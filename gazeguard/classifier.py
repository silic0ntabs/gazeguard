"""
Classifier backends for the coarse-to-fine pipeline.

A `Classifier` extracts *descriptive* metadata for a single 480p frame (JPEG bytes)
across the 5 location-agnostic semantic primitives from the design (exposure_level,
physical_action, situational_state, camera_intent, visual_layer) + a confidence + a
short summary. It passes NO judgement — the policy engine (policy.py / the Lua hook)
decides what to do based on a chosen profile.

Backends:
  * MockClassifier            - deterministic opencv colour heuristic (no API, no key).
                                Used for local dry-runs and verifying the search tree.
  * GeminiClassifier          - real frontier vision via Google Gemini using only stdlib
                                urllib (no SDK install). BLOCK_NONE so a suggestive frame
                                is described rather than suppressed; any API/safety error
                                fails SHUT (treated as explicit exposure at max severity).

Interface: classify(frames) -> Dict[timestamp, SemanticMeta]  (semantic description)
Decision:  pipeline uses policy.tree_explicit() to decide discovery.
"""

from __future__ import annotations

import abc
import base64
import os
import json
import urllib.error
import urllib.request
from typing import Dict, List, Tuple

from .policy import SemanticMeta

# Keep the older tri-state labels available for code that still reasons in labels.
CLEAN, FLAG, AMBI = "clean", "flag", "ambiguous"
VERDICTS = (CLEAN, FLAG, AMBI)


# --- Base -------------------------------------------------------------------
class Classifier(abc.ABC):
    name = "base"
    model = ""  # model id used by the ledger for cost accounting

    def min_frame_confidence(self) -> float:
        """Below this confidence the search tree treats a frame as unreliable."""
        return 0.5

    def n_requests(self, frames: Dict[float, bytes]) -> int:
        """Number of API requests `classify(frames)` will make (Gemini bills per request)."""
        return len(frames)

    @abc.abstractmethod
    def classify(self, frames: Dict[float, bytes]) -> Dict[float, SemanticMeta]:
        """
        Describe a dict of {timestamp_seconds: jpeg_bytes}.
        Returns {timestamp: SemanticMeta}. Timestamps with no result are absent.
        """

    def close(self) -> None:
        pass


# --- Mock (local, deterministic) --------------------------------------------
class MockClassifier(Classifier):
    """cv2 skin-tone colour heuristic. No network, no key. For dry runs + tests."""

    name = "mock"

    def __init__(self, flag_fraction: float = 0.35, ambi_fraction: float = 0.15):
        self.flag_fraction = flag_fraction
        self.ambi_fraction = ambi_fraction

    def min_frame_confidence(self) -> float:
        return 0.5

    def _classify_one(self, jpeg: bytes) -> SemanticMeta:
        import cv2
        import numpy as np

        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return SemanticMeta(summary="undecodable frame", source="mock", confidence=0.0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 40, 30), (25, 255, 255))
        mask2 = cv2.inRange(hsv, (340, 40, 30), (360, 255, 255))
        skin = cv2.bitwise_or(mask, mask2)
        frac = float(cv2.countNonZero(skin)) / float(skin.shape[0] * skin.shape[1])

        if frac >= self.flag_fraction:
            return SemanticMeta(
                exposure_level="moderate",
                physical_action="solitary_or_distant",
                situational_state="ambient_entertainment",
                intimacy_level="sensual_romance",
                camera_intent="neutral_cinematic",
                visual_layer="foreground_focal",
                summary="large skin-tone region (heuristic)",
                source="mock",
                confidence=min(1.0, 0.5 + (frac - self.flag_fraction)),
            )
        if frac >= self.ambi_fraction:
            return SemanticMeta(
                exposure_level="mild",
                physical_action="solitary_or_distant",
                situational_state="casual_routine",
                camera_intent="neutral_cinematic",
                visual_layer="foreground_focal",
                summary="moderate skin-tone region (heuristic)",
                source="mock",
                confidence=0.4 + (frac - self.ambi_fraction) / 4.0,
            )
        return SemanticMeta(
            exposure_level="none",
            physical_action="solitary_or_distant",
            situational_state="casual_routine",
            camera_intent="neutral_cinematic",
            visual_layer="foreground_focal",
            summary="no skin-tone signal",
            source="mock",
            confidence=0.2,
        )

    def classify(self, frames: Dict[float, bytes]) -> Dict[float, SemanticMeta]:
        return {t: self._classify_one(b) for t, b in frames.items()}


# --- Gemini (frontier vision, batch) ----------------------------------------
class GeminiClassifier(Classifier):
    """
    Frontier vision via Google Gemini (e.g. gemini-2.5-flash). Returns descriptive
    5-primitive metadata. Uses only stdlib urllib.
    """

    name = "gemini"

    def __init__(self, model: str = "", api_key: str = "", endpoint_base: str = "",
                 max_concurrent: int = 8, retries: int = 3):
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.endpoint_base = endpoint_base or os.environ.get(
            "GEMINI_BASE", "https://generativelanguage.googleapis.com"
        )
        self.max_concurrent = max_concurrent
        self.retries = retries
        if not self.api_key:
            raise RuntimeError(
                "GeminiClassifier requires GEMINI_API_KEY env var (or api_key=). "
                "Set it to use the frontier vision tier."
            )

    def min_frame_confidence(self) -> float:
        return 0.3

    def _prompt(self) -> str:
        # Descriptive, location-agnostic. NO moral judgement, NO object-specific rules.
        # P0-B: constrained "skin-gate" — a fully-clothed context caps the intimacy rung at
        # platonic/mild regardless of close physical proximity (fixes ballroom over-block).
        return (
            "You are an objective visual content analyzer for video preprocessing. "
            "Extract multi-dimensional contextual, physical, and atmospheric metadata "
            "from the provided frame. Evaluate the scene across foreground and background "
            "planes. Do not apply personal moral judgment; extract precise categorical "
            "descriptions based strictly on visible evidence and situational context, for "
            "any environment (car, bedroom, elevator, pool, desk, alley — do not key off "
            "the location or furniture).\n\n"
            "### SKIN-GATE (hard rule):\n"
            "Answer these discriminating binaries first:\n"
            "  1. any exposed skin beyond face/hands/forearms? (yes/no)\n"
            "  2. intimate setting actually shown? (bedroom / bed / undressing / steamy vehicle / private encounter)\n"
            "  3. make-out, oral, or undressing actually visible? (yes/no)\n"
            "If (1) is NO and the contact is brief/formal/fully-clothed (a handshake, a "
            "formal dance embrace, a ballroom waltz, characters just standing or sitting "
            "close, talking in bed in pyjamas), then CAP intimacy_level at 'platonic' or "
            "'mild_romance' — it must NEVER be 'sensual_romance' unless real skin is "
            "exposed (beyond face/hands) or an actual make-out/undressing is visible. "
            "Tight clothes, cargo pants, or a bare-armed woman are NOT 'sensual_romance' "
            "by themselves.\n\n"
            "### FEW-SHOT EXEMPLARS (use as anchors, not a floor):\n"
            "  - Two ballroom dancers, fully clothed, mid-waltz, close embrace -> intimacy='platonic', "
            "exposure='none', camera='neutral_cinematic'.\n"
            "  - Couple lying under a sheet, bare shoulders visible, post-intimate -> intimacy='sensual_romance', "
            "exposure='moderate', situational='implied_intimacy'.\n"
            "  - Someone in a bikini / topless by a pool -> exposure='moderate'/'explicit', intimacy='none', "
            "situational='casual_routine' (exposure is the signal, not intimacy).\n"
            "  - Two people speaking in an office, handshake or arm-shoulder -> intimacy='none'/'platonic', "
            "situational='casual_routine'.\n"
            "  - Heavy open-mouth make-out with touching, both fully or partially unclothed -> intimacy='sensual_romance' "
            "or 'explicit'.\n\n"
            "Be decisive and specific. When unsure between two rungs/levels, pick the LESS "
            "severe one unless exposing skin or a clearly physical act is shown."
        )

    def _request_payload(self, frame_b64: str) -> dict:
        return {
            "contents": [{
                "parts": [
                    {"text": self._prompt()},
                    {"inline_data": {"mime_type": "image/jpeg", "data": frame_b64}},
                ]
            }],
            "safetySettings": [
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "exposure_level": {
                            "type": "STRING",
                            "enum": ["none", "mild", "moderate", "explicit"],
                        },
                        "physical_action": {
                            "type": "STRING",
                            "enum": [
                                "solitary_or_distant",
                                "platonic_or_combat",
                                "affectionate_mild",
                                "sensual_intimate",
                                "explicit_sexual",
                            ],
                        },
                        "situational_state": {
                            "type": "STRING",
                            "enum": [
                                "casual_routine",
                                "clinical_or_distress",
                                "ambient_entertainment",
                                "implied_intimacy",
                                "active_encounter",
                            ],
                        },
                        "intimacy_level": {
                            "type": "STRING",
                            "enum": [
                                "none",
                                "platonic",
                                "mild_romance",
                                "sensual_romance",
                                "explicit",
                            ],
                            "description": "Precisely which rung of physical intimacy the frame SHOWS: none=dialogue/distance/combat; platonic=handshake, formal hug, cheek kiss, shoulder arm; mild_romance=tame closed-mouth kiss, holding hands, brief embrace, casual dance, fully-clothed; sensual_romance=lingering/passionate make-out, caressing over clothes, heavy contact that is clearly turning sexual; explicit=any sex act, undressing, nudity-driven intimacy.",
                        },
                        "camera_intent": {
                            "type": "STRING",
                            "enum": ["neutral_cinematic", "voyeuristic_objectifying"],
                        },
                        "visual_layer": {
                            "type": "STRING",
                            "enum": ["foreground_focal", "background_ambient", "full_frame"],
                        },
                        "summary": {
                            "type": "STRING",
                            "description": "Short factual 5-8 word summary of the frame.",
                        },
                    },
                    "required": [
                        "exposure_level", "physical_action", "situational_state",
                        "intimacy_level", "camera_intent", "visual_layer", "summary",
                    ],
                },
            },
        }

    def _payload_schema(self) -> dict:
        """The generationConfig.responseSchema block for structured output."""
        payload = self._request_payload("")
        return payload["generationConfig"]

    def _call_strip(self, jpegs: List[bytes], caption: str = "") -> SemanticMeta:
        """One call carrying several CONSECUTIVE frames; judge the scene as a whole.

        caption: optional SUBTITLE/DIALOGUE text for the scene's time window. This is the
        low-cost narrative ground (an episode is ~6-8k text tokens) that stops the model
        confabulating an intimate encounter from warm lighting / a bedroom: "The Soviets are
        mobilizing" alongside a dim living room keeps it a briefing, not a liaison.
        """
        b64s = [base64.b64encode(b).decode("ascii") for b in jpegs]
        dialogue = ""
        if caption and caption.strip():
            dialogue = (
                "\n\nSPOKEN DIALOGUE during this scene (from the subtitles):\n"
                f"```{caption.strip()}```\n"
                "Use this dialogue as narrative ground. It tells you the CONTEXT of what you "
                "see. Do not invent an intimate/acquainted reading if the dialogue indicates "
                "a professional, casual, familial, or default-non-intimate circumstance."
            )
        parts = [{"text": (
            self._prompt() +
            "\n\nThese frames are a CONSECUTIVE STRIP from ONE SHOT in a video. Judge the "
            "SCENE AS A WHOLE using temporal and spatial context (setting, motion, posture, "
            "whether anyone is actually moving to undress or escalate), NOT any single frame "
            "in isolation. A person standing and talking in a normal daytime location in "
            "ordinary clothes is 'casual_routine' regardless of a low camera angle or a modest "
            "neckline. Return the metadata for the scene." +
            dialogue
        )}] + [{"inline_data": {"mime_type": "image/jpeg", "data": b}} for b in b64s]
        body = json.dumps({
            "contents": [{"parts": parts}],
            "safetySettings": [
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
                "responseSchema": self._payload_schema()["responseSchema"],
            },
        }).encode("utf-8")

        url = f"{self.endpoint_base}/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        import time as _t
        data = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                err = e.read().decode("utf-8", "replace")
                if e.code in (429, 500, 502, 503) and attempt < self.retries - 1:
                    _t.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Gemini strip HTTP {e.code}: {err}")
            except (urllib.error.URLError, TimeoutError, ConnectionResetError) as e:
                if attempt < self.retries - 1:
                    _t.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Gemini strip network error: {e}")
        if data is None:
            raise RuntimeError("Gemini strip call exhausted retries")
        candidates = data.get("candidates") or []
        if not candidates:
            fr = (data.get("promptFeedback") or {}).get("blockReason")
            raise RuntimeError(f"Gemini strip: no candidates (blockReason={fr})")
        cand = candidates[0]
        if cand.get("finishReason") == "SAFETY":
            raise RuntimeError("Gemini strip safety block (fail-shut)")
        parts = (cand.get("content") or {}).get("parts") or []
        text = "\n".join(p.get("text", "") for p in parts if p.get("text"))
        return self._parse_meta(text)

    def classify(self, frames: Dict[float, bytes]) -> Dict[float, SemanticMeta]:
        from concurrent.futures import ThreadPoolExecutor

        out: Dict[float, SemanticMeta] = {}
        b64 = {t: base64.b64encode(b).decode("ascii") for t, b in frames.items()}

        def _one(item: Tuple[float, str]):
            t, data = item
            try:
                ver = self._call_batch(data)
                return (t, ver)
            except Exception:
                # FAIL-SHUT: any API/safety error -> explicit exposure at max severity.
                return (t, SemanticMeta(exposure_level="explicit",
                                        situational_state="active_encounter",
                                        summary="API/safety error (fail-shut)",
                                        source="gemini", confidence=1.0))

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as ex:
            for t, ver in ex.map(_one, b64.items()):
                if ver:
                    out[t] = ver
        return out

    def _call_batch(self, frame_b64: str) -> SemanticMeta:
        """One frame inside a Batch API generateContent request -> descriptive metadata."""
        url = (f"{self.endpoint_base}/v1beta/models/{self.model}:generateContent?key={self.api_key}")
        body = json.dumps(self._request_payload(frame_b64)).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})

        import time as _t
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                err = e.read().decode("utf-8", "replace")
                # transient: rate limit / server hiccup -> back off and retry
                if e.code in (429, 500, 502, 503) and attempt < self.retries - 1:
                    _t.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Gemini HTTP {e.code}: {err}")
            except (urllib.error.URLError, TimeoutError, ConnectionResetError) as e:
                if attempt < self.retries - 1:
                    _t.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Gemini network error: {e}")
        else:
            raise RuntimeError("Gemini call exhausted retries")

        candidates = data.get("candidates") or []
        if not candidates:
            fr = (data.get("promptFeedback") or {}).get("blockReason") or data.get("error") or {}
            raise RuntimeError(f"no candidate (blocked? {fr})")
        cand = candidates[0]
        if cand.get("finishReason") in ("SAFETY", "PROHIBITED_CONTENT"):
            raise RuntimeError(f"Gemini safety block: {cand.get('finishReason')}")
        text = (cand.get("content") or {}).get("parts", [{}])[0].get("text", "")
        return self._parse_meta(text)

    def _parse_meta(self, text: str) -> SemanticMeta:
        import re

        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError(f"no JSON in model reply: {text[:120]!r}")
        raw = m.group(0)
        cleaned = re.sub(r"[\r\n\t]+", "", raw)
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            raise ValueError(f"bad JSON in model reply: {raw[:120]!r}")

        _ALIAS = {
            # checked in order; more specific/severe phrasing wins (e.g. 'passionate kiss' -> sensual)
            "sensual_romance": ["make out", "makeout", "passionate", "heavy", "caressing",
                                "groping", "grinding", "dry hump", "tongue", "foreplay",
                                "sensual"],
            "explicit": ["sex", "intercourse", "undressing", "naked", "topless", "nudity"],
            "mild_romance": ["mild kiss", "tame kiss", "kiss", "peck", "closed mouth",
                             "holding hands", "hand hold", "brief embrace", "dance",
                             "slow dance", "romance", "romantic", "affectionate"],
            "platonic": ["handshake", "hug", "cheek kiss", "peck on cheek", "shoulder",
                         "fist bump", "friendship", "sober"],
        }

        def _enum(key: str, allowed, default: str) -> str:
            v = str(obj.get(key, default)).strip().lower()
            # exact or substring first
            for a in allowed:
                if v == a or v in a or a in v:
                    return a
            # alias fallback: find which allowed value is suggested by the phrasing
            for canon, syns in _ALIAS.items():
                if canon in allowed and any(s in v for s in syns):
                    return canon
            return default

        return SemanticMeta(
            exposure_level=_enum("exposure_level",
                                 ["none", "mild", "moderate", "explicit"], "none"),
            physical_action=_enum("physical_action",
                                  ["solitary_or_distant", "platonic_or_combat",
                                   "affectionate_mild", "sensual_intimate", "explicit_sexual"],
                                  "solitary_or_distant"),
            situational_state=_enum("situational_state",
                                    ["casual_routine", "clinical_or_distress",
                                     "ambient_entertainment", "implied_intimacy",
                                     "active_encounter"],
                                    "casual_routine"),
            intimacy_level=_enum("intimacy_level",
                                 ["none", "platonic", "mild_romance",
                                  "sensual_romance", "explicit"],
                                 "none"),
            camera_intent=_enum("camera_intent",
                                ["neutral_cinematic", "voyeuristic_objectifying"],
                                "neutral_cinematic"),
            visual_layer=_enum("visual_layer",
                               ["foreground_focal", "background_ambient", "full_frame"],
                               "foreground_focal"),
            summary=str(obj.get("summary", "")).strip()[:80],
            confidence=float(obj.get("confidence", 0.5)),
            source="gemini",
        )