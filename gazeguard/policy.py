"""
Playback policy engine for the descriptive semantic metadata.

The model only *describes* a frame across 5 location-agnostic primitives; it passes no
judgement. Deciding what to do with that description lives HERE, in a policy layer that
is mirrored (1:1) in the IINA/mpv Lua so sensitivity can be re-tuned at playback with
zero additional API calls.

Primitives (all location-agnostic — a car, elevator, pool, desk or alley all behave the
same way; a bed is NOT special):
  exposure_level       none | mild | moderate | explicit
  physical_action      solitary_or_distant | platonic_or_combat | affectionate_mild
                       | sensual_intimate | explicit_sexual
  situational_state    casual_routine | clinical_or_distress | ambient_entertainment
                       | implied_intimacy | active_encounter
  camera_intent        neutral_cinematic | voyeuristic_objectifying
  visual_layer         foreground_focal | background_ambient | full_frame
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class Action(str, Enum):
    NONE = "none"            # nothing (or just subtitles preserved / audio continues)
    BLUR_MILD = "blur_mild"  # frosted blur: softens suggestive-but-uncertain content
    BLUR_STRONG = "blur_strong"  # heavy blur: obscures confirmed explicit content


PROFILES = ("STRICT", "BALANCED", "MINIMAL")


@dataclass
class SemanticMeta:
    """One frame's descriptive metadata (extracted by the VLM, not judged)."""
    exposure_level: str = "none"
    physical_action: str = "solitary_or_distant"
    situational_state: str = "casual_routine"
    # NEW: precise articulation of any physical intimacy actually shown as one ordered rung.
    intimacy_level: str = "none"
    camera_intent: str = "neutral_cinematic"
    visual_layer: str = "foreground_focal"
    summary: str = ""
    confidence: float = 0.5
    source: str = "gemini"  # gemini | mock

    def as_dict(self) -> dict:
        return {
            "exposure_level": self.exposure_level,
            "physical_action": self.physical_action,
            "situational_state": self.situational_state,
            "intimacy_level": self.intimacy_level,
            "camera_intent": self.camera_intent,
            "visual_layer": self.visual_layer,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "SemanticMeta":
        return cls(
            exposure_level=d.get("exposure_level", "none"),
            physical_action=d.get("physical_action", "solitary_or_distant"),
            situational_state=d.get("situational_state", "casual_routine"),
            intimacy_level=d.get("intimacy_level", "none"),
            camera_intent=d.get("camera_intent", "neutral_cinematic"),
            visual_layer=d.get("visual_layer", "foreground_focal"),
            summary=d.get("summary", ""),
            confidence=float(d.get("confidence", 0.5)),
            source=d.get("source", "gemini"),
        )


# --- Severity ordering used to aggregate many frames into one interval -----------------
_EXP = {"none": 0, "mild": 1, "moderate": 2, "explicit": 3}
_ACT = {"solitary_or_distant": 0, "platonic_or_combat": 1, "affectionate_mild": 2,
        "sensual_intimate": 3, "explicit_sexual": 4}
_SIT = {"casual_routine": 0, "clinical_or_distress": 1, "ambient_entertainment": 2,
        "implied_intimacy": 3, "active_encounter": 4}
# Ordered intimacy rung — the PRIMARY policy driver: the model articulates HOW intimate the
# shown physical contact is; the policy decides how much of it a profile permits.
_INT = {"none": 0, "platonic": 1, "mild_romance": 2, "sensual_romance": 3, "explicit": 4}


def severity(meta: SemanticMeta) -> int:
    return (_EXP.get(meta.exposure_level, 0) * 5
            + _INT.get(meta.intimacy_level, 0) * 4
            + _ACT.get(meta.physical_action, 0) * 3
            + _SIT.get(meta.situational_state, 0) * 2
            + (1 if meta.camera_intent == "voyeuristic_objectifying" else 0))


def combine_metas(metas: List[SemanticMeta]) -> SemanticMeta:
    """
    Aggregate many frame-metas in an interval into one representative.

    Prefers genuine frames over fail-shut frames: a fail-shut frame is worst-severity
    ('explicit'/'active_encounter') so it would always win combine_metas if included, which
    would overwrite the REAL scene description whenever a single flaky frame failed shut.
    Fail-shut still forces a block (discovery already treats it as explicit), but it should
    only supply METADATA when there is no real signal to report.
    """
    if not metas:
        return SemanticMeta()
    real = [m for m in metas if "fail-shut" not in (m.summary or "")]
    pool = real if real else metas
    worst = max(pool, key=severity)
    return SemanticMeta(
        exposure_level=worst.exposure_level,
        physical_action=worst.physical_action,
        situational_state=worst.situational_state,
        intimacy_level=worst.intimacy_level,
        camera_intent=worst.camera_intent,
        visual_layer=worst.visual_layer,
        summary=worst.summary,
        confidence=max(m.confidence for m in pool),
        source=pool[0].source,
    )


def hard_trigger(meta: SemanticMeta) -> bool:
    """
    Unconditional blackout across ANY profile — only genuinely explicit acts/undressing.
    A tame, clothed kiss registered as intimacy=mild_romance must NOT land here, so MINIMAL
    (and BLUR for BALANCED) can recover. This is the direct fix for the Titanic false-positive
    where exposure="none"/"mild" combined with over-called active_encounter got hard-blocked.
    """
    return (meta.exposure_level == "explicit"
            or meta.intimacy_level == "explicit"
            or (meta.intimacy_level == "sensual_romance"
                and meta.exposure_level in ("moderate", "explicit")))


_EXP_ORDER = ["none", "mild", "moderate", "explicit"]


def _at_or_above(level: str, floor: str) -> bool:
    """True if `level` is >= `floor` in the exposure severity ladder."""
    try:
        return _EXP_ORDER.index(level) >= _EXP_ORDER.index(floor)
    except ValueError:
        return False


def action_for(meta: SemanticMeta, profile: str = "STRICT",
               exposure_floor: str = "moderate") -> Action:
    """
    Deterministic decision. MIRRORED IN gazeguard_blackout.lua (keep them in sync).
    intimacy_level is the primary driver; exposure/situational/camera contribute.

    exposure_floor: minimum exposure-level at which intimacy/implied_intimacy count as
    skinnable (moderate = current; change here + Lua + config to re-tune later).
    """
    if hard_trigger(meta):
        # confirmed explicit acts/undressing -> STRONG blur (content obscured, subs/sound stay)
        return Action.BLUR_STRONG

    # Clinical/medical exemption: a doctor examining a patient (gown, brief chest exposure,
    # surgical context) is NOT a violation. Mirrors "Pass or Mild Blur".
    clinical = (meta.situational_state == "clinical_or_distress"
                and meta.physical_action == "platonic_or_combat"
                and meta.exposure_level != "explicit")

    if profile == "STRICT":
        # The model can be "skittish in private spaces" — it may read a couple talking in bed,
        # a car conversation, or two friends chatting as "implied_intimacy"/"romance" even with
        # NOTHING exposed. So the intimacy/situational classes are only treated as sensitive
        # when backed by skin. This is the exposure gate: none/mild = people are just close,
        # let it pass; the confirmed-bad content (bra, lingerie) is moderate/explicit and
        # still blurred.
        skin = _at_or_above(meta.exposure_level, exposure_floor)
        objectifying = (meta.camera_intent == "voyeuristic_objectifying"
                        or meta.visual_layer == "background_ambient")
        explicit_ambient = (meta.situational_state == "ambient_entertainment"
                            and meta.exposure_level == "explicit")
        if not clinical and (
                (meta.intimacy_level in ("mild_romance", "sensual_romance") and skin)
                or (meta.situational_state == "implied_intimacy" and skin)
                or explicit_ambient
                or (meta.camera_intent == "voyeuristic_objectifying"
                    and meta.exposure_level in ("mild", "moderate"))):
            return Action.BLUR_STRONG
        # objectifying scene with visible-but-not-explicit exposure -> STRONG in STRICT
        if objectifying and meta.exposure_level in ("mild", "moderate"):
            return Action.BLUR_STRONG
        if clinical and meta.exposure_level == "moderate":
            return Action.BLUR_MILD

    elif profile == "BALANCED":
        # Same exposure gate: a couple "in bed / in a car" with NO exposed skin (none/mild)
        # is just people talking close — blur at most, never heavy. Real content = skin.
        skin = _at_or_above(meta.exposure_level, exposure_floor)
        if not clinical and (
                (meta.intimacy_level == "sensual_romance" and skin)
                or (meta.situational_state == "implied_intimacy" and skin)):
            return Action.BLUR_STRONG
        # tame-but-visible romance, mid exposure, voyeuristic/objectifying angle, or
        # background activity -> frosted blur (lets mild kisses through as a soft blur)
        if meta.exposure_level == "moderate" \
                or (meta.intimacy_level == "mild_romance" and skin) \
                or meta.camera_intent == "voyeuristic_objectifying" \
                or meta.visual_layer == "background_ambient":
            return Action.BLUR_MILD

    elif profile == "MINIMAL":
        # hard triggers only (actual explicit acts/nudity); tame kisses and implied romance pass
        pass

    return Action.NONE


def tree_explicit(meta: SemanticMeta) -> bool:
    """
    Discovery-time predicate used by the coarse-to-fine pipeline to decide whether a
    frame belongs in a flagged region.

    Delegates to action_for(STRICT) so discovery and playback share ONE decision
    function — no drift between what gets captured and what gets blacked out at playback.
    """
    return action_for(meta, "STRICT") != Action.NONE