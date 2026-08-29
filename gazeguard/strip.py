"""
Frame-strip classification: give the VLM the temporal/spatial context it needs.

A single isolated 480p frame strips away everything a vision-LANGUAGE model is good at —
narrative, motion, setting. A neckline crop of a woman standing in a parking lot mid-chat
reads as "woman near man, cut neckline" and the model fixates on it like an ML classifier.
By sending a strip of 3-5 consecutive frames in ONE call, the model sees the scene as a
shot ("parking lot, daytime, talking, standing") and reasons about context instead of a
single ambiguous frame. Also ~Nx cheaper per classified frame.

The verdict for the whole strip is assigned to every frame in it — for coarse-to-fine
discovery a scene is one perceptual unit anyway.
"""

from typing import Dict, List, Tuple

from .classifier import Classifier, GeminiClassifier
from .frame_io import subtitle_slice
from .policy import SemanticMeta


class StripClassifier(Classifier):
    """
    Wraps a GeminiClassifier and re-groups per-timestamp frames into consecutive strips,
    sending each strip as a single call. `classify()` returns one SemanticMeta per frame,
    with frames in the same strip sharing the strip's verdict.

    Optional dialogue grounding: set `subtitles` (list of (start_s, end_s, text)) and the
    classifier injects the dialogue overlapping each strip's time window as narrative context.
    """
    name = "strip"

    def __init__(self, inner: "GeminiClassifier", strip_size: int = 4,
                 max_concurrent: int = 4):
        self.inner = inner
        self.strip_size = max(2, int(strip_size))
        self.max_concurrent = max(1, max_concurrent)
        self.model = getattr(inner, "model", "gemini-2.5-flash")
        self.subtitles: List[tuple] = []  # set by the pipeline before run()

    def min_frame_confidence(self) -> float:
        return self.inner.min_frame_confidence()

    def n_requests(self, frames: Dict[float, bytes]) -> int:
        """One API request per strip of strip_size frames (Gemini bills per request)."""
        n = len(frames)
        return (n + self.strip_size - 1) // self.strip_size

    def classify(self, frames: Dict[float, bytes]) -> Dict[float, SemanticMeta]:
        if not frames:
            return {}
        from concurrent.futures import ThreadPoolExecutor

        order = sorted(frames.keys())
        strips: List[Tuple[List[float], List[bytes]]] = []
        i = 0
        while i < len(order):
            chunk = order[i:i + self.strip_size]
            strips.append((chunk, [frames[t] for t in chunk]))
            i += self.strip_size

        out: Dict[float, SemanticMeta] = {}

        def _one_strip(strip: Tuple[List[float], List[bytes]]):
            ts, jpegs = strip
            try:
                caption = ""
                if self.subtitles:
                    caption = subtitle_slice(self.subtitles, min(ts), max(ts))
                meta = self.inner._call_strip(jpegs, caption=caption)
                return {t: meta for t in ts}
            except Exception:
                # FAIL-SHUT: any strip call error -> explicit exposure at max severity.
                fs = SemanticMeta(exposure_level="explicit",
                                  situational_state="active_encounter",
                                  summary="API/safety error (fail-shut)",
                                  source="gemini", confidence=1.0)
                return {t: fs for t in ts}

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as ex:
            for result in ex.map(_one_strip, strips):
                out.update(result)
        return out