"""
Merge & padding engine.

Turns raw flagged-timestamp clusters into clean blackout intervals:
  1. merge adjacent flagged segments separated by < `gap_merge` s (anti-strobe)
  2. prepend `lead_in` and append `trail_out` padding so no exposed frames slip through
  3. clamp to [0, duration]
  4. drop any surviving block shorter than `min_block`

Intervals use the schema expected by the mpv/IINA lua: {"start": ..., "end": ...}.
"""

from __future__ import annotations

from typing import List, Dict


def _merge_clusters(flagged: List[float], gap_merge: float) -> List[List[float]]:
    """Group ascending timestamps into clusters where adjacent gaps <= gap_merge."""
    if not flagged:
        return []
    flagged = sorted(flagged)
    clusters: List[List[float]] = [[flagged[0]]]
    for t in flagged[1:]:
        if t - clusters[-1][-1] <= gap_merge:
            clusters[-1].append(t)
        else:
            clusters.append([t])
    return clusters


def build_intervals(
    flagged_timestamps: List[float],
    *,
    gap_merge: float = 1.0,
    lead_in: float = 0.3,
    trail_out: float = 0.2,
    min_block: float = 0.0,
    duration_s: float | None = None,
    fps: float = 1.0,
) -> List[Dict[str, float]]:
    """
    Convert a list of flagged frame timestamps into final blackout intervals.

    Each flagged timestamp contributes a [t - 1/(2*fps), t + 1/(2*fps)] span so a single
    flagged frame still occludes its own frame, then clusters are merged/padded.
    """
    # Build per-frame spans from the fps grid (so a lone flagged frame still occludes itself)
    half = 0.5 / max(1.0, fps)
    edges: List[float] = []
    for t in flagged_timestamps:
        edges.append(t - half)
        edges.append(t + half)

    if not edges:
        return []

    # Sort all edges, sweep to merge overlapping / padding-close spans.
    edges.sort()
    intervals: List[Dict[str, float]] = []
    cur_start, cur_end = edges[0], edges[0]
    for e in edges[1:]:
        if e <= cur_end + gap_merge:
            cur_end = max(cur_end, e)
        else:
            intervals.append({"start": cur_start, "end": cur_end})
            cur_start, cur_end = e, e
    intervals.append({"start": cur_start, "end": cur_end})

    # Apply lead / trail padding, clamp, filter length.
    result: List[Dict[str, float]] = []
    for iv in intervals:
        s = max(0.0, iv["start"] - lead_in)
        e = iv["end"] + trail_out
        if duration_s is not None:
            e = min(e, duration_s)
        if (e - s) >= min_block:
            result.append({"start": round(s, 3), "end": round(e, 3)})
    return result