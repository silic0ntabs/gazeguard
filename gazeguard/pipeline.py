"""
Hierarchical Coarse-to-Fine (Binary Drill-Down) pipeline.

Level 1 (Macro scan): evenly-spaced low-res frames across the whole timeline (~1 per
`sample_interval` s). Mark fully clean regions safe with ZERO extra inference.

Level 2 (Micro cut resolution): for each flagged coarse sample, extract real scene-cut
timestamps within its ±`cut_window` envelope and re-sample the envelope denser (every
`refine_step` s) to find where the flagged region actually starts/ends.

Level 3 (Binary boundary pinpointing): recursively bisect each boundary between a known
flag and known-clean frame until the interval is within `precision` s (±~200 ms).

Discovery runs at STRICT sensitivity (see policy.tree_explicit) so the sidecar
captures ENOUGH descriptive metadata that BALANCED/MINIMAL profiles can relax the decision
later at playback — a single extraction pass, re-tunable at ₹0.

The whole thing is designed so a *frontier vision* classifier (e.g. Gemini via Batch)
is called ~225-1200 times per episode instead of scanning all cuts linearly. Each call
returns descriptive 5-primitive metadata which is aggregated per interval into the sidecar.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import frame_io as fi
from .classifier import Classifier
from .merger import build_intervals
from .policy import SemanticMeta, combine_metas, tree_explicit


@dataclass
class ScanStat:
    l1_frames: int = 0
    l1_inferred: int = 0
    l2_frames: int = 0
    l3_calls: int = 0
    l3_recursive_calls: int = 0
    total_classified: int = 0
    flagged_timestamps: List[float] = field(default_factory=list)
    elapsed_s: float = 0.0
    prior: Optional[dict] = None   # grounding prior (steering only)
    n_subtitle_events: int = 0     # how many dialogue cues were loaded for context

    def as_dict(self) -> dict:
        return {
            "l1_frames": self.l1_frames,
            "l1_inferred": self.l1_inferred,
            "l2_frames": self.l2_frames,
            "l3_calls": self.l3_calls,
            "l3_recursive_calls": self.l3_recursive_calls,
            "total_classified": self.total_classified,
            "n_flagged": len(self.flagged_timestamps),
            "elapsed_s": round(self.elapsed_s, 2),
            "prior": self.prior,
        }


@dataclass
class Options:
    sample_interval: float = 12.0   # Level 1 coarse step (s)
    cut_window: float = 10.0        # +/- window around a flagged sample for Level 2 (s)
    refine_step: float = 1.0        # Level 2 denser sampling step inside envelope (s)
    precision: float = 0.2          # Level 3 final boundary precision (s) - fine / sporadic
    edge_precision: float = 1.0     # Level 3 edge precision for SUSTAINED runs (coarser, cheap)
    sustained_min_span_s: float = 24.0  # contiguous flagged coarse run >= this = "sustained"
    max_l3_depth: int = 8           # cap on recursive bisection depth
    scene_threshold: float = 0.3    # ffmpeg scene cut sensitivity
    venue_of_video_scale: int = 480 # 480p short edge
    gap_merge: float = 1.0          # merge flagged clusters separated by < 1s (anti-strobe)
    lead_in: float = 0.3
    trail_out: float = 0.2
    min_block: float = 0.0
    concurrency: int = 4
    per_episode_cap: Optional[int] = None  # optional: grounding can lower this (never raises verdict)
    scene_sampling: bool = False  # P1: sample one frame per ffmpeg-detected scene instead of by time
    dedup_samples: bool = False   # P1: perceptual-hash dedup of near-identical L1 samples
    resolution: str = "broad"     # 'broad'=coarse blocks from L1 flags (cheap, current default);
                                  # 'fine'=L2 dense scan + L3 ms bisection (expensive, deferred)
    strip_size: int = 0        # >1 => wrap Gemini in StripClassifier (context-aware scene verdicts)
    subtitle_grounding: bool = False  # load subtitles + inject dialogue into strip prompts
    dry_run: bool = False
    verbose: bool = False


class _NotSustained(Exception):
    """Raised when a sustained run's midpoint probe comes back clean -> should be re-scanned finely."""


class CoarseToFinePipeline:
    def __init__(self, classifier: Classifier, opts: Optional[Options] = None,
                 call_budget=None):
        self.classifier = classifier
        # mid-run spend guard: every batch is counted before it's sent
        self.call_budget = call_budget
        self.opts = opts or Options()
        # Episode grounding prior (P0-A) — steers density/cap only, never a verdict.
        self.prior = None
        self._grounded_interval: Optional[float] = None
        # Registry of every frame's metadata keyed by (rounded) timestamp.
        self._metas: Dict[float, SemanticMeta] = {}

    def _guess(self, frames: Dict[float, bytes]) -> Dict[float, SemanticMeta]:
        """Send a batch through the optional spend guard, then the classifier."""
        if self.call_budget is not None:
            return self.call_budget.classify(frames)
        return self.classifier.classify(frames)

    def _record(self, verdicts: Dict[float, SemanticMeta]) -> None:
        """Store per-frame metadata so intervals can be annotated after discovery."""
        for t, m in verdicts.items():
            self._metas[round(t, 3)] = m

    # ---- Episode grounding (P0-A): steer L1 density / cap before the scan ------
    def grounded(self, show: str, season: Optional[int] = None,
                 episode: Optional[int] = None, *, use_cache: bool = True) -> "CoarseToFinePipeline":
        """
        Resolve the episode prior and apply its density/cap steering to this run.
        The prior NEVER emits a safe-skip; anything unknown/failed leaves the full scan.
        """
        from .grounding import ground_episode, apply_prior
        prior = ground_episode(show, season, episode, use_cache=use_cache)
        self.prior = prior
        new_interval, new_cap = apply_prior(prior, self.opts.sample_interval,
                                            getattr(self.opts, "per_episode_cap", None))
        self._grounded_interval = new_interval
        if new_cap is not None:
            self.opts.per_episode_cap = new_cap
        return self

    # ---- Level 1: macro scan ------------------------------------------------
    def _level1(self, video: Path, probe: fi.Probe, stat: ScanStat) -> Dict[float, bool]:
        """Even-spaced coarse samples -> {timestamp: explicit?} for the whole timeline."""
        interval = self._grounded_interval or self.opts.sample_interval
        if self.opts.scene_sampling:
            # P1: one representative frame per detected scene (drama -> often 30-60 total)
            ts = fi.scene_representative_times(video, self.opts.scene_threshold)
            # SAFETY FLOOR: never let sampling collapse to near-zero (synthetic clips /
            # single-shot films produce few scenes). Merge in time-interval samples so L1
            # keeps a minimum density (every ~2*sample_interval) no matter how few scenes.
            n_floor = max(2, int(probe.duration_s // (interval * 2)))
            floor_ts = [min(max(0.0, i * interval * 2), probe.duration_s - 0.05)
                        for i in range(n_floor)]
            ts = sorted(set(ts) | set(round(t, 3) for t in floor_ts))
        else:
            n = max(2, int(probe.duration_s // interval))
            ts = [min(max(0.0, i * interval), probe.duration_s - 0.05)
                  for i in range(n)]

        # Extract all coarse candidates concurrently (avoid serial per-frame ffmpeg).
        frames = fi.extract_frames_at(video, ts, probe, self.opts.venue_of_video_scale,
                                      self.opts.concurrency)

        if self.opts.dedup_samples and frames:
            # P1 dedup: drop near-identical (dHash) frames in timestamp order, ₹0.
            order = sorted(frames.keys())
            keep: List[float] = []
            prev_hash = None
            for t in order:
                h = fi.perceptual_hash(frames[t])
                if h is None or prev_hash is None or fi._hamming(h, prev_hash) > 10:
                    keep.append(t)
                    prev_hash = h
            # Dedup must never collapse the whole L1 set (e.g. uniform/solid-color shots
            # all hash identically). Floor: keep at least 25% of the original candidates.
            min_keep = max(2, (len(order) + 3) // 4)
            if len(keep) < min_keep:
                # fall back to evenly-spaced subset of the original order
                step = max(1, len(order) // max(1, min_keep))
                keep = [order[i] for i in range(0, len(order), step)][:min_keep]
            frames = {t: frames[t] for t in keep}

        stat.l1_frames = len(frames)
        metas = self._guess(frames) if frames else {}
        self._record(metas)
        stat.l1_inferred = len(metas)
        stat.total_classified += len(metas)
        return {t: tree_explicit(m) for t, m in metas.items()}

    # ---- Level 2: micro cut resolution --------------------------------------
    def _level2(self, video: Path, probe: fi.Probe, stat: ScanStat,
                coarse: Dict[float, bool]) -> List[float]:
        """
        For each coarse frame that is explicit, find its exact-ish scene:
          - detect actual scene cuts in the merged flagged envelope
          - denser-scan the merged envelope every refine_step s to get candidate timestamps
        Returns candidate timestamps likely inside the explicit scene.

        Merging overlapping suspect envelopes (interval union) means two adjacent flagged
        coarse samples do NOT re-scan the same seconds — a full-flagged episode stays
        roughly O(region) instead of O(region * flagged_samples).
        """
        suspects = sorted(t for t, ex in coarse.items() if ex)
        if not suspects:
            return []

        # ---- interval union of all suspect envelopes ---------------------------
        envs = [(max(0.0, t - self.opts.cut_window),
                 min(probe.duration_s, t + self.opts.cut_window)) for t in suspects]
        envs.sort()
        merged: List[tuple] = []
        for lo, hi in envs:
            if merged and lo <= merged[-1][1] + self.opts.refine_step:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))

        # Also detect scene cuts across EVERY merged envelope (cheap, no per-sample).
        candidates: List[float] = []
        for lo, hi in merged:
            candidates.extend(fi.detect_scene_cuts(video, lo, hi, self.opts.scene_threshold))

        # Denser-scan the merged envelopes once.
        denser: List[float] = []
        for lo, hi in merged:
            n = max(1, int((hi - lo) / self.opts.refine_step))
            denser.extend(lo + i * self.opts.refine_step for i in range(n))
        frames = fi.extract_frames_at(video, denser, probe, self.opts.venue_of_video_scale,
                                      self.opts.concurrency)
        stat.l2_frames += len(frames)
        metas = self._guess(frames) if frames else {}
        self._record(metas)
        stat.total_classified += len(metas)
        for dt, m in metas.items():
            if tree_explicit(m):
                candidates.append(dt)

        return sorted(set(round(c, 3) for c in candidates))

    # ---- Level 3: binary boundary pinpointing --------------------------------
    def _point_boundary(self, video: Path, probe: fi.Probe,
                        known_flag: float, known_clean: float,
                        stat: ScanStat, depth: int = 0,
                        precision: Optional[float] = None) -> float:
        """Bisect [known_clean, known_flag] to the clean<->explicit boundary, <= precision."""
        if precision is None:
            precision = self.opts.precision
        width = abs(known_flag - known_clean)
        if width <= precision or depth >= self.opts.max_l3_depth:
            return known_flag
        stat.l3_calls += 1
        mid = (known_flag + known_clean) / 2.0
        frames = fi.extract_frames_at(video, [mid], probe, self.opts.venue_of_video_scale, 1)
        if not frames:
            return known_flag
        metas = self._guess(frames)
        self._record(metas)
        stat.total_classified += len(metas)
        stat.l3_recursive_calls += 1
        m = next(iter(metas.values()))
        if tree_explicit(m):
            # mid still explicit -> boundary is lower
            return self._point_boundary(video, probe, mid, known_clean, stat,
                                        depth + 1, precision)
        # mid clean -> boundary is higher
        return self._point_boundary(video, probe, known_flag, mid, stat,
                                    depth + 1, precision)

    def _find_clean_anchor(self, video: Path, probe: fi.Probe,
                           from_t: float, direction: str,
                           stat: ScanStat, max_steps: int = 6) -> float:
        """Walk away from a flagged frame until we find a reliably non-explicit anchor."""
        step = self.opts.cut_window / 2.0
        t = from_t
        for _ in range(max_steps):
            next_t = t - step if direction == "start" else t + step
            if next_t < 0:
                return 0.0
            if next_t >= probe.duration_s:
                return probe.duration_s
            frames = fi.extract_frames_at(video, [next_t], probe,
                                          self.opts.venue_of_video_scale, 1)
            if frames:
                metas = self._guess(frames)
                self._record(metas)
                stat.total_classified += len(metas)
                m = next(iter(metas.values()), None)
                if m is not None and not tree_explicit(m):
                    return next_t
            t = next_t
        return t

    def _refine_boundaries(self, video: Path, probe: fi.Probe,
                           flagged: List[float], stat: ScanStat) -> List[Dict[str, float]]:
        """For each cluster, pinpoint start/end edges with L3 bisection."""
        from .merger import _merge_clusters

        clusters = _merge_clusters(flagged, self.opts.gap_merge)
        result: List[Dict[str, float]] = []
        for cl in clusters:
            f_lo, f_hi = cl[0], cl[-1]
            clean_before = self._find_clean_anchor(video, probe, f_lo, "start", stat)
            start = self._point_boundary(video, probe, f_lo, clean_before, stat)
            clean_after = self._find_clean_anchor(video, probe, f_hi, "end", stat)
            end = self._point_boundary(video, probe, f_hi, clean_after, stat)
            result.append({"start": round(start, 3), "end": round(end, 3)})
        return result

    def _annotate(self, intervals: List[Dict[str, float]]) -> List[Dict[str, object]]:
        """Attach combined per-interval descriptive metadata to each interval."""
        out: List[Dict[str, object]] = []
        for iv in intervals:
            lo, hi = iv["start"], iv["end"]
            metas = [m for t, m in self._metas.items() if lo - 0.5 <= t <= hi + 0.5]
            out.append({"start": iv["start"], "end": iv["end"],
                        "meta": combine_metas(metas).as_dict()})
        return out

    def _triage_coarse(self, coarse: Dict[float, bool], probe: fi.Probe):
        """
        Split explicit coarse samples into:
          * sustained runs  - a contiguous run of flagged coarse samples spanning >=
                              sustained_min_span_s. We already know (with high confidence) this
                              is a long explicit block, so we do NOT waste L2/L3 resolving its
                              interior second-by-second — we black the whole thing and only
                              pin the two outer edges at edge_precision.
          * sporadic        - isolated / short explicit samples that genuinely need L2 dense
                              scan + fine L3. (This is where a 2-min Sopranos scene lives.)
        Returns (sustained_runs, sporadic_flags).
        """
        flags = sorted(t for t, ex in coarse.items() if ex)
        if not flags:
            return [], []

        # Cluster contiguous flagged coarse samples (gap <= sample_interval*1.5 keeps them in one run)
        runs: List[List[float]] = []
        for t in flags:
            if runs and t - runs[-1][-1] <= self.opts.sample_interval * 1.5:
                runs[-1].append(t)
            else:
                runs.append([t])

        sustained: List[List[float]] = []
        sporadic: List[float] = []
        for r in runs:
            span = r[-1] - r[0]
            if span >= self.opts.sustained_min_span_s:
                sustained.append(r)
            else:
                sporadic.extend(r)
        return sustained, sporadic

    def _handle_sustained_run(self, video: Path, probe: fi.Probe,
                              run: List[float], stat: ScanStat) -> List[Dict[str, float]]:
        """
        For a sustained explicit run (no interior need): pin ONLY the two outer boundaries
        at edge_precision, treating the entire run span as one blackout block. This replaces
        what would otherwise be a dense L2 scan of the whole run interior.

        Health-check: probe the RUN MIDPOINT first. If the middle comes back genuinely clean,
        the "sustained" classification was likely a string of false-positives — so we do NOT
        bulk-block; we fall back to the full fine path for the run's frames. (This is the guard
        that stops a whole clean episode from going black due to a contiguous run of flaky L1
        flags — the exact Titanic failure.)
        """
        f_lo, f_hi = run[0], run[-1]
        mid = (f_lo + f_hi) / 2.0

        frames = fi.extract_frames_at(video, [mid], probe,
                                      self.opts.venue_of_video_scale, 1)
        if frames:
            metas = self._guess(frames)
            self._record(metas)
            stat.total_classified += len(metas)
            m = next(iter(metas.values()), None)
            # If the run's middle is solidly clean, treat the whole run as possibly bogus:
            # drop it to the sporadic bucket so it gets a proper fine scan.
            if m is not None and (m.exposure_level == "none"
                                  and m.intimacy_level == "none"
                                  and m.situational_state == "casual_routine"
                                  and "fail-shut" not in (m.summary or "")):
                raise _NotSustained()

        clean_before = self._find_clean_anchor(video, probe, f_lo, "start", stat,
                                               max_steps=2)
        start = self._point_boundary(video, probe, f_lo, clean_before, stat,
                                     precision=self.opts.edge_precision)

        clean_after = self._find_clean_anchor(video, probe, f_hi, "end", stat,
                                              max_steps=2)
        end = self._point_boundary(video, probe, f_hi, clean_after, stat,
                                   precision=self.opts.edge_precision)

        return [{"start": round(start, 3), "end": round(end, 3)}]

    # ---- Broad-blank (default milestone): coarse blocks straight from L1 flags -----
    def _broad_intervals(self, coarse: Dict[float, bool],
                         probe: fi.Probe) -> List[Dict[str, float]]:
        """
        Cheap path: no L2 dense scan, no L3 bisection. Each flagged coarse sample represents
        a ~sample_interval-wide window centered on it; we simply block the whole window and
        merge adjacent flagged windows into one wide block. Good enough to BLANK the content,
        bad at ms-precision edges — explicitly the current milestone (7b decision).
        """
        interval = self._grounded_interval or self.opts.sample_interval
        half = interval / 2.0
        flagged = sorted(t for t, ex in coarse.items() if ex)
        blocks: List[Dict[str, float]] = []
        cur_start = cur_end = -1.0  # sentinel: no open block yet
        for t in flagged:
            s = max(0.0, t - half)
            e = min(probe.duration_s, t + half)
            s = max(0.0, s - self.opts.lead_in)
            e = min(probe.duration_s, e + self.opts.trail_out)
            if cur_start < 0:
                cur_start, cur_end = s, e
            elif s <= cur_end + self.opts.gap_merge:
                cur_end = max(cur_end, e)
            else:
                if (cur_end - cur_start) >= self.opts.min_block:
                    blocks.append({"start": round(cur_start, 2), "end": round(cur_end, 2)})
                cur_start, cur_end = s, e
        if cur_start >= 0 and (cur_end - cur_start) >= self.opts.min_block:
            blocks.append({"start": round(cur_start, 2), "end": round(cur_end, 2)})
        return blocks

    # ---- Level 1b: scene-burst (strip) macro scan -----------------------------
    def _level1_strip(self, video: Path, probe: fi.Probe, stat: ScanStat) -> Dict[float, bool]:
        """
        Context-aware L1. For each detected scene, extract `strip_size` frames spread across
        that shot and classify them as ONE StripClassifier call, so the VLM judges the whole
        shot (setting / motion / posture) instead of a single ambiguous frame. This is the
        fix for "the model fixated on a neckline crop instead of seeing a parking-lot chat."
        """
        scenes = fi.detect_scenes(video, self.opts.scene_threshold)
        if not scenes:
            scenes = [(0.0, probe.duration_s)]
        if len(scenes) > self.opts.strip_size * 8:
            # too many scenes -> thin to avoid cost blowup; sample a subset
            step = max(1, len(scenes) // 40)
            scenes = scenes[::step]

        all_sets: List[Dict[float, bytes]] = []
        for (s, e) in scenes:
            if (e - s) < 0.5:
                continue
            # spread strip_size sample times across the shot
            n = self.opts.strip_size
            ts = [s + (e - s) * (k + 0.5) / n for k in range(n)]
            ts = [min(max(0.0, t), probe.duration_s - 0.05) for t in ts]
            frames = fi.extract_frames_at(video, ts, probe, self.opts.venue_of_video_scale, 1)
            if frames:
                all_sets.append(frames)

        explicit: Dict[float, bool] = {}
        import concurrent.futures as cf

        def _guess_strip(frames: Dict[float, bytes]):
            metas = self._guess(frames)  # StripClassifier groups into one strip
            return metas

        with cf.ThreadPoolExecutor(max_workers=self.opts.concurrency) as ex:
            results = list(ex.map(_guess_strip, all_sets))

        for frames, metas in zip(all_sets, results):
            self._record(metas)
            stat.total_classified += len(metas)
            for t, m in metas.items():
                explicit[t] = tree_explicit(m)
        stat.l1_frames = len(explicit)
        return explicit

    def run(self, video: Path) -> Tuple[List[Dict[str, object]], ScanStat]:
        t0 = time.time()
        stat = ScanStat()
        stat.prior = self.prior.as_dict() if self.prior else None
        self._metas = {}
        probe = fi.probe_video(video)
        if probe.duration_s <= 0:
            raise ValueError(f"could not determine duration for {video}")

        # Dialogue grounding: load subtitles once, attach to the strip classifier.
        # Cheap (~6-8k text tokens / episode) but provides the narrative context that
        # stops vision-only confabulation in ambiguous scenes.
        subs = fi.extract_subtitles(video) if self.opts.subtitle_grounding else []
        strip_clf = getattr(self.classifier, "strip_size", 0)
        if subs and strip_clf:
            try:
                self.classifier.subtitles = subs  # type: ignore[attr-defined]
                stat.n_subtitle_events = len(subs)
            except AttributeError:
                pass

        # Level 1: coarse scan of the whole timeline
        if self.opts.strip_size > 1:
            coarse = self._level1_strip(video, probe, stat)
        else:
            coarse = self._level1(video, probe, stat)

        # BROAD (default): coarse blocks from L1 flags — cheap, no L2/L3.
        if self.opts.resolution == "broad":
            stat.flagged_timestamps = sorted(t for t, ex in coarse.items() if ex)
            intervals = self._broad_intervals(coarse, probe)
            annotated = self._annotate(intervals)
            stat.elapsed_s = time.time() - t0
            return annotated, stat

        # ---- FINE (deferred): sustained shortcut + sporadic L2/L3 ----
        sustained, sporadic = self._triage_coarse(coarse, probe)

        # ---- Sustained runs: cheap, pin outer edges, absorb interior wholesale ----
        sustained_intervals: List[Dict[str, float]] = []
        demoted: List[float] = []   # sustained runs whose midpoint came back clean -> rescan fine
        for run in sustained:
            try:
                sustained_intervals.extend(self._handle_sustained_run(video, probe, run, stat))
            except _NotSustained:
                demoted.extend(run)

        # ---- Sporadic flags: full L2 dense scan + fine L3 refinement ----
        sporadic_intervals: List[Dict[str, float]] = []
        sporadic_all = sporadic + demoted
        if sporadic_all:
            sporadic_coarse = {t: True for t in sporadic_all}
            sporadic_flags = self._level2(video, probe, stat, sporadic_coarse)
            stat.flagged_timestamps = sporadic_flags
            if sporadic_flags:
                self._refine_boundaries(video, probe, sporadic_flags, stat)
                sporadic_intervals = build_intervals(
                    sporadic_flags, gap_merge=self.opts.gap_merge,
                    lead_in=self.opts.lead_in, trail_out=self.opts.trail_out,
                    min_block=self.opts.min_block, duration_s=probe.duration_s,
                    fps=1.0 / max(self.opts.refine_step, 0.5),
                )

        # ---- Combine + apply lead/trail padding to sustained blocks + merge ----
        padded_sustained: List[Dict[str, float]] = []
        for b in sustained_intervals:
            s = max(0.0, b["start"] - self.opts.lead_in)
            e = min(probe.duration_s, b["end"] + self.opts.trail_out)
            if (e - s) >= self.opts.min_block:
                padded_sustained.append({"start": round(s, 3), "end": round(e, 3)})

        intervals = self._merge_intervals(sporadic_intervals + padded_sustained)
        annotated = self._annotate(intervals)
        stat.elapsed_s = time.time() - t0
        return annotated, stat

    def _merge_intervals(self, intervals: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Merge overlapping/adjacent intervals (within gap_merge) and sort."""
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x["start"])
        merged: List[Dict[str, float]] = [dict(intervals[0])]
        for iv in intervals[1:]:
            last = merged[-1]
            if iv["start"] <= last["end"] + self.opts.gap_merge:
                last["end"] = max(last["end"], iv["end"])
            else:
                merged.append(dict(iv))
        return merged


def load_sidecar(video: Path) -> List[Dict[str, float]]:
    """Load an existing .halal.json sidecar (or fall back to .json for older files)."""
    for suffix in (".halal.json", ".json"):
        p = video.with_suffix(suffix)
        if p.exists():
            data = json.loads(p.read_text())
            if isinstance(data, dict) and "intervals" in data:
                return data["intervals"]
            return data if isinstance(data, list) else []
    return []


def write_sidecar(video: Path, intervals: List[Dict[str, object]]) -> Path:
    """Write Show.S01E01.mkv -> Show.S01E01.halal.json"""
    out = video.with_suffix(".halal.json")
    out.write_text(json.dumps(intervals, indent=2))
    return out