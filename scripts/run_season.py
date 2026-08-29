#!/usr/bin/env python3
"""
GazeGuard season-batch utility: preprocess a folder of episodes overnight.

For each episode it:
  1. Parses {show, season, episode} from the filename (robust regex + fuzzy fallback —
     filenames are messy: "Show.S01E02.mkv", "Show - s1e2.mp4", "Show.S1.E2.mkv", etc.)
  2. Runs the sync pipeline with grounding (websearch prior for the show/ep) + strips +
     broad resolution — the proven cheap/fast default.
  3. Writes a .halal.json sidecar per episode and records spend to the ledger.
  4. Applies the too-many-blackouts fail-safe at episode AND show level.

Genuinely zero new infrastructure (unlike the Batch API path, which needs GCS). At
~$0.05 for a 47-min episode this is already affordable for overnight runs.

Usage:
    python scripts/run_season.py --dir /path/to/season \
        [--show "The Americans"] [--season 1] \
        [--model gemini] [--cap 500] [--force]
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        if k and k not in os.environ:
            os.environ[k] = v.strip().strip("'\"\"")


_load_dotenv(ROOT / ".env")

from gazeguard.pipeline import CoarseToFinePipeline
from gazeguard.classifier import GeminiClassifier, MockClassifier
from gazeguard.strip import StripClassifier
from gazeguard.budget import CallBudget, load_ledger, save_ledger, record_run, BudgetExceeded
from gazeguard.pipeline import load_sidecar

VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".mov", ".webm", ".m4v", ".ts"}
_ILLEGAL = re.compile(r"[<>:/\\|?*\"]")


def parse_episode(filename: str, fallback_show: str = "", fallback_season: int = 1):
    """Best-effort {show, season, episode} from a filename. Never-None episode."""
    stem = Path(filename).stem
    # SxxEyy / sxxeyy / S1E2 / S1.E2 (upper/lower, optional dots/spaces)
    m = re.search(r"[sS](\d{1,2})\s*\.?\s*[eE](\d{1,3})", stem)
    if m:
        season, ep = int(m.group(1)), int(m.group(2))
    else:
        # "1x02" or "102" (season 1 ep 02) style
        m = re.search(r"(\d{1,2})x(\d{1,3})", stem)
        if m:
            season, ep = int(m.group(1)), int(m.group(2))
        else:
            m = re.search(r"(?:^|[^0-9])(\d{1,2})(\d{2})(?:[^0-9]|$)", stem)
            season, ep = (int(m.group(1)), int(m.group(2))) if m else (None, None)
    # show name = everything before the season marker (or fallback)
    show = None
    idx = re.search(r"[sS]\d{1,2}\s*\.?\s*[eE]\d{1,3}|\d{1,2}x\d{1,3}|(?:^|[^0-9])\d{1,2}\d{2}(?:[^0-9]|$)", stem)
    if idx:
        show = _clean_show(stem[:idx.start()])
    show = (show or fallback_show or "").strip()
    if season is None:
        season = fallback_season
    ep = ep if ep is not None else 1
    return show, season, ep


def _clean_show(raw: str) -> str:
    raw = re.sub(r"[_\-.]", " ", raw)
    raw = raw.strip()
    # drop common suffixes that slip in
    for tok in ("WEB", "BluRay", "1080p", "720p", "2160p", "HDR", "x264", "x265",
                "HEVC", "AAC", "DDP5", "NF", "AMZN", "Si", "UHD"):
        raw = re.sub(r"\b" + tok + r"\b", " ", raw, flags=re.I)
    return re.sub(r"\s+", " ", raw).strip()


def list_episodes(dir_path: Path, force: bool) -> list:
    out = []
    if dir_path.is_file():
        dir_path = dir_path.parent
    for f in sorted(dir_path.iterdir()):
        if f.suffix.lower() in VIDEO_EXTS:
            sidecar = f.with_suffix(".halal.json")
            if sidecar.exists() and not force:
                continue  # already done
            out.append(f)
    return out


def build_pipeline(model: str, gemini_model: str, strip_size: int, cap: int, note: str):
    """Per-episode pipeline; cap is per-episode and ledger updated on completion."""
    if model == "gemini":
        base = GeminiClassifier(model=gemini_model)
        clf = StripClassifier(base, strip_size=strip_size) if strip_size > 1 else base
        guard = CallBudget(clf, cap, note=note) if cap else None
    else:
        clf = MockClassifier()
        guard = None
    return clf, guard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Season folder (or a single video file)")
    ap.add_argument("--show", default="", help="Show title override (else parsed from filename)")
    ap.add_argument("--season", type=int, default=None, help="Season override (else parsed)")
    ap.add_argument("--model", default="mock", choices=["mock", "gemini"])
    ap.add_argument("--gemini-model", default="")
    ap.add_argument("--strip-size", type=int, default=4)
    ap.add_argument("--cap", type=int, default=0,
                    help="Per-episode call cap (0 = GAZEGUARD_MAX_CALLS_PER_RUN / unlimited)")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--note", default="season-batch")
    ap.add_argument("--force", action="store_true", help="Re-scan episodes that already have sidecars")
    args = ap.parse_args()

    import json
    conf = json.loads(Path(args.config).read_text())
    c = conf.get("coarse", {})
    from gazeguard.pipeline import Options as _Opts
    opts = _Opts(
        sample_interval=float(c.get("sample_interval", 12.0)),
        cut_window=float(c.get("cut_window", 10.0)),
        refine_step=float(c.get("refine_step", 1.0)),
        precision=float(c.get("precision", 0.2)),
        edge_precision=float(c.get("edge_precision", 1.0)),
        sustained_min_span_s=float(c.get("sustained_min_span_s", 24.0)),
        gap_merge=float(c.get("gap_merge", 1.0)),
        lead_in=float(c.get("lead_in", 0.3)),
        trail_out=float(c.get("trail_out", 0.2)),
        min_block=float(c.get("min_block", 0.0)),
        scene_sampling=bool(c.get("scene_sampling", False)),
        dedup_samples=bool(c.get("dedup_samples", False)),
        resolution=str(c.get("resolution", "broad")),
        strip_size=args.strip_size if args.strip_size > 1 else int(c.get("strip_size", 0) or 0),
        subtitle_grounding=bool(c.get("subtitle_grounding", False)),
    )
    # mask strip entirely if --strip-size 0
    if args.strip_size <= 1:
        opts.strip_size = 0

    cap = args.cap or int(os.environ.get("GAZEGUARD_MAX_CALLS_PER_RUN", "0") or 0) or None
    ledger = load_ledger()
    episodes = list_episodes(Path(args.dir).expanduser(), args.force)
    if not episodes:
        print("No episodes to process (all have sidecars; use --force to redo).")
        return
    print(f"=== season-batch: {len(episodes)} episode(s) | model={args.model} "
          f"strip={opts.strip_size} resolution={opts.resolution} cap={cap or 'unlimited'} ===\n")

    flagged = 0
    total_calls = 0
    for video in episodes:
        show, season, ep = parse_episode(video.name, args.show)
        show = args.show or show or "unknown"
        if args.season is not None:
            season = args.season
        note = f"{args.note}-{show}-s{season}e{ep}" if show != "unknown" else args.note
        print(f"[RUN] {video.name}  ->  {{show: '{show}', s{season}e{ep}}}")

        clf, guard = build_pipeline(args.model, args.gemini_model, opts.strip_size,
                                    cap, note=note)
        pipe = CoarseToFinePipeline(clf, opts, call_budget=guard)
        if args.model == "gemini":
            try:
                pipe.grounded(show, season, ep)
                prior = pipe.prior
                if prior:
                    print(f"      prior: risk={prior.risk} ({prior.decided_by})")
            except Exception as e:
                print(f"      grounding skipped ({e}); full scan")

        t0 = time.time()
        try:
            intervals, stat = pipe.run(video)
        except BudgetExceeded as be:
            print(f"  [CAP] {be}")
            continue
        save_ledger(ledger)
        total_calls += stat.total_classified

        from gazeguard.pipeline import write_sidecar
        sidecar = write_sidecar(video, intervals)
        blocked = sum(float(i.get("end", 0)) - float(i.get("start", 0)) for i in intervals)
        from gazeguard.main import _video_duration
        dur = _video_duration(video) or 0.0
        frac = (blocked / dur) if dur else 0.0
        flag_cfg = conf.get("flag", {})
        ep_frac = float(flag_cfg.get("episode_blocked_fraction", 0.60))
        mark = ""
        if frac > ep_frac:
            flagged += 1
            mark = f"  [FLAG] {frac*100:.0f}% blocked > {ep_frac*100:.0f}%"
        print(f"      {len(intervals)} intervals, {blocked:.0f}s (~{frac*100:.0f}%) "
              f"blocked | {stat.total_classified} calls | {time.time()-t0:.0f}s{mark}")

    if args.model != "mock" and total_calls:
        run_model = args.gemini_model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        cost = record_run(ledger, total_calls, run_model, note=args.note)
        print(f"\nseason-batch done: {total_calls} calls, ~Rs{cost:.2f}. "
              f"Ledger total Rs{ledger.cumulative_cost_rs:.2f}")
    else:
        print(f"\nseason-batch done: {total_calls} mock calls (no spend).")
    if flagged:
        print(f"\n{flagged}/{len(episodes)} episodes over-blocked. Re-run at MINIMAL profile "
              f"or review — see DECISIONS.md.")


if __name__ == "__main__":
    main()