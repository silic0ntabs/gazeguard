import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List
import typer
from rich.console import Console
from .pipeline import CoarseToFinePipeline, Options, write_sidecar
from .classifier import MockClassifier, GeminiClassifier, Classifier
from . import frame_io as fi
from .budget import (load_ledger, save_ledger, Ledger, record_run, check_budget,
                     reset_ledger, BudgetExceeded, estimate_cost, CallBudget)

app = typer.Typer(help="GazeGuard: Compute content-filter sidecars so modest viewers can blur explicit scenes on playback.")
console = Console()


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (stdlib, no python-dotenv)."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = val


# Load .env before anything so GEMINI_API_KEY / budget vars are visible.
load_dotenv()

def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        console.print(f"[yellow]Config not found at {config_path}, using defaults.[/yellow]")
        return {
            "strictness_level": 0.5,
            "fps_sample_rate": 1.0,
            "min_block_duration": 1.0,
            "gap_merge_threshold": 2.0,
            "explicit_classes": ["NSFW"]
        }
    with open(config_path, 'r') as f:
        return json.load(f)


def _video_duration(video: Path) -> Optional[float]:
    """Probe a video's duration in seconds via ffprobe (used by the too-many-blackouts flag)."""
    import subprocess as sp
    try:
        out = sp.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, timeout=60,
        )
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip())
    except Exception:
        return None
    return None

@app.command(name="preprocess")
def preprocess_cmd(
    path: str = typer.Option(..., "--path", help="Path to a video file or directory"),
    sample_interval: Optional[float] = typer.Option(None, "--interval",
        help="Level 1 coarse sample spacing in seconds (default: 12)"),
    precision: Optional[float] = typer.Option(None, "--precision",
        help="Level 3 boundary precision in seconds (default: 0.2)"),
    model: str = typer.Option("mock", "--model",
        help="Classifier backend: 'mock' (default, local) or 'gemini'"),
    gemini_model: str = typer.Option("", "--gemini-model",
        help="Gemini model id when --model gemini (default: GEMINI_MODEL env or gemini-flash-lite)"),
    dry_run: bool = typer.Option(False, "--dry-run",
        help="Build the pipeline but don't flush LLM calls (stats only)"),
    force: bool = typer.Option(False, "--force", help="Re-process even if sidecar exists"),
    budget_rs: Optional[float] = typer.Option(None, "--budget-rs",
        help="Cumulative spend cap in INR (default: GAZEGUARD_BUDGET env, else unlimited)"),
    max_calls: Optional[int] = typer.Option(None, "--max-calls",
        help="Per-run classifier call cap (default: GAZEGUARD_MAX_CALLS_PER_RUN env, else unlimited)"),
    note: str = typer.Option("", "--note", help="Attach a label to this run in the ledger"),
    show: str = typer.Option("", "--show",
        help="Show title for episode grounding (P0-A). Enables websearch prior that steers L1 density"),
    season: Optional[int] = typer.Option(None, "--season", help="Season number for grounding"),
    episode: Optional[int] = typer.Option(None, "--episode", help="Episode number for grounding"),
    no_grounding: bool = typer.Option(False, "--no-grounding",
        help="Disable episode grounding (skip TMDB/DDTD/websearch prior)"),
    config: Path = typer.Option(Path("config.json"), "--config", help="Path to config file"),
):
    """
    Hierarchical Coarse-to-Fine preprocessing. Writes Show.S01E01.halal.json sidecars
    (no re-encoding; the video file is never touched).
    """
    conf = load_config(config)
    opts = Options(
        sample_interval=sample_interval or float(conf.get("coarse", {}).get("sample_interval", 12.0)),
        precision=precision or float(conf.get("coarse", {}).get("precision", 0.2)),
        edge_precision=float(conf.get("coarse", {}).get("edge_precision", 1.0)),
        sustained_min_span_s=float(conf.get("coarse", {}).get("sustained_min_span_s", 24.0)),
        cut_window=float(conf.get("coarse", {}).get("cut_window", 10.0)),
        refine_step=float(conf.get("coarse", {}).get("refine_step", 1.0)),
        gap_merge=float(conf.get("coarse", {}).get("gap_merge", 1.0)),
        lead_in=float(conf.get("coarse", {}).get("lead_in", 0.3)),
        trail_out=float(conf.get("coarse", {}).get("trail_out", 0.2)),
        min_block=float(conf.get("coarse", {}).get("min_block", 0.0)),
        scene_sampling=bool(conf.get("coarse", {}).get("scene_sampling", False)),
        dedup_samples=bool(conf.get("coarse", {}).get("dedup_samples", False)),
        resolution=str(conf.get("coarse", {}).get("resolution", "broad")),
        strip_size=int(conf.get("coarse", {}).get("strip_size", 0) or 0),
        subtitle_grounding=bool(conf.get("coarse", {}).get("subtitle_grounding", False)),
        dry_run=dry_run,
    )

    strip_size = int(conf.get("coarse", {}).get("strip_size", 0) or opts.strip_size or 0)

    if model == "gemini":
        configured_model = (gemini_model or conf.get("coarse", {}).get("model") or "")
        base_classifier: Classifier = GeminiClassifier(model=configured_model)
        if strip_size > 1:
            from .strip import StripClassifier
            classifier = StripClassifier(base_classifier, strip_size=strip_size)
        else:
            classifier = base_classifier
        tag = f" ({classifier.model}, strip={strip_size})" if strip_size > 1 \
            else f" ({classifier.model})"
        console.print(f"[bold]Classifier: Gemini{tag}[/bold]")
    else:
        classifier: Classifier = MockClassifier()
        console.print("[bold]Classifier: Mock (local, no API). "
                      "Use --model gemini with GEMINI_API_KEY for frontier vision.[/bold]")

    # Budget guard (only meaningful for real API use; mock costs nothing but still works).
    ledger = load_ledger()
    run_model = getattr(classifier, "model", "gemini-2.5-flash")
    bgt = budget_rs if budget_rs is not None else _env_float("GAZEGUARD_BUDGET")
    mcr = max_calls if max_calls is not None else _env_int("GAZEGUARD_MAX_CALLS_PER_RUN")
    if model != "mock":
        console.print(f"[yellow]Ledger: {ledger.cumulative_calls} calls / "
                      f"Rs{ledger.cumulative_cost_rs:.2f} spent so far "
                      f"(budget: {'Rs' + ('%.2f' % bgt) if bgt else 'unlimited'}, "
                      f"max-calls: {mcr or 'unlimited'})[/yellow]")
    # Mid-run hard spend guard: wrap the classifier so every batch stops the run at the cap.
    guard = CallBudget(classifier, mcr, note="gemini") if (mcr is not None and model != "mock") else None
    pipe = CoarseToFinePipeline(classifier, opts, call_budget=guard)

    # Too-many-blackouts fail-safe (config-driven). If an episode is blocked for > the
    # episode threshold, flag it; if > the show threshold fraction of episodes in one batch
    # get flagged, flag the SHOW itself. Fail-open (warn/flag), never deletes sidecars.
    flag_cfg = conf.get("flag", {})
    ep_frac = float(flag_cfg.get("episode_blocked_fraction", 0.60))
    show_min_flagged_episodes = int(flag_cfg.get("show_flag_min_episodes", 2))
    show_flagged_ratio = float(flag_cfg.get("show_flag_ratio", 0.50))
    flagged_episodes = 0
    processed_episodes = 0

    path_str = path.strip("'\" ")
    path_obj = Path(path_str).expanduser().resolve()
    videos = _collect_videos(path_obj)

    if not videos:
        console.print("[yellow]No supported video files found.[/yellow]")
        return

    console.print(f"[bold green]Found {len(videos)} video(s) to preprocess.[/bold green]")
    total_calls = 0
    for video in videos:
        sidecar = video.with_suffix(".halal.json")
        if sidecar.exists() and not force:
            console.print(f"[blue]Skipping {video.name} (already has .halal.json). Use --force.[/blue]")
            continue
        console.print(f"\n[bold]Processing: {video.name}[/bold]")

        # Episode grounding (P0-A): steers L1 density / cap from a cheap prior.
        if show and not no_grounding and model == "gemini":
            try:
                pipe.grounded(show, season, episode)
                prior = pipe.prior
                if prior:
                    console.print(f"  [dim]prior: risk={prior.risk} (density x{prior.density_multiplier}, "
                                  f"cap x{prior.cap_multiplier}) via {prior.decided_by}[/dim]")
                    # grounding may have lowered opts.per_episode_cap -> rebuild the guard
                    if opts.per_episode_cap and pipe.call_budget is not None:
                        pipe.call_budget = CallBudget(classifier, opts.per_episode_cap, note="grounded")
            except Exception as e:
                console.print(f"  [dim]grounding skipped ({e}); full scan[/dim]")
        elif show and not no_grounding and model != "gemini":
            console.print("  [dim]grounding requires --model gemini (prior needs a websearch call); full scan[/dim]")

        try:
            intervals, stat = pipe.run(video)
        except BudgetExceeded as be:
            console.print(f"  [red]Budget guard stopped the run: {be}[/red]")
            # write current ledger before bailing
            save_ledger(ledger)
            continue
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
            continue

        total_calls += stat.total_classified
        # enforce per-run call cap AFTER the fact (cheap safety net when no pre-estimate)
        if mcr is not None and stat.total_classified > mcr:
            console.print(f"  [red]Run used {stat.total_classified} calls > cap {mcr}. "
                          f"Sidecar kept but consider tuning coarse settings.[/red]")

        sidecar = write_sidecar(video, intervals)
        if intervals:
            console.print(f"  [green]{len(intervals)} explicit interval(s).[/green]")
        else:
            console.print("  [white]No explicit content detected.[/white]")
        console.print(f"  Saved [bold]{sidecar}[/bold]")
        console.print(f"  Frames: L1 coarse={stat.l1_frames}, refined={stat.l2_frames}, "
                      f"L3 calls={stat.l3_calls} | classified={stat.total_classified} "
                      f"| {stat.elapsed_s:.1f}s")

        # Too-many-blackouts episode flag
        processed_episodes += 1
        dur = _video_duration(video)
        if dur and dur > 0:
            blocked = sum(float(iv.get("end", 0)) - float(iv.get("start", 0)) for iv in intervals)
            frac = blocked / dur
            if frac > ep_frac:
                flagged_episodes += 1
                console.print(f"  [red]FLAG episode: {video.name} is {frac*100:.0f}% blocked "
                              f"(> {ep_frac*100:.0f}%). Higher than expected for ordinary "
                              f"content — review or use MINIMAL.[/red]")

    # Too-many-blackouts SHOW flag (after all episodes in this batch)
    if show and processed_episodes >= show_min_flagged_episodes and \
            (flagged_episodes / processed_episodes) >= show_flagged_ratio:
        console.print(f"\\n[bold red]FLAG SHOW: {show} — {flagged_episodes}/"
                      f"{processed_episodes} episodes over-blocked.This is a red flag for "
                      f"the whole series; consider MINIMAL or manual review.[/bold red]")

    if model != "mock" and total_calls:
        run_cost = record_run(ledger, total_calls, run_model, note)
        console.print(f"\n[bold green]Preprocessing complete. "
                      f"This batch: {total_calls} calls / ~Rs{run_cost:.2f}. "
                      f"Total: Rs{ledger.cost_rs:.2f} across {ledger.cumulative_calls} calls.[/bold green]")
    else:
        console.print("\n[bold green]Preprocessing complete![/bold green]")

def _collect_videos(path_obj: Path) -> List[Path]:
    """Resolve a file or directory into a list of video paths."""
    video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".m4v"}
    if path_obj.is_file():
        return [path_obj] if path_obj.suffix.lower() in video_extensions else []
    if path_obj.is_dir():
        out = []
        for root, _, files in os.walk(path_obj):
            for f in files:
                if Path(f).suffix.lower() in video_extensions:
                    out.append(Path(root) / f)
        return out
    return []

def _env_float(name: str) -> Optional[float]:
    v = os.environ.get(name)
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _env_int(name: str) -> Optional[int]:
    v = os.environ.get(name)
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        return None


@app.command(name="budget")
def budget_cmd(
    action: str = typer.Option("show", "--action",
        help="'show' (default) to view the ledger, or 'reset' to zero it"),
):
    """
    Show or reset the local spend ledger.
    """
    ledger = load_ledger()
    if action == "reset":
        reset_ledger()
        console.print("[green]Ledger reset to zero.[/green]")
        return
    console.print(f"[bold]GazeGuard spend ledger[/bold]")
    console.print(f"  Model:           {ledger.model}")
    console.print(f"  Cumulative calls: {ledger.cumulative_calls}")
    console.print(f"  Estimated cost:   Rs{ledger.cost_rs:.2f}")
    console.print(f"  Recent runs:")
    for h in ledger.history[-10:]:
        console.print(f"    - {h['calls']} calls, Rs{h['cost']:.2f} (total Rs{h['total']:.2f})"
                      + (f" [{h['note']}]" if h.get('note') else ""))
    console.print("[dim]Estimate only; Gemini bills per token, not per call.[/dim]")


@app.command()
def version():
    """Show the version of GazeGuard."""
    console.print("GazeGuard v0.1.0")

if __name__ == "__main__":
    app()
