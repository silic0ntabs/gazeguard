"""
Frame I/O and video probing.

Thin wrapper around ffprobe / ffmpeg to:
  * probe duration + fps
  * extract individual frames at ~480p (of the video's native resolution) as JPEG bytes
  * detect scene-cut timestamps within a time window (for Level 2 micro-cut resolution)

All calls are subprocess -> ffmpeg, no external SDK dependency.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, check=True)


def which_binary(name: str) -> Optional[str]:
    import shutil

    for base in ["/usr/local/bin", "/opt/homebrew/bin", "/usr/bin"]:
        p = Path(base) / name
        if p.exists():
            return str(p)
    return shutil.which(name)


def which_ffmpeg() -> str:
    p = which_binary("ffmpeg")
    if not p:
        raise RuntimeError("ffmpeg not found on PATH. Install it first (brew install ffmpeg).")
    return p


def which_ffprobe() -> str:
    p = which_binary("ffprobe")
    if not p:
        raise RuntimeError("ffprobe not found on PATH. Install it first (brew install ffmpeg).")
    return p


@dataclass
class Probe:
    duration_s: float
    fps: float
    width: int
    height: int


def probe_video(video: Path) -> Probe:
    """ffprobe duration / fps / resolution."""
    cmd = [
        which_ffprobe(),
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration",
        "-of", "json",
        str(video),
    ]
    out = _run(cmd).stdout
    data = json.loads(out)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {video}")
    s = streams[0]
    w = s.get("width") or 0
    h = s.get("height") or 0
    fps_num, fps_den = 0.0, 1.0
    if s.get("r_frame_rate"):
        try:
            n, d = s["r_frame_rate"].split("/")
            fps_num, fps_den = float(n), float(d) or 1.0
        except ValueError:
            fps_num, fps_den = 0.0, 1.0
    fps = fps_num / fps_den if fps_num else 0.0
    duration = float(s.get("duration") or 0.0)
    # fallback to format duration
    if not duration:
        cmd = [
            which_ffprobe(), "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", str(video),
        ]
        fmt = json.loads(_run(cmd).stdout)
        duration = float(fmt.get("format", {}).get("duration") or 0.0)
    return Probe(duration, fps, w, h)


def _scale_filter(width: int, height: int, max_dim: int = 480) -> str:
    """Scale to keep tokens small (~480p short edge) while preserving aspect ratio."""
    if not width or not height:
        return "scale=-2:480"
    scale = f"scale='min({max_dim},iw)':-2"
    if width > height:
        scale = f"scale=-2:'min({max_dim},ih)'"
    return scale


def extract_frame(video: Path, timestamp: float, width: int, height: int,
                  max_dim: int = 480) -> Optional[bytes]:
    """Extract a single frame at `timestamp` as JPEG bytes (None if the video is too short)."""
    timestamp = max(0.0, timestamp)
    scale = _scale_filter(width, height, max_dim)
    cmd = [
        which_ffmpeg(),
        "-v", "error",
        "-ss", f"{timestamp:.3f}",
        "-i", str(video),
        "-frames:v", "1",
        "-vf", f"{scale},format=yuvj420p",
        "-f", "image2pipe",
        "-c:v", "mjpeg",
        "-q:v", "5",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except subprocess.SubprocessError:
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    return proc.stdout


def extract_frames_at(video: Path, timestamps: List[float], probe: Probe,
                      max_dim: int = 480, concurrency: int = 4) -> dict:
    """
    Extract frames at a list of timestamps.

    Returns {timestamp: jpeg_bytes}. Missing/out-of-range timestamps are skipped.
    Concurrency is capped so we never spawn unbounded ffmpeg processes.
    """
    from concurrent.futures import ThreadPoolExecutor

    results: dict = {}
    unique = sorted({round(t, 3) for t in timestamps if 0.0 <= t < probe.duration_s})

    def _one(t: float):
        b = extract_frame(video, t, probe.width, probe.height, max_dim)
        if b:
            return (t, b)
        return (t, None)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for t, b in ex.map(_one, unique):
            if b:
                results[t] = b
    return results


def detect_scene_cuts(video: Path, start_s: float, end_s: float,
                      threshold: float = 0.30) -> List[float]:
    """
    Probe the QUICK scene transitions in [start_s, end_s) using ffmpeg's scene filter.

    Returns cut timestamps (rising edges crossing `threshold`). Used for Level 2:
    within a flagged envelope, find the actual cut points to refine the search window.
    """
    start_s = max(0.0, start_s)
    end_s = min(end_s, float(probe_video(video).duration_s))
    if end_s <= start_s:
        return []
    scale = _scale_filter(probe_video(video).width, probe_video(video).height, 480)
    cmd = [
        which_ffmpeg(),
        "-v", "info",
        "-ss", f"{start_s:.3f}",
        "-to", f"{end_s:.3f}",
        "-i", str(video),
        "-vf", f"{scale},select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0:
            return []
    except subprocess.SubprocessError:
        return []
    cuts: List[float] = []
    # parse showinfo pts_time from stderr
    import re

    for line in proc.stderr.decode("utf-8", "replace").splitlines():
        m = re.search(r"pts_time:([0-9.]+)", line)
        if m:
            t = start_s + float(m.group(1))
            cuts.append(round(t, 3))
    return sorted(set(cuts))


def detect_scenes(video: Path, threshold: float = 0.30, min_len_s: float = 0.5) -> List[tuple]:
    """
    Detect ALL scene boundaries across the whole video (one ffmpeg pass, CPU-only, ₹0).
    Returns a list of (start_s, end_s) scene intervals covering [0, duration].
    Used for P1 scene-aware L1 sampling: one frame per scene is enough to represent a shot.
    """
    probe = probe_video(video)
    if probe.duration_s <= 0:
        return [(0.0, 0.0)]
    cuts = detect_scene_cuts(video, 0.0, probe.duration_s, threshold)
    bounds = [0.0] + cuts + [probe.duration_s]
    scenes: List[tuple] = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if (e - s) < min_len_s:
            # absorb tiny slivers into the previous scene
            if scenes:
                scenes[-1] = (scenes[-1][0], e)
            continue
        scenes.append((round(s, 3), round(e, 3)))
    return scenes


def scene_representative_times(video: Path, threshold: float = 0.30) -> List[float]:
    """One representative timestamp per detected scene, used for scene-aware L1 sampling."""
    scenes = detect_scenes(video, threshold)
    return [round((s + e) / 2.0, 3) for s, e in scenes if e > s]


def perceptual_hash(jpeg: bytes, bits: int = 64) -> Optional[bytes]:
    """
    Cheap perceptual (aHash/dHash-ish) similarity signature for deduping near-identical
    samples (slow pan / crossfade / held shot). Returns bytes(8). None if undecodable.
    """
    try:
        import numpy as np

        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img = __import__("cv2").imdecode(arr, __import__("cv2").IMREAD_GRAYSCALE)
        if img is None:
            return None
        small = __import__("cv2").resize(img, (9, 8), interpolation=__import__("cv2").INTER_AREA)
        # dHash: compare adjacent pixel columns; 8x8 diff -> 64 bits
        diff = small[:, 1:] > small[:, :-1]
        bits_arr = diff.flatten()[:bits]
        out = 0
        for b in bits_arr:
            out = (out << 1) | int(b)
        return out.to_bytes((bits + 7) // 8, "big")
    except Exception:
        return None


def _hamming(a: bytes, b: bytes) -> int:
    return sum(bin(x ^ y).count("1") for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Subtitle grounding (dialogue context for the strip classifier)
# ---------------------------------------------------------------------------
def extract_subtitles(video: Path, srt_path: Optional[Path] = None) -> List[tuple]:
    """
    Load subtitle events as [(start_s, end_s, text), ...] for narrative grounding.

    Source priority:
      1. an explicit .srt/.vtt sidecar next to the video (same stem),
      2. any subtitle stream embedded in the video (extracted once, cached to a temp .srt).

    Returns an empty list if no subtitles exist (classification then runs vision-only).
    """
    candidates = []
    if srt_path is not None:
        candidates.append(srt_path)
    # adjacent sidecar, same stem
    for ext in (".srt", ".vtt", ".ass", ".ssa"):
        p = video.with_suffix(ext)
        if p.exists():
            candidates.append(p)
    for p in candidates:
        try:
            events = parse_subtitles(p)
            if events:
                return events
        except Exception:
            continue

    # embedded subtitle stream -> extract once to a temp .srt
    try:
        tmp = video.with_suffix(".extracted.srt")
        r = _run_ffmpeg_extract_subs(video, tmp)
        if r and tmp.exists():
            events = parse_subtitles(tmp)
            if events:
                return events
    except Exception:
        pass
    return []


def _run_ffmpeg_extract_subs(video: Path, out: Path) -> bool:
    import subprocess as sp
    r = sp.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video),
         "-map", "0:s:0?", "-c:s", "srt", str(out)],
        capture_output=True, text=True, timeout=120)
    return r.returncode == 0


def _parse_timestamp(t: str) -> float:
    """'HH:MM:SS,mmm' or 'HH:MM:SS.mmm' or 'MM:SS.mmm' -> seconds."""
    t = t.strip().replace(",", ".")
    parts = t.split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = "0", parts[0], parts[1]
    else:
        h, m, s = "0", "0", parts[0]
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_subtitles(path: Path) -> List[tuple]:
    """
    Parse .srt or .vtt into [(start_s, end_s, text), ...].
    Consecutive cue blocks are joined so one line == one spoken segment.
    """
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".vtt" or "WEBVTT" in text[:20].upper():
        return _parse_vtt(text)
    return _parse_srt(text)


def _parse_srt(text: str) -> List[tuple]:
    events = []
    # block: index \\n start --> end \\n text...
    block_re = re.compile(r"(?:\d+\s*\n)?([0-9:,\.]+)\s*-->\s*([0-9:,\.]+)(.*?)(?=(\n\s*\d+\s*\n[0-9:,\.]+\s*-->)|$)",
                          re.S)
    for m in re.finditer(block_re, text):
        start_t, end_t, body = m.group(1), m.group(2), m.group(3)
        cue = " ".join(line.strip() for line in body.strip().splitlines() if line.strip())
        # strip vtt position tags / HTML
        cue = re.sub(r"<[^>]+>", "", cue).replace("&nbsp;", " ").replace("\\N", " ").strip()
        if not cue:
            continue
        try:
            events.append((_parse_timestamp(start_t), _parse_timestamp(end_t), cue))
        except Exception:
            continue
    return events


def _parse_vtt(text: str) -> List[tuple]:
    events = []
    dia = re.findall(
        r"([0-9:\.]{4,}(?::[0-9:\.]+)?)\s*-->\s*([0-9:\.]{4,}[0-9:\.]?)(.*?)(?=(\n[0-9:\.]+\s*-->)|$)",
        text, re.S)
    for start_t, end_t, body in dia:
        cue = " ".join(line.strip() for line in body.strip().splitlines() if line.strip())
        cue = re.sub(r"<[^>]+>", "", cue).strip()
        if not cue or "-->" in cue:
            continue
        try:
            events.append((_parse_timestamp(start_t), _parse_timestamp(end_t), cue))
        except Exception:
            continue
    return events


def subtitle_slice(events: List[tuple], start: float, end: float,
                   pad: float = 1.0, max_speakers: int = 6) -> str:
    """
    Collect dialogue overlapping [start, end] (padded a little) into one readable line.

    Returns plain text to inject into the strip prompt, or '' if no dialogue. This is the
    missing narrative context: "Senator, the KGB intercepted the courier in Berlin" tells
    the model this is a political briefing, not a hotel-room encounter.
    """
    hits = []
    for (s, e, t) in events:
        if s <= end + pad and e >= start - pad:
            hits.append((s, t))
    if not hits:
        return ""
    hits.sort()
    lines = [f"[{int(round(s))}] {t}" for s, t in hits[:max_speakers]]
    return " ".join(lines)