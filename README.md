# GazeGuard

**GazeGuard** precomputes a content-filter sidecar for video so viewers can blur explicit
or objectionable scenes at playback — without re-encoding, and while keeping **audio and
subtitles running** the whole time.

A small [mpv](https://mpv.io)/IINA Lua script reads the sidecar and applies a **blur**
(strong for confirmed-explicit content, mild for suggestive-but-uncertain content) over the
flagged intervals. No frames are ever edited; nothing is re-encoded.

It's built for **TV series and movies** and is cheap to run — roughly **$0.05–0.07 per
45-minute episode** on the default vision tier.

---

## How it works

**Model describes; pipeline decides.** A vision model returns location-agnostic descriptive
metadata for each scene (exposure, intimacy, situation, camera intent). A deterministic
policy engine maps that into a screen action per profile. This keeps the model honest and
lets you re-tune sensitivity **at playback at no additional cost** by cycling profiles.

```
video ──► grounding (episode prior) ──► scene detection ──► strip of frames per shot
                                             │
                                             └─► vision model: descriptive metadata
                                             │
                                   policy engine ──► {start,end,meta} sidecar
                                             │
                                   GazeGuard.lua (mpv) ──► blur / pass at playback
```

Key properties:

- **Broad by default.** L1 samples scenes; flagged regions become a few wide merged blocks.
  No millisecond edge refinement needed — a blur is a blur. This is what keeps it ~$0.05–0.07
  per episode instead of ~$1+.
- **Context strips.** Consecutive frames per shot are sent in one call, so the model judges
  the *scene* (setting, motion, posture), not a single ambiguous frame.
- **Dialogue grounding.** Subtitles are injected alongside the frames (a few cents of text
  tokens per episode), giving the model the narrative context that prevents inventing
  intimacy where none exists.
- **Grounding steers effort, never decides.** Episode metadata (TMDB / DoesTheDogDie /
  web search) only adjusts scan density and spend cap — it never says "safe, skip."
- **Fail-open + safe.** Any API/safety error fails the frame *shut* (blur), never a silent
  pass. A too-many-blackouts heuristic flags episodes/shows that look over-blocked.

---

## Install

Requires **Python 3.12+** and **FFmpeg**.

```bash
git clone https://github.com/<you>/gazeguard.git
cd gazeguard
python3 -m venv venv && source venv/bin/activate
pip install -e .
```

Then create `.env` from `.env.example` and add a
[Gemini API key](https://aistudio.google.com/app/apikey) (the pipeline works without one in
mock mode — it just verifies logic with no cost):

```bash
cp .env.example .env   # fill in GEMINI_API_KEY
```

---

## Usage

```bash
# Process a video or a whole folder of episodes (mock: no API key, verifies logic)
gazeguard preprocess --path "Movies/Series"

# Real vision + episode grounding
gazeguard preprocess --path "ep.mkv" --model gemini \
  --show "The Americans" --season 1 --episode 2

# Whole season overnight (parses show/season/episode from filenames)
python scripts/run_season.py --dir /path/to/season --model gemini --cap 500

# Spend ledger (the Gemini API has no native cap on a plain key)
gazeguard budget --action show
```

### Playback

Drop `gazeguard_blackout.lua` into your mpv scripts folder
(`~/.config/mpv/scripts/`). The sidecar (`Show.Ep.halal.json`) sits next to the video and is
auto-loaded.

- **`Ctrl+B`** — cycle `STRICT → BALANCED → MINIMAL` live
- **`Shift+B`** — manually unlock

---

## Configuration

All tuning lives in `config.json` — no code edits:

| Key | Default | Meaning |
|---|---|---|
| `resolution` | `broad` | `broad` (cheap) or `fine` (millisecond edge refinement) |
| `strip_size` | `4` | frames per context call (0 = single frame) |
| `scene_sampling` | `true` | sample one strip per detected scene |
| `dedup_samples` | `true` | drop near-identical frames before classification |
| `subtitle_grounding` | `true` | inject dialogue into strip prompts |
| `model` | `gemini-2.5-flash` | vision model tier |
| `flag.*` | — | too-many-blackouts fail-safe thresholds |

Spend guards live in `.env`: `GAZEGUARD_BUDGET` (hard cumulative cap) and
`GAZEGUARD_MAX_CALLS_PER_RUN` (per-scrape hard cap). The ledger is at
`~/.gazeguard/ledger.json`.

---

## Design notes

- **No re-encoding, ever.** The source file is read-only; only the small sidecar is written.
- **Location-agnostic.** It never keys off "bed" or "car" — an encounter can happen
  anywhere. The model reports *intimacy*, not furniture.
- **Model doesn't judge.** It describes severity and situation; the policy (Python, mirrored
  1:1 in the Lua) decides blackout/blur/pass.
- **False negatives are preferred over false positives.** An innocent scene passing is
  tolerable; blurring an innocent scene (a couple talking) is the
  failure we optimise against. Over-blocked episodes are recoverable by dropping to a lower
  profile at playback.

## Project layout

```
gazeguard/            core package (pipeline, classifier, policy, grounding, budget…)
gazeguard_blackout.lua  mpv/IINA playback hook
scripts/run_season.py   folder → overnight sidecars
config.json, .env.example
```

---

License: MIT. Contributions welcome — this is early-stage but functional and the
design/decisions are documented so others can pick it up.
