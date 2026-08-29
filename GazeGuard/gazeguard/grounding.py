"""
Episode grounding (P0-A): cheap external priors that STEER search effort — never a verdict.

The prior's only job is to adjust L1 density, default model tier, and the per-episode call
cap. It can NEVER emit "safe, skip" (a hallucinated clean label must be impossible to use
as a pass). `unknown` / API failure / no-web-signal all fall through to the full scan.
Fail-shut is unchanged.

Pipeline of sources, cheapest-first:
  1. TMDB show baseline (optional; needs TMDB_API_KEY) — certification / genre / type.
  2. DoesTheDogDie episode metadata  — CACHED to disk keyed by show|season|episode so the
     5k/month cap is not burned re-querying the same episode. Needs DOES_THE_DOG_DIE_API_KEY.
  3. Gemini flash/lite WITH web-search grounding — one text call per episode →
     structured {risk, notes, sources[]}. This folds DDTD / Fandom / r/Drama pages into one
     prompt rather than many separate API calls.

The per-episode websearch (step 3) is the primary prior. Steps 1 & 2 are cheap fallbacks /
baselines used when step 3 returns `unknown` (or fails).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# The one thing external metadata is allowed to do: multiply L1 density.
# Less risk -> scan fewer coarse frames. `unknown`/failure -> multiplier 1.0 (full scan).
_RISK_DENSITY = {"none": 3.0, "low": 2.0, "medium": 1.0, "high": 0.6, "unknown": 1.0}
# Per-risk cap multiplier (default per-episode cap = GAZEGUARD_MAX_CALLS_PER_RUN).
_RISK_CAP = {"none": 0.4, "low": 0.7, "medium": 1.0, "high": 1.3, "unknown": 1.0}

CACHE_DIR = Path.home() / ".gazeguard" / "cache"
GROUNDING_CACHE = CACHE_DIR / "grounding.json"
_RISKS = ("none", "low", "medium", "high", "unknown")


@dataclass
class EpisodePrior:
    risk: str = "unknown"
    notes: str = ""
    sources: List[str] = field(default_factory=list)
    show_cert: str = ""
    show_genres: List[str] = field(default_factory=list)
    media_type: str = ""          # movie | tv | music_video | unknown
    density_multiplier: float = 1.0   # applied to sample_interval
    cap_multiplier: float = 1.0       # applied to per-episode call cap
    decided_by: str = "none"      # which source set the risk

    def as_dict(self) -> dict:
        return {
            "risk": self.risk,
            "notes": self.notes,
            "sources": self.sources,
            "show_cert": self.show_cert,
            "show_genres": self.show_genres,
            "media_type": self.media_type,
            "density_multiplier": self.density_multiplier,
            "cap_multiplier": self.cap_multiplier,
            "decided_by": self.decided_by,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "EpisodePrior":
        return cls(
            risk=d.get("risk", "unknown"),
            notes=d.get("notes", ""),
            sources=list(d.get("sources") or []),
            show_cert=d.get("show_cert", ""),
            show_genres=list(d.get("show_genres") or []),
            media_type=d.get("media_type", ""),
            density_multiplier=float(d.get("density_multiplier", 1.0)),
            cap_multiplier=float(d.get("cap_multiplier", 1.0)),
            decided_by=d.get("decided_by", ""),
        )


def _norm_risk(r: Optional[str]) -> str:
    """Coerce any risk string to the allowed set (unknown default)."""
    if not r:
        return "unknown"
    r = str(r).strip().lower()
    if r in _RISKS:
        return r
    for cand in _RISKS:           # fuzzy: "medium-high"->medium
        if cand in r:
            return cand
    return "unknown"


def apply_prior(prior: EpisodePrior, sample_interval: float,
                per_episode_cap: Optional[int]) -> tuple:
    """Turn a prior into (sample_interval, cap) — the ONLY surface it may touch."""
    new_interval = sample_interval * prior.density_multiplier
    new_cap = per_episode_cap
    if per_episode_cap is not None:
        new_cap = max(100, int(per_episode_cap * prior.cap_multiplier))
    return new_interval, new_cap


# ---------------------------------------------------------------------------
# Disk cache (atomic, per-episode key) — protects DDTD 5k/month cap and repeat runs.
# ---------------------------------------------------------------------------
def _key(*parts) -> str:
    return "|".join(str(p).strip().lower() for p in parts if p)


def _load_cache() -> Dict[str, dict]:
    if GROUNDING_CACHE.exists():
        try:
            return json.loads(GROUNDING_CACHE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: Dict[str, dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = GROUNDING_CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2))
    os.replace(tmp, GROUNDING_CACHE)


def get_cached_prior(show: str, season: Optional[int], episode: Optional[int]) -> Optional[EpisodePrior]:
    k = _key("ep", show, season, episode)
    c = _load_cache()
    if k in c:
        return EpisodePrior.from_dict(c[k])
    return None


def cache_prior(show: str, season: Optional[int], episode: Optional[int], prior: EpisodePrior) -> None:
    k = _key("ep", show, season, episode)
    c = _load_cache()
    c[k] = prior.as_dict()
    _save_cache(c)


# ---------------------------------------------------------------------------
# TMDB show baseline (optional; needs TMDB_API_KEY). Fails clean -> empty.
# ---------------------------------------------------------------------------
def _tmdb_baseline(show: str) -> Dict:
    key = os.environ.get("TMDB_API_KEY", "")
    if not key:
        return {}
    try:
        q = urllib.parse.quote(show)
        url = f"https://api.themoviedb.org/3/search/tv?api_key={key}&query={q}&page=1"
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.loads(resp.read().decode())
        res = (data.get("results") or [])[:1]
        if not res:
            return {}
        r = res[0]
        return {
            "certification": "",
            "genres": [g.get("name") for g in (r.get("genre_ids") or [])],
            "media_type": "tv",
            "name": r.get("name"),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# DoesTheDogDie (cached). Requires DOES_THE_DOG_DIE_API_KEY. Never uncached-hammered.
# v3 API: search /items?q= -> id, then /items/{id} -> topicItemStats (yes/no votes).
# Returns show-level modesty-relevant topics; cached by title so the 5k/month free cap
# is not burned re-querying the same show.
# ---------------------------------------------------------------------------
_DDTD_HOST = "https://www.doesthedogdie.com/api/v3"
# Topic names we treat as modesty-risk signal (case-insensitive substring).
_DDTD_RISK_TOPICS = (
    "sex", "nude", "nudity", "sexual", "seduct", "strip", "topless", "bare breast",
    "explicit", "porn", "masturbat", "prostitut", "orgy", "affair",
)


def _ddtd_topics(show: str, year: Optional[int] = None) -> Dict:
    key = os.environ.get("DOES_THE_DOG_DIE_API_KEY", "")
    if not key:
        return {}
    ck = _key("ddtd", show, year)
    c = _load_cache()
    if ck in c:
        return c[ck]
    result = {"topics": [], "risk_yes": 0, "risk_no": 0, "found": False}
    try:
        # 1) search for the item by title (response is a bare JSON array). Try with
        #    year first; fall back to bare title if DDTD's entry has no releaseYear.
        queries = []
        if year:
            queries.append(show + f" ({year})")
        queries.append(show)
        items = []
        for qs in queries:
            q = urllib.parse.quote(qs)
            url = f"{_DDTD_HOST}/items?q={q}"
            req = urllib.request.Request(url, headers={"X-API-KEY": key})
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode())
            items = data if isinstance(data, list) else (data.get("items") or [])
            if items:
                break
        if not items:
            _save_cache({**_load_cache(), ck: result})
            return result
        item_id = items[0].get("id")
        # 2) get topic stats for the item (yes/no votes per trigger)
        url2 = f"{_DDTD_HOST}/items/{item_id}"
        req2 = urllib.request.Request(url2, headers={"X-API-KEY": key})
        with urllib.request.urlopen(req2, timeout=25) as resp2:
            d2 = json.loads(resp2.read().decode())
        stats = d2.get("topicItemStats") or []
        topics = []
        risk_yes = risk_no = 0
        for s in stats:
            name = s.get("topicName") or ""
            if not name:
                continue
            topics.append(name)
            if any(w in name.lower() for w in _DDTD_RISK_TOPICS):
                risk_yes += int(s.get("yesSum") or 0)
                risk_no += int(s.get("noSum") or 0)
        result = {
            "topics": topics[:30],
            "risk_yes": risk_yes,
            "risk_no": risk_no,
            "found": bool(topics),
        }
    except Exception:
        result = {"topics": [], "risk_yes": 0, "risk_no": 0, "found": False}
    _save_cache({**_load_cache(), ck: result})  # cached -> never re-burn the 5k cap
    return result


# ---------------------------------------------------------------------------
# Gemini grounded websearch — the primary per-episode prior.
# ---------------------------------------------------------------------------
def _grounded_prompt(show: str, season: Optional[int], episode: Optional[int],
                     show_baseline: Dict) -> str:
    media_type = show_baseline.get("media_type", "")
    genres = ", ".join(show_baseline.get("genres") or []) or "unknown"
    cert = show_baseline.get("certification") or "unknown"
    ident = f"{show}"
    if season is not None and episode is not None:
        ident += f" S{season:02d}E{episode:02d}"
    return (
        "You are a metadata researcher helping decide how much visual scanning effort a "
        "video episode deserves. Do a web search for the given TV show episode and return "
        "a structured judgement about whether it likely contains MODESTY-VIOLATING visual "
        "content: nudity, immodest/revealing attire (swimwear, lingerie, sheer, exposed "
        "cleavage/midriff), explicit or implied sexual/intimate scenes, or objectifying "
        "voyeuristic camera framing (e.g. strip-club / heavy make-out / sex scenes).\n\n"
        f"Show: {ident}\nCertification: {cert}\nGenres: {genres}\nMedia type: {media_type}\n\n"
        "Use episode recaps (Fandom/wikis), parents-guide forums (e.g. doesitencourage.com / "
        "Reddit r/Drama), or topic pages. IMPORTANT: only report what you can actually "
        "verify from search results. If this specific episode has little or no web coverage, "
        "return risk=unknown rather than guessing.\n\n"
        "Respond ONLY with JSON:\n"
        '{"risk": "none|low|medium|high|unknown", "notes": "1-2 sentence summary", '
        '"sources": ["url1","url2"]}'
    )


def ground_episode(show: str, season: Optional[int] = None, episode: Optional[int] = None,
                   *, use_cache: bool = True, model: str = "") -> EpisodePrior:
    """
    Return a per-episode prior, cached. Never raises on network/API failure; any failure
    yields risk=unknown (full scan) so fail-shut is preserved.
    """
    cached = get_cached_prior(show, season, episode) if use_cache else None
    if cached is not None:
        return cached

    base = EpisodePrior(decided_by="grounded-websearch")
    baseline = _tmdb_baseline(show)
    base.show_genres = baseline.get("genres") or []
    base.media_type = baseline.get("media_type", "")

    # 1) grounded websearch (primary). Use the cheap text tier for the prior by default;
    #    a caller can pin an explicit `model`, else GEMINI_GROUNDING_MODEL / flash-lite.
    risk_str, notes, sources = "", "", []
    model = model or os.environ.get("GEMINI_GROUNDING_MODEL", "gemini-2.5-flash-lite")
    prompt = _grounded_prompt(show, season, episode, baseline)
    try:
        llm = _grounded_generate(model, prompt)
        parsed = _parse_grounded(llm)
        if parsed:
            risk_str, notes, sources = parsed
    except Exception:
        risk_str, notes, sources = "", "", []

    # 2) fallback to DDTD baseline if the websearch gave nothing useful
    if not risk_str or risk_str == "unknown":
        d = _ddtd_topics(show)
        if d.get("found"):
            base.notes = "DDTD topics: " + ", ".join(d["topics"][:12])
            base.sources = ["doesitencourage.com / doesthedogdie.com"]
            if d.get("risk_yes", 0) > 0:
                risk_str = "medium"
            elif d.get("risk_no", 0) > 0 and d.get("risk_yes", 0) == 0:
                risk_str = "low"     # topic exists but overwhelmingly voted "no" -> mild risk
            elif base.notes:
                risk_str = "unknown"  # show covered but no modesty-signal topics -> don't guess
            base.decided_by = "ddtd"
        else:
            risk_str = "unknown"
            base.decided_by = "websearch-empty"
    else:
        base.decided_by = "grounded-websearch"

    base.risk = _norm_risk(risk_str)
    base.notes = base.notes or notes
    base.sources = (sources or base.sources)[:8]
    base.density_multiplier = _RISK_DENSITY[base.risk]
    base.cap_multiplier = _RISK_CAP[base.risk]

    if use_cache:
        cache_prior(show, season, episode, base)
    return base


# ---------------------------------------------------------------------------
# Raw grounded generateContent (websearch tool) via stdlib urllib.
# ---------------------------------------------------------------------------
def _grounded_generate(model: str, prompt: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY required for grounded prior")
    base = os.environ.get("GEMINI_BASE", "https://generativelanguage.googleapis.com")
    url = f"{base}/v1beta/models/{model}:generateContent?key={api_key}"
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        # NOTE: responseMimeType=application/json is INCOMPATIBLE with google_search tool
        # (Gemini returns HTTP 400). Ask for JSON in the prompt text instead; the model
        # reliably emits valid JSON for a well-specified schema.
        "generationConfig": {"temperature": 0.0},
    }).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    data = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode())
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"grounding HTTP {e.code}")
        except (urllib.error.URLError, TimeoutError):
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise
    if data is None:
        raise RuntimeError("grounding: exhausted retries")
    cand = (data.get("candidates") or [{}])[0]
    if cand.get("finishReason") in ("SAFETY", "PROHIBITED_CONTENT"):
        raise RuntimeError("grounding safety block")
    parts = (cand.get("content") or {}).get("parts") or []
    # web grounding citations sometimes come as a separate tool part; grab all text
    texts = [p.get("text", "") for p in parts if p.get("text")]
    return "\n".join(texts)


def _parse_grounded(text: str):
    """Return (risk, notes, sources) tuple or None. Defensive."""
    if not text:
        return None
    import re
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    cleaned = re.sub(r"[\r\n\t]+", "", m.group(0))
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    risk = _norm_risk(obj.get("risk"))
    notes = str(obj.get("notes", "")).strip()
    sources = [str(s) for s in (obj.get("sources") or [])][:8]
    return risk, notes, sources