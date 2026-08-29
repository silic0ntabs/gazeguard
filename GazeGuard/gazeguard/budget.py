"""
Local spend ledger + budget guard.

The Gemini (and most) vision APIs have NO native hard cash-cap on a plain API key, so
'protecting the bank account' must be a local-loc condition. This module keeps a durable
ledger of cumulative classifier calls + estimated cost and enforces a configurable cap.

The ledger file lives at ~/.gazeguard/ledger.json and is only ever written atomically.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

LEDGER_DIR = Path.home() / ".gazeguard"


@dataclass
class ModelPrice:
    """Pricing per 1M tokens ($) + tokens-per-480p-frame estimate."""
    input_per_1m: float
    output_per_1m: float
    tokens_per_frame: int = 258  # ~1 480p JPEG
    prompt_tokens: int = 100     # shared system prompt overhead
    output_tokens: int = 25      # tiny structured JSON reply
    batch_discount: float = 1.0  # 0.5 if we ever send through the real Batch API


# Source-of-truth pricing. Keep input/output matching the doc's tier table.
PRICES = {
    "gemini-2.5-flash-lite": ModelPrice(0.10, 0.40),
    "gemini-2.5-flash":       ModelPrice(0.30, 2.50),
    "gemini-3.7-flash":      ModelPrice(0.75, 3.75, batch_discount=0.5),
}

# fallback for unknown model names (conservative: use flash pricing)
_DEFAULT_PRICE = PRICES["gemini-2.5-flash"]


def fx_rate() -> float:
    """INR per USD. Allow override via env; default 95.5."""
    try:
        return float(os.environ.get("GAZEGUARD_FX", "95.5"))
    except ValueError:
        return 95.5


def estimate_cost(calls: int, model: str) -> float:
    """Estimated cost in INR for `calls` classifier calls against a model."""
    p = PRICES.get(model.split(":")[-1]) or _DEFAULT_PRICE
    in_tok = p.tokens_per_frame * calls + p.prompt_tokens * calls
    out_tok = p.output_tokens * calls
    usd = (in_tok / 1e6 * p.input_per_1m * p.batch_discount) + \
          (out_tok / 1e6 * p.output_per_1m * p.batch_discount)
    return usd * fx_rate()


@dataclass
class Ledger:
    model: str = "gemini-2.5-flash"
    cumulative_calls: int = 0
    cumulative_cost_rs: float = 0.0
    history: list = field(default_factory=list)

    @property
    def cost_rs(self) -> float:
        return round(self.cumulative_cost_rs, 2)


def ledger_path() -> Path:
    return LEDGER_DIR / "ledger.json"


def load_ledger() -> Ledger:
    p = ledger_path()
    if p.exists():
        try:
            data = json.loads(p.read_text())
            return Ledger(**{k: data.get(k) for k in ("model", "cumulative_calls", "cumulative_cost_rs", "history")})
        except (json.JSONDecodeError, TypeError):
            pass
    return Ledger()


def save_ledger(ledger: Ledger) -> None:
    p = ledger_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "model": ledger.model,
        "cumulative_calls": ledger.cumulative_calls,
        "cumulative_cost_rs": round(ledger.cumulative_cost_rs, 4),
        "history": ledger.history[-200:],  # keep last 200 entries
    }, indent=2))
    os.replace(tmp, p)  # atomic


def record_run(ledger: Ledger, calls: int, model: str, note: str = "") -> float:
    """Add a run's cost to the ledger and persist. Returns this run's cost in INR."""
    cost = estimate_cost(calls, model)
    ledger.model = model
    ledger.cumulative_calls += calls
    ledger.cumulative_cost_rs += cost
    # keep history compact: list of small dicts
    ledger.history.append(
        {"calls": calls, "cost": round(cost, 4), "total": round(ledger.cumulative_cost_rs, 2), "note": note}
    )
    save_ledger(ledger)
    return cost


class BudgetExceeded(Exception):
    """Raised when a run must be aborted to protect the spend cap."""


class CallBudget:
    """
    HARD mid-run call cap. Wraps a classifier and raises BudgetExceeded as soon as the
    cumulative per-run call count would exceed `max_calls`. This is what actually STOPS
    the pipeline from draining money on a pathological (fully-flagged) clip — a running
    counter on classify(), not an after-the-fact warning.
    """

    def __init__(self, classifier, max_calls: int | None, note: str = ""):
        self.inner = classifier
        self.max_calls = max_calls
        self.calls = 0
        self.note = note

    @property
    def model(self) -> str:
        return getattr(self.inner, "model", "gemini-2.5-flash")

    def classify(self, frames):
        if self.max_calls is None:
            return self.inner.classify(frames)
        # Count API REQUESTS, not frames: a StripClassifier sends ceil(N/strip) requests
        # for N frames, and Gemini bills per request. n_requests() on the inner classifier
        # reports the true request count.
        n = getattr(self.inner, "n_requests", None)
        need = n(frames) if callable(n) else len(frames)
        budget_left = self.max_calls - self.calls
        if need > budget_left:
            raise BudgetExceeded(
                f"[{self.note}] call cap hit: {budget_left} calls left, "
                f"this batch needs {need} (cap={self.max_calls}). "
                "Aborting to protect spend. Raise GAZEGUARD_MAX_CALLS_PER_RUN and re-run "
                "--force to continue."
            )
        self.calls += need
        return self.inner.classify(frames)


def check_budget(ledger: Ledger, projected_calls: int, model: str,
                 budget_rs: float | None, max_calls_per_run: int | None) -> None:
    """
    Refuse to run if either cap would be exceeded.
      budget_rs        - cumulative lifetime spend cap (INR)
      max_calls_per_run - per-run frame/call cap
    """
    projected_cost = estimate_cost(projected_calls, model)
    if budget_rs is not None and (ledger.cumulative_cost_rs + projected_cost) > budget_rs:
        raise BudgetExceeded(
            f"budget cap exceeded: projected {ledger.cumulative_cost_rs + projected_cost:.2f} INR "
            f"> budget {budget_rs:.2f} INR (ledger already {ledger.cumulative_cost_rs:.2f}). "
            "Raise GAZEGUARD_BUDGET or reset the ledger."
        )
    if max_calls_per_run is not None and projected_calls > max_calls_per_run:
        raise BudgetExceeded(
            f"projected {projected_calls} calls > per-run cap {max_calls_per_run}. "
            "Raise GAZEGUARD_MAX_CALLS_PER_RUN or tune coarse settings."
        )


def reset_ledger() -> Ledger:
    led = Ledger()
    save_ledger(led)
    return led