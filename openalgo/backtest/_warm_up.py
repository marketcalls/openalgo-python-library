"""Automatic indicator warm-up (decision D25): the one place its rule is written.

    An indicator is ready once its own series has closed its warm-up of bars, and a
    strategy decides only when all its indicators are ready.

- **The warm-up** of an indicator registered with `self.I` is its window-stability length
  measured on the engine's reference bars (`_warm_up_reference`): a function of the
  indicator call alone (the function, its parameters and the bar fields it reads),
  counted in the bars of the indicator's own series. It never depends on the run's
  prices, its `end` or its symbol, so no decision depends on a later bar, and live
  trading keeps the same window (08-live-trading 11).
- **What does not count**: an anchored indicator (session VWAP) restarts at each anchor,
  so it needs no warm-up. One that no window settles on the reference bars (a
  cumulative series) cannot be made ready by any history, and one that reads values of
  the user's own (an array) cannot be measured: the strategy does not wait for either,
  and a warning names it.
- **The history**: the run loads the bars before `start` that its indicators need from
  its own source (`_warm_up_history`) and the indicators are computed over them. The
  strategy makes no decision before `start`: `on_bar`, timers, conditions and arrays all
  begin there (the history only refreshes indicators, as missed bars do live), and
  equity, trades, charges, metrics and the benchmark start at `start` as before.
- **Too little history** (own data that begins at `start`, a new listing, a futures
  series): the strategy waits inside the run until its indicators are ready, deciding
  nothing, and never decides when the run has fewer bars than a need; a warning counts
  the strategy bars each symbol waited and says why, and `bt.backtest` raises a
  `WarmUpWarning` (`_warm_up_notice`).

A strategy that keeps its own guard (`if len(self.close) < self.slow: return`) still
works: the history counts in `len`, so it returns early less often.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._calendar_convert import format_ist
from ._causality_checks import CheckOutcome
from ._messages import counted
from ._series import ist_time

# The Assumptions line of the rule, ahead of what the run measured.
RULE = (
    "automatic: no decision before start, nor before every self.I indicator has closed "
    "its warm-up of bars (its window-stability length on the engine's reference bars, the "
    "same for any data and in live trading; bars before start count)"
)
# A decision instant later than any bar's close: the strategy never decides.
NEVER = int(np.iinfo(np.int64).max)
# The manifest's first decision of a run whose indicators were never ready.
NEVER_DECIDED = "never"


@dataclass(frozen=True)
class SeriesNeed:
    """The bars one series must close before the strategy decides: its symbol (or
    futures underlying) and interval, the longest warm-up of its indicators in bars of
    that interval, and the indicator that needs it."""

    symbol: str
    interval: str
    bars: int
    label: str

    def describe(self) -> str:
        """For example "ema(SBIN.close, 50) needs 512 D bars of SBIN"."""
        return f"{self.label} needs {counted(self.bars, f'{self.interval} bar')} of {self.symbol}"


def series_needs(
    outcomes: Iterable[CheckOutcome],
) -> tuple[tuple[SeriesNeed, ...], tuple[CheckOutcome, ...]]:
    """(the need of each series, in first-registration order; the outcomes of the
    indicators with no warm-up length, which set no need). An anchored indicator sets
    none and is not listed."""
    needs: dict[tuple[str, str], SeriesNeed] = {}
    unsettled: list[CheckOutcome] = []
    for outcome in outcomes:
        if outcome.anchored or outcome.series is None:
            continue
        if outcome.warm_up is None:
            unsettled.append(outcome)
            continue
        symbol, interval = outcome.series
        held = needs.get(outcome.series)
        if held is None or outcome.warm_up > held.bars:
            needs[outcome.series] = SeriesNeed(symbol, interval, outcome.warm_up, outcome.label)
    return tuple(needs.values()), tuple(unsettled)


@dataclass(frozen=True)
class Shortfall:
    """A need the bars before `start` did not meet: `had` bars of its series closed
    before the run's first bar, and `why` says what the history lacked."""

    need: SeriesNeed
    had: int
    why: str

    def describe(self, start: Any) -> str:
        """For example "ema(SBIN.close, 50) needs 512 D bars of SBIN and 0 bars closed
        before 2023-01-02: <why>"."""
        had = counted(self.had, "bar")
        return f"{self.need.describe()} and {had} closed before {start}: {self.why}"


@dataclass(frozen=True)
class Readiness:
    """When a run's strategy may decide, and what it waited for.

    needs: the need of each series. unsettled: indicators with no warm-up length.
    ready_at: UTC nanoseconds from which decisions are made, None when the strategy
    decides from `start` on, NEVER when a series has fewer bars than its need. waited:
    (name, strategy bars of the run that closed before then) for each symbol or futures
    underlying, in order. short: the needs not met by `start`. history: what the run
    kept before `start` (`_warm_up_history`).
    """

    needs: tuple[SeriesNeed, ...]
    unsettled: tuple[CheckOutcome, ...]
    ready_at: int | None
    waited: tuple[tuple[str, int], ...]
    short: tuple[Shortfall, ...]
    history: str

    def assumption(self) -> str:
        """The Assumptions line: the rule, each series' need, the history and when the
        strategy first decided."""
        parts = [RULE]
        if not self.needs:
            parts.append("no indicator needs a warm-up")
        parts += [need.describe() for need in self.needs]
        parts.append(f"history before start: {self.history}")
        if self.ready_at == NEVER:
            parts.append("no decision: the indicators were never ready")
        elif self.ready_at is not None:
            parts.append(f"first decision {format_ist(self.ready_at)}")
        elif self.needs:
            parts.append("ready at start")
        return "; ".join(parts)

    def _waits(self) -> str:
        counts = [(name, bars) for name, bars in self.waited if bars]
        if not counts:
            return "no strategy bar"
        (name, bars), rest = counts[0], counts[1:]
        told = [f"{name} waited {counted(bars, 'strategy bar')}"]
        return ", ".join([*told, *(f"{other} {count}" for other, count in rest)])

    def warnings(self, start: Any) -> list[str]:
        """What the reader must know: a wait inside the run, and indicators the strategy
        did not wait for."""
        notes = []
        reasons = " ".join(f"{short.describe(start)}." for short in self.short)
        if self.ready_at == NEVER:
            notes.append(
                "The strategy made no decision: its indicators were never ready within the "
                f"run's bars ({self._waits()}). {reasons}"
            )
        elif self.ready_at is not None:
            notes.append(
                f"The strategy waited inside the run for its indicators: {self._waits()}, to "
                f"{format_ist(self.ready_at)}. {reasons}"
            )
        for outcome in self.unsettled:
            notes.append(f"{outcome.label} {outcome.note}: the strategy did not wait for it.")
        return [note.rstrip() for note in notes]

    def manifest(self) -> dict[str, Any]:
        """The warm-up used, as the run manifest records it."""
        first = None if self.ready_at is None else format_ist(self.ready_at)
        return {
            "needs": [
                {"symbol": n.symbol, "interval": n.interval, "bars": n.bars, "indicator": n.label}
                for n in self.needs
            ],
            "not_waited_for": [outcome.label for outcome in self.unsettled],
            "first_decision": NEVER_DECIDED if self.ready_at == NEVER else first,
            "waited": dict(self.waited),
        }


def _run_part(store: Any) -> tuple[np.ndarray, np.ndarray]:
    """(opens, closes) of a store's bars from `start` on (UTC nanoseconds)."""
    first = store.history_bars
    return store.column("ts_open")[first:], store.column("ts_close")[first:]


def _ready_at(need: SeriesNeed, context: Any) -> int:
    """The close of the need-th bar of its series, history included; NEVER when the
    series has fewer bars."""
    closes = context.store(need.symbol, need.interval).column("ts_close")
    return int(closes[need.bars - 1]) if len(closes) >= need.bars else NEVER


def _shortfall(need: SeriesNeed, context: Any, history: Any, run_open: int) -> Shortfall:
    store = context.store(need.symbol, need.interval)
    had = int(np.searchsorted(store.column("ts_close"), run_open, side="right"))
    opens = store.column("ts_open")
    first_day: dt.date | None = ist_time(int(opens[0])).date() if len(opens) else None
    return Shortfall(need, had, history.shortfall(need.symbol, first_day))


def readiness(outcomes: Iterable[CheckOutcome], context: Any, history: Any) -> Readiness:
    """The readiness of a run (see the module docstring) from its indicators'
    `outcomes`, its `RunContext` (the stores by symbol and interval, the names it
    trades) and its history before `start` (`_warm_up_history.RunHistory`: `describe()`
    and `shortfall(symbol, first bar day)`)."""
    needs, unsettled = series_needs(outcomes)
    runs = {name: _run_part(context.store(name)) for name in context.names}
    first_closes = [int(closes[0]) for _, closes in runs.values() if len(closes)]
    first_opens = [int(opens[0]) for opens, _ in runs.values() if len(opens)]
    at = {need: _ready_at(need, context) for need in needs}
    latest = max(at.values(), default=None)
    if latest is None or not first_closes or latest <= min(first_closes):
        return Readiness(needs, unsettled, None, (), (), history.describe())
    decided_from, run_open = min(first_closes), min(first_opens)
    short = tuple(
        _shortfall(need, context, history, run_open) for need in needs if at[need] > decided_from
    )
    waited = tuple(
        (name, int(np.searchsorted(closes, latest, side="left")))
        for name, (_, closes) in runs.items()
    )
    return Readiness(needs, unsettled, latest, waited, short, history.describe())


__all__ = [
    "NEVER",
    "NEVER_DECIDED",
    "RULE",
    "Readiness",
    "SeriesNeed",
    "Shortfall",
    "readiness",
    "series_needs",
]
