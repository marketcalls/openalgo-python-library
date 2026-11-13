"""Automatic indicator warm-up (decision D25): the one place its rule is written.

    An indicator is ready once its own series has closed its warm-up of bars, and a
    strategy decides only when all its indicators are ready.

- **The warm-up** of an indicator registered with `self.I` is the length its window-
  stability check measures (`_causality_checks`): the shortest rolling window, doubling
  from 16 bars, whose values agree with the values computed on all the bars at hand
  within the window tolerance. It is the window live trading keeps, so a backtest waits
  exactly as long as live trading does before its first decision (08-live-trading 11,
  "the same warm-up length in both").
- **What does not count**: an anchored indicator (session VWAP) restarts at each anchor,
  so it needs no warm-up; one that no window settles (a cumulative series, or a warm-up
  longer than the bars at hand) cannot be made ready by any history, and the strategy
  does not wait for it (a warning names it).
- **The history**: the run loads bars before `start` from its own source
  (`_warm_up_history`) and the indicators are computed over them. The strategy makes no
  decision before `start`: `on_bar`, timers, conditions and arrays all begin there (the
  history only refreshes indicators, as missed bars do live), and equity, trades,
  charges, metrics and the benchmark start at `start` as before.
- **Too little history** (own data that begins at `start`, a new listing, a futures
  series): the strategy waits inside the run until its indicators are ready, deciding
  nothing; a warning says how many strategy bars it waited and why.

A strategy that keeps its own guard (`if len(self.close) < self.slow: return`) still
works: the history counts in `len`, so it returns early less often.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._calendar_convert import format_ist
from ._causality_checks import CheckOutcome
from ._messages import counted

# The Assumptions line of the rule, ahead of what the run measured.
RULE = (
    "automatic: no decision before start, nor before every self.I indicator has closed "
    "its warm-up of bars (the window-stability length; bars before start count)"
)


@dataclass(frozen=True)
class SeriesNeed:
    """The bars one series must close before the strategy decides: its symbol (or
    futures underlying) and interval, the longest warm-up of its indicators, and the
    indicator that needs it."""

    symbol: str
    interval: str
    bars: int
    label: str

    def describe(self) -> str:
        """For example "SBIN D needs 512 bars (ema(SBIN.close, 50))"."""
        return f"{self.symbol} {self.interval} needs {counted(self.bars, 'bar')} ({self.label})"


def series_needs(outcomes: Iterable[CheckOutcome]) -> tuple[tuple[SeriesNeed, ...], list[str]]:
    """(the need of each series, in first-registration order; the labels of the
    indicators no window settles, which set no need). An anchored indicator sets none."""
    needs: dict[tuple[str, str], SeriesNeed] = {}
    unsettled: list[str] = []
    for outcome in outcomes:
        if outcome.anchored or outcome.series is None:
            continue
        if outcome.warm_up is None:
            unsettled.append(outcome.label)
            continue
        symbol, interval = outcome.series
        held = needs.get(outcome.series)
        if held is None or outcome.warm_up > held.bars:
            needs[outcome.series] = SeriesNeed(symbol, interval, outcome.warm_up, outcome.label)
    return tuple(needs.values()), unsettled


@dataclass(frozen=True)
class Readiness:
    """When a run's strategy may decide, and what it waited for.

    needs: the need of each series. unsettled: indicators no window settles. ready_at:
    UTC nanoseconds from which decisions are made, None when the strategy decides from
    `start` on. waited: strategy bars of the run that closed before then. short: the need
    that set the wait, None without one, and had: the bars its series had before
    `start`. history: what the run loaded before `start` (`History.describe`).

    A need is always met within the bars at hand: the window check finds only a warm-up
    shorter than the series it measures.
    """

    needs: tuple[SeriesNeed, ...]
    unsettled: tuple[str, ...]
    ready_at: int | None
    waited: int
    short: SeriesNeed | None
    had: int
    history: str

    def assumption(self) -> str:
        """The Assumptions line: the rule, each series' need, the history and when the
        strategy first decided."""
        parts = [RULE]
        if not self.needs:
            parts.append("no indicator needs a warm-up")
        parts += [need.describe() for need in self.needs]
        parts.append(f"history before start: {self.history}")
        if self.ready_at is not None:
            parts.append(f"first decision {format_ist(self.ready_at)}")
        elif self.needs:
            parts.append("ready at start")
        return "; ".join(parts)

    def warnings(self, start: Any) -> list[str]:
        """What the reader must know: a wait inside the run, and indicators the strategy
        did not wait for."""
        notes = []
        if self.short is not None and self.ready_at is not None:
            notes.append(
                f"The strategy waited {counted(self.waited, 'strategy bar')} inside the run "
                f"for its indicators, to {format_ist(self.ready_at)}: {self.short.describe()} "
                f"and {counted(self.had, 'bar')} closed before {start}. Load data from "
                f"earlier than {start} to trade from it."
            )
        for label in self.unsettled:
            notes.append(
                f"{label} does not settle within the bars at hand (it depends on its whole "
                "history, or needs a longer warm-up than they allow): the strategy did not "
                "wait for it."
            )
        return notes

    def manifest(self) -> dict[str, Any]:
        """The warm-up used, as the run manifest records it."""
        return {
            "needs": [
                {"symbol": n.symbol, "interval": n.interval, "bars": n.bars, "indicator": n.label}
                for n in self.needs
            ],
            "unsettled": list(self.unsettled),
            "first_decision": None if self.ready_at is None else format_ist(self.ready_at),
            "waited_bars": self.waited,
        }


def _latest_ready(
    needs: tuple[SeriesNeed, ...], store_of: Callable[[str, str], Any]
) -> tuple[int | None, SeriesNeed | None]:
    """(UTC nanoseconds at which the last series becomes ready, the need of that
    series), or (None, None) for no need. A series is ready at the close of its
    need-th bar, history included."""
    ready_at, short = None, None
    for need in needs:
        at = int(store_of(need.symbol, need.interval).column("ts_close")[need.bars - 1])
        if ready_at is None or at > ready_at:
            ready_at, short = at, need
    return ready_at, short


def readiness(
    outcomes: Iterable[CheckOutcome],
    store_of: Callable[[str, str], Any],
    decisions: Any,
    history: str,
) -> Readiness:
    """The readiness of a run (see the module docstring) from its indicators'
    `outcomes`, the store of a series by (symbol, interval), the store of the strategy
    bars its decisions follow (history first, `history_bars` of them) and the
    description of its history."""
    needs, unsettled = series_needs(outcomes)
    ready_at, short = _latest_ready(needs, store_of)
    run_closes = decisions.column("ts_close")[decisions.history_bars :]
    if ready_at is None or short is None or not len(run_closes) or ready_at <= run_closes[0]:
        return Readiness(needs, tuple(unsettled), None, 0, None, 0, history)
    first_open = decisions.column("ts_open")[decisions.history_bars]
    opens = store_of(short.symbol, short.interval).column("ts_open")
    had = int(np.searchsorted(opens, first_open, side="left"))
    waited = int(np.searchsorted(run_closes, ready_at, side="left"))
    return Readiness(needs, tuple(unsettled), ready_at, waited, short, had, history)


__all__ = ["RULE", "Readiness", "SeriesNeed", "readiness", "series_needs"]
