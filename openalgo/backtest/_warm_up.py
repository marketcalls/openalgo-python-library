"""Automatic indicator warm-up (decision D25): the one place its rule is written.

    An indicator is ready once its own series has closed its warm-up of bars, and a
    strategy decides for a symbol only when all the indicators of that symbol are ready,
    counting only the bars up to the decision.

- **The warm-up** of an indicator registered with `self.I` is its window-stability length
  measured on the engine's reference bars (`_warm_up_measure`): a function of the
  indicator call alone (the function, its parameters and the bar fields it reads),
  counted in the bars of the indicator's own series. It never depends on the run's
  prices, its `end` or its symbol, so no decision depends on a later bar, and live
  trading keeps the same window (08-live-trading 11). An indicator of other views needs
  their warm-up too. One that needs more than the longest window measured waits that
  long at least, and a warning says it may need more.
- **An anchored indicator** (session or monthly VWAP, floor pivots) restarts at each
  anchor period: it needs the bars since the start of the first period it reads (its
  own, or the whole one before for pivots), and is ready from there (`_warm_up_anchor`).
- **What does not count**: an indicator no window settles (a running sum) cannot be made
  ready by any history, and one that reads values of the user's own (an array) or
  cannot be computed on the reference bars cannot be measured: the strategy does not
  wait for them, and a warning names each.
- **The history**: the run loads the bars before `start` that its indicators need from
  its own source (`_warm_up_history`) and its series hold them before `init` runs, so
  every indicator is computed over them. The strategy makes no decision before `start`:
  `on_bar`, timers, conditions and arrays all begin there, and equity, trades, charges,
  metrics and the benchmark start at `start` as before.
- **Readiness is per symbol and causal** (`Readiness.ready_from`): a symbol (or futures
  underlying) is ready from the close of the bar at which the last of its needs is met,
  the need-th bar of its series, history counted, or for an anchored need the bar from
  which the periods it reads are whole. That bar is found from the bars up to it alone,
  so whether a symbol is ready at a bar never depends on a later bar or on how many bars
  follow. At a decision a symbol not ready yet is left out: its bar is not passed to
  `on_bar`, its orders are skipped with skip reason "not_ready" (`_context`), and its
  indicators read NaN, or false for a condition, wherever they are read (`_views`), so
  a decision for another symbol cannot rest on a value that the bars after it decide
  (an EMA of 20 bars is not computed on fewer than 20, and is from the first bar on
  more); its bar fields stay readable. It joins at the first bar it is ready, and the
  symbols that are ready trade. Nothing is decided before the first symbol is ready (no
  timer, no `on_bar`, no condition), so a run of one symbol simply waits. A series too
  short to compute an indicator at all (`_indicators`) has fewer bars than the warm-up of
  that indicator, so it is not ready at any of its bars, as a series long enough to
  compute it would not be by then.
- **Too little history** (own data that begins at `start`, a new listing, a futures
  series): the symbol waits inside the run until its indicators are ready, and one whose
  needs are not met by its last bar makes no decision at all; a warning counts the
  strategy bars each symbol waited and says why, and `bt.backtest` raises a
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
from ._warm_up_anchor import PeriodAnchor, anchored_ready_at

# The Assumptions line of the rule, ahead of what the run measured.
RULE = (
    "automatic: no decision before start, nor for a symbol before every self.I indicator "
    "of it has closed its warm-up of bars (its window-stability length on the engine's "
    "reference bars, the same for any data and in live trading; bars before start count)"
)
# A decision instant later than any bar's close: the strategy never decides.
NEVER = int(np.iinfo(np.int64).max)
# The manifest's first decision of a run or a symbol whose indicators were never ready.
NEVER_DECIDED = "never"
# How far back history reaches at most: thirty years before `start`. Longer than the
# indicators in common use need on daily bars (a 200-bar EMA needs 2,048 sessions, about
# eight years) and a bound on the load of an intraday base; an indicator that needs more
# than the longest window measured is given all of it.
HISTORY_YEARS_CAP = 30


@dataclass(frozen=True)
class SeriesNeed:
    """The bars one series must close before the strategy decides: its symbol (or
    futures underlying) and interval, the longest warm-up of its indicators in bars of
    that interval, and the indicator that needs it. `anchor` is the anchor of an
    anchored indicator, whose need is the bars since the first period it reads began
    (`bars` 0); `more` is True when the indicator needs more than `bars`, the longest
    window measured."""

    symbol: str
    interval: str
    bars: int
    label: str
    anchor: PeriodAnchor | None = None
    more: bool = False

    def key(self) -> tuple:
        """What needs of one kind on one series share: the longest of them is kept."""
        return (self.symbol, self.interval, self.anchor)

    def describe(self) -> str:
        """For example "ema(SBIN.close, 50) needs 512 D bars of SBIN"."""
        if self.anchor is not None:
            return (
                f"{self.label} needs the {self.interval} bars of {self.symbol} since "
                f"{self.anchor.since()}"
            )
        bars = counted(self.bars, f"{self.interval} bar")
        return f"{self.label} needs {'more than ' if self.more else ''}{bars} of {self.symbol}"

    def manifest(self) -> dict[str, Any]:
        """The need as the run manifest records it."""
        record: dict[str, Any] = {"symbol": self.symbol, "interval": self.interval}
        record["bars"] = None if self.anchor is not None else self.bars
        record["indicator"] = self.label
        if self.anchor is not None:
            record["since"] = self.anchor.period
            if self.anchor.periods_before:
                record["periods_before"] = self.anchor.periods_before
        if self.more:
            record["more_than"] = True
        return record


def _need_of(outcome: CheckOutcome) -> SeriesNeed | None:
    """The need an indicator's outcome sets, None when it has no warm-up length. A
    series too short to compute it keeps its warm-up as its need (module docstring)."""
    if outcome.series is None:
        return None
    symbol, interval = outcome.series
    if outcome.anchor is not None:
        return SeriesNeed(symbol, interval, 0, outcome.label, outcome.anchor)
    if outcome.warm_up is not None:
        return SeriesNeed(symbol, interval, outcome.warm_up, outcome.label, more=outcome.more)
    return None


def series_needs(
    outcomes: Iterable[CheckOutcome],
) -> tuple[tuple[SeriesNeed, ...], tuple[CheckOutcome, ...]]:
    """(the needs of each series in first-registration order: its longest warm-up and
    one for each anchor its anchored indicators restart on; the outcomes of the
    indicators with no warm-up length, which set no need)."""
    needs: dict[tuple, SeriesNeed] = {}
    unsettled: list[CheckOutcome] = []
    for outcome in outcomes:
        found = _need_of(outcome)
        if found is None:
            if outcome.series is not None:
                unsettled.append(outcome)
            continue
        held = needs.get(found.key())
        if held is None or (found.bars, found.more) > (held.bars, held.more):
            needs[found.key()] = found
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


def _when(instant: int | None) -> str | None:
    """A symbol's first decision as the manifest records it: None at start, "never", or
    the IST time."""
    if instant is None:
        return None
    return NEVER_DECIDED if instant == NEVER else format_ist(instant)


def _whose(names: list[str]) -> str:
    return "its" if len(names) == 1 else "their"


@dataclass(frozen=True)
class Readiness:
    """When a run's strategy decides for each name it addresses, and what it waited for.

    needs: the needs of each series. unsettled: indicators with no warm-up length.
    ready_from: (name, UTC nanoseconds of the close from which the strategy decides for
    it) for every symbol or futures underlying, in order: None when it is ready at the
    run's first decision instant, NEVER when it is never ready within the run. waited:
    (name, strategy bars of the run that closed before it was ready) for each name not
    ready at start, in order. short: the needs not met by `start`. history: what the run
    kept before `start` (`_warm_up_history`).
    """

    needs: tuple[SeriesNeed, ...]
    unsettled: tuple[CheckOutcome, ...]
    ready_from: tuple[tuple[str, int | None], ...]
    waited: tuple[tuple[str, int], ...]
    short: tuple[Shortfall, ...]
    history: str

    @property
    def first_decision(self) -> int | None:
        """The instant of the run's first decision (UTC nanoseconds): None when a name
        is ready at start, NEVER when none is ever ready, else the earliest name's."""
        instants = [instant for _, instant in self.ready_from]
        if not instants or None in instants:
            return None
        return min(instant for instant in instants if instant is not None)

    @property
    def never(self) -> tuple[str, ...]:
        """The names never ready within the run, in order."""
        return tuple(name for name, instant in self.ready_from if instant == NEVER)

    def waiting(self) -> dict[str, int]:
        """The names not ready at start, with the instant each is ready from (NEVER for
        never): what the run context leaves out until then (`_context`)."""
        return {name: instant for name, instant in self.ready_from if instant is not None}

    def _all_never(self) -> bool:
        return bool(self.ready_from) and len(self.never) == len(self.ready_from)

    def _first_decisions(self) -> str | None:
        """The Assumptions part that says when the strategy first decided."""
        instants = [instant for _, instant in self.ready_from]
        if self._all_never():
            return "no decision: the indicators were never ready"
        if all(instant is None for instant in instants):
            return "ready at start" if self.needs else None
        if len(self.ready_from) == 1:
            return f"first decision {format_ist(instants[0])}"
        told = (f"{name} {_when(when) or 'at start'}" for name, when in self.ready_from)
        return f"first decisions: {', '.join(told)}"

    def assumption(self) -> str:
        """The Assumptions line: the rule, each series' need, the history and when the
        strategy first decided for each symbol."""
        parts = [RULE]
        if not self.needs:
            parts.append("no indicator needs a warm-up")
        parts += [need.describe() for need in self.needs]
        parts.append(f"history before start: {self.history}")
        decided = self._first_decisions()
        return "; ".join(parts if decided is None else [*parts, decided])

    def _waits(self) -> str:
        counts = [(name, bars) for name, bars in self.waited if bars]
        if not counts:
            return "no strategy bar"
        (name, bars), rest = counts[0], counts[1:]
        told = [f"{name} waited {counted(bars, 'strategy bar')}"]
        return ", ".join([*told, *(f"{other} {count}" for other, count in rest)])

    def _reasons(self, start: Any, names: Iterable[str]) -> str:
        """The shortfalls of the needs of `names`."""
        chosen = set(names)
        found = [short for short in self.short if short.need.symbol in chosen]
        return " ".join(f"{short.describe(start)}." for short in found)

    def _one_name_notes(self, start: Any) -> list[str]:
        """The wait of a run of one name, or of names none of which was ever ready."""
        reasons = self._reasons(start, (name for name, _ in self.ready_from))
        if self._all_never():
            return [
                "The strategy made no decision: its indicators were never ready within the "
                f"run's bars ({self._waits()}). {reasons}"
            ]
        ((_, instant),) = self.ready_from
        return [
            f"The strategy waited inside the run for its indicators: {self._waits()}, to "
            f"{format_ist(instant)}. {reasons}"
        ]

    def _names_notes(self, start: Any) -> list[str]:
        """The waits of a run of several names, some of which decided."""
        notes = []
        never = list(self.never)
        if never:
            whose = _whose(never)
            notes.append(
                f"The strategy made no decision for {', '.join(never)}: {whose} indicators "
                f"were never ready within {whose} bars, so {whose} orders were skipped while "
                f"the other symbols traded. {self._reasons(start, never)}"
            )
        ready_from = dict(self.ready_from)
        waiting = [(name, bars) for name, bars in self.waited if ready_from[name] != NEVER]
        if waiting:
            names = [name for name, _ in waiting]
            whose = _whose(names)
            each = "; ".join(
                f"{name} waited {counted(bars, 'strategy bar')}, to {format_ist(ready_from[name])}"
                for name, bars in waiting
            )
            notes.append(
                f"The strategy waited inside the run for the indicators of {', '.join(names)}: "
                f"{each}. Until then {whose} bars were not passed to on_bar and {whose} orders "
                f"were skipped (skip reason 'not_ready'). {self._reasons(start, names)}"
            )
        return notes

    def warnings(self, start: Any) -> list[str]:
        """What the reader must know: a wait inside the run, the names never ready, a
        need longer than the longest window measured, and indicators the strategy did not
        wait for."""
        notes: list[str] = []
        if self.waited:
            several = len(self.ready_from) > 1 and not self._all_never()
            notes = self._names_notes(start) if several else self._one_name_notes(start)
        for need in self.needs:
            if need.more:
                notes.append(
                    f"{need.describe()}: no window of the engine's reference bars settles it "
                    "and its difference shrinks as the window grows, as a very long "
                    f"average's does; the run loaded up to {HISTORY_YEARS_CAP} years of history "
                    f"before start and waited {counted(need.bars, 'bar')} at least, so its "
                    "first values may still differ from those over a longer history."
                )
        for outcome in self.unsettled:
            notes.append(f"{outcome.unsettled()}: the strategy did not wait for it.")
        return [note.rstrip() for note in notes]

    def manifest(self) -> dict[str, Any]:
        """The warm-up used, as the run manifest records it."""
        record = {
            "needs": [need.manifest() for need in self.needs],
            "not_waited_for": [outcome.label for outcome in self.unsettled],
            "first_decision": _when(self.first_decision),
            "waited": dict(self.waited),
            "first_decisions": {name: _when(instant) for name, instant in self.ready_from},
        }
        if self.never and not self._all_never():
            record["no_decision"] = list(self.never)
        return record


def _run_part(store: Any) -> tuple[np.ndarray, np.ndarray]:
    """(opens, closes) of a store's bars from `start` on (UTC nanoseconds)."""
    first = store.history_bars
    return store.column("ts_open")[first:], store.column("ts_close")[first:]


def _ready_at(need: SeriesNeed, context: Any) -> int:
    """The close of the bar from which the need is met, history included: the need-th
    bar of its series, or for an anchored need the bar from which the periods it reads
    are whole (`_warm_up_anchor`); NEVER when the series has no such bar. Either is
    found from the bars up to it alone (module docstring)."""
    store = context.store(need.symbol, need.interval)
    if need.anchor is not None:
        run = context.run
        ready = anchored_ready_at(store, need.anchor, run.calendar, run.exchange)
        return NEVER if ready is None else ready
    closes = store.column("ts_close")
    return int(closes[need.bars - 1]) if len(closes) >= need.bars else NEVER


def _shortfall(need: SeriesNeed, context: Any, history: Any, run_open: int) -> Shortfall:
    store = context.store(need.symbol, need.interval)
    had = int(np.searchsorted(store.column("ts_close"), run_open, side="right"))
    opens = store.column("ts_open")
    first_day: dt.date | None = ist_time(int(opens[0])).date() if len(opens) else None
    return Shortfall(need, had, history.shortfall(need, first_day))


def readiness(outcomes: Iterable[CheckOutcome], context: Any, history: Any) -> Readiness:
    """The readiness of a run (see the module docstring) from its indicators'
    `outcomes`, its `RunContext` (the stores by symbol and interval, the names it
    trades, the calendar and exchange) and its history before `start`
    (`_warm_up_history.RunHistory`: `describe()` and `shortfall(need, first bar day)`).
    A name is ready at start when its needs are met by the run's first decision
    instant, and never ready when they are met after the run's last strategy bar."""
    needs, unsettled = series_needs(outcomes)
    at = {need: _ready_at(need, context) for need in needs}
    names = context.names
    runs = {name: _run_part(context.store(name)) for name in names}
    described = history.describe()
    closes_of = [closes for _, closes in runs.values() if len(closes)]
    if not closes_of:
        ready = tuple((name, None) for name in names)
        return Readiness(needs, unsettled, ready, (), (), described)
    run_first = min(int(closes[0]) for closes in closes_of)
    run_last = max(int(closes[-1]) for closes in closes_of)
    run_open = min(int(opens[0]) for opens, _ in runs.values() if len(opens))
    short = tuple(
        _shortfall(need, context, history, run_open) for need in needs if at[need] > run_first
    )
    ready_from: list[tuple[str, int | None]] = []
    waited = []
    for name in names:
        latest = max((at[need] for need in needs if need.symbol == name), default=None)
        if latest is None or latest <= run_first:
            ready_from.append((name, None))
            continue
        ready_from.append((name, NEVER if latest > run_last else latest))
        waited.append((name, int(np.searchsorted(runs[name][1], latest, side="left"))))
    return Readiness(needs, unsettled, tuple(ready_from), tuple(waited), short, described)


__all__ = [
    "HISTORY_YEARS_CAP",
    "NEVER",
    "NEVER_DECIDED",
    "RULE",
    "Readiness",
    "SeriesNeed",
    "Shortfall",
    "readiness",
    "series_needs",
]
