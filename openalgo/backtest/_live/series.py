"""The strategy layer's live stores (16-session-design 1.5, 2.5 A.10, B and C, 3.1 step
11; unit 6): bars that grow as slices close, indicators recomputed per close, broker
candles that replace tick-built bars, and missing-data marks.

- `LiveSeries` holds one instrument's base bars of a session: the history before today
  (the warm-up's broker candles, 08-live-trading section 5 rule 6) and today's closed
  base slices, each a bar built from ticks or a broker candle. A candle replaces the
  tick bar of its slice for later closes: the accepted candle feeds the strategy's
  series while the kernel keeps the tick bar it ran (D27; merge notes, bars review). A
  slice that closed while the feed was dead with no bar is marked missing until a bar
  of it arrives (2.5 B and C).
- `LiveStore` is the series at one interval as a strategy reads it: the history's bars,
  then today's bars closed by the decision instant, built from the base bars with the
  resampler a backtest builds its bars with (`resample.resample_bars`), so the store
  equals a backtest's store of the same bars bar by bar. Its views and `self.I`
  indicators (`_series_indicators`) recompute once per change of the store.
- `LiveRunContext` (`_series_context`, re-exported here) is the backtest's run context
  over live stores: the same strategy class reads, orders and registers indicators
  through it, a bar is visible once the kernel's decision instant passes its close, and
  `self.I` is registered in `init` only (the strategy's own rule).

Times are UTC nanoseconds, prices rupees, volume units. Not thread-safe: the session loop
owns these objects, as it owns the engines.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .._calendar_convert import DateLike
from .._layout import BAR_COLUMNS, NANOS_PER_SECOND, parse_interval
from .._series import FIELDS, TIMES, Bar, SeriesStore
from .._timeframes import _check_interval
from .._views import field_recipe
from ..calendar import NO_BAR, Calendar, default_calendar
from ..errors import CalendarError, ConfigError, LiveError
from ..resample import resample_bars
from ._bars_candle import CANDLE_SOURCE, TICKS_SOURCE, check_interval
from ._series_context import LiveRunContext
from ._series_indicators import LiveIndicator, LiveView
from ._types import LiveBar

#: The base interval of a live session's bars: one minute, the interval of the broker
#: candles they are checked against (08-live-trading section 5 rule 2; 1.1 Tier 1).
DEFAULT_BASE = "1m"

#: The decision instant a store's bars are visible up to (UTC nanoseconds), None before
#: the first decision.
Clock = Callable[[], int | None]
_COLUMNS = (*TIMES, *FIELDS)


def _empty_columns() -> dict[str, np.ndarray]:
    return {
        name: np.array([], dtype=np.int64 if name in TIMES else np.float64) for name in _COLUMNS
    }


def _session_slices(
    calendar: Calendar, exchange: str, day: DateLike, base_ns: int
) -> tuple[np.ndarray, np.ndarray]:
    """(opens, closes) of the exchange's base slices of `day`, UTC nanoseconds: the grid
    the bar builders and the kernel's live session use (`bars.TickBarBuilder`)."""
    if calendar.session(exchange, day) is None:
        raise CalendarError(
            f"{exchange} has no session on {day}: a live series is kept for a trading day "
            "only; check the date, or add the session to the calendar"
        )
    grid = calendar.union_grid([exchange], day, day, base=f"{base_ns // NANOS_PER_SECOND}s")
    closes = grid[f"close_{grid.attrs['exchanges'][0]}"].to_numpy()
    has_bar = closes != NO_BAR
    return grid["ts"].to_numpy()[has_bar], closes[has_bar]


def _history(frame: pd.DataFrame | None, symbol: str, today_ns: int) -> pd.DataFrame:
    """The history's base bars in the canonical layout. Raises ConfigError for a frame
    without the layout's columns, unsorted or repeated opens, or a bar after today's
    first slice opens."""
    if frame is None:
        frame = pd.DataFrame(_empty_columns())
    missing = [name for name in _COLUMNS if name not in frame]
    if missing:
        raise ConfigError(
            f"the history of {symbol} lacks the columns {missing}: pass the base bars before "
            f"today in the bars layout ({', '.join(BAR_COLUMNS)})"
        )
    opens = frame["ts_open"].to_numpy()
    if len(opens) and not (np.all(np.diff(opens) > 0) and int(frame["ts_close"].max()) <= today_ns):
        raise ConfigError(
            f"the history of {symbol} must hold base bars before today, sorted by ts_open "
            "with no repeats"
        )
    history = frame.reset_index(drop=True)
    return history if "oi" in history else history.assign(oi=np.nan)


class LiveSeries:
    """One instrument's base bars of a live session's IST `day` (module docstring).

    `history` is the base bars before today in the bars layout (`_layout`; the warm-up's
    `SymbolHistory.base`), None for none. `base` is the session's base interval, an
    intraday interval ("1m"); `calendar` is the package default when None.

    Raises ConfigError for a base that is not a whole number of seconds shorter than a
    day or an unusable history, and CalendarError for an unknown exchange or a day the
    exchange has no session on.
    """

    def __init__(
        self,
        symbol: str,
        exchange: str,
        day: DateLike,
        *,
        history: pd.DataFrame | None = None,
        base: str = DEFAULT_BASE,
        calendar: Calendar | None = None,
    ) -> None:
        kind, seconds = parse_interval(base)
        if kind != "intraday":
            raise ConfigError(
                f"a live session's base interval is intraday, such as '1m', not {base!r}"
            )
        base_ns = check_interval(seconds * NANOS_PER_SECOND)
        self.symbol, self.exchange, self.base = symbol, exchange, base
        self._calendar = default_calendar() if calendar is None else calendar
        self._opens, self._closes = _session_slices(self._calendar, exchange, day, base_ns)
        self._history = _history(history, symbol, int(self._opens[0]))
        self._bars: dict[int, LiveBar] = {}
        self._missing: set[int] = set()
        self._version = 0

    @property
    def version(self) -> int:
        """A number that changes whenever a base bar is taken (a store rebuilds then)."""
        return self._version

    @property
    def slices(self) -> tuple[tuple[int, int], ...]:
        """(open, close) of every base slice of the day in order, UTC nanoseconds."""
        return tuple(zip(self._opens.tolist(), self._closes.tolist(), strict=True))

    def take(self, bar: LiveBar) -> None:
        """Take today's closed base bar of one slice: a bar built from ticks, given once
        and before any candle of its slice, or a broker candle, which replaces whatever
        bar the slice had (a candle accepted by `CandleCheck`, or the candle of a slice
        that made no tick bar). Either lifts the slice's missing-data mark.

        Raises ConfigError for a bar of another instrument, of a source other than
        "ticks" or "candle", that is not one of today's base slices, or a tick bar for a
        slice that already has a bar.
        """
        if (bar.symbol, bar.exchange) != (self.symbol, self.exchange):
            raise ConfigError(
                f"a bar of {bar.exchange}:{bar.symbol} reached the series of "
                f"{self.exchange}:{self.symbol}"
            )
        if bar.source not in (TICKS_SOURCE, CANDLE_SOURCE):
            raise ConfigError(f"a live bar is built from ticks or is a candle, not {bar.source!r}")
        slot = self._slot(bar.ts_open_ns)
        if int(self._closes[slot]) != bar.ts_close_ns:
            raise ConfigError(
                f"the {self.symbol} bar opening at {bar.ts_open_ns} closes at "
                f"{bar.ts_close_ns}, not at its slice's close {int(self._closes[slot])}"
            )
        if bar.source == TICKS_SOURCE and slot in self._bars:
            raise ConfigError(
                f"the {self.symbol} slice opening at {bar.ts_open_ns} has a bar already: a "
                "slice's tick bar is given once, before any candle of it"
            )
        self._bars[slot] = bar
        self._missing.discard(slot)
        self._version += 1

    def bar(self, slice_open_ns: int) -> LiveBar | None:
        """The bar today's slice opening at `slice_open_ns` has now, None when it has
        none. Raises ConfigError for an instant that opens no slice of the day."""
        return self._bars.get(self._slot(slice_open_ns))

    def mark_missing(self, slice_open_ns: int) -> None:
        """Mark today's slice opening at `slice_open_ns` missing: it closed while the
        feed was dead and made no bar (2.5 B). A bar of it lifts the mark (`take`).

        Raises ConfigError for an instant that opens no slice of the day, or a slice
        that has a bar.
        """
        slot = self._slot(slice_open_ns)
        if slot in self._bars:
            raise ConfigError(
                f"the {self.symbol} slice opening at {slice_open_ns} has a bar: only a "
                "slice without one is marked missing"
            )
        self._missing.add(slot)

    def missing_slices(self, ts_open_ns: int, ts_close_ns: int) -> tuple[int, ...]:
        """The opens (UTC nanoseconds) of the slices marked missing that open in
        [ts_open_ns, ts_close_ns), oldest first: a strategy bar with any is not decided
        on (2.5 C; 08-live-trading section 5 rule 3)."""
        return tuple(
            int(self._opens[slot])
            for slot in sorted(self._missing)
            if ts_open_ns <= self._opens[slot] < ts_close_ns
        )

    def store(self, interval: str, clock: Clock, *, name: str | None = None) -> LiveStore:
        """The store of these bars at `interval` (a whole multiple of the base, or a
        calendar interval such as "D"), visible up to the decision instant `clock` gives.
        `name` prefixes its views' labels, the symbol when None (a strategy's own
        interval; another is named `SBIN[15m]`, as in a backtest, and the indicators'
        warm-ups are keyed on those labels). Raises ConfigError for an interval the base
        bars cannot give."""
        _check_interval(interval, self.base)
        return LiveStore(self, interval, clock, name)

    def history_columns(self, interval: str) -> dict[str, np.ndarray]:
        """The bars of `interval` the history makes, as store columns."""
        return self._columns(self._history, interval)

    def today_columns(self, interval: str) -> dict[str, np.ndarray]:
        """The bars of `interval` today's base bars make now, as store columns; the last
        may still be forming (a store shows the closed ones only)."""
        rows = [self._bars[slot] for slot in sorted(self._bars)]
        frame = pd.DataFrame(
            {
                "ts_open": np.array([bar.ts_open_ns for bar in rows], dtype=np.int64),
                "ts_close": np.array([bar.ts_close_ns for bar in rows], dtype=np.int64),
                **{
                    name: np.array([getattr(bar, name) for bar in rows], dtype=np.float64)
                    for name in FIELDS
                },
                "oi": np.full(len(rows), np.nan),
            }
        )
        return self._columns(frame, interval)

    def _columns(self, frame: pd.DataFrame, interval: str) -> dict[str, np.ndarray]:
        """The bars of `interval` the base bars `frame` make, built as a backtest builds
        them (`_warm_up_history`, `_timeframes`)."""
        if frame.empty:
            return _empty_columns()
        if interval != self.base:
            frame = resample_bars(
                frame,
                interval,
                exchange=self.exchange,
                calendar=self._calendar,
                source_interval=self.base,
                on_closed_day="drop",
            )
        return {name: frame[name].to_numpy() for name in _COLUMNS}

    def _slot(self, slice_open_ns: int) -> int:
        slot = int(np.searchsorted(self._opens, slice_open_ns))
        if slot == len(self._opens) or int(self._opens[slot]) != slice_open_ns:
            raise ConfigError(
                f"{slice_open_ns} opens no {self.base} slice of the {self.exchange} session "
                f"the {self.symbol} series is kept for"
            )
        return slot


class LiveStore(SeriesStore):
    """One instrument's bars at one interval in a live session (module docstring): the
    history's bars (`history_bars` of them), then today's bars closed by the decision
    instant `clock` gives. None from `clock` (no decision yet, `init` running) shows no
    bar, and every bar the store holds is visible once a decision has come, as in a
    backtest, where `init` sees none and the history is visible from the first pause.

    `version` changes whenever the bars change: a decision revealed a bar, or a candle
    replaced a base bar. Its views (`view`) and indicators are live views that recompute
    once per change (`_series_indicators`). Made by `LiveSeries.store`.
    """

    def __init__(
        self, series: LiveSeries, interval: str, clock: Clock, name: str | None = None
    ) -> None:
        super().__init__(series.symbol, interval, series.history_columns(interval), name)
        self.history_bars = len(self._series["ts_open"])
        self._history = dict(self._series)
        self._source = series
        self._clock = clock
        # The series' version the today part was built at, today's bars, how many of them
        # the store holds, the store's own version and the recipe input of that version.
        self._built_at = -1
        self._today = _empty_columns()
        self._held = 0
        self._version = 0
        self._bars_of: tuple[int, dict[str, Any]] | None = None

    @property
    def version(self) -> int:
        """A number that changes whenever the bars the store holds change."""
        self._sync()
        return self._version

    def _sync(self) -> None:
        """Bring the bars up to the series and the clock: today's bars rebuilt when the
        series took a bar, then cut at the decision instant."""
        source = self._source
        rebuilt = source.version != self._built_at
        if rebuilt:
            self._today = source.today_columns(self.interval)
            self._built_at = source.version
        now = self._clock()
        held = 0 if now is None else int(np.searchsorted(self._today["ts_close"], now, "right"))
        if rebuilt or held != self._held:
            self._series = {
                name: np.concatenate((self._history[name], self._today[name][:held]))
                for name in _COLUMNS
            }
            self._times = None
            self._held = held
            self._version += 1

    def visible(self) -> int:
        self._sync()
        return 0 if self._clock() is None else len(self._series["ts_open"])

    def begin(self) -> None:
        """Nothing to do: the history is visible from the first decision (`visible`).
        The run context calls it when `init` has returned."""

    def column(self, name: str) -> np.ndarray:
        self._sync()
        return super().column(name)

    def times(self) -> pd.DatetimeIndex:
        self._sync()
        return super().times()

    def bars(self) -> dict[str, Any]:
        """The bar fields of the bars the store holds, as a recipe reads them
        (`_views.Recipe`): the columns and "times", the IST opens."""
        version = self.version
        if self._bars_of is None or self._bars_of[0] != version:
            self._bars_of = (version, {**self._series, "times": self.times()})
        return self._bars_of[1]

    def view(self, name: str) -> LiveView:
        """The live view of one bar field."""
        return LiveView(
            self,
            lambda: self.column(name),
            f"{self.name}.{name}",
            field_recipe(name),
            bar_field=True,
        )

    def reveal(self, row: tuple) -> Bar:
        """The bar of the kernel's strategy-bar row (index, ts_open, ...) as the store
        holds it, after the run context has moved the clock to the row's decision: the
        series' bar, built from the accepted candles where they came (module docstring).

        Raises LiveError when the store holds no bar opening at the row's open: the
        kernel was given a bar the series was not.
        """
        opens = self.column("ts_open")
        index = int(np.searchsorted(opens, row[1]))
        if index == len(opens) or int(opens[index]) != row[1]:
            raise LiveError(
                f"{self.symbol}: the kernel closed a strategy bar opening at {row[1]} that "
                "the live series does not hold: every bar the kernel is given goes to the "
                "series too (LiveSeries.take)"
            )
        return Bar(self.symbol, *(self._series[name].item(index) for name in _COLUMNS))

    def indicator_adapter(self, call: Any, names: list[str], causal: tuple[str, ...]) -> Any:
        """What `_indicators.register` needs for an indicator on this store."""
        return LiveIndicator(self, call, names, causal)


__all__ = ["DEFAULT_BASE", "Clock", "LiveRunContext", "LiveSeries", "LiveStore"]
