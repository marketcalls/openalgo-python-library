"""The strategy layer's live stores (16-session-design 1.5, 2.5 A.10, B and C, 3.1 step
11; unit 6): bars that grow as slices close, indicators recomputed per close, broker
candles that replace tick-built bars, and missing-data marks.

- `LiveSeries` holds one instrument's base bars of a session: the history before today
  (the warm-up's broker candles, 08-live-trading section 5 rule 6) and today's closed
  base slices, each a bar built from ticks or a broker candle. A candle replaces the
  tick bar of its slice for later closes: the accepted candle feeds the strategy's
  series while the kernel keeps the tick bar it ran (D27; merge notes, bars review). The
  slice's tick bar stays at hand (`tick_bar`), so the loop checks every candle of a
  slice against it (`CandleCheck.accepted`) and takes what the check gives; it skips a
  candle equal to the bar the slice has (a candle read again), so the drift log keeps
  each candle once. A slice that closed while the feed was dead with no bar is marked
  missing until a bar of it arrives (2.5 B and C).
- `LiveStore` (`_series_store`, re-exported here) is the series at one interval as a
  strategy reads it: the bars closed by the decision instant, built from the base bars
  with the resampler a backtest builds its bars with (`resample.resample_bars`), so the
  store equals a backtest's store of the same bars bar by bar. The history's bars are
  built once; the bar of a period that today continues (a week, month, quarter or year
  still forming) is built with today's from all its base bars, the history's and
  today's, as a backtest builds it, and like every bar it is shown only once it has
  closed. Its views and `self.I` indicators (`_series_indicators`) recompute once per
  change of the store.
- `LiveRunContext` (`_series_context`, re-exported here) is the backtest's run context
  over live stores: the same strategy class reads, orders and registers indicators
  through it, a bar is visible once the kernel's decision instant passes its close, and
  `self.I` is registered in `init` only (the strategy's own rule).

Times are UTC nanoseconds, prices rupees, volume units. Not thread-safe: the session loop
owns these objects, as it owns the engines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._calendar_convert import DateLike
from .._layout import BAR_COLUMNS, NANOS_PER_SECOND, parse_interval
from .._series import FIELDS, TIMES
from .._timeframes import _check_interval
from ..calendar import NO_BAR, Calendar, default_calendar
from ..errors import CalendarError, ConfigError
from ..resample import resample_bars
from ._bars_candle import CANDLE_SOURCE, TICKS_SOURCE, check_interval
from ._series_context import LiveRunContext
from ._series_store import COLUMNS, Clock, LiveStore
from ._types import LiveBar

#: The base interval of a live session's bars: one minute, the interval of the broker
#: candles they are checked against (08-live-trading section 5 rule 2; 1.1 Tier 1).
DEFAULT_BASE = "1m"


def _empty_columns() -> dict[str, np.ndarray]:
    return {name: np.array([], dtype=np.int64 if name in TIMES else np.float64) for name in COLUMNS}


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
    missing = [name for name in COLUMNS if name not in frame]
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
        # Per interval: the history's bars closed by today's first open, and the history's
        # base bars of the period today continues (`_split`).
        self._splits: dict[str, tuple[dict[str, np.ndarray], pd.DataFrame]] = {}
        self._bars: dict[int, LiveBar] = {}
        self._tick_bars: dict[int, LiveBar] = {}
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
        that made no tick bar). Either lifts the slice's missing-data mark. The slice's
        own tick bar again (what `CandleCheck.accepted` gives back for a candle that does
        not cover the slice) and a candle equal to the slice's bar change nothing.

        Raises ConfigError for a bar of another instrument, of a source other than
        "ticks" or "candle", that is not one of today's base slices, or a tick bar for a
        slice that already has another bar.
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
        if bar.source == TICKS_SOURCE:
            if self._tick_bars.get(slot) == bar:
                return
            if slot in self._bars:
                raise ConfigError(
                    f"the {self.symbol} slice opening at {bar.ts_open_ns} has a bar already: "
                    "a slice's tick bar is given once, before any candle of it"
                )
            self._tick_bars[slot] = bar
        elif self._bars.get(slot) == bar:
            return
        self._bars[slot] = bar
        self._missing.discard(slot)
        self._version += 1

    def bar(self, slice_open_ns: int) -> LiveBar | None:
        """The bar today's slice opening at `slice_open_ns` has now, None when it has
        none. Raises ConfigError for an instant that opens no slice of the day."""
        return self._bars.get(self._slot(slice_open_ns))

    def tick_bar(self, slice_open_ns: int) -> LiveBar | None:
        """The bar built from ticks that today's slice opening at `slice_open_ns` was
        given, kept after a candle replaced it: what each candle of the slice is checked
        against. None when the slice made none. Raises ConfigError for an instant that
        opens no slice of the day."""
        return self._tick_bars.get(self._slot(slice_open_ns))

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
        """The bars of `interval` the history makes that have closed when today's first
        slice opens, as store columns. The bar of a period today continues is not among
        them: `today_columns` builds it."""
        return self._split(interval)[0]

    def today_columns(self, interval: str) -> dict[str, np.ndarray]:
        """The bars of `interval` today's base bars make now, the bar of a period today
        continues built with the history's base bars of it, as store columns; the last
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
        carried = self._split(interval)[1]
        if len(carried):
            frame = pd.concat((carried, frame), ignore_index=True)
        return self._columns(frame, interval)

    def _split(self, interval: str) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
        """(the history's bars of `interval` closed by today's first open, as columns;
        the history's base bars of the period today continues, empty for none), made
        once per interval: the history never changes. Only the history's last bar can
        close after today's first open, and only for a period interval: an intraday or a
        daily bar ends with its session."""
        split = self._splits.get(interval)
        if split is None:
            columns = self._columns(self._history, interval)
            closed = int(np.searchsorted(columns["ts_close"], self._opens[0], "right"))
            base = self._history.loc[:, list(BAR_COLUMNS)]
            if closed < len(columns["ts_open"]):
                base = base[base["ts_open"] >= columns["ts_open"][closed]]
            else:
                base = base.iloc[:0]
            settled = {name: values[:closed] for name, values in columns.items()}
            split = (settled, base.reset_index(drop=True))
            self._splits[interval] = split
        return split

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
        return {name: frame[name].to_numpy() for name in COLUMNS}

    def _slot(self, slice_open_ns: int) -> int:
        slot = int(np.searchsorted(self._opens, slice_open_ns))
        if slot == len(self._opens) or int(self._opens[slot]) != slice_open_ns:
            raise ConfigError(
                f"{slice_open_ns} opens no {self.base} slice of the {self.exchange} session "
                f"the {self.symbol} series is kept for"
            )
        return slot


__all__ = ["DEFAULT_BASE", "Clock", "LiveRunContext", "LiveSeries", "LiveStore"]
