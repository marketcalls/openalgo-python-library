"""What a strategy reads: strategy bars as they close, series views over them, the `bar`
handed to `on_bar`, and positions (14-strategy-api 2.2).

A store holds the bars of one instrument for the whole run and reveals them one at a
time. `BarStore` holds the strategy bars, taken from the kernel before the run
(`Engine.strategy_bars`), and reveals each when a pause delivers it, checking that the
two agree; nothing after the current decision is ever visible. `[-1]` is the latest
closed bar. Other intervals (`self.data[symbol, "15m"]`) are in `_timeframes`.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd

from ._calendar_convert import format_ist
from ._layout import IST_OFFSET_NANOS, NANOS_PER_SECOND
from ._messages import not_supported
from ._views import TimelineView
from .errors import BacktestError

IST = dt.timezone(dt.timedelta(seconds=IST_OFFSET_NANOS // NANOS_PER_SECOND), "IST")
FIELDS = ("open", "high", "low", "close", "volume")
TIMES = ("ts_open", "ts_close")


# The Unix epoch in IST. IST has a fixed offset, so adding the microseconds since the
# epoch to it gives the same datetime as converting the UTC instant, without a time-zone
# conversion on every call.
EPOCH_IST = dt.datetime(1970, 1, 1, tzinfo=dt.UTC).astimezone(IST)


def ist_time(ts_ns: int) -> dt.datetime:
    """An IST datetime from UTC nanoseconds (to the microsecond)."""
    return EPOCH_IST + dt.timedelta(microseconds=ts_ns // 1000)


class Bar(NamedTuple):
    """One strategy bar: prices in rupees, volume in units, the open and close instants
    as UTC nanoseconds (`ts_open`, `ts_close`) and in IST (`time`, `close_time`: the bar
    opens at `time` and is closed at `close_time`). Immutable.

    The IST datetimes are made when they are read: making two for every bar was most of
    the cost of an `on_bar` call (NFR-PERF-04), and many strategies never read them."""

    symbol: str
    ts_open: int
    ts_close: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def time(self) -> dt.datetime:
        """The bar's open, IST."""
        return ist_time(self.ts_open)

    @property
    def close_time(self) -> dt.datetime:
        """The instant the bar is closed, IST."""
        return ist_time(self.ts_close)


class SeriesStore:
    """The bars of one instrument and interval over the whole run, as arrays: `ts_open`
    and `ts_close` int64 UTC nanoseconds, prices in rupees, volume in units. A subclass
    says how many are visible. `name` prefixes the labels of its views."""

    def __init__(
        self, symbol: str, interval: str, series: dict[str, np.ndarray], name: str | None = None
    ) -> None:
        self.symbol = symbol
        self.interval = interval
        self.name = name or symbol
        self._series = {name: np.asarray(series[name]) for name in (*TIMES, *FIELDS)}
        self._times: pd.DatetimeIndex | None = None

    def visible(self) -> int:
        """How many bars are revealed now."""
        raise NotImplementedError

    def __len__(self) -> int:
        return self.visible()

    def column(self, name: str) -> np.ndarray:
        """One column of the whole run (read-only use: views slice it)."""
        return self._series[name]

    @property
    def columns(self) -> dict[str, np.ndarray]:
        """The price and volume columns of the revealed bars only (read-only views)."""
        stop = self.visible()
        return {name: self._series[name][:stop] for name in FIELDS}

    def view(self, name: str) -> SeriesView:
        """The series view of one column (a futures series gives its own, `_spliced`)."""
        return SeriesView(self, name)

    def times(self) -> pd.DatetimeIndex:
        """IST open of every bar of the run."""
        if self._times is None:
            utc = pd.to_datetime(self._series["ts_open"], utc=True)
            self._times = pd.DatetimeIndex(utc.tz_convert(IST), name="time")
        return self._times

    def frame(self, last: int | None = None) -> pd.DataFrame:
        """The revealed bars as a DataFrame indexed by IST bar open; the last `last`."""
        stop = self.visible()
        start = 0 if last is None else max(stop - last, 0)
        data = {name: self._series[name][start:stop].copy() for name in FIELDS}
        return pd.DataFrame(data, index=self.times()[start:stop])


class BarStore(SeriesStore):
    """The strategy bars of one instrument, revealed by the kernel's pauses."""

    def __init__(self, symbol: str, interval: str, series: dict[str, np.ndarray]) -> None:
        super().__init__(symbol, interval, series)
        self._count = 0
        self._opens = self._series["ts_open"]
        self._size = len(self._opens)

    def visible(self) -> int:
        return self._count

    def reveal(self, row: tuple) -> Bar:
        """Reveal the next bar from the kernel's tuple (index, ts_open, ts_close, o, h, l,
        c, v). Raises BacktestError when it is not the next bar of the series (a kernel
        defect: indicators would be misaligned)."""
        count = self._count
        if count >= self._size or self._opens.item(count) != row[1]:
            raise BacktestError(
                f"{self.symbol}: the kernel delivered a strategy bar opening at "
                f"{format_ist(row[1])} that is not the next bar of its series; please report "
                "this"
            )
        self._count = count + 1
        # The row has the fields of a Bar after the instrument; tuple.__new__ is what
        # Bar._make calls, without its frame and length check, once per callback.
        return tuple.__new__(Bar, (self.symbol, *row[1:]))


class SeriesView(TimelineView):
    """One column of a store's bars: `[-1]` is the latest closed bar, slices and `len()`
    work, `.values` is a NumPy copy; arithmetic and comparisons as `TimelineView`."""

    def __init__(self, store: SeriesStore, name: str) -> None:
        super().__init__(store, store.column(name), f"{store.name}.{name}")


class InstrumentData:
    """`self.data[symbol]`: series views of one instrument's bars."""

    def __init__(self, store: SeriesStore) -> None:
        self._store = store

    def __getattr__(self, name: str) -> SeriesView:
        if name in FIELDS:
            return self._store.view(name)
        if name == "oi":
            raise not_supported(
                "open interest in strategy bars",
                "keep it in a DataFrame of your own and look it up in on_bar by bar.time",
            )
        raise AttributeError(name)

    def __len__(self) -> int:
        return self._store.visible()


@dataclass(frozen=True)
class PositionView:
    """A position as a strategy sees it: quantity in signed units, prices and P&L in
    rupees, `tranches` as (number, open units, price, opened at UTC nanoseconds)."""

    symbol: str
    qty: float
    avg_price: float | None
    adds: int
    unrealized: float
    realized: float
    tranches: tuple[tuple, ...]

    @property
    def is_long(self) -> bool:
        return self.qty > 0

    @property
    def is_short(self) -> bool:
        return self.qty < 0

    @property
    def is_flat(self) -> bool:
        return self.qty == 0

    @classmethod
    def flat(cls, symbol: str) -> PositionView:
        """A position never opened."""
        return cls(symbol, 0.0, None, 0, 0.0, 0.0, ())

    @classmethod
    def from_native(cls, symbol: str, native: dict | None) -> PositionView:
        """From the binding's position dict (None when never opened)."""
        if native is None:
            return cls.flat(symbol)
        return cls(
            symbol,
            native["quantity_units"],
            native["average_price_rupees"],
            native["adds"],
            native["unrealized_rupees"],
            native["realized_rupees"],
            tuple(native["tranches"]),
        )


__all__ = [
    "FIELDS",
    "IST",
    "Bar",
    "BarStore",
    "InstrumentData",
    "PositionView",
    "SeriesStore",
    "SeriesView",
    "ist_time",
]
