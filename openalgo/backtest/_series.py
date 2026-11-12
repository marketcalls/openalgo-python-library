"""What a strategy reads: strategy bars as they close, series views over them, the `bar`
handed to `on_bar`, and positions (14-strategy-api 2.2).

Bars are revealed one strategy bar at a time, at its close, from the kernel's pauses;
nothing after the current decision is ever visible. `[-1]` is the latest closed bar.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._expr import ConditionOperators, Expr, column
from ._layout import IST_OFFSET_NANOS, NANOS_PER_SECOND

IST = dt.timezone(dt.timedelta(seconds=IST_OFFSET_NANOS // NANOS_PER_SECOND), "IST")
FIELDS = ("open", "high", "low", "close", "volume")


def ist_time(ts_ns: int) -> dt.datetime:
    """An IST datetime from UTC nanoseconds (to the microsecond)."""
    utc = dt.datetime(1970, 1, 1, tzinfo=dt.UTC) + dt.timedelta(microseconds=ts_ns // 1000)
    return utc.astimezone(IST)


@dataclass(frozen=True)
class Bar:
    """One strategy bar: prices in rupees, volume in units, times in IST (the bar opens at
    `time` and is closed at `close_time`)."""

    symbol: str
    time: dt.datetime
    close_time: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class BarStore:
    """The strategy bars of one instrument revealed so far."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.ts_open: list[int] = []
        self.ts_close: list[int] = []
        self.columns: dict[str, list[float]] = {name: [] for name in FIELDS}

    def append(self, row: tuple) -> Bar:
        """Reveal one bar from the kernel's tuple (index, ts_open, ts_close, o, h, l, c, v)."""
        _, ts_open, ts_close, *values = row
        self.ts_open.append(ts_open)
        self.ts_close.append(ts_close)
        for name, value in zip(FIELDS, values, strict=True):
            self.columns[name].append(value)
        return Bar(self.symbol, ist_time(ts_open), ist_time(ts_close), *values)

    def __len__(self) -> int:
        return len(self.ts_open)

    def frame(self, last: int | None = None) -> pd.DataFrame:
        """The revealed bars as a DataFrame indexed by IST bar open."""
        start = 0 if last is None else max(len(self) - last, 0)
        index = pd.DatetimeIndex([ist_time(ts) for ts in self.ts_open[start:]], name="time")
        data = {name: self.columns[name][start:] for name in FIELDS}
        return pd.DataFrame(data, index=index)


class SeriesView(ConditionOperators):
    """One column of revealed bars: `[-1]` is the latest closed bar, slices and `len()`
    work, `.values` is a NumPy copy. In conditions (`entry_when`) it stands for the
    whole column (`_expr`)."""

    def __init__(self, store: BarStore, name: str) -> None:
        self._store = store
        self._name = name

    def __getitem__(self, key: int | slice) -> float | np.ndarray:
        column = self._store.columns[self._name]
        if isinstance(key, slice):
            return np.asarray(column[key], dtype=float)
        return column[key]

    def __len__(self) -> int:
        return len(self._store)

    @property
    def values(self) -> np.ndarray:
        return np.asarray(self._store.columns[self._name], dtype=float)

    def __repr__(self) -> str:
        return f"SeriesView({self._store.symbol}.{self._name}, {len(self)} bars)"

    def __condition_expr__(self) -> Expr:
        """The column as a condition operand."""
        return column(self._store.symbol, self._name)


class InstrumentData:
    """`self.data[symbol]`: series views of one instrument's strategy bars."""

    def __init__(self, store: BarStore) -> None:
        self._store = store

    def __getattr__(self, name: str) -> SeriesView:
        if name in FIELDS:
            return SeriesView(self._store, name)
        if name == "oi":
            raise NotImplementedError(
                "open interest in strategy bars is not implemented yet (Phase 1 work in progress)"
            )
        raise AttributeError(name)

    def __len__(self) -> int:
        return len(self._store)


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
    "SeriesView",
    "ist_time",
]
