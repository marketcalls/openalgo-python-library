"""`self.data[symbol, interval]`: an instrument's bars at another interval, each visible
only once it has closed (03-architecture 4.2, "Higher timeframes").

The bars are built once from the run's base bars with the same resampler and calendar
the kernel uses, so a 15-minute bar is the aggregate of the base bars present in it. A
bar is visible when its close is at or before the current decision instant: on a
1-minute strategy the 09:15 to 09:30 bar appears at the decision of 09:30, never at
09:29. Any interval that is a whole multiple of the base interval works ("15m", "1h",
"D", "W", ...), coarser or finer than the strategy interval.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from ._layout import parse_interval
from ._series import FIELDS, TIMES, SeriesStore
from .errors import ConfigError
from .resample import resample_bars


class TimeframeStore(SeriesStore):
    """Bars of another interval, visible once closed by the run's clock."""

    def __init__(
        self,
        symbol: str,
        interval: str,
        series: dict[str, np.ndarray],
        clock: Callable[[], int | None],
    ) -> None:
        super().__init__(symbol, interval, series, name=f"{symbol}[{interval}]")
        self._clock = clock

    def visible(self) -> int:
        now = self._clock()
        if now is None:
            return 0
        return int(np.searchsorted(self.column("ts_close"), now, side="right"))


def _check_interval(interval: str, base: str) -> None:
    """Raise ConfigError unless `interval` can be built from bars of `base`."""
    kind, size = parse_interval(interval)
    base_kind, base_size = parse_interval(base)
    if kind == "intraday" and (base_kind != "intraday" or size % base_size):
        raise ConfigError(
            f"self.data[symbol, {interval!r}]: {interval} bars cannot be built from the "
            f"run's {base} base bars; use a whole multiple of {base}"
        )


def timeframe_store(
    symbol: str,
    exchange: str,
    interval: str,
    base: str,
    base_bars: pd.DataFrame,
    calendar: object,
    clock: Callable[[], int | None],
) -> TimeframeStore:
    """The store of `symbol` at `interval`, built from the run's base bars (canonical
    layout, already cut to the run's slices). Raises ConfigError for an interval that
    cannot be built from the base interval."""
    _check_interval(interval, base)
    bars = base_bars
    if interval != base:
        bars = resample_bars(
            base_bars,
            interval,
            exchange=exchange,
            calendar=calendar,
            source_interval=base,
            on_closed_day="drop",
        )
    series = {name: bars[name].to_numpy() for name in (*TIMES, *FIELDS)}
    return TimeframeStore(symbol, interval, series, clock)


__all__ = ["TimeframeStore", "timeframe_store"]
