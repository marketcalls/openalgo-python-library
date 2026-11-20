"""The live store of one instrument at one interval (16-session-design 1.5 and 2.5 C,
unit 6): the bars a strategy reads, built from a `series.LiveSeries` and cut at the
kernel's decision instant. Part of `series`, which re-exports `LiveStore` and `Clock`.

Times are UTC nanoseconds, prices rupees, volume units. Not thread-safe: the session loop
owns the stores, as it owns the engines.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .._series import FIELDS, TIMES, Bar, SeriesStore
from .._views import field_recipe
from ..errors import LiveError
from ._series_indicators import LiveIndicator, LiveView

if TYPE_CHECKING:
    from .series import LiveSeries

#: The decision instant a store's bars are visible up to (UTC nanoseconds), None before
#: the first decision.
Clock = Callable[[], int | None]
#: The columns of a store, times first, as `_series.SeriesStore` holds them.
COLUMNS = (*TIMES, *FIELDS)


class LiveStore(SeriesStore):
    """One instrument's bars at one interval in a live session (`series` module
    docstring): the history's bars closed before today (`history_bars` of them), then the
    bars today makes, each held once the decision instant `clock` gives is at or after
    its close. None from `clock` (no decision yet, `init` running) holds the history's
    bars and shows none, and every bar the store holds is visible once a decision has
    come, as in a backtest, where `init` sees none and the history is visible from the
    first pause.

    `version` changes whenever the bars change: a decision revealed a bar, or a candle
    filled a base slice that had none. Its views (`view`) and indicators are live views
    that recompute once per change (`_series_indicators`). Made by `LiveSeries.store`.
    """

    def __init__(
        self, series: LiveSeries, interval: str, clock: Clock, name: str | None = None
    ) -> None:
        super().__init__(series.symbol, interval, series.history_columns(interval), name)
        self.history_bars = len(self._series["ts_open"])
        self._history = dict(self._series)
        self._source = series
        self._clock = clock
        # The series' version the bars were built at, every bar built (history first),
        # how many of them the store holds, the store's own version and the recipe input
        # of that version.
        self._built_at = -1
        self._all = self._history
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
        series took a bar, then every bar, the history's too, cut at the decision
        instant, so the store holds a bar once the decision is at or after its close."""
        source = self._source
        rebuilt = source.version != self._built_at
        if rebuilt:
            today = source.today_columns(self.interval)
            self._all = {
                name: np.concatenate((self._history[name], today[name])) for name in COLUMNS
            }
            self._built_at = source.version
        now = self._clock()
        closes = self._all["ts_close"]
        held = self.history_bars if now is None else int(np.searchsorted(closes, now, "right"))
        if rebuilt or held != self._held:
            self._series = {name: values[:held] for name, values in self._all.items()}
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
        series' bar, built from the bars built from ticks and the candles of minutes that
        made none (`series` module docstring).

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
        return Bar(self.symbol, *(self._series[name].item(index) for name in COLUMNS))

    def indicator_adapter(self, call: Any, names: list[str], causal: tuple[str, ...]) -> Any:
        """What `_indicators.register` needs for an indicator on this store."""
        return LiveIndicator(self, call, names, causal)


__all__ = ["COLUMNS", "Clock", "LiveStore"]
