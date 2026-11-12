"""The spliced series of a futures strategy, back-adjusted as known at each decision
point (04-execution 7; 14-strategy-api 2.1; decision SPEC-30; lead rulings LEAD-04 and
LEAD-16; 06-data-layer FR-DATA-16).

A futures strategy reads one series per underlying: the strategy bars of the contract
it trades at each bar's open, the near contract's before a switch and the next
contract's from it (`futures::Switch` in the kernel). At each switch the history is
re-adjusted by the switch's spread: every earlier bar's prices move by the spread, so
the series does not jump at the roll and the current contract's prices stay exact.

What is known changes with the decision point, so the values do too:

- an "epoch" is the number of switches known at a decision instant (a switch is known
  from its slice's open);
- every view of the series (`self.close`, the expressions built from it, `self.I`
  indicators) shows the values computed on the history as re-adjusted in the current
  epoch, so `[-k]` comparisons across a roll are on one basis: the store re-adjusts its
  columns in place at each switch, and every other view recomputes its values on the
  re-adjusted history the first time it is read in a new epoch (`SplicedView`);
- the bar handed to `on_bar` is the contract's own bar, so values recorded in the result
  (decisions, logs) are those used and are never revised;
- the causality checks of an indicator run point in time: each bar's value is the one
  computed on the history known at that bar's close, and a truncated computation must
  reproduce it (03-architecture 5).

Volume is never adjusted.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ._series import FIELDS, TIMES, Bar, BarStore, SeriesStore
from ._views import TimelineView

PRICE_FIELDS = ("open", "high", "low", "close")
EpochValues = Callable[[int], np.ndarray]


class SplicedView(TimelineView):
    """A view of a spliced series whose values follow the epoch (module docstring).
    `at_epoch(e)` gives the values of the whole run as computed in epoch `e`; the view's
    own values are those of the store's epoch, recomputed in place when it is read."""

    def __init__(
        self, store: SplicedStore, values: np.ndarray, label: str, at_epoch: EpochValues
    ) -> None:
        super().__init__(store, values, label)
        self._compute_at = at_epoch
        self._cached: tuple[int, np.ndarray] | None = None
        self._epoch = store.epoch

    def at_epoch(self, epoch: int) -> np.ndarray:
        """The values of the whole run as computed in `epoch` (cached for the last one)."""
        if self._cached is None or self._cached[0] != epoch:
            self._cached = (epoch, np.asarray(self._compute_at(epoch)))
        return self._cached[1]

    def sync(self) -> None:
        """Bring the values to the store's epoch (in place, once per epoch)."""
        epoch = self._store.epoch
        if epoch != self._epoch:
            self._values[...] = self.at_epoch(epoch)
            self._epoch = epoch

    def _shown(self) -> np.ndarray:
        self.sync()
        return super()._shown()

    def _wrap(self, method: str, *args: Any) -> Any:
        """The view `TimelineView.<method>` builds, as a spliced view that rebuilds itself
        from its operands' values in any epoch."""
        for value in (self, *args):
            if isinstance(value, SplicedView):
                value.sync()
        built = getattr(TimelineView, method)(self, *args)
        if not isinstance(built, TimelineView):
            return built

        def at_epoch(epoch: int) -> np.ndarray:
            def plain(value: Any) -> Any:
                if isinstance(value, SplicedView):
                    return TimelineView(value._store, value.at_epoch(epoch), value.label)
                return value

            rebuilt = getattr(TimelineView, method)(plain(self), *(plain(a) for a in args))
            return rebuilt._values

        return SplicedView(self._store, built._values, built.label, at_epoch)

    def _derive(self, other: Any, function: Callable, symbol: str, flip: bool = False) -> Any:
        return self._wrap("_derive", other, function, symbol, flip)

    def __neg__(self) -> Any:
        return self._wrap("__neg__")

    def __abs__(self) -> Any:
        return self._wrap("__abs__")

    def __invert__(self) -> Any:
        return self._wrap("__invert__")

    def _cross(self, other: Any, function: str, name: str) -> Any:
        return self._wrap("_cross", other, function, name)

    def shift(self, bars: int) -> Any:
        return self._wrap("shift", bars)

    def _window(self, bars: int, function: str, name: str) -> Any:
        return self._wrap("_window", bars, function, name)


def _splice(
    sources: list[SeriesStore], starts: list[int]
) -> tuple[dict[str, np.ndarray], list[tuple[int, int]], list[int]]:
    """The bars of each source from its start instant (UTC nanoseconds) to the next
    source's: the columns, the (source, row) of every bar, and where each source after
    the first begins."""
    pieces: dict[str, list[np.ndarray]] = {name: [] for name in (*TIMES, *FIELDS)}
    origin: list[tuple[int, int]] = []
    begins: list[int] = []
    bounds = [None, *starts, None]
    for k, source in enumerate(sources):
        opens = source.column("ts_open")
        keep = np.ones(len(opens), dtype=bool)
        if bounds[k] is not None:
            keep &= opens >= bounds[k]
        if bounds[k + 1] is not None:
            keep &= opens < bounds[k + 1]
        if k > 0:
            begins.append(len(origin))
        rows = np.flatnonzero(keep)
        origin += [(k, int(row)) for row in rows]
        for name in pieces:
            pieces[name].append(np.asarray(source.column(name))[rows])
    columns = {name: np.concatenate(parts) for name, parts in pieces.items()}
    return columns, origin, begins


class SplicedStore(SeriesStore):
    """One underlying's spliced strategy bars (module docstring).

    sources: the contracts' strategy-bar stores in series order; switches: the kernel's
    switches of this series in order (dicts with `slice_ts` and `spread_rupees`)."""

    def __init__(
        self, name: str, interval: str, sources: list[BarStore], switches: list[dict]
    ) -> None:
        starts = [int(switch["slice_ts"]) for switch in switches]
        columns, self._origin, self._begins = _splice(list(sources), starts)
        self._raw = {key: values.copy() for key, values in columns.items()}
        super().__init__(name, interval, columns, name)
        self._known_at = np.asarray(starts, dtype=np.int64)
        self._spreads = [float(switch["spread_rupees"]) for switch in switches]
        self._epoch = 0
        self._count = 0

    # ----- revealing and the epoch ------------------------------------------------

    def visible(self) -> int:
        return self._count

    @property
    def epoch(self) -> int:
        """The number of switches known now."""
        return self._epoch

    def advance(self, decided_at: int | None) -> None:
        """Move to the epoch of a decision instant: re-adjust the history by every switch
        now known (the store's columns in place; the other views when they are read)."""
        epoch = (
            0 if decided_at is None else int(np.searchsorted(self._known_at, decided_at, "right"))
        )
        if epoch <= self._epoch:
            return
        for name in PRICE_FIELDS:
            self._series[name][...] = self.column_at(name, epoch)
        self._epoch = epoch

    def reveal_from(self, source: int, row: int, bar: Bar) -> Bar | None:
        """Reveal the next spliced bar when it is row `row` of source `source`; returns
        the contract's bar (its own prices, as used), or None when it is not."""
        if self._count < len(self._origin) and self._origin[self._count] == (source, row):
            self._count += 1
            return bar
        return None

    # ----- values by epoch ----------------------------------------------------------

    def column_at(self, name: str, epoch: int) -> np.ndarray:
        """A column of the whole run as re-adjusted in `epoch` (prices only)."""
        values = self._raw[name].copy()
        if name in PRICE_FIELDS:
            for begin, spread in zip(self._begins[:epoch], self._spreads[:epoch], strict=True):
                values[:begin] += spread
        return values

    def bar_epochs(self) -> np.ndarray:
        """The epoch in which each bar is the latest closed one (at its close)."""
        return np.searchsorted(self._known_at, self._raw["ts_close"], "right")

    # ----- views ----------------------------------------------------------------------

    def view(self, name: str) -> SplicedView:
        """The view of one column; it shares the store's live column, which `advance`
        re-adjusts in place."""
        return SplicedView(
            self, self.column(name), f"{self.name}.{name}", lambda e: self.column_at(name, e)
        )

    def indicator_adapter(self, call: Any, names: list[str], causal: tuple[str, ...]) -> Any:
        """What `_indicators.register` needs for an indicator on this series."""
        return IndicatorAdapter(self, call, names, causal)


class IndicatorAdapter:
    """An indicator on a spliced series: its outputs in any epoch, its point-in-time
    values for the causality checks, and its views (module docstring).

    call: the indicator call of `_indicators` (func, inputs, params, label); names:
    its output names; causal: the names that do not look ahead."""

    def __init__(
        self, store: SplicedStore, call: Any, names: list[str], causal: tuple[str, ...]
    ) -> None:
        self._store, self._call, self._names = store, call, names
        self._causal = [k for k, name in enumerate(names) if name in causal]
        self._cached: tuple[int, list[np.ndarray]] | None = None
        self._epochs = store.bar_epochs()
        self.whole = self._point_in_time()

    def _outputs(self, epoch: int, start: int, stop: int) -> list[np.ndarray]:
        """Every output on bars [start, stop) with the inputs as re-adjusted in `epoch`."""
        from ._indicators import _outputs

        times = self._store.times()[start:stop]
        arguments = [
            pd.Series(value.at_epoch(epoch)[start:stop], index=times)
            if isinstance(value, SplicedView)
            else value
            for value in self._call.inputs
        ]
        returned = self._call.func(*arguments, **self._call.params)
        return _outputs(returned, stop - start, self._call.label)

    def outputs_at(self, epoch: int) -> list[np.ndarray]:
        """Every output of the whole run in `epoch` (cached for the last epoch asked)."""
        if self._cached is None or self._cached[0] != epoch:
            length = len(self._store.column("ts_open"))
            self._cached = (epoch, self._outputs(epoch, 0, length))
        return self._cached[1]

    def _point_in_time(self) -> list[np.ndarray]:
        """The causal outputs as used: each bar's value from the epoch of its close."""
        whole = [np.array(self.outputs_at(0)[k], copy=True) for k in self._causal]
        for epoch in np.unique(self._epochs):
            mask = self._epochs == epoch
            outputs = self.outputs_at(int(epoch))
            for values, k in zip(whole, self._causal, strict=True):
                values[mask] = outputs[k][mask]
        return whole

    def compute(self, start: int, stop: int) -> list[np.ndarray]:
        """The causal outputs on bars [start, stop) for the checks: the last bar's value
        computed on that history as known at its close, the earlier ones as used."""
        epoch = int(self._epochs[stop - 1])
        outputs = self._outputs(epoch, start, stop)
        parts = [outputs[k] for k in self._causal]
        for part, used in zip(parts, self.whole, strict=True):
            part[:-1] = used[start : stop - 1]
        return parts

    def _output_of(self, index: int) -> EpochValues:
        """Output `index` of the whole run, by epoch."""

        def at_epoch(epoch: int) -> np.ndarray:
            return self.outputs_at(epoch)[index]

        return at_epoch

    def adopt(self, views: list[Any]) -> list[Any]:
        """The register's views, as spliced views recomputed at every switch; their
        values are those of the store's epoch (the register computed them on its inputs
        as they stood, which may be an earlier epoch)."""
        adopted = []
        epoch = self._store.epoch
        for k, view in enumerate(views):
            if isinstance(view, TimelineView):
                values = np.array(self.outputs_at(epoch)[k], copy=True)
                view = SplicedView(self._store, values, view.label, self._output_of(k))
            adopted.append(view)
        return adopted


__all__ = ["IndicatorAdapter", "SplicedStore", "SplicedView"]
