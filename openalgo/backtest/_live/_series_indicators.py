"""Views and `self.I` indicators of a live store (`series.LiveStore`), recomputed as the
store grows (16-session-design 1.5 and 2.5 C, unit 6).

A backtest computes every view and indicator once over the whole run and shows a growing
part of it (`_views`). A live store's bars arrive one close at a time, and a broker
candle may fill a minute that had no bar after later bars were shown (D27), so here a
view holds how its values are computed instead of the values themselves:

- the view of a bar field reads the store's column;
- a view built from other views (arithmetic, comparisons, `&`, `|`, `~`, `shift`,
  `rolling_max`, `rolling_min`, a crossing) applies the backtest's own operation
  (`TimelineView`) to its operands' values as they stand;
- an indicator calls its `openalgo.ta` function on its inputs' values as they stand.

A view recomputes the first time it is read after the store changed (a close revealed a
bar, or a candle filled a minute) and keeps the values until the next change, so each view
is computed at most once per change, and a view built in `init` stays current all day.
Only causal outputs reach a strategy: `_indicators.register` refuses a function whose
outputs all look ahead and turns a look-ahead output into a `LookaheadOutput`, as in a
backtest (the registry `openalgo.indicators.causality`). Because a causal value uses
only the bars up to its own, the value recomputed at each close equals the value a
backtest computes over its whole series for that bar.

Not live: an indicator or view over values of the user's own (an array, a pandas Series
or DataFrame), which hold one value per bar of a run known in advance; the session's
bars are not. Such a call raises NotImplementedError saying what to do instead.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .._indicator_values import outputs_of
from .._messages import not_supported
from .._views import Recipe, TimelineView

#: What a live view's values are computed by: no argument, the values now.
ValuesNow = Callable[[], np.ndarray]

_OWN_VALUES = (
    "values of your own (an array, a pandas Series or DataFrame) in a live session",
    "compute it from series views such as self.close, which the session extends at every "
    "close; values of your own hold one value per bar of a run known in advance",
)


class LiveSource(Protocol):
    """What a live view needs of its store (`series.LiveStore`): `version`, a number that
    changes whenever the store's bars change, and `bars()`, the bar fields a recipe
    reads (`_views.Recipe`), plus the `Store` attributes every view uses."""

    symbol: str
    interval: str
    waiting: bool

    @property
    def version(self) -> int:
        """Changes whenever the store's bars change."""
        ...

    def visible(self) -> int:
        """How many bars are shown now."""
        ...

    def times(self) -> pd.DatetimeIndex:
        """IST open of every bar the store holds."""
        ...

    def column(self, name: str) -> np.ndarray:
        """One column of the bars the store holds."""
        ...

    def bars(self) -> Mapping[str, Any]:
        """The bar fields of the store's bars, as a recipe reads them."""
        ...


class LiveView(TimelineView):
    """A view of a live store whose values are those `compute` gives for the store's
    bars now, recomputed once per change of the store (module docstring). Reads,
    arithmetic, comparisons and labels are the backtest's (`TimelineView`); every view
    built from a live view is a live view. `recipe` and `need` are as on `TimelineView`.
    """

    def __init__(
        self,
        store: LiveSource,
        compute: ValuesNow,
        label: str,
        recipe: Recipe | None = None,
        need: int | None = 1,
        bar_field: bool = False,
    ) -> None:
        self._compute = compute
        self._cached: tuple[int, np.ndarray] | None = None
        super().__init__(store, None, label, recipe, need, bar_field)

    @property  # type: ignore[override]
    def _values(self) -> np.ndarray:
        """The view's values on the store's bars now (one per bar the store holds)."""
        version = self._store.version
        cached = self._cached
        if cached is None or cached[0] != version:
            cached = (version, np.asarray(self._compute()))
            self._cached = cached
        return cached[1]

    @_values.setter
    def _values(self, values: np.ndarray | None) -> None:
        # `TimelineView.__init__` assigns the values it is given: a live view is built
        # with none and computes them when it is first read.
        self._cached = None if values is None else (self._store.version, np.asarray(values))

    def _built(self, method: str, *args: Any) -> Any:
        """The view `TimelineView.<method>` builds from this view and `args`, as a live
        view that applies the same method to its operands' values whenever the store
        changes."""
        operands = tuple(adopted(value) for value in args)
        built = getattr(TimelineView, method)(self, *operands)
        if not isinstance(built, TimelineView):
            return built

        def compute() -> np.ndarray:
            return getattr(TimelineView, method)(self, *operands)._values

        return LiveView(self._store, compute, built.label, built._recipe, built.need)

    def _derive(self, other: Any, function: Callable, symbol: str, flip: bool = False) -> Any:
        return self._built("_derive", other, function, symbol, flip)

    def _unary(
        self, compute: Callable[[np.ndarray], np.ndarray], label: str, added: int = 0
    ) -> TimelineView:
        return self._built("_unary", compute, label, added)

    def _cross(self, other: Any, function: str, name: str) -> TimelineView:
        return self._built("_cross", other, function, name)


def adopted(value: Any) -> Any:
    """`value` as an operand of a live view: a live view or anything that is not a view
    as it is; a view of a live store that is not live yet (a clock condition such as
    `bt.time_between` meeting a live view) as a live view computed by its recipe on the
    store's bars. Raises NotImplementedError for a view of values of the user's own,
    which has no recipe."""
    if isinstance(value, LiveView) or not isinstance(value, TimelineView):
        return value
    store, recipe = value._store, value._recipe
    if recipe is None:
        raise not_supported(f"{value.label}: a view over {_OWN_VALUES[0]}", _OWN_VALUES[1])

    def compute() -> np.ndarray:
        return recipe(store.bars())

    return LiveView(store, compute, value.label, recipe, value.need, value.bar_field)


class LiveIndicator:
    """An `self.I` indicator on a live store: what `_indicators.register` asks of a store
    that has an `indicator_adapter` (as the spliced series of `_spliced` has).

    `whole` are its causal outputs on the store's bars at registration, and `compute`
    gives them on part of those bars, for the causality checks (`_causality_checks`);
    `adopt` turns the views `register` built into live views of `outputs`, every output
    on the store's bars now, and keeps them (`views`, the causal outputs' only, for the
    session's decisions file). `call` is the indicator call of `_indicators` (its
    function, inputs, parameters and label), `names` its output names and `causal` the
    names that do not look ahead.

    Raises NotImplementedError for a call over values of the user's own (module
    docstring).
    """

    def __init__(
        self, store: LiveSource, call: Any, names: list[str], causal: tuple[str, ...]
    ) -> None:
        if call.own_values():
            raise not_supported(f"{call.label}: an indicator over {_OWN_VALUES[0]}", _OWN_VALUES[1])
        self._store, self._call = store, call
        self._causal = [k for k, name in enumerate(names) if name in causal]
        self._cached: tuple[int, list[np.ndarray]] | None = None
        self.whole = [self.outputs()[k] for k in self._causal]
        self.views: list[LiveView] = []

    def _on(self, start: int, stop: int) -> list[np.ndarray]:
        """Every output on the store's bars [start, stop), the inputs' values as they
        stand (each live input recomputed when the store changed)."""
        call, times = self._call, self._store.times()[start:stop]
        arguments = [
            pd.Series(value._values[start:stop], index=times)
            if isinstance(value, TimelineView)
            else value
            for value in (adopted(value) for value in call.inputs)
        ]
        returned = call.func(*arguments, **call.params)
        return outputs_of(returned, stop - start, call.label)

    def outputs(self) -> list[np.ndarray]:
        """Every output on all the store's bars now, computed once per change."""
        version = self._store.version
        if self._cached is None or self._cached[0] != version:
            length = len(self._store.column("ts_open"))
            self._cached = (version, self._on(0, length))
        return self._cached[1]

    def compute(self, start: int, stop: int) -> list[np.ndarray]:
        """The causal outputs on the store's bars [start, stop), for the truncation
        check."""
        outputs = self._on(start, stop)
        return [outputs[k] for k in self._causal]

    def _output(self, index: int) -> ValuesNow:
        def compute() -> np.ndarray:
            return self.outputs()[index]

        return compute

    def adopt(self, views: list[Any]) -> list[Any]:
        """The views `register` built, each output a live view of `outputs`; a
        `LookaheadOutput` stays as it is (any use of it raises)."""
        adopted = [
            LiveView(self._store, self._output(k), view.label, view._recipe, view.need)
            if isinstance(view, TimelineView)
            else view
            for k, view in enumerate(views)
        ]
        self.views = [view for view in adopted if isinstance(view, LiveView)]
        return adopted


__all__ = ["LiveIndicator", "LiveSource", "LiveView", "ValuesNow", "adopted"]
