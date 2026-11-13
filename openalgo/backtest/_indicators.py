"""`self.I(func, *inputs, **params)`: indicators from `openalgo.ta` only (14-strategy-api
2.2; product decision "Indicators come only from openalgo.ta").

An indicator is computed once over the run's whole series (in Rust, by `ta`) and
revealed with its input's bars: `[-1]` is its value at the latest closed bar. Inputs are
series views (`self.close`, `self.data["SBIN", "15m"].high`) or other indicators, all of
one symbol and interval, plus plain values passed through (`self.I(ta.ema, self.close,
20)`). Series reach `ta` as pandas Series indexed by IST bar open, so anchored
indicators such as session VWAP see the bar times.

Outputs the causality registry marks as looking ahead are rejected by name: a function
whose outputs all look ahead (`ta.fractals`) is refused when registered, and a
look-ahead output of a function with causal ones (`ta.ichimoku`'s lagging span) is a
`LookaheadOutput` that raises ConfigError on any use. The causal outputs then go
through the checks of `_causality_checks`: truncation on the run's bars, and the
window-stability warm-up on the engine's reference bars, computed there from the
inputs' recipes (`_views.Recipe`), which the run waits for (`_warm_up`). An indicator of
views with a need of their own (`self.close.rolling_max(100).shift(1)`, another
indicator) needs theirs before its own (`_warm_up_measure.ReferenceCall`), and its views
carry the result as their need. An anchored indicator (`ta.vwap`) needs the bars since
the start of its anchor period, the call's `anchor` (its default when not given;
`anchor=None` is one running period, measured as any other). An input of the user's own
(a NumPy array, a pandas Series or DataFrame, as a series or as a keyword such as
`timestamps=`) has no counterpart on the reference bars, so such an indicator has no
warm-up length and is not waited for; the run says so. The series include the history
loaded before `start`, so an indicator is computed over it; one over values of the
user's own, which hold one value per bar of the run, is computed on the run's bars and
unknown (NaN, or false for a condition) on the history's.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from openalgo.indicators import causality

from ._messages import counted, not_supported
from ._views import Recipe, TimelineView, need_of
from ._warm_up_measure import ReferenceCall, fixed_inputs
from .errors import ConfigError

# The registry's word for an indicator replayed from an anchor, not a rolling window, and
# the argument that names the anchor; with an argument of `ANCHOR_OVERRIDES` the bar times
# or period starts are the user's own.
ANCHORED_NOTE = "Anchored"
ANCHOR_ARGUMENT = "anchor"
ANCHOR_OVERRIDES = ("timestamps", "session_starts")
# Plain inputs that are values of the user's own, one per bar: they have no counterpart
# on the reference bars the warm-up is measured on (`_warm_up_reference`).
OWN_VALUES = (np.ndarray, pd.Series, pd.DataFrame)
# Every output of an indicator computed on any bars (`_Call.on_bars`).
OnBars = Callable[[Mapping[str, Any]], list[np.ndarray]]


class LookaheadOutput:
    """A `ta` output that looks ahead: any use raises ConfigError naming it."""

    def __init__(self, function: str, output: str, reason: str, causal: tuple[str, ...]) -> None:
        self._message = (
            f"ta.{function} output {output!r} looks ahead and cannot be used in a strategy: "
            f"{reason}. Its causal outputs are {list(causal)}"
        )
        self._name = f"{function}.{output}"

    def _refuse(self, *args: Any) -> Any:
        raise ConfigError(self._message)

    def __getattr__(self, name: str) -> Any:
        raise ConfigError(self._message)

    def __repr__(self) -> str:
        return f"LookaheadOutput({self._name}: rejected)"


for _method in (
    "__getitem__",
    "__len__",
    "__bool__",
    "__iter__",
    "__float__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__eq__",
    "__ne__",
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    "__truediv__",
    "__rtruediv__",
    "__neg__",
    "__abs__",
):
    setattr(LookaheadOutput, _method, LookaheadOutput._refuse)
LookaheadOutput.__hash__ = None  # type: ignore[assignment]


def _record(func: Any) -> dict:
    """The causality record of a `ta` function. Raises ConfigError for anything else."""
    if not callable(func):
        raise ConfigError(f"self.I takes a function of openalgo.ta, got {func!r}")
    try:
        return causality.causality(func)
    except (KeyError, TypeError):
        raise ConfigError(
            f"self.I takes a function of openalgo.ta (for example ta.ema); "
            f"{getattr(func, '__name__', func)!r} is not one. Indicators come only from "
            "openalgo.ta, whose outputs are checked for look-ahead"
        ) from None


def _store_of(inputs: tuple) -> Any:
    """The one store every series input belongs to. Raises ConfigError for none, and
    NotImplementedError for inputs of several symbols or intervals."""
    for value in inputs:
        if isinstance(value, LookaheadOutput):
            value._refuse()
    stores = [value._store for value in inputs if isinstance(value, TimelineView)]
    if not stores:
        raise ConfigError("self.I needs at least one series input, such as self.close")
    if any(store is not stores[0] for store in stores):
        raise not_supported(
            "an indicator over series of several symbols or intervals",
            "compute one indicator per series and combine their values in on_bar",
        )
    return stores[0]


def _label(name: str, inputs: tuple, params: dict) -> str:
    shown = [value.label if isinstance(value, TimelineView) else repr(value) for value in inputs]
    shown += [f"{key}={value!r}" for key, value in params.items()]
    return f"{name}({', '.join(shown)})"


def _outputs(returned: Any, length: int, label: str) -> list[np.ndarray]:
    """The function's outputs as 1-D arrays of the series length."""
    if isinstance(returned, pd.DataFrame):
        parts = [returned[column] for column in returned.columns]
    elif isinstance(returned, tuple | list):
        parts = list(returned)
    else:
        parts = [returned]
    arrays = []
    for part in parts:
        array = np.asarray(part)
        if array.dtype != np.bool_:
            array = array.astype(float)
        if array.ndim != 1 or len(array) != length:
            raise ConfigError(f"{label} returned {array.shape} values for {counted(length, 'bar')}")
        arrays.append(array)
    return arrays


def _own_span(inputs: tuple, params: dict, length: int) -> int:
    """The bars at the end of a series of `length` an indicator is computed on: as many as
    its values of the user's own hold (one per bar of the run, whose series has the
    history before `start` in front), else all of them."""
    sizes = [
        len(value)
        for value in (*inputs, *params.values())
        if isinstance(value, OWN_VALUES) and np.ndim(value) == 1
    ]
    return min(sizes[0], length) if sizes else length


def _padded(values: np.ndarray, added: int) -> np.ndarray:
    """`values` after `added` unknown values: false for a condition, else NaN."""
    if not added:
        return values
    if values.dtype == np.bool_:
        return np.concatenate((np.zeros(added, dtype=bool), values))
    return np.concatenate((np.full(added, np.nan), values))


class _Call:
    """One indicator call: the function, its inputs and parameters, on any part of the
    input series from bar `offset` on (`_own_span`)."""

    def __init__(
        self, func: Any, inputs: tuple, params: dict, store: Any, label: str, offset: int = 0
    ) -> None:
        self.func, self.inputs, self.params, self.store = func, inputs, params, store
        self.label = label
        self.offset = offset

    def arguments(self, start: int, stop: int) -> list[Any]:
        """The inputs on bars [start, stop) after `offset`: series as pandas Series by IST
        bar open."""
        first, last = self.offset + start, self.offset + stop
        times = self.store.times()[first:last]
        return [
            pd.Series(value._values[first:last], index=times)
            if isinstance(value, TimelineView)
            else value
            for value in self.inputs
        ]

    def __call__(self, start: int, stop: int) -> list[np.ndarray]:
        returned = self.func(*self.arguments(start, stop), **self.params)
        return _outputs(returned, stop - start, self.label)

    def digest(self) -> str:
        """SHA-256 of the series inputs: the data part of the check key."""
        hasher = hashlib.sha256()
        for value in self.inputs:
            if isinstance(value, TimelineView):
                hasher.update(np.ascontiguousarray(value._values).tobytes())
        return hasher.hexdigest()[:16]

    def own_values(self) -> bool:
        """True when an input or a keyword holds values of the user's own (an array, or
        a view without a recipe): the reference bars cannot reproduce them."""
        values = (*self.inputs, *self.params.values())
        if any(isinstance(value, OWN_VALUES) for value in values):
            return True
        return any(isinstance(v, TimelineView) and v._recipe is None for v in self.inputs)

    def on_any_bars(self) -> tuple[Callable[[Mapping[str, Any]], list[Any]], Callable]:
        """(the call's arguments on any bars, from the inputs' recipes (`_views.Recipe`),
        series as pandas Series by IST bar open; the outputs from such arguments)."""
        recipes = [v._recipe if isinstance(v, TimelineView) else None for v in self.inputs]

        def arguments(bars: Mapping[str, Any]) -> list[Any]:
            times = bars["times"]
            return [
                value if recipe is None else pd.Series(recipe(bars), index=times)
                for value, recipe in zip(self.inputs, recipes, strict=True)
            ]

        def apply(values: list[Any]) -> list[np.ndarray]:
            length = next(len(v) for v in values if isinstance(v, pd.Series))
            return _outputs(self.func(*values, **self.params), length, self.label)

        return arguments, apply

    def on_bars(self) -> OnBars:
        """Every output computed on any bars (`on_any_bars`). The outputs of the last bars
        asked for are kept, so that the views of several outputs compute the indicator
        once."""
        arguments, apply = self.on_any_bars()
        last: list[tuple[Mapping[str, Any], list[np.ndarray]]] = []

        def outputs(bars: Mapping[str, Any]) -> list[np.ndarray]:
            if last and last[0][0] is bars:
                return last[0][1]
            found = apply(arguments(bars))
            last[:] = [(bars, found)]
            return found

        return outputs

    def reference(self, names: list[str], causal: tuple[str, ...]) -> ReferenceCall | None:
        """The call as the reference bars measure it (`_warm_up_measure.ReferenceCall`):
        its causal outputs and the largest need of its series inputs; None when it reads
        values of the user's own."""
        if self.own_values():
            return None
        arguments, apply = self.on_any_bars()

        def causal_outputs(values: list[Any]) -> list[np.ndarray]:
            return [part for n, part in zip(names, apply(values), strict=True) if n in causal]

        input_need = need_of(*self.inputs)
        fixed = fixed_inputs(arguments, causal_outputs) if input_need > 1 else None
        return ReferenceCall(lambda bars: causal_outputs(arguments(bars)), input_need, fixed)


def _output_recipe(on_bars: OnBars, index: int) -> Recipe:
    """The recipe of output `index` of an indicator."""

    def recipe(bars: Mapping[str, Any]) -> np.ndarray:
        return on_bars(bars)[index]

    return recipe


def _anchor(record: dict, func: Any, inputs: tuple, params: dict) -> str | None:
    """The anchor period of an anchored indicator call (module docstring), None for any
    other call, and for one whose bar times or period starts are the user's own."""
    if ANCHORED_NOTE not in (record["notes"] or ""):
        return None
    bound = inspect.signature(func).bind_partial(*inputs, **params)
    bound.apply_defaults()
    if any(bound.arguments.get(name) is not None for name in ANCHOR_OVERRIDES):
        return None
    return bound.arguments.get(ANCHOR_ARGUMENT)


def _computed(call: _Call, span: int) -> Any:
    """What the function returns on the last `span` bars of its series. Raises
    ConfigError when it fails there: on the run's series, history included, or on the
    engine's reference bars, where `init` first runs to learn the needs."""
    try:
        return call.func(*call.arguments(0, span), **call.params)
    except ValueError as exc:
        bars = counted(span, "bar")
        where = (
            f"the engine's {bars} of reference bars (init runs on them first to learn what "
            "history to load)"
            if getattr(call.store, "reference", False)
            else f"the {bars} of its series, the history before start included"
        )
        raise ConfigError(f"{call.label} failed on {where}: {exc}") from exc


def _causal_names(record: dict, outputs: list[np.ndarray]) -> tuple[list[str], tuple[str, ...]]:
    """(the name of each output, the causal ones). Raises ConfigError when none is."""
    lookahead = record["non_causal"]
    names = [*record["outputs"], *(f"output_{k}" for k in range(len(outputs)))][: len(outputs)]
    causal = tuple(n for n in names if n not in lookahead)
    if not causal:
        reasons = "; ".join(f"{n}: {lookahead[n]}" for n in names)
        raise ConfigError(
            f"ta.{record['name']} cannot be used in a strategy, it looks ahead ({reasons})"
        )
    return names, causal


def register(checks: Any, func: Any, inputs: tuple, params: dict) -> Any:
    """Register an indicator (see the module docstring): a view, or a tuple of views for
    a function with several outputs. Raises ConfigError for a function outside
    openalgo.ta, an indicator that looks ahead, or a call that fails on its series."""
    record = _record(func)
    name, lookahead = record["name"], record["non_causal"]
    if set(record["outputs"]) <= set(lookahead):
        reasons = "; ".join(f"{n}: {lookahead[n]}" for n in record["outputs"])
        raise ConfigError(f"ta.{name} cannot be used in a strategy, it looks ahead ({reasons})")
    store = _store_of(inputs)
    label = _label(name, inputs, params)
    length = len(store.column("ts_open"))
    span = _own_span(inputs, params, length)
    call = _Call(func, inputs, params, store, label, length - span)
    returned = _computed(call, span)
    outputs = _outputs(returned, span, label)
    names, causal = _causal_names(record, outputs)
    whole = [values for n, values in zip(names, outputs, strict=True) if n in causal]
    # The function, its series inputs and their data: never a parameter value, so the
    # combinations of a sweep share one check.
    series = ",".join(value.label for value in inputs if isinstance(value, TimelineView))
    key = f"{name}:{series}:{call.digest()}"

    def compute(start: int, stop: int) -> list[np.ndarray]:
        parts = call(start, stop)
        return [part for n, part in zip(names, parts, strict=True) if n in causal]

    # A futures series is re-adjusted at each roll: its indicators are checked point in
    # time and recomputed at every roll (`_spliced`).
    adapter = getattr(store, "indicator_adapter", None)
    if adapter is not None:
        adapter = adapter(call, names, causal)
        compute, whole = adapter.compute, adapter.whole
    reference = call.reference(names, causal)
    anchor = _anchor(record, func, inputs, params)
    series_of = (store.symbol, store.interval)
    outcome = checks.check(key, label, compute, whole, anchor, series_of, reference)
    on_bars = None if reference is None else call.on_bars()
    views = [
        LookaheadOutput(name, n, lookahead[n], causal)
        if n in lookahead
        else TimelineView(
            store,
            _padded(values, length - span),
            label if len(outputs) == 1 else f"{label}.{n}",
            None if on_bars is None else _output_recipe(on_bars, k),
            outcome.warm_up,
        )
        for k, (n, values) in enumerate(zip(names, outputs, strict=True))
    ]
    if adapter is not None:
        views = adapter.adopt(views)
    return views[0] if len(views) == 1 and not isinstance(returned, tuple | list) else tuple(views)


__all__ = ["LookaheadOutput", "register"]
