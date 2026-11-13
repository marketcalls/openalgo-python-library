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
carry the result as their need. An anchored indicator (the registry's `anchor`:
`ta.vwap`, `ta.pivot_points`) needs the bars since the start of the first anchor period
it reads, by the call's `anchor` (its default when not given; `anchor=None` is one
running period, or no anchor at all, measured as any other). An input of the user's own
(a NumPy array, a pandas Series or DataFrame, as a series or as a keyword such as
`timestamps=`) has no counterpart on the reference bars, so such an indicator has no
warm-up length and is not waited for; the run says so. The series include the history
loaded before `start`, so an indicator is computed over it; one over values of the
user's own, which hold one value per bar of the run, is computed on the run's bars and
unknown (NaN, or false for a condition) on the history's.

**A series too short for its indicator**: when the bars an indicator is computed on are
fewer than its own window (its `period` or `length` argument), or, for a call without
one, than its warm-up, it is not computed (`_indicator_values.short_outputs`): every
value is unknown and its
outcome records the bars it needs (`CheckOutcome.required`), so the strategy makes no
decision for that symbol (`_warm_up`). The series length decides, never the error a call
raises: any other error of the call stops the run with a ConfigError.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from openalgo.indicators import causality

from ._indicator_values import outputs_of, padded, required_bars, short_outputs
from ._messages import counted, not_supported
from ._views import Recipe, TimelineView, need_of
from ._warm_up_anchor import PeriodAnchor
from ._warm_up_measure import ReferenceCall, fixed_inputs
from .errors import ConfigError

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
        return outputs_of(returned, stop - start, self.label)

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
            return outputs_of(self.func(*values, **self.params), length, self.label)

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

    def reference(self, record: dict) -> ReferenceCall | None:
        """The call as the reference bars measure it (`_warm_up_measure.ReferenceCall`):
        its causal outputs by the registry `record` and the largest need of its series
        inputs; None when it reads values of the user's own."""
        if self.own_values():
            return None
        arguments, apply = self.on_any_bars()

        def causal_outputs(values: list[Any]) -> list[np.ndarray]:
            outputs = apply(values)
            names, causal = _causal_names(record, outputs)
            return [part for n, part in zip(names, outputs, strict=True) if n in causal]

        input_need = need_of(*self.inputs)
        fixed = fixed_inputs(arguments, causal_outputs) if input_need > 1 else None
        return ReferenceCall(lambda bars: causal_outputs(arguments(bars)), input_need, fixed)


def _output_recipe(on_bars: OnBars, index: int) -> Recipe:
    """The recipe of output `index` of an indicator."""

    def recipe(bars: Mapping[str, Any]) -> np.ndarray:
        return on_bars(bars)[index]

    return recipe


def _bound(func: Any, inputs: tuple, params: dict) -> dict[str, Any]:
    """Every argument of the call by name, defaults included."""
    bound = inspect.signature(func).bind_partial(*inputs, **params)
    bound.apply_defaults()
    return bound.arguments


def _anchor(record: dict, func: Any, inputs: tuple, params: dict) -> PeriodAnchor | None:
    """The anchor of an anchored indicator call (module docstring, the registry's
    `anchor`), None for any other call, for one that names no anchor period, and for one
    whose bar times or period starts are the user's own."""
    anchored = record["anchor"]
    if anchored is None:
        return None
    arguments = _bound(func, inputs, params)
    if any(arguments.get(name) is not None for name in anchored["overrides"]):
        return None
    period = arguments.get(anchored["argument"])
    return None if period is None else PeriodAnchor(period, anchored["periods_before"])


def _failure(call: _Call, bars: int, exc: ValueError) -> ConfigError:
    """The error of a call that failed on `bars` bars: of the run's series, history
    included, or of the engine's reference bars, where `init` first runs to learn the
    needs."""
    counted_bars = counted(bars, "bar")
    where = (
        f"the engine's {counted_bars} of reference bars (init runs on them first to learn "
        "what history to load)"
        if getattr(call.store, "reference", False)
        else f"the {counted_bars} of its series, the history before start included"
    )
    return ConfigError(f"{call.label} failed on {where}: {exc}")


def _computed(call: _Call, span: int) -> Any:
    """What the function returns on the last `span` bars of its series. Raises
    ConfigError when it fails there (`_failure`)."""
    try:
        return call.func(*call.arguments(0, span), **call.params)
    except ValueError as exc:
        raise _failure(call, span, exc) from exc


def _short(call: _Call, span: int, required: int) -> tuple[bool, list[np.ndarray]]:
    """`_indicator_values.short_outputs` of a call on the last `span` bars of a series
    too short for it. Raises ConfigError when the call fails (`_failure`)."""
    arguments = call.arguments(0, span)
    try:
        return short_outputs(call.func, arguments, call.params, span, required, call.label)
    except ValueError as exc:
        raise _failure(call, span, exc) from exc


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


def _checked(
    checks: Any,
    call: _Call,
    record: dict,
    outputs: list[np.ndarray],
    found: tuple[PeriodAnchor | None, ReferenceCall | None, int | None],
) -> tuple[Any, Any]:
    """(the outcome of the checks of a call, the futures adapter of its series or None).
    `outputs` are its outputs on its series, `found` its anchor, reference call and, for
    a series too short for it, the bars it needs (it was not computed, so neither its
    truncation is checked nor an adapter made). Raises ConfigError as `checks.check`
    does."""
    anchor, reference, required = found
    names, causal = _causal_names(record, outputs)
    whole = [values for n, values in zip(names, outputs, strict=True) if n in causal]
    # The function, its series inputs and their data: never a parameter value, so the
    # combinations of a sweep share one check.
    series = ",".join(value.label for value in call.inputs if isinstance(value, TimelineView))
    key = f"{record['name']}:{series}:{call.digest()}"

    def compute(start: int, stop: int) -> list[np.ndarray]:
        parts = call(start, stop)
        return [part for n, part in zip(names, parts, strict=True) if n in causal]

    # A futures series is re-adjusted at each roll: its indicators are checked point in
    # time and recomputed at every roll (`_spliced`).
    adapter = None if required is not None else getattr(call.store, "indicator_adapter", None)
    if adapter is not None:
        adapter = adapter(call, names, causal)
        compute, whole = adapter.compute, adapter.whole
    if required is not None:
        whole = []
    series_of = (call.store.symbol, call.store.interval)
    outcome = checks.check(
        key, call.label, compute, whole, anchor, series_of, reference, required=required
    )
    return outcome, adapter


def register(checks: Any, func: Any, inputs: tuple, params: dict) -> Any:
    """Register an indicator (see the module docstring): a view, or a tuple of views for
    a function with several outputs. Raises ConfigError for a function outside
    openalgo.ta, an indicator that looks ahead, or a call that fails on its series for
    any reason but its length."""
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
    reference = call.reference(record)
    anchor = _anchor(record, func, inputs, params)
    found = checks.warm_up(label, anchor, reference)
    required = required_bars(found, _bound(func, inputs, params))
    if required is not None and span < required:
        several, outputs = _short(call, span, required)
    else:
        required = None
        returned = _computed(call, span)
        several, outputs = isinstance(returned, tuple | list), outputs_of(returned, span, label)
    names, causal = _causal_names(record, outputs)
    outcome, adapter = _checked(checks, call, record, outputs, (anchor, reference, required))
    on_bars = None if reference is None else call.on_bars()
    views = [
        LookaheadOutput(name, n, lookahead[n], causal)
        if n in lookahead
        else TimelineView(
            store,
            padded(values, length - span),
            label if len(outputs) == 1 else f"{label}.{n}",
            None if on_bars is None else _output_recipe(on_bars, k),
            outcome.warm_up,
        )
        for k, (n, values) in enumerate(zip(names, outputs, strict=True))
    ]
    if adapter is not None:
        views = adapter.adopt(views)
    return views[0] if len(views) == 1 and not several else tuple(views)


__all__ = ["LookaheadOutput", "register"]
