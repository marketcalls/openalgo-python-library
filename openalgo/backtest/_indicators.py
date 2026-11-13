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
inputs' recipes (`_views.Recipe`), which the run waits for (`_warm_up`). An input of
the user's own (a NumPy array, a pandas Series or DataFrame) has no counterpart on the
reference bars, so such an indicator has no warm-up length and is not waited for; the
run says so. The series include the history loaded before `start`, so an indicator is
computed over it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from openalgo.indicators import causality

from ._messages import counted, not_supported
from ._views import Recipe, TimelineView
from .errors import ConfigError

# The registry's word for an indicator replayed from an anchor, not a rolling window.
ANCHORED_NOTE = "Anchored"
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


class _Call:
    """One indicator call: the function, its inputs and parameters, on any part of the
    input series."""

    def __init__(self, func: Any, inputs: tuple, params: dict, store: Any, label: str) -> None:
        self.func, self.inputs, self.params, self.store = func, inputs, params, store
        self.label = label

    def arguments(self, start: int, stop: int) -> list[Any]:
        """The inputs on bars [start, stop): series as pandas Series by IST bar open."""
        times = self.store.times()[start:stop]
        return [
            pd.Series(value._values[start:stop], index=times)
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

    def on_bars(self) -> OnBars | None:
        """Every output computed on any bars from the inputs' recipes (`_views.Recipe`),
        or None when an input holds values of the user's own: an array, or a view
        without a recipe. The outputs of the last bars asked for are kept, so that the
        views of several outputs compute the indicator once."""
        recipes: list[Recipe | None] = []
        for value in self.inputs:
            if isinstance(value, OWN_VALUES):
                return None
            if isinstance(value, TimelineView):
                if value._recipe is None:
                    return None
                recipes.append(value._recipe)
            else:
                recipes.append(None)
        last: list[tuple[Mapping[str, Any], list[np.ndarray]]] = []

        def outputs(bars: Mapping[str, Any]) -> list[np.ndarray]:
            if last and last[0][0] is bars:
                return last[0][1]
            times = bars["times"]
            arguments = [
                value if recipe is None else pd.Series(recipe(bars), index=times)
                for value, recipe in zip(self.inputs, recipes, strict=True)
            ]
            found = _outputs(self.func(*arguments, **self.params), len(times), self.label)
            last[:] = [(bars, found)]
            return found

        return outputs


def _causal_part(on_bars: OnBars, names: list[str], causal: tuple[str, ...]) -> OnBars:
    """The causal outputs of `on_bars`, in order."""

    def outputs(bars: Mapping[str, Any]) -> list[np.ndarray]:
        return [part for n, part in zip(names, on_bars(bars), strict=True) if n in causal]

    return outputs


def _output_recipe(on_bars: OnBars, index: int) -> Recipe:
    """The recipe of output `index` of an indicator."""

    def recipe(bars: Mapping[str, Any]) -> np.ndarray:
        return on_bars(bars)[index]

    return recipe


def register(checks: Any, func: Any, inputs: tuple, params: dict) -> Any:
    """Register an indicator (see the module docstring): a view, or a tuple of views for
    a function with several outputs. Raises ConfigError for a function outside
    openalgo.ta, an indicator that looks ahead, or a call that fails on the run's bars."""
    record = _record(func)
    name = record["name"]
    lookahead = record["non_causal"]
    if set(record["outputs"]) <= set(lookahead):
        reasons = "; ".join(f"{n}: {lookahead[n]}" for n in record["outputs"])
        raise ConfigError(f"ta.{name} cannot be used in a strategy, it looks ahead ({reasons})")
    store = _store_of(inputs)
    label = _label(name, inputs, params)
    call = _Call(func, inputs, params, store, label)
    length = len(store.column("ts_open"))
    try:
        returned = func(*call.arguments(0, length), **params)
    except ValueError as exc:
        bars = counted(length, "bar")
        raise ConfigError(f"{label} failed on the run's {bars}: {exc}") from exc
    outputs = _outputs(returned, length, label)
    names = [*record["outputs"], *(f"output_{k}" for k in range(len(outputs)))][: len(outputs)]
    causal = tuple(n for n in names if n not in lookahead)
    if not causal:
        reasons = "; ".join(f"{n}: {lookahead[n]}" for n in names)
        raise ConfigError(f"ta.{name} cannot be used in a strategy, it looks ahead ({reasons})")
    whole = [values for n, values in zip(names, outputs, strict=True) if n in causal]
    anchored = ANCHORED_NOTE in (record["notes"] or "")
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
    on_bars = call.on_bars()
    reference = None if on_bars is None else _causal_part(on_bars, names, causal)
    series_of = (store.symbol, store.interval)
    checks.check(key, label, compute, whole, anchored, series=series_of, reference=reference)
    views = [
        LookaheadOutput(name, n, lookahead[n], causal)
        if n in lookahead
        else TimelineView(
            store,
            values,
            label if len(outputs) == 1 else f"{label}.{n}",
            None if on_bars is None else _output_recipe(on_bars, k),
        )
        for k, (n, values) in enumerate(zip(names, outputs, strict=True))
    ]
    if adapter is not None:
        views = adapter.adopt(views)
    return views[0] if len(views) == 1 and not isinstance(returned, tuple | list) else tuple(views)


__all__ = ["LookaheadOutput", "register"]
