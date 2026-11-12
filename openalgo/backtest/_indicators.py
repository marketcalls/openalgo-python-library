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
through the truncation and window-stability checks of `_causality_checks`.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from openalgo.indicators import causality

from ._views import TimelineView
from .errors import ConfigError

# The registry's word for an indicator replayed from an anchor, not a rolling window.
ANCHORED_NOTE = "Anchored"


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
        raise NotImplementedError(
            "indicators over series of several symbols or intervals are not implemented yet "
            "(Phase 1 work in progress)"
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
            raise ConfigError(f"{label} returned {array.shape} values for {length} bars")
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
        raise ConfigError(f"{label} failed on the {length} bars of the run: {exc}") from exc
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
    checks.check(key, label, compute, whole, anchored)
    views = [
        LookaheadOutput(name, n, lookahead[n], causal)
        if n in lookahead
        else TimelineView(store, values, label if len(outputs) == 1 else f"{label}.{n}")
        for n, values in zip(names, outputs, strict=True)
    ]
    if adapter is not None:
        views = adapter.adopt(views)
    return views[0] if len(views) == 1 and not isinstance(returned, tuple | list) else tuple(views)


__all__ = ["LookaheadOutput", "register"]
