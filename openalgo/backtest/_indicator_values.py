"""The values of a `self.I` indicator call as the run holds them (`_indicators`).

- `outputs_of`: what a `ta` function returns, as 1-D arrays of the series length (each
  column of a DataFrame, each item of a tuple, or the one value), booleans kept for a
  condition and everything else as floats.
- `padded`: unknown values in front of an indicator computed on part of its series (the
  run's bars of a call over values of the user's own): NaN, or false for a condition.
- **A series too short for its call** (decision D25): `required_bars` is what decides it,
  the call's own window (`period`, `length`), else its warm-up when it has a length, so
  the series length decides and never the error a call raises; `short_outputs` gives
  such a call unknown values of the right kind and number without computing it on the
  series.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from ._messages import counted
from ._warm_up_measure import WarmUp
from .errors import ConfigError

# The arguments a `ta` function names its one window in bars with (`ta.sma`'s `period`,
# `ta.roc`'s `length`), which a series shorter than it cannot compute (`required_bars`).
# A function of several windows (`ta.macd`'s fast, slow and signal periods) has none.
WINDOW_ARGUMENTS = ("period", "length")
# A series too short for its call is shaped on its inputs filled in front to this many
# times the bars it needs (`short_outputs`): enough for a function that checks one bar
# more than its period.
SHAPE_BARS_FACTOR = 2


def outputs_of(returned: Any, length: int, label: str) -> list[np.ndarray]:
    """The function's outputs as 1-D arrays of the series length. Raises ConfigError for
    an output of another shape."""
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


def padded(values: np.ndarray, added: int) -> np.ndarray:
    """`values` after `added` unknown values: false for a condition, else NaN."""
    if not added:
        return values
    if values.dtype == np.bool_:
        return np.concatenate((np.zeros(added, dtype=bool), values))
    return np.concatenate((np.full(added, np.nan), values))


def required_bars(found: WarmUp, arguments: Mapping[str, Any]) -> int | None:
    """The bars a series must hold for a call to be computed: its own window (an
    argument of `WINDOW_ARGUMENTS`, from `arguments` by name, defaults included), or
    without one its warm-up `found`; None when neither is known. The window comes first,
    so a call its series can compute is computed even when the series is shorter than
    its warm-up (it is then never ready, `_warm_up`); the warm-up is never shorter than
    the window."""
    for name in WINDOW_ARGUMENTS:
        window = arguments.get(name)
        if isinstance(window, int) and not isinstance(window, bool):
            return window
    return found.bars


def _filled_in_front(value: Any, added: int, span: int) -> Any:
    """`value` with `added` copies of its first value in front when it is a series of
    the call's `span` bars (a pandas Series or a 1-D array); any other value as it is."""
    if isinstance(value, pd.Series) and len(value) == span:
        values = value.to_numpy()
    elif isinstance(value, np.ndarray) and value.ndim == 1 and len(value) == span:
        values = value
    else:
        return value
    first = values[0] if len(values) else np.ones(1, dtype=values.dtype)[0]
    filled = np.concatenate((np.full(added, first, dtype=values.dtype), values))
    return pd.Series(filled) if isinstance(value, pd.Series) else filled


def short_outputs(
    call: Callable[..., Any],
    arguments: list[Any],
    params: dict[str, Any],
    span: int,
    required: int,
    label: str,
) -> tuple[bool, list[np.ndarray]]:
    """(True when the function returns several outputs, its outputs on a series of
    `span` bars too short for it, `required` bars: all unknown, NaN or false for a
    condition). `arguments` and `params` are the call's on that series. The kind and
    number of outputs are those the function returns on them filled in front to
    `SHAPE_BARS_FACTOR` times `required` bars, whose values are dropped. Raises
    ValueError when that call fails, and ConfigError for an output of another shape."""
    added = SHAPE_BARS_FACTOR * max(required, 1) - span
    filled = [_filled_in_front(value, added, span) for value in arguments]
    named = {name: _filled_in_front(value, added, span) for name, value in params.items()}
    returned = call(*filled, **named)
    shaped = outputs_of(returned, span + added, label)
    unknown = [
        np.zeros(span, dtype=bool) if part.dtype == np.bool_ else np.full(span, np.nan)
        for part in shaped
    ]
    return isinstance(returned, tuple | list), unknown


__all__ = [
    "SHAPE_BARS_FACTOR",
    "WINDOW_ARGUMENTS",
    "outputs_of",
    "padded",
    "required_bars",
    "short_outputs",
]
