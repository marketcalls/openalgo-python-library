"""The values of a `self.I` indicator call as the run holds them (`_indicators`).

- `outputs_of`: what a `ta` function returns, as 1-D arrays of the series length (each
  column of a DataFrame, each item of a tuple, or the one value), booleans kept for a
  condition and everything else as floats.
- `padded`: unknown values in front of an indicator computed on part of its series (the
  run's bars of a call over values of the user's own): NaN, or false for a condition.
- **A series too short for its call** (decision D25): a call that raises ValueError on
  its series is tried once more on the same inputs with bars in front, up to
  `LENGTH_TEST_BARS` (`short_outputs`). When it computes there, more bars cure it, so
  the series was too short for it: its values are unknown, of the kind and number the
  function returns. When it fails there too, the failure is not the length (a parameter
  it refuses, values it cannot take) and the run stops on the original error. Nothing is
  guessed from the names of the call's arguments: a function that checks a length of its
  own, longer than its period (a 20-bar ulcer index checks for 52 bars), is told apart
  from a bad parameter by what it does on a longer series.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ._messages import counted
from ._warm_up_reference import REFERENCE_BARS
from .errors import ConfigError

# The bars a call that failed on its series is tried on again (`short_outputs`): as many
# as the reference bars hold, the longest series a warm-up is measured on, so a call
# with a warm-up computes on them.
LENGTH_TEST_BARS = REFERENCE_BARS


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
    label: str,
) -> tuple[bool, list[np.ndarray]] | None:
    """For a call that raised ValueError on its series of `span` bars (`arguments` and
    `params` are the call's there): (True when the function returns several outputs,
    its outputs on that series, all unknown, NaN or false for a condition) when the
    series was too short for it, else None (module docstring). The call is tried on its
    inputs filled in front to `LENGTH_TEST_BARS` bars, each series input repeating its
    first value. Raises ConfigError for an output of another shape."""
    added = LENGTH_TEST_BARS - span
    if added <= 0:
        return None
    filled = [_filled_in_front(value, added, span) for value in arguments]
    named = {name: _filled_in_front(value, added, span) for name, value in params.items()}
    try:
        returned = call(*filled, **named)
    except ValueError:
        return None
    shaped = outputs_of(returned, span + added, label)
    unknown = [
        np.zeros(span, dtype=bool) if part.dtype == np.bool_ else np.full(span, np.nan)
        for part in shaped
    ]
    return isinstance(returned, tuple | list), unknown


__all__ = ["LENGTH_TEST_BARS", "outputs_of", "padded", "short_outputs"]
