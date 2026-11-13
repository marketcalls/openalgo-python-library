"""How the warm-up of a `self.I` indicator is measured on the engine's reference bars
(decision D25; the bars are in `_warm_up_reference`, the rule that uses the warm-up in
`_warm_up`).

**The window-stability rule** (`settle_length`): the value an indicator gives at a bar
when computed on a rolling window of bars ending there is compared with the value it
gives there when computed on all the reference bars; the window doubles from
`FIRST_WINDOW` bars until every compared bar agrees within `WINDOW_TOLERANCE`, relative
to the larger value or, when that is larger, to a thousandth of the output's median
magnitude (`near_zero`: an oscillator at 1e-13 against 0 is not a difference of
warm-up). The
bars compared at each window are `WINDOW_POINTS` spread over the bars the window can end
on (a path-dependent indicator, such as a parabolic stop-and-reverse, differs on a few
bars only), and, for an output of booleans, up to `EVENT_POINTS` of the bars where it is
true over all the bars: a condition is false on most bars, so a window too short to see
an event shows only where the event is.

**An indicator of other views** (`ReferenceCall`): its inputs may have a warm-up of their
own (`self.close.rolling_max(100).shift(1)` is known from its 101st bar, another
indicator from its warm-up). The indicator needs at least that many bars before its own
warm-up begins, so its warm-up is the larger of the window measured on the whole
computation and the inputs' warm-up plus the indicator's own, measured on the inputs as
computed over all the reference bars (less one, the bar they share).

**No window settles it**: when a window was too short to compute the indicator at all,
or its difference from the whole-series values shrinks by at least `DECAY_FACTOR` at
each of the last `DECAY_WINDOWS` doublings, the indicator needs more bars than the
longest window (a very long average); otherwise it depends on all its history, as a
running sum does, and no history makes it ready. A computation that fails on all the
reference bars has no warm-up either, and says why.

Each warm-up is measured once per indicator label; `ReferenceNeeds` keeps them, and a
sweep hands them to its worker processes. Nothing here is random: the bars and the
compared points are fixed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._warm_up_reference import REFERENCE_BARS, reference_columns

# Default relative tolerance of the window-stability rule (03-architecture 5).
WINDOW_TOLERANCE = 1e-6
# The first window tried, and the bars compared at each window (module docstring).
FIRST_WINDOW = 16
WINDOW_POINTS = 1024
EVENT_POINTS = 64
# Values are compared relative to at least this fraction of the output's median
# magnitude (module docstring).
NEAR_ZERO_FRACTION = 1e-3
# A difference that shrinks by this factor at each of this many last doublings belongs to
# a long lookback, not to a running sum (module docstring).
DECAY_FACTOR = 0.5
DECAY_WINDOWS = 2

# compute(start, stop): the indicator's outputs on the reference bars [start, stop).
Compute = Callable[[int, int], list[np.ndarray]]
# The causal outputs of an indicator computed on any bars: it takes the bar fields as a
# view's recipe reads them (`_views.Recipe`).
Outputs = Callable[[Mapping[str, Any]], list[np.ndarray]]

# Why an indicator has no warm-up length (`WarmUp.note`).
NEVER_SETTLES = (
    "does not settle on the engine's reference bars (it depends on its whole history, as a "
    "running sum does)"
)
OWN_VALUES = (
    "reads values of your own (an array, or a view the reference bars cannot "
    "reproduce), so its warm-up cannot be measured"
)
FAILED = (
    "could not be computed on the engine's reference bars ({error}), so its warm-up "
    "cannot be measured"
)


@dataclass(frozen=True)
class WarmUp:
    """An indicator's warm-up: `bars` of its own series before it is ready (None when it
    has no length, and `note` says why); `more` is True when it needs more than `bars`,
    the longest window measured."""

    bars: int | None
    note: str = ""
    more: bool = False


@dataclass(frozen=True)
class ReferenceCall:
    """An indicator call as the reference bars measure it (module docstring).

    outputs: its causal outputs on any bars, inputs computed from the bars by their
    recipes. input_need: the largest warm-up of its series inputs in bars (1 for bar
    fields). fixed: given all the reference bars, the outputs on bars [start, stop) of
    the inputs as computed over all of them; None when `input_need` is 1."""

    outputs: Outputs
    input_need: int = 1
    fixed: Callable[[Mapping[str, Any]], Compute] | None = None


@dataclass(frozen=True)
class Settling:
    """What the window-stability rule found: the window that settles (None when none
    does), the longest window tried, and with no window, whether the difference was
    still shrinking (module docstring)."""

    bars: int | None
    longest: int
    shrinking: bool = False


def _difference(part: Any, whole: Any, floor: float) -> float:
    """The difference of two values relative to the larger magnitude, or to `floor`
    when that is larger: 0 for equal values or both NaN, infinite for a NaN against a
    number or differing booleans."""
    if isinstance(part, np.bool_ | bool) or isinstance(whole, np.bool_ | bool):
        return 0.0 if bool(part) == bool(whole) else np.inf
    left, right = float(part), float(whole)
    if np.isnan(left) or np.isnan(right):
        return 0.0 if np.isnan(left) and np.isnan(right) else np.inf
    scale = max(abs(left), abs(right), floor)
    return 0.0 if scale == 0.0 else abs(left - right) / scale


def near_zero(values: np.ndarray) -> float:
    """The magnitude a value near zero of an output is compared on: `NEAR_ZERO_FRACTION`
    of the median magnitude of its numbers (0 for booleans or none), so that an
    oscillator at 1e-13 against 0 is not taken for a difference of its warm-up."""
    if values.dtype == np.bool_:
        return 0.0
    numbers = np.abs(values[~np.isnan(values)])
    return NEAR_ZERO_FRACTION * float(np.median(numbers)) if len(numbers) else 0.0


def _spread(first: int, last: int, count: int) -> list[int]:
    """`count` bars spread from `first` to `last`, both included."""
    if last < first:
        return []
    steps = max(count - 1, 1)
    return sorted({first + (last - first) * k // steps for k in range(count)})


def _points(whole: Sequence[np.ndarray], window: int, length: int) -> list[int]:
    """The bars compared for `window` (module docstring)."""
    points = set(_spread(window - 1, length - 1, WINDOW_POINTS))
    for values in whole:
        if values.dtype == np.bool_:
            events = np.flatnonzero(values[window - 1 :]) + window - 1
            chosen = _spread(0, len(events) - 1, EVENT_POINTS)
            points.update(int(events[k]) for k in chosen)
    return sorted(points)


def _window_difference(
    compute: Compute,
    whole: Sequence[np.ndarray],
    window: int,
    length: int,
    stop_above: float | None,
) -> float | None:
    """The largest relative difference at the bars compared for `window`, or the first
    one above `stop_above` when that is given (the window has failed, and only the last
    windows' sizes matter); None when the indicator cannot be computed on so few bars
    (the ValueError of a period longer than the window, which is what a too-short window
    means here)."""
    floors = [near_zero(values) for values in whole]
    largest = 0.0
    for bar in _points(whole, window, length):
        try:
            parts = compute(bar - window + 1, bar + 1)
        except ValueError:
            return None
        for part, values, floor in zip(parts, whole, floors, strict=True):
            largest = max(largest, _difference(part[-1], values[bar], floor))
        if stop_above is not None and largest > stop_above:
            return largest
    return largest


def _shrinking(differences: list[float | None]) -> bool:
    """True when the windows show a long lookback (module docstring): a window too short
    to compute the indicator at all, or a difference shrinking at the last doublings."""
    if any(difference is None for difference in differences):
        return True
    last = differences[-(DECAY_WINDOWS + 1) :]
    if len(last) <= DECAY_WINDOWS or any(d is None or not np.isfinite(d) for d in last):
        return False
    pairs = zip(last[:-1], last[1:], strict=True)
    return all(later <= DECAY_FACTOR * earlier for earlier, later in pairs)


def settle_length(
    compute: Compute, whole: Sequence[np.ndarray], length: int, tolerance: float
) -> Settling:
    """The window-stability rule (module docstring) over windows from `FIRST_WINDOW`
    bars, doubling, shorter than `length`; `whole` are the outputs on all `length` bars
    and `compute(start, stop)` gives them on bars [start, stop)."""
    differences: list[float | None] = []
    window = FIRST_WINDOW
    while window < length:
        last = window << (DECAY_WINDOWS + 1) >= length
        stop_above = None if last else tolerance
        difference = _window_difference(compute, whole, window, length, stop_above)
        if difference is not None and difference <= tolerance:
            return Settling(window, window)
        differences.append(difference)
        window *= 2
    return Settling(None, window // 2, _shrinking(differences))


def _part(argument: Any, start: int, stop: int) -> Any:
    """Bars [start, stop) of an argument computed over all the bars (a constant as it is)."""
    return argument.iloc[start:stop] if isinstance(argument, pd.Series) else argument


def fixed_inputs(
    arguments: Callable[[Mapping[str, Any]], list[Any]],
    apply: Callable[[list[Any]], list[np.ndarray]],
) -> Callable[[Mapping[str, Any]], Compute]:
    """`ReferenceCall.fixed` of an indicator whose `arguments` on any bars `apply`
    turns into its causal outputs."""

    def on(bars: Mapping[str, Any]) -> Compute:
        whole = arguments(bars)
        return lambda start, stop: apply([_part(value, start, stop) for value in whole])

    return on


class ReferenceNeeds:
    """Warm-ups measured on the reference bars, each once per indicator label (a run's
    or a sweep's; the bars are built on first use). `tolerance` is the window tolerance,
    relative; `measured` are warm-ups measured elsewhere (a sweep's), by label."""

    def __init__(
        self, tolerance: float = WINDOW_TOLERANCE, measured: Mapping[str, WarmUp] | None = None
    ) -> None:
        self.tolerance = tolerance
        self._columns: dict[str, np.ndarray] | None = None
        self._times: pd.DatetimeIndex | None = None
        self._found: dict[str, WarmUp] = dict(measured or {})

    def columns(self) -> dict[str, np.ndarray]:
        """The reference bars as store columns (`_warm_up_reference.reference_columns`)."""
        if self._columns is None:
            self._columns = reference_columns()
        return self._columns

    def bars(self, start: int, stop: int) -> dict[str, Any]:
        """Bars [start, stop) of the reference as a recipe reads them (`_views.Recipe`):
        the price and volume fields, `ts_open`, `ts_close` and "times", their IST opens."""
        columns = self.columns()
        if self._times is None:
            utc = pd.to_datetime(columns["ts_open"], utc=True)
            self._times = pd.DatetimeIndex(utc.tz_convert("Asia/Kolkata"), name="time")
        window: dict[str, Any] = {name: columns[name][start:stop] for name in columns}
        window["times"] = self._times[start:stop]
        return window

    def measured(self) -> dict[str, WarmUp]:
        """Every warm-up measured so far, by label (for a sweep's worker processes)."""
        return dict(self._found)

    def need(self, label: str, call: ReferenceCall) -> WarmUp:
        """The warm-up of the indicator labelled `label` (module docstring)."""
        if label not in self._found:
            self._found[label] = self._measure(call)
        return self._found[label]

    def _verdict(self, settling: Settling) -> WarmUp:
        if settling.bars is not None:
            return WarmUp(settling.bars)
        if settling.shrinking:
            return WarmUp(settling.longest, more=True)
        return WarmUp(None, NEVER_SETTLES)

    def _measure(self, call: ReferenceCall) -> WarmUp:
        everything = self.bars(0, REFERENCE_BARS)
        try:
            whole = call.outputs(everything)
        except ValueError as exc:
            return WarmUp(None, FAILED.format(error=exc))

        def windowed(start: int, stop: int) -> list[np.ndarray]:
            return call.outputs(self.bars(start, stop))

        found = self._verdict(settle_length(windowed, whole, REFERENCE_BARS, self.tolerance))
        if found.bars is None or found.more or call.fixed is None or call.input_need <= 1:
            return found
        own = self._verdict(
            settle_length(call.fixed(everything), whole, REFERENCE_BARS, self.tolerance)
        )
        if own.bars is None or own.more:
            return own
        return WarmUp(max(found.bars, call.input_need + own.bars - 1))


__all__ = [
    "EVENT_POINTS",
    "NEAR_ZERO_FRACTION",
    "FAILED",
    "FIRST_WINDOW",
    "NEVER_SETTLES",
    "OWN_VALUES",
    "WINDOW_POINTS",
    "WINDOW_TOLERANCE",
    "Compute",
    "Outputs",
    "ReferenceCall",
    "ReferenceNeeds",
    "Settling",
    "WarmUp",
    "fixed_inputs",
    "settle_length",
    "near_zero",
]
