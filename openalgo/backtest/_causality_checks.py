"""The run-time causality checks of `self.I` indicators (03-architecture 5).

The registry `openalgo.indicators.causality` says which `ta` outputs look ahead; those
are rejected before anything here runs. These checks verify, on the run's own data,
what the registry claims for the rest:

- **Truncation.** The indicator is computed again on the series cut at sampled points;
  every value up to a cut must equal the value computed on the whole series (within
  `TRUNCATION_AGREEMENT`, relative and absolute, as the library's own causality test).
  An indicator that fails reads later bars and is rejected with a ConfigError.
- **Window stability**, for live trading. A value computed from a rolling window ending
  at a bar is compared with the whole-series value at that bar within a relative
  tolerance (`WINDOW_TOLERANCE`, printed in the Assumptions block); the window doubles
  from `FIRST_WINDOW` bars until every sampled bar agrees. The length found is the
  warm-up a live run needs. Anchored indicators (the registry's notes say so) are
  replayed from their anchor instead, and an indicator no window settles (a cumulative
  series) needs its whole history; both are stated, neither is an error.

They run once per strategy and data: an `IndicatorChecks` remembers what it has checked
by strategy source hash, function, inputs and data digest, never by parameter value, so
the combinations of a sweep share one check. Cut points and sampled bars are fixed by
the series length: nothing is random.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .errors import ConfigError

# Cut points of the truncation check, spread over the second half of the series.
TRUNCATION_CUTS = 4
# Two values agree when they differ by at most this, relative to the larger magnitude
# or absolutely below 1 (the tolerance of test/test_indicators_causality.py).
TRUNCATION_AGREEMENT = 1e-9
# Default relative tolerance of the window-stability check (03-architecture 5).
WINDOW_TOLERANCE = 1e-6
# Bars compared at each window length, and the first window length tried.
WINDOW_POINTS = 4
FIRST_WINDOW = 16

# compute(start, stop): the indicator's outputs on the input bars [start, stop).
Compute = Callable[[int, int], list[np.ndarray]]


@dataclass(frozen=True)
class CheckOutcome:
    """What the checks found for one indicator.

    label: the indicator as named in messages. cuts: truncation cut points checked.
    warm_up: bars of rolling window that reproduce the whole-series values within the
    tolerance, or None. note: why there is no warm-up length, or "".
    """

    label: str
    cuts: int
    warm_up: int | None
    note: str

    def describe(self) -> str:
        """One line for the Assumptions block."""
        settled = f"warm-up {self.warm_up} bars" if self.warm_up is not None else self.note
        return f"{self.label}: truncation passed at {self.cuts} cuts, {settled}"


def _agree(window: np.ndarray, whole: np.ndarray, tolerance: float, floor: float) -> bool:
    """Element-wise agreement: equal NaN, equal booleans, or a difference within
    `tolerance` of the larger magnitude (or of `floor`, whichever is larger)."""
    if window.dtype == np.bool_ or whole.dtype == np.bool_:
        return bool(np.array_equal(window, whole))
    window, whole = window.astype(float), whole.astype(float)
    both_nan = np.isnan(window) & np.isnan(whole)
    scale = np.maximum(np.maximum(np.abs(window), np.abs(whole)), floor)
    with np.errstate(invalid="ignore"):
        close = np.abs(window - whole) <= tolerance * scale
    return bool(np.all(both_nan | close))


def _cuts(length: int) -> list[int]:
    """Cut points in the second half of a series of `length` bars."""
    spacing = 2 * (TRUNCATION_CUTS + 1)
    points = {length // 2 + length * k // spacing for k in range(1, TRUNCATION_CUTS + 1)}
    return sorted(cut for cut in points if 1 <= cut < length)


def _sample(first: int, last: int) -> list[int]:
    """`WINDOW_POINTS` bars spread from `first` to `last`, both included."""
    if last < first:
        return []
    steps = max(WINDOW_POINTS - 1, 1)
    return sorted({first + (last - first) * k // steps for k in range(WINDOW_POINTS)})


class IndicatorChecks:
    """The checks of one run or one sweep. See the module docstring.

    strategy_hash: the strategy's source hash. known: keys checked elsewhere (a sweep's
    first combination), skipped here. tolerance: the window-stability tolerance.
    """

    def __init__(
        self,
        strategy_hash: str,
        known: frozenset[str] = frozenset(),
        tolerance: float = WINDOW_TOLERANCE,
    ) -> None:
        self.strategy_hash = strategy_hash
        self.tolerance = tolerance
        self._known = known
        self._outcomes: dict[str, CheckOutcome] = {}

    @property
    def keys(self) -> frozenset[str]:
        """Every key checked, here or before."""
        return self._known | frozenset(self._outcomes)

    def check(
        self,
        key: str,
        label: str,
        compute: Compute,
        whole: Sequence[np.ndarray],
        anchored: bool,
    ) -> None:
        """Run both checks for an indicator unless `key` was checked; `whole` are its
        causal outputs on the whole series, `compute` recomputes them on a part.

        Raises ConfigError when a value changes once later bars are removed."""
        key = f"{self.strategy_hash}:{key}"
        if key in self._known or key in self._outcomes:
            return
        length = len(whole[0]) if whole else 0
        cuts = self._truncation(label, compute, whole, length)
        if anchored:
            outcome = CheckOutcome(label, cuts, None, "anchored, replayed from its anchor")
        else:
            outcome = CheckOutcome(label, cuts, *self._window(compute, whole, length))
        self._outcomes[key] = outcome

    def _truncation(
        self, label: str, compute: Compute, whole: Sequence[np.ndarray], length: int
    ) -> int:
        checked = 0
        for cut in _cuts(length):
            try:
                parts = compute(0, cut)
            except ValueError:
                continue  # the indicator needs more bars than the cut leaves
            checked += 1
            for part, values in zip(parts, whole, strict=True):
                if not _agree(part, values[:cut], TRUNCATION_AGREEMENT, 1.0):
                    raise ConfigError(
                        f"{label} looks ahead: its values before bar {cut} change when the "
                        "bars after it are removed, so it cannot be used in a strategy"
                    )
        return checked

    def _window(
        self, compute: Compute, whole: Sequence[np.ndarray], length: int
    ) -> tuple[int | None, str]:
        window = FIRST_WINDOW
        while window < length:
            if self._stable(compute, whole, window, length):
                return window, ""
            window *= 2
        return None, "no window settles within the tolerance: needs the whole history"

    def _stable(
        self, compute: Compute, whole: Sequence[np.ndarray], window: int, length: int
    ) -> bool:
        for bar in _sample(window - 1, length - 1):
            try:
                parts = compute(bar - window + 1, bar + 1)
            except ValueError:
                return False
            for part, values in zip(parts, whole, strict=True):
                if not _agree(part[-1:], values[bar : bar + 1], self.tolerance, 0.0):
                    return False
        return True

    def assumption(self) -> str | None:
        """The Assumptions line: the tolerance and each indicator's outcome."""
        if not self._outcomes:
            return None
        lines = "; ".join(outcome.describe() for outcome in self._outcomes.values())
        return f"window tolerance {self.tolerance:g} (relative); {lines}"


__all__ = [
    "FIRST_WINDOW",
    "TRUNCATION_AGREEMENT",
    "TRUNCATION_CUTS",
    "WINDOW_TOLERANCE",
    "CheckOutcome",
    "IndicatorChecks",
]
