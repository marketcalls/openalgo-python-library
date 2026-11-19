"""The parameter-neighbourhood rule (design 17, section 2.16): is a setting on a plateau
or a lone spike? The parameter neighbourhood check, the noise-robust optimization (on its
noisy score) and `bt.optimize` (for its best) apply it.

The figures of a chosen setting among its neighbours (`Grid.neighbours`, `_grid`), each
measured by an objective with a break-even value (0, or 1 for a profit factor,
`_bar_common.BREAK_EVEN`):

    profitable_share   the share of neighbours that made money (net profit of at least
                       half a paisa, `_trade_common.BREAK_EVEN_RUPEES`)
    median, low        the median and the 25th percentile of the neighbours' objectives
                       (linear between order statistics)
    median_ratio, low_ratio
                       each over the chosen objective, both measured from break-even:
                       (x - break-even) / (chosen - break-even)
    percentile         the share of every other combination run below the chosen one

Pass: the 25th percentile neighbour above break-even (by the objective, it made money)
and the median neighbour keeping at least `keep` (half) of the chosen objective. A
neighbour whose objective is undefined is left out of the percentiles and the ratios;
with none defined the rule cannot be applied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import _words as words

#: The percentile of the neighbours that must be above break-even (design 17, 2.16).
LOW_PERCENTILE = 0.25
#: The share of the chosen objective the median neighbour keeps by default.
DEFAULT_KEEP = 0.5


@dataclass(frozen=True)
class Plateau:
    """The figures of the module docstring for one chosen setting."""

    chosen: float
    level: float
    keep: float
    neighbours: int
    measured: int
    profitable_share: float
    median: float
    low: float
    percentile: float

    @property
    def median_ratio(self) -> float:
        """The median neighbour's objective over the chosen one, from break-even."""
        return _ratio(self.median, self.chosen, self.level)

    @property
    def low_ratio(self) -> float:
        """The 25th percentile neighbour's objective over the chosen one, from
        break-even."""
        return _ratio(self.low, self.chosen, self.level)

    @property
    def passed(self) -> bool:
        """Whether the setting is on a plateau by the rule of the module docstring."""
        return bool(self.low > self.level and self.median_ratio >= self.keep)

    def summary(self) -> dict[str, float]:
        """The figures by name, fractions as fractions."""
        return {
            "chosen": self.chosen,
            "neighbours": float(self.neighbours),
            "measured": float(self.measured),
            "profitable_share": self.profitable_share,
            "median": self.median,
            "low": self.low,
            "median_ratio": self.median_ratio,
            "low_ratio": self.low_ratio,
            "percentile": self.percentile,
        }


def _ratio(value: float, chosen: float, level: float) -> float:
    gain = chosen - level
    return (value - level) / gain if gain > 0 and math.isfinite(value) else math.nan


def plateau(
    chosen: float,
    neighbours: np.ndarray,
    profitable: np.ndarray,
    others: np.ndarray,
    level: float,
    keep: float,
) -> Plateau | None:
    """The `Plateau` of a setting whose objective is `chosen` among `neighbours` (their
    objectives) that made money where `profitable`, with `others` the objectives of every
    other combination run; None when no neighbour's objective is defined."""
    values = np.asarray(neighbours, dtype=float)
    known = values[np.isfinite(values)]
    if not len(known):
        return None
    median, low = np.quantile(known, [0.5, LOW_PERCENTILE], method="linear")
    rest = np.asarray(others, dtype=float)
    rest = rest[np.isfinite(rest)]
    below = float((rest < chosen).mean()) if len(rest) else math.nan
    return Plateau(
        chosen=float(chosen),
        level=float(level),
        keep=float(keep),
        neighbours=len(values),
        measured=len(known),
        profitable_share=float(np.mean(np.asarray(profitable, dtype=bool))),
        median=float(median),
        low=float(low),
        percentile=below,
    )


def rule_words(name: str, level: float, keep: float) -> str:
    """The rule, in words, for an objective named `name` with break-even `level`."""
    return (
        f"the 25th percentile neighbour's {name} is above {level:g} and the median "
        f"neighbour keeps {words.percent(keep, 0)} of the chosen {name}"
    )


def figure_words(found: Plateau, name: str) -> str:
    """The figure of a `Plateau`, in words."""
    return (
        f"{words.percent(found.profitable_share)} of {words.whole(found.neighbours, 'neighbour')} "
        f"profitable; the median keeps {words.percent(found.median_ratio)} and the 25th "
        f"percentile {words.percent(found.low_ratio)} of the chosen {name} of "
        f"{words.number(found.chosen)}; the chosen one above "
        f"{words.percent(found.percentile)} of the other settings"
    )


__all__ = [
    "DEFAULT_KEEP",
    "LOW_PERCENTILE",
    "Plateau",
    "figure_words",
    "plateau",
    "rule_words",
]
