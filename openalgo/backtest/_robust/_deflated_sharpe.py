"""The deflated Sharpe ratio of the best of a sweep (design 17, section 5): the probability
that its true Sharpe ratio is above what the best of that many tries would show by luck
alone (Bailey and Lopez de Prado 2014, "The Deflated Sharpe Ratio: Correcting for
Selection Bias, Backtest Overfitting and Non-Normality", Journal of Portfolio Management
40(5)).

Every figure is per session, on the combinations' session returns (`Performance.
session_returns`, with no risk-free rate, as a run's metrics measure them):

    sharpe      a combination's mean session return over its sample standard deviation
                (divisor n - 1); undefined when its returns never vary
    variance    the sample variance (divisor n - 1) of the defined Sharpe ratios of the
                combinations tried
    expected    the Sharpe ratio the best of N tries would show with no skill:
                sqrt(variance) ((1 - g) Z(1 - 1/N) + g Z(1 - 1/(N e))), with Z the
                inverse of the standard normal distribution function, e Euler's number
                and g the Euler-Mascheroni constant; N counts every combination tried
    deflated    Phi((S - expected) sqrt(n - 1) / sqrt(1 - g1 S + (g2 + 2) / 4 S^2)) for
                the best's Sharpe ratio S over its n session returns, with g1 and g2 the
                moment ratios of those returns (skewness m3 / m2^1.5 and excess kurtosis
                m4 / m2^2 - 3, population moments): the probabilistic Sharpe ratio of the
                kernel's metrics (`oa_bt::metrics::adjusted`) against `expected` in place
                of zero, a probability

The deflated ratio is NaN with fewer than two combinations (nothing was selected), fewer
than two defined Sharpe ratios, a best whose Sharpe ratio is undefined, or a variance
term that is not positive. The normal distribution is the standard library's
(`statistics.NormalDist`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

#: The Euler-Mascheroni constant.
EULER_GAMMA = 0.5772156649015329
STANDARD_NORMAL = NormalDist()


def sharpe(returns: np.ndarray) -> float:
    """The per-session Sharpe ratio of `returns` (module docstring)."""
    values = np.asarray(returns, dtype=float)
    if len(values) < 2 or (values == values[0]).all():
        return math.nan
    mean = math.fsum(values) / len(values)
    deviation = math.sqrt(math.fsum((values - mean) ** 2) / (len(values) - 1))
    return mean / deviation


def moment_ratios(returns: np.ndarray) -> tuple[float, float]:
    """(g1, g2): skewness m3 / m2^1.5 and excess kurtosis m4 / m2^2 - 3 of `returns`,
    population moments; NaN below two values or without variation."""
    values = np.asarray(returns, dtype=float)
    if len(values) < 2 or (values == values[0]).all():
        return math.nan, math.nan
    centred = values - math.fsum(values) / len(values)
    m2, m3, m4 = (math.fsum(centred**k) / len(values) for k in (2, 3, 4))
    return m3 / m2**1.5, m4 / (m2 * m2) - 3.0


def expected_best(variance: float, trials: int) -> float:
    """The Sharpe ratio the best of `trials` tries with Sharpe ratios of `variance`
    shows with no skill (module docstring); NaN below two tries."""
    if trials < 2 or not math.isfinite(variance) or variance < 0:
        return math.nan
    first = STANDARD_NORMAL.inv_cdf(1.0 - 1.0 / trials)
    second = STANDARD_NORMAL.inv_cdf(1.0 - 1.0 / (trials * math.e))
    return math.sqrt(variance) * ((1.0 - EULER_GAMMA) * first + EULER_GAMMA * second)


@dataclass(frozen=True)
class DeflatedSharpe:
    """The deflated Sharpe ratio of a sweep's best and what it rests on (module
    docstring): the best's per-session Sharpe ratio and its sessions, the combinations
    tried, the variance of their Sharpe ratios, the expected best and the deflated
    ratio, a probability."""

    best_sharpe: float
    sessions: int
    trials: int
    variance: float
    expected: float
    deflated: float


def deflated_sharpe(best: np.ndarray, everyone: Sequence[np.ndarray]) -> DeflatedSharpe:
    """The `DeflatedSharpe` of the best combination's session returns `best` among the
    session returns of every combination tried (`everyone`, the best included)."""
    ratios = np.array([sharpe(returns) for returns in everyone], dtype=float)
    known = ratios[np.isfinite(ratios)]
    variance = float(np.var(known, ddof=1)) if len(known) >= 2 else math.nan
    expected = expected_best(variance, len(everyone))
    best_sharpe, count = sharpe(best), len(best)
    g1, g2 = moment_ratios(best)
    spread = 1.0 - g1 * best_sharpe + (g2 + 2.0) / 4.0 * best_sharpe**2
    deflated = math.nan
    if math.isfinite(expected) and math.isfinite(spread) and spread > 0 and count >= 2:
        score = (best_sharpe - expected) * math.sqrt(count - 1) / math.sqrt(spread)
        deflated = STANDARD_NORMAL.cdf(score)
    return DeflatedSharpe(best_sharpe, count, len(everyone), variance, expected, deflated)


__all__ = [
    "EULER_GAMMA",
    "DeflatedSharpe",
    "deflated_sharpe",
    "expected_best",
    "moment_ratios",
    "sharpe",
]
