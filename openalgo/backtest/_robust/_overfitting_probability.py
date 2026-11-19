"""The probability of backtest overfitting of a sweep (design 17, section 5), by
combinatorially symmetric cross-validation (Bailey, Borwein, Lopez de Prado and Zhu 2017,
"The Probability of Backtest Overfitting", Journal of Computational Finance 20(4)).

The sweep's session returns form a matrix, a row per session and a column per
combination (a session a combination has no return for counts 0: its equity did not
move). Its rows are cut in time into `blocks` (16) blocks of near-equal length, block b
being sessions round(b T / S) to round((b + 1) T / S) - 1 of T. Each way of choosing half
of the blocks (C(S, S/2) splits, in lexicographic order of the chosen blocks) is one
split: the chosen blocks are in sample, the others out of sample. In each split:

    1. every combination is measured in sample by its Sharpe ratio, its mean session
       return over its sample standard deviation; an undefined one (returns that never
       vary) is never best;
    2. the combination best in sample (the first of equal ones) is ranked out of sample
       among all of them by the same ratio, from 1 (the worst) to N (the best), ties
       counting half, an undefined ratio below every defined one;
    3. its relative rank w = rank / (N + 1) gives the logit ln(w / (1 - w)).

The probability of backtest overfitting is the share of splits whose logit is below
zero, how often the best in sample ranks below the median out of sample, a split whose
logit is exactly zero (the best at the median, which an odd number of combinations
allows and identical combinations always give) counting half. On returns with no edge
the best in sample is as likely to rank above the median out of sample as below it, so
the probability averages one half for any number of combinations; counting a split at
the median in full would raise it to two thirds with three combinations. A split with
no defined Sharpe ratio in sample is left out of the count.

Fewer blocks are used when the sessions are few: the largest even number up to
`blocks` that leaves `BLOCK_SESSIONS` sessions in each block; below `FEWEST_BLOCKS`, or
with fewer than two combinations, the probability is NaN. Sums run in NumPy's pairwise
order over the blocks' sums, never through a matrix library, so the figure is the same on
every platform.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

#: The blocks of the paper's examples (Bailey et al. 2017).
DEFAULT_BLOCKS = 16
#: The sessions each block must hold: a sample standard deviation needs two.
BLOCK_SESSIONS = 2
#: The fewest blocks that make a split: four give six splits.
FEWEST_BLOCKS = 4
#: The splits measured at once, keeping the arrays of a chunk to a few megabytes.
VALUES_PER_CHUNK = 2_000_000


@dataclass(frozen=True, eq=False)
class Overfitting:
    """The probability of backtest overfitting (module docstring) and what it rests on:
    the blocks and splits used, the splits counted, and each counted split's logit."""

    probability: float
    blocks: int
    splits: int
    logits: np.ndarray


def blocks_for(sessions: int, wanted: int = DEFAULT_BLOCKS) -> int:
    """The blocks a run of `sessions` is cut into (module docstring); 0 for too few."""
    most = min(wanted, sessions // BLOCK_SESSIONS)
    most -= most % 2
    return most if most >= FEWEST_BLOCKS else 0


def _block_sums(matrix: np.ndarray, blocks: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(count, sum, sum of squares) of each block's returns, by combination."""
    sessions = matrix.shape[0]
    edges = [round(b * sessions / blocks) for b in range(blocks + 1)]
    parts = [matrix[edges[b] : edges[b + 1]] for b in range(blocks)]
    counts = np.array([len(part) for part in parts], dtype=float)
    sums = np.stack([part.sum(axis=0) for part in parts])
    squares = np.stack([(part * part).sum(axis=0) for part in parts])
    return counts, sums, squares


def _ratios(counts: np.ndarray, sums: np.ndarray, squares: np.ndarray) -> np.ndarray:
    """The Sharpe ratio of each combination of each split from its summed blocks: (splits,
    combinations), NaN where its returns do not vary."""
    n = counts[:, None]
    mean = sums / n
    variance = (squares - sums * mean) / (n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = mean / np.sqrt(variance)
    return np.where(variance > 0, ratio, np.nan)


def _logits(inside: np.ndarray, outside: np.ndarray) -> np.ndarray:
    """The logit of each split's best in sample, ranked out of sample (module docstring);
    NaN for a split with no defined ratio in sample."""
    splits, combos = inside.shape
    known = np.isfinite(inside).any(axis=1)
    best = np.argmax(np.where(np.isfinite(inside), inside, -np.inf), axis=1)
    rows = np.arange(splits)
    chosen = outside[rows, best]
    filled = np.where(np.isfinite(outside), outside, -np.inf)
    picked = np.where(np.isfinite(chosen), chosen, -np.inf)[:, None]
    below = (filled < picked).sum(axis=1)
    ties = (filled == picked).sum(axis=1) - 1
    rank = 1.0 + below + 0.5 * ties
    share = rank / (combos + 1.0)
    return np.where(known, np.log(share / (1.0 - share)), np.nan)


def overfitting(matrix: np.ndarray, wanted: int = DEFAULT_BLOCKS) -> Overfitting:
    """The `Overfitting` of `matrix`, (sessions, combinations) session returns (module
    docstring)."""
    sessions, combos = matrix.shape
    blocks = blocks_for(sessions, wanted)
    if not blocks or combos < 2:
        return Overfitting(math.nan, blocks, 0, np.empty(0))
    counts, sums, squares = _block_sums(np.asarray(matrix, dtype=float), blocks)
    chosen = np.array(list(itertools.combinations(range(blocks), blocks // 2)))
    rest = np.array([[b for b in range(blocks) if b not in row] for row in chosen])
    size = max(1, VALUES_PER_CHUNK // max(combos * blocks, 1))
    logits = []
    for first in range(0, len(chosen), size):
        inside_blocks, outside_blocks = chosen[first : first + size], rest[first : first + size]
        inside = _ratios(
            counts[inside_blocks].sum(axis=1),
            sums[inside_blocks].sum(axis=1),
            squares[inside_blocks].sum(axis=1),
        )
        outside = _ratios(
            counts[outside_blocks].sum(axis=1),
            sums[outside_blocks].sum(axis=1),
            squares[outside_blocks].sum(axis=1),
        )
        logits.append(_logits(inside, outside))
    every = np.concatenate(logits)
    counted = every[np.isfinite(every)]
    return Overfitting(below_median_share(counted), blocks, len(chosen), counted)


def below_median_share(logits: np.ndarray) -> float:
    """The share of `logits` below zero, one at exactly zero counting half (module
    docstring); NaN for none."""
    if not len(logits):
        return math.nan
    return float(((logits < 0).sum() + 0.5 * (logits == 0).sum()) / len(logits))


__all__ = [
    "BLOCK_SESSIONS",
    "DEFAULT_BLOCKS",
    "FEWEST_BLOCKS",
    "Overfitting",
    "below_median_share",
    "blocks_for",
    "overfitting",
]
