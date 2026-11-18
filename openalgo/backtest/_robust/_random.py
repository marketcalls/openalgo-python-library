"""Seeded random draws of the robustness checks: the same numbers for a seed on every
platform.

The stream is SplitMix64 (Steele, Lea and Flood 2014, "Fast splittable pseudorandom
number generators", OOPSLA), the generator of the kernel's Monte Carlo
(`oa_bt::metrics::rng`), read as a counter: its j-th 64-bit output (counting from 0) is
the mixing function applied to seed + (j + 1) x `GOLDEN_GAMMA` modulo 2^64, so any
stretch of the stream is computed at once with NumPy's wrapping unsigned arithmetic.

A whole number below a bound is the high word of the 128-bit product of an output and
the bound (Lemire 2019, "Fast random integer generation in an interval", ACM TOMACS
29(1)); a fraction is the top 53 bits of an output over 2^53. Both use integer
arithmetic and exactly representable doubles only, so no platform rounds differently.
The kernel's integer draw also rejects the rare outputs that make some results likelier;
that happens with a probability below bound / 2^64 per draw (under 1e-12 for any bound
under ten million), and here no output is rejected, so a draw can differ from the
kernel's only in that case.

Each check reads its own stream, seeded with the report's seed plus the check's number
(`check_seed`), so one check's draws never depend on which other checks run. The draws
of run r of a check are positions r x width to (r + 1) x width - 1 of its stream, so its
first 200 runs are the same whether 200 or 1,000 are asked for.
"""

from __future__ import annotations

import numpy as np

#: The counter increment and the two multipliers of the mixing function (Steele, Lea
#: and Flood 2014); the same constants as the kernel's generator.
GOLDEN_GAMMA = np.uint64(0x9E37_79B9_7F4A_7C15)
MIX_1 = np.uint64(0xBF58_476D_1CE4_E5B9)
MIX_2 = np.uint64(0x94D0_49BB_1331_11EB)
#: The largest seed a report takes: a check adds its number (at most 17) and the sum
#: must stay a 64-bit unsigned seed; a signed 64-bit bound keeps it one in NumPy too.
MAX_SEED = 2**63 - 1
#: The largest bound of a whole-number draw: the multiply-high below splits an output
#: into two 32-bit halves, so the bound must fit in 32 bits.
MAX_BOUND = 2**32 - 1
_LOW_32 = np.uint64(0xFFFF_FFFF)
_HALF = np.uint64(32)
#: The three shifts of the mixing function, and the shift that keeps an output's top 53
#: bits.
_MIX_SHIFTS = (np.uint64(30), np.uint64(27), np.uint64(31))
_TOP_53 = np.uint64(11)
#: 2^-53, the weight of the lowest of the 53 bits a fraction keeps.
_FRACTION_UNIT = 2.0**-53


def check_seed(seed: int, number: int) -> int:
    """The seed of check `number`'s stream: the report's `seed` plus that number."""
    return seed + number


def part_seed(seed: int, part: int) -> int:
    """The seed of part `part` of a check that draws for several things (random entries
    and their bars, two generators of candidates): output `part` of the check's stream,
    so each part reads a stream of its own, the same on every platform."""
    return int(outputs(seed, part, 1)[0])


def outputs(seed: int, start: int, count: int) -> np.ndarray:
    """Outputs `start` to `start + count - 1` of the stream seeded with `seed`, as
    unsigned 64-bit integers."""
    positions = np.arange(start + 1, start + count + 1, dtype=np.uint64)
    z = np.uint64(seed) + positions * GOLDEN_GAMMA
    first, second, third = _MIX_SHIFTS
    z = (z ^ (z >> first)) * MIX_1
    z = (z ^ (z >> second)) * MIX_2
    return z ^ (z >> third)


def block(seed: int, first_run: int, runs: int, width: int) -> np.ndarray:
    """The draws of runs `first_run` to `first_run + runs - 1`, `width` outputs each, as
    a (runs, width) array: run r reads positions r x width onwards (module docstring)."""
    return outputs(seed, first_run * width, runs * width).reshape(runs, width)


def below(bits: np.ndarray, bound: int) -> np.ndarray:
    """A whole number from 0 to `bound` - 1 for each output in `bits`: the high word of
    output x bound (module docstring). `bound` is from 1 to `MAX_BOUND`."""
    size = np.uint64(bound)
    high, low = bits >> _HALF, bits & _LOW_32
    # high x size and the carry of low x size both fit in 64 bits for a 32-bit size.
    product_high = (high * size + ((low * size) >> _HALF)) >> _HALF
    return product_high.astype(np.int64)


def fractions(bits: np.ndarray) -> np.ndarray:
    """A fraction in [0, 1) for each output in `bits`: its top 53 bits over 2^53."""
    return (bits >> _TOP_53).astype(np.float64) * _FRACTION_UNIT


def orders(seed: int, first_run: int, runs: int, size: int) -> np.ndarray:
    """A random order of `size` positions for each of the runs, (runs, size): the
    positions sorted by one draw each, ties (two equal 64-bit draws) kept in position
    order, so every order is a permutation and every permutation is equally likely but
    for those ties."""
    keys = block(seed, first_run, runs, size)
    return np.argsort(keys, axis=1, kind="stable")


def run_chunks(runs: int, width: int, values_per_chunk: int) -> list[tuple[int, int]]:
    """(first run, runs) pairs that cover `runs` runs of `width` draws each, in order,
    with at most `values_per_chunk` draws per pair (one run at least), so the memory a
    check needs does not grow with the number of runs."""
    size = max(1, values_per_chunk // max(width, 1))
    return [(first, min(size, runs - first)) for first in range(0, runs, size)]


__all__ = [
    "MAX_BOUND",
    "MAX_SEED",
    "below",
    "block",
    "check_seed",
    "fractions",
    "orders",
    "outputs",
    "part_seed",
    "run_chunks",
]
