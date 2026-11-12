"""Consistency rules for the bars of a case. One function per rule.

Every rule takes a case and returns its violations; each violation names the fixture,
the slice and the rule that is broken.
"""

from __future__ import annotations

from .bars import bars_by_slice, is_on_tick, tick_size_of
from .timegrid import Case, exchange_of, grid_slices, parse_ist

Violations = list[str]


def bars_are_valid_ohlc(case: Case) -> Violations:
    """Rule: prices are positive, high is the highest and low the lowest of the four."""
    problems = []
    for instrument, bars in bars_by_slice(case).items():
        for slice_ist, bar in bars.items():
            if min(bar.open, bar.high, bar.low, bar.close) <= 0:
                problems.append(f"{case['id']} {slice_ist}: {instrument} has a non-positive price")
            if bar.high < max(bar.open, bar.close) or bar.low > min(bar.open, bar.close):
                problems.append(
                    f"{case['id']} {slice_ist}: {instrument} open or close lies outside "
                    f"the range {bar.low} to {bar.high}"
                )
    return problems


def bars_are_in_time_order(case: Case) -> Violations:
    """Rule: bar rows are strictly increasing in time (sorted, no duplicates)."""
    problems = []
    for instrument, rows in case["bars"]["rows"].items():
        times = [parse_ist(row[0]) for row in rows]
        for earlier, later, row in zip(times, times[1:], rows[1:], strict=False):
            if later <= earlier:
                problems.append(
                    f"{case['id']} {row[0]}: {instrument} bar is not later than the one before it"
                )
    return problems


def bars_are_on_the_declared_grid(case: Case) -> Violations:
    """Rule: every bar opens on a base-interval slot of the run window."""
    problems = []
    for instrument, bars in bars_by_slice(case).items():
        grid = set(grid_slices(case, exchange_of(case, instrument)))
        for slice_ist in bars:
            if slice_ist not in grid:
                problems.append(
                    f"{case['id']} {slice_ist}: {instrument} bar is not on the "
                    f"{case['settings']['base']} grid of the run window"
                )
    return problems


def missing_slices_are_declared(case: Case) -> Violations:
    """Rule: a grid slot without a bar is listed in missing_slices, and nothing else is."""
    declared = {(gap["instrument"], gap["slice_ist"]) for gap in case["missing_slices"]}
    actual = set()
    for instrument, bars in bars_by_slice(case).items():
        for slice_ist in grid_slices(case, exchange_of(case, instrument)):
            if slice_ist not in bars:
                actual.add((instrument, slice_ist))
    problems = [
        f"{case['id']} {slice_ist}: {instrument} has no bar and the gap is not declared"
        for instrument, slice_ist in sorted(actual - declared)
    ]
    problems += [
        f"{case['id']} {slice_ist}: {instrument} is declared missing but has a bar"
        for instrument, slice_ist in sorted(declared - actual)
    ]
    return problems


def bar_prices_are_on_the_tick_grid(case: Case) -> Violations:
    """Rule: every bar price is a whole number of ticks of its instrument."""
    problems = []
    for instrument, bars in bars_by_slice(case).items():
        tick_size = tick_size_of(case, instrument)
        for slice_ist, bar in bars.items():
            for price in (bar.open, bar.high, bar.low, bar.close):
                if not is_on_tick(price, tick_size):
                    problems.append(
                        f"{case['id']} {slice_ist}: {instrument} price {price} is not a "
                        f"multiple of the tick {tick_size}"
                    )
    return problems
