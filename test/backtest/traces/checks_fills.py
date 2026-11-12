"""Consistency rules for the expected fills of a case. One function per rule.

Every rule takes a case and returns its violations; each violation names the fixture,
the slice and the rule that is broken. None of the rules follows the fixture's
derivation: they use only the bars, the fills and the stated end state.
"""

from __future__ import annotations

from decimal import Decimal

from . import ledger
from . import vocabulary as vocab
from .bars import bars_by_slice, is_on_tick, rupees, tick_size_of
from .timegrid import Case, parse_ist

Violations = list[str]

# Steps of the sequence in which a fill may happen, by where in the bar it is priced
# (03-architecture 4.2: step 2 fills at the open, step 3 at the open or at a level,
# step 6 at the last close when the session has no later bar).
STEPS_BY_FILL_POINT = {
    "open": (vocab.STEP_FILLS_AT_OPEN, vocab.STEP_LEVELS),
    "level": (vocab.STEP_LEVELS,),
    "close": (vocab.STEP_AGGREGATE_RULES,),
}


def fills_are_in_sequence_order(case: Case) -> Violations:
    """Rule: fills are listed in slice order and, inside a slice, in step order."""
    problems = []
    fills = case["expected"]["fills"]
    keys = [(parse_ist(fill["slice_ist"]), fill["step"]) for fill in fills]
    for earlier, later, fill in zip(keys, keys[1:], fills[1:], strict=False):
        if later < earlier:
            problems.append(
                f"{case['id']} {fill['slice_ist']}: fill in step {fill['step']} is listed "
                "after a later slice or step"
            )
    return problems


def fill_prices_lie_inside_their_bar(case: Case) -> Violations:
    """Rule: a fill is priced inside its bar's range; an open fill at the open, a close fill
    at the close."""
    problems = []
    bars = bars_by_slice(case)
    for fill in case["expected"]["fills"]:
        where = f"{case['id']} {fill['slice_ist']}: {fill['reason']} fill"
        bar = bars.get(fill["instrument"], {}).get(fill["slice_ist"])
        if bar is None:
            problems.append(f"{where} is in a slice where {fill['instrument']} has no bar")
            continue
        price = rupees(fill["price_rupees"])
        if not bar.low <= price <= bar.high:
            problems.append(f"{where} at {price} is outside the bar range {bar.low} to {bar.high}")
        if fill["fill_at"] == "open" and price != bar.open:
            problems.append(f"{where} is an open fill at {price} but the bar opens at {bar.open}")
        if fill["fill_at"] == "close" and price != bar.close:
            problems.append(f"{where} is a close fill at {price} but the bar closes at {bar.close}")
    return problems


def fill_points_agree_with_steps(case: Case) -> Violations:
    """Rule: step 2 fills at the open, step 3 at the open or a level, step 6 at the last
    close and flagged."""
    problems = []
    for fill in case["expected"]["fills"]:
        where = f"{case['id']} {fill['slice_ist']}: {fill['reason']} fill"
        if fill["step"] not in STEPS_BY_FILL_POINT[fill["fill_at"]]:
            problems.append(
                f"{where} at the {fill['fill_at']} cannot happen in step {fill['step']}"
            )
        flagged = "filled_at_last_close" in fill["flags"]
        if flagged != (fill["fill_at"] == "close"):
            problems.append(f"{where}: the flag filled_at_last_close marks exactly the close fills")
    return problems


def fill_prices_are_on_the_tick_grid(case: Case) -> Violations:
    """Rule: every fill price is a whole number of ticks of its instrument."""
    problems = []
    for fill in case["expected"]["fills"]:
        tick_size = tick_size_of(case, fill["instrument"])
        price = rupees(fill["price_rupees"])
        if not is_on_tick(price, tick_size):
            problems.append(
                f"{case['id']} {fill['slice_ist']}: fill price {price} is not a multiple "
                f"of the tick {tick_size}"
            )
    return problems


def fill_quantities_are_whole_lots(case: Case) -> Violations:
    """Rule: every fill quantity (in units) is a whole number of lots of its instrument."""
    lot_sizes = {item["symbol"]: item["lot_size_units"] for item in case["instruments"]}
    problems = []
    for fill in case["expected"]["fills"]:
        lot_size = lot_sizes[fill["instrument"]]
        if fill["quantity_units"] % lot_size:
            problems.append(
                f"{case['id']} {fill['slice_ist']}: fill quantity {fill['quantity_units']} "
                f"units is not a multiple of the lot size {lot_size}"
            )
    return problems


def fills_net_to_the_final_positions(case: Case) -> Violations:
    """Rule: the fills, netted at average cost, give the stated final positions."""
    problems = []
    replayed = ledger.final_positions(case)
    for stated in case["expected"]["final_positions"]:
        position = replayed[stated["instrument"]]
        stated_average = stated["average_price_rupees"]
        average = None if stated_average is None else rupees(stated_average)
        if position.quantity_units != stated["quantity_units"] or position.average_price != average:
            problems.append(
                f"{case['id']} end of run: {stated['instrument']} fills net to "
                f"{position.quantity_units} units at {position.average_price}, the fixture "
                f"states {stated['quantity_units']} at {stated_average}"
            )
    return problems


def gross_pnl_matches_the_fills(case: Case) -> Violations:
    """Rule: realized, unrealized and total gross P&L recomputed from the fills equal the
    stated figures."""
    stated = case["expected"]["gross_pnl"]
    realized = ledger.realized_pnl(case)
    unrealized = ledger.unrealized_pnl(case)
    recomputed: dict[str, Decimal] = {
        "realized_rupees": realized,
        "unrealized_rupees": unrealized,
        "total_rupees": realized + unrealized,
    }
    return [
        f"{case['id']} end of run: {key} recomputed from the fills is {value}, "
        f"the fixture states {stated[key]}"
        for key, value in recomputed.items()
        if value != rupees(stated[key])
    ]
