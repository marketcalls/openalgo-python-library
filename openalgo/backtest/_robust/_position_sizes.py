"""How large a run's positions grew, by symbol (design 17, section 2.14): what the
delayed-orders check compares to tell an entry made late from an entry made twice.

Under a delay (`bt.backtest(delay_entries=)`) the strategy keeps deciding at every strategy
bar while its decided entry is held, and a held entry is a working order (in
`self.orders`), not a position. A strategy that decides its entries by its position alone
(`self.position.is_flat`) sees none while its entry is held and enters again, so the late
run holds more than the run ever did, and a check on it would judge a bigger position, not
a late entry. A condition strategy counts a held entry as pending (SPEC-29) and never
does; nor does a strategy that also checks `self.orders`. On time a market entry fills at
the next open, before the strategy decides again, so it does not happen there.

`PositionSize` of one symbol of a run: the fills that added to an open position (`adds`,
the fills table's reason "add") and the largest position it held after any fill, in units
of one product (`largest_units`). The position after each fill is the running sum of the
fills, so a position that grew and shrank inside one bar counts at its largest, which the
positions table, a row at a bar's end, does not show.

A rerun grew a symbol's position (`grown`) when it added to it more often than the run and
held more units of it than the run's largest: both, because a late entry sized from the
equity can buy a few more units than the run did without adding, and a strategy that adds
can add once more at other prices and still hold no more than the run's largest.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from . import _words as words

#: The fills table's reason of a fill that added to an open position.
ADD = "add"
#: The fills table's side of a buy; every other fill is a sell.
BUY = "BUY"
#: What a strategy sees of a held entry (module docstring).
WORKING_ORDER = "A held entry is a working order (self.orders), not a position."


@dataclass(frozen=True)
class PositionSize:
    """How large one symbol's positions grew in a run (module docstring): `instrument` its
    index among the run's symbols, `adds` the fills that added to an open position,
    `largest_units` the largest position held after any fill, units (never lots)."""

    instrument: int
    adds: int
    largest_units: float


#: The size of a symbol a run never traded.
NEVER_TRADED = PositionSize(-1, 0, 0.0)


def position_sizes(result: Any) -> tuple[PositionSize, ...]:
    """The `PositionSize` of each symbol `result` traded, in instrument order."""
    fills = result._output["fills"]
    held: dict[tuple[int, str], float] = {}
    adds: dict[int, int] = {}
    largest: dict[int, float] = {}
    rows = zip(
        fills["instrument"],
        fills["product"],
        fills["side"],
        fills["quantity_units"],
        fills["reason"],
        strict=True,
    )
    for instrument, product, side, units, reason in rows:
        key = (instrument, product)
        held[key] = held.get(key, 0.0) + (units if side == BUY else -units)
        largest[instrument] = max(largest.get(instrument, 0.0), abs(held[key]))
        adds[instrument] = adds.get(instrument, 0) + (reason == ADD)
    return tuple(PositionSize(k, adds[k], largest[k]) for k in sorted(largest))


def grown(sizes: Sequence[PositionSize], run: Sequence[PositionSize]) -> tuple[PositionSize, ...]:
    """The sizes among `sizes` (a rerun's) of the symbols whose position the rerun grew
    beyond the run's `run` (module docstring), in their order."""
    before = _by_symbol(run)
    return tuple(
        size
        for size in sizes
        if size.adds > before.get(size.instrument, NEVER_TRADED).adds
        and size.largest_units > before.get(size.instrument, NEVER_TRADED).largest_units
    )


def entered_again(
    grew: Sequence[PositionSize], run: Sequence[PositionSize], symbols: Sequence[str]
) -> str:
    """Why the run with entries one bar late cannot judge a late entry: what it did to the
    symbols it grew (`grew`, by `grown`) against the run's sizes `run`, each named from
    `symbols` (the run's, by instrument), and what to do."""
    before = _by_symbol(run)
    parts = "; ".join(
        _grew_words(size, before.get(size.instrument, NEVER_TRADED), symbols[size.instrument])
        for size in grew
    )
    return (
        f"With entries one bar late the strategy entered again while its entry was held: "
        f"{parts}. {WORKING_ORDER} A strategy that decides its entries by its position alone "
        "(self.position.is_flat) sees none while its entry is held and enters twice, so "
        "this run would judge a bigger position, not a late entry. To judge it, decide "
        "entries by the working orders too (self.position.is_flat and not self.orders), "
        "or as conditions."
    )


def _grew_words(size: PositionSize, before: PositionSize, symbol: str) -> str:
    """How a rerun grew `symbol`'s position to `size` from the run's `before`."""
    return (
        f"it added to {symbol}'s position {words.whole(size.adds, 'time')} where the run "
        f"added {words.whole(before.adds, 'time')}, and held up to "
        f"{_units(size.largest_units)} units where the run held at most "
        f"{_units(before.largest_units)}"
    )


def _by_symbol(sizes: Sequence[PositionSize]) -> dict[int, PositionSize]:
    return {size.instrument: size for size in sizes}


def _units(value: float) -> str:
    """Units as a whole number when they are one, else to two places."""
    return words.number(value, 0 if float(value).is_integer() else 2)


__all__ = [
    "NEVER_TRADED",
    "WORKING_ORDER",
    "PositionSize",
    "entered_again",
    "grown",
    "position_sizes",
]
