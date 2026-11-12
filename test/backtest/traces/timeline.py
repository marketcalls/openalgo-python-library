"""Read access to the level timeline of a case.

`expected.levels` holds one row per instrument at the end of every slice in which its
position, stop, targets, trail state or breakeven state changed. A value written at the
end of slice t is the value in force from slice t + 1 (03-architecture 4.2 step 4), so
the row "in force" during a slice is the latest row of an earlier slice.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from .bars import rupees
from .timegrid import Case, parse_ist
from .vocabulary import STEP_FILLS_AT_OPEN

LevelRow = dict[str, Any]


def rows_of(case: Case, instrument: str) -> list[LevelRow]:
    """Level rows of one instrument, in the order listed."""
    return [row for row in case["expected"]["levels"] if row["instrument"] == instrument]


def row_in_force(case: Case, instrument: str, slice_ist: str) -> LevelRow | None:
    """The latest row of an earlier slice: the levels a bar of this slice is tested against."""
    moment = parse_ist(slice_ist)
    earlier = [row for row in rows_of(case, instrument) if parse_ist(row["slice_ist"]) < moment]
    return earlier[-1] if earlier else None


def is_long(row: LevelRow) -> bool:
    """True for a long position row. Callers skip flat rows first."""
    return row["position_quantity_units"] > 0


def stop_level(row: LevelRow) -> Decimal | None:
    """Stop level of a row in rupees, or None when the leg has no stop."""
    return None if row["stop"] is None else rupees(row["stop"]["level_rupees"])


def working_targets(row: LevelRow) -> list[dict[str, Any]]:
    """Targets of a row that have not filled yet."""
    return [target for target in row["targets"] if target["state"] == "working"]


def level_for_exit(row: LevelRow, reason: str, target_index: int | None) -> Decimal | None:
    """The level a stop exit (reason leg_sl) or a target exit is measured against in a row."""
    if reason == "leg_sl":
        return stop_level(row)
    for target in working_targets(row):
        if target["index"] == target_index:
            return rupees(target["level_rupees"])
    return None


def fills_at(case: Case, instrument: str, slice_ist: str) -> list[dict[str, Any]]:
    """Expected fills of one instrument in one slice, in the order listed."""
    return [
        fill
        for fill in case["expected"]["fills"]
        if fill["instrument"] == instrument and fill["slice_ist"] == slice_ist
    ]


def has_fill_at_the_open(case: Case, instrument: str, slice_ist: str) -> bool:
    """True when the instrument has a step 2 fill in the slice.

    Such a fill (an entry, an add, a square-off, a roll) changes the levels before step 3,
    so the row in force no longer describes what step 3 tested.
    """
    return any(fill["step"] == STEP_FILLS_AT_OPEN for fill in fills_at(case, instrument, slice_ist))


def is_beyond(price: Decimal, level: Decimal, upward: bool) -> bool:
    """True when a price has reached a level approached from below (upward) or above."""
    return price >= level if upward else price <= level
