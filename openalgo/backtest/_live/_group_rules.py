"""The rules of an instrument group against the broker's net position (18-multi-strategy
4.6, 4.7, 4.8 and 4.10; unit M6, before the other group rows of unit M7). Pure: the
reconciler's group comparison (`_reconcile_groups`) and the restart's (`_restart_groups`)
act on what these give.

A broker nets the strategies of a session that trade one instrument into one position, so
their legs are judged together: the sum of the members' units against the broker's net.

- **The group's kind** (`group_kind`): `_reconcile_rules.judge` of the sum, but a
  difference trades of other orders explain is `Kind.EXPLAINED_SHARED`: the members it
  closes take it by the allocation of 4.7 (`_reconcile_allocate`), each its `Share`.
- **At a restart** (`group_restart_kind`): `_reconcile_rules.restart_kind` of the sum, but
  a broker position is unknown only when every member's journal is flat.
- **The words** (`GroupFigure`, `and_join`): an alert of a member names the group and the
  sum it was judged by, never the member's units against the broker's net alone.

Quantities are signed units (long above 0), prices rupees.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ._reconcile_rules import Kind, RestartKind, judge


@dataclass(frozen=True)
class Share:
    """One member's part of a difference other orders' trades explain on its shared
    instrument (`_reconcile_allocate.allocate`, 18-multi-strategy 4.7): the strategy's
    `name`, the signed `units` it held, the signed units `taken` from it (a sale takes them
    from a long, below 0; none from a member the allocation leaves as it is), and the price
    in rupees of the other orders' trades it adopts (None when it takes none, or when no
    such trade is on the side it takes)."""

    name: str
    units: int
    taken: int
    price: float | None = None

    @property
    def left(self) -> int:
        """The signed units the member holds once it has adopted its share."""
        return self.units + self.taken


@dataclass(frozen=True)
class GroupFigure:
    """A shared instrument as a comparison judged it: its `members` (strategy names, in
    call order), `units`, the sum of their signed units (the journal's at a restart), and,
    for a difference other orders' trades explain, its allocation (`shares`, in the order
    the members take; none otherwise)."""

    members: tuple[str, ...]
    units: int
    shares: tuple[Share, ...] = ()

    def others(self, name: str) -> tuple[str, ...]:
        """The members but strategy `name`, in call order."""
        return tuple(member for member in self.members if member != name)

    def share(self, name: str) -> Share | None:
        """Strategy `name`'s share of the allocation, None when it has none."""
        return next((share for share in self.shares if share.name == name), None)


def and_join(names: Sequence[str]) -> str:
    """`names` in words: "A", "A and B", "A, B and C"."""
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def group_kind(group_units: int, broker_units: int | None, unexplained: int) -> Kind:
    """The group's position against the broker's (module docstring): `group_units` the sum
    of the members' units, `broker_units` the broker's net (None for a book that could
    not be read), `unexplained` what the trades of other orders, less what the members
    adopted of them, do not account for of the difference
    (`_reconcile_rules.unexplained_by`)."""
    kind = judge(group_units, broker_units, unexplained)
    if kind in (Kind.EXPLAINED_FLAT, Kind.EXPLAINED_FEWER):
        return Kind.EXPLAINED_SHARED
    return kind


def group_restart_kind(
    member_units: Sequence[int], broker_units: int, unexplained: int
) -> RestartKind:
    """What a restart does with a shared instrument (module docstring): `member_units` the
    members' journal positions, `broker_units` the broker's net, `unexplained` what the
    trades of other orders do not account for of the difference from their sum."""
    if sum(member_units) == broker_units:
        return RestartKind.RESUME
    if not any(member_units):
        return RestartKind.UNKNOWN_POSITION
    if broker_units == 0 and unexplained != 0:
        return RestartKind.UNEXPLAINED_FLAT
    return RestartKind.DIFFERS


__all__ = ["GroupFigure", "Share", "and_join", "group_kind", "group_restart_kind"]
