"""A restart judged per instrument group (18-multi-strategy 4.10; unit M6: what a restart
needs once two strategies of a session may share an instrument, before the other group
rows of unit M7).

A broker nets the members of an instrument into one position, so a restart compares the
sum of the members' journal positions (16-session-design 3.6, each computed as for a leg)
with the broker's net (`judge_group`), never one member's with it. A group of one is
judged as a leg always was (`_restart_books.judge_start`).

- **The sum is the broker's net** (RESUME): every member resumes.
- **A broker position, every member's journal flat** (UNKNOWN_POSITION): the start is
  refused, naming the instrument and its members.
- **The members hold units, the broker none, and no trade of another order explains it**
  (UNEXPLAINED_FLAT): refused, naming the group.
- **Any other difference** (DIFFERS): every member resumes with its entries locked,
  judged by the group's kind (`_group_rules.group_kind`). A difference trades of other
  orders explain is shared out among the members by their journal positions, the most
  recently opened first (`_reconcile_allocate`, when each opened told by its saved
  position), each adopting its share; while the broker holds fewer units that nothing
  adopts, or the other side, every member's exits are held by the exit guard
  (18-multi-strategy 4.8).

Quantities are signed units. Run by the main thread at the start.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ._group_rules import GroupFigure, and_join, group_kind, group_restart_kind
from ._groups import InstrumentGroup
from ._reconcile_allocate import allocate
from ._reconcile_rules import Kind, RestartKind, Trade, unexplained_by


@dataclass(frozen=True)
class GroupStart:
    """How a restart judged one shared instrument: its verdict, the kind of its difference
    (`Kind.AGREE` for none), the members and their sum (`figure`), and the words of its
    refusal (None unless the verdict refuses the start)."""

    restart: RestartKind
    kind: Kind
    figure: GroupFigure
    refusal: str | None = None


def judge_group(
    group: InstrumentGroup,
    journal_units: Mapping[str, int],
    broker_units: int,
    external: Sequence[Trade],
    adopted_units: int,
    opened_ns: Mapping[str, int | None] | None = None,
) -> GroupStart:
    """The members of `group` judged together (module docstring): `journal_units` the
    journal position of each member judged (by name; a member missing from it is not
    judged), `broker_units` the broker's net, `external` the trades of orders the session
    did not send (oldest first), `adopted_units` the signed units the members adopted of
    them before the restart, `opened_ns` when each member's saved position opened (UTC
    nanoseconds; a member missing, or None, not known)."""
    names = tuple(name for name in group.members if name in journal_units)
    units = [journal_units[name] for name in names]
    total = sum(units)
    external_units = sum(trade.signed for trade in external)
    unexplained = unexplained_by(total, broker_units, external_units, adopted_units)
    verdict = group_restart_kind(units, broker_units, unexplained)
    kind = group_kind(units, broker_units, unexplained)
    shares = ()
    if kind is Kind.EXPLAINED_SHARED:
        opened = opened_ns or {}
        holders = [(name, held, opened.get(name)) for name, held in zip(names, units, strict=True)]
        shares = allocate(holders, broker_units - total, external)
    figure = GroupFigure(names, total, shares, tuple(units), external_units - adopted_units)
    where, who = f"{group.symbol} {group.exchange}", and_join(names)
    refusal = None
    if verdict is RestartKind.UNKNOWN_POSITION:
        refusal = (
            f"OpenAlgo's books show {broker_units} units of {where}, which {who} trade, and "
            "the journal holds none for them: close them or start with another symbol"
        )
    elif verdict is RestartKind.UNEXPLAINED_FLAT:
        each = ", ".join(f"{name} {held}" for name, held in zip(names, units, strict=True))
        refusal = (
            f"the journal holds {total} units of {where} for {who} together ({each}), "
            "OpenAlgo's books show none and no trade explains it: check the account in "
            "OpenAlgo"
        )
    return GroupStart(verdict, kind, figure, refusal)


__all__ = ["GroupStart", "judge_group"]
