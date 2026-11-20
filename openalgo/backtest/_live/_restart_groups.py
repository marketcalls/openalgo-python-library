"""A restart judged per instrument group (18-multi-strategy 4.10; unit M6: what a restart
needs once two strategies of a session may share an instrument, before the group rows and
the allocation of unit M7).

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
  judged by the group's kind (`_group_rules.group_kind`: nothing is adopted before unit
  M7's allocation), its exits held by the exit guard while the broker holds fewer units,
  none or the other side (18-multi-strategy 4.8).

Quantities are signed units. Run by the main thread at the start.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ._group_rules import GroupFigure, and_join, group_kind, group_restart_kind
from ._groups import InstrumentGroup
from ._reconcile_rules import Kind, RestartKind, unexplained_by


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
    external_units: int,
    adopted_units: int,
) -> GroupStart:
    """The members of `group` judged together (module docstring): `journal_units` the
    journal position of each member judged (by name; a member missing from it is not
    judged), `broker_units` the broker's net, `external_units` the signed units of the
    trades of orders the session did not send, `adopted_units` what the members adopted of
    them before the restart."""
    names = tuple(name for name in group.members if name in journal_units)
    units = [journal_units[name] for name in names]
    total = sum(units)
    unexplained = unexplained_by(total, broker_units, external_units, adopted_units)
    verdict = group_restart_kind(units, broker_units, unexplained)
    figure = GroupFigure(names, total)
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
    return GroupStart(verdict, group_kind(total, broker_units, unexplained), figure, refusal)


__all__ = ["GroupStart", "judge_group"]
