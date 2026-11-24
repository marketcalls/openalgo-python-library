"""What the close of a live session names besides each leg it holds (16-session-design 5.4
"As built", rounds 7 to 9; 18-multi-strategy 4.11; moved from `_session_close` in unit M7,
which adds the shared instruments).

- **The legs named** (`named_legs`, `NamedLegs`): each leg whose last comparison was late
  and saw a difference, or whose difference with OpenAlgo's books no later read confirms
  (`_reconcile_close`), each a mismatch of the day unless comparisons in time counted it
  then; then each with none that no read judged after a fill of its own orders landed
  during a read, which is no mismatch. A difference of an instrument several strategies
  of the session share is its group's: named once, in the group's words, and counted
  once, whichever member's sighting holds it.
- **Their alerts** (`alert_not_confirmed`): one per leg no read judged after such a fill,
  to check OpenAlgo's position book, and one ERROR alert per leg whose last comparison was
  late and saw a difference, with that read's figures.
- **The shared instruments** (`shared_positions`, `SharedPosition`): for each instrument
  several strategies of the session trade, OpenAlgo's net at the last read a comparison of
  its group took, carried by the members' own fills since (each moved OpenAlgo's books by
  the same units), the members' sum now, and whether they agree. A read compared late (a
  member in flight past the limit) may already hold a fill its member hears only later, so
  its net is named as that read saw it, neither carried nor confirmed (`LATE_GROUP_READ`),
  as a leg's late sighting is (`_reconcile_close.LATE_READ`). The close lists the
  instrument in `open_positions` when OpenAlgo's net is a position, or when it is flat
  and differs from the members' sum while a member is listed open
  (`SharedPosition.listed`, `SharedPosition.note`: a flat net no member is listed against
  leaves nothing open at the broker), and gives it a line of the DAILY_SUMMARY
  (`SharedPosition.line`). A net no comparison of this start read is named as not read,
  never guessed.

Quantities are signed whole units. Read while the kernel sessions run: a finished one has
no units to read. Used by the close only (`_session_close`).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from ._group_rules import GroupFigure, and_join
from ._reconcile_close import (
    NOT_CONFIRMED_ALERT,
    Differing,
    differing,
    late_alert,
    late_group_summary,
    late_summary,
    not_confirmed_alert,
    not_confirmed_summary,
)
from ._reconcile_findings import Sighting
from ._types import AlertKind

#: How the note and the line of a shared instrument say OpenAlgo's net was carried to the
#: close by the members' own fills since the read that saw it.
CARRIED = " and their own fills since"
#: How they name a net the group's last read saw late (module docstring), as it saw it.
LATE_GROUP_READ = (
    "at a late read, while an order of a member was in flight past the limit, neither "
    "carried by their fills since nor confirmed by a later read"
)


@dataclass(frozen=True)
class NamedLegs:
    """The legs the summary names at the close (`named_legs`), in its words: `differs`,
    those whose position differed from OpenAlgo's books (a shared instrument's group once),
    `unsure`, those no read confirmed, and `new_mismatches`, how many of `differs` the day
    has not counted yet (a late difference comparisons in time acted on was counted
    then)."""

    differs: tuple[str, ...] = ()
    unsure: tuple[str, ...] = ()
    new_mismatches: int = 0


def named_legs(session: Any) -> NamedLegs:
    """The legs the summary names at the close, in its words (module docstring): each whose
    last comparison was late and saw a difference (`_reconcile_close.late_summary`) or
    whose difference with OpenAlgo's books no later read confirms
    (`_reconcile_close.Differing.summary`), a shared instrument's once in its group's words;
    then each with none that no read judged after a fill of its own orders landed during a
    read (`_reconcile_close.not_confirmed_summary`). Each differing leg is a mismatch of the
    day but a late one comparisons in time acted on, counted then
    (`Reconciler.late_counted`). `session` is the `_session_loop.Session`."""
    reconcile = session.reconcile
    differs, unsure, counted = [], [], 0
    groups: set[tuple[str, str, str]] = set()
    for trader in session.traders:
        late = reconcile.late(trader.name)
        seen = differing(reconcile, trader)
        group = _group_of(late, seen)
        if group is not None:
            key = (trader.symbol, trader.exchange, trader.product)
            if key in groups:
                continue
            groups.add(key)
        if late is not None:
            words = late_summary if group is None else late_group_summary
            differs.append(words(trader, late))
            counted += reconcile.late_counted(trader.name)
        elif seen is not None:
            differs.append(seen.summary(trader) if group is None else seen.group_summary(trader))
        elif reconcile.unjudged(trader.name):
            unsure.append(not_confirmed_summary(trader, round(trader.open_units())))
    return NamedLegs(tuple(differs), tuple(unsure), len(differs) - counted)


def _group_of(late: Sighting | None, seen: Differing | None) -> GroupFigure | None:
    """The group of a leg's difference the close names, the late one first; None for a
    strategy alone or a leg with no difference to name."""
    if late is not None:
        return late.group
    return None if seen is None else seen.seen.group


def alert_not_confirmed(session: Any) -> None:
    """One alert per leg no read judged after a fill of its own orders landed during a
    read, to check OpenAlgo's position book (`_reconcile_close.not_confirmed_alert`), and
    one ERROR alert per leg whose last comparison was late and saw a difference, with that
    read's figures (`_reconcile_close.late_alert`)."""
    for trader in session.traders:
        late = session.reconcile.late(trader.name)
        if late is not None:
            session.alert(AlertKind.ERROR, late_alert(trader, late))
        elif session.reconcile.unjudged(trader.name):
            differs = differing(session.reconcile, trader)
            text = not_confirmed_alert(trader, round(trader.open_units()), differs)
            session.alert(NOT_CONFIRMED_ALERT, text)


@dataclass(frozen=True)
class SharedPosition:
    """An instrument several strategies of the session trade, at the close (module
    docstring): `symbol`, `exchange`, its `members` in call order, OpenAlgo's `net` (signed
    units, None when no comparison of this start read it), whether the members' own fills
    since that read carried it (`carried`), `held`, the members' sum now, and whether that
    read compared `late` (its net as it saw it, not confirmed)."""

    symbol: str
    exchange: str
    members: tuple[str, ...]
    net: int | None
    held: int
    carried: bool = False
    late: bool = False

    @property
    def agrees(self) -> bool:
        """True when OpenAlgo's net is known, from a read in time, and equals the
        members' sum."""
        return not self.late and self.net == self.held

    def listed(self, members_open: bool) -> bool:
        """True when the close lists the instrument in `open_positions`: OpenAlgo's net is
        known and is a position, or it is flat and differs from the members' sum while a
        member is listed open (`members_open`). A flat net no member is listed against
        leaves nothing open at the broker (16-session-design 5.4 "As built", round 8)."""
        if self.net is None:
            return False
        return self.net != 0 or (members_open and self.net != self.held)

    def note(self) -> str:
        """The instrument's note in `open_positions`: OpenAlgo's net, how it is known,
        against the members' sum, and what to do when they differ."""
        whose = f"OpenAlgo's net position of {self.symbol} for {and_join(self.members)} together"
        if self.late:
            return (
                f"{whose}, {LATE_GROUP_READ}: their sum is {self.held}; check the position "
                "book in OpenAlgo"
            )
        known = f"by the last read{CARRIED if self.carried else ''}"
        if self.agrees:
            return f"{whose}, {known}: it equals their sum; each holds its own stop on its row"
        return (
            f"{whose}, {known}: their sum is {self.held}, so they differ; check the position "
            "book in OpenAlgo"
        )

    def line(self) -> str:
        """The instrument's line of the DAILY_SUMMARY: OpenAlgo's net, the members' sum,
        whether they agree."""
        where = f"{self.symbol} {self.exchange} ({and_join(self.members)})"
        if self.net is None:
            return f"{where}: OpenAlgo's net not read, their sum {self.held}"
        if self.late:
            return (
                f"{where}: OpenAlgo's net {self.net} {LATE_GROUP_READ}; their sum "
                f"{self.held}: not confirmed"
            )
        verdict = "agree" if self.agrees else "differ"
        carried = CARRIED if self.carried else ""
        return (
            f"{where}: OpenAlgo's net {self.net} by the last read{carried}, their sum "
            f"{self.held}: they {verdict}"
        )


def shared_positions(session: Any) -> tuple[SharedPosition, ...]:
    """Each instrument several strategies of `session` (the `_session_loop.Session`)
    trade, in the order of its first member, at the close (module docstring)."""
    return tuple(_shared(session))


def _shared(session: Any) -> Iterator[SharedPosition]:
    groups = session.groups
    for group in groups.groups:
        if not group.shared:
            continue
        members = groups.of_group(group)
        held = sum(round(member.open_units()) for member in members)
        read = session.reconcile.group_read(group)
        where = (group.symbol, group.exchange, group.members)
        if read is None:
            yield SharedPosition(*where, None, held)
        elif read.late:
            yield SharedPosition(*where, read.net, held, late=True)
        else:
            net = read.net + held - read.held
            yield SharedPosition(*where, net, held, carried=held != read.held)


__all__ = [
    "CARRIED",
    "LATE_GROUP_READ",
    "NamedLegs",
    "SharedPosition",
    "alert_not_confirmed",
    "named_legs",
    "shared_positions",
]
