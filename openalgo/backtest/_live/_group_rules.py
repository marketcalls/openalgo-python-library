"""The rules of an instrument group against the broker's net position (18-multi-strategy
4.6, 4.7, 4.8 and 4.10; units M6 and M7). Pure: the reconciler's group comparison
(`_reconcile_groups`) and the restart's (`_restart_groups`) act on what these give.

A broker nets the strategies of a session that trade one instrument into one position, so
their legs are judged together: the sum of the members' units (`E`) against the broker's
net (`B`), with `X` the units of other orders' trades not taken in yet.

- **The group's kind** (`group_kind`), the rows of 4.6 in their general form, judged in
  this order: the books unreadable; agreeing (`B == E`); the other side (`B` strictly on
  the other side of `E`, `Kind.OTHER_SIDE`); an excess no member's closing orders can
  overshoot (`B > E` with no short member, or `B < E` with no long member, `Kind.MORE`);
  explained and absorbed (`X` accounts for `B - E` and the members on the side it closes
  hold at least its units, `Kind.EXPLAINED_SHARED`: they take it by the allocation of 4.7,
  `_reconcile_allocate`, each its `Share`); else unexplained. The side gate keeps a live
  group's open members on one side (4.13), so the excess and the other side take their
  one-sided forms there; with one member the kinds are a leg's (`_reconcile_rules.judge`),
  an explained difference named `EXPLAINED_SHARED`.
- **What an excess and the other side do** (`GROUP_WORDS`): nothing adopted and nothing
  sent for the difference, every member's entries locked; on the other side each member's
  stop is cancelled (4.7's cut with the broker on the other side: every stop on the
  members' side cut to nothing) and one ERROR alert names the instrument
  (`other_side_alert`); the words say to end the session before closing by hand.
- **At a restart** (`group_restart_kind`): `_reconcile_rules.restart_kind` of the sum, but
  a broker position is unknown only when every member's journal is flat.
- **The words** (`GroupFigure`, `and_join`): an alert of a member names the group and the
  sum it was judged by, never the member's units against the broker's net alone.

Quantities are signed units (long above 0), prices rupees.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from ._reconcile_rules import Kind, RestartKind

#: The words of a member's RECONCILE_MISMATCH alert for the group rows that adopt nothing
#: (module docstring): what is done, and what to do. `{excess}` is the broker's units
#: beyond the members' sum.
GROUP_WORDS: Mapping[Kind, str] = MappingProxyType(
    {
        Kind.MORE: (
            "the {excess} units beyond what they hold are no strategy's of the session: "
            "nothing is sent for them and they have no stop at the broker. End the session, "
            "then close them by hand in OpenAlgo"
        ),
        Kind.OTHER_SIDE: (
            "the broker holds the other side of what they hold together: nothing is "
            "adopted, each member's stop is cancelled and nothing else is sent. End the "
            "session, then close the position by hand in OpenAlgo"
        ),
    }
)


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
    the members take; none otherwise). `held` is each member's signed units in the order
    of `members` (empty when not given) and `external` the signed units of the other
    orders' trades of the instrument not taken in yet (`X`; None when the trade book was
    not read)."""

    members: tuple[str, ...]
    units: int
    shares: tuple[Share, ...] = ()
    held: tuple[int, ...] = ()
    external: int | None = None

    def others(self, name: str) -> tuple[str, ...]:
        """The members but strategy `name`, in call order."""
        return tuple(member for member in self.members if member != name)

    def share(self, name: str) -> Share | None:
        """Strategy `name`'s share of the allocation, None when it has none."""
        return next((share for share in self.shares if share.name == name), None)

    def member_units(self) -> str:
        """Each member's units in words, call order: "A 100, B 50"; "" when not given."""
        return ", ".join(f"{n} {u}" for n, u in zip(self.members, self.held, strict=False))

    def allocation(self) -> str:
        """The allocation made, in words, in the order the members take: each member that
        takes a share, its signed units taken and what it holds after ("B -50 to 0, A -10
        to 90"); "" for none."""
        taken = [share for share in self.shares if share.taken]
        return ", ".join(f"{s.name} {s.taken:+d} to {s.left}" for s in taken)


def and_join(names: Sequence[str]) -> str:
    """`names` in words: "A", "A and B", "A, B and C"."""
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def group_kind(member_units: Sequence[int], broker_units: int | None, unexplained: int) -> Kind:
    """The group's position against the broker's (module docstring): `member_units` each
    member's signed units, `broker_units` the broker's net (None for a book that could not
    be read), `unexplained` what the trades of other orders, less what the members adopted
    of them, do not account for of the difference (`_reconcile_rules.unexplained_by` of the
    members' sum)."""
    if broker_units is None:
        return Kind.UNREADABLE
    held = sum(member_units)
    difference = broker_units - held
    if difference == 0:
        return Kind.AGREE
    if held * broker_units < 0:
        return Kind.OTHER_SIDE
    if _excess(member_units, difference):
        return Kind.MORE
    if unexplained == 0 and _absorbed(member_units, difference):
        return Kind.EXPLAINED_SHARED
    return Kind.UNEXPLAINED


def group_words(kind: Kind, figure: GroupFigure, broker_units: int) -> str:
    """The words of a member's alert for an excess or the other side of the group of
    `figure` against `broker_units` (`GROUP_WORDS`). Raises KeyError for another kind."""
    return GROUP_WORDS[kind].format(excess=abs(broker_units - figure.units))


def other_side_alert(symbol: str, exchange: str, figure: GroupFigure, broker_units: int) -> str:
    """The one ERROR alert of a shared instrument whose broker holds the other side of its
    members' sum (module docstring)."""
    return (
        f"{symbol} {exchange}: OpenAlgo's books show {broker_units} units, the other side of "
        f"the {figure.units} units {and_join(figure.members)} hold together: end the session, "
        "then close the position by hand in OpenAlgo"
    )


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


def _excess(member_units: Sequence[int], difference: int) -> bool:
    """Whether the broker's `difference` from the members' sum adds units no member's
    closing orders can overshoot: more with no short member, fewer with no long one."""
    if difference > 0:
        return not any(units < 0 for units in member_units)
    return not any(units > 0 for units in member_units)


def _absorbed(member_units: Sequence[int], difference: int) -> bool:
    """Whether the members on the side the broker's `difference` closes hold at least its
    units."""
    closing = sum(abs(units) for units in member_units if units * difference < 0)
    return closing >= abs(difference)


__all__ = [
    "GROUP_WORDS",
    "GroupFigure",
    "Share",
    "and_join",
    "group_kind",
    "group_restart_kind",
    "group_words",
    "other_side_alert",
]
