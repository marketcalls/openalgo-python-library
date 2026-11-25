"""What a GTT's leg's units say of it, and the references they are judged against (the
lead's rulings M1 to M4 on N3a's review round 5; 20-positional-live 3.1, 3.3). Pure: the
GTT follower (`_gtt_follow`) keeps the references, and it and the GTT sender
(`_gtt_send`) judge by them.

**A reference** of a leg is a settled read of its broker units (`_own_fills.OwnFills.
settled`: the broker's books can be taken to show every fill of the session's own on it)
with what the session's own orders had filled on it then. A read that is not settled is
never a reference. `References` keeps the latest per leg (`take`), from the GTT parts' own
reads (each read of the GTT book reads the legs' units first, `_gtt_reads`) and from the
session's (`GttFollower.reference`).

**A GTT's reference** is one made when it could not yet have fired: the leg's latest when
its place is sent, and the leg's latest again each time a read shows it resting (its leg's
units were read just before that read of the GTT book, and a GTT that fired never rests
again): `pin`. A GTT taken back from the journal on a restart has the units the journal
kept for it (`restore`): they are net of every own fill the earlier process heard, so the
own fills they are moved by count from the first time this process knows them (`of`).

**The judgement** (`judge`): the units the leg would hold had the GTT not fired are its
reference's, moved by the session's own filled units on the leg since
(`Reference.expected`). A settled read of the leg's units against them says: moved by
exactly the GTT's units on its side (a SELL lowers them, a BUY raises them: a long's stop,
a short's stop and a strategy's GTT entry alike), it FIRED; not moved, UNCHANGED; moved by
anything else, OTHER. A read that is not settled, or whose units could not be read, judges
nothing (UNSETTLED); a GTT with no reference cannot be judged (NO_REFERENCE). Each caller
says what it answers to each, in the words of `UNTOLD` when the units cannot tell.

Times are UTC nanoseconds, quantities signed whole units.
"""

from __future__ import annotations

import dataclasses
import enum
from dataclasses import dataclass

from ._gtt_reads import UNITS_WAIT_S, BookRead, LegKey
from ._journal_gtt_rows import GttTerms


class UnitsSign(enum.Enum):
    """What a read of a GTT's leg's units says of the GTT (module docstring)."""

    #: Settled, and moved by exactly the GTT's units on its side: it fired.
    FIRED = "fired"
    #: Settled, and as its reference moved by the session's own fills: it did not fire.
    UNCHANGED = "unchanged"
    #: Settled, and moved by anything else: not by the GTT's units alone (a gone GTT's
    #: follower cannot tell; a cancel or a place takes it as no move by the GTT).
    OTHER = "other"
    #: Read while the session's own fills on the leg had not settled, or not read: the
    #: read judges nothing.
    UNSETTLED = "unsettled"
    #: The GTT has no reference: nothing can judge it.
    NO_REFERENCE = "no_reference"


#: Why a GTT's leg's units cannot tell whether it fired, by the sign that says so: the
#: words of the answers and outcomes the GTT parts give for want of a judgement. An
#: UNSETTLED sign is answered only once the wait for a settled read is over
#: (`_gtt_reads.BookRead.waited_out`).
UNTOLD = {
    UnitsSign.NO_REFERENCE: (
        "no settled read of its leg's units was made when it could not have fired"
    ),
    UnitsSign.UNSETTLED: (
        f"its leg's units were not settled {UNITS_WAIT_S:g} s after the wait for them began "
        "and after the session's newest own fill on the leg (an own order still working on "
        "it, or its books not read)"
    ),
}


@dataclass(frozen=True)
class Reference:
    """A settled read of a leg's broker units, `units`, with `own`, what the session's
    own orders had filled on the leg then (None for a reference taken back from the
    journal, until this process knows them)."""

    units: int
    own: int | None

    def expected(self, own: int) -> int:
        """The leg's units had nothing but the session's own orders moved them since: the
        reference's, moved by what those orders filled from then to `own`, what they have
        filled by now."""
        return self.units if self.own is None else self.units + own - self.own


@dataclass(frozen=True)
class LegUnits:
    """What a read says of a GTT by its leg's units: the sign, the units the leg would
    hold had the GTT not fired and the units read (both None when the read judges
    nothing)."""

    sign: UnitsSign
    expected: int | None = None
    units: int | None = None


class References:
    """The references of the legs and of the GTTs (module docstring), GTTs by journal
    number."""

    def __init__(self) -> None:
        self._legs: dict[LegKey, tuple[int, Reference]] = {}
        self._gtts: dict[int, Reference] = {}

    def take(self, leg: LegKey, units: int, own: int, read_ns: int) -> None:
        """A settled read of `leg` made at `read_ns` (UTC ns): `units` at the broker, `own`
        filled by the session's own orders. It is the leg's latest reference unless the
        one held was read later."""
        held = self._legs.get(leg)
        if held is None or read_ns >= held[0]:
            self._legs[leg] = (read_ns, Reference(units, own))

    def take_read(self, read: BookRead) -> None:
        """Take each leg `read` read settled (`take`)."""
        for leg in read.settled:
            units, own = read.units.get(leg), read.own.get(leg)
            if units is not None and own is not None:
                self.take(leg, units, own, read.read_ns)

    def latest(self, leg: LegKey) -> Reference | None:
        """`leg`'s latest reference, or None when none was taken."""
        held = self._legs.get(leg)
        return None if held is None else held[1]

    def pin(self, seq: int, leg: LegKey) -> None:
        """GTT `seq` on `leg` could not have fired when the leg's latest reference was
        read (its place was not sent yet, or a read that showed it resting came after):
        that reference is the GTT's. A leg with none leaves the GTT's as it was."""
        latest = self.latest(leg)
        if latest is not None:
            self._gtts[seq] = latest

    def restore(self, seq: int, units: int) -> None:
        """GTT `seq`'s reference taken back from the journal: the `units` it kept, net of
        every own fill the earlier process heard (module docstring)."""
        self._gtts[seq] = Reference(units, None)

    def of(self, seq: int, own: int | None) -> Reference | None:
        """GTT `seq`'s reference, or None; `own` is what the session's own orders have
        filled on its leg now (None: not known), the own units a reference taken back from
        the journal is moved from."""
        reference = self._gtts.get(seq)
        if reference is not None and reference.own is None and own is not None:
            reference = dataclasses.replace(reference, own=own)
            self._gtts[seq] = reference
        return reference

    def drop(self, seq: int) -> None:
        """Forget GTT `seq`'s reference."""
        self._gtts.pop(seq, None)


def judge(terms: GttTerms, reference: Reference | None, read: BookRead, leg: LegKey) -> LegUnits:
    """What `read` says by `leg`'s units of a GTT of `terms` whose reference is
    `reference` (module docstring)."""
    if reference is None:
        return LegUnits(UnitsSign.NO_REFERENCE)
    units, own = read.units.get(leg), read.own.get(leg)
    if leg not in read.settled or units is None or own is None:
        return LegUnits(UnitsSign.UNSETTLED)
    expected = reference.expected(own)
    signed = terms.quantity if terms.side == "BUY" else -terms.quantity
    if units - expected == signed:
        sign = UnitsSign.FIRED
    else:
        sign = UnitsSign.UNCHANGED if units == expected else UnitsSign.OTHER
    return LegUnits(sign, expected, units)


__all__ = ["UNTOLD", "LegUnits", "Reference", "References", "UnitsSign", "judge"]
