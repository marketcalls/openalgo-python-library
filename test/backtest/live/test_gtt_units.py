"""The references a GTT's leg's units are judged against, and what the units say
(`_live/_gtt_units.py`; the lead's ruling M1 on N3a's review round 5, 20-positional-live
3.1 and 3.3): one store keeps each leg's latest settled read; a GTT's reference is the
leg's latest when it could not yet have fired; units read while the session's own fills on
the leg had not settled judge nothing.

The leg is SBIN NSE CNC; the stop sells 10 when SBIN trades at 790.00 or below (child
limit 780.00). Expected answers are worked by hand from each row.
"""

from __future__ import annotations

import pytest

from openalgo.backtest._live._gtt_reads import BookRead
from openalgo.backtest._live._gtt_units import Reference, References, UnitsSign, judge
from openalgo.backtest._live.journal import GttTerms

NS = 10**9
LEG = ("SBIN", "NSE", "CNC")
OTHER_LEG = ("SBIN", "NSE", "NRML")
STOP = GttTerms("SELL", 790.0, 780.0, 10)
#: A strategy's GTT entry: buy 10 when SBIN trades at 810.00 or above.
ENTRY = GttTerms("BUY", 810.0, 820.0, 10)


def book_read(units=10, own=0, *, settled=True, read_s=100.0, leg=LEG):
    """A read of the GTT book with `leg`'s units and the session's own fills on it."""
    return BookRead(
        rows={},
        read_ns=round(read_s * NS),
        units={leg: units},
        own={leg: own},
        settled=frozenset({leg}) if settled else frozenset(),
    )


def test_a_settled_read_of_a_leg_is_its_reference():
    references = References()

    references.take_read(book_read(units=10, own=0))

    assert references.latest(LEG) == Reference(10, 0)


def test_a_read_made_before_the_sessions_own_fills_on_the_leg_settled_is_never_a_reference():
    references = References()

    references.take_read(book_read(units=10, own=0, settled=False))

    assert references.latest(LEG) is None


def test_a_reference_read_before_the_one_held_does_not_replace_it():
    references = References()
    references.take(LEG, 10, 0, read_ns=200 * NS)

    # The session's start read, given late: it was read before the GTT parts' read.
    references.take(LEG, 0, 0, read_ns=100 * NS)

    assert references.latest(LEG) == Reference(10, 0)


def test_a_gtts_reference_is_its_legs_latest_when_pinned_and_later_reads_leave_it():
    references = References()
    references.take(LEG, 0, 0, read_ns=100 * NS)
    references.pin(1, LEG)

    references.take(LEG, 10, 10, read_ns=200 * NS)

    pinned, latest = references.of(1, own=10), references.latest(LEG)

    assert (pinned, latest) == (Reference(0, 0), Reference(10, 10))


def test_a_gtt_pinned_while_its_leg_has_no_reference_has_none():
    references = References()
    references.take(OTHER_LEG, 10, 0, read_ns=100 * NS)

    references.pin(1, LEG)

    assert references.of(1, own=0) is None


def test_a_reference_taken_back_from_the_journal_counts_own_fills_from_when_they_are_known():
    references = References()
    references.restore(1, 10)

    # The order path counts 3 units of its own on the leg when the GTT is first judged; an
    # exit sells 5 more later.
    first = references.of(1, own=3)
    later = references.of(1, own=-2)

    assert (first.expected(3), later.expected(-2)) == (10, 5)


@pytest.mark.parametrize(
    ("terms", "reference", "read", "sign"),
    [
        (STOP, None, book_read(units=0), UnitsSign.NO_REFERENCE),
        (STOP, Reference(10, 0), book_read(units=0), UnitsSign.FIRED),
        (STOP, Reference(10, 0), book_read(units=10), UnitsSign.UNCHANGED),
        (STOP, Reference(10, 0), book_read(units=6), UnitsSign.OTHER),
        (STOP, Reference(10, 0), book_read(units=0, settled=False), UnitsSign.UNSETTLED),
        (STOP, Reference(10, 0), book_read(units=None), UnitsSign.UNSETTLED),
        # The reference from before the session's own entry of 10: 0 + 10 if nothing fired.
        (STOP, Reference(0, 0), book_read(units=10, own=10), UnitsSign.UNCHANGED),
        (STOP, Reference(0, 0), book_read(units=0, own=10), UnitsSign.FIRED),
        # A partial exit sold 5 of the 10, the stop cut to the 5 left: 10 - 5 if nothing
        # fired.
        (
            GttTerms("SELL", 790.0, 780.0, 5),
            Reference(10, 0),
            book_read(units=0, own=-5),
            UnitsSign.FIRED,
        ),
        (ENTRY, Reference(0, 0), book_read(units=10), UnitsSign.FIRED),
    ],
    ids=[
        "no reference",
        "fallen by the stop's units",
        "unchanged",
        "moved by other units",
        "not settled",
        "not read",
        "the entry's units only",
        "the entry's units and the stop's fall",
        "a partial exit and the stop's fall",
        "an entry risen by its units",
    ],
)
def test_what_a_settled_read_of_a_legs_units_says_of_its_gtt(terms, reference, read, sign):
    assert judge(terms, reference, read, LEG).sign is sign
