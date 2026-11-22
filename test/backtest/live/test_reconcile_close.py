"""What the close says of a leg whose difference with OpenAlgo's books no later read will
confirm (`_live/_reconcile_close.py`, `_live/_reconcile_findings.py`; 16-session-design 5.4
"As built", rounds 7 to 9; unit 12).

A comparison saw OpenAlgo's books differ from the engine for the first time (a
`Sighting`), and the session ended before a later read confirmed it. The engine's units
may have moved since by its own orders' fills, which the broker made too: OpenAlgo's
figure moves by the same units, and the close names the read's figures, the fills since
and what they leave. A comparison whose read carried a fill of the leg's own orders, or
that was made late (the leg in flight past the limit, its trades not attributed), takes no
sighting and keeps an earlier one; after such a fill no read judged the leg until a later
comparison does, and the close says so (round 9). Expected values come from the scenarios:
`StopHold` long 10 SBIN with 4 or 10 sold by hand, its stop of 10 filling or not after the
read. A member of an instrument several strategies of the session share is named by its
group's net against the members' sum, and listed at its own units (18-multi-strategy
4.11): B holding 50 of the 150 it holds with A, 60 or all of them sold by hand.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from openalgo.backtest._live._group_rules import GroupFigure
from openalgo.backtest._live._reconcile_close import (
    Differing,
    differing,
    late_alert,
    not_confirmed_alert,
    not_confirmed_summary,
)
from openalgo.backtest._live._reconcile_findings import Findings, Sighting
from openalgo.backtest._live._reconcile_rules import Kind, LegBooks

#: The leg as the close names it.
TRADER = SimpleNamespace(name="StopHold", symbol="SBIN", exchange="NSE")


@pytest.mark.parametrize(
    ("seen", "engine_now", "broker_now"),
    [
        (Sighting(10, 6), 10, 6),
        (Sighting(10, 6), 0, -4),
        (Sighting(10, 0), 10, 0),
        (Sighting(10, 6), 4, 0),
        (Sighting(0, -4), 0, -4),
        (Sighting(-5, -2), 0, 3),
    ],
    ids=[
        "4 sold by hand, nothing since",
        "4 sold by hand, the stop of 10 filled since",
        "all sold by hand, nothing since",
        "4 sold by hand, the engine sold 6 since",
        "short 4 after the stop filled, nothing since",
        "short 5, 3 bought by hand, the engine bought 5 since",
    ],
)
def test_openalgos_figure_moves_by_the_engines_own_fills_since_the_read(
    seen, engine_now, broker_now
):
    assert Differing(seen, engine_now).broker_units == broker_now


@pytest.mark.parametrize(
    ("seen", "engine_now", "flat"),
    [(Sighting(10, 0), 10, True), (Sighting(10, 6), 4, True), (Sighting(10, 6), 0, False)],
    ids=["all sold by hand", "the rest sold by the engine since", "short 4"],
)
def test_a_leg_is_flat_at_openalgo_when_its_figure_comes_to_no_position(seen, engine_now, flat):
    assert Differing(seen, engine_now).flat_at_openalgo is flat


def test_the_words_name_the_reads_figures_alone_when_no_fill_came_after_it():
    assert Differing(Sighting(10, 6), 10).words() == (
        "OpenAlgo's books showed 6 units against the engine's 10 at the last read, and the "
        "session ended before they agreed: nothing was sent for the difference"
    )


def test_the_words_name_no_position_for_a_flat_figure():
    assert Differing(Sighting(10, 0), 10).words() == (
        "OpenAlgo's books showed no position against the engine's 10 at the last read, and "
        "the session ended before they agreed: nothing was sent for the difference"
    )


@pytest.mark.parametrize(
    ("seen", "engine_now", "since"),
    [
        (
            Sighting(10, 6),
            0,
            "the engine's own fills (10 units sold) came after it, which would leave -4 units "
            "in OpenAlgo's books against the engine's 0",
        ),
        (
            Sighting(10, 6),
            4,
            "the engine's own fills (6 units sold) came after it, which would leave no "
            "position in OpenAlgo's books against the engine's 4",
        ),
        (
            Sighting(-5, -2),
            0,
            "the engine's own fills (5 units bought) came after it, which would leave 3 units "
            "in OpenAlgo's books against the engine's 0",
        ),
    ],
    ids=["the stop of 10 filled", "the engine sold 6", "the engine bought 5"],
)
def test_the_words_name_the_fills_after_the_read_and_what_they_leave(seen, engine_now, since):
    words = Differing(seen, engine_now).words()
    read = (
        f"OpenAlgo's books showed {seen.broker_units} units against the engine's "
        f"{seen.engine_units} at the last read"
    )
    assert words == (
        f"{read}; {since}, and the session ended before they agreed: nothing was sent for "
        "the difference"
    )


@pytest.mark.parametrize(
    ("seen", "engine_now", "summary"),
    [
        (Sighting(10, 6), 10, "StopHold SBIN NSE 6 units in OpenAlgo's books, the engine's 10"),
        (
            Sighting(10, 6),
            0,
            "StopHold SBIN NSE -4 units in OpenAlgo's books by the last read and the engine's "
            "own fills since, the engine's 0",
        ),
        (
            Sighting(10, 0),
            10,
            "StopHold SBIN NSE no position in OpenAlgo's books, the engine's 10",
        ),
    ],
    ids=["nothing since", "the stop of 10 filled since", "all sold by hand"],
)
def test_the_summary_names_openalgos_figure_against_the_engines_units_now(
    seen, engine_now, summary
):
    assert Differing(seen, engine_now).summary(TRADER) == summary


def test_a_leg_with_no_unconfirmed_difference_has_none_at_the_close():
    reconcile = SimpleNamespace(unconfirmed=lambda name: None, unjudged=lambda name: False)
    trader = SimpleNamespace(name="StopHold", open_units=lambda: 10.0)
    assert differing(reconcile, trader) is None


@pytest.mark.parametrize("unjudged", [False, True], ids=["judged", "not judged since a fill"])
def test_the_close_takes_the_engines_units_now_beside_the_sighting(unjudged):
    seen = Sighting(10, 6)
    reconcile = SimpleNamespace(unconfirmed={"StopHold": seen}.get, unjudged=lambda name: unjudged)
    trader = SimpleNamespace(name="StopHold", open_units=lambda: 0.0)
    assert differing(reconcile, trader) == Differing(seen, 0, unjudged)


# ----- which comparisons take a sighting, and which judge the leg (rounds 7 to 9) -------

#: Why the close cannot confirm a leg no read judged after its fill.
WHY = (
    "a fill of its own orders landed during a read of the books and no later read judged "
    "the position"
)
#: The leg the comparisons are of.
LEG = SimpleNamespace(name="StopHold", symbol="SBIN", exchange="NSE", open_units=lambda: 10.0)


def findings() -> Findings:
    """A session's findings, with nothing acted on for real."""
    session = SimpleNamespace(now=lambda: 0, event=lambda *a, **k: None, alert=lambda *a: None)
    corrections = SimpleNamespace(act=lambda *a: "acted", alert_only=lambda *a: "alerted")
    return Findings(session, corrections, lag_alert_after_s=180.0)


def judged(engine: int, broker: int, *, late: bool = False):
    """A comparison that read both books: the engine's and OpenAlgo's units (`late` for a
    leg in flight past the limit)."""

    def compare(found: Findings) -> None:
        kind = Kind.AGREE if engine == broker else Kind.EXPLAINED_FEWER
        position = set() if kind is Kind.AGREE else {(kind, engine, broker)}
        found.judged(LEG, position, LegBooks(broker, (), ()), frozenset(), late=late)

    return compare


def unreadable(found: Findings) -> None:
    """A comparison whose position book could not be read."""
    found.judged(LEG, set(), LegBooks(None, (), ()), frozenset(), late=False)


def fill_during_the_read(found: Findings) -> None:
    """A read whose trade book carried a fill of the leg's own orders: not judged."""
    found.fill_during_read(LEG.name)


@pytest.mark.parametrize(
    ("comparisons", "sighting"),
    [
        ([judged(10, 6)], Sighting(10, 6)),
        ([judged(10, 6), judged(10, 6)], None),
        ([judged(10, 6), judged(10, 10)], None),
        ([judged(10, 6), unreadable], Sighting(10, 6)),
        ([judged(10, 0, late=True)], None),
        ([judged(10, 6), judged(10, 0, late=True)], Sighting(10, 6)),
        ([fill_during_the_read], None),
        ([judged(10, 6), fill_during_the_read], Sighting(10, 6)),
        ([judged(10, 6), fill_during_the_read, judged(0, -4)], Sighting(0, -4)),
    ],
    ids=[
        "a difference first seen",
        "seen twice: acted on",
        "then agreeing",
        "then unreadable: kept",
        "a late comparison takes none",
        "a late comparison keeps an earlier one",
        "a fill during the read takes none",
        "a fill during the read keeps an earlier one",
        "the read after the fill judges the leg",
    ],
)
def test_a_sighting_is_taken_only_by_a_comparison_that_judges_the_position(comparisons, sighting):
    found = findings()
    for compare in comparisons:
        compare(found)
    assert found.unconfirmed(LEG.name) == sighting


@pytest.mark.parametrize(
    ("comparisons", "unjudged"),
    [
        ([judged(10, 6)], False),
        ([fill_during_the_read], True),
        ([fill_during_the_read, judged(0, 0)], False),
        ([fill_during_the_read, judged(0, -4)], False),
        ([fill_during_the_read, unreadable], True),
        ([fill_during_the_read, judged(0, 0, late=True)], True),
        ([judged(10, 6), fill_during_the_read, fill_during_the_read], True),
    ],
    ids=[
        "judged",
        "a fill during the read",
        "then a read that agrees",
        "then a read that differs",
        "then an unreadable read",
        "then a late comparison",
        "fills during two reads",
    ],
)
def test_a_leg_is_unjudged_from_a_fill_during_a_read_until_a_comparison_judges_it(
    comparisons, unjudged
):
    found = findings()
    for compare in comparisons:
        compare(found)
    assert found.unjudged(LEG.name) is unjudged


def test_the_words_of_a_sighting_no_read_judged_after_a_fill_say_it_is_not_confirmed():
    assert Differing(Sighting(10, 6), 0, unjudged=True).words() == (
        "OpenAlgo's books showed 6 units against the engine's 10 at the last read to judge "
        "it; the engine's own fills (10 units sold) came after it, which would leave -4 "
        "units in OpenAlgo's books against the engine's 0, and the session ended before "
        f"they agreed: nothing was sent for the difference; not confirmed, since {WHY}"
    )


def test_the_summary_of_a_sighting_no_read_judged_after_a_fill_says_it_is_not_confirmed():
    assert Differing(Sighting(10, 6), 0, unjudged=True).summary(TRADER) == (
        "StopHold SBIN NSE -4 units in OpenAlgo's books by the last read to judge it and the "
        f"engine's own fills since, the engine's 0, not confirmed, since {WHY}"
    )


@pytest.mark.parametrize(
    ("engine", "summary"),
    [
        (0, f"StopHold SBIN NSE the engine's 0 units, not confirmed, since {WHY}"),
        (6, f"StopHold SBIN NSE the engine's 6 units, not confirmed, since {WHY}"),
    ],
    ids=["flat", "6 left"],
)
def test_the_summary_of_a_leg_no_read_judged_after_a_fill_names_the_engines_units(engine, summary):
    assert not_confirmed_summary(TRADER, engine) == summary


@pytest.mark.parametrize(
    ("engine", "differs", "figures"),
    [
        (0, None, "the engine holds no position"),
        (6, None, "the engine holds 6 units"),
        (
            0,
            Differing(Sighting(10, 6), 0, unjudged=True),
            "the engine holds no position against -4 units in OpenAlgo's books by the last "
            "read to judge it and the engine's own fills since",
        ),
        (
            10,
            Differing(Sighting(10, 6), 10, unjudged=True),
            "the engine holds 10 units against 6 units in OpenAlgo's books by the last read to "
            "judge it",
        ),
    ],
    ids=["flat", "6 left", "a sighting and a fill since", "a sighting and no fill since"],
)
def test_the_alert_of_a_leg_no_read_judged_after_a_fill_asks_to_check_the_position_book(
    engine, differs, figures
):
    assert not_confirmed_alert(TRADER, engine, differs) == (
        f"StopHold SBIN NSE: {figures}, not confirmed, since {WHY}: check OpenAlgo's position book"
    )


# ----- a member of a shared instrument (18-multi-strategy 4.11) ---------------------------

#: The group of SBIN as a comparison judged it: A and B holding 150 together; B holds 50.
GROUP = GroupFigure(("A", "B"), 150)
SHARED = Sighting(50, 90, GROUP)


@pytest.mark.parametrize(
    ("seen", "group_now", "broker_now", "listed"),
    [
        (SHARED, 150, 90, 50),
        (SHARED, 50, -10, 50),
        (Sighting(50, 0, GROUP), 150, 0, 0),
    ],
    ids=[
        "60 sold by hand, nothing since",
        "60 sold, A's stop of 100 filled since",
        "all sold by hand: nothing to close",
    ],
)
def test_a_members_figure_is_its_groups_net_and_it_is_listed_at_its_own_units(
    seen, group_now, broker_now, listed
):
    differs = Differing(seen, 50, group_units=group_now, symbol="SBIN")
    assert (differs.broker_units, differs.open_units) == (broker_now, listed)


def test_a_members_words_name_its_groups_net_against_the_members_sum():
    assert Differing(SHARED, 50, group_units=150, symbol="SBIN").words() == (
        "OpenAlgo's books showed 90 units of SBIN for A and B together, who held 150, at the "
        "last read, and the session ended before they agreed: nothing was sent for the "
        "difference"
    )


def test_a_members_words_name_every_members_fills_since_the_read():
    assert Differing(SHARED, 50, group_units=50, symbol="SBIN").words() == (
        "OpenAlgo's books showed 90 units of SBIN for A and B together, who held 150, at the "
        "last read; their own fills (100 units sold) came after it, which would leave -10 "
        "units in OpenAlgo's books against their 50, and the session ended before they "
        "agreed: nothing was sent for the difference"
    )


def test_a_members_summary_names_its_groups_net():
    assert Differing(SHARED, 50, group_units=150, symbol="SBIN").summary(TRADER) == (
        "StopHold SBIN NSE 90 units of SBIN for A and B together in OpenAlgo's books, the "
        "engine's 50"
    )


def test_a_late_members_alert_names_its_groups_net_against_the_members_sum():
    assert late_alert(TRADER, SHARED) == (
        "StopHold SBIN NSE: OpenAlgo's books showed 90 units of SBIN for A and B together "
        "against their 150 at a late read, while an order of the leg was in flight past the "
        "limit, neither carried by the engine's fills since nor confirmed by a later read; "
        "nothing was sent for the difference: check the position and the order book in "
        "OpenAlgo"
    )


def test_the_close_weighs_a_members_sighting_against_its_groups_units_now():
    reconcile = SimpleNamespace(
        unconfirmed={"StopHold": SHARED}.get,
        unjudged=lambda name: False,
        group_units=lambda trader: 150,
    )
    trader = SimpleNamespace(name="StopHold", symbol="SBIN", open_units=lambda: 50.0)
    assert differing(reconcile, trader) == Differing(SHARED, 50, False, 150, "SBIN")


def test_a_comparison_of_a_group_keeps_the_group_with_its_sighting():
    found = findings()
    finding = (Kind.EXPLAINED_SHARED, 10, 6)
    found.judged(LEG, {finding}, LegBooks(6, (), ()), frozenset(), late=False, group=GROUP)
    assert found.unconfirmed(LEG.name) == Sighting(10, 6, GROUP)
