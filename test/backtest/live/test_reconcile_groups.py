"""A shared instrument's rules against the broker's net position, on their own
(`_live/_group_rules.py`, `_live/_reconcile_allocate.py`, `_live/_restart_groups.py`;
18-multi-strategy 4.6 to 4.8 and 4.10, unit M6): the group's kind, the allocation of a
difference other orders' trades explain, what a restart does with the group, and the words
that name it.

The comparison of a session is tested on the fake OpenAlgo (`test_session_shared_*.py`).
Expected values are worked by hand: signed units, long above 0; "A" holds 100 and "B" 50
of SBIN unless a row says otherwise.
"""

from __future__ import annotations

import pytest
from test_reconcile_rules import TABLE_5_3, TABLE_5_5

from openalgo.backtest._live._group_rules import (
    GroupFigure,
    Share,
    and_join,
    group_kind,
    group_restart_kind,
    group_words,
    other_side_alert,
)
from openalgo.backtest._live._groups import InstrumentGroup
from openalgo.backtest._live._reconcile_allocate import allocate, allocation_words
from openalgo.backtest._live._reconcile_rules import (
    Kind,
    RestartKind,
    Trade,
    adoption_price,
    unexplained_by,
)
from openalgo.backtest._live._restart_groups import judge_group

SBIN = InstrumentGroup("SBIN", "NSE", "MIS", ("A", "B"))
#: When each position opened: A at 10:05 IST, B at 11:20 IST (UTC nanoseconds, any day).
TEN_FIVE, ELEVEN_TWENTY = 16_500 * 10**9, 21_000 * 10**9


def sold(units: int, price: float = 499.0, order_id: str = "h1") -> Trade:
    """Another order's sale of `units` at `price`."""
    return Trade(order_id, "SELL", units, price)


def bought(units: int, price: float = 501.0, order_id: str = "h1") -> Trade:
    """Another order's purchase of `units` at `price`."""
    return Trade(order_id, "BUY", units, price)


@pytest.mark.parametrize(
    ("members", "broker", "unexplained", "kind"),
    [
        ((100, 50), 150, 0, Kind.AGREE),
        ((100, 50), None, 0, Kind.UNREADABLE),
        ((100, 50), 160, 10, Kind.MORE),
        ((100, 50), 160, 0, Kind.MORE),
        ((0, 0), 30, 30, Kind.MORE),
        ((0, 0), -30, -30, Kind.MORE),
        ((-100, -50), -160, -10, Kind.MORE),
        ((100, 50), -50, 0, Kind.OTHER_SIDE),
        ((100, 50), -30, -180, Kind.OTHER_SIDE),
        ((100, 50), 90, 0, Kind.EXPLAINED_SHARED),
        ((100, 50), 0, 0, Kind.EXPLAINED_SHARED),
        ((-100, -50), -120, 0, Kind.EXPLAINED_SHARED),
        ((100, 50), 90, -60, Kind.UNEXPLAINED),
        ((100, 50), 0, -150, Kind.UNEXPLAINED),
        ((100, -50), 80, 0, Kind.EXPLAINED_SHARED),
        ((100, -50), 120, 0, Kind.UNEXPLAINED),
        ((100, -100), 30, 30, Kind.UNEXPLAINED),
        ((100, -50), -10, 0, Kind.EXPLAINED_SHARED),
        ((100, -50), -110, 0, Kind.OTHER_SIDE),
    ],
    ids=[
        "agree",
        "unreadable",
        "a buy by hand on an all-long group: an excess",
        "an excess the trades explain is still an excess",
        "an excess of a flat group",
        "a short excess of a flat group",
        "a sale by hand on an all-short group: an excess",
        "a sale beyond the members, explained: the other side",
        "the other side, unexplained",
        "fewer, explained: absorbed",
        "flat, explained: absorbed",
        "a buy by hand on a short group: absorbed",
        "fewer, unexplained",
        "flat, unexplained",
        "mixed sides: a buy the short member absorbs",
        "mixed sides: a buy beyond the short member",
        "mixed sides at no net: more, with a short member to close",
        "mixed sides: a sale the long member absorbs, though the net turns short",
        "mixed sides: a sale beyond the long member, the net short: the other side",
    ],
)
def test_the_groups_kind_judges_the_members_sum_by_the_rows_of_4_6(
    members, broker, unexplained, kind
):
    assert group_kind(members, broker, unexplained) is kind


@pytest.mark.parametrize(("engine", "broker", "external", "adopted", "kind"), TABLE_5_3)
def test_a_group_of_one_is_judged_by_the_table_of_5_3_as_a_leg(
    engine, broker, external, adopted, kind
):
    # A group of one member is 16's rows exactly (4.6): an explained difference, row 1 or
    # 2, is the group's explained one, which the allocation gives that member whole.
    unexplained = 0 if broker is None else unexplained_by(engine, broker, external, adopted)
    explained = (Kind.EXPLAINED_FLAT, Kind.EXPLAINED_FEWER)
    expected = Kind.EXPLAINED_SHARED if kind in explained else kind
    assert group_kind((engine,), broker, unexplained) is expected


@pytest.mark.parametrize(("journal", "broker", "unexplained", "verdict"), TABLE_5_5)
def test_a_restart_of_a_group_of_one_is_judged_by_the_table_of_5_5_as_a_leg(
    journal, broker, unexplained, verdict
):
    assert group_restart_kind((journal,), broker, unexplained) is verdict


@pytest.mark.parametrize(
    ("engine", "broker", "trades"),
    [
        (10, 0, [sold(10, 498.0)]),
        (10, 4, [sold(6, 498.0)]),
        (-10, -4, [bought(6, 502.0)]),
        (10, 4, [sold(5, 499.0, "h1"), bought(2, 501.0, "h2"), sold(3, 496.0, "h3")]),
    ],
    ids=["flat", "fewer", "fewer short", "the newest trades first"],
)
def test_a_group_of_one_adopts_an_explained_difference_as_a_leg_does(engine, broker, trades):
    # The leg adopts the whole difference at the newest of the other orders' trades on its
    # side (5.3 rows 1 and 2): the allocation gives a member alone the same units and price.
    side = "SELL" if broker < engine else "BUY"
    (share,) = allocate([("A", engine, None)], broker - engine, trades)
    assert (share.taken, share.left) == (broker - engine, broker)
    assert share.price == pytest.approx(adoption_price(trades, abs(broker - engine), side))


@pytest.mark.parametrize(
    ("members", "broker", "unexplained", "verdict"),
    [
        ((100, 50), 150, 0, RestartKind.RESUME),
        ((100, 0), 100, 0, RestartKind.RESUME),
        ((0, 0), 0, 0, RestartKind.RESUME),
        ((0, 0), 30, 30, RestartKind.UNKNOWN_POSITION),
        ((100, 50), 0, -150, RestartKind.UNEXPLAINED_FLAT),
        ((100, 50), 0, 0, RestartKind.DIFFERS),
        ((100, 50), 90, 0, RestartKind.DIFFERS),
        ((100, 50), 90, -60, RestartKind.DIFFERS),
        ((100, 0), 150, 50, RestartKind.DIFFERS),
        ((100, 50), -30, -180, RestartKind.DIFFERS),
    ],
    ids=[
        "the sum agrees",
        "one member flat, the sum agrees",
        "every member flat, the broker flat",
        "every member flat, the broker holds: refused",
        "the broker flat, nothing explains it: refused",
        "the broker flat, explained",
        "fewer, explained",
        "fewer, unexplained",
        "one member flat, the broker holds more",
        "the other side",
    ],
)
def test_a_restart_judges_the_members_sum_against_the_brokers_net(
    members, broker, unexplained, verdict
):
    assert group_restart_kind(members, broker, unexplained) is verdict


def test_a_restart_of_a_group_whose_sum_agrees_resumes_with_nothing_to_act_on():
    start = judge_group(SBIN, {"A": 100, "B": 50}, 150, (), 0)
    assert (start.restart, start.kind, start.refusal) == (RestartKind.RESUME, Kind.AGREE, None)
    assert start.figure == GroupFigure(("A", "B"), 150, (), (100, 50), 0)


def test_a_restart_of_a_group_names_the_instrument_and_every_member_when_it_refuses():
    unknown = judge_group(SBIN, {"A": 0, "B": 0}, 30, (bought(30),), 0)
    assert unknown.refusal == (
        "OpenAlgo's books show 30 units of SBIN NSE, which A and B trade, and the journal "
        "holds none for them: close them or start with another symbol"
    )
    # Sold by hand: the trades of other orders explain the broker's flat book.
    closed = judge_group(SBIN, {"A": 100, "B": 50}, 0, (sold(150),), 0)
    assert closed.refusal is None
    gone = judge_group(SBIN, {"A": 100, "B": 50}, 0, (), 0)
    assert gone.refusal == (
        "the journal holds 150 units of SBIN NSE for A and B together (A 100, B 50), "
        "OpenAlgo's books show none and no trade explains it: check the account in OpenAlgo"
    )


def test_a_restart_of_a_group_after_a_sale_by_hand_shares_it_out_by_the_saved_openings():
    # 60 sold by hand while the session was down: the other orders' trades explain it, and
    # B, whose saved position opened last, takes first.
    opened = {"A": TEN_FIVE, "B": ELEVEN_TWENTY}
    start = judge_group(SBIN, {"A": 100, "B": 50}, 90, (sold(60),), 0, opened)
    assert (start.restart, start.kind) == (RestartKind.DIFFERS, Kind.EXPLAINED_SHARED)
    assert start.figure.shares == (Share("B", 50, -50, 499.0), Share("A", 100, -10, 499.0))
    # The members' units as judged, and the 60 sold by hand not taken in yet.
    assert (start.figure.held, start.figure.external) == ((100, 50), -60)


def test_a_restart_of_a_group_whose_openings_are_not_saved_shares_out_by_call_order():
    start = judge_group(SBIN, {"A": 100, "B": 50}, 90, (sold(60),), 0)
    assert start.figure.shares == (Share("A", 100, -60, 499.0), Share("B", 50, 0))


def test_a_member_moved_away_is_left_out_of_its_groups_sum():
    start = judge_group(SBIN, {"A": 100}, 100, (), 0)
    figure = GroupFigure(("A",), 100, (), (100,), 0)
    assert (start.restart, start.figure) == (RestartKind.RESUME, figure)


@pytest.mark.parametrize(
    ("names", "words"),
    [((), ""), (("A",), "A"), (("A", "B"), "A and B"), (("A", "B", "C"), "A, B and C")],
)
def test_names_are_joined_in_words(names, words):
    assert and_join(names) == words


def test_a_figure_names_the_other_members_in_call_order():
    assert GroupFigure(("A", "B", "C"), 0).others("B") == ("A", "C")


# ----- the allocation of an explained difference (4.7) ------------------------------------


@pytest.mark.parametrize(
    ("holders", "difference", "external", "shares"),
    [
        (
            [("A", 100, TEN_FIVE), ("B", 50, ELEVEN_TWENTY)],
            -60,
            [sold(60)],
            (Share("B", 50, -50, 499.0), Share("A", 100, -10, 499.0)),
        ),
        (
            [("A", 100, TEN_FIVE), ("B", 50, TEN_FIVE)],
            -60,
            [sold(60)],
            (Share("A", 100, -60, 499.0), Share("B", 50, 0)),
        ),
        (
            [("A", 100, TEN_FIVE), ("B", 50, ELEVEN_TWENTY)],
            -150,
            [sold(150)],
            (Share("B", 50, -50, 499.0), Share("A", 100, -100, 499.0)),
        ),
        (
            [("A", -100, TEN_FIVE), ("B", -50, ELEVEN_TWENTY)],
            30,
            [bought(30)],
            (Share("B", -50, 30, 501.0), Share("A", -100, 0)),
        ),
        (
            [("A", 100, ELEVEN_TWENTY), ("B", 50, None)],
            -60,
            [sold(60)],
            (Share("B", 50, -50, 499.0), Share("A", 100, -10, 499.0)),
        ),
        (
            [("A", 100, TEN_FIVE), ("B", 0, None)],
            -60,
            [sold(60)],
            (Share("A", 100, -60, 499.0), Share("B", 0, 0)),
        ),
    ],
    ids=[
        "the position opened last takes first",
        "positions opened together: call order",
        "all of the group's units sold",
        "a buy by hand on a short group",
        "an opening not known counts as the latest",
        "a flat member takes nothing",
    ],
)
def test_an_explained_difference_is_shared_out_the_latest_opened_position_first(
    holders, difference, external, shares
):
    assert allocate(holders, difference, external) == shares


def test_the_newest_of_the_other_orders_trades_price_the_first_share():
    # 60 sold by hand at 499.00, then 40 at 498.00 (and a purchase, on the other side): B,
    # opened last, takes the newest 50 (40 at 498.00, 10 at 499.00), A the 50 left at 499.00.
    trades = [sold(60, 499.0, "h1"), bought(10, 502.0, "h2"), sold(40, 498.0, "h3")]
    shares = allocate([("A", 100, TEN_FIVE), ("B", 50, ELEVEN_TWENTY)], -100, trades)
    assert [(s.name, s.taken) for s in shares] == [("B", -50), ("A", -50)]
    assert shares[0].price == pytest.approx(498.2)
    assert shares[1].price == pytest.approx(499.0)


def test_a_share_beyond_the_other_orders_trades_is_priced_at_the_newest_of_them():
    # Only 30 sold units are in the trade book (the rest adopted on an earlier day's start):
    # B's 50 averages the 30 there, and A's 10 is priced from the newest again.
    shares = allocate([("A", 100, TEN_FIVE), ("B", 50, ELEVEN_TWENTY)], -60, [sold(30, 497.0)])
    assert [(s.name, s.taken, s.price) for s in shares] == [("B", -50, 497.0), ("A", -10, 497.0)]


def test_the_allocation_names_each_share_and_what_it_leaves():
    figure = GroupFigure(("A", "B"), 150, (Share("B", 50, -50, 499.0), Share("A", 100, -10, 499.0)))
    assert allocation_words(figure, "A") == (
        "60 units sold by other orders are shared out, the most recently opened position "
        "first: taken from B 50 (flat) and A 10 (90 left); each takes its share as its own, "
        "its stop cut down to what it is left with, or cancelled"
    )


def test_the_allocation_tells_a_member_it_takes_nothing_from_that_it_keeps_its_units():
    figure = GroupFigure(("A", "B"), 150, (Share("A", 100, -60, 499.0), Share("B", 50, 0)))
    assert "taken from A 60 (40 left); B keeps its 50 units;" in allocation_words(figure, "B")


def test_a_figure_gives_each_members_share_and_none_for_a_member_without_one():
    figure = GroupFigure(("A", "B"), 150, (Share("A", 100, -60, 499.0),))
    assert (figure.share("A"), figure.share("B")) == (Share("A", 100, -60, 499.0), None)
    assert Share("A", 100, -60).left == 40


def test_the_allocation_tells_a_flat_member_it_takes_none():
    figure = GroupFigure(("A", "Flat"), 100, (Share("A", 100, -60, 499.0), Share("Flat", 0, 0)))
    words = allocation_words(figure, "Flat")
    assert "taken from A 60 (40 left); Flat is flat and takes none;" in words
    assert "keeps its 0 units" not in words


def test_a_figure_names_each_members_units_and_the_allocation_made():
    shares = (Share("B", 50, -50, 499.0), Share("A", 100, -10, 499.0), Share("C", 0, 0))
    figure = GroupFigure(("A", "B", "C"), 150, shares, (100, 50, 0), -60)
    assert figure.member_units() == "A 100, B 50, C 0"
    assert figure.allocation() == "B -50 to 0, A -10 to 90"
    assert GroupFigure(("A", "B"), 150).member_units() == ""
    assert GroupFigure(("A", "B"), 150).allocation() == ""


# ----- the excess and the other side (4.6) ------------------------------------------------


def test_an_excess_says_the_units_beyond_the_members_are_no_strategys_to_close_after_the_end():
    words = group_words(Kind.MORE, GroupFigure(("A", "B"), 150), 160)
    assert words == (
        "the 10 units beyond what they hold are no strategy's of the session: nothing is sent "
        "for them and they have no stop at the broker. End the session, then close them by "
        "hand in OpenAlgo"
    )


def test_the_other_side_says_each_members_stop_is_cancelled_and_nothing_adopted():
    words = group_words(Kind.OTHER_SIDE, GroupFigure(("A", "B"), 150), -50)
    assert words == (
        "the broker holds the other side of what they hold together: nothing is adopted, each "
        "member's stop is cancelled and nothing else is sent. End the session, then close the "
        "position by hand in OpenAlgo"
    )


def test_the_other_sides_error_alert_names_the_instrument_its_members_and_both_figures():
    text = other_side_alert("SBIN", "NSE", GroupFigure(("A", "B"), 150), -50)
    assert text == (
        "SBIN NSE: OpenAlgo's books show -50 units, the other side of the 150 units A and B "
        "hold together: end the session, then close the position by hand in OpenAlgo"
    )


def test_only_an_excess_and_the_other_side_have_group_words():
    with pytest.raises(KeyError):
        group_words(Kind.UNEXPLAINED, GroupFigure(("A", "B"), 150), 90)
