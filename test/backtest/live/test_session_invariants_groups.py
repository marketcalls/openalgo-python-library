"""The invariants of instrument groups on order changes and runs written by hand
(`session_invariants_groups`, 18-multi-strategy 5.1, unit M6): I4 the sum rule, I5
isolation and I6 one side each find what they must and nothing more; on a session run on
the fake OpenAlgo (`session_hooks`), I5's hooks note a member made to act while another
member's order word is applied, and an order word handed to a member it is not bound to;
and a run whose members' books the trace could read at none of its changes fails I4 and
I6 rather than pass them unjudged.

The session's client is "s", another client of the account "o"; "A" and "B" trade SBIN in
one session. Each change carries the members' books as the trace samples them: a
member's kernel units, the units it adopted, and what each of its bound orders' fill
ledgers gave its kernel.
"""

from __future__ import annotations

import pytest
import session_trace
from fake_openalgo.fixtures import *  # noqa: F403
from session_invariants_groups import isolation, one_side, sampled, sum_rule
from session_trace import ISOLATION, ONE_SIDE, SUM_RULE, Member, OrderChange, Run
from shared_samples import SbinBreakout, SbinTrend
from stops_samples import stop_world

from openalgo.backtest._live._session_news import News
from openalgo.backtest._live._types import Mode
from openalgo.backtest._live.stops import StopsSupervisor
from openalgo.backtest.errors import LiveError

MINE = frozenset({"s"})


def change(
    seq: int,
    order_id: str,
    strategy: str,
    action: str,
    *,
    filled: int,
    fill: int,
    position: int,
    members: tuple[Member, ...],
    client: str = "s",
) -> OrderChange:
    """A MARKET order of SBIN as it stood after a change, the fake's position after it and
    the members' books then."""
    status = "complete" if filled else "open"
    return OrderChange(
        seq, seq, Mode.LIVE, order_id, client, strategy, "SBIN", "NSE", action, "MARKET",
        status, filled or 100, filled, fill, 500.0 if fill else 0.0, position, members, MINE,
    )  # fmt: skip


def member(strategy: str, units: int, *given: tuple[str, int], adopted: int = 0) -> Member:
    return Member(strategy, units, adopted, tuple(given))


#: A buys 100 (order "a1"), then B buys 50 (order "b1"); each kernel took its own fill.
A_BOUGHT = change(
    0, "a1", "A", "BUY", filled=100, fill=100, position=100,
    members=(member("A", 100, ("a1", 100)), member("B", 0)),
)  # fmt: skip
B_BOUGHT = change(
    1, "b1", "B", "BUY", filled=50, fill=50, position=150,
    members=(member("A", 100, ("a1", 100)), member("B", 50, ("b1", 50))),
)  # fmt: skip


def run(*strayed: str) -> Run:
    return Run(session=None, clients=MINE, strayed=list(strayed))


# ----- I4, the sum rule ------------------------------------------------------------------


def test_the_sum_rule_holds_when_each_kernel_took_its_own_fill():
    sum_rule([A_BOUGHT, B_BOUGHT], MINE)


def test_the_sum_rule_counts_a_fill_the_kernel_has_not_been_given_yet():
    waiting = change(
        1, "b1", "B", "BUY", filled=50, fill=50, position=150,
        members=(member("A", 100, ("a1", 100)), member("B", 0, ("b1", 0))),
    )  # fmt: skip
    sum_rule([A_BOUGHT, waiting], MINE)


def test_the_sum_rule_finds_a_fill_given_to_two_kernels():
    # B's fill reached A's kernel as well as B's: the members hold 200 against 150.
    twice = change(
        1, "b1", "B", "BUY", filled=50, fill=50, position=150,
        members=(member("A", 150, ("a1", 100)), member("B", 50, ("b1", 50))),
    )  # fmt: skip
    with pytest.raises(AssertionError, match="I4"):
        sum_rule([A_BOUGHT, twice], MINE)


def test_the_sum_rule_counts_another_clients_sale_until_a_member_adopts_it():
    sale = change(
        2, "h1", "manual", "SELL", filled=60, fill=60, position=90, client="o",
        members=(member("A", 100, ("a1", 100)), member("B", 50, ("b1", 50))),
    )  # fmt: skip
    sum_rule([A_BOUGHT, B_BOUGHT, sale], MINE)


# ----- I6, one side ----------------------------------------------------------------------


def test_one_side_holds_while_the_members_are_long_or_flat():
    one_side([A_BOUGHT, B_BOUGHT])


def test_one_side_finds_two_members_on_opposite_sides():
    opposite = change(
        1, "b1", "B", "SELL", filled=50, fill=50, position=50,
        members=(member("A", 100, ("a1", 100)), member("B", -50, ("b1", 50))),
    )  # fmt: skip
    with pytest.raises(AssertionError, match="I6"):
        one_side([A_BOUGHT, opposite])


# ----- I5, isolation ---------------------------------------------------------------------


def test_isolation_holds_when_each_member_binds_its_own_orders():
    isolation(run(), [A_BOUGHT, B_BOUGHT], MINE)


def test_isolation_finds_a_word_a_kernel_took_of_another_members_order():
    with pytest.raises(AssertionError, match="I5"):
        isolation(run("A took order b1, recorded for B"), [A_BOUGHT, B_BOUGHT], MINE)


def test_isolation_finds_an_order_bound_in_two_members_venue_maps():
    both = change(
        1, "b1", "B", "BUY", filled=50, fill=50, position=150,
        members=(member("A", 100, ("a1", 100), ("b1", 0)), member("B", 50, ("b1", 50))),
    )  # fmt: skip
    with pytest.raises(AssertionError, match="bound to both A and B"):
        isolation(run(), [A_BOUGHT, both], MINE)


def test_isolation_finds_a_member_binding_an_order_the_fake_records_for_another():
    # A's venue map binds B's order and B's binds none: no order is bound twice, but the
    # fake records "b1" for B.
    taken = change(
        1, "b1", "B", "BUY", filled=50, fill=50, position=150,
        members=(member("A", 150, ("a1", 100), ("b1", 50)), member("B", 0)),
    )  # fmt: skip
    with pytest.raises(AssertionError, match="which the fake records for B"):
        isolation(run(), [A_BOUGHT, taken], MINE)


def test_isolation_does_not_take_another_clients_order_for_a_members():
    # Another client's order tagged with A's name (18-multi-strategy F115) is bound to no
    # member: it is not A's.
    tagged = change(
        2, "h1", "A", "BUY", filled=10, fill=10, position=160, client="o",
        members=(member("A", 100, ("a1", 100)), member("B", 50, ("b1", 50))),
    )  # fmt: skip
    isolation(run(), [A_BOUGHT, B_BOUGHT, tagged], MINE)


# ----- I5's hooks on a session run on the fake --------------------------------------------


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path):
    return stop_world(make_fake_openalgo, monkeypatch, tmp_path)


def test_the_hooks_note_a_member_made_to_act_while_anothers_order_word_is_applied(
    world, monkeypatch
):
    # A session broken on purpose: each order word of one member asks for the other
    # members' stops to be placed again (a no-op while they work), a cause passed from one
    # member to another.
    world.trace.waive(ISOLATION, "the test breaks isolation and reads what the hooks noted")
    updated = StopsSupervisor.updated

    def leaking(stops, trader, update, result):
        for other in stops._session.traders:
            if other is not trader:
                stops._session.margin.protect(other)
        return updated(stops, trader, update, result)

    monkeypatch.setattr(StopsSupervisor, "updated", leaking)
    world.run([SbinTrend, SbinBreakout])

    strayed = world.trace.runs[-1].strayed
    assert "protect of SbinTrend while a word of SbinBreakout's order was applied" in strayed


def test_the_hooks_note_an_order_word_handed_to_a_member_it_is_not_bound_to(world, monkeypatch):
    # A session broken on purpose: the first word of B's order reaches A's kernel. The
    # hooks note it twice (not bound in A's venue map; recorded for B at the fake) before
    # A's venue map refuses it and the session stops.
    world.trace.waive(ISOLATION, "the test breaks isolation and reads what the hooks noted")
    owner, misrouted = News._owner, []

    def to_the_other_member(news, order_id, intent_id):
        trader = owner(news, order_id, intent_id)
        if trader is not None and trader.name == "SbinBreakout" and not misrouted:
            misrouted.append(order_id)
            return next(t for t in news._session.traders if t.name == "SbinTrend")
        return trader

    monkeypatch.setattr(News, "_owner", to_the_other_member)
    with pytest.raises(LiveError, match="is not bound in this venue map"):
        world.run([SbinTrend, SbinBreakout])

    (order_id,) = misrouted
    assert world.trace.runs[-1].strayed == [
        f"SbinTrend took order {order_id}, not bound to it",
        f"SbinTrend took order {order_id}, recorded for SbinBreakout",
    ]


# ----- I4 and I6 judge only where the members' books were read -----------------------------


def test_a_run_whose_members_books_were_read_at_some_change_passes_the_sample_check():
    sampled(Run(None, MINE, sampled=1, unsampled=4))
    sampled(Run(None, MINE))


def test_a_run_whose_members_books_were_read_at_no_change_fails_the_sample_check():
    with pytest.raises(AssertionError, match="none of the 3 order changes"):
        sampled(Run(None, MINE, unsampled=3))


def test_the_trace_counts_the_changes_whose_members_books_it_could_not_read(world, monkeypatch):
    # Every read of a member's books fails: no change is sampled, so I4 and I6 judge
    # nothing, which the sample check finds.
    for check in (SUM_RULE, ONE_SIDE):
        world.trace.waive(check, "the test reads the sample check itself")

    def unreadable(session, trader):
        raise RuntimeError("the kernel cannot be read")

    monkeypatch.setattr(session_trace, "_member", unreadable)
    world.run([SbinTrend, SbinBreakout])
    run = world.trace.runs[-1]
    assert (run.sampled, run.unsampled > 0) == (0, True)
    with pytest.raises(AssertionError, match="I4 and I6"):
        sampled(run)


def test_a_session_run_samples_the_members_books_at_its_changes(world):
    world.run([SbinTrend, SbinBreakout])
    run = world.trace.runs[-1]
    assert (run.sampled > 0, run.unsampled) == (True, 0)
