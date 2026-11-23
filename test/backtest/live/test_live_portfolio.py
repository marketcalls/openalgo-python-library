"""`SessionResult.portfolio`: a live session's strategies as one account, a `bt.PortfolioResult`
(`_live/_result_portfolio.py`, `session_result.py`; 18-multi-strategy 1.3 and 4.11, unit M5).

The market and the scripts are `session_samples`' and `watch_samples`': the session starts
at 14:40:05 IST on Monday 2026-11-16, each symbol trades 10 units every 10 s at 500.00.
In the account's session, SBIN and INFY fall to 492.00 from 14:43:40; `HoldSbin` and
`HoldInfy` (capital Rs 1,00,000 each) buy 10 units at 500.00 at their first decision
(14:42:01), and the account's daily loss limit of Rs 150 fires the kill switch at the
first trade at 492.00, which sells both at 492.00: -80.00 each, with no charges modelled
(`costs=None`); Ctrl+C at 14:47. In the capped session (test_session_daily_limits) the
account's trade cap is 1 entry: HoldSbin, deciding first, buys SBIN at 14:42:01, and
HoldInfy finds the account at its cap at each of its decisions to the Ctrl+C at 14:47.
Expected values are worked from the scenario and the fake's books, never from the
portfolio's code.
"""

from __future__ import annotations

import html
import logging

import pandas as pd
import pytest
from fake_openalgo import LIVE_CASH
from fake_openalgo.fixtures import *  # noqa: F403
from live_samples import live, seam
from session_samples import CAPITAL, BuyAndHold, StopFlag, ist, rig
from shared_samples import A_UNITS, B_UNITS, SbinBreakout, SbinTrend
from stops_samples import at_price
from test_session_figures import DAY_ENDS, TwoRoundTrips
from watch_samples import HoldInfy, HoldSbin

import openalgo.backtest as bt
from openalgo.backtest import metrics
from openalgo.backtest._live._result_portfolio import account_rows, restart_scope
from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest._portfolio_tables import REFUSED_COLUMNS

IST = "Asia/Kolkata"

#: The account's daily loss limit: at or past it once both hold 10 units at 492.00.
ACCOUNT_LIMIT = 150.0
FALL_AT = ist(14, 43, 40)
#: Each strategy's loss: 10 units bought at 500.00 and sold at 492.00.
LOSS_EACH = 10 * (492.0 - 500.0)


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    made = rig(make_fake_openalgo, monkeypatch, tmp_path)
    seam(made, monkeypatch)
    return made


@pytest.fixture
def account(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    """The account's session (module docstring) through `bt.live`: its result."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    made = rig(make_fake_openalgo, monkeypatch, tmp_path, symbols=("SBIN", "INFY"))
    made.script.price = lambda now_ns: 492.0 if now_ns >= FALL_AT else 500.0
    made.stop_at(ist(14, 47))
    seam(made, monkeypatch)
    return live(made, [HoldSbin, HoldInfy], account_max_daily_loss=ACCOUNT_LIMIT, costs=None)


@pytest.fixture
def capped(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    """The capped session (module docstring) through `bt.live`: its result."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    made = rig(make_fake_openalgo, monkeypatch, tmp_path, symbols=("SBIN", "INFY"))
    made.stop_at(ist(14, 47))
    seam(made, monkeypatch)
    return live(made, [HoldSbin, HoldInfy], account_max_trades_per_day=1)


def as_of_sum(results, column: str, before: dict[str, float]) -> pd.Series:
    """Each strategy's `column` of its equity rows at its last row at or before every
    instant one of them has (`before[name]` before its first), added up."""
    frames = {name: result.equity[column] for name, result in results.items()}
    times = sorted(set().union(*(frame.index for frame in frames.values())))
    aligned = pd.DataFrame({name: frame.reindex(times).ffill() for name, frame in frames.items()})
    return aligned.fillna(before).sum(axis=1)


# ----- the account's rule, on rows worked by hand ------------------------------------------------


class Rows:
    """What the account reads of a strategy's `Result`: its equity rows by IST time and its
    manifest's capital and risk basis."""

    def __init__(self, capital: float, basis: str, rows: dict[str, tuple]) -> None:
        columns = ["equity", "realized", "unrealized", "costs", "margin"]
        index = pd.DatetimeIndex([pd.Timestamp(f"2026-11-16 {t}", tz=IST) for t in rows])
        self.equity = pd.DataFrame(
            list(rows.values()), index=index, columns=[f"{c}_rupees" for c in columns]
        )
        self.manifest = {"arguments": {"capital": capital, "risk": {"risk_basis": basis}}}


def test_the_accounts_rows_add_each_strategys_figures_at_its_last_close():
    # A (Rs 1,00,000, net basis) closes slices at 09:16, 09:17 and 09:18; B (Rs 50,000,
    # gross basis) at 09:17 only, so before it B holds its capital, all free. A's measure
    # on the net basis takes its charges off: -5 - 5 = -10 at 09:17, -10 - 6 = -16 at
    # 09:18; B's is its realized 20.
    a = Rows(
        100_000,
        "net",
        {
            "09:16": (100_000.0, 0.0, 0.0, 0.0, 0.0),
            "09:17": (99_990.0, 0.0, -5.0, 5.0, 2_000.0),
            "09:18": (99_984.0, -10.0, 0.0, 6.0, 0.0),
        },
    )
    b = Rows(50_000, "gross", {"09:17": (50_020.0, 20.0, 0.0, 0.0, 0.0)})
    rows = account_rows({"A": a, "B": b})
    # 09:16 IST is 03:46 UTC.
    assert rows["ts"][0] == pd.Timestamp("2026-11-16 03:46", tz="UTC").value
    figures = ("equity_rupees", "free_cash_rupees", "margin_rupees", "daily_loss_rupees")
    assert [list(rows[name]) for name in figures] == [
        [150_000.0, 150_010.0, 150_004.0],
        [150_000.0, 148_010.0, 150_004.0],
        [0.0, 2_000.0, 0.0],
        [0.0, 10.0, 4.0],
    ]


# ----- the strategies ------------------------------------------------------------------------


def test_the_portfolio_is_a_portfolio_result_of_the_sessions_strategies(account):
    pf = account.portfolio
    assert isinstance(pf, bt.PortfolioResult)
    assert pf.strategies == account.strategies == ("HoldSbin", "HoldInfy")


def test_each_strategy_of_the_portfolio_is_the_sessions_own_result(account):
    # The same objects: the portfolio covers what the results cover, this start.
    pf = account.portfolio
    assert all(pf.strategy(name) is account.results[name] for name in account.strategies)


# ----- the account -------------------------------------------------------------------------------


def test_the_account_is_the_sum_of_the_capitals_and_ends_with_both_losses(account):
    rows = account.portfolio.account
    assert rows["equity_rupees"].iloc[0] == 2 * CAPITAL
    assert rows["equity_rupees"].iloc[-1] == 2 * CAPITAL + 2 * LOSS_EACH
    assert rows.iloc[-1][["free_cash_rupees", "margin_rupees"]].tolist() == [
        2 * CAPITAL + 2 * LOSS_EACH,
        0.0,
    ]


def test_the_accounts_equity_is_the_combined_equity_of_its_strategies(account):
    # Fixed allocations: every strategy's equity at its last slice close added up, as
    # metrics.combine_equity adds them with each strategy's capital.
    pf = account.portfolio
    equities = {name: result.equity["equity_rupees"] for name, result in pf.results.items()}
    combined = metrics.combine_equity(equities, opening_wealth=dict.fromkeys(equities, CAPITAL))
    assert pf.equity.index.tolist() == combined.index.tolist()
    assert pf.equity.tolist() == pytest.approx(combined.tolist(), abs=0.005)


def test_the_accounts_margin_is_the_margin_its_strategies_block(account):
    pf = account.portfolio
    held = as_of_sum(pf.results, "margin_rupees", dict.fromkeys(pf.strategies, 0.0))
    assert pf.account["margin_rupees"].tolist() == pytest.approx(held.tolist(), abs=0.005)
    assert pf.account["margin_rupees"].max() > 0


def test_the_accounts_daily_loss_measure_ends_at_both_losses_together(account):
    # Both were sold before the 14:43 bar closed: from then on the measure is the two
    # losses realized, -160.00, the most the account lost.
    measure = account.portfolio.account["daily_loss_rupees"]
    assert measure.iloc[-1] == 2 * LOSS_EACH
    assert measure.min() == 2 * LOSS_EACH


def test_the_kill_switchs_lock_leaves_its_refusals_in_the_strategies_orders(account):
    # Flat and locked from 14:43:40, each strategy is refused at its later decisions by
    # the lock, live_lock, which its own orders hold: the account refused nothing itself.
    refused = account.portfolio.refused
    assert list(refused.columns) == list(REFUSED_COLUMNS)
    assert refused.empty
    for result in account.results.values():
        assert set(result.orders["reject_code"].dropna()) == {"live_lock"}


#: The card of the account's refusals in the account's session, as the page writes it: the
#: session set the account's daily loss limit alone, no trade cap and no profit target.
LOCKED_OUT_CARD = (
    "The account refused no order; the entries the account&#x27;s daily loss limit locked "
    "out are in the strategies&#x27; orders with the code live_lock, and the working entries "
    "and adds its kill switch cancelled are in their orders as cancelled."
)


def refused_card(page: str) -> str:
    """The words of the empty card of the account's refusals on a combined page."""
    after = page.split("Refused by the Account", 1)[1]
    return after.split("<p class='empty'>", 1)[1].split("</p>", 1)[0]


def test_the_empty_refused_card_after_the_kill_switch_says_where_its_refusals_are(account):
    # The account refused nothing itself, but its loss limit fired the kill switch: the
    # card must not read as if the account had refused and ended nothing, nor name a trade
    # cap or a profit target the session did not set.
    assert refused_card(account.portfolio.tearsheet()) == LOCKED_OUT_CARD


def test_the_refused_cards_words_are_not_an_assumptions_line_too(account):
    page = account.portfolio.tearsheet()
    assert page.count("locked out are in the strategies&#x27; orders") == 1


def test_a_session_with_no_account_limit_refuses_nothing_in_a_backtests_words(world):
    page = live(world, BuyAndHold).portfolio.tearsheet()
    assert refused_card(page) == "The account refused no order and ended none."


def test_the_account_note_names_the_kill_switchs_cancels(account):
    note = account.portfolio.assumptions["account"]
    assert "the working entries and adds the kill switch cancelled" in note


def test_the_entries_the_accounts_trade_cap_refused_are_listed_by_strategy(capped):
    # HoldInfy's five entries, decided on the bars opening 14:41 to 14:45, its orders 1 to 5.
    refused = capped.portfolio.refused
    opens = [pd.Timestamp(f"2026-11-16 14:{minute}", tz=IST) for minute in range(41, 46)]
    assert refused["strategy"].tolist() == ["HoldInfy"] * 5
    assert refused["order_id"].tolist() == [1, 2, 3, 4, 5]
    assert refused["time"].tolist() == opens
    assert set(zip(refused["symbol"], refused["side"], refused["reason"], strict=True)) == {
        ("INFY", "BUY", "account_trade_limit")
    }


def test_the_contribution_counts_the_entries_the_account_refused(capped):
    assert capped.portfolio.contribution["orders_refused"].tolist() == [0, 5]


def test_the_combined_page_lists_the_entries_the_account_refused(capped):
    page = capped.portfolio.tearsheet()
    assert "The account refused no order and ended none." not in page
    assert "5 orders the account refused or ended, with the times" in page


def test_each_strategys_contribution_is_its_own_loss(account):
    table = account.portfolio.contribution
    assert table["net_pnl_rupees"].tolist() == [LOSS_EACH, LOSS_EACH]
    assert table["share_pct"].tolist() == [50.0, 50.0]


def test_the_combined_trades_are_both_kill_switch_exits_named_by_strategy(account):
    trades = account.portfolio.trades
    assert sorted(trades["strategy"]) == ["HoldInfy", "HoldSbin"]
    assert set(trades["exit_reason"]) == {"account_loss_limit"}


# ----- what it says ------------------------------------------------------------------------------


def test_the_assumptions_name_the_capitals_and_the_funds_openalgo_reported_at_the_start(
    account,
):
    notes = account.portfolio.assumptions
    assert notes["account_capital"] == 2 * CAPITAL
    assert notes["account_max_daily_loss"] == ACCOUNT_LIMIT
    assert "Rs 2,00,000 here is the sum of the strategies' capitals" in notes["account"]
    # The fake's live book reports Rs 2,50,000 available (Indian grouping).
    assert LIVE_CASH == 250_000
    assert "OpenAlgo reported Rs 2,50,000 of funds at the start" in notes["account"]
    assert "the broker's margin check is the account's" in notes["account"]
    assert "the entries the account's trade cap refused are listed in refused" in notes["account"]
    assert "stay in its orders with the code live_lock" in notes["account"]


def test_a_session_without_a_shared_instrument_states_no_rule_of_one(account):
    notes = account.portfolio.assumptions
    assert notes["shared_instruments"] is None
    assert "shared_instrument_rules" not in notes


def test_the_summary_gives_the_accounts_figures(account):
    lines = account.portfolio.summary().splitlines()
    assert lines[:2] == ["Net profit   : Rs -160.00", "Return       : -0.08%"]


def test_the_combined_tearsheet_is_titled_by_the_strategies_with_the_portfolio_section(account):
    page = account.portfolio.tearsheet()
    assert "<h1 class='serif'>Portfolio: HoldSbin and HoldInfy</h1>" in page
    assert "<section id='portfolio'" in page


# ----- one strategy, a shared instrument, a restart ----------------------------------------------


def test_a_session_of_one_strategy_is_an_account_of_that_strategy(world):
    result = live(world, BuyAndHold)
    pf = result.portfolio
    own = result.results["BuyAndHold"].equity["equity_rupees"]
    assert pf.strategies == ("BuyAndHold",)
    assert pf.equity.index.tolist() == own.index.tolist()
    assert pf.equity.tolist() == pytest.approx(own.tolist(), abs=0.005)


def test_a_shared_instruments_net_position_is_the_members_units_added_up(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # A buys 100 SBIN and B 50 at their first decision, both filled at once; Ctrl+C at
    # 14:45 keeps both positions.
    made = rig(make_fake_openalgo, monkeypatch, tmp_path)
    made.stop_at(ist(14, 45))
    seam(made, monkeypatch)
    pf = live(made, [SbinTrend, SbinBreakout], on_stop="keep").portfolio
    first = pf.net_positions.iloc[0]
    assert (first["symbol"], first["exchange"], first["product"]) == ("SBIN", "NSE", "MIS")
    assert (first["SbinTrend_units"], first["SbinBreakout_units"]) == (A_UNITS, B_UNITS)
    assert (first["net_units"], first["holders"]) == (A_UNITS + B_UNITS, "SbinTrend, SbinBreakout")
    notes = pf.assumptions
    assert notes["shared_instruments"] == "SBIN (NSE): SbinTrend (MIS) and SbinBreakout (MIS)"
    assert notes["shared_instrument_rules"].startswith(
        "the strategies trading one instrument share one broker position; each holds its own "
        "stop; they trade one side at a time and with market orders only"
    )


def test_after_a_restart_the_portfolio_covers_the_restarted_start_only(world):
    # TwoRoundTrips' first round trip closes before the Ctrl+C at 14:46:30, its second in
    # the restarted start (test_live_result.py): that start's account holds one trade and
    # its own money only, as its result does.
    at_price(world, ist(14, 44, 30), 505.0)
    at_price(world, ist(14, 48, 30), 510.0)
    world.stop_at(ist(14, 46, 30))
    live(world, TwoRoundTrips)
    world.stop = StopFlag()
    world.stop_at(DAY_ENDS)
    second = live(world, TwoRoundTrips)
    pf = second.portfolio
    assert len(pf.trades) == 1
    own = second.results["TwoRoundTrips"].metrics["net_profit"]
    assert pf.equity.iloc[-1] - CAPITAL == pytest.approx(own, abs=0.005)
    # Its daily loss measure counts from the restart too, and its note says so: the
    # restarted start's round trip alone, 10 units bought at 505.00 and sold at 510.00 on
    # the gross basis, where the day's measure holds both round trips (about Rs 100).
    assert "the daily loss measure counts this start only" in pf.assumptions["account"]
    assert pf.account["daily_loss_rupees"].iloc[-1] == pytest.approx(10 * (510.0 - 505.0))


# ----- the scope of a restarted start's portfolio (2.4.1 item W-1) -----------------------------

#: When the restarted start of these tests begins: after the first start's Ctrl+C at
#: 14:46:30 and its close, early enough to decide on the 14:48 bar and make its round trip
#: (16-session-design 3.1 step 14). It is the day's second start, from 14:46 IST.
RESTART_AT = ist(14, 46, 40)
#: How a restarted start's portfolio states what it covers.
RESTART_SCOPE = (
    "from the restart at 14:46 IST (start 2 of the day): the day's figures across restarts "
    "are in the session summary (SessionResult.summary_text())"
)


def restarted(world) -> bt.SessionResult:
    """TwoRoundTrips stopped by Ctrl+C at 14:46:30 after its first round trip, then started
    again at `RESTART_AT` and run to the day's end: the second start's result. Both starts
    charge the zerodha preset, on which the figures below are worked."""
    at_price(world, ist(14, 44, 30), 505.0)
    at_price(world, ist(14, 48, 30), 510.0)
    world.stop_at(ist(14, 46, 30))
    live(world, TwoRoundTrips, costs="zerodha")
    now = world.fake.clock.now()
    assert now < RESTART_AT
    world.script.sleep((RESTART_AT - now) / 10**9)
    world.stop = StopFlag()
    world.stop_at(DAY_ENDS)
    return live(world, TwoRoundTrips, costs="zerodha")


def test_after_a_restart_the_portfolio_summary_states_its_scope_at_the_top(world):
    second = restarted(world)
    assert second.restarts == 1
    lines = second.portfolio.summary().splitlines()
    assert lines[0] == "F" + RESTART_SCOPE[1:]
    # Below it, the account's own figures: this start's round trip alone, 10 SBIN bought at
    # 505.00 and sold at 510.00 (gross Rs 50.00), less its contract note on the zerodha
    # preset, each line the round trip's sum rounded once: brokerage 0.03 percent of
    # Rs 5,050.00 and Rs 5,100.00 (3.045, 3.05), STT 0.025 percent of the sale (1.275, Rs 1),
    # exchange charges 0.00307 percent of Rs 10,150.00 (0.3116, 0.31), SEBI fee Rs 10 a
    # crore (0.0102, 0.01), stamp duty 0.003 percent of the buy (0.1515, Rs 0), GST 18
    # percent of brokerage, exchange charges and SEBI fee (0.6060, 0.61): Rs 4.98, so
    # 50.00 - 4.98 = 45.02.
    assert lines[1] == "Net profit   : Rs 45.02"
    assert lines[4] == "Trades       : 1 (100% won)"


def test_after_a_restart_the_combined_tearsheet_states_its_scope_in_the_assumptions(world):
    pf = restarted(world).portfolio
    assert pf.assumptions["scope"] == RESTART_SCOPE
    row = f"<td class='l'>scope</td><td class='l'>{html.escape(RESTART_SCOPE)}</td>"
    assert row in pf.tearsheet()


def test_a_first_starts_portfolio_states_no_scope(account):
    pf = account.portfolio
    assert "scope" not in pf.assumptions
    assert pf.summary().splitlines()[0] == "Net profit   : Rs -160.00"
    assert "<td class='l'>scope</td>" not in pf.tearsheet()


def test_the_scope_of_a_days_third_start_names_its_start_and_its_time():
    # Restarted at 10:05:30 IST after two starts of the day.
    assert restart_scope(ist(10, 5, 30), restarts=2) == (
        "from the restart at 10:05 IST (start 3 of the day): the day's figures across "
        "restarts are in the session summary (SessionResult.summary_text())"
    )


def test_a_first_start_has_no_scope_to_state():
    assert restart_scope(ist(14, 40, 5), restarts=0) is None
