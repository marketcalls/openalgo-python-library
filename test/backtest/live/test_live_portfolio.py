"""`SessionResult.portfolio`: a live session's strategies as one account, a `bt.PortfolioResult`
(`_live/_result_portfolio.py`, `session_result.py`; 18-multi-strategy 1.3 and 4.11, unit M5).

The market and the scripts are `session_samples`' and `watch_samples`': the session starts
at 14:40:05 IST on Monday 2026-11-16, each symbol trades 10 units every 10 s at 500.00.
In the account's session, SBIN and INFY fall to 492.00 from 14:43:40; `HoldSbin` and
`HoldInfy` (capital Rs 1,00,000 each) buy 10 units at 500.00 at their first decision
(14:42:01), and the account's daily loss limit of Rs 150 fires the kill switch at the
first trade at 492.00, which sells both at 492.00: -80.00 each, with no charges modelled
(`costs=None`); Ctrl+C at 14:47. Expected values are worked from the scenario and the
fake's books, never from the portfolio's code.
"""

from __future__ import annotations

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
from openalgo.backtest._live._result_portfolio import account_rows
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


def test_the_account_refuses_nothing_the_brokers_margin_check_is_its(account):
    refused = account.portfolio.refused
    assert list(refused.columns) == list(REFUSED_COLUMNS)
    assert refused.empty


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
