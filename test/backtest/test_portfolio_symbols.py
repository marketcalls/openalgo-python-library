"""The run settings of a portfolio run that are keyed by symbol (`instrument=`,
`corporate_actions=`) or by underlying (`margin=`): one value for the run, of which each
strategy is given the entries of what it trades, so that each runs as it would alone with
them; an entry no strategy uses is refused, or for `margin=` warned about once, for the run
(18-multi-strategy 1.1; `_portfolio_symbols`).

The futures runs trade the two NIFTY contracts of `test_run_futures_margin` (September
2024 with lot 25, October 2024 with lot 50) beside a daily SBIN strategy on the same
sessions; the minute runs are `portfolio_samples`' scripts. Costs and slippage are off and
nothing contacts a server.
"""

from __future__ import annotations

import datetime as dt
import warnings

import pandas as pd
import pytest
from portfolio_samples import Scripted, flat_bars, no_costs
from strategy_samples import DAY
from test_run_futures_margin import ROLL_BARS, ROLL_CONTRACTS, TwoLotsOfNifty

import openalgo.backtest as bt
from openalgo.backtest.data import CorporateAction

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
CAPITAL = 1_000_000
LAKH = 100_000
#: Every table of a `Result` the stand-alone comparisons check.
TABLES = (
    "fills",
    "orders",
    "trades",
    "positions",
    "equity",
    "events",
    "costs",
    "contract_notes",
    "final_positions",
    "skipped",
)
UNUSED_FOR_SHARES = (
    "margin= sets the margin of futures contracts; this run trades none, so it is unused"
)


def september(rows: dict[int, tuple[float, float, float, float]]) -> pd.DataFrame:
    """Daily bars of September 2024 sessions, (open, high, low, close) by day of month."""
    index = pd.DatetimeIndex(
        [dt.datetime(2024, 9, day, 9, 15, tzinfo=IST) for day in rows], name="timestamp"
    )
    frame = pd.DataFrame(list(rows.values()), index=index, columns=["open", "high", "low", "close"])
    frame["volume"] = 1_000.0
    return frame


class SbinDaily(bt.Strategy):
    """Buys 10 SBIN at the 09-20 close and exits at the 09-27 close."""

    symbol, exchange, interval = "SBIN", "NSE", "D"

    def on_bar(self, bar):
        if bar.time.day == 20:
            self.buy(qty=10)
        elif bar.time.day == 27:
            self.exit()


#: The NSE sessions from 20 to 30 September 2024.
SESSIONS = (20, 23, 24, 25, 26, 27, 30)
SBIN_ROWS = {day: (800.0 + day, 805.0 + day, 795.0 + day, 800.0 + day) for day in SESSIONS}
FUTURES_RUN = {
    "capital": CAPITAL,
    "data": {
        **{symbol: september(rows) for symbol, rows in ROLL_BARS.items()},
        "SBIN": september(SBIN_ROWS),
    },
    "start": dt.date(2024, 9, 20),
    "end": dt.date(2024, 9, 30),
    **no_costs(),
}
NIFTY_MARGIN = {"NIFTY": bt.rupees(100_000)}


@pytest.fixture(scope="module")
def futures_beside_shares():
    # instrument= holds NIFTY contracts only and margin= names NIFTY and BANKNIFTY, which
    # no strategy trades: the futures strategy is given the contracts and NIFTY's margin,
    # the share strategy neither.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pf = bt.backtest(
            [TwoLotsOfNifty, SbinDaily],
            instrument=ROLL_CONTRACTS,
            margin={**NIFTY_MARGIN, "BANKNIFTY": bt.pct(12)},
            **FUTURES_RUN,
        )
        alone = {
            "TwoLotsOfNifty": bt.backtest(
                TwoLotsOfNifty, instrument=ROLL_CONTRACTS, margin=NIFTY_MARGIN, **FUTURES_RUN
            ),
            "SbinDaily": bt.backtest(SbinDaily, **FUTURES_RUN),
        }
    return pf, alone


def test_each_strategy_is_given_its_own_instruments_and_margin_and_runs_as_alone(
    futures_beside_shares,
):
    pf, alone = futures_beside_shares
    for name, own in alone.items():
        member = pf.strategy(name)
        for table in TABLES:
            pd.testing.assert_frame_equal(getattr(member, table), getattr(own, table), obj=table)
        assert member.warnings == own.warnings
        assert member.manifest == own.manifest
        block = member.assumptions
        assert block.pop("account").startswith("ran in a shared account")
        assert block == own.assumptions
    # Two September lots of 25 at 1,00,000 a lot, then one October lot of 50 after the roll.
    assert pf.strategy("TwoLotsOfNifty").rolls["to_symbol"].tolist() == ["NIFTY31OCT24FUT"]


def test_a_margin_entry_no_strategy_trades_is_one_warning_of_the_run(futures_beside_shares):
    pf, _ = futures_beside_shares
    margin_notes = [text for text in pf.warnings if "margin=" in text]
    assert margin_notes == [
        "margin= names BANKNIFTY, which this run does not trade as futures; its value is unused"
    ]


class OnSbin(Scripted):
    """Buys 100 SBIN at its first decision."""


class OnInfy(Scripted):
    """Buys 100 INFY at its first decision."""

    symbol = "INFY"


SHARES_RUN = {
    "data": {"SBIN": flat_bars(100.0), "INFY": flat_bars(200.0)},
    "capital": LAKH,
    **no_costs(),
}


@pytest.mark.parametrize("margin", [bt.pct(12), {"NIFTY": bt.pct(12)}], ids=["one", "by name"])
def test_a_margin_no_strategy_can_use_is_one_warning_of_the_run(margin):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pf = bt.backtest([OnSbin, OnInfy], margin=margin, **SHARES_RUN)
    assert [text for text in pf.warnings if "margin=" in text] == [UNUSED_FOR_SHARES]
    assert pf.strategy("OnSbin").manifest["arguments"]["margin"] is None


def test_each_strategy_records_only_the_corporate_actions_of_its_own_symbols():
    checked = [CorporateAction.none(DAY)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pf = bt.backtest([OnSbin, OnInfy], corporate_actions={"SBIN": checked}, **SHARES_RUN)
    declared = {
        name: result.manifest["arguments"]["corporate_actions"]
        for name, result in pf.results.items()
    }
    assert declared == {
        "OnSbin": {"SBIN": [{"ex_date": DAY.isoformat(), "kind": "none", "factor": 1.0}]},
        "OnInfy": None,
    }
