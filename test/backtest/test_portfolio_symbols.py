"""The run settings of a portfolio run that are keyed by symbol (`instrument=`,
`corporate_actions=`) or by underlying (`margin=`): one value for the run, of which each
strategy is given the entries of what it trades, so that each runs as it would alone with
them; an entry no strategy uses is refused, or for `margin=` warned about once, for the run
(18-multi-strategy 1.1; `_portfolio_symbols`).

The futures runs trade the two NIFTY contracts of `test_run_futures_margin` (September
2024 with lot 25, October 2024 with lot 50), or two SBIN contracts (lot 750), beside a
daily SBIN strategy on the same sessions; the minute runs are `portfolio_samples`'
scripts. Costs and slippage are off and nothing contacts a server.
"""

from __future__ import annotations

import datetime as dt
import re
import warnings

import pandas as pd
import pytest
from portfolio_samples import Scripted, flat_bars, no_costs
from strategy_samples import DAY
from test_run_futures_margin import ROLL_BARS, ROLL_CONTRACTS, TwoLotsOfNifty

import openalgo.backtest as bt
from openalgo.backtest.data import CorporateAction
from openalgo.backtest.errors import ConfigError
from openalgo.backtest.reference import Instrument

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


def assert_each_runs_as_alone(pf, alone: dict) -> None:
    """Each strategy's own `Result` in the portfolio run `pf` equals its run alone in
    `alone` (by name): every table, the warnings, the manifest and the Assumptions block
    but for the line naming the account it ran in."""
    for name, own in alone.items():
        member = pf.strategy(name)
        for table in TABLES:
            pd.testing.assert_frame_equal(getattr(member, table), getattr(own, table), obj=table)
        assert member.warnings == own.warnings
        assert member.manifest == own.manifest
        block = member.assumptions
        assert block.pop("account").startswith("ran in a shared account")
        assert block == own.assumptions


def test_each_strategy_is_given_its_own_instruments_and_margin_and_runs_as_alone(
    futures_beside_shares,
):
    pf, alone = futures_beside_shares
    assert_each_runs_as_alone(pf, alone)
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


# ----- a share and the futures of the same name ------------------------------------------

SBIN_SEPTEMBER = "SBIN26SEP24FUT"
SBIN_OCTOBER = "SBIN31OCT24FUT"


def sbin_future(symbol: str, last_day: dt.date) -> Instrument:
    """An SBIN futures contract on NFO: lot 750, tick 0.05, expiring on `last_day`."""
    return Instrument(
        symbol=symbol,
        exchange="NFO",
        kind="future",
        lot_size=750,
        tick_size=0.05,
        expiry=last_day,
        underlying="SBIN",
        settlement="cash",
    )


SBIN_CONTRACTS = {
    SBIN_SEPTEMBER: sbin_future(SBIN_SEPTEMBER, dt.date(2024, 9, 26)),
    SBIN_OCTOBER: sbin_future(SBIN_OCTOBER, dt.date(2024, 10, 31)),
}
SBIN_SHARE = {"SBIN": Instrument.cash_equity("SBIN", "NSE")}
#: The September contract 2 rupees above the share to its last trading day, the October
#: contract 4 rupees above from the 09-25 roll on.
SBIN_FUTURE_BARS = {
    SBIN_SEPTEMBER: september(
        {day: tuple(price + 2.0 for price in SBIN_ROWS[day]) for day in SESSIONS[:5]}
    ),
    SBIN_OCTOBER: september(
        {day: tuple(price + 4.0 for price in SBIN_ROWS[day]) for day in SESSIONS[3:]}
    ),
}
SHARE_AND_FUTURES_RUN = {**FUTURES_RUN, "data": {**FUTURES_RUN["data"], **SBIN_FUTURE_BARS}}
#: A checked date of SBIN: a session of the run with no split or bonus.
CHECKED_SBIN = [CorporateAction.none(dt.date(2024, 9, 23))]


class SbinFutures(bt.Strategy):
    """Buys one lot of the current SBIN future at the 09-20 close and exits at the 09-27
    close; the position rolls at the 09-25 open."""

    underlying, exchange, interval = "SBIN", "NFO", "D"

    def on_bar(self, bar):
        if bar.time.day == 20:
            self.buy(lots=1)
        elif bar.time.day == 27:
            self.exit()


class SbinOnBse(SbinDaily):
    """SbinDaily on BSE."""

    exchange = "BSE"


@pytest.mark.parametrize(
    "actions", [{"SBIN": CHECKED_SBIN}, CHECKED_SBIN], ids=["by symbol", "one list"]
)
def test_a_share_and_the_futures_of_its_name_each_take_only_their_own_entries(actions):
    # instrument= names the share and the contracts, corporate_actions= the share: the
    # share strategy is given the share's instrument and actions, the futures strategy the
    # contracts only (corporate actions apply to shares, and it trades none).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pf = bt.backtest(
            [SbinDaily, SbinFutures],
            instrument={**SBIN_CONTRACTS, **SBIN_SHARE},
            corporate_actions=actions,
            **SHARE_AND_FUTURES_RUN,
        )
        alone = {
            "SbinDaily": bt.backtest(
                SbinDaily,
                instrument=SBIN_SHARE,
                corporate_actions=actions,
                **SHARE_AND_FUTURES_RUN,
            ),
            "SbinFutures": bt.backtest(
                SbinFutures, instrument=SBIN_CONTRACTS, **SHARE_AND_FUTURES_RUN
            ),
        }
    assert_each_runs_as_alone(pf, alone)
    assert pf.strategy("SbinFutures").rolls["to_symbol"].tolist() == [SBIN_OCTOBER]
    assert pf.strategy("SbinDaily").manifest["arguments"]["corporate_actions"] is not None


def test_one_symbol_on_two_exchanges_takes_an_instrument_on_its_own_exchange_only():
    # instrument= holds SBIN on NSE: the NSE strategy is given it, the BSE strategy the
    # generic BSE share, as each alone.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pf = bt.backtest([SbinDaily, SbinOnBse], instrument=SBIN_SHARE, **FUTURES_RUN)
        alone = {
            "SbinDaily": bt.backtest(SbinDaily, instrument=SBIN_SHARE, **FUTURES_RUN),
            "SbinOnBse": bt.backtest(SbinOnBse, **FUTURES_RUN),
        }
    assert_each_runs_as_alone(pf, alone)


#: Settings that no strategy of the run can take, refused before any data is loaded:
#: (id, strategies, arguments, the refusal's words).
REFUSED = [
    (
        "an instrument on an exchange no strategy trades on",
        [SbinDaily, SbinFutures],
        {"instrument": Instrument.cash_equity("SBIN", "BSE")},
        "instrument= gives SBIN (equity, BSE), which no strategy of the run trades "
        "(SBIN futures on NFO, SBIN on NSE)",
    ),
    (
        "a future of an underlying no strategy trades",
        [SbinDaily, SbinFutures],
        {"instrument": ROLL_CONTRACTS},
        "instrument= gives NIFTY26SEP24FUT (future, NFO), NIFTY31OCT24FUT (future, NFO), "
        "which no strategy of the run trades (SBIN futures on NFO, SBIN on NSE)",
    ),
    (
        "corporate actions of a futures underlying",
        [SbinDaily, TwoLotsOfNifty],
        {"corporate_actions": {"NIFTY": CHECKED_SBIN}},
        "corporate_actions names ['NIFTY'], which no strategy of the run trades as a share "
        "on NSE or BSE (SBIN)",
    ),
    (
        "a list of corporate actions with no share traded",
        [TwoLotsOfNifty, SbinFutures],
        {"corporate_actions": CHECKED_SBIN},
        "corporate_actions apply to shares on NSE and BSE, and no strategy of the run "
        "trades one: leave them out",
    ),
    (
        "one table of bars for a share and its futures",
        [SbinDaily, SbinFutures],
        {"data": september(SBIN_ROWS)},
        "the strategies trade SBIN, SBIN futures: give data= as a dict of bars by symbol",
    ),
]


@pytest.mark.parametrize(
    ("strategies", "given", "words"),
    [row[1:] for row in REFUSED],
    ids=[row[0] for row in REFUSED],
)
def test_a_setting_no_strategy_can_take_is_refused_before_any_data_is_loaded(
    strategies, given, words
):
    # No client and, but for the one table, no data: a refusal after the checks would be
    # the data layer's, for the missing source.
    arguments = {"capital": CAPITAL, "start": dt.date(2024, 9, 20), "end": dt.date(2024, 9, 30)}
    with pytest.raises(ConfigError, match=re.escape(words)):
        bt.backtest(strategies, **arguments, **given)
