"""The futures margin of a backtest (decision D23; 04-execution 5): by default every
futures contract blocks 20 percent of its contract value (5x), on any exchange and
underlying; `margin=` replaces it with the run's own figure, one value for every
contract or a dict by underlying, as `bt.pct(x)` percent of contract value or
`bt.rupees(x)` rupees per lot of the contract traded.

The intraday runs use `accounting_runs`: capital 10,00,000, costs and slippage off, the
entry filling at the 09:30 open of 2026-11-04 at 1,000.00 and blocking its margin from
then until the exit at the 09:45 open. The roll runs use two NIFTY contracts of
`test_futures` (September 2024 with lot 25, here October 2024 with lot 50).
"""

import datetime as dt

import pandas as pd
import pytest
from accounting_runs import (
    CAPITAL,
    NIFTY_FUTURE,
    at,
    daily_bars,
    future_round_trip,
    round_trip_strategy,
    session_bars,
)

import openalgo.backtest as bt
from openalgo.backtest.errors import ConfigError
from openalgo.backtest.reference import Instrument

OFF = {"costs": None, "slippage": 0}
DEFAULT_LINE = "20 percent of contract value (default)"


def margin_after_entry(result) -> float:
    """Margin blocked at the close of the entry slice (09:30 to 09:31)."""
    return result.equity.loc[at(9, 31), "margin_rupees"]


def _future(symbol: str, exchange: str, underlying: str, lot: int, expiry: dt.date) -> Instrument:
    return Instrument(
        symbol=symbol,
        exchange=exchange,
        kind="future",
        lot_size=lot,
        tick_size=0.05,
        expiry=expiry,
        underlying=underlying,
        settlement="cash",
    )


SBIN_FUTURE = _future("SBIN24NOV26FUT", "NFO", "SBIN", 750, dt.date(2026, 11, 24))
SENSEX_FUTURE = _future("SENSEX26NOV26FUT", "BFO", "SENSEX", 20, dt.date(2026, 11, 26))


# ----- the default: 20 percent of contract value for every futures contract ----------


def test_an_index_future_blocks_twenty_percent_of_its_contract_value_by_default():
    # One lot of 75 at 1,000.00 is 75,000; 20 percent of it is 15,000.
    result = future_round_trip({"lots": 1}, **OFF)
    assert margin_after_entry(result) == 15_000.0


def test_a_stock_future_blocks_twenty_percent_of_its_contract_value_by_default():
    # One lot of 750 at 1,000.00 is 7,50,000; 20 percent of it is 1,50,000.
    result = future_round_trip({"lots": 1}, SBIN_FUTURE, **OFF)
    assert margin_after_entry(result) == 150_000.0


def test_a_sensex_future_has_the_default_margin():
    # Daily bars: bought at the 11-02 close, filled at the 11-03 open of 1,000.00 and sold
    # at the 11-04 open. One lot of 20 is 20,000; 20 percent of it is 4,000.
    strategy = round_trip_strategy("D", SENSEX_FUTURE.symbol, "BFO", lots=1)
    result = bt.backtest(
        strategy,
        data=daily_bars(),
        capital=CAPITAL,
        instrument=SENSEX_FUTURE,
        product="NRML",
        **OFF,
    )
    assert list(result.equity["margin_rupees"]) == [0.0, 4_000.0, 0.0]


def test_the_default_is_stated_in_the_assumptions_and_does_not_warn():
    result = future_round_trip({"lots": 1}, **OFF)
    assert result.assumptions["futures_margin"] == f"NIFTY {DEFAULT_LINE}"
    assert not any("margin" in text for text in result.warnings)


def test_a_run_without_futures_states_no_futures_margin():
    result = bt.backtest(
        round_trip_strategy(qty=10), data=session_bars(), capital=CAPITAL, base="1m", **OFF
    )
    assert "futures_margin" not in result.assumptions


# ----- margin=: one value for every futures contract --------------------------------


def test_one_percent_sets_the_margin_of_every_futures_contract():
    # 12.5 percent of 75,000 is 9,375.
    result = future_round_trip({"lots": 1}, margin=bt.pct(12.5), **OFF)
    assert margin_after_entry(result) == 9_375.0
    assert result.assumptions["futures_margin"] == (
        "NIFTY 12.5 percent of contract value (margin=)"
    )


def test_rupees_per_lot_block_that_amount_for_each_lot():
    # Two lots at 50,000 a lot are 1,00,000, whatever the price.
    result = future_round_trip({"lots": 2}, margin=bt.rupees(50_000), **OFF)
    assert margin_after_entry(result) == 100_000.0
    assert result.assumptions["futures_margin"] == "NIFTY 50000 rupees per lot (margin=)"


def test_an_entry_is_rejected_when_the_own_margin_is_more_than_the_free_cash():
    # Two lots at 6,00,000 a lot need 12,00,000 of the 10,00,000 free; at the default
    # they need 2 x 15,000 = 30,000 and fill.
    refused = future_round_trip({"lots": 2}, margin=bt.rupees(600_000), **OFF)
    assert refused.fills.empty
    assert list(refused.orders["reject_code"]) == ["margin"]
    assert len(future_round_trip({"lots": 2}, **OFF).fills) == 2


# ----- margin=: a dict by underlying ---------------------------------------------------


class NiftyAndSbin(bt.Strategy):
    """Buys one lot of the NIFTY and the SBIN futures at its first bar and exits both at
    its second."""

    symbols = (NIFTY_FUTURE.symbol, SBIN_FUTURE.symbol)
    exchange, interval = "NFO", "15m"

    def on_bar(self, bar):
        bars = self.state.get("bars", 0) + 1
        self.state["bars"] = bars
        for symbol in self.symbols:
            if bars == 1:
                self.buy(lots=1, symbol=symbol)
            elif bars == 2:
                self.exit(symbol=symbol)


def pair_run(**settings):
    return bt.backtest(
        NiftyAndSbin,
        data={symbol: session_bars(minutes=385) for symbol in NiftyAndSbin.symbols},
        instrument={NIFTY_FUTURE.symbol: NIFTY_FUTURE, SBIN_FUTURE.symbol: SBIN_FUTURE},
        capital=CAPITAL,
        base="1m",
        product="NRML",
        **OFF,
        **settings,
    )


def test_an_underlying_the_dict_does_not_name_keeps_the_default():
    # SBIN at its own 10 percent of 7,50,000 is 75,000; NIFTY, not named, at the
    # default 20 percent of 75,000 is 15,000: 90,000 in all.
    result = pair_run(margin={"SBIN": bt.pct(10)})
    assert margin_after_entry(result) == 90_000.0
    assert result.assumptions["futures_margin"] == (
        f"NIFTY {DEFAULT_LINE}; SBIN 10 percent of contract value (margin=)"
    )


def test_names_the_run_does_not_trade_are_reported_as_unused():
    result = future_round_trip({"lots": 1}, margin={"BANKNIFTY": bt.pct(15)}, **OFF)
    assert margin_after_entry(result) == 15_000.0
    assert any("BANKNIFTY" in text and "unused" in text for text in result.warnings)


# ----- rupees per lot across a roll with a lot size change ---------------------------

SEPTEMBER = "NIFTY26SEP24FUT"
OCTOBER = "NIFTY31OCT24FUT"
ROLL_BARS = {
    SEPTEMBER: {
        20: (24950.0, 25050.0, 24900.0, 25000.0),
        23: (25000.0, 25150.0, 24950.0, 25100.0),
        24: (25100.0, 25250.0, 25050.0, 25200.0),
        25: (25200.0, 25300.0, 25150.0, 25250.0),
        26: (25250.0, 25350.0, 25200.0, 25300.0),
    },
    OCTOBER: {
        25: (25300.0, 25400.0, 25250.0, 25350.0),
        26: (25350.0, 25450.0, 25300.0, 25400.0),
        27: (25400.0, 25500.0, 25350.0, 25450.0),
        30: (25450.0, 25550.0, 25400.0, 25500.0),
    },
}
ROLL_CONTRACTS = {
    SEPTEMBER: _future(SEPTEMBER, "NFO", "NIFTY", 25, dt.date(2024, 9, 26)),
    OCTOBER: _future(OCTOBER, "NFO", "NIFTY", 50, dt.date(2024, 10, 31)),
}
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def _frame(rows: dict[int, tuple]) -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [dt.datetime(2024, 9, day, 9, 15, tzinfo=IST) for day in rows], name="timestamp"
    )
    frame = pd.DataFrame(list(rows.values()), index=index, columns=["open", "high", "low", "close"])
    frame["volume"] = 1_000.0
    return frame


class TwoLotsOfNifty(bt.Strategy):
    """Buys two lots of the current NIFTY future at the 09-20 close and exits at the
    09-27 close; the position rolls at the 09-25 open."""

    underlying, exchange, interval = "NIFTY", "NFO", "D"

    def on_bar(self, bar):
        if bar.time.day == 20:
            self.buy(lots=2)
        elif bar.time.day == 27:
            self.exit()


def test_rupees_per_lot_follow_the_lots_of_each_contract_across_a_roll():
    # Two September lots of 25 (50 units) at 1,00,000 a lot block 2,00,000 from the 09-23
    # open. At the 09-25 roll the 50 units are one October lot of 50: 1,00,000 until the
    # exit at the 09-30 open.
    result = bt.backtest(
        TwoLotsOfNifty,
        capital=CAPITAL,
        data={symbol: _frame(rows) for symbol, rows in ROLL_BARS.items()},
        instrument=ROLL_CONTRACTS,
        start=dt.date(2024, 9, 20),
        end=dt.date(2024, 9, 30),
        margin={"NIFTY": bt.rupees(100_000)},
        **OFF,
    )
    margin = result.equity["margin_rupees"]
    assert list(margin) == [0.0, 200_000.0, 200_000.0, 100_000.0, 100_000.0, 100_000.0, 0.0]
    assert list(result.rolls["to_symbol"]) == [OCTOBER]


# ----- validation ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("margin", "message"),
    [
        (20, "bare number"),
        ({"NIFTY": 15}, "bare number"),
        (bt.pct(120), "at most 100"),
        (bt.points(5), r"takes bt\.pct"),
        ("20%", r"takes bt\.pct"),
        ({"NIFTY": None}, r"takes bt\.pct"),
        (bt.rupees(float("inf")), "finite"),
        ({}, "empty dict"),
        ({"nifty": bt.pct(10)}, "'NIFTY', not 'nifty'"),
        ({"NIFTY24NOV26FUT": bt.pct(10)}, "per underlying, not per contract"),
        ({1: bt.pct(10)}, "keyed by underlying symbol"),
        ({"NIFTY 50": bt.pct(10)}, "keyed by underlying symbol"),
    ],
    ids=str,
)
def test_an_unusable_margin_raises_a_config_error_saying_what_to_write(margin, message):
    with pytest.raises(ConfigError, match=message):
        future_round_trip({"lots": 1}, margin=margin, **OFF)


# ----- the manifest, rerun and optimize ------------------------------------------------


def test_the_manifest_records_margin_and_a_rerun_reproduces_it():
    result = pair_run(margin={"SBIN": bt.rupees(40_000)})
    assert result.manifest["arguments"]["margin"] == {"SBIN": ["rupees_per_lot", 40_000.0]}
    again = result.rerun()
    # SBIN at 40,000 a lot and NIFTY at the default 15,000.
    assert margin_after_entry(again) == margin_after_entry(result) == 55_000.0
    assert again.manifest["arguments"]["margin"] == result.manifest["arguments"]["margin"]


def test_a_rerun_can_change_margin():
    result = future_round_trip({"lots": 1}, margin=bt.pct(10), **OFF)
    assert result.manifest["arguments"]["margin"] == ["percent", 10.0]
    assert margin_after_entry(result.rerun(margin=None)) == 15_000.0


class NiftyLots(bt.Strategy):
    """Buys `lots` lots of the NIFTY future at its first bar and exits at its second."""

    symbol, exchange, interval = NIFTY_FUTURE.symbol, "NFO", "15m"
    lots = bt.param(1, (1, 2, 3))

    def on_bar(self, bar):
        if len(self.close) == 1:
            self.buy(lots=self.lots)
        elif len(self.close) == 2 and not self.position.is_flat:
            self.exit()


@pytest.mark.parametrize("workers", [1, 2])
def test_optimize_runs_every_combination_with_the_margin(workers):
    # At 6,00,000 a lot one lot fills (6,00,000 of 10,00,000 free), and two and three
    # lots are rejected for margin (12,00,000 and 18,00,000); at the default all three
    # would fill. With two workers the second and third run in worker processes.
    result = bt.optimize(
        NiftyLots,
        data=session_bars(minutes=385),
        instrument=NIFTY_FUTURE,
        capital=CAPITAL,
        base="1m",
        product="NRML",
        margin=bt.rupees(600_000),
        workers=workers,
        **OFF,
    )
    trades = list(zip(result.table["lots"], result.table["trades"], strict=True))
    assert trades == [(1, 1), (2, 0), (3, 0)]
    assert ("2 worker processes" in result.plan) == (workers == 2)
    chosen = result.select(lots=1)
    assert chosen.manifest["arguments"]["margin"] == ["rupees_per_lot", 600_000.0]
