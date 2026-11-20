"""`bt.backtest` given a list of strategies: a portfolio run on one account
(18-multi-strategy 1.1, 1.2, 2.4, 2.5 and 3; `_portfolio_run`, `portfolio_result`).

The runs are `portfolio_samples`' scripts on in-process bars of NSE sessions of July 2025,
cash equity at 1x margin (an MIS share blocks its full value), no charges and no slippage
unless a test says otherwise; nothing contacts a server. Every expected figure is worked
by hand from the scripts and the bars in the comment above it.
"""

from __future__ import annotations

import datetime as dt
import re
import warnings

import pandas as pd
import pytest
from portfolio_samples import DAYS, DailyScript, Scripted, flat_bars, no_costs, stepped_bars
from strategy_samples import CAPITAL as STRATEGY_CAPITAL
from strategy_samples import (
    DAY,
    EmaCrossConditions,
    EmaCrossIntraday,
    at,
    daily_bars,
    wave_bars,
    wave_with_history,
)

import openalgo.backtest as bt
from openalgo.backtest.errors import ConfigError, OrdersNotPlacedWarning
from openalgo.backtest.portfolio_result import PORTFOLIO_TABLES

LAKH = 100_000


def ist(hour: int, minute: int, day: dt.date = DAY) -> pd.Timestamp:
    return pd.Timestamp(at(hour, minute, day)).tz_convert("Asia/Kolkata")


def nanos(hour: int, minute: int, day: dt.date = DAY) -> int:
    return ist(hour, minute, day).value


class First(Scripted):
    qty = 600


class Second(Scripted):
    qty = 600


@pytest.fixture(scope="module")
def contention():
    # Two strategies of Rs 1,00,000 on an account of Rs 1,00,000, SBIN at 100.00 all day;
    # both buy 600 at the 09:16 decision. At the 09:16 open First fills first (60,000 of
    # margin), leaving the account 1,00,000 - 2,00,000 + 40,000 + 1,00,000 = 40,000 of
    # free cash, less than Second's 60,000: Second is refused account_margin.
    data = {"SBIN": flat_bars(100.0)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pf = bt.backtest(
            [First, Second], data=data, capital=LAKH, account_capital=LAKH, **no_costs()
        )
    return pf, caught


def test_the_second_strategy_at_a_shared_open_is_refused_for_the_account_margin(contention):
    pf, _ = contention
    refused = pf.refused
    assert refused.drop(columns="time").to_dict("records") == [
        {
            "strategy": "Second",
            "order_id": 1,
            "symbol": "SBIN",
            "side": "BUY",
            "slice": 1,
            "slice_ts": nanos(9, 16),
            "step": 2,
            "reason": "account_margin",
        }
    ]
    assert refused["time"].tolist() == [ist(9, 16)]
    assert pf.strategy("Second").orders["reject_code"].tolist() == ["account_margin"]
    assert pf.strategy("First").trades["entry_units"].tolist() == [600.0]


def test_the_account_row_after_the_fill_holds_its_margin_and_free_cash(contention):
    pf, _ = contention
    # The round closing at 09:17: First holds 600 at 100.00.
    row = pf.account.loc[ist(9, 17)]
    assert row.to_dict() == {
        "equity_rupees": 100_000.0,
        "free_cash_rupees": 40_000.0,
        "margin_rupees": 60_000.0,
        "daily_loss_rupees": 0.0,
    }
    assert pf.equity.loc[ist(9, 17)] == 100_000.0


def test_kernel_ids_restart_per_strategy_and_combined_rows_carry_the_strategy(contention):
    pf, _ = contention
    # Both orders are each engine's order 1, decided at 09:16; call order breaks the tie.
    orders = pf.orders
    assert list(zip(orders["strategy"], orders["order_id"], strict=True)) == [
        ("First", 1),
        ("Second", 1),
    ]
    assert pf.trade_list().index.tolist() == [("First", 1)]


def test_each_strategy_result_says_the_account_it_ran_in(contention):
    pf, _ = contention
    assert pf.strategy("Second").assumptions["account"] == (
        "ran in a shared account of Rs 1,00,000 with First; 1 order refused for the "
        "account's margin"
    )
    assert pf.strategy("First").assumptions["account"] == (
        "ran in a shared account of Rs 1,00,000 with Second; 0 orders refused for the "
        "account's margin"
    )


def test_a_smaller_account_is_stated_in_the_assumptions(contention):
    pf, _ = contention
    assert pf.assumptions["account"] == (
        "shared account of Rs 1,00,000 for strategies of Rs 2,00,000 together: an opening "
        "fill needs its margin within the strategy's own free cash and within the "
        "account's, which the other strategies' margin and open losses take"
    )
    assert pf.assumptions["shared_instruments"] == "SBIN (NSE): First (MIS) and Second (MIS)"


def test_a_strategy_whose_order_the_account_refused_is_told_in_one_warning(contention):
    _, caught = contention
    told = [str(w.message) for w in caught if w.category is OrdersNotPlacedWarning]
    assert len(told) == 1
    assert told[0].startswith("Second: 1 of 1 order was rejected because in a portfolio run")


def test_the_contribution_counts_each_strategy_margin_and_refusals(contention):
    pf, _ = contention
    # Flat prices: nobody makes or loses anything, so no share and no drawdown.
    table = pf.contribution
    assert table.index.tolist() == ["First", "Second"]
    assert table["net_pnl_rupees"].tolist() == [0.0, 0.0]
    assert table["share_pct"].isna().all()
    assert table["drawdown_pnl_rupees"].isna().all()
    assert table["peak_margin_rupees"].tolist() == [60_000.0, 0.0]
    assert table["orders_refused"].tolist() == [0, 1]


def test_the_manifest_holds_the_account_and_each_strategy_manifest(contention):
    pf, _ = contention
    manifest = pf.manifest
    assert manifest["account"] == {
        "pool": "shared",
        "account_capital": 100_000.0,
        "account_max_daily_loss": None,
    }
    own = pf.strategy("First").manifest
    del own["engine_version"]
    assert manifest["strategies"]["First"] == own
    assert list(manifest["strategies"]) == ["First", "Second"]


def test_every_table_is_exported_the_strategies_prefixed_by_name(contention, tmp_path):
    pf, _ = contention
    tables = pf.to_dataframes()
    own = ["fills", "orders", "trades", "positions", "equity"]
    assert list(tables)[: len(PORTFOLIO_TABLES)] == list(PORTFOLIO_TABLES)
    assert {f"{name}_{table}" for name in ("First", "Second") for table in own} <= set(tables)
    written = pf.to_csv(tmp_path)
    assert len(written) == len(tables)
    assert len(pd.read_csv(tmp_path / "refused.csv")) == 1
    assert pd.read_csv(tmp_path / "contribution.csv")["strategy"].tolist() == ["First", "Second"]


def test_a_shared_account_of_the_sum_of_the_capitals_rarely_binds():
    # The same two buys on Rs 2,00,000: after First's fill the account has
    # 2,00,000 - 2,00,000 + 40,000 + 1,00,000 = 1,40,000 free, enough for Second.
    pf = bt.backtest([First, Second], data={"SBIN": flat_bars(100.0)}, capital=LAKH, **no_costs())
    assert pf.refused.empty
    assert pf.account.loc[ist(9, 17), "free_cash_rupees"] == 80_000.0
    assert pf.assumptions["account"] == (
        "shared account of Rs 2,00,000, the sum of the capitals: it refuses an order only "
        "when other strategies' open losses leave too little free cash; give "
        "account_capital to model a smaller account"
    )


class Twice(Scripted):
    qty, entries = 600, 2


class Once(Scripted):
    qty = 400


@pytest.fixture(scope="module")
def breach():
    # SBIN at 100.00 to the 09:24 bar, then 95.00. Twice holds 600 and Once 400 from the
    # 09:16 open. The round closing at 09:26 marks them at 95.00: the account's measure is
    # -3,000 - 2,000 = -5,000, at the limit of 5,000. Both exit at the 09:26 open at
    # 95.00; Twice, flat at the 09:27 decision, tries to enter again and is refused.
    data = {"SBIN": stepped_bars(100.0, 95.0, 10)}
    return bt.backtest(
        [Twice, Once], data=data, capital=LAKH, account_max_daily_loss=5_000, **no_costs()
    )


def test_the_account_limit_exits_every_strategy_with_its_own_reason(breach):
    trades = breach.trades
    assert trades["strategy"].tolist() == ["Twice", "Once"]
    assert trades["exit_reason"].tolist() == ["account_loss_limit", "account_loss_limit"]
    assert trades["exit_price_rupees"].tolist() == [95.0, 95.0]
    assert trades["gross_pnl_rupees"].tolist() == [-3_000.0, -2_000.0]
    hits = breach.risk_events
    assert hits.loc[hits["name"] == "account_loss_hit", "strategy"].tolist() == ["Twice", "Once"]
    assert breach.account.loc[ist(9, 26), "daily_loss_rupees"] == -5_000.0


def test_an_entry_after_the_account_limit_is_refused_until_the_next_session(breach):
    refused = breach.refused
    assert refused.drop(columns="time").to_dict("records") == [
        {
            "strategy": "Twice",
            "order_id": 2,
            "symbol": "SBIN",
            "side": "BUY",
            "slice": 11,
            "slice_ts": nanos(9, 26),
            "step": 7,
            "reason": "account_loss_gate",
        }
    ]
    assert (
        breach.strategy("Twice")
        .assumptions["account"]
        .endswith("; the account's daily loss limit Rs 5,000 refused 1 entry")
    )


def test_the_account_loss_is_shared_out_by_each_strategy_contribution(breach):
    # Net -5,000: Twice -3,000 (60 percent), Once -2,000 (40). The drawdown runs from the
    # 09:25 round (2,00,000) to the 09:26 one (1,95,000). The account's margin peaks first
    # at 09:17 (600 and 400 at 100.00).
    table = breach.contribution
    assert table["net_pnl_rupees"].tolist() == [-3_000.0, -2_000.0]
    assert table["share_pct"].tolist() == [60.0, 40.0]
    assert table["drawdown_pnl_rupees"].tolist() == [-3_000.0, -2_000.0]
    assert table["peak_margin_rupees"].tolist() == [60_000.0, 40_000.0]
    assert table["orders_refused"].tolist() == [1, 0]
    assert breach.metrics["net_profit"] == pytest.approx(-5_000.0)


class Long(Scripted):
    pass


class Short(Scripted):
    side, wait = "SELL", 1


@pytest.fixture(scope="module")
def opposite():
    # Long buys 100 at the 09:16 open; Short skips one decision and sells 100 at the 09:17
    # open; both are squared off at the 15:10 open.
    return bt.backtest([Long, Short], data={"SBIN": flat_bars(100.0)}, capital=LAKH, **no_costs())


def test_two_strategies_hold_one_symbol_on_opposite_sides_with_a_warning(opposite):
    assert (
        "SBIN (NSE, MIS): Long held it long while Short held it short, first from "
        "2025-07-01 09:17 IST; bt.live would have refused the later entry, since a broker "
        "nets one position per symbol and product"
    ) in opposite.warnings


def test_the_net_position_is_the_broker_view_of_a_shared_symbol(opposite):
    net = opposite.net_positions
    columns = ["time", "net_units", "holders", "Long_units", "Short_units"]
    assert net[columns].values.tolist() == [
        [ist(9, 16), 100.0, "Long", 100.0, 0.0],
        [ist(9, 17), 0.0, "Long, Short", 100.0, -100.0],
        [ist(15, 10), 0.0, "", 0.0, 0.0],
    ]
    assert set(net["symbol"]) == {"SBIN"} and set(net["product"]) == {"MIS"}


def test_combined_fills_are_in_time_order_one_instant_in_call_order(opposite):
    fills = opposite.fills
    assert list(zip(fills["strategy"], fills["side"], fills["time"], strict=True)) == [
        ("Long", "BUY", ist(9, 16)),
        ("Short", "SELL", ist(9, 17)),
        ("Long", "SELL", ist(15, 10)),
        ("Short", "BUY", ist(15, 10)),
    ]


class HolderA(DailyScript):
    buy_on, sell_on = 1, 3


class HolderB(DailyScript):
    buy_on, sell_on = 1, 3


class Buyer(DailyScript):
    buy_on = 3


def test_days_a_broker_would_net_the_delivery_trades_of_several_strategies_are_warned():
    # Daily bars, 1 to 4 July 2025. The holders buy at the 2 July open and sell at the 4
    # July open, the day Buyer buys: on 4 July SBIN is bought on CNC by one strategy and
    # sold on CNC by another, and sold from holdings by two.
    data = {"SBIN": daily_bars(DAYS)}
    pf = bt.backtest([HolderA, HolderB, Buyer], data=data, capital=LAKH, **no_costs())
    assert pf.warnings[:2] == [
        "SBIN (NSE) was bought on CNC by one strategy and sold on CNC by another on "
        "2025-07-04: a broker nets a scrip's buys and sells of a day across the account in "
        "one contract note and charges what offsets as intraday, where each strategy's "
        "delivery trades are charged here",
        "SBIN (NSE) was sold from holdings by two strategies on 2025-07-04: a broker "
        "charges the DP charge once per scrip and day, where each strategy is charged one "
        "here",
    ]


# ----- each strategy as it runs alone -----------------------------------------------------


class EmaInfy(EmaCrossConditions):
    """The condition strategy, decided in Rust, on INFY intraday."""

    symbol, product = "INFY", "MIS"


class TimedTcs(bt.Strategy):
    """Timers and callbacks: buys 50 TCS at 10:00 and exits at 14:00, counting its fills."""

    symbol, exchange, interval = "TCS", "NSE", "1m"

    def init(self):
        self.at("10:00", lambda: self.buy(qty=50))
        self.at("14:00", self.exit)

    def on_fill(self, fill):
        self.state["fills"] = self.state.get("fills", 0) + 1


THREE = [EmaCrossIntraday, EmaInfy, TimedTcs]
THREE_DATA = {
    "SBIN": wave_with_history(),
    "INFY": wave_with_history() + 1_000.0,
    "TCS": wave_bars() + 2_500.0,
}
# The default charges and slippage: every strategy pays its own.
THREE_RUN = {"data": THREE_DATA, "capital": STRATEGY_CAPITAL, "start": DAY, "end": DAY}
TABLES = (
    "fills",
    "orders",
    "trades",
    "positions",
    "equity",
    "events",
    "stale_marks",
    "costs",
    "contract_notes",
    "final_positions",
    "skipped",
)


@pytest.fixture(scope="module")
def alone():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {strategy.__name__: bt.backtest(strategy, **THREE_RUN) for strategy in THREE}


@pytest.mark.parametrize("pool", ["separate", "shared"])
def test_each_strategy_of_a_portfolio_gets_the_result_of_its_run_alone(alone, pool):
    # A separate pool, or a shared one of the sum that never binds: every table, figure,
    # warning and manifest of each strategy is that of its own bt.backtest call.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pf = bt.backtest(THREE, pool=pool, **THREE_RUN)
    assert pf.strategies == ("EmaCrossIntraday", "EmaInfy", "TimedTcs")
    for name, own in alone.items():
        member = pf.strategy(name)
        for table in TABLES:
            pd.testing.assert_frame_equal(getattr(member, table), getattr(own, table), obj=table)
        assert member.gross_pnl == own.gross_pnl
        assert member.warnings == own.warnings
        assert member.manifest == own.manifest
        block = member.assumptions
        assert block.pop("account").startswith("ran in ")
        assert block == own.assumptions
    assert len(alone["TimedTcs"].fills) == 2
    assert pf.refused.empty


def test_the_account_equity_of_a_separate_pool_is_the_sum_of_its_strategies(alone):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pf = bt.backtest(THREE, pool="separate", **THREE_RUN)
    # Every strategy closes its rounds on the same 1-minute grid: the account's equity at
    # each round is 3 x 10,00,000 - 3 x 10,00,000 + the three equities.
    together = sum(result.equity["equity_rupees"] for result in alone.values())
    pd.testing.assert_series_equal(pf.equity, together, check_names=False)
    assert pf.assumptions["account"].startswith("separate allocations of Rs 30,00,000 together")


def test_a_portfolio_of_one_is_the_strategy_alone(alone):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pf = bt.backtest([TimedTcs], **THREE_RUN)
    own = alone["TimedTcs"]
    for table in TABLES:
        pd.testing.assert_frame_equal(getattr(pf.strategy("TimedTcs"), table), getattr(own, table))
    assert pf.strategy("TimedTcs").assumptions["account"] == (
        "ran in a shared account of Rs 10,00,000 alone; 0 orders refused for the account's margin"
    )
    with pytest.raises(ConfigError, match="two"):
        pf.correlation  # noqa: B018 (a property that raises)


# ----- the account's figures --------------------------------------------------------------


class LongDaily(bt.Strategy):
    """Buys 100 SBIN at 09:16 every session (a timer: it fills at that open)."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.at("09:16", lambda: self.buy(qty=100))


class ShortDaily(LongDaily):
    """Sells 100 SBIN at 09:17 every session."""

    def init(self):
        self.at("09:17", lambda: self.sell(qty=100))


def test_the_correlation_of_two_strategies_on_opposite_sides_is_minus_one():
    # Two sessions: SBIN rises from 100.00 to 110.00 at 10:55 on the first and falls to
    # 95.00 on the second; the squares-off at 15:10 close every position. LongDaily makes
    # 100 x 10.00 = +1,000 then -500, ShortDaily -1,000 then +500: with two session returns
    # each, moving in opposite directions, the correlation is -1.
    first, second = DAYS[0], DAYS[1]
    data = {
        "SBIN": pd.concat(
            [
                stepped_bars(100.0, 110.0, 100, day=first),
                stepped_bars(100.0, 95.0, 100, day=second),
            ]
        )
    }
    pf = bt.backtest([LongDaily, ShortDaily], data=data, capital=LAKH, **no_costs())
    matrix = pf.correlation
    assert matrix.index.tolist() == ["LongDaily", "ShortDaily"]
    assert matrix.loc["LongDaily", "ShortDaily"] == pytest.approx(-1.0)
    assert pf.contribution["net_pnl_rupees"].tolist() == [500.0, -500.0]


# ----- refused before any data is loaded --------------------------------------------------


class OnInfy(Scripted):
    symbol = "INFY"


REFUSED = [
    ("pool unknown", {"pool": "pooled"}, ConfigError, "pool takes 'shared' or 'separate'"),
    (
        "separate pool of another capital",
        {"pool": "separate", "account_capital": 150_000},
        ConfigError,
        "pool='separate' gives each strategy its own capital, so the account is their sum, "
        "Rs 2,00,000.00: leave account_capital out",
    ),
    (
        "account capital not positive",
        {"account_capital": -5},
        ConfigError,
        "account_capital takes a positive amount of rupees",
    ),
    (
        "account limit not positive",
        {"account_max_daily_loss": 0},
        ConfigError,
        "account_max_daily_loss takes a positive amount of rupees",
    ),
    (
        "capital not positive for one strategy",
        {"capital": {"First": LAKH, "OnInfy": 0}},
        ConfigError,
        "OnInfy: capital must be a positive number of rupees",
    ),
    (
        "one table of bars for two symbols",
        {"data": flat_bars(100.0)},
        ConfigError,
        "the strategies trade INFY, SBIN: give data= as a dict of bars by symbol",
    ),
]


@pytest.mark.parametrize(
    ("given", "error", "words"),
    [row[1:] for row in REFUSED],
    ids=[row[0] for row in REFUSED],
)
def test_a_portfolio_run_refuses_unusable_arguments_before_loading_data(given, error, words):
    # No data and no client: a refusal after the checks would be the data layer's.
    arguments = {"capital": LAKH, **given}
    with pytest.raises(error, match=re.escape(words)):
        bt.backtest([First, OnInfy], **arguments)


@pytest.mark.parametrize("verb", [bt.optimize, bt.robustness, bt.walk_forward])
def test_the_searches_and_checks_take_one_strategy_at_a_time(verb):
    with pytest.raises(NotImplementedError, match="with a list of strategies is not supported"):
        verb([First, Second], capital=LAKH, workers=1)


@pytest.mark.parametrize(
    "run",
    [
        lambda: bt.backtest(First, capital=LAKH, pool="separate"),
        lambda: bt.optimize(First, capital=LAKH, qty=[100, 200], account_capital=LAKH, workers=1),
    ],
    ids=["backtest", "optimize"],
)
def test_a_run_of_one_strategy_refuses_the_account_settings_before_loading_data(run):
    with pytest.raises(ConfigError, match="set the account of a list of strategies"):
        run()


def test_one_strategy_result_of_a_portfolio_cannot_rerun_alone(contention):
    pf, _ = contention
    member = pf.strategy("First")
    with pytest.raises(NotImplementedError, match="result.rerun.. of First, one strategy"):
        member.rerun()
    with pytest.raises(NotImplementedError, match="result.robustness.. of First, one strategy"):
        member.robustness(checks="quick")


def test_a_strategy_the_run_does_not_have_is_a_key_error_naming_them(contention):
    pf, _ = contention
    with pytest.raises(KeyError, match=r"no strategy 'Third' in this run; it has \['First'"):
        pf.strategy("Third")


def test_a_list_gives_a_portfolio_result_and_a_class_a_result(contention, alone):
    pf, _ = contention
    assert isinstance(pf, bt.PortfolioResult)
    assert all(isinstance(result, bt.Result) for result in (*pf.results.values(), *alone.values()))
