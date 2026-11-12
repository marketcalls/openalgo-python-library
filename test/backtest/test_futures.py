"""Futures strategies end to end (14-strategy-api 2.1; 04-execution 7; SPEC-19, SPEC-20,
SPEC-30; LEAD-04, LEAD-05, LEAD-16): the contracts of a rank from the reference tables,
the roll dates from the expiry calendar, the roll in a run, the spliced series the
strategy reads, the daily mark to market and settlement at expiry.

The runs use two NIFTY contracts, September 2024 (last trading day Thursday 2024-09-26)
and October 2024 (2024-10-31), lot 25, with daily bars from 2024-09-20 to 2024-09-30 on
the built-in calendar:

    September  09-20 (24950, 25050, 24900, 25000)   09-23 (25000, 25150, 24950, 25100)
               09-24 (25100, 25250, 25050, 25200)   09-25 (25200, 25300, 25150, 25250)
               09-26 (25250, 25350, 25200, 25300)
    October    09-25 (25300, 25400, 25250, 25350)   09-26 (25350, 25450, 25300, 25400)
               09-27 (25400, 25500, 25350, 25450)   09-30 (25450, 25550, 25400, 25500)

A strategy buys one lot at the close of 09-20 (filled at 25,000.00 on 09-23) and exits
at the close of 09-27 (filled on 09-30). With `roll=1` it rolls at the 09-25 open: the
spread of the two opens is 25,300.00 - 25,200.00 = 100.00.
"""

import dataclasses
import datetime as dt
import warnings

import pandas as pd
import pytest
from accounting_runs import daily_round_trip

import openalgo.backtest as bt
from openalgo import ta
from openalgo.backtest import _futures, reference
from openalgo.backtest.calendar import default_calendar
from openalgo.backtest.errors import ConfigError, ReferenceDataError
from openalgo.backtest.reference import Instrument

CALENDAR = default_calendar()

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
SEPTEMBER = "NIFTY26SEP24FUT"
OCTOBER = "NIFTY31OCT24FUT"
BARS = {
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


def _frame(rows: dict[int, tuple]) -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [dt.datetime(2024, 9, day, 9, 15, tzinfo=IST) for day in rows], name="timestamp"
    )
    frame = pd.DataFrame(list(rows.values()), index=index, columns=["open", "high", "low", "close"])
    frame["volume"] = 1_000.0
    return frame


def _contract(symbol: str, last_day: dt.date) -> Instrument:
    return Instrument(
        symbol=symbol,
        exchange="NFO",
        kind="future",
        lot_size=25,
        tick_size=0.05,
        expiry=last_day,
        underlying="NIFTY",
        settlement="cash",
    )


CONTRACTS = {
    SEPTEMBER: _contract(SEPTEMBER, dt.date(2024, 9, 26)),
    OCTOBER: _contract(OCTOBER, dt.date(2024, 10, 31)),
}


def _strategy(log: list) -> type:
    class Rolling(bt.Strategy):
        underlying, exchange, interval = "NIFTY", "NFO", "D"

        def init(self):
            self.mean = self.I(ta.sma, self.close, 2)

        def on_bar(self, bar):
            previous = self.close[-2] if len(self.close) > 1 else None
            mean = self.mean[-1] if len(self.close) > 1 else None
            log.append((bar.time.day, bar.symbol, bar.close, previous, mean))
            if bar.time.day == 20:
                self.buy(lots=1)
            elif bar.time.day == 27:
                self.exit()

    return Rolling


def _run(roll=1, log=None, **settings):
    log = [] if log is None else log
    return bt.backtest(
        _strategy(log),
        capital=1_000_000,
        data={symbol: _frame(rows) for symbol, rows in BARS.items()},
        instrument=CONTRACTS,
        start=dt.date(2024, 9, 20),
        end=dt.date(2024, 9, 30),
        costs=settings.pop("costs", None),
        slippage=0,
        roll=roll,
        **settings,
    )


# ---------------------------------------------------------------- contracts and roll dates


class _Plan:
    exchange = "NFO"
    start = dt.date(2024, 9, 2)
    end = dt.date(2025, 1, 31)


def _resolved(rank: str, roll: int | None):
    class Futures(bt.Strategy):
        underlying, exchange, interval = "NIFTY", "NFO", "D"
        expiry = rank

    plan = _futures.resolve_futures(
        Futures, _Plan, {"instrument": None, "roll": roll}, default_calendar()
    )
    return plan.series[0]


def test_the_current_month_rolls_one_session_before_each_last_trading_day():
    # NIFTY's monthly contracts from the expiry calendar: last Thursdays 2024-09-26,
    # 10-31, 11-28, 12-26 and 2025-01-30. The session before each is the roll session;
    # 2024-12-25 is a holiday, so December rolls on the 24th.
    series = _resolved("current_month", 1)
    assert [i.symbol for i in series.contracts] == [
        "NIFTY26SEP24FUT",
        "NIFTY31OCT24FUT",
        "NIFTY28NOV24FUT",
        "NIFTY26DEC24FUT",
        "NIFTY30JAN25FUT",
        "NIFTY27FEB25FUT",
    ]
    assert series.hand_over == (
        dt.date(2024, 9, 25),
        dt.date(2024, 10, 30),
        dt.date(2024, 11, 27),
        dt.date(2024, 12, 24),
        dt.date(2025, 1, 29),
        None,
    )


def test_roll_none_moves_on_at_the_session_after_each_expiry():
    # After 2024-10-31 the next session is the Muhurat session of 2024-11-01.
    series = _resolved("current_month", None)
    assert series.hand_over[:2] == (dt.date(2024, 9, 27), dt.date(2024, 11, 1))


def test_the_next_month_holds_the_second_contract_and_rolls_with_the_nearest():
    series = _resolved("next_month", 1)
    assert [i.symbol for i in series.contracts][:2] == ["NIFTY31OCT24FUT", "NIFTY28NOV24FUT"]
    assert series.hand_over[0] == dt.date(2024, 9, 25)


def test_each_contract_has_the_lot_size_of_its_expiry():
    # NIFTY's lot became 75 for contracts expiring after December 2024's revision
    # (the reference tables' lot_sizes rows): January 2025 still 25, February 75.
    series = _resolved("current_month", 1)
    lots = {i.symbol: i.lot_size for i in series.contracts}
    assert (lots["NIFTY30JAN25FUT"], lots["NIFTY27FEB25FUT"]) == (25, 75)


def test_a_contract_is_loaded_from_the_session_it_takes_over_to_its_last_trading_day():
    series = _resolved("current_month", 1)
    assert series.windows[:2] == (
        (dt.date(2024, 9, 2), dt.date(2024, 9, 26)),
        (dt.date(2024, 9, 25), dt.date(2024, 10, 31)),
    )


# The reference tables were last checked on 2026-11-12 (tables.csv); a row used for a
# later date warns. NIFTY's monthly contracts expire on 2026-10-27, 2026-11-24 (listed
# for Monday 2026-11-23) and 2026-12-29.
CHECKED_ON = dt.date(2026, 11, 12)


class _AutumnPlan:
    exchange = "NFO"
    start = dt.date(2026, 10, 1)


def _reference_warnings(monkeypatch, end: dt.date, roll: int) -> list[str]:
    """The ReferenceDataWarnings of resolving a current-month NIFTY run from 2026-10-01
    to `end`, on reference tables of its own (a table warns once per row)."""
    fresh = reference.ReferenceTables()
    monkeypatch.setattr(reference, "default_tables", lambda: fresh)
    plan = type("Plan", (_AutumnPlan,), {"end": end})

    class Futures(bt.Strategy):
        underlying, exchange, interval = "NIFTY", "NFO", "D"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _futures.resolve_futures(Futures, plan, {"instrument": None, "roll": roll}, CALENDAR)
    return [str(w.message) for w in caught if w.category is reference.ReferenceDataWarning]


def test_a_run_whose_contracts_are_all_checked_does_not_warn_about_the_look_ahead(monkeypatch):
    # The run needs October (rolled on 10-26) and November, held past its end on 11-10;
    # the look-ahead also lists December and January, which the run never uses.
    tables = reference.default_tables()
    assert [tables.table_as_of(name) for name in ("expiry_rules", "contract_dates")] == [
        CHECKED_ON,
        CHECKED_ON,
    ]
    assert _reference_warnings(monkeypatch, dt.date(2026, 11, 10), roll=1) == []


def test_a_roll_set_by_an_expiry_after_the_check_date_warns(monkeypatch):
    # roll=10 moves to December inside the run, ten sessions before November's
    # 2026-11-24 expiry, a date after the tables were checked.
    found = _reference_warnings(monkeypatch, dt.date(2026, 11, 12), roll=10)
    assert len(found) == 1
    assert "is being used for 2026-11-24" in found[0]


class _GivenFutures(bt.Strategy):
    underlying, exchange, interval = "NIFTY", "NFO", "D"


def _given_plan(end: dt.date) -> type:
    return type("Plan", (), {"exchange": "NFO", "start": dt.date(2024, 9, 20), "end": end})


def _resolve_given(given, end: dt.date = dt.date(2024, 9, 30)):
    """Resolve `_GivenFutures` with `instrument=given` and roll=1 from 2024-09-20."""
    args = {"instrument": given, "roll": 1}
    return _futures.resolve_futures(_GivenFutures, _given_plan(end), args, CALENDAR)


def test_a_run_past_the_roll_session_of_the_last_given_contract_needs_the_next_one():
    # October rolls (roll=1) at 2024-10-30, the session before its last trading day
    # 10-31. A run to 10-30 trades November from that session, which is not given.
    with pytest.raises(ReferenceDataError, match="no contract is known after NIFTY31OCT24FUT"):
        _resolve_given(CONTRACTS, end=dt.date(2024, 10, 30))


def test_a_run_that_ends_before_the_last_given_contracts_roll_session_holds_it():
    series = _resolve_given(CONTRACTS, end=dt.date(2024, 10, 29)).series[0]
    assert series.hand_over == (dt.date(2024, 9, 25), None)


@pytest.mark.parametrize(
    "given",
    [[CONTRACTS[SEPTEMBER], CONTRACTS[OCTOBER]], {SEPTEMBER: "lot 25"}],
    ids=["a list", "a dict of text"],
)
def test_futures_instrument_takes_instruments_only(given):
    with pytest.raises(ConfigError, match="instrument= takes an Instrument or a dict of them"):
        _resolve_given(given)


def test_a_futures_instrument_keyed_by_another_symbol_is_refused():
    swapped = {SEPTEMBER: CONTRACTS[OCTOBER], OCTOBER: CONTRACTS[SEPTEMBER]}
    with pytest.raises(ConfigError, match=f"the key '{SEPTEMBER}' holds the Instrument of"):
        _resolve_given(swapped)


@pytest.mark.parametrize(
    "stray",
    [
        dataclasses.replace(
            CONTRACTS[SEPTEMBER], symbol="BANKNIFTY26SEP24FUT", underlying="BANKNIFTY"
        ),
        dataclasses.replace(CONTRACTS[SEPTEMBER], symbol="NIFTY26SEP24FUT-BFO", exchange="BFO"),
        Instrument.cash_equity("NIFTY", "NSE"),
    ],
    ids=["another underlying", "another exchange", "not a future"],
)
def test_a_given_instrument_no_futures_series_of_the_run_can_use_is_refused(stray):
    with pytest.raises(ConfigError, match=f"instrument= gives {stray.symbol} "):
        _resolve_given({**CONTRACTS, stray.symbol: stray})


def test_next_month_with_roll_none_is_refused():
    with pytest.raises(ConfigError, match="roll=N"):
        _resolved("next_month", None)


def test_without_expiry_the_strategy_trades_the_current_month():
    assert _resolved(None, 1).contracts[0].symbol == "NIFTY26SEP24FUT"


def test_an_unknown_rank_is_refused():
    with pytest.raises(ConfigError, match="expiry"):
        _resolved("far_month", 1)


@pytest.mark.parametrize("roll", [-1, 1.5, True])
def test_roll_takes_a_whole_number_of_sessions(roll):
    with pytest.raises(ConfigError, match="roll"):
        _futures.check_roll(roll)


def test_underlyings_in_symbols_are_futures_series_and_contracts_are_not():
    class Pair(bt.Strategy):
        symbols, exchange, interval = ("HDFCBANK", "ICICIBANK"), "NFO", "D"

    class Contracts(bt.Strategy):
        symbols, exchange, interval = (SEPTEMBER,), "NFO", "D"

    assert _futures.underlying_names(Pair, "NFO") == ("HDFCBANK", "ICICIBANK")
    assert _futures.underlying_names(Contracts, "NFO") == ()


def _refusal(underlying: str, exchange: str, start: dt.date) -> str:
    """The ReferenceDataError message of a futures run on `underlying` with no
    `instrument=`, from `start` for three weeks."""

    declared = {"underlying": underlying, "exchange": exchange, "interval": "D"}
    strategy = type("Futures", (bt.Strategy,), declared)
    with pytest.raises(ReferenceDataError) as caught:
        bt.backtest(strategy, capital=1_000_000, start=start, end=start + dt.timedelta(days=21))
    return str(caught.value)


def test_an_mcx_underlying_without_instrument_says_to_pass_each_contract():
    # The built-in reference tables have no MCX expiry rules, so the run cannot list
    # CRUDEOIL's monthly contracts on its own.
    message = _refusal("CRUDEOIL", "MCX", dt.date(2026, 10, 1))
    assert message.startswith(
        "CRUDEOIL futures on MCX without instrument= is not supported in this version: "
        "the reference tables have no expiry rule for CRUDEOIL on MCX, so pass every "
        "contract the run trades with instrument="
    )
    assert (
        'bt.backtest(Strategy, ..., instrument={"CRUDEOIL18NOV26FUT": Instrument('
        'symbol="CRUDEOIL18NOV26FUT", exchange="MCX", kind="future", underlying="CRUDEOIL", '
        "expiry=datetime.date(2026, 11, 18), lot_size=..., tick_size=...), ...})"
    ) in message


def test_an_underlying_whose_rules_miss_the_run_dates_says_which_dates_they_cover():
    # NIFTY's monthly expiry rules start on 2019-01-01, after a run in January 2015.
    message = _refusal("NIFTY", "NFO", dt.date(2015, 1, 1))
    assert "is not supported in this version" not in message
    assert "Rows for this key cover 2019-01-01 to the present" in message


def test_an_underlying_on_a_cash_exchange_is_refused():
    class Cash(bt.Strategy):
        underlying, exchange, interval = "NIFTY", "NSE", "D"

    with pytest.raises(ConfigError, match="derivatives exchange"):
        _futures.underlying_names(Cash, "NSE")


# ---------------------------------------------------------------- a rolled run


def test_the_roll_fills_carry_the_contracts_and_reason_roll():
    result = _run()
    fills = result.fills[["symbol", "side", "price_rupees", "reason"]].values.tolist()
    assert fills == [
        [SEPTEMBER, "BUY", 25000.0, "entry"],
        [SEPTEMBER, "SELL", 25200.0, "roll"],
        [OCTOBER, "BUY", 25300.0, "roll"],
        [OCTOBER, "SELL", 25450.0, "signal"],
    ]
    rolls = result.rolls
    assert list(rolls[["from_symbol", "to_symbol", "spread_rupees"]].iloc[0]) == [
        SEPTEMBER,
        OCTOBER,
        100.0,
    ]


def test_a_switch_and_a_settlement_are_dated_in_ist():
    # The roll fills at the 09-25 open; the first posting is at the 09-23 session close.
    result = _run()
    switch = result.rolls.iloc[0]
    assert (switch["date"], switch["time"].tz.utcoffset(None)) == (
        dt.date(2024, 9, 25),
        dt.timedelta(hours=5, minutes=30),
    )
    assert "day" not in result.rolls.columns
    first = result.settlements.iloc[0]
    assert (first["date"], first["time"].date()) == (dt.date(2024, 9, 23),) * 2


@pytest.mark.parametrize("table", ["rolls", "settlements"])
def test_the_time_column_is_an_ist_datetime_in_a_run_with_no_futures(table):
    # A stock run has no switch and no settlement: the empty column keeps its type, so
    # `.dt` works on it as on any other run's.
    times = getattr(daily_round_trip(costs=None), table)["time"]
    assert times.empty
    assert isinstance(times.dtype, pd.DatetimeTZDtype)
    assert times.dt.tz.utcoffset(None) == dt.timedelta(hours=5, minutes=30)


def test_on_bar_sees_each_contracts_own_bar_and_the_series_re_adjusted_at_the_roll():
    # On 09-24: close[-2] is 25,100.00 and sma(2) 25,150.00. On 09-25 (the roll): the
    # bar is October's, close[-2] is 25,200.00 + 100.00 = 25,300.00 and sma(2)
    # (25,300 + 25,350) / 2 = 25,325.00. The log keeps the values as used.
    log = []
    _run(log=log)
    assert log[2] == (24, SEPTEMBER, 25200.0, 25100.0, 25150.0)
    assert log[3] == (25, OCTOBER, 25350.0, 25300.0, 25325.0)


def test_the_trades_are_split_at_the_roll_and_their_pnl_adds_up():
    # September: (25,200 - 25,000) x 25 = 5,000; October: (25,450 - 25,300) x 25 = 3,750.
    result = _run()
    trades = result.trades[["symbol", "exit_reason", "gross_pnl_rupees"]].values.tolist()
    assert trades == [[SEPTEMBER, "roll", 5000.0], [OCTOBER, "signal", 3750.0]]
    assert result.gross_pnl["total"] == 8750.0


def test_the_daily_mark_to_market_posts_each_sessions_change():
    # 09-23: (25,100 - 25,000) x 25 = 2,500; 09-24: 2,500; 09-25: October from its fill
    # 25,300 to its close 25,350, 1,250 (September posted all it made); 09-26 and 09-27:
    # 1,250 each. The postings add up to the gross P&L.
    result = _run()
    rows = result.settlements[["date", "symbol", "amount_rupees"]].values.tolist()
    assert rows == [
        [dt.date(2024, 9, 23), SEPTEMBER, 2500.0],
        [dt.date(2024, 9, 24), SEPTEMBER, 2500.0],
        [dt.date(2024, 9, 25), OCTOBER, 1250.0],
        [dt.date(2024, 9, 26), OCTOBER, 1250.0],
        [dt.date(2024, 9, 27), OCTOBER, 1250.0],
    ]
    assert result.settlements["amount_rupees"].sum() == result.gross_pnl["total"]
    assert result.settlements["last_close_used"].all()


# These September 2024 trades are charged at the built-in rates of their trade dates.
CHARGED = bt.costs.CostModel()


def test_both_roll_legs_are_charged():
    result = _run(costs=CHARGED)
    charged = result.fills.loc[result.fills["reason"] == "roll", "costs_rupees"]
    assert len(charged) == 2 and (charged > 0).all()


def test_positions_are_addressed_by_the_underlying():
    seen = []

    class Watch(_strategy([])):
        def on_bar(self, bar):
            super().on_bar(bar)
            seen.append((bar.time.day, list(self.positions), self.instrument().symbol))

    bt.backtest(
        Watch,
        capital=1_000_000,
        data={symbol: _frame(rows) for symbol, rows in BARS.items()},
        instrument=CONTRACTS,
        start=dt.date(2024, 9, 20),
        end=dt.date(2024, 9, 30),
        costs=None,
        slippage=0,
    )
    assert seen[2] == (24, ["NIFTY"], SEPTEMBER)
    assert seen[3] == (25, ["NIFTY"], OCTOBER)


def _run_class(strategy: type):
    return bt.backtest(
        strategy,
        capital=1_000_000,
        data={symbol: _frame(rows) for symbol, rows in BARS.items()},
        instrument=CONTRACTS,
        start=dt.date(2024, 9, 20),
        end=dt.date(2024, 9, 30),
        costs=None,
        slippage=0,
    )


def test_a_truth_test_of_the_series_in_init_is_refused():
    class Branches(bt.Strategy):
        underlying, exchange, interval = "NIFTY", "NFO", "D"

        def init(self):
            if self.close > self.open:
                self.state["rising"] = True

        def on_bar(self, bar):
            pass

    rising = r"\(NIFTY\.close > NIFTY\.open\) was tested for truth in init"
    with pytest.raises(ConfigError, match=rising):
        _run_class(Branches)


def test_on_bar_reads_a_condition_of_the_series():
    seen = []

    class Reads(bt.Strategy):
        underlying, exchange, interval = "NIFTY", "NFO", "D"

        def on_bar(self, bar):
            seen.append(bool(self.close > 0))

    _run_class(Reads)
    # Seven sessions from 09-20 to 09-30, each close above zero.
    assert seen == [True] * 7


# ---------------------------------------------------------------- held to expiry


def test_with_roll_none_the_position_is_settled_at_expiry_on_the_expiry_session():
    # Held to 09-26 and settled at its close 25,300.00 in step 8, flagged; the strategy
    # trades October from 09-27, so its exit order finds no October position.
    result = _run(roll=None)
    last = result.fills.iloc[-1]
    assert (last["symbol"], last["price_rupees"], last["reason"], last["step"]) == (
        SEPTEMBER,
        25300.0,
        "expiry",
        8,
    )
    assert last["time"].date() == dt.date(2024, 9, 26)
    assert list(last["flags"]) == ["filled_at_last_close"]
    assert result.settlements["kind"].iloc[-1] == "expiry"


def test_with_roll_none_the_series_is_adjusted_by_the_last_trading_days_closes():
    # October 25,400.00 less September 25,300.00 at the close of 09-26.
    result = _run(roll=None)
    switch = result.rolls.iloc[0]
    assert (switch["spread_rupees"], bool(switch["rolls_positions"])) == (100.0, False)
    assert switch["time"].date() == dt.date(2024, 9, 27)


def test_settlement_at_expiry_is_not_charged():
    result = _run(roll=None, costs=CHARGED)
    assert result.fills.iloc[-1]["costs_rupees"] == 0.0
