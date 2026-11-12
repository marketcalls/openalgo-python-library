"""The cash a delivery (CNC) account can spend (04-execution-and-market-model 5): what
the margin check at the fill and `self.cash` count.

Delivery is paid in full, so a holding's open profit is not cash until it is sold and
its open loss takes nothing from the cash already left. The proceeds of selling
holdings are credited for trading by the dated rule of `sale_credit_defaults.csv`: all
of them on the day before 2020-12-01 and from 2024-10-07, 80 percent on the day from
2020-12-01 to 2024-10-06 with the rest on the next trading day. Units bought and sold
on one day are an intraday trade, whose proceeds are credited in full.

Every run: two NSE shares AAA and BBB (registered in that order, so at one open AAA's
order fills first), capital 10,00,000, no charges, no slippage. Daily bars open and
close at the prices given (high and low 10.00 either side), so an order decided at a
close fills at the next session's open. Expected values are worked by hand in each
test.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import openalgo.backtest as bt

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
CAPITAL = 1_000_000
SYMBOLS = ("AAA", "BBB")


def sessions(start: str, count: int) -> list[dt.date]:
    """The first `count` NSE sessions of the built-in calendar from `start`."""
    days = bt.calendar.default_calendar().sessions("NSE", start, "2026-12-31")["date"]
    return list(days)[:count]


def daily(days: list[dt.date], opens: list[float], closes: list[float]) -> pd.DataFrame:
    """One bar per session, stamped 09:15 IST."""
    index = pd.DatetimeIndex(
        [dt.datetime.combine(day, dt.time(9, 15), tzinfo=IST) for day in days], name="timestamp"
    )
    opens_, closes_ = np.array(opens, dtype=float), np.array(closes, dtype=float)
    columns = {
        "open": opens_,
        "high": np.maximum(opens_, closes_) + 10.0,
        "low": np.minimum(opens_, closes_) - 10.0,
        "close": closes_,
        "volume": np.full(len(days), 1_000.0),
    }
    return pd.DataFrame(columns, index=index)


def flat(days: list[dt.date], price: float = 1_000.0) -> pd.DataFrame:
    """Bars that open and close at `price` every session."""
    return daily(days, [price] * len(days), [price] * len(days))


def scripted(plan: dict[int, list[tuple]], interval: str = "D") -> type[bt.Strategy]:
    """A CNC strategy on AAA and BBB that, after its n-th bar, calls `self.<method>` with
    each (method, symbol, keyword arguments) of `plan[n]` and keeps the cash it saw."""

    class Scripted(bt.Strategy):
        symbols, exchange, product = SYMBOLS, "NSE", "CNC"
        allow_averaging_down = True

        def init(self):
            self.cash_seen = []

        def on_bar(self, bars):
            self.cash_seen.append(self.cash)
            for method, symbol, arguments in plan.get(len(self.data["AAA"].close), ()):
                getattr(self, method)(symbol=symbol, **arguments)

    Scripted.interval = interval
    return Scripted


def run(strategy: type[bt.Strategy], data: dict[str, pd.DataFrame], **settings):
    arguments = {"data": data, "capital": CAPITAL, "costs": None, "slippage": 0, **settings}
    return bt.backtest(strategy, **arguments)


def outcome(result, symbol: str, side: str = "BUY") -> list[tuple]:
    """(units, status, reject code) of the orders on one symbol and side."""
    orders = result.orders[(result.orders["symbol"] == symbol) & (result.orders["side"] == side)]
    codes = [None if pd.isna(code) else code for code in orders["reject_code"]]
    return list(zip(orders["quantity_units"], orders["status"], codes, strict=True))


# ----- open P&L of delivery holdings ---------------------------------------------------


def test_a_holdings_open_profit_is_not_cash():
    # 900 AAA bought at 1,000.00 leave 1,00,000 of cash. AAA closes the next session at
    # 1,100.00 (an open profit of 90,000); 150 BBB at 1,000.00 need 1,50,000, more than
    # the 1,00,000 of cash, so the buy is refused at its fill.
    days = sessions("2026-11-02", 3)
    data = {
        "AAA": daily(days, [1_000, 1_000, 1_100], [1_000, 1_100, 1_100]),
        "BBB": flat(days),
    }
    plan = {1: [("buy", "AAA", {"qty": 900})], 2: [("buy", "BBB", {"qty": 150})]}
    result = run(scripted(plan), data)
    assert outcome(result, "BBB") == [(150.0, "Rejected", "margin")]


def test_a_holdings_open_loss_takes_nothing_from_the_cash_left():
    # 900 AAA bought at 1,000.00 leave 1,00,000 of cash. AAA closes the next session at
    # 900.00 (an open loss of 90,000); 90 BBB at 1,000.00 need 90,000 of the 1,00,000.
    days = sessions("2026-11-02", 3)
    data = {
        "AAA": daily(days, [1_000, 1_000, 900], [1_000, 900, 900]),
        "BBB": flat(days),
    }
    plan = {1: [("buy", "AAA", {"qty": 900})], 2: [("buy", "BBB", {"qty": 90})]}
    result = run(scripted(plan), data)
    assert outcome(result, "BBB") == [(90.0, "Filled", None)]


def test_the_cash_a_strategy_sees_is_the_cash_left_after_paying_for_its_holdings():
    # After the 900 AAA fill at 1,000.00, at the close of 1,100.00: 1,00,000 of cash
    # (equity is 10,90,000 with the open profit).
    days = sessions("2026-11-02", 3)
    data = {
        "AAA": daily(days, [1_000, 1_000, 1_100], [1_000, 1_100, 1_100]),
        "BBB": flat(days),
    }
    strategy = scripted({1: [("buy", "AAA", {"qty": 900})]})
    seen = []
    strategy.on_stop = lambda self: seen.extend(self.cash_seen)
    run(strategy, data)
    assert seen == [1_000_000.0, 100_000.0, 100_000.0]


# ----- proceeds of selling holdings ----------------------------------------------------


def sale_then_buy(start: str, bbb_units: float) -> bt.Result:
    """900 AAA bought at the second session's open (1,00,000 of cash left), sold whole at
    the fourth session's open, where `bbb_units` BBB are bought too (AAA fills first)."""
    days = sessions(start, 5)
    data = {"AAA": flat(days), "BBB": flat(days)}
    plan = {
        1: [("buy", "AAA", {"qty": 900})],
        3: [("exit", "AAA", {}), ("buy", "BBB", {"qty": bbb_units})],
    }
    return run(scripted(plan), data)


def test_from_december_2020_to_october_2024_80_percent_of_a_sale_is_cash_the_same_day():
    # 2023: the sale of 900 AAA at 1,000.00 brings 9,00,000, of which 80 percent,
    # 7,20,000, is cash the same day: 1,00,000 + 7,20,000 = 8,20,000. 800 BBB (8,00,000)
    # fill; 830 BBB (8,30,000) do not.
    assert outcome(sale_then_buy("2023-03-01", 800), "BBB") == [(800.0, "Filled", None)]
    assert outcome(sale_then_buy("2023-03-01", 830), "BBB") == [(830.0, "Rejected", "margin")]


def test_from_october_2024_the_whole_sale_is_cash_the_same_day():
    # 2025: all 9,00,000 are credited: 1,00,000 + 9,00,000 = 10,00,000 cover 990 BBB.
    assert outcome(sale_then_buy("2025-03-03", 990), "BBB") == [(990.0, "Filled", None)]


def test_before_december_2020_the_whole_sale_is_cash_the_same_day():
    assert outcome(sale_then_buy("2019-03-01", 990), "BBB") == [(990.0, "Filled", None)]


def test_the_rest_of_a_sale_is_cash_the_next_trading_day():
    # 2023: after the sale and 800 BBB, 20,000 of cash and 1,80,000 withheld; the next
    # session 190 BBB (1,90,000) fill from the 2,00,000.
    days = sessions("2023-03-01", 6)
    data = {"AAA": flat(days), "BBB": flat(days)}
    plan = {
        1: [("buy", "AAA", {"qty": 900})],
        3: [("exit", "AAA", {}), ("buy", "BBB", {"qty": 800})],
        4: [("buy", "BBB", {"qty": 190})],
    }
    result = run(scripted(plan), data)
    assert outcome(result, "BBB") == [(800.0, "Filled", None), (190.0, "Filled", None)]


def test_the_cash_seen_at_the_close_of_a_sale_day_leaves_out_the_part_not_yet_credited():
    # 2023: at the close of the sale day, 20,000 of cash (1,80,000 are credited the next
    # day); at the next close, 2,00,000.
    days = sessions("2023-03-01", 6)
    data = {"AAA": flat(days), "BBB": flat(days)}
    plan = {
        1: [("buy", "AAA", {"qty": 900})],
        3: [("exit", "AAA", {}), ("buy", "BBB", {"qty": 800})],
    }
    strategy = scripted(plan)
    seen = []
    strategy.on_stop = lambda self: seen.extend(self.cash_seen)
    run(strategy, data)
    assert seen[3:5] == [20_000.0, 200_000.0]


def test_a_sale_of_units_bought_the_same_day_is_credited_in_full():
    # 2023, CNC on 15-minute bars of a 1-minute base: 900 AAA bought at 09:30 and sold at
    # 09:45 at 1,000.00 are an intraday trade, so all 9,00,000 are cash again and 990
    # BBB (9,90,000) fill at 10:00.
    day = dt.date(2023, 3, 1)
    stamps = [
        dt.datetime.combine(day, dt.time(9, 15), tzinfo=IST) + dt.timedelta(minutes=k)
        for k in range(375)
    ]
    index = pd.DatetimeIndex(stamps, name="timestamp")
    bars = pd.DataFrame(
        {"open": 1_000.0, "high": 1_001.0, "low": 999.0, "close": 1_000.0, "volume": 1_000.0},
        index=index,
    )
    plan = {
        1: [("buy", "AAA", {"qty": 900})],
        2: [("exit", "AAA", {})],
        3: [("buy", "BBB", {"qty": 990})],
    }
    result = run(scripted(plan, interval="15m"), {"AAA": bars, "BBB": bars.copy()}, base="1m")
    assert outcome(result, "BBB") == [(990.0, "Filled", None)]


def test_a_run_that_credits_a_sale_by_an_unverified_row_warns():
    # Before 2020-12-01 the full credit rests on one dated observation, applied to every
    # earlier date: the row is unverified and the run says so.
    result = sale_then_buy("2019-03-01", 100)
    assert any("sale_credit row" in text and "unverified" in text for text in result.warnings)
