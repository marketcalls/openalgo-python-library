"""The run arguments the delayed-orders and liquidity checks rerun with (design 17,
sections 2.14 and 2.15): `bt.backtest(fill=)` models and `delay_entries=`,
`delay_exits=`. Prices are worked by hand from own daily bars, with no charges and no
slippage: bar k opens at `OPENS[k]` and closes 2 rupees above its open.
"""

import pytest
from robust_runs import FREE, opens_bars

import openalgo.backtest as bt
from openalgo.backtest.errors import ConfigError

OPENS = [500.0, 501.0, 503.0, 506.0, 510.0, 515.0]


def bars():
    data = opens_bars(OPENS)
    data["close"] = data["open"] + 2.0
    data["high"] = data["open"] + 3.0
    return data


class BuyFirstSellThird(bt.Strategy):
    """Buys 10 shares at the close of bar 0 and exits at the close of bar 2."""

    symbol, exchange, interval = "SBIN", "NSE", "D"

    def init(self):
        self.bar_number = -1

    def on_bar(self, bar):
        self.bar_number += 1
        if self.bar_number == 0:
            self.buy(qty=10)
        elif self.bar_number == 2:
            self.exit()


def fill_prices(**settings) -> list[float]:
    result = bt.backtest(BuyFirstSellThird, data=bars(), **FREE, **settings)
    return list(result.fills["price_rupees"])


# (fill, entry price, exit price): the next bar after a decision at bar 0 is bar 1 (open
# 501, high 504, low 500, close 503) and after bar 2 it is bar 3 (506, 509, 505, 508).
FILLS = [
    ("next_open", 501.0, 506.0),
    ("next_close", 503.0, 508.0),
    ("next_mid", 502.0, 507.0),
    ("next_ohlc4", 502.0, 507.0),
    # with the bar after: bar 2 (503, 506, 502, 505) and bar 4 (510, 513, 509, 512)
    ("next_close_2", 504.0, 510.0),
    ("next_mid_2", 503.0, 509.0),
    ("next_ohlc4_2", 503.0, 509.0),
]


@pytest.mark.parametrize(("fill", "entry", "exit_"), FILLS)
def test_a_fill_model_prices_the_strategys_orders_at_its_price_of_the_next_bar(fill, entry, exit_):
    assert fill_prices(fill=fill) == [entry, exit_]


def test_a_held_entry_fills_a_bar_later():
    # Decided at bar 0's close and held one bar: it fills at bar 2's open, 503.
    assert fill_prices(delay_entries=1) == [503.0, 506.0]


def test_a_held_exit_fills_a_bar_later():
    # Decided at bar 2's close and held two bars: it fills at bar 5's open, 515.
    assert fill_prices(delay_exits=2) == [501.0, 515.0]


def test_a_run_that_holds_its_orders_says_so_and_no_other_does():
    held = bt.backtest(BuyFirstSellThird, data=bars(), **FREE, delay_entries=1)
    assert held.assumptions["delay"].startswith("entries held 1 and exits 0 strategy bars")
    plain = bt.backtest(BuyFirstSellThird, data=bars(), **FREE)
    assert "delay" not in plain.assumptions
    assert plain.assumptions["fill"] == "next_open"


@pytest.mark.parametrize("value", [-1, 1.5, "1", True])
def test_a_delay_is_a_whole_number_of_strategy_bars(value):
    with pytest.raises(ConfigError, match="delay_entries takes a whole number of strategy bars"):
        bt.backtest(BuyFirstSellThird, data=bars(), **FREE, delay_entries=value)


def test_an_unknown_fill_names_every_model():
    with pytest.raises(ConfigError, match="next_ohlc4_2"):
        bt.backtest(BuyFirstSellThird, data=bars(), **FREE, fill="next_vwap")
