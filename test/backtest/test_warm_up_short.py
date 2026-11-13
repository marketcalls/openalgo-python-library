"""A series shorter than its indicators need, and parameters the needs are keyed on
(decision D25, `_indicators`, `_warm_up`, `_warm_up_probe`).

An indicator whose series has fewer bars than its own window (`period`, `length`), or
than its warm-up when it has no single window, is not computed there; one its series
can compute but whose warm-up
is longer than the series is never ready. Either way the strategy makes no decision for
that symbol and a warning says so, while the other symbols of the run trade. Any other
error of an indicator still stops the run.

The bars are the seeded walks of `warm_up_bars`. The warm-ups used are those of the
engine's reference bars: 128 bars for a 100-bar simple average (a window of 64 cannot
compute it).
"""

from __future__ import annotations

import pytest
from warm_up_bars import SETTINGS, walk

import openalgo.backtest as bt
from openalgo import ta
from openalgo.backtest._warm_up_probe import params_key
from openalgo.backtest.errors import ConfigError, WarmUpWarning

SHORT = walk("2026-01-01", "2026-06-30", seed=2).iloc[:60]
LONG = walk("2020-01-01", "2026-06-30", seed=3)
# Indicators of more bars than SHORT has: each is longer than 60 bars, by its warm-up
# or by its period.
TOO_LONG = {
    "sma": lambda s: s.I(ta.sma, s.close, 100),
    "ema": lambda s: s.I(ta.ema, s.close, 100),
    "bbands": lambda s: s.I(ta.bbands, s.close, 100),
    "donchian": lambda s: s.I(ta.donchian, s.high, s.low, 100),
    "rsi": lambda s: s.I(ta.rsi, s.close, 14),
    "supertrend": lambda s: s.I(ta.supertrend, s.high, s.low, s.close, 10, 3),
    # No single window: its warm-up decides, and its slow average of 100 bars could not
    # be computed on 60.
    "macd": lambda s: s.I(ta.macd, s.close, 12, 100, 9),
}


def warm_up(result: bt.Result) -> dict:
    return result.manifest["arguments"]["warm_up"]


def reading(build) -> type[bt.Strategy]:
    class Reading(bt.Strategy):
        symbol, exchange, interval, product = "SBIN", "NSE", "D", "CNC"

        def init(self):
            self.line = build(self)

        def on_bar(self, bar):
            if self.position.is_flat:
                self.buy(qty=1)

    return Reading


@pytest.mark.parametrize("name", list(TOO_LONG))
def test_an_indicator_longer_than_its_series_makes_no_decision_and_says_so(name):
    result = bt.backtest(reading(TOO_LONG[name]), data=SHORT, **SETTINGS)
    assert result.orders.empty
    assert warm_up(result)["first_decision"] == "never"
    (warning,) = [note for note in result.warnings if note.startswith("The strategy made no")]
    assert f"{name}(SBIN." in warning


def test_an_indicator_over_values_of_your_own_shorter_than_its_period_makes_no_decision():
    # Its warm-up cannot be measured (an array of the user's own), so its period decides.
    volumes = SHORT.volume.to_numpy()

    class OwnWeights(bt.Strategy):
        symbol, exchange, interval, product = "SBIN", "NSE", "D", "CNC"

        def init(self):
            self.line = self.I(ta.vwma, self.close, volumes, 100)

        def on_bar(self, bar):
            self.buy(qty=1)

    result = bt.backtest(OwnWeights, data=SHORT, **SETTINGS)
    assert result.orders.empty
    assert warm_up(result)["first_decision"] == "never"


def test_any_other_error_of_an_indicator_still_stops_the_run():
    with pytest.raises(ConfigError, match="Period must be positive"):
        bt.backtest(reading(lambda s: s.I(ta.sma, s.close, 0)), data=SHORT, **SETTINGS)


class Pair(bt.Strategy):
    """Buys one share of each symbol it decides for, on its first decision."""

    symbols, exchange, interval, product = ("AAA", "BBB"), "NSE", "D", "CNC"

    def init(self):
        self.lines = {s: self.I(ta.sma, self.data[s].close, 100) for s in self.symbols}

    def on_bar(self, bars):
        for symbol in self.symbols:
            if self.positions[symbol].is_flat:
                self.buy(symbol=symbol, qty=1)


def test_a_symbol_too_short_for_its_indicator_is_left_out_and_the_others_trade():
    result = bt.backtest(Pair, data={"AAA": LONG, "BBB": SHORT}, start="2026-01-01", **SETTINGS)
    assert set(result.orders["symbol"]) == {"AAA"}
    assert warm_up(result)["first_decision"] is None
    assert warm_up(result)["no_decision"] == ["BBB"]


def test_the_orders_of_a_symbol_left_out_are_skipped_and_the_result_says_why():
    result = bt.backtest(Pair, data={"AAA": LONG, "BBB": SHORT}, start="2026-01-01", **SETTINGS)
    skipped = result.skipped
    assert set(zip(skipped["symbol"], skipped["reason"], strict=True)) == {("BBB", "not_ready")}
    (warning,) = [note for note in result.warnings if note.startswith("The strategy made no")]
    assert warning.startswith("The strategy made no decision for BBB: its indicators were never")
    assert "sma(BBB.close, 100) needs 128 D bars of BBB" in warning


def test_a_symbol_left_out_is_told_on_screen():
    with pytest.warns(WarmUpWarning) as caught:
        bt.backtest(Pair, data={"AAA": LONG, "BBB": SHORT}, start="2026-01-01", **SETTINGS)
    (warning,) = [w for w in caught if w.category is WarmUpWarning]
    assert str(warning.message) == (
        "The strategy made no decision for BBB: its indicators were never ready within its "
        "bars. See result.warnings for why."
    )


# ----- parameters the needs are keyed on -------------------------------------------------


class Ribbon(bt.Strategy):
    symbol, exchange, interval, product = "SBIN", "NSE", "D", "CNC"
    lengths = bt.param((10, 20, 30))

    def init(self):
        self.lines = [self.I(ta.ema, self.close, length) for length in self.lengths]


def test_a_list_parameter_runs_as_its_tuple_does():
    as_list, as_tuple = (
        bt.backtest(Ribbon, data=LONG, start="2026-01-01", params={"lengths": value}, **SETTINGS)
        for value in ([10, 20, 30], (10, 20, 30))
    )
    assert as_list.assumptions["warm_up"] == as_tuple.assumptions["warm_up"]


def test_a_key_of_list_and_dict_values_is_hashable():
    assert hash(params_key({"lengths": [10, 20], "weights": {"a": [1.5]}})) is not None


def test_parameters_in_another_order_of_items_are_the_same_key():
    assert params_key({"a": [1, 2], "b": 3}) == params_key({"b": 3, "a": [1, 2]})


@pytest.mark.parametrize(
    ("one", "other"),
    [
        ({"lengths": [10, 20]}, {"lengths": [20, 10]}),
        ({"lengths": [10, 20]}, {"lengths": (10, 20)}),
        ({"weights": {"a": 1}}, {"weights": {"a": 2}}),
    ],
)
def test_different_parameter_values_are_different_keys(one, other):
    assert params_key(one) != params_key(other)
