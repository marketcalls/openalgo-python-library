"""`reinvest` and the caps shared by every sizer, `size_caps = bt.size.caps(...)`
(12-position-sizing 5, FR-SIZE-07).

Bars are those of `accounting_runs`: one NSE share on 2026-11-04 at 1,000.00 until
09:45 IST and 1,100.00 after. The strategies trade 15-minute bars on MIS with no
charges, no slippage and whole shares, capital 10,00,000: a decision after the first
bar is taken at 1,000.00 and fills at 1,000.00.
"""

import pytest
from accounting_runs import CAPITAL, session_bars

import openalgo.backtest as bt
from openalgo.backtest.errors import ConfigError


def run(strategy: type) -> bt.Result:
    return bt.backtest(
        strategy,
        data=session_bars(),
        capital=CAPITAL,
        base="1m",
        costs=None,
        slippage=0,
        quantity_mode="venue",
    )


def first_entry(controls: dict, **order) -> type:
    """A strategy with the class attributes `controls` that buys with `order` after its
    first bar."""

    class First(bt.Strategy):
        symbol, exchange, interval, product = "SBIN", "NSE", "15m", "MIS"

        def on_bar(self, bar):
            if len(self.close) == 1:
                self.buy(**order)

    for name, value in controls.items():
        setattr(First, name, value)
    return First


def entry_units(result: bt.Result) -> float:
    return result.fills.loc[0, "quantity_units"]


HALF = {"size": bt.size.percent(50)}


class Compounding(bt.Strategy):
    """Makes 50,000.00 on 500 shares (1,000.00 to 1,100.00), then buys with half the
    equity at 1,100.00."""

    symbol, exchange, interval, product = "SBIN", "NSE", "15m", "MIS"

    def on_bar(self, bar):
        if len(self.close) == 1:
            self.buy(qty=500)
        elif len(self.close) == 2:
            self.exit()
        elif len(self.close) == 3:
            self.buy(size=bt.size.percent(50))


def test_reinvesting_sizes_from_the_current_equity():
    # Half of 10,50,000 at 1,100.00 is 477.3 shares: 477.
    result = run(Compounding)
    assert result.fills.loc[2, "quantity_units"] == 477.0


def test_without_reinvest_sizes_come_from_the_initial_capital():
    # Half of the initial 10,00,000 at 1,100.00 is 454.5 shares: 454.
    strategy = type("Fixed", (Compounding,), {"reinvest": False})
    result = run(strategy)
    assert result.fills.loc[2, "quantity_units"] == 454.0


@pytest.mark.parametrize(
    ("caps", "units"),
    [
        (bt.size.caps(position=100), 100.0),
        (bt.size.caps(position=bt.size.lots(120)), 120.0),
        (bt.size.caps(position=bt.rupees(50_000)), 50.0),
        (bt.size.caps(position=bt.pct(10)), 100.0),
        (bt.size.caps(leverage=0.2), 200.0),
    ],
)
def test_a_position_cap_cuts_a_sized_entry(caps, units):
    # Half the equity is 500 shares; 50,000 rupees is 50, 10 percent of 10,00,000 is
    # 100, and 0.2x the equity is 200.
    result = run(first_entry({"size_caps": caps}, **HALF))
    assert entry_units(result) == units


def test_the_risk_cap_limits_the_loss_of_the_order_at_its_stop():
    # A stop 10.00 away and at most 2,000 rupees at risk: 200 shares.
    caps = bt.size.caps(risk=bt.rupees(2_000))
    result = run(first_entry({"size_caps": caps}, stop=bt.points(10), **HALF))
    assert entry_units(result) == 200.0


def test_a_risk_cap_skips_a_sized_order_without_a_stop():
    caps = bt.size.caps(risk=bt.pct(0.5))
    result = run(first_entry({"size_caps": caps}, **HALF))
    assert result.fills.empty
    assert list(result.skipped["reason"]) == ["cap"]


def test_caps_leave_an_explicit_quantity_as_asked():
    result = run(first_entry({"size_caps": bt.size.caps(position=100)}, qty=300))
    assert entry_units(result) == 300.0


@pytest.mark.parametrize(
    "controls",
    [
        {"reinvest": "yes"},
        {"size_caps": 5},
        {"size_caps": {"position": 100}},
    ],
)
def test_unusable_controls_are_refused(controls):
    with pytest.raises(ConfigError):
        run(first_entry(controls, **HALF))


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"position": -1},
        {"position": bt.points(5)},
        {"leverage": 0},
        {"leverage": True},
        {"risk": bt.pct(100)},
        {"risk": bt.ticks(3)},
    ],
)
def test_caps_refuse_what_they_cannot_use(arguments):
    with pytest.raises(ConfigError):
        bt.size.caps(**arguments)


def test_the_assumptions_state_reinvest_and_the_caps():
    controls = {"reinvest": False, "size_caps": bt.size.caps(position=bt.pct(10), leverage=1)}
    assumptions = run(first_entry(controls, **HALF)).assumptions
    assert assumptions["reinvest"] is False
    assert assumptions["size_caps"] == "position 10 percent of equity; leverage 1x"
