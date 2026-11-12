"""In a strategy on two symbols of one exchange, every exit is priced from its own
instrument's bars, never from the bars of the instrument another leg entered on
(01-requirements 6.4: basket and pairs runners pricing exits as entries is one of the
pitfalls the engine must not repeat).

Two NSE shares on 2026-11-04, a 1-minute MIS strategy on 1-minute bars, no charges and
no slippage. The two series are far apart, so a price taken from the wrong one cannot
pass for the right one. The bar opening k minutes after 09:15 is:

    SBIN  open 800 + k,   high open + 2, low open - 2, close open + 0.50
    INFY  open 1500 - 2k, high open + 3, low open - 3, close open - 1

The strategy buys 10 SBIN and sells 5 INFY at the close of bar 0, so both enter at the
open of bar 1: SBIN at 801.00 and INFY at 1,498.00.
"""

import datetime as dt

import pandas as pd
import pytest

import openalgo.backtest as bt

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
SESSION_OPEN = dt.datetime(2026, 11, 4, 9, 15, tzinfo=IST)
MINUTES = 15
SBIN_UNITS, INFY_UNITS = 10, 5


def minute(k: int) -> dt.datetime:
    return SESSION_OPEN + dt.timedelta(minutes=k)


def series(first_open: float, step: float, reach: float, close_offset: float) -> pd.DataFrame:
    opens = [first_open + step * k for k in range(MINUTES)]
    frame = pd.DataFrame(
        {
            "open": opens,
            "high": [price + reach for price in opens],
            "low": [price - reach for price in opens],
            "close": [price + close_offset for price in opens],
            "volume": 1_000.0,
        },
        index=pd.DatetimeIndex([minute(k) for k in range(MINUTES)], name="timestamp"),
    )
    return frame


def two_series(sbin_overrides: dict[int, tuple] | None = None) -> dict[str, pd.DataFrame]:
    sbin = series(800.0, 1.0, 2.0, 0.5)
    for k, (o, h, low, c) in (sbin_overrides or {}).items():
        sbin.iloc[k] = [o, h, low, c, 1_000.0]
    return {"SBIN": sbin, "INFY": series(1500.0, -2.0, 3.0, -1.0)}


def pair_strategy(sbin_entry: dict, infy_entry: dict, exits: dict[int, tuple[str, ...]]):
    """Buys SBIN with `sbin_entry` and sells INFY with `infy_entry` at the close of bar
    0; at the close of bar n exits the symbols named in `exits[n]`."""

    class Pair(bt.Strategy):
        symbols, exchange, interval = ("SBIN", "INFY"), "NSE", "1m"

        def on_bar(self, bar):
            count = len(self.data["SBIN"].close)
            if count == 1:
                self.buy(qty=SBIN_UNITS, symbol="SBIN", **sbin_entry)
                self.sell(qty=INFY_UNITS, symbol="INFY", **infy_entry)
            for symbol in exits.get(count - 1, ()):
                self.exit(symbol)

    return Pair


def run(strategy, data=None, **settings) -> bt.Result:
    arguments = {
        "data": data or two_series(),
        "capital": 1_000_000,
        "costs": None,
        "slippage": 0,
        "base": "1m",
        "product": "MIS",
        "end": minute(MINUTES - 1),
        **settings,
    }
    return bt.backtest(strategy, **arguments)


def fills_by_symbol(result: bt.Result) -> dict[str, list[tuple]]:
    found: dict[str, list[tuple]] = {}
    for _, fill in result.fills.iterrows():
        found.setdefault(fill["symbol"], []).append(
            (fill["time"], fill["side"], fill["price_rupees"], fill["reason"])
        )
    return found


def test_both_entries_fill_at_their_own_opens():
    fills = fills_by_symbol(run(pair_strategy({}, {}, {})))
    assert fills["SBIN"][0] == (minute(1), "BUY", 801.0, "entry")
    assert fills["INFY"][0] == (minute(1), "SELL", 1498.0, "entry")


def test_exits_in_one_slice_each_fill_at_their_own_open():
    # Both exit at the close of bar 3, so at the open of bar 4: SBIN 804.00, INFY 1,492.00.
    fills = fills_by_symbol(run(pair_strategy({}, {}, {3: ("SBIN", "INFY")})))
    assert fills["SBIN"][1] == (minute(4), "SELL", 804.0, "signal")
    assert fills["INFY"][1] == (minute(4), "BUY", 1492.0, "signal")


def test_exits_at_different_bars_each_fill_at_their_own_open():
    # SBIN at the open of bar 3 (803.00), INFY at the open of bar 7 (1,486.00).
    fills = fills_by_symbol(run(pair_strategy({}, {}, {2: ("SBIN",), 6: ("INFY",)})))
    assert fills["SBIN"][1] == (minute(3), "SELL", 803.0, "signal")
    assert fills["INFY"][1] == (minute(7), "BUY", 1486.0, "signal")


def test_the_trades_carry_each_instruments_own_entry_and_exit_prices():
    # SBIN long 801.00 to 804.00 on 10: 30.00; INFY short 1,498.00 to 1,492.00 on 5: 30.00.
    trades = run(pair_strategy({}, {}, {3: ("SBIN", "INFY")})).trades.set_index("symbol")
    assert list(trades.loc["SBIN", ["entry_price_rupees", "exit_price_rupees"]]) == [801.0, 804.0]
    assert list(trades.loc["INFY", ["entry_price_rupees", "exit_price_rupees"]]) == [1498.0, 1492.0]
    assert list(trades["gross_pnl_rupees"]) == [30.0, 30.0]


def test_a_stop_and_a_target_fill_at_their_own_instruments_levels():
    # SBIN's stop is 801.00 - 5.00 = 796.00; its bar 3 is changed to dip to 795.00, so the
    # stop fills there at 796.00. INFY's target is 1,498.00 - 10.00 = 1,488.00; INFY's bar
    # 5 (open 1,490.00, low 1,487.00) is the first to reach it.
    strategy = pair_strategy({"stop": bt.points(5)}, {"target": bt.points(10)}, {})
    data = two_series({3: (802.0, 803.0, 795.0, 796.0)})
    fills = fills_by_symbol(run(strategy, data))
    assert fills["SBIN"][1] == (minute(3), "SELL", 796.0, "leg_sl")
    assert fills["INFY"][1] == (minute(5), "BUY", 1488.0, "leg_target")


def test_the_square_off_closes_each_position_at_its_own_open():
    # A 09:20 square-off fills both at the open of the 09:20 bar: SBIN 805.00, INFY
    # 1,490.00.
    fills = fills_by_symbol(run(pair_strategy({}, {}, {}), square_off="09:20"))
    assert fills["SBIN"][1] == (minute(5), "SELL", 805.0, "scheduler")
    assert fills["INFY"][1] == (minute(5), "BUY", 1490.0, "scheduler")


@pytest.mark.parametrize(
    ("bar", "expected"),
    [
        # Long 10 SBIN from 801.00 and short 5 INFY from 1,498.00, marked at each close.
        (1, 10 * (801.5 - 801.0) + 5 * (1498.0 - 1497.0)),
        (2, 10 * (802.5 - 801.0) + 5 * (1498.0 - 1495.0)),
    ],
)
def test_each_open_position_is_marked_at_its_own_close(bar, expected):
    result = run(pair_strategy({}, {}, {}))
    assert result.equity.loc[minute(bar + 1), "unrealized_rupees"] == expected
