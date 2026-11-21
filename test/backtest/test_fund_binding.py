"""The fund of a portfolio run through the binding (`Fund`; rust/oa_py/src/backtest/fund.rs,
18-multi-strategy section 2): engines advanced in lockstep over one account, each strategy
deciding on its own engine at the pauses the fund hands it, condition strategies decided
in Rust, and the account's records.

The runs are backtests of the NSE day 2025-07-01 on 1-minute slices from 09:15 IST, one
instrument each (SBIN or INFY, cash equity at 1x margin), no charges and no slippage,
with in-process bars; nothing contacts a server. A strategy is a script: the orders it
submits at the close of a minute.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pytest

from openalgo.backtest._calendar_convert import native_attr, to_day
from openalgo.backtest.calendar import default_calendar

DAY = dt.date(2025, 7, 1)
MINUTE_NS = 60 * 1_000_000_000
# 09:15 IST is 03:45 UTC.
OPEN_NS = int(dt.datetime(2025, 7, 1, 3, 45, tzinfo=dt.UTC).timestamp()) * 10**9
MINUTES = 375


def instrument(symbol: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "exchange": "NSE",
        "kind": "EQ",
        "lot_size": 1,
        "tick_size": 0.05,
        "multiplier": 1.0,
        "freeze_qty": None,
        "last_trading_day": None,
    }


def config(capital: float) -> dict[str, Any]:
    """A cost-free, slippage-free 1-minute run on 1-minute bars of DAY."""
    return {
        "first_day": to_day(DAY),
        "last_day": to_day(DAY),
        "base_nanos": MINUTE_NS,
        "strategy_interval": MINUTE_NS,
        "fill": "next_open",
        "quantity_mode": "venue",
        "capital": capital,
        "costs": None,
        "slippage": ("zero", 0.0, 0),
        "risk": {
            "risk_basis": "gross",
            "after_risk_exit": "lock_session",
            "round_levels": True,
            "strict_config": True,
        },
        "deliver_events": False,
    }


def bars(price: Any) -> tuple[np.ndarray, ...]:
    """The 375 bars of DAY, minute `k` priced (open, high, low, close) = `price(k)`."""
    rows = [price(k) for k in range(MINUTES)]
    ts = np.array([OPEN_NS + k * MINUTE_NS for k in range(MINUTES)], dtype=np.int64)
    columns = [np.array([row[i] for row in rows], dtype=np.float64) for i in range(4)]
    return (ts, *columns, np.full(MINUTES, 1_000.0))


def wave(base: float) -> Any:
    """A moving market around `base` in whole rupees: open and close step over 0 to 20."""

    def price(k: int) -> tuple[float, float, float, float]:
        open_, close = base + (k * 37) % 21, base + (k * 53) % 21
        return (open_, max(open_, close) + 1, min(open_, close) - 1, close)

    return price


def flat(price: float) -> Any:
    return lambda _k: (price, price, price, price)


def engine(symbol: str, capital: float, price: Any) -> Any:
    calendar = default_calendar()._cal
    return native_attr("Engine")(calendar, config(capital), [instrument(symbol)], [bars(price)])


def order(side: str, units: float, **fields: Any) -> dict[str, Any]:
    return {
        "instrument": 0,
        "side": side,
        "quantity": {"kind": "units", "value": units},
        "order_type": "MARKET",
        "validity": "DAY",
        "product": "MIS",
        "reduce_only": False,
        "reverse": False,
        **fields,
    }


def exit_all(side: str) -> dict[str, Any]:
    return order(side, 1.0, quantity={"kind": "fraction", "value": 1.0}, reduce_only=True)


def close_of(minute: int) -> int:
    """The close of minute `minute` after 09:15, UTC nanoseconds."""
    return OPEN_NS + (minute + 1) * MINUTE_NS


def decide(target: Any, script: dict[int, list], pause: tuple) -> None:
    """Submit the script's orders of this close (a bar-close pause)."""
    kind, decided_at = pause[0], pause[1]
    if kind == "bar_close":
        for request in script.get(decided_at, []):
            target.submit(request)


def alone(target: Any, script: dict[int, list]) -> tuple[dict, list]:
    """Run `target` alone with its script: (its result tables, its pauses as (kind,
    decision instant))."""
    seen = []
    while True:
        pause = target.advance()
        seen.append(pause[:2])
        decide(target, script, pause)
        if pause[0] == "finished":
            return target.finish(), seen


def in_fund(
    engines: list, scripts: list, account: dict[str, Any]
) -> tuple[list[dict], dict, list[list]]:
    """Run `engines` in a fund with their scripts: (each engine's result tables, the
    account's records, each strategy's pauses as (kind, decision instant))."""
    names = [f"S{index}" for index in range(len(engines))]
    fund = native_attr("Fund")(engines, names, account)
    seen: list[list] = [[] for _ in engines]
    while (step := fund.advance()) is not None:
        index, pause = step
        seen[index].append(pause[:2])
        decide(engines[index], scripts[index], pause)
    return [target.finish() for target in engines], fund.finish(), seen


def shared(capital: float | None = None, limit: float | None = None) -> dict[str, Any]:
    return {"pool": "shared", "account_capital": capital, "account_max_daily_loss": limit}


def same(left: Any, right: Any) -> bool:
    """Equal values, arrays element by element, nested in dicts, lists and tuples."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, list | tuple):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right, strict=True))
    return left == right


SBIN_SCRIPT = {
    close_of(3): [order("BUY", 100.0)],
    close_of(10): [order("BUY", 50.0)],
    close_of(40): [exit_all("SELL")],
    close_of(60): [order("SELL", 80.0)],
    close_of(70): [exit_all("BUY")],
}
INFY_SCRIPT = {close_of(20): [order("BUY", 300.0)], close_of(90): [exit_all("SELL")]}
SCRIPTS = [SBIN_SCRIPT, INFY_SCRIPT]


def two() -> list:
    return [engine("SBIN", 1_000_000.0, wave(100.0)), engine("INFY", 500_000.0, wave(1_500.0))]


def test_each_strategy_in_a_fund_gets_the_tables_and_pauses_it_gets_alone():
    solo = [alone(target, script) for target, script in zip(two(), SCRIPTS, strict=True)]
    outputs, _, seen = in_fund(two(), SCRIPTS, shared())
    for index, (tables, pauses) in enumerate(solo):
        assert same(outputs[index], tables), index
        assert seen[index] == pauses, index
    # The SBIN script's five orders: the buy, the add, the exit, the short, the cover.
    assert len(solo[0][0]["fills"]["side"]) == 5


def test_a_condition_strategy_is_decided_in_rust_as_run_signals_decides_it():
    conditions = engine("SBIN", 1_000_000.0, wave(100.0))
    opens = conditions.strategy_bars(0)["ts_open"]
    spec = {
        "instrument": 0,
        "product": "MIS",
        "bar_opens": opens,
        "entry": np.arange(len(opens)) % 50 == 5,
        "exit": np.arange(len(opens)) % 50 == 25,
        "short": None,
        "cover": None,
        "entry_order": order("BUY", 100.0),
        "short_order": None,
        "pyramiding": 0,
    }
    solo = engine("SBIN", 1_000_000.0, wave(100.0))
    solo_skipped = solo.run_signals(spec)
    solo_tables = solo.finish()
    infy = engine("INFY", 500_000.0, wave(1_500.0))
    fund = native_attr("Fund")([conditions, infy], ["Conditions", "Infy"], shared())
    fund.add_signals(0, spec)
    while (step := fund.advance()) is not None:
        index, pause = step
        assert index == 1, "the condition strategy never pauses in Python"
        decide(infy, INFY_SCRIPT, pause)
    assert same(conditions.finish(), solo_tables)
    assert fund.finish()["skipped"] == [list(solo_skipped), []]
    assert len(solo_tables["trades"]["entry_ts"]) >= 7


def test_the_first_strategy_takes_the_account_and_the_second_is_refused():
    # Two strategies of Rs 1,00,000 on an account of Rs 1,00,000, SBIN flat at 100.00:
    # both buy 600 at the close of minute 0; at the open of minute 1 the first fills
    # (60,000 of margin, the account's free cash 1,00,000 - 2,00,000 + 40,000 + 1,00,000 =
    # 40,000) and the second is refused account_margin.
    buy = {close_of(0): [order("BUY", 600.0)]}
    engines = [engine("SBIN", 100_000.0, flat(100.0)) for _ in range(2)]
    outputs, records, _ = in_fund(engines, [buy, buy], shared(capital=100_000.0))
    assert records["refused"] == {
        "strategy": [1],
        "order_id": [1],
        "slice": [1],
        "slice_ts": [OPEN_NS + MINUTE_NS],
        "step": [2],
        "reason": ["account_margin"],
    }
    account = records["account"]
    row = int(np.flatnonzero(account["ts"] == close_of(1))[0])
    figures = [account[key][row] for key in ("equity_rupees", "free_cash_rupees", "margin_rupees")]
    assert figures == [100_000.0, 40_000.0, 60_000.0]
    assert len(account["ts"]) == MINUTES
    assert outputs[1]["orders"]["reject_code"] == ["account_margin"]
    assert records["account_capital"] == 100_000.0


def test_the_account_limit_exits_every_strategy_and_is_recorded_once():
    # SBIN flat at 100.00 to 09:25, then 95.00; the strategies hold 600 and 400 from 09:16:
    # at the round closing at 09:26 the account's measure is -3,000 - 2,000 = -5,000, at
    # the limit of 5,000. Both exit at the next open at 95.00, reason account_loss_limit.
    falling = lambda k: (100.0,) * 4 if k < 10 else (95.0,) * 4  # noqa: E731
    first, second = {close_of(0): [order("BUY", 600.0)]}, {close_of(0): [order("BUY", 400.0)]}
    engines = [engine("SBIN", 100_000.0, falling) for _ in range(2)]
    outputs, records, _ = in_fund(engines, [first, second], shared(limit=5_000.0))
    at = close_of(10)
    assert records["breaches"] == [(at, to_day(DAY), 5_000.0, -5_000.0, "account_loss")]
    for tables in outputs:
        assert tables["trades"]["exit_reason"] == ["account_loss_limit"]
        assert tables["trades"]["exit_price_rupees"] == [95.0]


def test_a_fund_run_twice_gives_the_same_bytes():
    first = in_fund(two(), SCRIPTS, shared(capital=600_000.0, limit=20_000.0))
    second = in_fund(two(), SCRIPTS, shared(capital=600_000.0, limit=20_000.0))
    assert same(first, second)
    for key, column in first[1]["account"].items():
        assert column.tobytes() == second[1]["account"][key].tobytes(), key


SEPARATE_OF_1 = {**shared(), "pool": "separate", "account_capital": 1.0}


@pytest.mark.parametrize(
    ("names", "account", "message"),
    [
        (["A"], shared(), "2 engines and 1 names"),
        (["A", "B"], {**shared(), "pool": "pooled"}, "pool"),
        (["A", "B"], SEPARATE_OF_1, "sum of the capitals"),
        (["A", "B"], shared(limit=-1.0), "account_max_daily_loss"),
    ],
)
def test_a_fund_refuses_settings_it_cannot_run(names, account, message):
    with pytest.raises(native_attr("NativeEngineInputError"), match=message):
        native_attr("Fund")(two(), names, account)


def test_a_fund_refuses_one_engine_for_two_strategies():
    target = engine("SBIN", 100_000.0, flat(100.0))
    with pytest.raises(native_attr("NativeEngineInputError"), match="an engine is given twice"):
        native_attr("Fund")([target, target], ["A", "B"], shared())


def test_an_engine_advanced_outside_its_fund_stops_the_fund_naming_its_strategy():
    engines = two()
    fund = native_attr("Fund")(engines, ["SbinTrend", "InfyTrend"], shared())
    index, _ = fund.advance()
    engines[index].advance()
    outside = "SbinTrend: its engine was advanced outside the fund"
    with pytest.raises(native_attr("NativeEngineError"), match=outside):
        fund.advance()


def test_an_engine_finished_before_the_fund_ended_stops_the_fund_naming_its_strategy():
    # Both strategies end at the last round; SbinTrend's "finished" pause comes first, and
    # its driver finishes its engine there, as a single run's driver would.
    engines = two()
    fund = native_attr("Fund")(engines, ["SbinTrend", "InfyTrend"], shared())
    while (step := fund.advance())[1][0] != "finished":
        pass
    engines[step[0]].finish()
    early = "SbinTrend: its engine was finished before the portfolio run ended: finish each"
    with pytest.raises(native_attr("NativeEngineError"), match=early):
        fund.advance()


def test_an_engine_the_fund_refuses_is_named_by_its_strategy():
    advanced = engine("INFY", 500_000.0, wave(1_500.0))
    advanced.advance()
    engines = [engine("SBIN", 1_000_000.0, wave(100.0)), advanced]
    refused = "engines: InfyTrend has an engine that has advanced already"
    with pytest.raises(native_attr("NativeEngineInputError"), match=refused):
        native_attr("Fund")(engines, ["SbinTrend", "InfyTrend"], shared())


def test_an_engine_error_names_its_strategy():
    engines = two()
    fund = native_attr("Fund")(engines, ["SbinTrend", "InfyTrend"], shared())
    spec = {"instrument": 5, "product": "MIS", "bar_opens": np.array([], dtype=np.int64)}
    spec |= {"entry": None, "exit": None, "short": None, "cover": None}
    spec |= {"entry_order": None, "short_order": None, "pyramiding": 0}
    with pytest.raises(native_attr("NativeEngineInputError"), match="InfyTrend: conditions"):
        fund.add_signals(1, spec)


def test_a_fund_finishes_once_after_its_end():
    engines = two()
    fund = native_attr("Fund")(engines, ["A", "B"], shared())
    with pytest.raises(native_attr("NativeEngineError"), match="has not finished"):
        fund.finish()
    while fund.advance() is not None:
        pass
    with pytest.raises(native_attr("NativeEngineError"), match="has finished"):
        fund.advance()
    assert fund.finish()["account"]["ts"].dtype == np.int64
    with pytest.raises(native_attr("NativeEngineError"), match="already finished"):
        fund.finish()


def test_advancing_a_finished_fund_after_its_engines_finished_raises_the_funds_finished_error():
    # Each driver finishes its engine once the fund has returned None; a later advance is
    # the fund's finished error, not the error of an engine finished early.
    engines = two()
    fund = native_attr("Fund")(engines, ["SbinTrend", "InfyTrend"], shared())
    while fund.advance() is not None:
        pass
    for target in engines:
        target.finish()
    finished = "the portfolio run has finished; nothing to advance"
    with pytest.raises(native_attr("NativeEngineError"), match=finished):
        fund.advance()


def test_the_accounts_profit_target_exits_every_strategy_and_is_recorded_with_its_rule():
    # SBIN flat at 100.00 to 09:20, then 105.00; both strategies hold 100 from 09:16: at
    # the round closing at 09:21 each is up 500, the account 1,000, its target. Both exit
    # at the next open at 105.00, reason account_profit_target.
    rising = lambda k: (100.0,) * 4 if k < 5 else (105.0,) * 4  # noqa: E731
    buy = {close_of(0): [order("BUY", 100.0)]}
    engines = [engine("SBIN", 100_000.0, rising) for _ in range(2)]
    account = shared() | {"account_daily_profit_target": 1_000.0}
    outputs, records, _ = in_fund(engines, [buy, buy], account)
    assert records["breaches"] == [(close_of(5), to_day(DAY), 1_000.0, 1_000.0, "account_profit")]
    for tables in outputs:
        assert tables["trades"]["exit_reason"] == ["account_profit_target"]
        assert tables["trades"]["exit_price_rupees"] == [105.0]


def test_the_accounts_trade_cap_refuses_the_entry_past_it_with_its_code():
    # A cap of 2 entries: each strategy enters at 09:16 (one entry each), the first exits
    # at 09:18 and enters again at 09:20, which the cap refuses.
    first = {
        close_of(0): [order("BUY", 100.0)],
        close_of(2): [exit_all("SELL")],
        close_of(4): [order("BUY", 100.0)],
    }
    second = {close_of(0): [order("BUY", 100.0)]}
    engines = [engine("SBIN", 100_000.0, flat(100.0)) for _ in range(2)]
    account = shared() | {"account_max_trades_per_day": 2}
    outputs, records, _ = in_fund(engines, [first, second], account)
    assert records["refused"]["reason"] == ["account_trade_limit"]
    assert records["refused"]["strategy"] == [0]
    assert outputs[0]["orders"]["reject_code"] == [None, None, "account_trade_limit"]
