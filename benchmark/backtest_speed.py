"""Speed of the backtest engine on synthetic, seeded data (no server, no files).

    python benchmark/backtest_speed.py              # print the table
    python benchmark/backtest_speed.py --check      # also exit 1 when a target is missed
    python benchmark/backtest_speed.py --check --slack 2   # a slower machine (CI)

Rows:

- callbacks: an event strategy whose `on_bar` does nothing, on 1-minute bars of one
  instrument: strategy-bar callbacks per second of the whole `bt.backtest` call
  (NFR-PERF-04).
- no-callbacks: the same bars run with no Python call per bar (an array strategy that
  never trades, no metrics). The per-callback overhead printed under the table is the
  callbacks run less this one, per bar: the cost of calling Python once per bar.
- conditions-1M: an array strategy on one instrument over 1M 1-minute bars with the
  default charges and slippage and about 10k round trips, then every metric
  (`result.performance`), on one core (NFR-PERF-01).
- trades-N: that run on 375k bars with more and more round trips. Their cost must stay
  linear: the time each round trip adds is checked against a target. (Before the order
  book kept its working orders apart, every slice walked every order of the run, so a
  round trip cost 0.4 ms at this size, growing with the run's length.)
- bars-N: that run at a fixed number of round trips with more and more bars; the time
  per bar of the largest run may be at most 1.5 times that of the smallest.
- portfolio-3x1M: three array strategies, each the conditions-1M strategy on its own 1M
  bars, in a shared pool (18-multi-strategy 2.7); then fund round: the fund's own cost
  per round of the three engines (the phase calls and the ledger), the run less the
  three engines run alone, per round.
- portfolio-callbacks: three event strategies whose `on_bar` does nothing, each on the
  callbacks row's bars, in a shared pool: strategy-bar callbacks per second in total.

Until `bt.backtest` takes a list of strategies, the portfolio rows build each strategy's
native engine as `bt.backtest` builds it and run the engines and the fund through the
binding: the kernel, the strategies' drivers and the fund, without the result tables and
the metrics of a `bt.backtest` call (and, in portfolio-callbacks, with the pause loop
handing each strategy's bars to an `on_bar` that does nothing).

`--check` fails when a target is missed. `--slack` divides the rate target and
multiplies the time targets by its value, for a machine slower than the one the targets
were set on (a shared CI runner); the linearity ratio is a ratio and is not relaxed.

The bars are a random walk on the 0.05-rupee tick grid from a fixed seed, one bar a
minute from 09:15 to 15:29 IST on every weekday from 2014, on a calendar of regular NSE
hours with no holiday list (1M bars need more sessions than the built-in holiday lists
cover). The charges are the default broker's at the rates of each trade date, and at
the first schedule version's rates before that version starts. Each row is the best of
`--repeat` runs, so one disturbed run does not fail the check.
"""

from __future__ import annotations

import argparse
import datetime as dt
import inspect
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Run against this checkout when it holds a built extension (a local build copied next
# to the package); otherwise against the installed wheel.
ROOT = Path(__file__).resolve().parent.parent
if list((ROOT / "openalgo").glob("_oaindicators*")):
    sys.path.insert(0, str(ROOT))

import openalgo.backtest as bt  # noqa: E402
from openalgo.backtest._calendar_convert import native_attr  # noqa: E402
from openalgo.backtest._engine import build_engine  # noqa: E402
from openalgo.backtest._pipeline import prepare  # noqa: E402

# ----- targets ---------------------------------------------------------------------
# NFR-PERF-04 (02-prd section 8): an event strategy with a trivial on_bar makes at least
# 300k strategy-bar callbacks per second per process.
CALLBACKS_PER_SECOND_TARGET = 300_000
# NFR-PERF-01 (02-prd section 8): an array or condition strategy on one instrument over
# 1M bars with full metrics, one core. The PRD states it relative to another engine; the
# repository keeps an absolute target instead: 2 seconds for the whole call (data
# checks, the kernel with charges and slippage, the result tables, every metric), 2
# microseconds per bar, on a current desktop core (confirmed by the maintainer on
# 2026-11-12). Measured there: 1.1 to 1.5 seconds, depending on the clock speed the core
# holds under sustained load.
CONDITIONS_1M_SECONDS_TARGET = 2.0
# Linearity in trades: the run time one round trip adds (two orders, two fills, their
# charges, events and result rows) on the 375k-bar run, at most this many microseconds.
ROUND_TRIP_MICROSECONDS_TARGET = 50.0
# Linearity in bars: the time per bar of the largest bars-N run over that of the
# smallest, at most this.
BAR_GROWTH_LIMIT = 1.5
# 18-multi-strategy 2.7: three condition strategies over 1M bars each in a shared pool
# take at most three times the conditions-1M target plus 10 percent (6.6 seconds), and
# three empty event strategies make at least the callbacks target in total.
PORTFOLIO_STRATEGIES = 3
PORTFOLIO_3X1M_SECONDS_TARGET = PORTFOLIO_STRATEGIES * CONDITIONS_1M_SECONDS_TARGET * 1.1
# 18-multi-strategy 2.7: the fund's own cost per round of three engines, at most this
# many microseconds.
FUND_ROUND_MICROSECONDS_TARGET = 1.0

# ----- the synthetic market --------------------------------------------------------
SEED = 20_261_112
SYMBOL, EXCHANGE = "SYN", "NSE"
TICK_RUPEES = 0.05
START_PRICE_RUPEES = 1_000.0
# Whole ticks a bar moves at most, and a high or low beyond its open and close.
MAX_STEP_TICKS = 6
MAX_WICK_TICKS = 4
# A price floor in ticks (10 rupees), so the walk never reaches zero.
FLOOR_TICKS = 200
VOLUME_RANGE_UNITS = (1_000, 50_000)
# A 1-minute session from 09:15 to 15:29 IST has 375 bars.
BARS_PER_SESSION = 375
SESSION_OPEN = (9, 15)
FIRST_SESSION = dt.date(2014, 1, 1)
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
NANOS_PER_MINUTE = 60_000_000_000
CAPITAL_RUPEES = 1_000_000
QUANTITY_UNITS = 10

# ----- sizes ----------------------------------------------------------------------
# The callbacks row and the trades-N rows run the same bars, so their difference is the
# cost of calling Python per bar.
RUN_BARS = 375_000
MILLION = 1_000_000
CONDITIONS_1M_TRADES = 10_000
TRADE_COUNTS = (0, 1_000, 4_000, 16_000)
BAR_SCALING_TRADES = 1_000
BAR_COUNTS = (125_000, 250_000, 500_000, 1_000_000)


def weekdays(count: int) -> list[dt.date]:
    """The first `count` weekdays from FIRST_SESSION."""
    days, day = [], FIRST_SESSION
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += dt.timedelta(days=1)
    return days


def synthetic_bars(count: int, seed: int = SEED) -> pd.DataFrame:
    """`count` 1-minute bars of whole sessions (the last one cut): a random walk of whole
    ticks, each bar opening at the previous close."""
    rng = np.random.default_rng(seed)
    sessions = weekdays(-(-count // BARS_PER_SESSION))
    first_opens = [
        pd.Timestamp(dt.datetime(day.year, day.month, day.day, *SESSION_OPEN, tzinfo=IST)).value
        for day in sessions
    ]
    minutes = np.arange(BARS_PER_SESSION, dtype=np.int64) * NANOS_PER_MINUTE
    stamps = (np.asarray(first_opens, dtype=np.int64)[:, None] + minutes).ravel()[:count]
    steps = rng.integers(-MAX_STEP_TICKS, MAX_STEP_TICKS + 1, size=count)
    ticks = np.maximum(round(START_PRICE_RUPEES / TICK_RUPEES) + np.cumsum(steps), FLOOR_TICKS)
    close = ticks * TICK_RUPEES
    open_ = np.concatenate(([START_PRICE_RUPEES], close[:-1]))
    wicks = rng.integers(1, MAX_WICK_TICKS + 1, size=(2, count)) * TICK_RUPEES
    columns = {
        "open": open_,
        "high": np.maximum(open_, close) + wicks[0],
        "low": np.minimum(open_, close) - wicks[1],
        "close": close,
    }
    frame = pd.DataFrame({name: np.round(values, 2) for name, values in columns.items()})
    frame["volume"] = rng.integers(*VOLUME_RANGE_UNITS, size=count).astype(float)
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(stamps, unit="ns", utc=True).tz_convert(IST), name="timestamp"
    )
    return frame


def run_arguments(bars: pd.DataFrame) -> dict:
    """The `bt.backtest` arguments of every row: own data, the default slippage, the
    default broker's charges and the calendar of the module docstring."""
    return {
        "data": bars,
        "capital": CAPITAL_RUPEES,
        "calendar": bt.calendar.Calendar(builtin=False),
        "costs": bt.costs.CostModel(backfill=True),
        "start": bars.index[0].date(),
        "end": bars.index[-1].date(),
    }


class Trivial(bt.Strategy):
    """An event strategy whose on_bar does nothing."""

    symbol, exchange, interval = SYMBOL, EXCHANGE, "1m"

    def on_bar(self, bar):
        pass


def signal_arrays(bars: int, trades: int) -> tuple[np.ndarray, np.ndarray]:
    """Entry and exit arrays for about `trades` round trips spread evenly over `bars`
    1-minute bars: an entry every `period` bars and its exit half a period later. The
    kernel refuses entries outside the MIS entry window, so the count is approximate;
    the table shows the trades made."""
    entries = np.zeros(bars, dtype=bool)
    exits = np.zeros(bars, dtype=bool)
    if trades:
        period = max(bars // trades, 4)
        entries[::period] = True
        exits[period // 2 :: period] = True
    return entries, exits


def condition_strategy(entries: np.ndarray, exits: np.ndarray) -> type[bt.Strategy]:
    """An array strategy buying QUANTITY_UNITS on `entries` and exiting on `exits`."""
    return bt.from_arrays(
        entries, exits, symbol=SYMBOL, exchange=EXCHANGE, interval="1m", qty=QUANTITY_UNITS
    )


@dataclass(frozen=True)
class Row:
    """One line of the table: what ran (bars, trades made, best wall seconds) and, for
    a checked row, the value checked, its unit and target, and whether `--slack`
    relaxes the target (a time or a rate does; a ratio of two runs does not)."""

    name: str
    bars: int
    trades: int
    seconds: float
    value: float | None = None
    unit: str = ""
    target: float | None = None
    higher_is_better: bool = False
    relaxed: bool = True

    def meets(self, slack: float) -> bool:
        """True when the row has no target or meets it (relaxed by `slack`)."""
        if self.target is None or self.value is None:
            return True
        slack = slack if self.relaxed else 1.0
        if self.higher_is_better:
            return self.value >= self.target / slack
        return self.value <= self.target * slack


def best_time(repeat: int, run: Callable[[], object]) -> tuple[float, object]:
    """The shortest wall time of `repeat` calls of `run`, and the last call's value."""
    best, value = float("inf"), None
    for _ in range(repeat):
        start = time.perf_counter()
        value = run()
        best = min(best, time.perf_counter() - start)
    return best, value


def condition_run(bars: pd.DataFrame, trades: int) -> int:
    """The array strategy on `bars` with about `trades` round trips, then every metric;
    returns the trades made."""
    entries, exits = signal_arrays(len(bars), trades)
    result = bt.backtest(condition_strategy(entries, exits), **run_arguments(bars))
    result.performance  # noqa: B018 (computing every metric is part of the measured run)
    return len(result.trades)


def callbacks_row(bars: pd.DataFrame, repeat: int) -> Row:
    seconds, _ = best_time(repeat, lambda: bt.backtest(Trivial, **run_arguments(bars)))
    rate = len(bars) / seconds
    return Row(
        "callbacks", len(bars), 0, seconds, rate, "callbacks/s",
        CALLBACKS_PER_SECOND_TARGET, higher_is_better=True,
    )  # fmt: skip


def no_callbacks_row(bars: pd.DataFrame, repeat: int) -> Row:
    entries, exits = signal_arrays(len(bars), 0)
    strategy = condition_strategy(entries, exits)
    seconds, _ = best_time(repeat, lambda: bt.backtest(strategy, **run_arguments(bars)))
    return Row("no-callbacks", len(bars), 0, seconds)


def conditions_1m_row(repeat: int) -> Row:
    bars = synthetic_bars(MILLION)
    seconds, trades = best_time(repeat, lambda: condition_run(bars, CONDITIONS_1M_TRADES))
    return Row(
        "conditions-1M", len(bars), trades, seconds, seconds, "s", CONDITIONS_1M_SECONDS_TARGET
    )


def trade_rows(bars: pd.DataFrame, repeat: int) -> list[Row]:
    """The trades-N rows, then the round-trip cost of the largest above the first."""
    rows = []
    for count in TRADE_COUNTS:
        seconds, trades = best_time(repeat, lambda c=count: condition_run(bars, c))
        rows.append(Row(f"trades-{count}", len(bars), trades, seconds))
    first, last = rows[0], rows[-1]
    cost = (last.seconds - first.seconds) / last.trades * 1e6
    check = Row(
        "round trip", last.bars, last.trades, last.seconds, cost, "us",
        ROUND_TRIP_MICROSECONDS_TARGET,
    )  # fmt: skip
    return [*rows, check]


def bar_rows(repeat: int) -> list[Row]:
    """The bars-N rows, then the growth of the time per bar from the first to the last."""
    rows = []
    for count in BAR_COUNTS:
        bars = synthetic_bars(count)
        seconds, trades = best_time(repeat, lambda b=bars: condition_run(b, BAR_SCALING_TRADES))
        rows.append(Row(f"bars-{count}", len(bars), trades, seconds))
    first, last = rows[0], rows[-1]
    growth = (last.seconds / last.bars) / (first.seconds / first.bars)
    check = Row(
        "bar growth", last.bars, last.trades, last.seconds, growth, "x", BAR_GROWTH_LIMIT,
        relaxed=False,
    )  # fmt: skip
    return [*rows, check]


def native_engine(strategy: type[bt.Strategy], bars: pd.DataFrame) -> Any:
    """The native engine `bt.backtest` builds for `strategy` on `bars`, before it runs."""
    call = inspect.signature(bt.backtest).bind(strategy, None, **run_arguments(bars))
    call.apply_defaults()
    prepared = prepare(strategy, None, dict(call.arguments))
    market = prepared.market_settings()
    config = prepared.plan.native_config(deliver_events=False) | market.config
    calendar = prepared.session_calendar()._cal
    return build_engine(calendar, config, prepared.instruments, prepared.frames)


def condition_spec(engine: Any, entries: np.ndarray, exits: np.ndarray) -> dict:
    """The Rust driver's spec of the array strategy of `condition_strategy` on `engine`,
    whose strategy bars are its 1-minute base bars."""
    order = {
        "instrument": 0,
        "side": "BUY",
        "quantity": {"kind": "units", "value": float(QUANTITY_UNITS)},
        "order_type": "MARKET",
        "validity": "DAY",
        "product": "MIS",
        "reduce_only": False,
        "reverse": False,
    }
    opens = engine.strategy_bars(0)["ts_open"]
    return {"instrument": 0, "product": "MIS", "bar_opens": opens, "entry": entries} | {
        "exit": exits, "short": None, "cover": None,
        "entry_order": order, "short_order": None, "pyramiding": 0,
    }  # fmt: skip


def in_fund(engines: list, specs: list[dict | None]) -> tuple[int, int]:
    """Run `engines` in a shared fund to the end, the specs' strategies in Rust and the
    others through a pause loop whose `on_bar` does nothing; (callbacks, rounds)."""
    names = [f"S{index}" for index in range(len(engines))]
    account = {"pool": "shared", "account_capital": None, "account_max_daily_loss": None}
    fund = native_attr("Fund")(engines, names, account)
    for index, spec in enumerate(specs):
        if spec is not None:
            fund.add_signals(index, spec)
    callbacks = 0
    while (step := fund.advance()) is not None:
        kind, _, rows, _, _ = step[1]
        callbacks += kind == "bar_close" and bool(rows)
    for engine in engines:
        engine.finish()
    return callbacks, len(fund.finish()["account"]["ts"])


def portfolio_3x1m_rows(repeat: int) -> list[Row]:
    """portfolio-3x1M, then the fund's own cost per round (module docstring)."""
    entries, exits = signal_arrays(MILLION, CONDITIONS_1M_TRADES)
    strategy = condition_strategy(entries, exits)
    markets = [synthetic_bars(MILLION, SEED + index) for index in range(PORTFOLIO_STRATEGIES)]

    def alone() -> float:
        engines = [native_engine(strategy, bars) for bars in markets]
        start = time.perf_counter()
        for engine in engines:
            engine.run_signals(condition_spec(engine, entries, exits))
            engine.finish()
        return time.perf_counter() - start

    def together() -> tuple[float, int]:
        engines = [native_engine(strategy, bars) for bars in markets]
        specs = [condition_spec(engine, entries, exits) for engine in engines]
        start = time.perf_counter()
        _, rounds = in_fund(engines, specs)
        return time.perf_counter() - start, rounds

    solo = min(alone() for _ in range(repeat))
    runs = [together() for _ in range(repeat)]
    seconds, rounds = min(runs)
    bars = PORTFOLIO_STRATEGIES * MILLION
    cost = (seconds - solo) / rounds * 1e6
    return [
        Row("portfolio-3x1M", bars, 0, seconds, seconds, "s", PORTFOLIO_3X1M_SECONDS_TARGET),
        Row("fund round", bars, 0, seconds, cost, "us", FUND_ROUND_MICROSECONDS_TARGET),
    ]


def portfolio_callbacks_row(bars: pd.DataFrame, repeat: int) -> Row:
    """portfolio-callbacks (module docstring)."""

    def run() -> tuple[float, int]:
        engines = [native_engine(Trivial, bars) for _ in range(PORTFOLIO_STRATEGIES)]
        start = time.perf_counter()
        callbacks, _ = in_fund(engines, [None] * PORTFOLIO_STRATEGIES)
        return time.perf_counter() - start, callbacks

    seconds, callbacks = min(run() for _ in range(repeat))
    return Row(
        "portfolio-callbacks", PORTFOLIO_STRATEGIES * len(bars), 0, seconds, callbacks / seconds,
        "callbacks/s", CALLBACKS_PER_SECOND_TARGET, higher_is_better=True,
    )  # fmt: skip


def number(value: float | None, unit: str) -> str:
    """A value and its unit for the table: thousands separated, else 3 significant digits."""
    if value is None:
        return ""
    return f"{value:,.0f} {unit}" if abs(value) >= 1_000 else f"{value:.3g} {unit}"


def print_table(rows: list[Row], slack: float) -> None:
    print(f"{'row':<20}{'bars':>10}{'trades':>8}{'seconds':>9}  {'value':>20}  target")
    for row in rows:
        value = number(row.value, row.unit)
        target = number(row.target, row.unit)
        verdict = "" if row.target is None else ("met" if row.meets(slack) else "MISSED")
        print(
            f"{row.name:<20}{row.bars:>10,}{row.trades:>8,}{row.seconds:>9.3f}  "
            f"{value:>20}  {target} {verdict}".rstrip()
        )


def print_overhead(rows: list[Row]) -> None:
    """The per-callback figures of the documentation, when both rows ran."""
    named = {row.name: row for row in rows}
    callbacks, plain = named.get("callbacks"), named.get("no-callbacks")
    if callbacks is None:
        return
    print(f"per callback, whole run: {callbacks.seconds / callbacks.bars * 1e6:.2f} us")
    if plain is not None:
        overhead = (callbacks.seconds - plain.seconds) / callbacks.bars * 1e6
        print(f"per callback, the Python call (callbacks less no-callbacks): {overhead:.2f} us")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 when a target is missed")
    parser.add_argument("--slack", type=float, default=1.0, help="relax the targets by this")
    parser.add_argument("--repeat", type=int, default=3, help="runs per row; the best counts")
    groups = ("callbacks", "conditions", "trades", "bars", "portfolio")
    parser.add_argument("--rows", default=",".join(groups), help="row groups to run")
    args = parser.parse_args(argv)
    chosen = set(args.rows.split(","))
    # A misspelt group would run no row, and --check would then pass on nothing.
    unknown = sorted(chosen - set(groups))
    if unknown:
        parser.error(
            f"unknown row group {', '.join(map(repr, unknown))}: choose from {', '.join(groups)}"
        )
    # The runs start before the first cost schedule (backfilled) on a calendar with no
    # holiday list; the warnings saying so are expected here and would flood the table.
    warnings.simplefilter("ignore", UserWarning)
    run_bars = synthetic_bars(RUN_BARS)
    rows: list[Row] = []
    if "callbacks" in chosen:
        rows += [callbacks_row(run_bars, args.repeat), no_callbacks_row(run_bars, args.repeat)]
    if "conditions" in chosen:
        rows.append(conditions_1m_row(args.repeat))
    if "trades" in chosen:
        rows += trade_rows(run_bars, args.repeat)
    if "bars" in chosen:
        rows += bar_rows(args.repeat)
    if "portfolio" in chosen:
        rows += portfolio_3x1m_rows(args.repeat)
        rows.append(portfolio_callbacks_row(run_bars, args.repeat))
    print_table(rows, args.slack)
    print_overhead(rows)
    missed = [row.name for row in rows if not row.meets(args.slack)]
    if args.check and missed:
        print(f"targets missed (slack {args.slack}): {', '.join(missed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
