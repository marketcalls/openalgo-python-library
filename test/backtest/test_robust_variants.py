"""The checks that rerun the strategy on its own bars with one setting changed (design 17,
sections 2.7, 2.12, 2.14 and 2.15): shifted bars, randomized exits, delayed orders and
liquidity, on own bars with no server.

Every planted run below makes its money in a way worked out by hand, so each check's
verdict follows from the bars:

- `Every4`, a daily strategy that buys 10 shares at the close of bar 4j and exits at the
  close of bar 4j + 2, on daily bars whose opens repeat a block of four: the trade buys
  at the open of bar 4j + 1 and sells at the open of bar 4j + 3.
- `Flip5`, a 5-minute delivery (CNC) strategy that buys when flat (with no working order)
  and exits when long, on one-minute bars: its edge lives in where its 5-minute bars
  start, or not.
- `DipFlip`, a 5-minute MIS strategy that buys at a session's even bars with a stop
  below a dip planted in the bar after: the run never meets its stop, a tighter one does.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from robust_runs import FREE, nse_sessions, opens_bars

import openalgo.backtest as bt
from openalgo.backtest._robust._check_liquidity import _depths
from openalgo.backtest._robust._verdict import FAILED, NOT_JUDGED, PASSED

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
BLOCKS = 31


def quiet(line: str) -> None:
    """A progress function that drops the plan."""


def checked(result, name, **settings):
    """Check `name` alone on `result`, in this process, measured by net profit."""
    report = result.robustness(
        checks=[name], workers=1, runs=20, objective="net_profit", progress=quiet, **settings
    )
    return report.check(name)


class Every4(bt.Strategy):
    """Buys 10 shares at the close of bar 4j and exits at the close of bar 4j + 2."""

    symbol, exchange, interval = "SBIN", "NSE", "D"

    def init(self):
        self.bar_number = -1

    def on_bar(self, bar):
        self.bar_number += 1
        if self.bar_number % 4 == 0 and self.position.is_flat:
            self.buy(qty=10)
        elif self.bar_number % 4 == 2:
            self.exit()


class LimitEvery4(Every4):
    """`Every4` with resting orders: a LIMIT buy at bar 4j's close and a LIMIT sell at bar
    4j + 2's close, each at that close."""

    def on_bar(self, bar):
        self.bar_number += 1
        if self.bar_number % 4 == 0 and self.position.is_flat:
            self.buy(qty=10, price=bar.close)
        elif self.bar_number % 4 == 2 and self.position.is_long:
            self.sell(qty=10, price=bar.close)


class LimitInMarketOut(Every4):
    """`Every4` with a resting LIMIT buy at bar 4j's close and a MARKET exit."""

    def on_bar(self, bar):
        self.bar_number += 1
        if self.bar_number % 4 == 0 and self.position.is_flat:
            self.buy(qty=10, price=bar.close)
        elif self.bar_number % 4 == 2:
            self.exit()


def every4_bars(block, closes=None):
    """`BLOCKS` blocks of the daily opens `block` and one more bar; each bar's high and low
    1 rupee either side of its open, its close at its open unless `closes` (a block of
    closes) says otherwise."""
    data = opens_bars(block * BLOCKS + block[:1])
    if closes is not None:
        close = np.array(closes * BLOCKS + closes[:1], dtype=float)
        data["close"] = close
        data["high"] = np.maximum(data["high"], close + 0.5)
        data["low"] = np.minimum(data["low"], close - 0.5)
    return data


def every4_run(block, closes=None, strategy=Every4, **settings):
    """`strategy` (`Every4`) on `every4_bars(block, closes)`, with `settings` as run
    arguments."""
    return bt.backtest(strategy, data=every4_bars(block, closes), **FREE, **settings)


# ----- delayed orders -------------------------------------------------------------------


def test_an_edge_that_holds_an_entry_a_bar_late_passes():
    # Opens 500, 500, 500, 503: on time the trade buys at 500 (bar 1) and sells at 503
    # (bar 3), +30; a bar late it buys at bar 2's open, still 500: +30, all of it kept.
    result = every4_run([500.0, 500.0, 500.0, 503.0])
    assert result.metrics["net_profit"] == 30.0 * BLOCKS
    check = checked(result, "delayed")
    assert check.verdict == PASSED
    late = check.table.set_index(["delay_entries", "delay_exits"]).loc[(1, 0)]
    assert late["net_profit"] == 30.0 * BLOCKS and late["ratio"] == 1.0


def test_an_edge_that_a_bar_late_entry_misses_fails():
    # Opens 500, 500, 503, 503: the rise is between bars 1 and 2, so a bar late the trade
    # buys at 503 and sells at 503: nothing.
    result = every4_run([500.0, 500.0, 503.0, 503.0])
    assert result.metrics["net_profit"] == 30.0 * BLOCKS
    check = checked(result, "delayed")
    assert check.verdict == FAILED
    late = check.table.set_index(["delay_entries", "delay_exits"]).loc[(1, 0)]
    assert late["net_profit"] == 0.0
    assert check.figure.startswith("with entries one bar late the run made Rs 0 and keeps 0.0%")


def test_every_pair_of_delays_but_none_is_rerun():
    check = checked(every4_run([500.0, 500.0, 500.0, 503.0]), "delayed")
    pairs = list(zip(check.table["delay_entries"], check.table["delay_exits"], strict=True))
    assert pairs == [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]


def test_a_delay_of_one_bar_at_most_reruns_three_pairs():
    check = checked(every4_run([500.0, 500.0, 500.0, 503.0]), "delayed", delayed={"most": 1})
    assert len(check.table) == 3


def test_the_checks_delays_add_to_the_runs_own():
    # A run whose entries are held a bar already buys at bar 4j + 2's open (500) and sells
    # at bar 4j + 3's (503): +30 a trade. Entries one bar later than that run are entries
    # held two bars, which buy at bar 4j + 3's open (503) and make nothing on the move.
    block = [500.0, 500.0, 500.0, 503.0]
    result = every4_run(block, delay_entries=1)
    assert result.metrics["net_profit"] == 30.0 * BLOCKS
    check = checked(result, "delayed")
    late = check.table.set_index(["delay_entries", "delay_exits"]).loc[(1, 0)]
    held_two = every4_run(block, delay_entries=2).metrics["net_profit"]
    assert late["net_profit"] == held_two < 0
    assert check.verdict == FAILED
    assert "added to the run's own (entries 1, exits 0)" in check.note


# ----- liquidity ------------------------------------------------------------------------


def test_an_edge_that_lives_in_the_opens_fails_at_the_next_close():
    # Opens 500, 500, 500, 503 but every close 500: at the next bar's close the trade
    # buys at 500 and sells at 500, so the models on closes make nothing.
    result = every4_run([500.0, 500.0, 500.0, 503.0], closes=[500.0, 500.0, 500.0, 500.0])
    assert result.metrics["net_profit"] == 30.0 * BLOCKS
    check = checked(result, "liquidity")
    assert check.verdict == FAILED
    by_model = check.table.set_index("model")["net_profit"]
    assert by_model["next_close"] == 0.0


def test_an_edge_every_fill_price_keeps_passes():
    # Opens 500, 500, 499, 503 with each close at its open: the trade dips a rupee before
    # its exit under every model, and each model buys at or below 500 and sells at or
    # above 501.5.
    result = every4_run([500.0, 500.0, 499.0, 503.0])
    check = checked(result, "liquidity")
    assert check.verdict == PASSED
    assert (check.table["net_profit"] > 0).all()
    assert (check.table["drawdown_times"] <= 1.5).all()
    assert list(check.table["model"]) == [
        "next_close",
        "next_mid",
        "next_ohlc4",
        "next_close_2",
        "next_mid_2",
        "next_ohlc4_2",
    ]


def test_a_run_whose_fills_all_keep_their_prices_is_not_judged():
    # Every fill is a resting LIMIT order's, which keeps its price under every model: the
    # six reruns would each be the run itself.
    result = every4_run([500.0, 500.0, 500.0, 503.0], strategy=LimitEvery4)
    assert result.metrics["net_profit"] == 30.0 * BLOCKS
    check = checked(result, "liquidity")
    assert check.verdict == NOT_JUDGED
    assert check.note.startswith("Nothing to judge: no fill of the run was a market order")


def test_the_note_says_how_many_fills_the_models_priced():
    # The LIMIT buys keep their prices and the 31 MARKET exits move: 31 of 62 fills.
    result = every4_run([500.0, 500.0, 500.0, 503.0], strategy=LimitInMarketOut)
    check = checked(result, "liquidity")
    assert check.verdict != NOT_JUDGED
    assert "The fill models priced 31 of the run's 62 fills" in check.note


def test_a_run_on_a_fill_model_is_checked_against_the_others_and_the_next_open():
    result = every4_run([500.0, 500.0, 499.0, 503.0], fill="next_close")
    check = checked(result, "liquidity")
    assert list(check.table["model"]) == [
        "next_open",
        "next_mid",
        "next_ohlc4",
        "next_close_2",
        "next_mid_2",
        "next_ohlc4_2",
    ]


@pytest.mark.parametrize(
    ("drawdowns", "real", "times"),
    [
        ([-0.03, -0.01, 0.0], -0.02, [1.5, 0.5, 0.0]),
        ([-0.01, 0.0], 0.0, [np.inf, 0.0]),  # a fall where the run had none
    ],
)
def test_a_models_drawdown_is_measured_in_times_the_runs(drawdowns, real, times):
    assert list(_depths(np.array(drawdowns), real)) == times


# ----- shifted bars ---------------------------------------------------------------------


class Flip5(bt.Strategy):
    """Buys 10 shares when flat after a green bar and exits when long, on 5-minute bars,
    as delivery. It buys only with no working order, which on time it never has at a
    decision, so that with its entries held it enters late rather than twice."""

    symbol, exchange, interval = "SBIN", "NSE", "5m"
    product = "CNC"

    def on_bar(self, bar):
        if self.position.is_flat:
            if bar.close > bar.open and not self.orders:
                self.buy(qty=10)
        else:
            self.exit()


def minute_bars(prices_of, sessions: int = 2, minutes: int = 1) -> pd.DataFrame:
    """Bars of `minutes` minutes of `sessions` NSE sessions from 2025-03-03: bar k of the
    run opens and closes at `prices_of(k)`, a pair, its high and low around them."""
    stamps, opens, closes = [], [], []
    per_session = 375 // minutes
    for n, day in enumerate(nse_sessions("2025-03-03", sessions)):
        start = pd.Timestamp(dt.datetime.combine(day, dt.time(9, 15), tzinfo=IST))
        for k in range(per_session):
            stamps.append(start + pd.Timedelta(minutes=minutes * k))
            open_, close = prices_of(per_session * n + k)
            opens.append(open_)
            closes.append(close)
    opens, closes = np.array(opens), np.array(closes)
    frame = {
        "open": opens,
        "high": np.maximum(opens, closes),
        "low": np.minimum(opens, closes),
        "close": closes,
        "volume": 1_000.0,
    }
    return pd.DataFrame(frame, index=pd.DatetimeIndex(stamps, name="timestamp"))


def cycle(k: int) -> tuple[float, float]:
    """A 15-minute cycle: minutes 0 to 3 at 100, minute 4 closes 100.05 (its 5-minute bar
    is green), minutes 5 to 9 at 100.05, minute 10 opens 101 and closes 100.05, minute
    14 closes 100 (its bar is red)."""
    m = k % 15
    if m < 4:
        return 100.0, 100.0
    if m == 4:
        return 100.0, 100.05
    if m == 10:
        return 101.0, 100.05
    if m == 14:
        return 100.05, 100.0
    return 100.05, 100.05


def test_an_edge_planted_on_the_bars_boundaries_fails_when_they_move():
    # On time: the green bar of minutes 0 to 4 buys at minute 5's open (100.05) and the
    # exit at the close of minutes 5 to 9 sells at minute 10's open (101): +9.50 a trade,
    # 25 cycles a session. Shifted, the exit falls after minute 10 and sells at 100.05:
    # nothing.
    result = bt.backtest(Flip5, data=minute_bars(cycle), base="1m", **FREE)
    assert result.metrics["net_profit"] == pytest.approx(9.5 * 50)
    check = checked(result, "shifted")
    assert check.verdict == FAILED
    assert list(check.table["shift"]) == [1, 2, 3, 4]
    assert (check.table["net_profit"] == 0.0).all()
    assert check.summary["profitable_share"] == 0.0


def rising(k: int) -> tuple[float, float]:
    """Every minute opens 0.05 above the one before and closes 0.05 above its open."""
    return 500.0 + 0.05 * k, 500.05 + 0.05 * k


def test_an_edge_any_boundaries_keep_passes():
    # Every bar is green and every trade sells above its buy, wherever the bars start.
    result = bt.backtest(Flip5, data=minute_bars(rising), base="1m", **FREE)
    check = checked(result, "shifted")
    assert check.verdict == PASSED
    assert check.summary["profitable_share"] == 1.0
    assert list(check.table["label"])[0] == "1 base bar earlier (1 min)"


def test_strategy_bars_that_are_the_base_bars_have_nothing_to_shift():
    check = checked(every4_run([500.0, 500.0, 500.0, 503.0]), "shifted")
    assert check.verdict == NOT_JUDGED
    assert check.note.startswith("The strategy bars are the base bars (D on D)")


class DailyFlip(Flip5):
    interval = "D"


def test_a_daily_strategy_gets_days_ending_earlier_in_whole_base_bars():
    # On 5-minute base bars a day can end 15 (3 bars) or 30 (6 bars) minutes earlier, not
    # 7; on rising bars every trade of every shift makes money.
    data = minute_bars(rising, sessions=12, minutes=5)
    result = bt.backtest(DailyFlip, data=data, base="5m", **FREE)
    check = checked(result, "shifted", shifted={"minutes": [7, 15, 30], "min_trades": 2})
    assert list(check.table["shift"]) == [3, 6]
    assert list(check.table["label"]) == [
        "3 base bars earlier (15 min)",
        "6 base bars earlier (30 min)",
    ]
    assert "Days ending 7 minutes earlier are left out" in check.note
    assert check.verdict == PASSED


def test_a_daily_strategy_is_told_its_shifted_days_open_the_day_before():
    data = minute_bars(rising, sessions=12, minutes=5)
    result = bt.backtest(DailyFlip, data=data, base="5m", **FREE)
    check = checked(result, "shifted", shifted={"min_trades": 2})
    assert "bar.time is on the date of the session before" in check.note
    assert "the run's first day starts at its session's open" in check.note


def test_an_intraday_strategy_is_not_told_of_days():
    result = bt.backtest(Flip5, data=minute_bars(rising), base="1m", **FREE)
    assert "bar.time" not in checked(result, "shifted").note


def test_decisions_fixed_in_advance_cannot_be_shifted():
    data = minute_bars(rising)
    arrays = bt.from_arrays(
        np.arange(150) % 2 == 0,
        np.arange(150) % 2 == 1,
        symbol="SBIN",
        exchange="NSE",
        interval="5m",
        size=bt.size.qty(10),
        product="CNC",
    )
    result = bt.backtest(arrays, data=data, base="1m", **FREE)
    check = checked(result, "shifted")
    assert check.verdict == NOT_JUDGED
    assert check.note.startswith("Shifted bars cannot judge this strategy")


# ----- randomized exits -----------------------------------------------------------------


class DipFlip(bt.Strategy):
    """Buys 10 shares (MIS) at the close of a session's even 5-minute bars, with a stop
    `stop_points` below, and exits at the close of the bar after."""

    symbol, exchange, interval = "SBIN", "NSE", "5m"
    stop_points = 2.0

    def init(self):
        self.day, self.count = None, -1

    def on_bar(self, bar):
        day = pd.Timestamp(bar.ts_close - 1, tz="UTC").tz_convert(IST).date()
        self.day, self.count = day, (self.count + 1 if day == self.day else 0)
        if self.position.is_flat and self.count % 2 == 0:
            self.buy(qty=10, stop=bt.points(self.stop_points))
        elif self.position.is_long:
            self.exit()


def dip_run(dip: float):
    """`DipFlip` on two sessions of 5-minute bars: each odd bar opens at 500, dips `dip`
    below and closes at 500.50; each even bar opens at 500.50 and closes at 500. A trade
    buys at an odd bar's open and sells at the next even bar's open: 5 rupees."""

    def shape(k: int) -> tuple[float, float]:
        return (500.0, 500.5) if k % 75 % 2 else (500.5, 500.0)

    data = minute_bars(shape, minutes=5)
    odd = np.arange(len(data)) % 75 % 2 == 1
    data.loc[odd, "low"] = 500.0 - dip
    return bt.backtest(DipFlip, data=data, base="5m", **FREE)


def test_a_stop_fitted_just_below_the_dips_fails_when_its_distance_varies():
    # The 2-point stop sits 0.10 below each 1.90 dip, so the run never meets it and every
    # trade makes 5 rupees. A factor of 0.95 or less (two trades in five, within 0.75 to
    # 1.25) brings the stop above the dip: a loss of 15 to 19 rupees, three times a
    # winner, so a run with varied exits makes money only when fewer than a quarter of
    # its trades draw one, against two in five expected.
    result = dip_run(1.9)
    trades = result.trade_list()
    assert len(trades) >= 60 and (trades["net_pnl"] == 5.0).all()
    check = checked(result, "random_exits")
    assert check.verdict == FAILED
    assert check.summary["profitable_share"] < 0.7


def test_exits_no_factor_reaches_pass():
    # A dip of 1.00: a stop 2 points below, scaled to 1.50 at the closest, is never met.
    result = dip_run(1.0)
    check = checked(result, "random_exits")
    assert check.verdict == PASSED
    assert (check.table["net_profit"] == result.metrics["net_profit"]).all()


def test_the_runs_percentile_among_the_varied_exits_reads_as_an_ordinal():
    # The stop is never met, so every run makes the run's own net profit: none is below.
    report = dip_run(1.0).robustness(
        checks=["random_exits"], workers=1, runs=20, objective="net_profit", progress=quiet
    )
    (tiles,) = report._blocks["random_exits"][-1]
    shown = {tile.label: (tile.value, tile.sub) for tile in tiles.tiles}
    assert shown["The run's percentile"] == ("0th", "0.0% of the runs below it")


def test_exit_parameters_are_drawn_per_run_among_their_values():
    # A stop of 0.5 points (0.625 at most when scaled) is always met by the 1.00 dip, one
    # of 3.0 (2.25 at least) never: a run makes money exactly when it drew 3.0.
    exits = {"exit_params": {"stop_points": [0.5, 3.0]}}
    check = checked(dip_run(1.0), "random_exits", random_exits=exits)
    drawn = check.table["stop_points"]
    assert set(drawn) == {0.5, 3.0}
    assert ((check.table["net_profit"] > 0) == (drawn == 3.0)).all()


def test_a_strategy_with_no_engine_exit_has_nothing_to_vary():
    check = checked(every4_run([500.0, 500.0, 500.0, 503.0]), "random_exits")
    assert check.verdict == NOT_JUDGED
    assert check.note.startswith("Nothing to vary")


def test_an_exit_parameter_the_strategy_does_not_have_is_refused():
    with pytest.raises(bt.errors.ConfigError, match="not parameters of DipFlip"):
        checked(dip_run(1.0), "random_exits", random_exits={"exit_params": {"stop": [1.0]}})


def test_the_same_seed_gives_the_same_varied_exits():
    result = dip_run(1.9)
    once, again = checked(result, "random_exits"), checked(result, "random_exits")
    pd.testing.assert_frame_equal(once.table, again.table)


# ----- what they share ------------------------------------------------------------------


def test_the_plan_names_reruns_on_the_runs_own_bars():
    lines = []
    every4_run([500.0, 500.0, 500.0, 503.0]).robustness(
        checks=["delayed", "liquidity"], workers=1, progress=lines.append
    )
    assert (
        "rerun the backtest 14 times or fewer on the run's own bars, each with one setting "
        "changed (delayed 8, liquidity 6)"
    ) in lines[0]
    assert "A rerun on the run's own bars took" in lines[0]


def test_the_checks_change_no_figure_of_the_run():
    result = every4_run([500.0, 500.0, 500.0, 503.0])
    before = result.metrics_table()
    result.robustness(checks=["delayed", "liquidity"], workers=1, progress=quiet)
    pd.testing.assert_frame_equal(result.metrics_table(), before)


@pytest.mark.parametrize("name", ["delayed", "liquidity", "shifted"])
def test_each_section_draws_the_equity_and_a_table(name):
    result = bt.backtest(Flip5, data=minute_bars(rising), base="1m", **FREE)
    report = result.robustness(checks=[name], workers=1, progress=quiet)
    rows = report._blocks[name]
    kinds = [type(block).__name__ for row in rows for block in row]
    assert kinds[0] == "PathsBlock" and "TableBlock" in kinds
    page = report.tearsheet()
    assert page.isascii() and f"id='{name}'" in page
