"""The parts of the bar checks (design 17, sections 2.4 to 2.9): the stress rules on
session closes worked by hand, random entries placed among the strategy bars, the bars a
run read and their reruns, which reproduce the run on its own bars and are the same for
a seed in any process.
"""

import math

import numpy as np
import pandas as pd
import pytest
from robust_runs import edge_run, opens_bars

import openalgo.backtest as bt
from openalgo.backtest._robust._bar_common import break_even
from openalgo.backtest._robust._paths import fan_steps
from openalgo.backtest._robust._random_entries import (
    EntryPlan,
    Trade,
    entry_plan,
    schedule,
)
from openalgo.backtest._robust._rerun_one import Known, rerun
from openalgo.backtest._robust._reruns import (
    PERMUTED,
    REAL,
    Family,
    Reruns,
    Setup,
    Workers,
    choose_workers,
)
from openalgo.backtest._robust._run_bars import run_bars
from openalgo.backtest._robust._stress_rules import (
    deepest_below_average,
    held,
    longest_recovery,
    max_drawdown,
    prices,
    return_correlation,
    rules_value,
    stress_rules,
)
from openalgo.indicators import ta

# ----- stress rules --------------------------------------------------------------------


def test_the_maximum_drawdown_is_the_deepest_fall_below_the_running_high():
    # 120 then 90: a fall of a quarter; 130 is a new high.
    assert max_drawdown(np.array([100.0, 120.0, 90.0, 130.0])) == pytest.approx(-0.25)


def test_the_longest_recovery_counts_sessions_below_the_running_high():
    # Below the high of 120 at 90 and 110 (2 sessions), below 130 at 125 (1, to the end).
    assert longest_recovery(np.array([100.0, 120.0, 90.0, 110.0, 130.0, 125.0])) == 2
    assert longest_recovery(np.array([100.0, 101.0, 102.0])) == 0


def test_the_deepest_fall_below_the_average_is_measured_over_the_run():
    # 3-session averages: undefined, undefined, 10, 9, 9; the run is the last three
    # sessions, whose closes over their averages less one are 0, 7/9 - 1 and 10/9 - 1.
    closes = np.array([10.0, 10.0, 10.0, 7.0, 10.0])
    in_run = np.array([False, False, True, True, True])
    assert deepest_below_average(closes, in_run, 3) == pytest.approx(7 / 9 - 1)
    assert math.isnan(deepest_below_average(closes, in_run, 6))


def test_the_correlation_is_of_the_session_returns():
    up_down = np.array([100.0, 110.0, 99.0, 108.9])
    assert return_correlation(up_down, up_down) == pytest.approx(1.0)
    assert return_correlation(up_down, up_down[::-1]) == pytest.approx(-1.0, abs=0.05)
    assert math.isnan(return_correlation(up_down, np.full(4, 100.0)))


def _prices(opens):
    bars = opens_bars(opens)
    frame = pd.DataFrame(
        {
            "ts_open": bars.index.as_unit("ns").asi8,
            "ts_close": bars.index.as_unit("ns").asi8 + 1,
            "close": bars["close"].to_numpy(),
        }
    )
    return prices(frame, int(frame["ts_open"].iloc[0]))


SETTINGS = {"average": 200, "correlation": 0.5, "shock": 1.5}


# The real closes 100, 120, 90, 130, 140 fall a quarter from their high (120 to 90) and
# stay below it 1 session; a shock is half as deep again or as long again (`shock` 1.5).
@pytest.mark.parametrize(
    ("name", "candidate", "expected"),
    [
        ("drawdown", [100, 120, 70, 130], True),  # 120 to 70: 0.417, at least 0.375
        ("drawdown", [100, 120, 80, 130], False),  # 0.333: deeper, but not a shock
        ("drawdown", [100, 120, 100, 130], False),
        ("recovery", [100, 120, 110, 105, 130], True),  # 2 sessions below, against 1
        ("recovery", [100, 120, 110, 125, 130], False),
        ("correlation", [100, 90, 99, 89.1, 98], True),  # returns that move opposite
        ("correlation", [100, 120, 90, 130, 140], False),
    ],
)
def test_a_built_in_rule_holds_when_the_candidate_is_a_shock(name, candidate, expected):
    real = _prices([100, 120, 90, 130, 140])
    (rule,) = stress_rules([name], SETTINGS)
    assert held([rule], [real], [_prices(candidate)], permuted=False) == (expected,)


# With a 2-session average the real closes fall at most 90 / 105 - 1 = -0.143 below it;
# half as deep again is 0.214. 75 / 97.5 - 1 = -0.231 is a shock, 80 / 100 - 1 = -0.2 not.
@pytest.mark.parametrize(("low", "expected"), [(75, True), (80, False)])
def test_a_fall_below_the_average_is_a_shock_half_as_deep_again(low, expected):
    real, candidate = _prices([100, 120, 90, 130, 140]), _prices([100, 120, low, 130, 140])
    (rule,) = stress_rules(["below_average"], {**SETTINGS, "average": 2})
    assert held([rule], [real], [candidate], permuted=False) == (expected,)


def test_any_fall_is_a_shock_for_a_series_that_never_fell():
    real, candidate = _prices([100, 101, 102]), _prices([100, 99.5, 102])
    (rule,) = stress_rules(["drawdown"], SETTINGS)
    assert held([rule], [real], [candidate], permuted=False) == (True,)


def test_the_correlation_rule_reads_noisy_candidates_only():
    # A permuted path is the run's own moves in another order: on daily bars its returns
    # are uncorrelated with the real ones by construction, which is no shock.
    real, candidate = _prices([100, 120, 90, 130, 140]), _prices([100, 90, 99, 89.1, 98])
    (rule,) = stress_rules(["correlation"], SETTINGS)
    assert held([rule], [real], [candidate], permuted=False) == (True,)
    assert held([rule], [real], [candidate], permuted=True) == (False,)


def test_the_default_rules_are_the_three_that_find_a_shock_on_either_generator():
    from openalgo.backtest._robust._check_stress import SETTINGS as STRESS

    defaults = {setting.name: setting.default for setting in STRESS}
    assert defaults["rules"] == ("drawdown", "recovery", "below_average")
    assert defaults["shock"] == 1.5


def test_a_rule_given_as_a_function_reads_the_runs_bars():
    real, candidate = _prices([100, 101, 102]), _prices([100, 99, 98])
    (rule,) = stress_rules([lambda a, b: b["close"].iloc[-1] < a["close"].iloc[-1]], SETTINGS)
    assert held([rule], [real], [candidate], permuted=True) == (True,)
    assert rule.name == "<lambda>"


def _first(real, candidate):
    return True


def _second(real, candidate):
    return False


@pytest.mark.parametrize(
    "value",
    [
        "drawdown",
        [],
        ["drawdown", "bogus"],
        [3],
        ["drawdown", "drawdown"],  # one name twice
        [lambda a, b: True, lambda a, b: False],  # two functions both named <lambda>
    ],
)
def test_rules_that_name_no_rule_or_one_name_twice_are_refused(value):
    assert rules_value(value) is None


def test_rules_with_a_name_each_are_taken():
    assert rules_value(["drawdown", _first, _second]) == ("drawdown", _first, _second)


def test_two_rules_of_one_name_are_refused_with_what_to_do():
    from openalgo.backtest.errors import ConfigError

    with pytest.raises(ConfigError, match="each with a name of its own"):
        edge_run().robustness(checks=["stress"], stress={"rules": ["drawdown", "drawdown"]})


# ----- judging the kept candidates -----------------------------------------------------


def _set(net_profit):
    """A RerunSet of runs that ran, with these net profits, rupees."""
    from openalgo.backtest._robust._bar_common import RerunSet

    pnl = np.asarray(net_profit, dtype=float)
    count = np.zeros(len(pnl), dtype=np.int64)
    runs = len(pnl)
    return RerunSet(
        pnl,
        pnl,
        pnl,
        count,
        pnl,
        pnl,
        count,
        count,
        np.zeros((runs, 2)),
        ("",) * runs,
        ((),) * runs,
    )


def test_a_rules_kept_candidates_are_judged_against_their_generators_share():
    # Ten noisy candidates all make 100 (their generator's share: 100%); ten permuted
    # ones, of which the first four make 50 and the rest lose 50 (40%).
    #   a: noisy 0-4 and permuted 0-4: 9 of 10 profitable, expected (5 x 1 + 5 x 0.4) / 10
    #      = 0.7, so 1.29 times
    #   b: permuted 2-9: 2 of 8, expected 0.4, 0.625 times, at least 0.5: kept
    #   c: noisy 5-9: 5 kept, fewer than min_kept 6: not judged
    #   d: permuted 4-9: 0 of 6, 0 times: it fails the check
    from openalgo.backtest._robust._stress_rules import StressRule
    from openalgo.backtest._robust._stress_table import Candidates, rule_table

    generators = np.array(["noisy"] * 10 + ["permuted"] * 10)
    holds = {"a": [*range(5), *range(10, 15)], "b": range(12, 20), "c": range(5, 10)}
    holds["d"] = range(14, 20)
    flags = np.zeros((20, 4), dtype=bool)
    for k, rows in enumerate(holds.values()):
        flags[list(rows), k] = True
    candidates = Candidates(generators, flags, "")
    rules = [StressRule(name, name, lambda a, b: True, True) for name in holds]
    pnl = [100.0] * 10 + [50.0] * 4 + [-50.0] * 6
    settings = {"combine": "any", "min_kept": 6, "keep": 0.5}
    table = rule_table(rules, candidates, _set(pnl), settings)
    assert table["kept"].tolist() == [10, 8, 5, 6]
    assert table["profitable_share"].tolist() == pytest.approx([0.9, 0.25, 1.0, 0.0])
    assert table["expected_share"].tolist() == pytest.approx([0.7, 0.4, 1.0, 0.4])
    assert table["ratio"].tolist() == pytest.approx([0.9 / 0.7, 0.625, 1.0, 0.0])
    assert table["judged"].tolist() == [True, True, False, True]
    assert table["passed"].tolist() == [True, True, False, False]


# ----- random entries ------------------------------------------------------------------


def test_trades_of_opposite_directions_that_fill_every_bar_are_placed_back_to_back():
    # Six strategy bars, a long and a short held 2 bars: slack 6 - 2 - 4 = 0. The first
    # is decided at bar 0 and exited at bar 2, where the second, of the other direction,
    # is decided (a reversal fills cleanly); it is exited at bar 4, the last whose
    # decision can fill.
    closes = np.arange(6) * 100
    plan = EntryPlan(
        (Trade("SBIN", "BUY", 10, "CNC", 2), Trade("SBIN", "SELL", 5, "MIS", 2)), {"SBIN": closes}
    )
    assert plan.slack("SBIN") == 0
    decided = schedule(plan, seed=1, run=0)
    assert sorted(decided) == [0, 200, 400]
    assert [action.side for action in decided[200]][0] is None  # the exit comes first
    assert len(decided[0]) == 1 and len(decided[400]) == 1


def test_trades_of_one_direction_leave_a_bar_between_an_exit_and_the_next_entry():
    # Seven strategy bars, two longs held 2 bars: slack 7 - 2 - 4 - 1 = 0, the 1 being the
    # bar a long waits after the exit before it, when the position is flat again (an
    # entry decided at the exit's close would add to the open position). Decided: entry
    # at bar 0, exit at 2, entry at 3, exit at 5, the last whose decision can fill.
    closes = np.arange(7) * 100
    trades = (Trade("SBIN", "BUY", 10, "CNC", 2), Trade("SBIN", "BUY", 5, "CNC", 2))
    plan = EntryPlan(trades, {"SBIN": closes})
    assert plan.slack("SBIN") == 0
    decided = schedule(plan, seed=1, run=0)
    assert sorted(decided) == [0, 200, 300, 500]
    assert [len(decided[when]) for when in sorted(decided)] == [1, 1, 1, 1]


def test_two_longs_that_need_their_bar_between_do_not_fit_in_six_bars():
    closes = np.arange(6) * 100
    trades = (Trade("SBIN", "BUY", 10, "CNC", 2), Trade("SBIN", "BUY", 5, "CNC", 2))
    plan = EntryPlan(trades, {"SBIN": closes})
    assert plan.slack("SBIN") == -1 and plan.unplaced() == ["SBIN"]


@pytest.mark.parametrize("run", range(6))
def test_random_trades_keep_the_runs_order_of_directions(run):
    # The run went long, long, short, long, short: each place keeps its direction and
    # takes one of that direction's trades (its units and holding time), drawn.
    sides = ("BUY", "BUY", "SELL", "BUY", "SELL")
    trades = tuple(
        Trade("SBIN", side, units, "MIS", hold)
        for side, units, hold in zip(sides, (1, 2, 3, 4, 5), (1, 2, 3, 1, 2), strict=True)
    )
    decided = schedule(EntryPlan(trades, {"SBIN": np.arange(60) * 10}), seed=3, run=run)
    entered = [action for when in sorted(decided) for action in decided[when] if action.side]
    assert [action.side for action in entered] == list(sides)
    assert sorted(action.units for action in entered if action.side == "BUY") == [1, 2, 4]
    assert sorted(action.units for action in entered if action.side == "SELL") == [3, 5]


@pytest.mark.parametrize("run", range(5))
def test_random_trades_never_overlap_and_keep_their_holding_times(run):
    closes = np.arange(40) * 10
    holds = (3, 1, 5, 2)
    trades = tuple(Trade("SBIN", "BUY", 1, "CNC", hold) for hold in holds)
    decided = schedule(EntryPlan(trades, {"SBIN": closes}), seed=7, run=run)
    spans, open_at, last_exit = [], None, -1
    for when in sorted(decided):
        for action in decided[when]:
            if action.side is None:
                spans.append((open_at, when // 10))
                open_at, last_exit = None, when // 10
            else:
                assert open_at is None  # never two trades at once
                assert when // 10 > last_exit  # a long waits for the long before to close
                open_at = when // 10
    assert sorted(end - start for start, end in spans) == sorted(holds)
    assert spans[-1][1] <= 38  # the last exit can still fill


def test_trades_that_do_not_fit_are_named():
    closes = np.arange(5)
    plan = EntryPlan((Trade("SBIN", "BUY", 1, "CNC", 4),), {"SBIN": closes})
    assert plan.slack("SBIN") == -1 and plan.unplaced() == ["SBIN"]


@pytest.mark.parametrize(("hold", "slack"), [(4, -1), (5, -2)])
def test_a_schedule_of_trades_that_do_not_fit_is_refused(hold, slack):
    from openalgo.backtest.errors import ConfigError

    plan = EntryPlan((Trade("SBIN", "BUY", 1, "CNC", hold),), {"SBIN": np.arange(5)})
    assert plan.slack("SBIN") == slack
    with pytest.raises(ConfigError, match="the trades of SBIN do not fit among its 5"):
        schedule(plan, seed=1, run=0)


def test_the_entry_plan_holds_the_runs_trades_with_their_holding_times():
    result = edge_run()
    plan = entry_plan(result.trade_list(), {"SBIN": np.arange(10)})
    assert len(plan.trades) == 60
    assert {trade.hold for trade in plan.trades} == {1}  # Flip holds one bar
    assert {(trade.side, trade.units) for trade in plan.trades} == {("BUY", 10.0)}


# ----- the run's bars and its reruns ---------------------------------------------------


class Ema(bt.Strategy):
    """Long while a 5-bar EMA is above a 20-bar one: its indicators need history."""

    symbol, exchange, interval = "SBIN", "NSE", "D"

    def init(self):
        self.fast = self.I(ta.ema, self.close, 5)
        self.slow = self.I(ta.ema, self.close, 20)

    def on_bar(self, bar):
        if self.position.is_flat and self.fast[-1] > self.slow[-1]:
            self.buy(qty=10)
        elif self.position.is_long and self.fast[-1] < self.slow[-1]:
            self.exit()


@pytest.fixture(scope="module")
def ema_run():
    rng = np.random.default_rng(5)
    opens = 500 + np.cumsum(np.round(rng.normal(0.1, 3.0, 400) / 0.05) * 0.05)
    data = opens_bars(opens)
    return bt.backtest(
        Ema, data=data, start=data.index[150].date(), capital=100_000, costs=None, slippage=0
    )


def test_the_runs_bars_are_its_base_bars_after_the_history_its_indicators_keep(ema_run):
    bars, arguments = run_bars(ema_run)
    (frame,) = bars.frames
    assert bars.symbols == ("SBIN",) and bars.ticks == (0.05,)
    history = frame[frame["ts_open"] < bars.first_run_ns]
    assert len(history) == 150 and len(frame) == 400  # all 150 sessions before start
    assert arguments["data"] is None and arguments["tz"] is None and arguments["base"] == "D"


def test_a_rerun_on_the_runs_own_bars_reproduces_the_run(ema_run):
    bars, arguments = run_bars(ema_run)
    again, known = rerun(Ema, arguments, bars.symbols, list(bars.frames), Known())
    assert again.metrics["sharpe"] == ema_run.metrics["sharpe"]
    pd.testing.assert_frame_equal(again.trade_list(), ema_run.trade_list())
    assert known.measured  # the warm-ups measured once, for the reruns after it


@pytest.fixture(scope="module")
def setup():
    result = edge_run()
    bars, arguments = run_bars(result)
    _, known = rerun(strategy_of(result), arguments, bars.symbols, list(bars.frames), Known())
    dates = result.performance.session_returns.index
    return Setup(
        ("ref", strategy_of(result)),
        ("ref", "sharpe"),
        arguments,
        bars,
        known,
        dates,
        fan_steps(len(dates)),
        None,
    )


def strategy_of(result):
    """The strategy class of a run."""
    return result._record.strategy


def test_the_same_family_and_run_give_the_same_rerun(setup):
    family = Family(PERMUTED, 3)
    with Reruns(setup, Workers(1, "")) as first, Reruns(setup, Workers(1, "")) as second:
        a = first.rows(family, [0, 1, 2])
        b = second.rows(family, [2, 1, 0])
    assert [row.objective for row in a] == [row.objective for row in b][::-1]


def test_a_rerun_asked_twice_runs_once(setup):
    with Reruns(setup, Workers(1, "")) as reruns:
        reruns.rows(Family(PERMUTED, 3), [0, 1])
        reruns.rows(Family(PERMUTED, 3), [1, 2])
        assert reruns.done == 3


@pytest.mark.worker_pool
def test_reruns_in_two_worker_processes_are_the_reruns_of_this_process(setup):
    family = Family(PERMUTED, 8)
    with Reruns(setup, Workers(1, "")) as here:
        local = [row.objective for row in here.rows(family, range(4))]
    pooled_setup = Setup(
        ("ref", setup.strategy[1]),
        ("ref", "sharpe"),
        *[
            getattr(setup, name)
            for name in ("arguments", "bars", "known", "dates", "steps", "entries")
        ],
    )
    with Reruns(pooled_setup, Workers(2, "")) as pool:
        pooled = [row.objective for row in pool.rows(family, range(4))]
    assert pooled == local


def test_the_runs_own_bars_rerun_as_the_run(setup):
    with Reruns(setup, Workers(1, "")) as reruns:
        (row,) = reruns.rows(Family(REAL), [0])
    assert row.objective == edge_run().metrics["sharpe"]


@pytest.mark.parametrize(
    ("requested", "total", "expected"),
    [(1, 100, 1), (4, 100, 4), (4, 2, 2), (4, 1, 1)],
)
def test_the_workers_are_asked_for_but_never_more_than_the_reruns(requested, total, expected):
    from robust_runs import Flip

    assert choose_workers(requested, total, Flip, {"costs": None}).count == expected


def test_a_strategy_made_at_run_time_reruns_in_this_process():
    strategy = bt.from_arrays([True], [False], symbol="SBIN", exchange="NSE", interval="D")
    chosen = choose_workers(4, 100, strategy, {})
    assert chosen.count == 1 and "bt.from_arrays" in chosen.why


def net_profit_of(result):
    """A function of a Result: its net profit in rupees."""
    return float(result.metrics["net_profit"])


# (objective, break-even): what a run that makes nothing scores, or None for an objective
# that cannot say whether a run made money (a win rate, a drawdown, a trade count) and for
# any function of a Result, even one that measures profit: what a function gives for a run
# that makes nothing is not known (the end equity gives the capital, not 0).
BREAK_EVENS = [
    ("sharpe", 0.0),
    ("sortino", 0.0),
    ("net_profit", 0.0),
    ("expectancy", 0.0),
    ("profit_factor", 1.0),
    ("calmar", 0.0),
    ("net_pnl", 0.0),
    ("total_return", 0.0),
    ("cagr", 0.0),
    (net_profit_of, None),
    ("win_rate", None),
    ("max_drawdown", None),
    ("trades", None),
]


@pytest.mark.parametrize(("objective", "expected"), BREAK_EVENS)
def test_the_bar_checks_measure_an_objective_from_its_break_even(objective, expected):
    assert break_even(objective) == expected
