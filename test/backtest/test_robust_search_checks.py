"""The searching checks (design 17, sections 2.1, 2.3, 2.16 and 2.17) and `bt.walk_forward`
end to end, on own daily bars with no server (one walk-forward reads them through a fake
client), in this process but for the one test of the worker processes.

`robust_runs.Weekday` buys 10 shares at the close of a session on its weekday `day` and
sells two sessions on; `weekday_bars` plants the move each trade makes: 3 rupees a share
when its weekday is favoured, -1 when it is not, so a trade makes +30 or -10 rupees and
`weekday_trades` counts a run's trades by hand. A planted edge favours one weekday over
the whole run; a rotation changes the favoured weekday at each walk-forward boundary,
so what one window favours the next does not.
"""

import threading
from itertools import product

import numpy as np
import pandas as pd
import pytest
from data_fixtures.fakes import FakeClient
from robust_runs import (
    FREE,
    DailyCross,
    Flip,
    Weekday,
    bars,
    nse_sessions,
    random_walk,
    steps_for,
    weekday_bars,
    weekday_trades,
)
from test_quickstart import historify_rows

import openalgo.backtest as bt
from openalgo.backtest._robust import _check_parameters
from openalgo.backtest._robust._grid import Grid
from openalgo.backtest._robust._verdict import FAILED, INFORMATIONAL, NOT_JUDGED, PASSED
from openalgo.backtest._robust._walk import FoldOutcome, WalkOutcome
from openalgo.backtest._robust._walk_tables import fold_frame
from openalgo.backtest._robust._windows import folds

#: A trade on a favoured weekday and on another, rupees (10 shares x 3 and x -1).
WIN, LOSS = 30.0, -10.0
SESSIONS = 300
#: The walk-forward runs: 400 sessions in 5 rolling folds at 70 percent in sample, whose
#: out-of-sample windows start at sessions 127, 182, 236, 291 and 345 (round(400 (r + j) /
#: (r + 5)), r = 0.7 / 0.3).
WALK_SESSIONS = 400
BOUNDS = (127, 182, 236, 291, 345, 400)
FIVE = {"day": range(5)}


def quiet(line: str) -> None:
    """A progress function that drops the plan."""


def report(data, checks, objective="net_profit", params=None, **settings):
    """The searching `checks` on `Weekday` over `data` (with `params=`), in this process."""
    result = bt.backtest(Weekday, data=data, params=params, **FREE)
    settings.setdefault("progress", quiet)
    return result.robustness(checks=checks, workers=1, objective=objective, **settings)


def regime(session: int) -> int:
    """The walk-forward window a session falls in: 0 before the first out-of-sample one."""
    return sum(1 for bound in BOUNDS[:-1] if bound <= session)


@pytest.fixture(scope="module")
def wednesday():
    """Bars that favour Wednesday trades over the whole run."""
    return weekday_bars(SESSIONS, lambda s: (2,))


@pytest.fixture(scope="module")
def every_day():
    """Bars on which every trade wins."""
    return weekday_bars(SESSIONS, lambda s: range(5))


def test_the_hand_count_of_trades_is_the_runs():
    data = weekday_bars(SESSIONS, lambda s: (2,))
    result = bt.backtest(Weekday, data=data, params={"day": 2}, **FREE)
    trades = len(weekday_trades(SESSIONS, 2))
    assert result.metrics["trades"] == trades
    assert result.metrics["net_profit"] == WIN * trades


# ----- 2.16 parameter neighbourhood -------------------------------------------------------


def test_a_lone_spike_fails_the_parameter_neighbourhood(wednesday):
    # Day 2 at 10 percent: days 1, 2 and 3. Only Wednesday trades win, so both neighbours
    # lose 10 a trade: the median keeps -(n1 + n3) 10 / 2 of the chosen n2 x 30.
    check = report(wednesday, ["parameters"], params={"day": 2}).check("parameters")
    n1, n2, n3 = (len(weekday_trades(SESSIONS, day)) for day in (1, 2, 3))
    assert check.verdict == FAILED
    assert check.summary["profitable_share"] == 0.0
    assert check.summary["median_ratio"] == pytest.approx(LOSS * (n1 + n3) / 2 / (WIN * n2))
    assert list(check.table["day"]) == [1, 2, 3]
    assert list(check.table["neighbour"]) == [True, False, True]


def test_a_plateau_passes_the_parameter_neighbourhood(every_day):
    # Every trade wins 30: each day's run makes 30 a trade, so a neighbour keeps its
    # share of the chosen day's trades.
    check = report(every_day, ["parameters"], params={"day": 2}).check("parameters")
    n1, n2, n3 = (len(weekday_trades(SESSIONS, day)) for day in (1, 2, 3))
    assert check.verdict == PASSED
    assert check.summary["profitable_share"] == 1.0
    assert check.summary["median_ratio"] == pytest.approx((n1 + n3) / 2 / n2)


def test_with_ranges_the_neighbourhood_is_the_grids(wednesday):
    # Day 2 among days 0 to 4: its neighbours on the grid are days 1 and 3, and the
    # chosen one is above the 4 others.
    check = report(wednesday, ["parameters"], params={"day": 2}, **FIVE).check("parameters")
    assert list(check.table["neighbour"]) == [False, True, False, True, False]
    assert check.summary["percentile"] == 1.0


class DeclaredWeekday(Weekday):
    """`Weekday` with its weekday declared as a range to search, Wednesday by default."""

    day = bt.param(2, range(5))


@pytest.fixture(scope="module")
def declared(wednesday):
    """The searching checks of a `DeclaredWeekday` run whose params= name the setting a
    search picked, day 2."""
    result = bt.backtest(DeclaredWeekday, data=wednesday, params={"day": 2}, **FREE)
    return result.robustness(
        checks=["out_of_sample", "walk_forward", "parameters", "noise_optimization"],
        workers=1,
        objective="net_profit",
        progress=quiet,
        walk_forward={"folds": 2, "min_trades": 20},
        noise_optimization={"samples": 2},
    )


def test_params_given_are_the_chosen_point_inside_the_declared_ranges(declared):
    # params= day 2 is the run's chosen setting; the declared range of day, 0 to 4, stays
    # the space the searching checks search, so day 2 is judged among days 0 to 4.
    parameters = declared.check("parameters")
    assert list(parameters.table["day"]) == [0, 1, 2, 3, 4]
    assert list(parameters.table["chosen"]) == [False, False, True, False, False]
    assert list(declared.check("noise_optimization").table["day"]) == [0, 1, 2, 3, 4]
    searched = declared.check("out_of_sample").table.dropna(subset=["day"])
    assert list(searched["day"]) == [0, 1, 2, 3, 4]


def test_params_given_leave_the_declared_ranges_to_re_fit(declared):
    # Each of the two folds searches days 0 to 4 in sample and picks Wednesday.
    walked = declared.check("walk_forward")
    assert walked.verdict == PASSED, walked.note
    assert list(walked.table["day"]) == [2, 2]


def test_the_best_crossover_fitted_to_a_random_walk_is_a_lone_spike_of_its_declared_grid():
    # The trader's workflow on a history with no edge: optimize DailyCross's 81 declared
    # settings, then check the best by giving it as params=. The neighbourhood judges it
    # inside the declared grid, where the settings next to the luckiest did not hold up.
    # A slow crossover trades seldom, so the check is asked to judge from 10 trades.
    data = random_walk(1_000, 1)
    best = bt.optimize(
        DailyCross, data=data, **FREE, objective="sharpe", workers=1, progress=quiet
    ).best
    result = bt.backtest(DailyCross, data=data, params=best, **FREE)
    found = result.robustness(
        checks=["parameters"], workers=1, progress=quiet, parameters={"min_trades": 10}
    )
    check = found.check("parameters")
    chosen = check.table[check.table["chosen"]]
    assert len(check.table) == 81
    assert chosen[["fast", "slow"]].to_dict("records") == [best]
    assert check.verdict == FAILED


def test_bt_walk_forward_reads_params_as_bt_optimize_does():
    # A declared parameter that params= names is fixed, as bt.optimize fixes it, so a
    # strategy with one declared parameter has nothing left to search.
    data = weekday_bars(100, lambda s: (0,))
    with pytest.raises(bt.errors.ConfigError, match="nothing to search"):
        bt.walk_forward(DeclaredWeekday, data=data, **FREE, params={"day": 0}, workers=1)


def test_bt_walk_forward_says_params_fixed_the_declared_range_it_would_search():
    # The trader declared day's range; the error says params= fixed it, not that no range
    # was declared, and how to search it.
    data = weekday_bars(100, lambda s: (0,))
    with pytest.raises(bt.errors.ConfigError) as raised:
        bt.walk_forward(DeclaredWeekday, data=data, **FREE, params={"day": 0}, workers=1)
    assert str(raised.value).endswith(
        "params= fixes ['day'], whose declared ranges it would search: leave them out of "
        "params= to search them"
    )


def test_bt_walk_forward_refuses_a_parameter_both_fixed_by_params_and_given_a_range():
    # As bt.optimize refuses it: params= fixes day and day=range(5) sweeps it.
    data = weekday_bars(100, lambda s: (0,))
    with pytest.raises(bt.errors.ConfigError, match=r"\['day'\] are both fixed by params="):
        bt.walk_forward(Weekday, data=data, **FREE, params={"day": 0}, workers=1, **FIVE)


def test_a_strategy_without_a_number_parameter_is_not_judged():
    result = bt.backtest(Flip, data=bars(steps_for([3.0, -1.0], 40)), **FREE)
    check = result.robustness(checks=["parameters"], workers=1, progress=quiet).check("parameters")
    assert check.verdict == NOT_JUDGED and check.note.startswith("No parameter to vary")


def test_parameters_off_the_grid_given_are_not_judged(wednesday):
    check = report(wednesday, ["parameters"], params={"day": 2}, day=(0, 1, 3, 4))
    found = check.check("parameters")
    assert found.verdict == NOT_JUDGED
    assert found.note.startswith("The run's parameters (day 2) are not a combination")


class MondayOrTuesday(Weekday):
    """`Weekday` with its weekday declared as a range of Monday and Tuesday only."""

    day = bt.param(0, (0, 1))


def test_a_declared_parameter_set_outside_its_range_says_how_to_search_around_it():
    # params= day 3 is a value the declared range (0, 1) does not hold, so its
    # neighbourhood cannot be read off that range. The note names the range and gives the
    # keyword form around the run's value: 3 and a step of 10 percent either side, at
    # least one day, 2 and 4.
    data = weekday_bars(SESSIONS, lambda s: (3,))
    result = bt.backtest(MondayOrTuesday, data=data, params={"day": 3}, **FREE)
    check = result.robustness(checks=["parameters"], workers=1, progress=quiet).check("parameters")
    assert check.verdict == NOT_JUDGED
    assert check.note == (
        "The run's parameters (day 3) are not a combination of the ranges searched: day 3 "
        "is not among the values of its range (0, 1). Give it a range that holds it as a "
        "keyword named after the parameter, for example day=[2, 3, 4]."
    )


def test_a_long_range_a_value_is_off_reads_as_its_ends():
    # fast 5 to 50 in steps of 5, ten values; the run's 7 moves by 10 percent, at least 1.
    grid = Grid(("fast",), tuple({"fast": v} for v in range(5, 55, 5)), {"fast": 7}, True)
    assert _check_parameters._off_the_grid(grid, 0.10) == (
        "The run's parameters (fast 7) are not a combination of the ranges searched: fast 7 "
        "is not among the values of its range (5 to 50, 10 values). Give it a range that "
        "holds it as a keyword named after the parameter, for example fast=[6, 7, 8]."
    )


def test_each_value_off_its_range_is_named_with_a_keyword_range_around_it():
    # A name is not moved: its keyword range holds the run's value alone.
    combos = tuple({"fast": f, "label": n} for f in (5, 10) for n in ("a", "b"))
    grid = Grid(("fast", "label"), combos, {"fast": 7, "label": "c"}, True)
    assert _check_parameters._off_the_grid(grid, 0.10) == (
        "The run's parameters (fast 7, label c) are not a combination of the ranges searched: "
        "fast 7 is not among the values of its range (5, 10); label c is not among the values "
        "of its range (a, b). Give each a range that holds it as a keyword named after the "
        "parameter, for example fast=[6, 7, 8], label=['c']."
    )


def off_grid_of(ranges: dict[str, tuple], run_params: dict[str, object]) -> Grid:
    """The grid of `ranges` around a run whose parameters are `run_params`."""
    combos = tuple(dict(zip(ranges, values, strict=True)) for values in product(*ranges.values()))
    return Grid(tuple(ranges), combos, run_params, True)


#: A float residue: 0.1 + 0.2 is 0.30000000000000004 as a float, not 0.3.
RESIDUE = 0.1 + 0.2


@pytest.mark.parametrize(
    ("ranges", "run_params", "note"),
    [
        # The best of a sweep over np.arange(0.1, 0.7, 0.1) passed back in params= is off
        # the declared range by a float residue only: the note says so, writes both values
        # as they are, and gives the range's value to pass instead.
        (
            {"x": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)},
            {"x": RESIDUE},
            "The run's parameters (x 0.30000000000000004) are not a combination of the "
            "ranges searched: x 0.30000000000000004 differs from 0.3, a value of its range "
            "(0.1, 0.2, 0.3, 0.4, 0.5, 0.6), by a float residue only. Pass the range's value "
            "in params=, for example params={'x': 0.3}.",
        ),
        # A value of more than six digits is written whole, so the keyword range holds it;
        # its neighbours, 10 percent either side, need no more than twelve.
        (
            {"mult": (1.0, 1.5)},
            {"mult": 2.3456789},
            "The run's parameters (mult 2.3456789) are not a combination of the ranges "
            "searched: mult 2.3456789 is not among the values of its range (1.0, 1.5). Give "
            "it a range that holds it as a keyword named after the parameter, for example "
            "mult=[2.11111101, 2.3456789, 2.58024679].",
        ),
        # Both at once: each is named, and each piece of advice names its own.
        (
            {"x": (0.1, 0.2, 0.3), "day": (0, 1)},
            {"x": RESIDUE, "day": 3},
            "The run's parameters (x 0.30000000000000004, day 3) are not a combination of "
            "the ranges searched: x 0.30000000000000004 differs from 0.3, a value of its "
            "range (0.1, 0.2, 0.3), by a float residue only; day 3 is not among the values "
            "of its range (0, 1). Give it a range that holds it as a keyword named after the "
            "parameter, for example day=[2, 3, 4]. Pass the range's value in params=, for "
            "example params={'x': 0.3}.",
        ),
    ],
    ids=["a float residue", "more than six digits", "both"],
)
def test_a_fractional_value_off_its_range_is_written_as_it_is(ranges, run_params, note):
    assert _check_parameters._off_the_grid(off_grid_of(ranges, run_params), 0.10) == note


# ----- 2.1 out of sample ----------------------------------------------------------------


def test_a_planted_edge_keeps_its_figure_out_of_sample(wednesday):
    # Searched over days 0 to 4 on the first 210 sessions, Wednesday wins; out of sample
    # every Wednesday trade wins 30 too, so the expectancy keeps 100 percent.
    check = report(wednesday, ["out_of_sample"], "expectancy", **FIVE).check("out_of_sample")
    trades = len(weekday_trades(SESSIONS, 2, first=210))
    assert check.verdict == PASSED
    assert check.summary["ratio"] == pytest.approx(1.0)
    assert check.summary["out_of_sample_net_profit"] == WIN * trades
    chosen = check.table[check.table["chosen"] == True]  # noqa: E712
    assert list(chosen["day"]) == [2]


def test_an_edge_that_changed_after_the_search_fails_out_of_sample():
    # Monday wins on the first 210 sessions, Wednesday after: the search picks Monday,
    # whose trades lose 10 each out of sample.
    data = weekday_bars(SESSIONS, lambda s: (0,) if s - 2 < 210 else (2,))
    check = report(data, ["out_of_sample"], "expectancy", **FIVE).check("out_of_sample")
    trades = len(weekday_trades(SESSIONS, 0, first=210))
    assert check.verdict == FAILED
    assert check.summary["out_of_sample_net_profit"] == LOSS * trades
    assert check.summary["ratio"] == pytest.approx(LOSS / WIN)


def test_without_ranges_the_strategy_as_given_runs_on_both_parts(wednesday):
    check = report(wednesday, ["out_of_sample"], "expectancy", params={"day": 2}).check(
        "out_of_sample"
    )
    assert check.verdict == PASSED
    assert list(check.table["part"]) == ["in sample", "out of sample"]
    assert list(check.table["trades"]) == [
        len(weekday_trades(SESSIONS, 2, last=209)),
        len(weekday_trades(SESSIONS, 2, first=210)),
    ]


# ----- 2.3 walk-forward and bt.walk_forward ------------------------------------------------


def test_the_folds_bounds_are_the_designs():
    made = folds(WALK_SESSIONS, 5, 0.7, anchored=False)
    assert [fold.out_of_sample.first for fold in made] == list(BOUNDS[:-1])


@pytest.fixture(scope="module")
def planted_walk():
    data = weekday_bars(WALK_SESSIONS, lambda s: (0,))
    return data, report(data, ["walk_forward"], "expectancy", **FIVE).check("walk_forward")


def test_a_planted_edge_holds_when_refitted_through_time(planted_walk):
    # Every fold picks Monday and every out-of-sample Monday trade wins 30: efficiency 1,
    # every fold profitable, no drift.
    _, check = planted_walk
    assert check.verdict == PASSED
    assert check.summary["efficiency"] == pytest.approx(1.0)
    assert check.summary["profitable_share"] == 1.0 and check.summary["drift"] == 0.0
    assert list(check.table["day"]) == [0] * 5
    out = [
        WIN * len(weekday_trades(WALK_SESSIONS, 0, BOUNDS[k], BOUNDS[k + 1] - 1)) for k in range(5)
    ]
    assert list(check.table["net_profit"]) == out
    assert check.summary["net_profit"] == sum(out)


def test_bt_walk_forward_walks_as_the_check_does(planted_walk):
    data, check = planted_walk
    result = bt.walk_forward(
        Weekday, data=data, **FREE, objective="expectancy", workers=1, progress=quiet, **FIVE
    )
    pd.testing.assert_frame_equal(result.folds, check.table)
    assert result.efficiency == check.summary["efficiency"]
    assert result.net_profit == check.summary["net_profit"]
    assert result.drift.to_dict() == {"day": 0.0}
    assert isinstance(result, bt.WalkForwardResult)


def test_an_edge_that_rotates_fails_the_walk_forward():
    # Window k favours weekday k (mod 5): each fold picks the weekday its in-sample window
    # favoured most, never the one its out-of-sample window favours, and loses 10 a trade.
    data = weekday_bars(WALK_SESSIONS, lambda s: (regime(s - 2) % 5,))
    check = report(data, ["walk_forward"], "expectancy", **FIVE).check("walk_forward")
    assert check.verdict == FAILED
    assert check.summary["profitable_share"] == 0.0
    chosen = list(check.table["day"])
    assert all(day != (k + 1) % 5 for k, day in enumerate(chosen))
    lost = [
        LOSS * len(weekday_trades(WALK_SESSIONS, day, BOUNDS[k], BOUNDS[k + 1] - 1))
        for k, day in enumerate(chosen)
    ]
    assert list(check.table["net_profit"]) == lost


def test_a_walk_forward_with_no_edge_in_sample_is_not_judged_and_shows_its_folds():
    # Every trade loses 10, so the best of every fold's search loses money in sample: the
    # folds' mean in-sample net profit is below 0 and there is no edge to re-fit.
    data = weekday_bars(WALK_SESSIONS, lambda s: ())
    found = report(data, ["walk_forward"], "net_profit", **FIVE)
    check = found.check("walk_forward")
    assert check.verdict == NOT_JUDGED
    assert check.note.startswith("No edge to judge: the folds' mean in-sample net_profit per year")
    assert check.note.endswith("not above 0, the value of a run that makes nothing.")
    assert len(check.table) == 5
    assert found.tearsheet().count("class='fold-in'") == 5


class LateWeekday(Weekday):
    """`Weekday` that trades only from session `LATE_SESSION` of the walk-forward runs."""

    def on_bar(self, bar):
        if bar.time.date() >= nse_sessions("2023-01-02", WALK_SESSIONS)[LATE_SESSION]:
            super().on_bar(bar)


#: The first session `LateWeekday` trades on: after the in-sample windows of folds 1 to 3
#: (which end at sessions 126, 181 and 235) and inside those of folds 4 and 5.
LATE_SESSION = 240


def test_a_fold_that_chose_nothing_counts_as_a_fold_that_made_no_money():
    # Folds 1 to 3 have no trade in sample, so no defined expectancy, and choose nothing;
    # folds 4 and 5 choose Monday and win out of sample. Two of the five folds made money.
    data = weekday_bars(WALK_SESSIONS, lambda s: (0,))
    result = bt.walk_forward(
        LateWeekday, data=data, **FREE, objective="expectancy", workers=1, progress=quiet, **FIVE
    )
    assert list(result.folds["day"].isna()) == [True, True, True, False, False]
    assert list(result.folds["trades"].isna()) == [True, True, True, False, False]
    assert result.profitable_share == 2 / 5


def test_the_fold_table_reads_n_a_for_a_fold_that_chose_nothing():
    # Folds 1 to 3 choose nothing, so they have no out-of-sample run: no P&L, drawdown,
    # trades or win rate to show.
    data = weekday_bars(WALK_SESSIONS, lambda s: (0,))
    result = bt.backtest(LateWeekday, data=data, params={"day": 0}, **FREE)
    found = result.robustness(
        checks=["walk_forward"], workers=1, objective="expectancy", progress=quiet, **FIVE
    )
    (table,) = found._blocks["walk_forward"][2]
    rows = {row[0]: row[2:] for row in table.rows}
    assert rows["Fold 1"] == ("none", "n/a", "n/a", "n/a", "n/a", "n/a")
    assert rows["Fold 4"][0] == "day 0"
    assert "id='walk-forward'" in found.tearsheet()


def test_anchored_folds_grow_their_in_sample_window(planted_walk):
    data, _ = planted_walk
    result = bt.walk_forward(
        Weekday, data=data, **FREE, anchored=True, workers=1, progress=quiet, **FIVE
    )
    assert result.folds["in_sample_start"].nunique() == 1
    assert list(result.folds["day"]) == [0] * 5


def test_without_ranges_the_walk_forward_is_not_judged():
    result = bt.backtest(Flip, data=bars(steps_for([3.0, -1.0], 40)), **FREE)
    check = result.robustness(checks=["walk_forward"], workers=1, progress=quiet)
    assert check.check("walk_forward").note.startswith("Nothing to re-fit")


def test_bt_walk_forward_needs_something_to_search():
    data = weekday_bars(100, lambda s: (0,))
    with pytest.raises(bt.errors.ConfigError, match="nothing to search"):
        bt.walk_forward(Weekday, data=data, **FREE, workers=1)


def test_bt_walk_forward_refuses_too_few_sessions_for_its_folds():
    data = weekday_bars(12, lambda s: (0,))
    with pytest.raises(bt.errors.ConfigError, match="give fewer folds or a longer run"):
        bt.walk_forward(Weekday, data=data, **FREE, workers=1, progress=quiet, **FIVE)


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        ({"folds": 0}, "folds takes a whole number"),
        ({"in_sample": 1.0}, "in_sample takes a fraction above 0 and below 1"),
        ({"anchored": "yes"}, "anchored takes True or False"),
        ({"colour": "blue"}, "neither run arguments of bt.backtest nor parameters"),
    ],
)
def test_bt_walk_forward_refuses_unusable_settings(setting, message):
    data = weekday_bars(100, lambda s: (0,))
    with pytest.raises(bt.errors.ConfigError, match=message):
        bt.walk_forward(Weekday, data=data, **FREE, workers=1, **FIVE, **setting)


class LockedClient(FakeClient):
    """A fake OpenAlgo client that cannot be pickled, as the SDK's own client cannot: it
    holds a lock, as the SDK's pooled HTTP client does."""

    def __init__(self, **options):
        super().__init__(**options)
        self._lock = threading.RLock()


@pytest.fixture
def served_walk(planted_walk, tmp_path, monkeypatch):
    """The planted walk's bars served by a `LockedClient`, and `bt.walk_forward`'s settings
    that read them through it."""
    monkeypatch.setenv("OPENALGO_BT_CACHE", str(tmp_path))
    data, _ = planted_walk
    client = LockedClient(api={("SBIN", "NSE", "D"): historify_rows(data)})
    span = {"start": data.index[0].date(), "end": data.index[-1].date(), "source": "api"}
    return client, {**FREE, **span, **FIVE}


@pytest.mark.worker_pool
def test_a_walk_forward_through_a_client_runs_on_worker_processes(served_walk):
    # The client stays in this process: the bars are read through it once, and the
    # workers get the bars, so two workers run the folds this process runs.
    client, settings = served_walk
    here = bt.walk_forward(Weekday, client, workers=1, progress=quiet, **settings)
    pooled = bt.walk_forward(Weekday, client, workers=2, progress=quiet, **settings)
    assert "in 2 worker processes." in pooled.plan
    pd.testing.assert_frame_equal(pooled.folds, here.folds)
    assert list(here.folds["day"]) == [0] * 5


def test_a_walk_forward_whose_cost_model_cannot_be_sent_runs_here_and_says_why(planted_walk):
    data, _ = planted_walk
    settings = {**FREE, "costs": bt.costs.flat(20)}
    result = bt.walk_forward(Weekday, data=data, **settings, workers=2, progress=quiet, **FIVE)
    assert (
        "in this process (the cost model given as costs= cannot be sent to worker processes)."
        in result.plan
    )


def test_the_folds_count_closed_trades_in_whole_numbers(planted_walk):
    # Every fold trades Monday out of sample: its closed trades are the Mondays it buys on.
    _, check = planted_walk
    trades = check.table["trades"]
    assert trades.dtype == "Int64"
    assert trades.tolist() == [
        len(weekday_trades(WALK_SESSIONS, 0, BOUNDS[k], BOUNDS[k + 1] - 1)) for k in range(5)
    ]


def chosen_frame(values: tuple, chosen: list) -> pd.DataFrame:
    """The folds table of a hand-built walk-forward over one parameter `level` taking
    `values`, its folds choosing the positions `chosen` (None for nothing), none traded."""
    grid = Grid(("level",), tuple({"level": value} for value in values), {"level": values[0]}, True)
    windows = folds(100, len(chosen), 0.7, False)
    outcomes = tuple(
        FoldOutcome(fold, np.array([]), position, None, None)
        for fold, position in zip(windows, chosen, strict=True)
    )
    dates = nse_sessions("2023-01-02", 100)
    return fold_frame(WalkOutcome(outcomes, grid, "net_profit", None), dates)


# (the parameter's values, the folds' choices, whether its column is whole numbers, the
# column with None where it is missing): whole numbers are a whole-number column, as the
# trades are, missing for a fold that chose nothing; other values keep their own type.
PARAMETER_COLUMNS = [
    ((0, 1, 2), [None, 1, 2], True, [None, 1, 2]),
    ((0, 1, 2), [0, 2, 1], True, [0, 2, 1]),
    ((0.5, 1.5), [None, 1], False, [None, 1.5]),
    (("fast", "slow"), [None, 0], False, [None, "fast"]),
    ((True, False), [1, None], False, [False, None]),
]


@pytest.mark.parametrize(("values", "chosen", "whole", "column"), PARAMETER_COLUMNS)
def test_the_folds_show_whole_number_parameters_as_whole_numbers(values, chosen, whole, column):
    level = chosen_frame(values, chosen)["level"]
    assert (str(level.dtype) == "Int64") == whole
    assert [None if pd.isna(value) else value for value in level] == column


def test_a_runner_of_other_bars_shares_its_parents_worker_processes():
    from openalgo.backtest._robust._reruns import Reruns, Workers

    parent = Reruns(None, Workers(2, ""))
    child = parent.sibling(None)
    assert child._pool is parent._pool and child.setup is None


def test_bt_walk_forward_states_its_plan_before_it_runs():
    lines = []
    data = weekday_bars(100, lambda s: (0,))
    result = bt.walk_forward(Weekday, data=data, **FREE, workers=1, progress=lines.append, **FIVE)
    assert lines == [result.plan]
    assert result.plan.startswith(
        "bt.walk_forward: 5 folds of Weekday, rolling, each 70% in sample, searching 5 "
        "combinations: 30 reruns (25 in sample, 5 out of sample), each an ordinary backtest "
        "of the same kernel, in this process."
    )


def test_bt_walk_forward_suggests_the_condition_form_for_many_event_reruns():
    from openalgo.backtest._conditions import CONDITIONS_PATH, EVENT_PATH
    from openalgo.backtest._robust._reruns import Workers
    from openalgo.backtest._robust._walk_forward import _plan

    made = folds(WALK_SESSIONS, 5, 0.7, anchored=False)
    shape, here = {"anchored": False, "in_sample": 0.7}, Workers(1, "")
    # 5 folds of 200 combinations rerun 5 x 201 = 1,005 times, past 1,000; of 199, 1,000.
    many = _plan("Weekday", made, shape, 200, here, (0.01, EVENT_PATH))
    assert many.endswith("which runs without a Python call per bar.")
    assert "condition form" not in _plan("Weekday", made, shape, 199, here, (0.01, EVENT_PATH))
    assert "condition form" not in _plan("Weekday", made, shape, 200, here, (0.01, CONDITIONS_PATH))


# ----- 2.17 noise-robust optimization ------------------------------------------------------


def test_the_robust_choice_of_a_lone_spike_is_shown_off_a_plateau_without_a_verdict(wednesday):
    # On every noisy copy Wednesday still wins and its neighbours still lose: the
    # neighbourhood rule does not hold for the robust choice. Design 17 makes the check
    # informational, so the rule's outcome is its figure, not a failure counted.
    found = report(wednesday, ["noise_optimization"], noise_optimization={"samples": 3}, **FIVE)
    check = found.check("noise_optimization")
    robust = check.table[check.table["robust"]]
    assert list(robust["day"]) == [2]
    assert check.verdict == INFORMATIONAL
    assert check.summary["profitable_share"] == 0.0 and check.summary["plateau"] == 0.0
    assert "by the neighbourhood rule a lone spike, not a plateau" in check.figure
    assert list(found.verdicts["verdict"]) == [INFORMATIONAL]


def test_the_robust_choice_of_a_plateau_is_shown_on_a_plateau_without_a_verdict(every_day):
    check = report(
        every_day, ["noise_optimization"], noise_optimization={"samples": 3}, **FIVE
    ).check("noise_optimization")
    assert check.verdict == INFORMATIONAL
    assert check.summary["profitable_share"] == 1.0 and check.summary["plateau"] == 1.0
    assert "by the neighbourhood rule on a plateau" in check.figure


def test_without_ranges_the_noise_robust_optimization_is_not_judged(wednesday):
    check = report(wednesday, ["noise_optimization"], params={"day": 2})
    assert check.check("noise_optimization").note.startswith("No parameter range to search")


# ----- the report ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def searched(wednesday):
    return report(
        wednesday,
        ["out_of_sample", "walk_forward", "parameters", "noise_optimization"],
        params={"day": 2},
        noise_optimization={"samples": 2},
        walk_forward={"folds": 2},
        **FIVE,
    )


def test_the_plan_counts_the_searches_and_the_runs_they_share(wednesday):
    lines = []
    report(
        wednesday,
        ["parameters", "noise_optimization"],
        params={"day": 2},
        noise_optimization={"samples": 2},
        progress=lines.append,
        **FIVE,
    )
    # The neighbourhood runs the 5 days, the noise-robust optimization each on 2 noisy
    # copies and once on the run's bars, the same 5 runs as the neighbourhood's.
    assert (
        "the searching checks rerun the backtest 15 times or fewer on the run's own bars "
        "over parameter combinations or windows of the run (parameters 5, "
        "noise_optimization 15), each an ordinary backtest of the same kernel. The "
        "searching checks share 5 of their runs, the same combinations on the same bars"
    ) in lines[0]


def test_the_page_draws_each_searching_check(searched):
    page = searched.tearsheet()
    for title in ("Two parts in time", "Equity of the two parts", "In sample against out of"):
        assert title in page, title
    assert page.count("class='fold-in'") == 2 and page.count("class='fold-out'") == 2
    assert "aria-label='Objective by parameter'" in page
    assert "aria-label='The same grid as a surface'" in page
    assert "aria-label='Noisy score'" in page and "aria-label='Plain objective'" in page
    assert "class='bars spark'" in page


def test_the_same_seed_gives_the_same_searching_report(wednesday, searched):
    again = report(
        wednesday,
        ["out_of_sample", "walk_forward", "parameters", "noise_optimization"],
        params={"day": 2},
        noise_optimization={"samples": 2},
        walk_forward={"folds": 2},
        **FIVE,
    )
    assert again.tearsheet() == searched.tearsheet()
