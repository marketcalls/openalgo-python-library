"""The searching checks (design 17, sections 2.1, 2.3, 2.16 and 2.17) and `bt.walk_forward`
end to end, on own daily bars with no server, in this process.

`robust_runs.Weekday` buys 10 shares at the close of a session on its weekday `day` and
sells two sessions on; `weekday_bars` plants the move each trade makes: 3 rupees a share
when its weekday is favoured, -1 when it is not, so a trade makes +30 or -10 rupees and
`weekday_trades` counts a run's trades by hand. A planted edge favours one weekday over
the whole run; a rotation changes the favoured weekday at each walk-forward boundary,
so what one window favours the next does not.
"""

import pandas as pd
import pytest
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

import openalgo.backtest as bt
from openalgo.backtest._robust._verdict import FAILED, INFORMATIONAL, NOT_JUDGED, PASSED
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


def test_a_strategy_without_a_number_parameter_is_not_judged():
    result = bt.backtest(Flip, data=bars(steps_for([3.0, -1.0], 40)), **FREE)
    check = result.robustness(checks=["parameters"], workers=1, progress=quiet).check("parameters")
    assert check.verdict == NOT_JUDGED and check.note.startswith("No parameter to vary")


def test_parameters_off_the_grid_given_are_not_judged(wednesday):
    check = report(wednesday, ["parameters"], params={"day": 2}, day=(0, 1, 3, 4))
    found = check.check("parameters")
    assert found.verdict == NOT_JUDGED
    assert found.note.startswith("The run's parameters (day 2) are not a combination")


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
    assert result.profitable_share == 2 / 5


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


@pytest.mark.worker_pool
def test_a_walk_forward_in_two_worker_processes_is_the_one_in_this_process(planted_walk):
    data, _ = planted_walk
    here = bt.walk_forward(Weekday, data=data, **FREE, workers=1, progress=quiet, **FIVE)
    pooled = bt.walk_forward(Weekday, data=data, **FREE, workers=2, progress=quiet, **FIVE)
    assert "in 2 worker processes" in pooled.plan
    pd.testing.assert_frame_equal(pooled.folds, here.folds)


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
