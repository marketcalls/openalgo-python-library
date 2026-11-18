"""The parts of the searching checks that need no backtest (design 17, sections 2.1, 2.3
and 2.16): the parameter grid and the run's neighbourhood, a combination's neighbours,
the windows of the run's sessions, and the parameter-neighbourhood rule's figures, each
worked by hand."""

import math

import numpy as np
import pytest

import openalgo.backtest as bt
from openalgo.backtest._robust._grid import Grid, around, cube, search_grid
from openalgo.backtest._robust._neighbourhood import plateau
from openalgo.backtest._robust._windows import Window, folds, split, window_arguments
from openalgo.backtest.errors import ConfigError


class Crossing(bt.Strategy):
    """Two whole-number parameters, a fraction, a flag and a name."""

    symbol, exchange, interval = "SBIN", "NSE", "D"
    fast, slow, band, short, label = 10, 40, 2.0, True, "x"


class Declared(bt.Strategy):
    """A parameter declared with its sweep values, and one without."""

    symbol, exchange, interval = "SBIN", "NSE", "D"
    fast = bt.param(10, (5, 10, 15))
    slow = 40


# ----- the neighbourhood of the run's own parameters ----------------------------------


@pytest.mark.parametrize(
    ("value", "step", "expected"),
    [
        # A whole number moves by 10 percent of itself, rounded half up, at least a unit.
        (14, 0.10, (13, 14, 15)),
        (50, 0.10, (45, 50, 55)),
        (25, 0.10, (22, 25, 28)),
        (3, 0.10, (2, 3, 4)),
        # A count or a period stays at 1 or more, and a zero moves up only.
        (1, 0.10, (1, 2)),
        (0, 0.10, (0, 1)),
        (-5, 0.10, (-6, -5, -4)),
        # A fraction moves by 10 percent of itself; a zero fraction stays.
        (2.0, 0.10, (1.8, 2.0, 2.2)),
        (0.0, 0.10, (0.0,)),
        (200, 0.25, (150, 200, 250)),
    ],
)
def test_a_parameter_moves_by_its_step_at_least_a_unit_for_a_whole_number(value, step, expected):
    assert around(value, step) == pytest.approx(expected)


def test_the_neighbourhood_moves_every_number_parameter_and_keeps_the_others():
    grid = cube(Crossing, {"slow": 30}, 0.10)
    assert grid.names == ("fast", "slow", "band")
    assert len(grid.combos) == 27
    assert grid.axis("slow") == (27, 30, 33)
    assert all(combo["short"] is True and combo["label"] == "x" for combo in grid.combos)
    assert grid.swept(grid.chosen()) == {"fast": 10, "slow": 30, "band": 2.0}


def test_a_strategy_without_a_number_parameter_has_no_neighbourhood():
    class Flags(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "D"
        short = True

    assert cube(Flags, None, 0.10) is None


# ----- the grid of the ranges -----------------------------------------------------------


def test_the_ranges_given_make_the_grid_in_grid_order():
    grid = search_grid(Crossing, {"fast": (5, 10), "slow": (20, 40)}, None)
    assert grid.names == ("fast", "slow")
    assert [grid.swept(k) for k in range(4)] == [
        {"fast": 5, "slow": 20},
        {"fast": 5, "slow": 40},
        {"fast": 10, "slow": 20},
        {"fast": 10, "slow": 40},
    ]
    assert grid.given and grid.chosen() == 3


def test_a_parameter_chosen_in_params_and_given_a_range_is_searched_over_the_range():
    grid = search_grid(Crossing, {"fast": (5, 10, 15)}, {"fast": 15, "slow": 30})
    assert grid.names == ("fast",)
    assert grid.chosen() == 2
    # The other parameter stays as params= chose it.
    assert all(combo["slow"] == 30 for combo in grid.combos)


def test_without_ranges_the_declared_ones_make_the_grid():
    grid = search_grid(Declared, {}, None)
    assert grid.names == ("fast",) and grid.axis("fast") == (5, 10, 15)


def test_a_declared_range_fixed_by_params_leaves_nothing_to_search():
    assert search_grid(Declared, {}, {"fast": 10}) is None
    assert search_grid(Crossing, {}, None) is None


def test_a_combination_changes_only_the_parameters_that_differ_from_the_runs():
    grid = search_grid(Crossing, {"fast": (5, 10), "slow": (40,)}, None)
    assert grid.changes(0) == (("fast", 5),)
    assert grid.changes(1) == ()


def _square(size: int) -> Grid:
    combos = [{"x": x, "y": y} for x in range(size) for y in range(size)]
    return Grid(("x", "y"), tuple(combos), {"x": 0, "y": 0}, True)


@pytest.mark.parametrize(
    ("place", "expected"),
    [
        # The centre of a 3 by 3 grid has 8 neighbours; a corner 3; an edge 5.
        ((1, 1), [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]),
        ((0, 0), [(0, 1), (1, 0), (1, 1)]),
        ((0, 1), [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)]),
    ],
)
def test_a_neighbour_is_at_most_one_step_away_in_every_parameter(place, expected):
    grid = _square(3)
    position = 3 * place[0] + place[1]
    found = [(grid.combos[k]["x"], grid.combos[k]["y"]) for k in grid.neighbours(position)]
    assert found == expected


# ----- the windows of the run's sessions ----------------------------------------------


def test_out_of_sample_keeps_the_last_part_of_the_sessions():
    # 70 percent of 10 sessions: the first 7 in sample, the last 3 out.
    assert split(10, 0.7) == (Window(0, 6), Window(7, 9))


def test_out_of_sample_needs_two_sessions_in_each_part():
    with pytest.raises(ConfigError, match="each part needs 2"):
        split(4, 0.9)


#: 100 sessions in 5 folds at 70 percent: r = 0.7 / 0.3, b_j = round(100 (r + j) / (r + 5))
#: = 32, 45, 59, 73, 86 and 100 (31.8, 45.5, 59.1, 72.7, 86.4 rounded).
BOUNDS = (32, 45, 59, 73, 86, 100)


def test_rolling_folds_keep_an_in_sample_window_of_fixed_length_before_each_out_of_sample():
    made = folds(100, 5, 0.7, anchored=False)
    assert [(f.out_of_sample.first, f.out_of_sample.last) for f in made] == [
        (BOUNDS[j], BOUNDS[j + 1] - 1) for j in range(5)
    ]
    assert [(f.in_sample.first, f.in_sample.last) for f in made] == [
        (BOUNDS[j] - 32, BOUNDS[j] - 1) for j in range(5)
    ]


def test_anchored_folds_keep_every_session_before_in_sample():
    made = folds(100, 5, 0.7, anchored=True)
    assert [(f.in_sample.first, f.in_sample.last) for f in made] == [
        (0, BOUNDS[j] - 1) for j in range(5)
    ]
    assert [f.out_of_sample for f in made] == [
        f.out_of_sample for f in folds(100, 5, 0.7, anchored=False)
    ]


def test_folds_with_a_window_under_two_sessions_are_refused():
    with pytest.raises(ConfigError, match="give fewer folds or a longer run"):
        folds(12, 5, 0.7, anchored=False)


def test_a_window_reaching_the_runs_ends_keeps_the_runs_own_bounds():
    dates = ["d0", "d1", "d2", "d3"]
    assert window_arguments(Window(0, 1), dates, "first", "last") == (
        ("start", "first"),
        ("end", "d1"),
    )
    assert window_arguments(Window(2, 3), dates, "first", "last") == (
        ("start", "d2"),
        ("end", "last"),
    )


# ----- the parameter-neighbourhood rule --------------------------------------------------


def test_a_plateau_keeps_its_figures_among_its_neighbours():
    # Chosen 10; neighbours 6, 8, 9 and 12 (sorted 6, 8, 9, 12): the median 8.5 and the
    # 25th percentile 7.5 (linear between 6 and 8, then 8 and 9); from break-even 0 they
    # keep 85 and 75 percent. Of every other setting (the neighbours and 20, -4), 4 of 6
    # are below 10.
    found = plateau(
        10.0,
        np.array([6.0, 8.0, 9.0, 12.0]),
        np.array([True, True, True, True]),
        np.array([6.0, 8.0, 9.0, 12.0, 20.0, -4.0]),
        0.0,
        0.5,
    )
    assert found.median == pytest.approx(8.5) and found.low == pytest.approx(7.5)
    assert found.median_ratio == pytest.approx(0.85) and found.low_ratio == pytest.approx(0.75)
    assert found.percentile == pytest.approx(4 / 6) and found.profitable_share == 1.0
    assert found.passed


def test_a_spike_fails_when_its_low_neighbours_lose():
    # Neighbours -2, -1 and 9: the 25th percentile -1.5 is below break-even.
    found = plateau(
        10.0,
        np.array([-2.0, -1.0, 9.0]),
        np.array([False, False, True]),
        np.array([-2.0, -1.0, 9.0]),
        0.0,
        0.5,
    )
    assert found.low == pytest.approx(-1.5) and not found.passed
    assert found.profitable_share == pytest.approx(1 / 3)


def test_a_median_that_keeps_too_little_fails_though_every_neighbour_made_money():
    # Profit factor (break-even 1) chosen 3; neighbours 1.2, 1.4, 1.6: the median 1.4 keeps
    # (1.4 - 1) / (3 - 1) = 20 percent of the chosen.
    found = plateau(3.0, np.array([1.2, 1.4, 1.6]), np.ones(3, bool), np.array([1.2]), 1.0, 0.5)
    assert found.median_ratio == pytest.approx(0.2) and not found.passed


def test_neighbours_with_no_defined_objective_leave_the_rule_unapplied():
    assert plateau(1.0, np.array([math.nan]), np.zeros(1, bool), np.array([]), 0.0, 0.5) is None
