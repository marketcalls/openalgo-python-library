"""The overfitting figures `bt.optimize` gains from the grid it already ran (design 17,
section 5): the neighbourhood of the best with a spike warning, the deflated Sharpe ratio
and the probability of backtest overfitting, and the heatmap page that shows them; on
own daily bars with no server, in this process.

`robust_runs.Weekday` buys 10 shares at the close of a session on its weekday `day`;
`weekday_bars` plants +30 rupees on a trade of a favoured weekday and -10 on any other
(`test_robust_search_checks` describes them). The sweep is days 0 to 4.
"""

import itertools
import math
import tracemalloc
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from robust_runs import (
    FREE,
    DailyCross,
    Weekday,
    random_walk,
    weekday_bars,
    weekday_trades,
)
from test_report_tearsheet import FORBIDDEN_NAME_HASHES, Structure, section_ids

import openalgo.backtest as bt
from openalgo.backtest._robust._verdict import FAILED, NOT_JUDGED, PASSED
from openalgo.backtest._sweep_returns import ReturnsMatrix
from openalgo.backtest.errors import ConfigError

SESSIONS = 300
WIN, LOSS = 30.0, -10.0


def sweep(data, **settings):
    return bt.optimize(Weekday, data=data, **FREE, day=range(5), workers=1, **settings)


@pytest.fixture(scope="module")
def spike():
    """Wednesday trades win, every other day's lose."""
    return sweep(weekday_bars(SESSIONS, lambda s: (2,)))


@pytest.fixture(scope="module")
def plateau():
    """Every trade wins."""
    return sweep(weekday_bars(SESSIONS, lambda s: range(5)))


def test_the_best_of_a_lone_spike_fails_its_neighbourhood_and_warns(spike):
    # Wednesday makes 30 a trade; its neighbours Tuesday and Thursday lose 10 a trade.
    n1, n2, n3 = (len(weekday_trades(SESSIONS, day)) for day in (1, 2, 3))
    assert spike.best == {"day": 2}
    found = spike.neighbourhood
    assert found.verdict == FAILED
    assert found.summary["median_ratio"] == pytest.approx(LOSS * (n1 + n3) / 2 / (WIN * n2))
    assert list(found.table["neighbour"]) == [False, True, False, True, False]
    assert len(spike.warnings) == 1
    assert spike.warnings[0].startswith(
        "The best combination (day 2) looks like a lone spike, not a plateau"
    )


def test_the_best_of_a_plateau_passes_and_does_not_warn(plateau):
    assert plateau.neighbourhood.verdict == PASSED
    assert plateau.neighbourhood.summary["profitable_share"] == 1.0
    assert plateau.warnings == []


def test_the_figures_change_no_column_of_the_table(spike):
    assert list(spike.table.columns) == [
        "day",
        "net_profit",
        "total_return",
        "max_drawdown",
        "sharpe",
        "trades",
        "objective",
    ]


def test_the_sweep_keeps_its_session_returns_as_one_array(spike):
    # 8 bytes a session and combination, with no session index kept per combination.
    returns = spike._returns
    assert isinstance(returns, np.ndarray) and returns.shape == (SESSIONS, 5)
    assert returns.nbytes == 8 * SESSIONS * 5


def returns_array(*combinations: tuple[list[str], list[float]]) -> list[list[float]]:
    """The sweep's returns array of combinations given as (session dates, returns), in
    grid order."""
    built = ReturnsMatrix(len(combinations))
    for days, values in combinations:
        built.add(pd.Series(values, index=pd.DatetimeIndex(days)))
    return built.array().tolist()


JAN = ["2025-01-01", "2025-01-02", "2025-01-03"]


@pytest.mark.parametrize(
    ("combinations", "array"),
    [
        # The same sessions stand side by side, a missing number kept as it is.
        (
            [(JAN, [0.01, 0.02, 0.03]), (JAN, [0.0, -0.01, math.nan])],
            [[0.01, 0.0], [0.02, -0.01], [0.03, math.nan]],
        ),
        # The second has no return on 2 January: it counts 0 there, as its missing one.
        (
            [(JAN, [0.01, 0.02, 0.03]), (JAN[::2], [math.nan, -0.01])],
            [[0.01, 0.0], [0.02, 0.0], [0.03, -0.01]],
        ),
        # The second brings 2 January, a session the first lacks: the array grows to it
        # in date order, the first counting 0 there.
        (
            [(JAN[::2], [0.01, 0.03]), (JAN[1:], [0.02, -0.01])],
            [[0.01, 0.0], [0.0, 0.02], [0.03, -0.01]],
        ),
        # Grown after a combination on the first's sessions, which keeps its column.
        (
            [(JAN[:2], [0.01, 0.02]), (JAN[:2], [0.04, 0.05]), (JAN, [0.1, 0.2, 0.3])],
            [[0.01, 0.04, 0.1], [0.02, 0.05, 0.2], [0.0, 0.0, 0.3]],
        ),
    ],
    ids=["the same sessions", "fewer sessions", "a new session", "a new session later"],
)
def test_the_sweeps_returns_array_holds_each_combination_on_its_sessions(combinations, array):
    np.testing.assert_array_equal(returns_array(*combinations), array)


def test_a_sweep_with_nothing_measured_has_an_empty_returns_array():
    assert ReturnsMatrix(0).array().shape == (0, 0)


#: The size of the array the peak tests build: 200 combinations of 1,000 sessions, 1.6 MB.
PEAK_COMBINATIONS, PEAK_SESSIONS = 200, 1_000
#: A trading day of the peak tests' first session.
PEAK_START = "2022-01-03"


def built_peak(sessions_of: Callable[[int], pd.DatetimeIndex]) -> float:
    """The most traced memory building the returns array takes, from the first
    combination measured to the array made, over the array's size. Each combination's
    returns (on `sessions_of(position)`) are made as its run would end, a Series with its
    own session index, added, and let go, as a sweep does."""
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        built = ReturnsMatrix(PEAK_COMBINATIONS)
        for position in range(PEAK_COMBINATIONS):
            sessions = sessions_of(position)
            built.add(pd.Series(np.full(len(sessions), 0.001 * position), index=sessions))
        array = built.array()
        peak = tracemalloc.get_traced_memory()[1] - before
    finally:
        tracemalloc.stop()
    return peak / array.nbytes


def every_session(position: int) -> pd.DatetimeIndex:
    return pd.bdate_range(PEAK_START, periods=PEAK_SESSIONS)


def all_but_the_first_ten_after_the_first(position: int) -> pd.DatetimeIndex:
    """Every session for the first combination; the others start trading ten sessions
    later."""
    return every_session(position)[0 if position == 0 else 10 :]


def ten_more_after_the_first(position: int) -> pd.DatetimeIndex:
    """The first combination starts trading ten sessions later than the others."""
    return every_session(position)[10 if position == 0 else 0 :]


@pytest.mark.parametrize(
    "sessions_of",
    [every_session, all_but_the_first_ten_after_the_first, ten_more_after_the_first],
    ids=["the same sessions", "fewer sessions", "new sessions"],
)
def test_building_the_returns_array_holds_little_more_than_the_array(sessions_of):
    # Each combination's returns are copied into the array as they come back and let go,
    # so the build never holds them all beside it (which takes twice the array): at most
    # the array, one combination's Series and the session index, 20 percent over at most.
    # The second combination's new sessions grow the array while one column is filled,
    # and only that column is held twice.
    assert built_peak(sessions_of) < 1.2


def test_a_best_that_dominates_every_block_never_overfits(spike):
    # Wednesday's session returns are above every other day's in every block of the
    # cross-validation, so it is best in sample and out of sample in all C(16, 8) splits.
    figures = spike.overfitting
    assert spike.overfitting_probability == 0.0
    assert figures["blocks"] == 16 and figures["splits"] == 12_870
    assert figures["trials"] == 5 and figures["sessions"] == SESSIONS


def test_the_deflated_sharpe_ratio_is_a_probability(spike):
    figures = spike.overfitting
    assert 0.0 <= spike.deflated_sharpe <= 1.0
    assert figures["deflated_sharpe"] == spike.deflated_sharpe
    assert figures["expected_best_sharpe"] > 0.0


#: The sessions of a random walk a crossover's settings are fitted to.
WALK_SESSIONS = 1_000


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_best_crossover_fitted_to_a_random_walk_does_not_stand_out_from_luck(seed):
    # No setting has an edge on a random walk, so the best of DailyCross's 81 settings is
    # only the luckiest: its deflated Sharpe ratio stays below 0.95, and it does not sit
    # on a plateau of settings that made money (no edge to judge, or a lone spike).
    result = bt.optimize(
        DailyCross,
        data=random_walk(WALK_SESSIONS, seed),
        **FREE,
        objective="sharpe",
        workers=1,
        progress=lambda line: None,
    )
    assert result.deflated_sharpe < 0.95
    assert result.neighbourhood.verdict in (FAILED, NOT_JUDGED)


def test_a_sweep_that_minimises_leaves_the_neighbourhood_not_judged():
    result = sweep(weekday_bars(SESSIONS, lambda s: (2,)), maximize=False)
    assert result.neighbourhood.verdict == NOT_JUDGED
    assert "better higher" in result.neighbourhood.note
    assert result.warnings == []


def test_an_objective_without_a_break_even_leaves_the_neighbourhood_not_judged():
    result = sweep(weekday_bars(SESSIONS, lambda s: (2,)), objective="win_rate")
    assert result.neighbourhood.verdict == NOT_JUDGED
    assert result.neighbourhood.note.startswith("The objective win_rate has no value")


def net(result) -> float:
    """An objective function: the run's net profit."""
    return result.metrics["net_profit"]


def test_an_objective_function_is_named_where_the_neighbourhood_is_not_judged():
    # A function's value for a run that makes nothing is not known, so the rule has no
    # break-even to measure from; the note names the function once.
    result = sweep(weekday_bars(SESSIONS, lambda s: (2,)), objective=net)
    assert result.neighbourhood.verdict == NOT_JUDGED
    assert result.neighbourhood.note == (
        "The objective net has no value that says a run made nothing."
    )


def test_a_grid_of_one_combination_has_nothing_to_deflate():
    result = bt.optimize(
        Weekday, data=weekday_bars(SESSIONS, lambda s: (2,)), **FREE, day=(2,), workers=1
    )
    assert math.isnan(result.deflated_sharpe) and math.isnan(result.overfitting_probability)
    assert result.neighbourhood.verdict == NOT_JUDGED


# ----- the heatmap page ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def page(spike):
    return spike.tearsheet()


def test_the_page_is_the_search_the_neighbourhood_the_overfitting_and_the_notes(page):
    assert section_ids(page) == ["search", "neighbourhood", "overfitting", "notes"]


def test_the_page_draws_the_heatmap_and_the_surface_with_the_best_outlined(page):
    assert "aria-label='Objective'" in page and "aria-label='Objective as a surface'" in page
    assert page.count("class='heat-cell cell-marked'") == 1
    assert page.count("class='surf-top cell-marked'") == 1


def test_the_page_states_the_spike_warning_and_the_figures(spike, page):
    assert "Warning: The best combination (day 2) looks like a lone spike" in page
    assert ">Deflated Sharpe<" in page and ">Overfitting<" in page


def test_the_page_reads_a_low_probability_of_overfitting_as_no_proof_that_settings_hold(page):
    # A combination that held a lasting exposure to one history's trend also ranks well
    # in every block of it, so a probability near 0 cannot show the settings hold; near
    # 0.5 says only that the search did not pick settings that held across the blocks.
    assert "it does not show the settings hold" in page
    assert "the search did not pick settings that held up across the blocks" in page
    assert "picks noise" not in page and "picks settings that hold" not in page


def test_the_probability_of_overfitting_says_what_a_low_figure_cannot_show():
    from openalgo.backtest._optimize_result import OptimizeResult

    words = " ".join(OptimizeResult.overfitting_probability.__doc__.split())
    assert "near 0 does not show the settings hold" in words


def test_the_overfitting_histogram_counts_splits_not_runs(spike, page):
    # Each value is the logit of one of the C(16, 8) = 12,870 ways of splitting the blocks.
    assert spike.overfitting["splits_counted"] == 12_870
    assert "Bars: how many of the 12,870 splits fall in each bin." in page
    assert "12,870 runs" not in page


class TwoWeekday(Weekday):
    """`Weekday` with a second parameter it does not read, so a sweep has two."""

    lots = 1


def test_the_best_tile_shows_each_parameter_on_its_own_line():
    # Wednesday wins; both lots give the same run, so the first, 1, is the best.
    result = bt.optimize(
        TwoWeekday,
        data=weekday_bars(SESSIONS, lambda s: (2,)),
        **FREE,
        day=(1, 2),
        lots=(1, 2),
        workers=1,
        progress=lambda line: None,
    )
    assert result.best == {"day": 2, "lots": 1}
    shown = "<div class='stat-value nums wrap' title='day 2\nlots 1'>day 2\nlots 1</div>"
    assert f">Best</div>{shown}" in result.tearsheet()


def test_a_wrapped_tile_keeps_its_lines_and_breaks_a_long_one():
    from test_report_tearsheet import css_rule

    from openalgo.backtest.report._dashboard.style import stylesheet

    rule = css_rule(stylesheet(), ".stat-value.wrap")
    assert rule["white-space"] == "pre-line" and rule["overflow-wrap"] == "anywhere"


def test_the_page_is_one_self_contained_well_formed_page(page):
    assert page.startswith("<!DOCTYPE html>") and page.isascii()
    lower = page.lower()
    for forbidden in ("http://", "https://", "@import", "<link", "<iframe", "<img", "src="):
        assert forbidden not in lower, forbidden
    structure = Structure(page)
    assert structure.errors == [] and structure.stack == []


def test_the_page_names_no_third_party_product(page):
    import hashlib
    import re

    tokens = set(re.findall(r"[a-z0-9][a-z0-9.\-]*", page.lower()))
    tokens |= {part for token in tokens for part in token.split("-")}
    hashed = {hashlib.sha256(token.encode()).hexdigest()[:16] for token in tokens}
    assert not hashed & FORBIDDEN_NAME_HASHES


def test_the_page_is_written_to_a_file(spike, tmp_path):
    target = tmp_path / "sweep.html"
    assert spike.tearsheet(target) == str(target)
    assert target.read_text(encoding="utf-8") == spike.tearsheet()


def test_two_sweeps_of_one_grid_make_the_same_page(monkeypatch):
    # The plan times the first combination, so two sweeps report other seconds (here a
    # clock that steps 0.25 s, then one that steps 1.75 s, between its readings); the
    # page's notes keep the plan without its timings, so the pages are the same.
    from openalgo.backtest import _optimize

    data = weekday_bars(SESSIONS, lambda s: (2,))
    plans, pages = [], []
    for step in (0.25, 1.75):
        clock = itertools.count(0.0, step)
        monkeypatch.setattr(_optimize.time, "perf_counter", lambda clock=clock: next(clock))
        result = sweep(data)
        plans.append(result.plan)
        pages.append(result.tearsheet())
    assert plans[0] != plans[1]
    assert pages[0] == pages[1]
    assert "took" not in pages[0]
    assert (
        "<p>bt.optimize: 5 combinations of Weekday on the event path (on_bar runs in Python "
        "for every strategy bar), in this process.</p>"
    ) in pages[0]


def test_the_page_refuses_a_parameter_the_sweep_did_not_vary(spike):
    with pytest.raises(ConfigError, match="'fast' is not a swept parameter"):
        spike.tearsheet(x="fast")
