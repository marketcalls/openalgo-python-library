"""The quick overfitting checks of every backtest (design 17, section 4): the four
trade-list checks at 200 runs, in the tearsheet's "Overfitting checks" section, in one
line of `result.summary()` and in `result.warnings` when one fails; "too few trades to
judge" below 30 closed trades; never a change to an existing figure.

The runs are those of `robust_runs`: `Flip` with 60 trades on a planted edge (every
judged quick check passes; the trade reshuffle is informational) or on noise (the
random out of sample, the trade resample and the variance fail).
"""

from types import SimpleNamespace

import pandas as pd
import pytest
from robust_runs import FREE, Flip, bars, edge_run, noise_run, steps_for
from test_report_tearsheet import section_ids

import openalgo.backtest as bt
from openalgo.backtest import report
from openalgo.backtest._robust._quick import quick_checks

FAILED_NOISE = ("random out of sample", "trade resample", "variance")


@pytest.fixture(scope="module")
def edge():
    return edge_run()


@pytest.fixture(scope="module")
def noise():
    return noise_run()


def overfitting_section(page: str) -> str:
    return page[page.index("<section id='overfitting'") : page.index("<section id='notes'")]


@pytest.mark.parametrize("mode", ["full", "basic"])
def test_the_tearsheet_has_an_overfitting_section_after_the_trades(edge, mode):
    ids = section_ids(edge.tearsheet(mode=mode))
    assert ids[ids.index("trades") + 1] == "overfitting"


@pytest.mark.parametrize("mode", ["metrics", "charts"])
def test_the_metrics_and_charts_modes_leave_it_out(edge, mode):
    assert "overfitting" not in section_ids(edge.tearsheet(mode=mode))


def test_the_section_scores_the_four_quick_checks_and_points_to_the_full_set(edge):
    section = overfitting_section(edge.tearsheet())
    assert section.count("<tr>") == 1 + 4
    assert "3 passed, 1 informational." in section
    assert "200 runs each, seed 0" in section
    assert "result.robustness() runs the full set of checks, with charts." in section


def test_the_section_says_too_few_trades_to_judge_below_30_closed_trades():
    result = bt.backtest(Flip, data=bars(steps_for([2.0, -1.0], 10)), **FREE)
    section = overfitting_section(result.tearsheet())
    assert "Too few trades to judge: 10 closed trades; the checks need 30 to judge." in section
    assert "4 not judged." in section


def test_the_summary_has_one_line_for_the_quick_checks(edge):
    lines = edge.summary().splitlines()
    assert (
        lines[-1]
        == "Overfitting  : quick checks: 3 passed, 1 informational; see result.robustness()"
    )


def test_the_summary_names_the_quick_checks_that_failed(noise):
    line = noise.summary().splitlines()[-1]
    assert line == (
        "Overfitting  : quick checks: 3 failed (random_out_of_sample, resample, variance), "
        "1 informational; see result.robustness()"
    )


def test_a_quick_check_that_fails_adds_a_warning(noise):
    added = [line for line in noise.warnings if line.startswith("Overfitting check failed")]
    assert [line.split(":")[0] for line in added] == [
        f"Overfitting check failed, {title}" for title in FAILED_NOISE
    ]
    assert all("result.robustness() runs the full set" in line for line in added)


def test_quick_checks_that_pass_add_no_warning(edge):
    assert not [line for line in edge.warnings if line.startswith("Overfitting")]


def test_the_failed_quick_checks_are_in_the_tearsheet_notes(noise):
    notes = noise.tearsheet()[noise.tearsheet().index("<section id='notes'") :]
    assert notes.count("Overfitting check failed") == 3


def test_a_tearsheet_shows_the_quick_checks_its_run_already_computed(edge):
    given = quick_checks(edge.performance, runs=50)
    page = report.tearsheet(SimpleNamespace(performance=edge.performance, overfitting=given))
    assert "50 runs each, seed 0" in overfitting_section(page)


def test_a_results_tearsheet_shows_its_own_quick_checks(edge):
    section = overfitting_section(edge.tearsheet())
    assert all(result.figure.replace("'", "&#x27;") in section for result in edge._quick.results)


def test_the_quick_checks_change_no_figure_of_the_run():
    untouched = edge_run()
    checked = edge_run()
    checked.summary(), checked.warnings, checked.tearsheet()
    assert pd.Series(dict(checked.metrics)).equals(pd.Series(dict(untouched.metrics)))
    pd.testing.assert_frame_equal(checked.trades, untouched.trades)
    pd.testing.assert_frame_equal(checked.equity, untouched.equity)


def test_the_quick_checks_are_the_first_200_runs_of_the_full_checks(edge):
    full = edge.robustness(checks="quick", progress=lambda line: None)
    for quick in edge._quick.results:
        first = full.check(quick.name).table
        if quick.name == "variance":  # a row per path at each of the two rates
            first = first.groupby("win_rate", sort=False).head(200)
        pd.testing.assert_frame_equal(quick.table, first.iloc[: len(quick.table)])
