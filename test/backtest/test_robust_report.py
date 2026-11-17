"""`bt.robustness`, `Result.robustness` and `RobustnessReport` end to end, on own daily
bars with no server (`robust_runs`): a strategy with a planted edge passes every judged
check, a small edge in much noise fails them, a strategy with no edge is not judged, the
same seed gives the same report, the plan is reported before the checks run, and the
HTML report is the scorecard, a section per check and the Notes in one self-contained
page.

`Flip` trades 60 times. With a planted edge its trades make 30, 20, 40, -10, 30, 20,
-20, 50, 10 and -10 rupees, repeated: 7 of 10 win and the mean is 16. With a small edge
they make +25 and -20 in turn, 150 in all; with no edge, +20 and -20, nothing in all.
"""

import logging
import re

import pandas as pd
import pytest
from robust_runs import FREE, Flip, bars, edge_run, no_edge_run, small_edge_run, steps_for
from test_report_tearsheet import FORBIDDEN_NAME_HASHES, Structure, section_ids

import openalgo.backtest as bt
from openalgo.backtest._robust._verdict import (
    FAILED,
    INFORMATIONAL,
    NOT_BUILT,
    NOT_JUDGED,
    PASSED,
    WORDING_RULE,
)
from openalgo.backtest.errors import BacktestError, ConfigError

THIS_VERSION = ["random_out_of_sample", "reshuffle", "resample", "variance"]
EDGE_DATA = bars(steps_for([3.0, 2.0, 4.0, -1.0, 3.0, 2.0, -2.0, 5.0, 1.0, -1.0], 60))


@pytest.fixture(scope="module")
def edge():
    return edge_run()


@pytest.fixture(scope="module")
def edge_report(edge):
    return edge.robustness(progress=lambda line: None)


@pytest.fixture(scope="module")
def small_edge_report():
    return small_edge_run().robustness(progress=lambda line: None)


@pytest.fixture(scope="module")
def page(edge_report):
    return edge_report.tearsheet()


def test_the_public_names_come_from_openalgo_backtest(edge_report):
    assert {"robustness", "RobustnessReport", "CheckResult"} <= set(bt.__all__)
    assert isinstance(edge_report, bt.RobustnessReport)
    assert isinstance(edge_report.check("resample"), bt.CheckResult)


def test_a_planted_edge_passes_every_judged_check(edge_report):
    verdicts = edge_report.verdicts.set_index("check")["verdict"]
    assert verdicts[THIS_VERSION].tolist() == [PASSED, INFORMATIONAL, PASSED, PASSED]
    assert edge_report.passed


def test_a_small_edge_in_much_noise_fails(small_edge_report):
    verdicts = small_edge_report.verdicts.set_index("check")["verdict"]
    assert verdicts[["random_out_of_sample", "resample", "variance"]].tolist() == [FAILED] * 3
    assert not small_edge_report.passed


def test_a_strategy_with_no_edge_is_not_judged_and_nothing_passed():
    report = no_edge_run().robustness(checks="quick", progress=lambda line: None)
    verdicts = report.verdicts.set_index("check")["verdict"]
    assert verdicts[["random_out_of_sample", "resample", "variance"]].tolist() == [NOT_JUDGED] * 3
    assert not report.passed


def test_the_scorecard_lists_every_check_and_the_ones_not_built(edge_report):
    verdicts = edge_report.verdicts
    assert list(verdicts.columns) == ["check", "question", "figure", "rule", "verdict", "note"]
    assert list(verdicts.index) == list(range(1, 18))
    assert (verdicts["verdict"] == NOT_BUILT).sum() == 13
    noise = verdicts.set_index("check").loc["noise"]
    assert noise["note"].startswith("Comes with robustness unit RB2")


def test_one_check_has_its_table_summary_and_settings(edge_report):
    resample = edge_report.check("resample")
    assert len(resample.table) == 1_000
    assert resample.summary["profitable_share"] == 1.0
    assert resample.settings["runs"] == 1_000 and resample.seed == 11


def test_asking_for_a_check_not_built_names_its_unit(edge_report):
    with pytest.raises(NotImplementedError, match="robustness unit RB2"):
        edge_report.check("noise")


def test_asking_for_a_check_the_report_did_not_run_is_refused(edge):
    report = edge.robustness(checks=["resample"], progress=lambda line: None)
    with pytest.raises(ConfigError, match="did not run variance"):
        report.check("variance")
    assert list(report.verdicts["check"]) == ["resample"]


def test_the_same_seed_gives_the_same_report(edge):
    first = edge.robustness(seed=4, runs=300, progress=lambda line: None)
    again = edge.robustness(seed=4, runs=300, progress=lambda line: None)
    for name in THIS_VERSION:
        pd.testing.assert_frame_equal(first.check(name).table, again.check(name).table)
    assert first.tearsheet() == again.tearsheet()


def test_another_seed_draws_other_runs(edge):
    first = edge.robustness(seed=4, runs=300, progress=lambda line: None)
    other = edge.robustness(seed=5, runs=300, progress=lambda line: None)
    assert not first.check("resample").table.equals(other.check("resample").table)


def test_bt_robustness_runs_the_backtest_once_and_checks_it_as_the_result_does(edge_report):
    report = bt.robustness(Flip, data=EDGE_DATA, **FREE, progress=lambda line: None)
    pd.testing.assert_frame_equal(report.verdicts, edge_report.verdicts)


def test_result_robustness_reuses_the_run_and_takes_no_run_argument(edge):
    with pytest.raises(ConfigError, match=r"result.rerun\(capital=...\).robustness\(\)"):
        edge.robustness(capital=50_000)


def test_a_check_setting_reaches_the_check(edge):
    report = edge.robustness(reshuffle={"max_drawdown": 0.5}, progress=lambda line: None)
    assert report.check("reshuffle").verdict == PASSED


def test_too_few_trades_leave_every_check_not_judged_and_nothing_passed():
    result = bt.backtest(Flip, data=bars(steps_for([2.0, -1.0], 10)), **FREE)
    report = result.robustness(checks="quick", progress=lambda line: None)
    assert set(report.verdicts["verdict"]) == {NOT_JUDGED}
    assert not report.passed


def test_the_plan_is_reported_once_before_the_checks_run(edge):
    lines = []
    edge.robustness(progress=lines.append)
    assert len(lines) == 1
    plan = lines[0]
    assert plan.startswith("bt.robustness: 4 checks of 17 for Flip")
    assert "on 60 closed trades" in plan and "the backtest is not rerun" in plan
    assert "13 checks are not built in this version" in plan


def test_bt_robustness_states_how_long_the_backtest_took():
    lines = []
    bt.robustness(Flip, data=EDGE_DATA, **FREE, checks="quick", progress=lines.append)
    assert re.search(r"The backtest took \d+\.\d\d s\.$", lines[0])


def test_the_page_leaves_out_how_long_the_backtest_took(edge_report, page):
    # A wall-clock time differs from call to call; the page is the same for a seed.
    report = bt.robustness(Flip, data=EDGE_DATA, **FREE, progress=lambda line: None)
    assert "took" not in report.plan
    assert report.tearsheet() == page


def test_without_progress_the_plan_goes_to_the_robustness_logger(edge, caplog):
    with caplog.at_level(logging.INFO, logger="openalgo.backtest.robustness"):
        edge.robustness(checks="quick")
    assert caplog.records[0].getMessage().startswith("bt.robustness: 4 checks")


def test_a_progress_that_is_not_a_function_is_refused(edge):
    with pytest.raises(ConfigError, match="progress takes a function"):
        edge.robustness(progress="loud")


def test_a_result_not_made_by_bt_backtest_has_no_run_to_check(edge):
    orphan = bt.Result(
        edge._output,
        edge._symbols,
        edge._assumptions,
        [],
        [],
        edge._manifest,
        strategy_closes=edge._strategy_closes,
    )
    with pytest.raises(BacktestError, match="not made by bt.backtest"):
        orphan.robustness()


# ----- the HTML report -----------------------------------------------------------------


def test_the_page_is_the_scorecard_a_section_per_check_and_the_notes(page):
    ids = ["random-out-of-sample", "reshuffle", "resample", "variance"]
    assert section_ids(page) == ["scorecard", *ids, "notes"]


def test_the_scorecard_counts_the_verdicts_and_states_the_wording_rule(page):
    scorecard = page[page.index("<section id='scorecard'") : page.index("<section id='random")]
    assert "3 passed, 0 failed, 0 not judged, 1 informational, 13 not built." in scorecard
    assert WORDING_RULE in scorecard
    assert scorecard.count("<tr>") == 1 + 17


def test_the_page_is_one_self_contained_well_formed_page(page):
    assert page.startswith("<!DOCTYPE html>") and page.isascii()
    lower = page.lower()
    for forbidden in ("http://", "https://", "@import", "<link", "<iframe", "<img", "src="):
        assert forbidden not in lower, forbidden
    structure = Structure(page)
    assert structure.errors == [] and structure.stack == []
    assert "@media (prefers-color-scheme: dark)" in page and "width=device-width" in page


def test_the_page_names_no_third_party_product(page):
    import hashlib

    tokens = set(re.findall(r"[a-z0-9][a-z0-9.\-]*", page.lower()))
    tokens |= {part for token in tokens for part in token.split("-")}
    hashed = {hashlib.sha256(token.encode()).hexdigest()[:16] for token in tokens}
    assert not hashed & FORBIDDEN_NAME_HASHES


def test_each_fan_has_its_bands_100_sampled_paths_and_the_run_bold(page):
    # Reshuffle, resample and the two of variance.
    assert page.count("class='fan-real'") == 4
    assert page.count("class='fan-outer'") == 4 and page.count("class='fan-inner'") == 4
    assert page.count("class='fan-path'") == 4 * 100  # 100 of the 1,000 runs of each fan
    assert page.count("class='fan-median'") == 4


def test_each_histogram_marks_its_reference_or_the_runs_own_value(page):
    assert page.count("class='mark-real'") == 1  # the run's order among the reshuffles
    assert page.count("class='mark-reference'") == 1  # zero among the differences


def test_the_notes_hold_the_settings_seeds_and_the_runs_assumptions(page, edge):
    notes = page[page.index("<section id='notes'") :]
    assert "seed 11" in notes and "points 5.0" in notes
    first_assumption = next(iter(edge.assumptions))
    assert f"<td class='l'>{first_assumption}</td>" in notes


def test_the_report_is_written_to_a_file(edge_report, tmp_path):
    target = tmp_path / "robustness.html"
    assert edge_report.tearsheet(target) == str(target)
    assert target.read_text(encoding="utf-8") == edge_report.tearsheet()
