"""`bt.robustness`, `Result.robustness` and `RobustnessReport` end to end, on own daily
bars with no server (`robust_runs`): a strategy with a planted edge passes every judged
check, a small edge in much noise fails them, a strategy with no edge is not judged, the
same seed gives the same report, the plan is reported before the checks run, and the
HTML report is the scorecard, a section per check and the Notes in one self-contained
page.

`Flip` trades 60 times. With a planted edge its trades make 30, 20, 40, -10, 30, 20,
-20, 50, 10 and -10 rupees, repeated: 7 of 10 win and the mean is 16. With a small edge
they make +25 and -20 in turn, 150 in all; with no edge, +20 and -20, nothing in all.

The full reports keep their bar checks small (`robust_runs.SMALL_BARS`: one process, 20
runs a check, stress on 10 noisy candidates); `test_robust_bar_checks` holds the bar
checks' own specifications.
"""

import dataclasses
import logging
import re

import pandas as pd
import pytest
from robust_runs import (
    FREE,
    SMALL_BARS,
    Flip,
    bars,
    edge_run,
    no_edge_run,
    small_edge_run,
    steps_for,
)
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
from openalgo.backtest.report._format import text as fmt_text

THIS_VERSION = [
    "random_out_of_sample",
    "vs_random",
    "permutation",
    "noise",
    "other_markets",
    "stress",
    "reshuffle",
    "resample",
    "variance",
]
TRADE_LIST = ["random_out_of_sample", "reshuffle", "resample", "variance"]
EDGE_DATA = bars(steps_for([3.0, 2.0, 4.0, -1.0, 3.0, 2.0, -2.0, 5.0, 1.0, -1.0], 60))
#: Stress on noisy candidates only: the planted edge holds under noise.
STRESS = {"noise_candidates": 10, "permutation_candidates": 0, "min_kept": 5}
SMALL = {**SMALL_BARS, "stress": STRESS}


def quiet(line: str) -> None:
    """A progress function that drops the plan."""


@pytest.fixture(scope="module")
def edge():
    return edge_run()


@pytest.fixture(scope="module")
def edge_report(edge):
    return edge.robustness(progress=quiet, **SMALL)


@pytest.fixture(scope="module")
def small_edge_report():
    return small_edge_run().robustness(checks="quick", progress=quiet)


@pytest.fixture(scope="module")
def page(edge_report):
    return edge_report.tearsheet()


def test_the_public_names_come_from_openalgo_backtest(edge_report):
    assert {"robustness", "RobustnessReport", "CheckResult"} <= set(bt.__all__)
    assert isinstance(edge_report, bt.RobustnessReport)
    assert isinstance(edge_report.check("resample"), bt.CheckResult)


def test_a_planted_edge_passes_every_judged_check(edge_report):
    verdicts = edge_report.verdicts.set_index("check")["verdict"]
    assert verdicts[THIS_VERSION].tolist() == [
        PASSED,
        PASSED,
        PASSED,
        PASSED,
        NOT_JUDGED,  # no others= given
        PASSED,
        INFORMATIONAL,
        PASSED,
        PASSED,
    ]
    assert edge_report.passed


def test_a_small_edge_in_much_noise_fails(small_edge_report):
    verdicts = small_edge_report.verdicts.set_index("check")["verdict"]
    assert verdicts[["random_out_of_sample", "resample", "variance"]].tolist() == [FAILED] * 3
    assert not small_edge_report.passed


def test_a_strategy_with_no_edge_is_not_judged_and_nothing_passed():
    report = no_edge_run().robustness(checks="quick", progress=quiet)
    verdicts = report.verdicts.set_index("check")["verdict"]
    assert verdicts[["random_out_of_sample", "resample", "variance"]].tolist() == [NOT_JUDGED] * 3
    assert not report.passed


def test_the_scorecard_lists_every_check_and_the_ones_not_built(edge_report):
    verdicts = edge_report.verdicts
    assert list(verdicts.columns) == ["check", "question", "figure", "rule", "verdict", "note"]
    assert list(verdicts.index) == list(range(1, 18))
    assert (verdicts["verdict"] == NOT_BUILT).sum() == 8
    delayed = verdicts.set_index("check").loc["delayed"]
    assert delayed["note"] == (
        "Not built in this version: it needs late orders, other fill prices and shifted bar "
        "boundaries in the engine."
    )


def test_a_check_result_is_made_of_its_public_fields_only():
    names = [field.name for field in dataclasses.fields(bt.CheckResult)]
    assert names == [
        "name",
        "title",
        "question",
        "verdict",
        "figure",
        "rule",
        "note",
        "table",
        "summary",
        "settings",
        "seed",
    ]


def test_one_check_has_its_table_summary_and_settings(edge_report):
    resample = edge_report.check("resample")
    assert len(resample.table) == 1_000
    assert resample.summary["profitable_share"] == 1.0
    assert resample.settings["runs"] == 1_000 and resample.seed == 11


def test_asking_for_a_check_not_built_says_what_it_needs(edge_report):
    with pytest.raises(NotImplementedError, match="it needs late orders"):
        edge_report.check("delayed")


def test_asking_for_a_check_the_report_did_not_run_is_refused(edge):
    report = edge.robustness(checks=["resample"], progress=quiet)
    with pytest.raises(ConfigError, match="did not run variance"):
        report.check("variance")
    assert list(report.verdicts["check"]) == ["resample"]


def test_the_same_seed_gives_the_same_report(edge):
    first = edge.robustness(checks="quick", seed=4, runs=300, progress=quiet)
    again = edge.robustness(checks="quick", seed=4, runs=300, progress=quiet)
    for name in TRADE_LIST:
        pd.testing.assert_frame_equal(first.check(name).table, again.check(name).table)
    assert first.tearsheet() == again.tearsheet()


def test_another_seed_draws_other_runs(edge):
    first = edge.robustness(checks="quick", seed=4, runs=300, progress=quiet)
    other = edge.robustness(checks="quick", seed=5, runs=300, progress=quiet)
    assert not first.check("resample").table.equals(other.check("resample").table)


def test_bt_robustness_checks_its_run_as_the_result_does_and_the_page_has_no_time(
    edge_report, page
):
    # A wall-clock time differs from call to call; the page is the same for a seed.
    report = bt.robustness(Flip, data=EDGE_DATA, **FREE, progress=quiet, **SMALL)
    pd.testing.assert_frame_equal(report.verdicts, edge_report.verdicts)
    assert "took" not in report.plan
    assert report.tearsheet() == page


def test_result_robustness_reuses_the_run_and_takes_no_run_argument(edge):
    with pytest.raises(ConfigError, match=r"result.rerun\(capital=...\).robustness\(\)"):
        edge.robustness(capital=50_000)


def test_a_check_setting_reaches_the_check(edge):
    report = edge.robustness(checks="quick", reshuffle={"max_drawdown": 0.5}, progress=quiet)
    assert report.check("reshuffle").verdict == PASSED


def test_too_few_trades_leave_every_check_not_judged_and_nothing_passed():
    result = bt.backtest(Flip, data=bars(steps_for([2.0, -1.0], 10)), **FREE)
    report = result.robustness(checks="quick", progress=quiet)
    assert set(report.verdicts["verdict"]) == {NOT_JUDGED}
    assert not report.passed


def test_a_quick_plan_says_nothing_is_rerun(edge):
    lines = []
    edge.robustness(checks="quick", progress=lines.append)
    assert lines == [
        "bt.robustness: 4 checks of 17 for Flip (random_out_of_sample, reshuffle, resample, "
        "variance), 1,000 runs each from seed 0, on 60 closed trades over 122 sessions. They "
        "read the run's trade list, so the backtest is not rerun and no worker process starts."
    ]


def test_the_plan_counts_the_reruns_before_they_run(edge):
    # Versus random runs 20 of each of its three families, 60 in all, of which the 20 on
    # permuted bars are the permutation test's own runs: 60 + 20 - 20 = 60 reruns.
    lines = []
    report = edge.robustness(
        checks=["vs_random", "permutation", "resample"],
        workers=1,
        vs_random={"runs": 20},
        permutation={"runs": 20},
        progress=lines.append,
    )
    assert len(lines) == 1
    assert lines[0].startswith(report.plan)
    assert report.plan == (
        "bt.robustness: 3 checks of 17 for Flip (vs_random, permutation, resample), 1,000 runs "
        "each from seed 0, on 60 closed trades over 122 sessions. The trade-list checks read "
        "the run's trade list; the bar checks rerun the backtest 60 times or fewer on "
        "generated bars or other symbols (vs_random 60, permutation 20), each an ordinary "
        "backtest of the same kernel. Versus random shares 20 runs with the permutation test."
    )
    assert re.search(
        r"A rerun on the run's own bars took \d+\.\d\d s; in this process the reruns should "
        r"take about \d+\.\d s\.$",
        lines[0],
    )


def test_bt_robustness_states_how_long_the_backtest_took():
    lines = []
    bt.robustness(Flip, data=EDGE_DATA, **FREE, checks="quick", progress=lines.append)
    assert re.search(r"The backtest took \d+\.\d\d s\.$", lines[0])


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
    ids = [name.replace("_", "-") for name in THIS_VERSION]
    assert section_ids(page) == ["scorecard", *ids, "notes"]


def test_the_scorecard_counts_the_verdicts_and_states_the_wording_rule(page):
    scorecard = page[page.index("<section id='scorecard'") : page.index("<section id='random")]
    assert "7 passed, 0 failed, 1 not judged, 1 informational, 8 not built." in scorecard
    assert fmt_text(WORDING_RULE) in scorecard
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


def test_each_fan_has_its_bands_its_sampled_paths_and_the_run_bold(page):
    # Reshuffle, resample and the two of variance draw 100 of their 1,000 runs; versus
    # random draws its 60 runs, the permutation test and noise their 20.
    assert page.count("class='fan-real'") == 7
    assert page.count("class='fan-outer'") == 7 and page.count("class='fan-inner'") == 7
    assert page.count("class='fan-path'") == 4 * 100 + 60 + 20 + 20
    assert page.count("class='fan-median'") == 7


def test_each_histogram_marks_its_reference_or_the_runs_own_value(page):
    # The run's order among the reshuffles, and the run among the random, permuted and
    # noise objectives; zero among the differences; the limit of permutation, and the
    # two of noise (its top percentile and the line its median must keep).
    assert page.count("class='mark-real'") == 4
    assert page.count("class='mark-reference'") == 1
    assert page.count("class='mark-limit'") == 3


def test_the_notes_hold_the_settings_seeds_and_the_runs_assumptions(page, edge):
    notes = page[page.index("<section id='notes'") :]
    assert "seed 11" in notes and "points 5.0" in notes
    first_assumption = next(iter(edge.assumptions))
    assert f"<td class='l'>{first_assumption}</td>" in notes


def test_the_report_is_written_to_a_file(edge_report, tmp_path):
    target = tmp_path / "robustness.html"
    assert edge_report.tearsheet(target) == str(target)
    assert target.read_text(encoding="utf-8") == edge_report.tearsheet()
