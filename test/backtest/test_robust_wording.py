"""The words of the robustness reports (design 17, section 0 as amended on 2026-11-18):
the wording rule the full report prints at its top and the quick checks' reading share
their sentences; a failure says the edge did not hold up, never that it proves
overfitting; the trade-list checks say they cannot see a pick from many tries and name
the checks that can; no check is listed as not built, and no report names an internal
planning name; figures read without a minus zero or a doubled colon."""

import re
from pathlib import Path

import pytest
from robust_runs import SMALL_BARS, check_of, edge_run, trade_performance

import openalgo.backtest as bt
from openalgo.backtest._robust import _words as words
from openalgo.backtest._robust._quick import QUICK_READING
from openalgo.backtest._robust._verdict import (
    FAILING,
    FIXED,
    PASSING,
    PICKED,
    UNJUDGED,
    WORDING_RULE,
)
from openalgo.backtest.report._format import text as fmt_text

ROBUST = Path(bt.__file__).parent / "_robust"
RUN = Path(bt.__file__).parent / "run.py"


@pytest.fixture(scope="module")
def page():
    return edge_run().robustness(checks="quick", progress=lambda line: None).tearsheet()


@pytest.fixture(scope="module")
def every_check():
    """The report of every check on the planted edge, with few runs, in this process."""
    return edge_run().robustness(checks="all", progress=lambda line: None, **SMALL_BARS)


def test_passing_a_check_proves_nothing_on_its_own():
    assert PASSING == "Passing a check proves nothing on its own."
    assert WORDING_RULE.startswith(PASSING)


def test_a_failure_says_the_edge_did_not_hold_up_and_is_not_proof():
    assert FAILING == (
        "Failing one says the edge did not hold up under that check, so it may be fitted to "
        "the past or too small to tell apart from the noise of the data tested; a single "
        "failure is not proof of overfitting."
    )
    assert FAILING in WORDING_RULE


def test_the_rule_says_what_the_trade_list_checks_cannot_see_and_names_the_checks_that_can():
    assert PICKED == (
        "The checks of the trade list (random out of sample, trade reshuffle, trade resample "
        "and variance) cannot see a strategy picked as the best of many tried; the parameter "
        "neighbourhood, the noise-robust optimization and the optimizer's deflated Sharpe "
        "ratio and probability of backtest overfitting can."
    )
    assert PICKED in WORDING_RULE


def test_the_rule_says_the_rerun_checks_judge_the_chosen_settings_not_the_search():
    # Measured on an EMA crossover picked as the best of 100 settings on a random walk:
    # noise passed it in 3 of 3 seeds and the permutation test in 1 of 3, since both
    # rerun the settings chosen and never the search that chose them.
    assert FIXED == (
        "Nor can the checks that rerun the strategy with its parameters as chosen (versus "
        "random, permutation, noise, shifted bars, other markets, synthetic stress, "
        "randomized exits, delayed orders and liquidity): they judge the settings chosen, not "
        "the search that chose them."
    )
    assert WORDING_RULE.index(PICKED) < WORDING_RULE.index(FIXED) < WORDING_RULE.index(UNJUDGED)


def test_a_check_with_too_little_to_judge_never_passes():
    assert UNJUDGED == "A check with too little to judge says not judged and why, and never passes."
    assert WORDING_RULE.endswith(UNJUDGED)


def test_the_quick_checks_reading_shares_the_rules_sentences():
    assert QUICK_READING.startswith(PASSING)
    assert QUICK_READING.endswith(UNJUDGED)


def test_the_scorecard_prints_the_amended_rule_at_its_top(page):
    scorecard = page[page.index("<section id='scorecard'") : page.index("<section id='random")]
    for sentence in (PASSING, FAILING, PICKED, FIXED, UNJUDGED):
        assert fmt_text(sentence) in scorecard


def test_no_report_text_calls_a_failure_strong_evidence_of_overfitting(page):
    sources = [RUN, *sorted(ROBUST.glob("*.py"))]
    for path in sources:
        assert "strong evidence" not in path.read_text(encoding="utf-8"), path.name
    assert "strong evidence" not in page


def test_bt_robustness_says_a_win_rate_leaves_random_out_of_sample_not_judged():
    doc = " ".join(bt.robustness.__doc__.split())
    assert '"win_rate" (a win rate is not a measure of profit)' in doc
    assert '"profit_factor" or "win_rate"' not in doc


def test_the_report_lists_no_check_as_not_built_and_names_no_internal_unit(every_check):
    report = every_check
    page = report.tearsheet()
    assert len(report.verdicts) == 17
    assert "not built" not in page.lower() and "not built" not in " ".join(report.verdicts["note"])
    assert not re.search(r"\bRB\d", page)


# (fraction, decimals, text): a nonzero figure that would round to zero shows its first
# significant digit; a zero, of either sign, reads 0.
PERCENTS = [
    (0.875, 1, "87.5%"),
    (0.0, 1, "0.0%"),
    (-0.0, 1, "0.0%"),
    (-0.0004, 1, "-0.04%"),
    (-0.00000123, 1, "-0.0001%"),
    (0.004, 0, "0.4%"),
    (0.5, 0, "50%"),
    (1e-15, 1, "0.0%"),
]


@pytest.mark.parametrize(("fraction", "decimals", "text"), PERCENTS)
def test_a_percentage_never_reads_minus_zero(fraction, decimals, text):
    assert words.percent(fraction, decimals) == text


# (fraction, decimals, text): a percentile reads as an ordinal, rounded to `decimals`
# places with trailing zeros dropped; 11th to 13th take "th", a part of a hundred too.
ORDINALS = [
    (0.66, 0, "66th"),
    (0.6604, 0, "66th"),
    (0.95, 0, "95th"),
    (0.95, 1, "95th"),
    (0.975, 1, "97.5th"),
    (0.01, 0, "1st"),
    (0.02, 0, "2nd"),
    (0.03, 0, "3rd"),
    (0.04, 0, "4th"),
    (0.11, 0, "11th"),
    (0.12, 0, "12th"),
    (0.13, 0, "13th"),
    (0.21, 0, "21st"),
    (0.22, 0, "22nd"),
    (0.23, 0, "23rd"),
    (0.29, 0, "29th"),
    (0.0, 0, "0th"),
    (1.0, 0, "100th"),
    (float("nan"), 0, "n/a"),
    # Short of either end, a percentile shows the places that keep it from reading as
    # the end: only none below and every run below read "0th" and "100th".
    (0.996, 0, "99.6th"),
    (0.004, 0, "0.4th"),
    (0.9999, 1, "99.99th"),
    (0.99951, 0, "99.95th"),
]


@pytest.mark.parametrize(("fraction", "decimals", "text"), ORDINALS)
def test_a_percentile_reads_as_an_ordinal(fraction, decimals, text):
    assert words.ordinal(fraction, decimals) == text


def test_no_robustness_text_prints_a_percentile_as_a_percentage(every_check):
    sources = sorted(ROBUST.glob("*.py"))
    for path in sources:
        source = path.read_text(encoding="utf-8")
        assert not re.search(r"percent\([^)]*\)\} percentile", source), path.name
    page = every_check.tearsheet()
    assert "id='noise'" in page
    assert not re.search(r"\d% percentile", page)


def test_no_robustness_page_prints_a_minus_zero_percentage(every_check):
    # The planted edge's drawdowns are shallow: a chart key or a bin's hover readout of
    # -0.0002 reads "-0.02%", as the figures do.
    page = every_check.tearsheet()
    assert not re.search(r"-0\.0+%", page)


def test_the_win_rate_note_has_one_colon():
    result = check_of(
        trade_performance([150.0, 50.0] * 50), "random_out_of_sample", objective="win_rate"
    )
    assert result.note == (
        "The objective win_rate cannot judge this check: a win rate is not a measure of "
        "profit, since a strategy can lose money at any win rate; give objective= one of "
        "sharpe, sortino, net_profit, expectancy, profit_factor to judge it."
    )
