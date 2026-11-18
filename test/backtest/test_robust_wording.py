"""The words of the robustness reports (design 17, section 0 as amended on 2026-11-18):
the wording rule the full report prints at its top and the quick checks' reading share
their sentences; a failure says the edge did not hold up, never that it proves
overfitting; the trade-list checks say they cannot see a pick from many tries and name
the checks that can; a check not built says what it needs in plain words, with no
internal planning name; figures read without a minus zero or a doubled colon."""

import re
from pathlib import Path

import pytest
from robust_runs import SMALL_BARS, check_of, edge_run, trade_performance

import openalgo.backtest as bt
from openalgo.backtest._robust import _words as words
from openalgo.backtest._robust._quick import QUICK_READING
from openalgo.backtest._robust._request import build_request
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
#: The checks a later version brings, and what each needs.
LATER = {
    "shifted": "late orders, other fill prices and shifted bar boundaries in the engine",
    "random_exits": "late orders, other fill prices and shifted bar boundaries in the engine",
    "delayed": "late orders, other fill prices and shifted bar boundaries in the engine",
    "liquidity": "late orders, other fill prices and shifted bar boundaries in the engine",
    "out_of_sample": "walk-forward and the parameter searches",
    "walk_forward": "walk-forward and the parameter searches",
    "parameters": "walk-forward and the parameter searches",
    "noise_optimization": "walk-forward and the parameter searches",
}


def request(**changes):
    settings = {
        "checks": "all",
        "objective": "sharpe",
        "runs": 1_000,
        "seed": 0,
        "workers": None,
        "others": None,
        "check_settings": {},
        "ranges": {},
    }
    return build_request(**{**settings, **changes})


@pytest.fixture(scope="module")
def page():
    return edge_run().robustness(checks="quick", progress=lambda line: None).tearsheet()


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
        "random, permutation, noise, other markets and synthetic stress): they judge the "
        "settings chosen, not the search that chose them."
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


@pytest.mark.parametrize("name", list(LATER))
def test_a_check_not_built_says_what_it_needs_in_plain_words(name):
    with pytest.raises(NotImplementedError) as raised:
        request(checks=[name])
    message = str(raised.value)
    assert f"it needs {LATER[name]}" in message
    assert not re.search(r"\bRB\d", message) and "unit" not in message


def test_parameter_ranges_say_what_they_need_in_plain_words():
    with pytest.raises(NotImplementedError) as raised:
        request(ranges={"fast": range(5, 30, 5)})
    message = str(raised.value)
    assert "walk-forward and the parameter searches" in message
    assert not re.search(r"\bRB\d", message) and "unit" not in message


def test_the_report_names_no_internal_unit(page):
    report = edge_run().robustness(checks="all", progress=lambda line: None, **SMALL_BARS)
    notes = report.verdicts.set_index("check")["note"]
    assert notes["delayed"] == f"Not built in this version: it needs {LATER['delayed']}."
    assert not re.search(r"\bRB\d", report.tearsheet())


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


def test_the_win_rate_note_has_one_colon():
    result = check_of(
        trade_performance([150.0, 50.0] * 50), "random_out_of_sample", objective="win_rate"
    )
    assert result.note == (
        "The objective win_rate cannot judge this check: a win rate is not a measure of "
        "profit, since a strategy can lose money at any win rate; give objective= one of "
        "sharpe, sortino, net_profit, expectancy, profit_factor to judge it."
    )
