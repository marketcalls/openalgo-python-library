"""What the bar checks judge and what the plan counts (design 17, sections 2.4, 2.5, 2.9
and 1, the lead's rules of 2026-11-18):

- synthetic stress judges a rule only when it singles out shocks: it keeps at most half
  of the candidates of its generators that ran;
- on generated bars a strategy of decisions fixed in advance (`bt.from_arrays`) is not
  judged: the permutation test, synthetic stress and versus random's strategy on
  permuted bars leave it out, with one reason;
- the plan counts and times only the reruns the checks will run, from the same reasons
  their runs use.

The runs are `robust_runs`' own daily bars, with no server.
"""

import numpy as np
import pytest
from robust_runs import FREE, bars, edge_run, no_edge_run, steps_for

import openalgo.backtest as bt
from openalgo.backtest._robust._check_stress import MOST_KEPT, why_unjudged
from openalgo.backtest._robust._verdict import NOT_JUDGED, PASSED

EDGE_MOVES = [3.0, 2.0, 4.0, -1.0, 3.0, 2.0, -2.0, 5.0, 1.0, -1.0]
#: The one reason a check on generated bars gives for decisions fixed in advance.
FIXED = (
    "its decisions are arrays fixed in advance (bt.from_arrays), which never read the "
    "generated bars, so on them it trades at the same bars and only its fills move"
)


def quiet(line: str) -> None:
    """A progress function that drops the plan."""


@pytest.fixture(scope="module")
def edge():
    return edge_run()


@pytest.fixture(scope="module")
def fixed():
    """Flip's decisions as arrays fixed in advance (bt.from_arrays) on the edge bars."""
    data = bars(steps_for(EDGE_MOVES, 60))
    bar = np.arange(len(data))
    arrays = bt.from_arrays(
        bar % 2 == 0,
        bar % 2 == 1,
        symbol="SBIN",
        exchange="NSE",
        interval="D",
        size=bt.size.qty(10),
    )
    return bt.backtest(arrays, data=data, **FREE)


def every(nth: int):
    """A stress rule that holds for every `nth` candidate it is asked about, from the
    first: a rule that keeps a known minority, deterministic in one process."""
    asked = []

    def every_nth(real, candidate):
        asked.append(1)
        return (len(asked) - 1) % nth == 0

    return every_nth


def always(real, candidate):
    return True


# ----- synthetic stress: a rule that keeps most candidates singles out no shock --------


# (kept, candidates, expected share, min_kept, why), the reasons in the order they apply.
WHY = [
    (4, 10, 1.0, 5, "keeps 4 of 10 candidates, fewer than 5"),
    (6, 10, 1.0, 5, "keeps 6 of 10 candidates, so it does not single out shocks"),
    (5, 10, 1.0, 5, ""),  # exactly half is judged
    (
        5,
        10,
        0.0,
        5,
        "keeps 5 of 10 candidates, of generators on whose candidates the strategy never "
        "makes money",
    ),
]


@pytest.mark.parametrize(("kept", "candidates", "expected", "least", "why"), WHY)
def test_a_rule_is_judged_when_it_keeps_enough_and_at_most_half(
    kept, candidates, expected, least, why
):
    assert MOST_KEPT == 0.5
    assert why_unjudged(kept, candidates, expected, least) == why


def test_a_rule_that_keeps_every_candidate_is_not_judged_and_cannot_pass(edge):
    stress = {"rules": [always], "noise_candidates": 10, "permutation_candidates": 0}
    report = edge.robustness(
        checks=["stress"], workers=1, stress={**stress, "min_kept": 1}, progress=quiet
    )
    result = report.check("stress")
    assert result.verdict == NOT_JUDGED
    row = result.table.set_index("rule").loc["always"]
    assert (row["kept"], row["candidates"], bool(row["judged"])) == (10, 10, False)
    assert row["why"] == "keeps 10 of 10 candidates, so it does not single out shocks"
    assert result.note.startswith(
        "No rule can judge: always keeps 10 of 10 candidates, so it does not single out shocks."
    )


def test_a_rule_that_keeps_a_minority_is_judged_as_before(edge):
    # Every third of 12 noisy candidates: 4 kept, a third of them, and the planted edge
    # makes money on every noisy copy.
    stress = {"rules": [every(3)], "noise_candidates": 12, "permutation_candidates": 0}
    report = edge.robustness(
        checks=["stress"], workers=1, stress={**stress, "min_kept": 1}, progress=quiet
    )
    result = report.check("stress")
    row = result.table.set_index("rule").loc["every_nth"]
    assert (row["kept"], row["candidates"], bool(row["judged"]), row["why"]) == (4, 12, True, "")
    assert result.verdict == PASSED


def test_the_default_rules_judge_only_the_rules_that_select(edge):
    stress = {"noise_candidates": 40, "permutation_candidates": 40}
    report = edge.robustness(checks=["stress"], workers=1, stress=stress, progress=quiet)
    table = report.check("stress").table.set_index("rule")
    judged = table[table["judged"]]
    assert (judged["kept"] <= MOST_KEPT * judged["candidates"]).all()
    many = table[table["kept"] > MOST_KEPT * table["candidates"]]
    assert not many["judged"].any()
    assert all(why.endswith("so it does not single out shocks") for why in many["why"])


def test_the_stress_rule_names_the_most_a_rule_may_keep(edge):
    stress = {"noise_candidates": 4, "permutation_candidates": 0}
    report = edge.robustness(checks=["stress"], workers=1, stress=stress, progress=quiet)
    assert report.check("stress").rule == (
        "under every rule that keeps 10 or more candidates and at most 50% of its "
        "generators', the kept ones make money at least 50% as often as all the candidates "
        "of their generators"
    )


# ----- decisions fixed in advance are not judged on generated bars ----------------------


@pytest.mark.parametrize(
    ("name", "title"),
    [("permutation", "The permutation test"), ("stress", "Synthetic stress")],
)
def test_checks_on_generated_bars_leave_decisions_fixed_in_advance_unjudged(fixed, name, title):
    settings = {"stress": {"noise_candidates": 4, "permutation_candidates": 4}}
    report = fixed.robustness(
        checks=[name],
        workers=1,
        runs=20,
        progress=quiet,
        **{key: value for key, value in settings.items() if key == name},
    )
    result = report.check(name)
    assert result.verdict == NOT_JUDGED
    assert result.note == f"{title} cannot judge this strategy: {FIXED}."


def test_versus_random_keeps_only_its_random_entries_for_fixed_decisions(fixed):
    result = fixed.robustness(checks=["vs_random"], workers=1, runs=5, progress=quiet)
    check = result.check("vs_random")
    assert set(check.table["family"]) == {"random_entries", "random_on_permuted"}
    assert check.note.count(f" The strategy on permuted bars is left out: {FIXED}.") == 1


def test_versus_random_of_the_strategy_on_permuted_bars_alone_is_not_judged(fixed):
    report = fixed.robustness(
        checks=["vs_random"],
        workers=1,
        runs=5,
        vs_random={"families": ["permuted"]},
        progress=quiet,
    )
    check = report.check("vs_random")
    assert check.verdict == NOT_JUDGED
    assert check.note == (
        f"No family of random runs can run. The strategy on permuted bars is left out: {FIXED}."
    )


# ----- the plan counts only the reruns that will run ------------------------------------

#: The plan's words when no bar check will rerun.
NONE_RERUN = (
    "the bar checks will not rerun the backtest, since each is not judged before any "
    "rerun, as its note says."
)


def plan_of(result, checks, **settings):
    lines = []
    result.robustness(checks=checks, workers=1, runs=20, progress=lines.append, **settings)
    return lines[0]


def test_the_plan_of_a_run_with_no_edge_says_no_rerun_will_run():
    plan = plan_of(no_edge_run(), ["vs_random", "permutation", "noise", "stress"])
    assert NONE_RERUN in plan and "times or fewer" not in plan
    assert plan.endswith("no other will run.")


def test_the_plan_of_decisions_fixed_in_advance_counts_no_rerun_on_generated_bars(fixed):
    plan = plan_of(fixed, ["permutation", "noise", "stress"])
    assert NONE_RERUN in plan
    # Versus random's random entries still run: 20 on the run's bars, 20 on permuted bars.
    plan = plan_of(fixed, ["vs_random", "permutation"])
    assert "rerun the backtest 40 times or fewer" in plan and "(vs_random 40)" in plan


@pytest.mark.parametrize(
    ("objective", "counted"),
    [
        ("win_rate", "(permutation 20)"),  # noise needs a break-even win_rate lacks
        ("net_profit", "(permutation 20, noise 20)"),
    ],
)
def test_the_plan_counts_noise_only_when_its_objective_can_judge(edge, objective, counted):
    lines = []
    edge.robustness(
        checks=["permutation", "noise"],
        objective=objective,
        workers=1,
        runs=20,
        progress=lines.append,
    )
    assert counted in lines[0]


def test_the_plan_counts_no_rerun_of_a_check_with_too_few_trades(edge):
    plan = plan_of(edge, ["noise", "permutation"], noise={"min_trades": 100})
    assert "(permutation 20)" in plan and "noise 20" not in plan


def test_a_strategy_of_fixed_decisions_on_its_own_bars_is_still_backtested(fixed):
    # The plan reruns nothing on generated bars; the run itself and its trade-list checks
    # stand.
    report = fixed.robustness(checks=["resample", "noise"], workers=1, runs=20, progress=quiet)
    assert report.check("resample").verdict != NOT_JUDGED
    assert report.check("noise").verdict == NOT_JUDGED


def test_a_strategy_that_reads_its_bars_plans_every_family_of_versus_random(edge):
    plan = plan_of(edge, ["vs_random"])
    assert "(vs_random 60)" in plan
