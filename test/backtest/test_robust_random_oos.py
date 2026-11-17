"""Random out of sample (design 17, section 2.2): sessions put out of sample at random,
each closed trade in the part of its entry session, worked by hand on trade lists of
one trade per session (`robust_runs.trade_performance`).

- Every session makes +100: each part's net profit per year is 100 x sessions per year,
  the same in both parts, so every assignment keeps all of it.
- Sessions alternate +100 and -100 and the trades sum to zero: the out-of-sample P&L is
  minus the in-sample P&L, so whenever the in-sample figure is above zero the
  out-of-sample one is below it, and no assignment keeps anything.
- Sessions alternate +150 and +50: every part has a mean near 100 and a spread near 50,
  so the out-of-sample Sharpe ratio stays near the in-sample one.
"""

from dataclasses import replace

import numpy as np
import pytest
from robust_runs import check_of, trade_performance

from openalgo.backtest._robust._blocks import SessionsBlock
from openalgo.backtest._robust._part_objective import Part, part_measure
from openalgo.backtest._robust._trades import trade_run
from openalgo.backtest._robust._verdict import FAILED, NOT_JUDGED, PASSED
from openalgo.backtest.report._kernel import call

STEADY = [100.0] * 100
NOISE = [100.0, -100.0] * 50
EDGE = [150.0, 50.0] * 50


def test_each_assignment_puts_30_percent_of_the_sessions_out_of_sample():
    result = check_of(trade_performance(STEADY), "random_out_of_sample", runs=200)
    assert result.summary["out_of_sample_sessions"] == 30
    assert (result.table["out_of_sample_trades"] == 30).all()


def test_a_trade_belongs_to_the_part_of_its_entry_session():
    # 40 trades on every other session, then 20 sessions without one: 100 sessions.
    performance = trade_performance([100.0] * 40, sessions_apart=2, extra_sessions=20)
    table = check_of(performance, "random_out_of_sample", runs=200).table
    assert ((table["in_sample_trades"] + table["out_of_sample_trades"]) == 40).all()
    assert table["out_of_sample_trades"].nunique() > 1


def test_steady_profit_per_session_keeps_everything_in_every_assignment():
    result = check_of(
        trade_performance(STEADY), "random_out_of_sample", runs=200, objective="net_profit"
    )
    assert result.verdict == PASSED
    assert result.summary["kept_share"] == 1.0
    assert result.table["difference"].to_numpy() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("objective", ["sharpe", "net_profit", "expectancy"])
def test_trades_that_sum_to_zero_never_keep_their_in_sample_figure(objective):
    result = check_of(trade_performance(NOISE), "random_out_of_sample", objective=objective)
    assert result.verdict == FAILED
    assert result.summary["kept_share"] == 0.0


@pytest.mark.parametrize("objective", ["sharpe", "sortino"])
def test_parts_that_break_even_are_not_above_zero_whatever_the_rounding(objective):
    # 0.2 rupees on 1,00,000 has no exact binary fraction; parts with as many wins as
    # losses make exactly nothing, so their ratio is 0, never a rounding residue above it.
    result = check_of(
        trade_performance([0.2, -0.2] * 50), "random_out_of_sample", objective=objective
    )
    assert result.summary["kept_share"] == 0.0
    inside = result.table["in_sample"]
    assert ((inside == 0) | (inside.abs() > 1e-6)).all()
    assert (inside == 0).any()


@pytest.mark.parametrize("objective", ["sharpe", "sortino"])
@pytest.mark.parametrize("rate", [0.0, 0.065])
def test_a_parts_ratio_is_the_kernels_ratio_of_its_session_returns(objective, rate):
    # The ratio of the whole run's sessions, as a part holding all of them.
    pnl = [120.0, -40.0, 75.5, -60.25, 10.0, 33.0] * 10
    performance = trade_performance(pnl)
    run = trade_run(performance)
    session_pnl = np.bincount(run.entry_session, weights=run.pnl, minlength=len(run.sessions))
    whole = Part(session_pnl, run.pnl)
    measured = part_measure(objective, replace(run, risk_free_rate=rate))(whole)
    annual = (run.sessions_per_year, rate, run.elapsed_nanos)
    kernel = call("path", session_pnl / run.capital, *annual)[objective]
    assert measured == pytest.approx(kernel, rel=1e-12)


def test_a_planted_edge_keeps_its_sharpe_ratio_out_of_sample():
    result = check_of(trade_performance(EDGE), "random_out_of_sample")
    assert result.verdict == PASSED
    assert result.summary["kept_share"] >= 0.95


def test_the_share_out_of_sample_is_a_setting():
    result = check_of(trade_performance(STEADY), "random_out_of_sample", runs=50, oos=0.5)
    assert result.summary["out_of_sample_sessions"] == 50


@pytest.mark.parametrize("objective", ["cagr", lambda result: 1.0])
def test_an_objective_a_part_cannot_be_measured_by_is_not_judged(objective):
    result = check_of(trade_performance(STEADY), "random_out_of_sample", objective=objective)
    assert result.verdict == NOT_JUDGED
    assert "cannot be measured on a part of the trades" in result.note
    assert "sharpe, sortino, net_profit, expectancy, profit_factor, win_rate" in result.note


def test_a_share_that_leaves_a_part_empty_is_not_judged():
    result = check_of(trade_performance([100.0] * 30), "random_out_of_sample", oos=0.01)
    assert result.verdict == NOT_JUDGED
    assert result.note.startswith("Too few sessions to judge")


def test_the_chart_shades_the_first_assignments_sessions():
    result = check_of(trade_performance(STEADY), "random_out_of_sample", runs=20)
    block = result._blocks[0][0]
    assert isinstance(block, SessionsBlock)
    assert block.marked.sum() == 30 and len(block.dates) == 100


def test_the_same_seed_gives_the_same_assignments_and_another_seed_others():
    performance = trade_performance(EDGE)
    first = check_of(performance, "random_out_of_sample", runs=40, seed=2).table
    again = check_of(performance, "random_out_of_sample", runs=40, seed=2).table
    other = check_of(performance, "random_out_of_sample", runs=40, seed=3).table
    assert first.equals(again) and not first.equals(other)
