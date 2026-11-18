"""The bar checks (design 17, sections 2.4, 2.5, 2.6, 2.8 and 2.9) on own daily bars with
no server (`robust_runs`): a planted edge passes, an edge that comes from the drift alone
or from a digit of the price fails, a check with too little to judge is not judged, and
the same seed gives the same report.

The runs:
- `edge_run`: `Flip` with a planted edge, its trades 30, 20, 40, -10, 30, 20, -20, 50, 10
  and -10 rupees in turn; the order of the moves makes it, so permuted bars lose it, and
  noise of a fifth of a bar's range leaves it.
- `rising_run`: `Flip` on bars that rise 1 rupee a session, 60 trades of +10: random
  entries with the same holding time make the same, and every permuted path makes about
  as much (it rises as much), so the edge is the drift's.
- `odd_close_run`: `OddCloseFlip`, 40 trades of +30 planted on closes that end in 5 paisa:
  noise moves most closes, so the signal fires at random.
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from data_fixtures.fakes import FakeClient
from robust_runs import (
    FREE,
    Flip,
    bars,
    edge_run,
    no_edge_run,
    odd_close_run,
    rising_run,
    steps_for,
)
from test_quickstart import QUICKSTART, historify_rows

import openalgo.backtest as bt
from openalgo.backtest._robust._blocks import (
    CandlesBlock,
    FanBlock,
    HistogramBlock,
    LinesBlock,
    PathsBlock,
    SessionsBlock,
    TableBlock,
)
from openalgo.backtest._robust._check_permutation import p_value
from openalgo.backtest._robust._run_bars import run_bars
from openalgo.backtest._robust._verdict import FAILED, NOT_JUDGED, PASSED
from openalgo.backtest.errors import ConfigError

EDGE_MOVES = [3.0, 2.0, 4.0, -1.0, 3.0, 2.0, -2.0, 5.0, 1.0, -1.0]
#: Rules that keep every candidate, so stress reads the generator alone.
ALWAYS = {"rules": [lambda real, candidate: True], "min_kept": 5}


def quiet(line: str) -> None:
    """A progress function that drops the plan."""


def checked(result, name, runs=40, **settings):
    """The report of check `name` alone on `result`, in this process."""
    return result.robustness(checks=[name], workers=1, runs=runs, progress=quiet, **settings)


@pytest.fixture(scope="module")
def edge():
    return edge_run()


@pytest.fixture(scope="module")
def rising():
    return rising_run()


@pytest.fixture(scope="module")
def odd_close():
    return odd_close_run()


# ----- permutation test ----------------------------------------------------------------

# (values, real, p, at or above, measured): p = (at or above + 1) / (measured + 1).
P_VALUES = [
    ([1.0, 2.0, 3.0], 2.5, 2 / 4, 1, 3),
    ([1.0, 2.0, 3.0], 3.0, 2 / 4, 1, 3),  # a tie counts as at or above
    ([1.0, 2.0, 3.0], 4.0, 1 / 4, 0, 3),
    ([1.0, math.nan, 3.0], 2.0, 2 / 3, 1, 2),  # an undefined run is left out
]


@pytest.mark.parametrize(("values", "real", "p", "above", "measured"), P_VALUES)
def test_the_p_value_counts_the_runs_at_or_above_the_run(values, real, p, above, measured):
    assert p_value(np.array(values), real) == pytest.approx((p, above, measured))


def test_a_planted_edge_in_the_order_of_the_moves_passes_the_permutation_test(edge):
    result = checked(edge, "permutation").check("permutation")
    assert result.verdict == PASSED
    assert result.summary["at_or_above"] == 0 and result.summary["p_value"] == pytest.approx(1 / 41)


def test_an_edge_of_the_drift_alone_fails_the_permutation_test(rising):
    # Every permuted path rises as much, so about half the runs make as much.
    result = checked(rising, "permutation", objective="net_profit").check("permutation")
    assert result.verdict == FAILED
    assert result.summary["p_value"] > 0.2


def test_too_few_runs_to_reach_the_level_are_not_judged(edge):
    # With 10 runs the smallest p is 1 / 11, above 0.05: 19 runs are needed.
    result = checked(edge, "permutation", runs=10).check("permutation")
    assert result.verdict == NOT_JUDGED
    assert result.note.startswith("Too few runs to judge: 10 permuted runs measured")


def test_the_permutation_section_draws_the_fan_the_histogram_and_the_summary(edge):
    report = checked(edge, "permutation", runs=20)
    (fan, histogram), (tiles,) = report._blocks["permutation"]
    assert isinstance(fan, FanBlock) and fan.fan.runs == 20 and len(fan.dates) == 122
    assert isinstance(histogram, HistogramBlock) and len(histogram.values) == 20
    labels = [tile.label for tile in tiles.tiles]
    assert {"Best drawdown", "Worst drawdown", "Longest win streak", "Average loss streak"} <= set(
        labels
    )


# ----- noise ---------------------------------------------------------------------------


def test_a_planted_edge_survives_noise(edge):
    result = checked(edge, "noise").check("noise")
    assert result.verdict == PASSED
    assert result.summary["ratio"] >= 0.5 and result.summary["profitable_share"] >= 0.8


def test_an_edge_on_a_digit_of_the_close_fails_under_noise(odd_close):
    result = checked(odd_close, "noise").check("noise")
    assert result.verdict == FAILED
    assert result.summary["ratio"] < 0.5


def test_noise_settings_reach_the_noisy_bars(edge):
    still = {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0}
    result = checked(edge, "noise", runs=5, noise=still).check("noise")
    # No price moves, so every noisy run is the run itself.
    assert (result.table["objective"] == edge.metrics["sharpe"]).all()


def test_a_win_rate_cannot_judge_the_noise_ratio(edge):
    result = checked(edge, "noise", runs=5, objective="win_rate").check("noise")
    assert result.verdict == NOT_JUDGED
    assert "a win rate is not a measure of profit" in result.note


def test_the_noise_section_shows_the_bars_beside_a_noisy_copy(edge):
    report = checked(edge, "noise", runs=5)
    rows = report._blocks["noise"]
    real, noisy = rows[1]
    assert isinstance(real, CandlesBlock) and isinstance(noisy, CandlesBlock)
    assert len(real.bars) == len(noisy.bars) == 60
    assert real.bars["ts_open"].tolist() == noisy.bars["ts_open"].tolist()


def test_the_noise_histogram_marks_the_run_its_top_percentile_and_the_keep_line(edge):
    # Sharpe breaks even at 0, so the median must stay at or above half the run's.
    report = checked(edge, "noise", runs=5)
    check = report.check("noise")
    (_, histogram), *_ = report._blocks["noise"]
    marks = {mark.label: mark.value for mark in histogram.marks}
    assert marks["the run"] == edge.metrics["sharpe"]
    assert marks["95% percentile"] == check.summary["top_limit"]
    assert marks["the median keeps 50%"] == pytest.approx(0.5 * edge.metrics["sharpe"])


# ----- versus random -------------------------------------------------------------------


def test_a_planted_edge_beats_every_random_run(edge):
    result = checked(edge, "vs_random", runs=20).check("vs_random")
    assert result.verdict == PASSED
    assert result.summary["percentile"] == 1.0
    assert set(result.table["family"]) == {"random_entries", "permuted", "random_on_permuted"}


def test_random_entries_hold_the_runs_trade_count(edge):
    table = checked(edge, "vs_random", runs=10).check("vs_random").table
    random_entries = table[table["family"] == "random_entries"]
    assert (random_entries["trades"] == 60).all()


def test_random_entries_keep_every_trade_when_the_position_is_under_water_at_its_exit():
    # Each bar closes a rupee below its open, so at the close where a long is exited it
    # is under water: a long decided at that same close would add to it below its
    # average, which the engine refuses. The random longs wait a bar, so all 60 fill.
    data = bars(steps_for(EDGE_MOVES, 60))
    data["close"] = data["open"] - 1.0
    result = bt.backtest(Flip, data=data, **FREE)
    table = checked(result, "vs_random", runs=10, vs_random={"families": ["random_entries"]})
    assert (table.check("vs_random").table["trades"] == 60).all()


def test_an_edge_random_entries_match_fails(rising):
    # On bars that rise every session every one-bar trade makes 10 rupees, wherever it is.
    result = checked(rising, "vs_random", runs=10, vs_random={"families": ["random_entries"]})
    check = result.check("vs_random")
    assert check.verdict == FAILED
    assert check.summary["best_random"] == pytest.approx(rising.metrics["sharpe"])


def test_a_margin_asks_for_more_than_the_best_random_run(edge):
    result = checked(edge, "vs_random", runs=10, vs_random={"margin": 5.0}).check("vs_random")
    assert result.verdict == FAILED  # 2.6 times the best, not 6


def test_versus_random_shares_the_permutation_tests_runs():
    lines = []
    report = edge_run().robustness(
        checks=["vs_random", "permutation"], workers=1, runs=20, progress=lines.append
    )
    table = report.check("vs_random").table
    permuted = table[table["family"] == "permuted"]["objective"].to_numpy()
    np.testing.assert_array_equal(permuted, report.check("permutation").table["objective"])


def test_the_versus_random_section_has_a_table_by_family(edge):
    rows = checked(edge, "vs_random", runs=5)._blocks["vs_random"]
    (table,) = rows[1]
    assert isinstance(table, TableBlock) and len(table.rows) == 3
    # The random runs' average trades sit beside the run's own count, so a trader sees
    # when the run's rules (an entry window, margin) refused some random entries.
    assert table.subtitle == "The sharpe of each family's runs; the run has 60 closed trades"
    assert table.columns[-1] == "Average trades"


# ----- other markets -------------------------------------------------------------------


@pytest.fixture(scope="module")
def three_symbols():
    """The planted edge on SBIN and ICICIBANK, and moves that sum to zero on HDFCBANK."""
    edge_bars = bars(steps_for(EDGE_MOVES, 60))
    flat_bars = bars(steps_for([2.0, -2.0], 60))
    return {"SBIN": edge_bars, "ICICIBANK": edge_bars, "HDFCBANK": flat_bars}


def test_the_logic_holding_on_half_the_other_symbols_passes(three_symbols):
    result = bt.backtest(Flip, data=three_symbols, **FREE)
    report = result.robustness(
        checks=["other_markets"], others=["ICICIBANK", "HDFCBANK"], progress=quiet
    )
    check = report.check("other_markets")
    assert check.verdict == PASSED
    table = check.table.set_index("symbol")
    assert table.loc["ICICIBANK", "net_profit"] == 960.0  # the run's own trades again
    assert table.loc["HDFCBANK", "net_profit"] == 0.0
    assert check.summary["profitable_share"] == 0.5


def test_the_logic_failing_on_every_other_symbol_fails(three_symbols):
    data = {**three_symbols, "ICICIBANK": three_symbols["HDFCBANK"]}
    result = bt.backtest(Flip, data=data, **FREE)
    report = result.robustness(
        checks=["other_markets"], others=["ICICIBANK", "HDFCBANK"], progress=quiet
    )
    assert report.check("other_markets").verdict == FAILED


def test_other_markets_without_others_is_not_judged(edge):
    result = edge.robustness(checks=["other_markets"], progress=quiet).check("other_markets")
    assert result.verdict == NOT_JUDGED and result.note.startswith("No related symbols given")


def test_other_markets_need_the_others_bars_in_a_dict(edge):
    with pytest.raises(ConfigError, match="data= must be a dict by symbol"):
        edge.robustness(checks=["other_markets"], others=["ICICIBANK"], progress=quiet)


def test_other_markets_of_a_strategy_on_several_symbols_are_not_supported(three_symbols):
    class Pair(bt.Strategy):
        symbols, exchange, interval = ("SBIN", "ICICIBANK"), "NSE", "D"

    with pytest.raises(NotImplementedError, match="other markets for a strategy on several"):
        bt.robustness(
            Pair, data=three_symbols, **FREE, checks=["other_markets"], others=["HDFCBANK"]
        )


def test_the_other_markets_section_draws_each_symbol_and_its_buy_and_hold(three_symbols):
    result = bt.backtest(Flip, data=three_symbols, **FREE)
    report = result.robustness(checks=["other_markets"], others=["ICICIBANK"], progress=quiet)
    (chart,), (table,) = report._blocks["other_markets"]
    assert isinstance(chart, LinesBlock)
    assert [line.style for line in chart.lines] == ["bold", "dotted", "thin", "dotted"]
    assert isinstance(table, TableBlock) and table.rows[0][0] == "ICICIBANK"


# ----- synthetic stress ----------------------------------------------------------------


def test_a_planted_edge_holds_on_every_noisy_candidate(edge):
    stress = {"noise_candidates": 20, "permutation_candidates": 0, **ALWAYS}
    result = checked(edge, "stress", stress=stress).check("stress")
    assert result.verdict == PASSED
    assert result.table.set_index("rule").loc["<lambda>", "kept"] == 20


def test_an_edge_on_a_digit_of_the_close_fails_on_permuted_candidates(odd_close):
    stress = {"noise_candidates": 0, "permutation_candidates": 40, **ALWAYS}
    assert checked(odd_close, "stress", stress=stress).check("stress").verdict == FAILED


def test_rules_that_keep_too_few_candidates_are_not_judged(edge):
    never = {"noise_candidates": 5, "permutation_candidates": 5, "rules": [lambda a, b: False]}
    result = checked(edge, "stress", stress=never).check("stress")
    assert result.verdict == NOT_JUDGED and result.note.startswith("Too few kept candidates")


def test_any_keeps_a_candidate_one_rule_holds_for_and_all_needs_every_rule(edge):
    # One rule always holds and one never does: "any" keeps the 4 candidates, under the
    # first rule only, and "all" keeps none.
    def always(real, candidate):
        return True

    def never(real, candidate):
        return False

    stress = {
        "rules": [always, never],
        "noise_candidates": 4,
        "permutation_candidates": 0,
        "min_kept": 1,
    }
    any_kept = checked(edge, "stress", stress=stress).check("stress")
    assert any_kept.summary["candidates"] == 4
    assert any_kept.table.set_index("rule")["kept"].to_dict() == {"always": 4, "never": 0}
    assert any_kept.verdict == PASSED  # judged on the rule that kept candidates
    all_kept = checked(edge, "stress", stress={**stress, "combine": "all"}).check("stress")
    assert all_kept.verdict == NOT_JUDGED


def test_the_stress_section_draws_the_kept_equity_by_rule(edge):
    stress = {"noise_candidates": 8, "permutation_candidates": 0, **ALWAYS}
    (chart,), (table,) = checked(edge, "stress", stress=stress)._blocks["stress"]
    assert isinstance(chart, PathsBlock) and chart.groups[0].paths.shape[0] == 8
    assert isinstance(table, TableBlock)


def test_a_rule_given_as_a_function_is_named_in_the_settings_and_the_page_repeats(edge):
    # A function's own text carries where it sits in memory, which differs from process
    # to process; the page names it, so the same run and seed give the same page.
    def deeper_than_ever(real, candidate):
        return True

    stress = {
        "rules": [deeper_than_ever, "drawdown"],
        "noise_candidates": 4,
        "permutation_candidates": 0,
        "min_kept": 1,
    }
    page = checked(edge, "stress", stress=stress).tearsheet()
    assert "Settings: min_trades 30, noise_candidates 4" in page
    assert "rules [deeper_than_ever, drawdown]" in page
    assert " at 0x" not in page and "&lt;function" not in page


# ----- what every bar check shares -----------------------------------------------------


@pytest.mark.parametrize("name", ["vs_random", "permutation", "noise", "stress"])
def test_a_run_with_no_edge_is_not_judged(name):
    result = checked(no_edge_run(), name, runs=5).check(name)
    assert result.verdict == NOT_JUDGED and result.note.startswith("No edge to judge")


def test_the_same_seed_gives_the_same_bar_checks(edge):
    settings = {"stress": {"noise_candidates": 6, "permutation_candidates": 6}}
    once = edge.robustness(workers=1, runs=20, seed=3, progress=quiet, **settings)
    again = edge.robustness(workers=1, runs=20, seed=3, progress=quiet, **settings)
    for name in ("vs_random", "permutation", "noise", "stress"):
        pd.testing.assert_frame_equal(once.check(name).table, again.check(name).table)
    assert once.tearsheet() == again.tearsheet()


def test_with_the_bars_random_out_of_sample_draws_the_price(edge):
    report = edge.robustness(
        checks=["random_out_of_sample", "permutation"], workers=1, runs=20, progress=quiet
    )
    chart = report._blocks["random_out_of_sample"][0][0]
    assert isinstance(chart, SessionsBlock) and not chart.percent
    assert chart.values[0] == 500.0 and chart.line.startswith("SBIN's close")
    assert "Line: SBIN&#x27;s close at each session close" in report.tearsheet()


def test_a_run_on_openalgo_data_is_read_once_more_and_rerun_without_the_client(
    tmp_path, monkeypatch
):
    # The report reads the run's bars through the client once more, as the run read
    # them; every rerun takes them from memory, and the first gives the run's own figure,
    # the warm-up of its averages included, so the plan reports no change of the bars.
    monkeypatch.setenv("OPENALGO_BT_CACHE", str(tmp_path))
    client = FakeClient(api={("SBIN", "NSE", "D"): historify_rows(QUICKSTART.daily_bars())})
    span = {"start": QUICKSTART.START, "end": QUICKSTART.END, "capital": QUICKSTART.CAPITAL}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = bt.backtest(QUICKSTART.EmaCross, client, source="api", **span)
    client.calls.clear()
    run_bars(result)
    one_read = len(client.calls)
    client.calls.clear()
    lines = []
    report = result.robustness(
        checks=["noise"], workers=1, runs=3, noise={"min_trades": 2}, progress=lines.append
    )
    assert one_read > 0 and len(client.calls) == one_read
    assert "changed since the run" not in lines[0]
    assert report.check("noise").summary["measured"] == 3


def test_the_bar_checks_change_no_figure_of_the_run(edge):
    before = edge.metrics_table()
    edge.robustness(checks=["permutation"], workers=1, runs=5, progress=quiet)
    pd.testing.assert_frame_equal(edge.metrics_table(), before)


def test_a_strategy_on_two_symbols_is_permuted_on_one_grid():
    class PairFlip(bt.Strategy):
        symbols, exchange, interval = ("SBIN", "ICICIBANK"), "NSE", "D"

        def on_bar(self, bar):
            if self.positions["SBIN"].is_flat:
                self.buy(qty=10, symbol="SBIN")
            else:
                self.exit("SBIN")

    edge_bars = bars(steps_for(EDGE_MOVES, 60))
    result = bt.backtest(PairFlip, data={"SBIN": edge_bars, "ICICIBANK": edge_bars}, **FREE)
    check = checked(result, "permutation", runs=20).check("permutation")
    assert check.verdict == PASSED and (check.table["refused"] == "").all()


def test_the_bar_checks_of_a_futures_series_are_not_supported():
    from test_futures import _run as futures_run

    with pytest.raises(NotImplementedError, match="for a strategy on a futures series"):
        checked(futures_run(), "permutation", runs=5)


def test_the_other_markets_page_is_one_well_formed_page(three_symbols):
    from test_report_tearsheet import Structure

    result = bt.backtest(Flip, data=three_symbols, **FREE)
    page = result.robustness(
        checks=["other_markets"], others=["ICICIBANK"], progress=quiet
    ).tearsheet()
    structure = Structure(page)
    assert structure.errors == [] and structure.stack == []
    assert page.count("class='line-bold'") == 1 and page.count("class='line-dotted'") == 2
