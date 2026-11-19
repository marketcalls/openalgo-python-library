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

import dataclasses
import math
import re
import warnings

import numpy as np
import pandas as pd
import pytest
from data_fixtures.fakes import FakeClient
from robust_runs import (
    FREE,
    SMALL_BARS,
    Flip,
    bars,
    edge_run,
    no_edge_run,
    nse_sessions,
    odd_close_run,
    opens_bars,
    rising_run,
    steps_for,
)
from test_quickstart import QUICKSTART, historify_rows

import openalgo.backtest as bt
from openalgo.backtest._robust import _check_noise
from openalgo.backtest._robust._bar_common import joined, rerun_set
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
from openalgo.backtest._robust._reruns import Reruns
from openalgo.backtest._robust._run_bars import run_bars
from openalgo.backtest._robust._verdict import FAILED, NOT_JUDGED, PASSED
from openalgo.backtest.errors import ConfigError

EDGE_MOVES = [3.0, 2.0, 4.0, -1.0, 3.0, 2.0, -2.0, 5.0, 1.0, -1.0]
#: Rules that keep every candidate: stress reads the generator alone, and judges no rule,
#: since one that keeps every candidate singles out no shock.
ALWAYS = {"rules": [lambda real, candidate: True], "min_kept": 5}


def half():
    """A stress rule that keeps every second candidate it is asked about (the second,
    the fourth, ...): half of them or one fewer, so it is judged. It reads the first
    instrument's calls only, so a candidate of several instruments counts once."""
    asked: dict[int, int] = {}

    def every_other(real, candidate):
        key = id(real)
        asked[key] = asked.get(key, 0) + 1
        return key == next(iter(asked)) and asked[key] % 2 == 0

    return every_other


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


@pytest.mark.parametrize("name", ["noise", "vs_random"])
def test_a_function_objective_cannot_judge_the_ratio_checks(edge, name):
    # A ratio is measured from what a run that makes nothing scores, which a function
    # does not say: the end equity, for one, scores the capital, so every noisy run would
    # keep about all of it.
    def end_equity(result):
        return float(result.metrics["end_equity"])

    result = checked(edge, name, runs=5, objective=end_equity).check(name)
    assert result.verdict == NOT_JUDGED
    assert result.note.startswith(
        "The objective end_equity cannot judge this check: what a function of a Result gives "
        "for a run that makes nothing is not known"
    )


@pytest.fixture(scope="module")
def one_session():
    """Flip on 5-minute bars of one session: 34 trades of +10 rupees, and a Sharpe ratio
    that is undefined, since it needs returns over two sessions or more."""
    day = nse_sessions("2024-03-01", 1)[0]
    stamps = pd.DatetimeIndex(
        [
            pd.Timestamp(f"{day} 09:15", tz="Asia/Kolkata") + pd.Timedelta(minutes=5 * k)
            for k in range(75)
        ],
        name="timestamp",
    )
    steps = np.where(np.arange(75) % 2 == 1, 1.0, -0.5)
    opens = 500 + np.concatenate(([0.0], np.cumsum(steps)[:-1]))
    columns = {"open": opens, "high": opens + 0.5, "low": opens - 0.5, "close": opens}
    data = pd.DataFrame({**columns, "volume": 1_000.0}, index=stamps)

    class FiveMinuteFlip(Flip):
        interval = "5m"

    return bt.backtest(FiveMinuteFlip, data=data, **FREE)


@pytest.mark.parametrize(("name", "compared"), [("noise", "noise"), ("vs_random", "random")])
def test_an_undefined_objective_is_named_undefined(one_session, name, compared):
    assert math.isnan(one_session.metrics["sharpe"]) and one_session.metrics["net_profit"] > 0
    result = checked(one_session, name, runs=5).check(name)
    assert result.verdict == NOT_JUDGED
    assert result.note.startswith("The run's sharpe is undefined")
    assert f"so the {compared} runs have no figure to be measured against" in result.note


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


def test_noise_cannot_judge_decisions_fixed_in_advance(fixed):
    # Arrays made from the real bars stay the same on every noisy copy, so noise could
    # move only the fills: the strategy would keep its edge whatever it was fitted to.
    noise = checked(fixed, "noise", runs=5).check("noise")
    assert noise.verdict == NOT_JUDGED
    assert noise.note == (
        "Noise cannot judge this strategy: its decisions are arrays fixed in advance "
        "(bt.from_arrays), which never read the generated bars, so on them it trades at the "
        "same bars and only its fills move."
    )


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
    assert marks["95th percentile"] == check.summary["top_limit"]
    assert marks["the median keeps 50%"] == pytest.approx(0.5 * edge.metrics["sharpe"])


def test_the_noise_words_say_how_many_noise_runs_the_run_is_above(edge):
    # No price moves, so every noisy run is the run itself: each is profitable and none
    # is below the run.
    still = {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0}
    report = checked(edge, "noise", runs=5, noise=still)
    check = report.check("noise")
    assert check.figure.endswith(
        "100.0% of noise runs profitable; the run above 0.0% of the noise runs"
    )
    assert check.rule.endswith("and the run is not above their 95th percentile")
    (tiles,) = report._blocks["noise"][-1]
    shown = {tile.label: (tile.value, tile.sub) for tile in tiles.tiles}
    assert shown["Above"] == ("0.0%", "of the noise runs")


#: The noise runs of the planted sets at the top limit.
NOISE_RUNS = 1_000


def planted_noise(real: float, below: int):
    """A stand-in for the noise check's set of reruns: its first rerun repeated
    `NOISE_RUNS` times, each profitable, with objectives 1 to 1,000 parts of the run's
    `real` / (`below` + 0.5), so exactly `below` of them are below the run."""

    def made(rows):
        many = joined([rerun_set(rows[:1])] * NOISE_RUNS)
        parts = np.arange(1.0, NOISE_RUNS + 1.0)
        return dataclasses.replace(
            many, objective=real * parts / (below + 0.5), net_profit=np.full(NOISE_RUNS, 100.0)
        )

    return made


# (noise runs below the run, verdict, their share): the 95th percentile of 1 to 1,000
# parts, linear between order statistics, is 950.05 parts (0.95 x 999 = 949.05 places
# past the first), so a run of 949.5 parts passes and one of 950.5 or 951.5 is above it.
# The median, 500.5 parts, keeps over half of each run.
TOP_LIMIT_ROWS = [(949, PASSED, "94.9%"), (950, FAILED, "95.0%"), (951, FAILED, "95.1%")]


@pytest.mark.parametrize(("below", "verdict", "share"), TOP_LIMIT_ROWS)
def test_the_noise_figure_at_the_top_limit_reads_as_its_verdict(
    edge, monkeypatch, below, verdict, share
):
    monkeypatch.setattr(_check_noise, "rerun_set", planted_noise(edge.metrics["sharpe"], below))
    report = checked(edge, "noise", runs=1)
    check = report.check("noise")
    assert check.verdict == verdict
    assert check.figure.endswith(f"the run above {share} of the noise runs")
    assert check.rule.endswith("and the run is not above their 95th percentile")
    (tiles,) = report._blocks["noise"][-1]
    shown = {tile.label: tile.value for tile in tiles.tiles}
    assert shown["Above"] == share


# (top, its name in the rule): to a tenth, or as many places as keep a top short of 1
# from reading as the 100th.
TOP_NAMES = [(0.975, "97.5th"), (0.9999, "99.99th")]


@pytest.mark.parametrize(("top", "name"), TOP_NAMES)
def test_another_top_percentile_is_named_as_it_was_set(edge, top, name):
    report = checked(edge, "noise", runs=5, noise={"top": top})
    assert report.check("noise").rule.endswith(f"not above their {name} percentile")


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


class TargetInsideTheBar(bt.Strategy):
    """Buys 10 shares when flat with a target half a rupee above its fill: on the edge
    bars each trade reaches it inside its entry bar (held 0 bars, placed as 1) and the
    next long is decided at that bar's close, so its 121 trades, each with the bar a long
    waits after the one before, need about twice the run's 122 strategy bars."""

    symbol, exchange, interval = "SBIN", "NSE", "D"

    def on_bar(self, bar):
        if self.position.is_flat:
            self.buy(qty=10, target=bt.points(0.5))


class StopInsideTheBar(TargetInsideTheBar):
    """`TargetInsideTheBar` with a stop half a rupee below the fill instead: every trade
    loses inside its entry bar."""

    def on_bar(self, bar):
        if self.position.is_flat:
            self.buy(qty=10, stop=bt.points(0.5))


@pytest.mark.parametrize("checks", [["vs_random"], ["vs_random", "permutation"]])
def test_random_entries_that_do_not_fit_are_left_out_and_the_report_goes_on(checks):
    result = bt.backtest(TargetInsideTheBar, data=bars(steps_for(EDGE_MOVES, 60)), **FREE)
    lines = []
    report = result.robustness(checks=checks, workers=1, runs=5, progress=lines.append)
    check = report.check("vs_random")
    assert set(check.table["family"]) == {"permuted"}
    assert (
        " Random entries are left out: the trades of SBIN do not fit among its strategy bars "
        "of the run."
    ) in check.note
    # The plan counts the runs of the families that can run, and times one of them.
    assert "vs_random 5" in lines[0] and "A rerun on generated bars took" in lines[0]


def test_a_losing_run_whose_random_entries_do_not_fit_has_no_edge_to_judge():
    # No check reruns a run that did not make money, so the plan times no generated rerun.
    result = bt.backtest(StopInsideTheBar, data=bars(steps_for(EDGE_MOVES, 60)), **FREE)
    lines = []
    report = result.robustness(checks=["vs_random"], workers=1, runs=5, progress=lines.append)
    check = report.check("vs_random")
    assert check.verdict == NOT_JUDGED and check.note.startswith("No edge to judge")
    assert "A rerun on the run's own bars took" in lines[0]


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
    assert [line.group for line in chart.lines] == [0, 0, 1, 1]
    assert isinstance(table, TableBlock) and table.rows[0][0] == "ICICIBANK"


def test_the_other_markets_chart_tells_each_symbol_apart_without_colour(three_symbols):
    # Each symbol's two lines share its colour and the shape at their end; the key names
    # every line's style and shape, and says what a buy and hold measures.
    result = bt.backtest(Flip, data=three_symbols, **FREE)
    page = result.robustness(
        checks=["other_markets"], others=["ICICIBANK", "HDFCBANK"], progress=quiet
    ).tearsheet()
    section = page[page.index("Equity by symbol") :]
    classes = re.findall(r"<polyline class='([a-z0-9-]+)'", section)
    assert classes == ["line-bold", "hold-0", "series-1", "hold-1", "series-2", "hold-2"]
    ends = re.findall(r"class='end-(\d)'", section)
    assert ends == ["0", "0", "1", "1", "2", "2"]
    key = re.search(r"<p class='key'>(SBIN, the run[^<]*)</p>", section).group(1)
    assert key == (
        "SBIN, the run: bold, ending in a circle; SBIN buy and hold: dotted, ending in a "
        "circle; ICICIBANK: thin, ending in a square; ICICIBANK buy and hold: dotted, ending "
        "in a square; HDFCBANK: thin, ending in a triangle; HDFCBANK buy and hold: dotted, "
        "ending in a triangle. A strategy line is its cumulative return on the capital; a buy "
        "and hold is the return of the price from the open of its first bar of the run, which "
        "a run invested in part of the time does not compare with directly."
    )


def test_the_other_markets_chart_draws_three_other_symbols_and_the_table_every_one(
    three_symbols,
):
    # Four colours and shapes tell four symbols apart and the run's own takes the first,
    # so a fourth other symbol would look like the run's own buy and hold: it is in the
    # table, and the key says so.
    data = {
        **three_symbols,
        "AXISBANK": three_symbols["SBIN"],
        "KOTAKBANK": three_symbols["HDFCBANK"],
    }
    result = bt.backtest(Flip, data=data, **FREE)
    others = ["ICICIBANK", "HDFCBANK", "AXISBANK", "KOTAKBANK"]
    report = result.robustness(checks=["other_markets"], others=others, progress=quiet)
    page = report.tearsheet()
    section = page[page.index("Equity by symbol") :]
    classes = re.findall(r"<polyline class='([a-z0-9-]+)'", section)
    assert classes == [
        "line-bold",
        "hold-0",
        "series-1",
        "hold-1",
        "series-2",
        "hold-2",
        "series-3",
        "hold-3",
    ]
    key = re.search(r"<p class='key'>(SBIN, the run[^<]*)</p>", section).group(1)
    assert "KOTAKBANK" not in key
    assert key.endswith(
        "The chart draws the first 3 of the 4 other symbols; the table has every one."
    )
    assert report.check("other_markets").table["symbol"].tolist() == others


def test_the_chart_has_a_shape_for_each_group_of_lines_it_tells_apart():
    from openalgo.backtest._robust._blocks import LINE_GROUPS
    from openalgo.backtest._robust._charts_bars import SHAPES

    assert len(set(SHAPES)) == len(SHAPES) == LINE_GROUPS


# ----- synthetic stress ----------------------------------------------------------------


def test_a_planted_edge_holds_on_the_noisy_candidates_a_rule_keeps(edge):
    stress = {"noise_candidates": 20, "permutation_candidates": 0, "rules": [half()]}
    result = checked(edge, "stress", stress={**stress, "min_kept": 5}).check("stress")
    assert result.verdict == PASSED
    assert result.table.set_index("rule").loc["every_other", "kept"] == 10


def test_a_rule_that_keeps_every_candidate_judges_nothing(edge):
    stress = {"noise_candidates": 20, "permutation_candidates": 0, **ALWAYS}
    result = checked(edge, "stress", stress=stress).check("stress")
    assert result.verdict == NOT_JUDGED
    assert result.table.set_index("rule").loc["<lambda>", "kept"] == 20


def test_a_planted_edge_passes_synthetic_stress_at_its_defaults(edge):
    # The permutation takes the planted edge away (it lives in the order of the moves), so
    # the strategy makes money on about half the permuted paths, shocks or not; judged
    # against that share, the shocks among them take nothing more from it.
    result = checked(edge, "stress").check("stress")
    assert result.verdict == PASSED
    assert result.summary["permuted_profitable"] < 0.6
    assert result.summary["noisy_profitable"] == 1.0


def test_a_strategy_whose_losses_fall_on_the_shocks_fails(edge):
    # A rule that keeps exactly the permuted paths on which Flip loses: Flip makes money
    # on about half of them all, and on none of those kept.
    def flip_loses(real, candidate):
        opens = candidate["open"].to_numpy()
        return (opens[2:121:2] - opens[1:120:2]).sum() < 0

    stress = {"noise_candidates": 10, "permutation_candidates": 60, "rules": [flip_loses]}
    result = checked(edge, "stress", stress=stress).check("stress")
    assert result.verdict == FAILED
    row = result.table.set_index("rule").loc["flip_loses"]
    assert row["kept_noisy"] == 0 and row["kept"] >= 10
    assert row["profitable_share"] == 0.0 and row["expected_share"] > 0.3


def test_an_edge_of_the_drift_alone_has_no_shock_to_judge(rising):
    # Bars that rise a rupee every session: a permutation of equal moves is the same path
    # and noise of a fifth of a bar's range never makes the price fall, so no candidate
    # falls deeper, longer or further below its average than the run's own bars.
    report = checked(rising, "stress", objective="net_profit")
    result = report.check("stress")
    assert result.verdict == NOT_JUDGED
    assert result.note.startswith(
        "No rule can judge: drawdown keeps 0 of 400 candidates, fewer than 10; recovery "
        "keeps 0 of 400 candidates, fewer than 10; below_average keeps 0 of 400 candidates, "
        "fewer than 10."
    )
    # The rule table stays, with the kept counts the design asks for, and so does the
    # section's chart and table.
    table = result.table.set_index("rule")
    assert table.index.tolist() == ["drawdown", "recovery", "below_average"]
    assert table["kept"].tolist() == [0, 0, 0] and not table["judged"].any()
    assert result.summary["candidates"] == 400 and result.summary["judged_rules"] == 0
    (chart,), (rules,) = report._blocks["stress"]
    assert isinstance(chart, PathsBlock) and chart.groups == ()
    assert [row[0] for row in rules.rows] == ["drawdown", "recovery", "below_average"]


@pytest.fixture(scope="module")
def walk():
    """An EMA crossover on a random walk of 600 sessions: a realistic path, with history
    before start for its averages."""
    from test_robust_bar_parts import Ema

    rng = np.random.default_rng(5)
    opens = 500 + np.cumsum(np.round(rng.normal(0.1, 3.0, 600) / 0.05) * 0.05)
    data = opens_bars(opens)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bt.backtest(Ema, data=data, start=data.index[150].date(), **FREE)


def test_stress_keeps_only_the_candidates_that_are_shocks(walk):
    # A noisy copy moves each price by at most a fifth of the bar's range, which never
    # makes a fall half as deep again as the run's own; a few permuted paths do.
    stress = {"noise_candidates": 50, "permutation_candidates": 50, "min_trades": 2, "min_kept": 1}
    result = checked(walk, "stress", stress=stress).check("stress")
    table = result.table.set_index("rule")
    assert table["kept_noisy"].sum() == 0
    assert 0 < result.summary["kept"] <= 25  # a quarter of the 100 candidates at most


def test_the_stress_rule_says_which_rules_are_judged(edge):
    result = checked(edge, "stress", stress={"noise_candidates": 4, "permutation_candidates": 0})
    rule = result.check("stress").rule
    assert rule == (
        "under every rule that keeps 10 or more candidates and at most 50% of its "
        "generators', the kept ones make money at least 50% as often as all the candidates "
        "of their generators"
    )


def test_stress_reruns_the_permutation_tests_paths_and_the_noise_checks_copies(edge, monkeypatch):
    done = []
    close = Reruns.close

    def recording_close(reruns):
        done.append(reruns.done)
        close(reruns)

    monkeypatch.setattr(Reruns, "close", recording_close)
    lines = []
    edge.robustness(
        checks=["permutation", "noise", "stress"],
        workers=1,
        runs=20,
        stress={"noise_candidates": 10, "permutation_candidates": 20},
        progress=lines.append,
    )
    assert (
        "the bar checks rerun the backtest 40 times or fewer on generated bars or other "
        "symbols (permutation 20, noise 20, stress 30), each an ordinary backtest of the same "
        "kernel. The checks share 30 of the runs listed, which read the same generated bars, "
        "so the total counts each once."
    ) in lines[0]
    assert done == [40]  # 20 permuted paths and 20 noisy copies, each run once


def test_rules_that_keep_too_few_candidates_are_not_judged(edge):
    never = {"noise_candidates": 5, "permutation_candidates": 5, "rules": [lambda a, b: False]}
    result = checked(edge, "stress", stress=never).check("stress")
    assert result.verdict == NOT_JUDGED
    assert result.note.startswith("No rule can judge: <lambda> keeps 0 of 10 candidates, fewer")


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
    all_kept = checked(edge, "stress", stress={**stress, "combine": "all"}).check("stress")
    assert all_kept.verdict == NOT_JUDGED
    # A rule that keeps half the candidates is judged under "any".
    selecting = {**stress, "rules": [half(), never]}
    judged = checked(edge, "stress", stress=selecting).check("stress")
    assert judged.table.set_index("rule")["kept"].to_dict() == {"every_other": 2, "never": 0}
    assert judged.verdict == PASSED  # judged on the rule that kept candidates


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


class PairFlip(bt.Strategy):
    """Flip on SBIN, with ICICIBANK's bars read too."""

    symbols, exchange, interval = ("SBIN", "ICICIBANK"), "NSE", "D"

    def on_bar(self, bar):
        if self.positions["SBIN"].is_flat:
            self.buy(qty=10, symbol="SBIN")
        else:
            self.exit("SBIN")


def test_a_strategy_on_two_symbols_is_permuted_on_one_grid():
    edge_bars = bars(steps_for(EDGE_MOVES, 60))
    result = bt.backtest(PairFlip, data={"SBIN": edge_bars, "ICICIBANK": edge_bars}, **FREE)
    check = checked(result, "permutation", runs=20).check("permutation")
    assert check.verdict == PASSED and (check.table["refused"] == "").all()


@pytest.fixture(scope="module")
def missing_bar():
    """PairFlip with one session missing from ICICIBANK's bars: 122 and 121 bars."""
    edge_bars = bars(steps_for(EDGE_MOVES, 60))
    other = edge_bars.drop(edge_bars.index[50])
    return bt.backtest(PairFlip, data={"SBIN": edge_bars, "ICICIBANK": other}, **FREE)


GRID = "the bars of SBIN and ICICIBANK are not at the same times (122 and 121 bars)"


def test_stress_on_symbols_with_a_missing_bar_judges_its_noisy_candidates(missing_bar):
    stress = {"noise_candidates": 10, "permutation_candidates": 10, "rules": [half()]}
    check = checked(missing_bar, "stress", stress={**stress, "min_kept": 5}).check("stress")
    assert check.verdict == PASSED and check.summary["candidates"] == 10
    # The note names the candidates made, then why the permuted ones were not.
    assert check.note.startswith(
        "Candidates: the noise check's first 10 noisy copies of the run's bars; the "
        "strategy ran on each,"
    )
    assert "permuted paths" not in check.note
    assert f"Permuted candidates are left out: {GRID}" in check.note


def test_the_plan_counts_no_permuted_candidate_of_symbols_with_a_missing_bar(missing_bar):
    lines = []
    stress = {"noise_candidates": 10, "permutation_candidates": 10}
    missing_bar.robustness(checks=["stress"], workers=1, stress=stress, progress=lines.append)
    assert "rerun the backtest 10 times or fewer" in lines[0] and "(stress 10)" in lines[0]


def test_a_full_report_on_symbols_with_a_missing_bar_keeps_every_check(missing_bar):
    report = missing_bar.robustness(progress=quiet, **SMALL_BARS)
    permutation = report.check("permutation")
    assert permutation.verdict == NOT_JUDGED
    assert permutation.note.startswith(f"No permuted path can be made: {GRID}")
    vs_random = report.check("vs_random")
    assert set(vs_random.table["family"]) == {"random_entries"}
    assert f"The families on permuted bars are left out: {GRID}" in vs_random.note
    assert report.check("noise").verdict == PASSED


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
    assert page.count("class='line-bold'") == 1 and page.count("class='series-1'") == 1
    assert page.count("class='hold-0'") == 1 and page.count("class='hold-1'") == 1
