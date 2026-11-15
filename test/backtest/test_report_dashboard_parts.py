"""The parts of the dashboard: axis arithmetic, the time-series pane, heatmap colours,
the weekly table, the period figures and which metrics a benchmark is shown for. Every
expected value is worked by hand in the comment beside it."""

import math
import re

import numpy as np
import pandas as pd
import pytest

from openalgo.backtest import metrics
from openalgo.backtest.report._dashboard import data, heatmaps, scale
from openalgo.backtest.report._dashboard import timeseries as ts

IST = "Asia/Kolkata"


@pytest.mark.parametrize(
    ("low", "high", "count", "step"),
    [
        # 0.2347 / 6 = 0.039: 3.9 hundredths lies between sqrt(10) and sqrt(50), so 5.
        (-0.144, 0.0907, 6, 0.05),
        # 0.2347 / 10 = 0.023: 2.3 lies between sqrt(2) and sqrt(10), so 2.
        (-0.144, 0.0907, 10, 0.02),
        # 150 / 5 = 30: 3.0 lies below sqrt(10), so 2 tens.
        (0.0, 150.0, 5, 20.0),
        # 1 / 10 = 0.1 exactly: 1.
        (0.0, 1.0, 10, 0.1),
    ],
)
def test_the_round_step_is_the_nearest_on_a_log_scale(low, high, count, step):
    assert scale.round_step(low, high, count) == pytest.approx(step)


def test_ticks_are_the_multiples_of_the_step_inside_the_span():
    assert scale.ticks(-0.15, 0.10, 6) == [-0.15, -0.1, -0.05, 0.0, 0.05, 0.1]


def test_an_axis_is_widened_to_round_ends():
    # Step 0.02 (above): floor(-0.144 / 0.02) = -8 and ceil(0.0907 / 0.02) = 5.
    assert scale.widen(-0.144, 0.0907) == pytest.approx((-0.16, 0.10))


def test_bands_leave_a_padding_share_between_and_half_at_the_ends():
    # Step 100 / (3 - 0.3 + 0.6) = 30.30; first offset (100 - 30.30 x 2.7) / 2 = 9.09.
    starts, width = scale.bands(3, 100.0, 0.3)
    assert starts == pytest.approx([9.0909, 39.3939, 69.6970], abs=1e-4)
    assert width == pytest.approx(21.2121, abs=1e-4)


def test_histogram_bins_have_round_edges_and_a_closed_last_bin():
    # Step 0.2 for four parts of [0, 1]; 1.0 falls in the last bin.
    counted = scale.bins(np.array([0.0, 0.1, 0.25, 1.0]), 0.0, 1.0, 4)
    assert [n for _, _, n in counted] == [2, 1, 0, 0, 1]
    assert [left for left, _, _ in counted] == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8])


def test_quartiles_interpolate_between_order_statistics():
    assert scale.quartiles(np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan])) == (1, 2, 3, 4, 5)


def test_a_linear_map_runs_from_its_start_to_its_end():
    assert scale.Linear(0.0, 10.0, 100.0, 0.0)(2.5) == 75.0


def _chart(values, dates, height=128, log=False) -> str:
    line = ts.Line("Strategy", np.asarray(values, dtype=float), "var(--text)")
    return ts.chart(dates, [line], height=height, percent=True, label="t", ident="t", log_scale=log)


def test_the_pane_leaves_a_fifth_above_and_a_tenth_below_the_values():
    # Values 0 to 0.7 take 70% of the pane: the top is 0.7 + 0.2, the bottom -0.1.
    dates = pd.date_range("2025-01-01", periods=2)
    page = _chart([0.0, 0.7], dates)
    top = float(re.search(r"data-top='([^']+)'", page).group(1))
    bottom = float(re.search(r"data-bottom='([^']+)'", page).group(1))
    assert (top, bottom) == (pytest.approx(0.9), pytest.approx(-0.1))


def test_value_labels_are_at_least_thirty_pixels_apart():
    dates = pd.date_range("2025-01-01", periods=50)
    page = _chart(np.linspace(-0.3, 1.2, 50), dates, height=320)
    # Labels run from the lowest value (largest top) to the highest.
    tops = [float(t) for t in re.findall(r"<span style='top:([\d.]+)px'>", page)]
    assert len(tops) >= 3
    assert all(lower - higher >= 30 for higher, lower in zip(tops[1:], tops, strict=False))


def test_a_long_span_is_labelled_at_the_first_session_of_each_year():
    dates = pd.bdate_range("2022-06-01", "2025-06-30")
    page = _chart(np.linspace(0, 1, len(dates)), dates)
    assert re.findall(r"%'>(\d{4})</span>", page) == ["2023", "2024", "2025"]


def test_a_log_scale_spaces_equal_growth_equally():
    # Growth 1 to 4: 100% and 300% returns sit one and two doublings above zero.
    dates = pd.date_range("2025-01-01", periods=3)
    page = _chart([0.0, 1.0, 3.0], dates, log=True)
    top = float(re.search(r"data-top='([^']+)'", page).group(1))
    bottom = float(re.search(r"data-bottom='([^']+)'", page).group(1))
    span = math.log(4) / 0.7
    assert (top, bottom) == (
        pytest.approx(math.log(4) + 0.2 * span),
        pytest.approx(-0.1 * span),
    )


def test_fewer_than_two_values_draw_no_chart():
    dates = pd.date_range("2025-01-01", periods=3)
    assert "<svg" not in _chart([np.nan, 0.1, np.nan], dates)


@pytest.mark.parametrize(
    ("value", "css"),
    [
        # Half the largest: 0.5 x 88 = 44 percent of the gain colour.
        (0.05, "color-mix(in oklab, var(--heat-pos) 44%, var(--heat-mid))"),
        (-0.10, "color-mix(in oklab, var(--heat-neg) 88%, var(--heat-mid))"),
        (math.nan, "transparent"),
    ],
)
def test_a_heatmap_cell_mixes_in_the_gain_or_loss_colour_by_its_size(value, css):
    assert heatmaps.heat(value, 0.10) == css


def test_weeks_compound_their_sessions_and_are_named_by_their_sunday():
    returns = pd.Series(
        [0.01, 0.02, -0.01],
        index=pd.DatetimeIndex(["2026-01-05", "2026-01-06", "2026-01-12"], name="date"),
    )
    weeks = data.weekly(returns)
    # 1.01 x 1.02 - 1 = 0.0302 in ISO week 2 of 2026 (1 January 2026 is a Thursday).
    assert [(w.week, w.label) for w in weeks[2026]] == [(2, "Jan 11"), (3, "Jan 18")]
    assert [w.value for w in weeks[2026]] == pytest.approx([0.0302, -0.01])


@pytest.fixture(scope="module")
def performance() -> metrics.Performance:
    # Three sessions: +10%, -20%, +5% from Rs 100: 110, 88, 92.4.
    closes = pd.DatetimeIndex(
        [pd.Timestamp(f"2026-01-{d} 15:30", tz=IST) for d in ("29", "30")]
        + [pd.Timestamp("2026-02-02 15:30", tz=IST)]
    )
    equity = pd.Series([110.0, 88.0, 92.4], index=closes)
    return metrics.analyse(equity, exchanges="NSE", opening_wealth=100.0)


def test_month_and_year_to_date_are_the_last_cells_of_the_tables(performance):
    # February holds only the last session: +5%; 2026 compounds all three: -7.6%.
    assert data.month_to_date(performance) == pytest.approx(0.05)
    assert data.year_to_date(performance) == pytest.approx(0.924 - 1)


def test_the_drawdown_of_a_session_is_its_deepest(performance):
    # Peak 110, then 88 (-20%) and 92.4 (-16%).
    assert data.session_drawdown(performance).to_numpy() == pytest.approx([0.0, -0.2, -0.16])


def test_cumulative_return_compounds_and_skips_an_unknown_session():
    returns = pd.Series([0.1, np.nan, -0.5])
    assert data.cumulative(returns).to_numpy() == pytest.approx([0.1, np.nan, -0.45], nan_ok=True)


def test_a_benchmark_is_shown_for_path_figures_and_not_for_trade_figures():
    assert {"total_return", "sharpe", "max_drawdown", "skew"} <= data.COMPARABLE
    assert not {"exposure", "win_rate", "profit_factor", "beta"} & data.COMPARABLE


def test_a_run_without_a_benchmark_has_no_benchmark_figures(performance):
    assert data.benchmark_performance(performance) is None
