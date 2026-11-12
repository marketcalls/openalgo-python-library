"""Report text: Indian digit grouping, compact rupees, percentages, durations, axes."""

import math

import pytest

from openalgo.backtest.report import _charts as charts
from openalgo.backtest.report import _format as fmt

R = "&#8377;"


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (0.0, "0.00"),
        (999.0, "999.00"),
        (1000.0, "1,000.00"),
        (123456.789, "1,23,456.79"),
        (12345678.0, "1,23,45,678.00"),
        (-1234567.5, "-12,34,567.50"),
    ],
)
def test_indian_digit_grouping(value, text):
    assert fmt.indian_digits(value) == text


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (2_500_000.0, f"{R}25 L"),
        (12_345_678.0, f"{R}1.2 Cr"),
        (1_500.0, f"{R}1.5 K"),
        (999.0, f"{R}999"),
        (-250_000.0, f"-{R}2.5 L"),
    ],
)
def test_compact_rupees_use_lakh_and_crore(value, text):
    assert fmt.rupees_compact(value) == text


def test_rupees_and_percentages():
    assert fmt.rupees(-500.0) == f"-{R}500.00"
    assert fmt.percent(0.1234) == "12.34%"
    assert fmt.percent(0.05, 1, signed=True) == "+5.0%"


def test_undefined_and_infinite_figures():
    assert fmt.rupees(math.nan) == "n/a"
    assert fmt.number(math.inf) == "&#8734;"
    assert fmt.figure("max_drawdown_recovery", None) == "n/a"


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(2_700, "45 min"), (12_000, "3 h 20 min"), (2 * 86_400 + 4 * 3_600, "2 d 4 h")],
)
def test_durations(seconds, text):
    assert fmt.duration(seconds) == text


def test_figures_are_formatted_by_their_unit():
    assert fmt.figure("win_rate", 0.5) == "50.00%"
    assert fmt.figure("trades", 1234) == "1,234"
    assert fmt.figure("longest_drawdown_days", 12.25) == "12.2 d"
    assert fmt.figure("ruined", False) == "no"


def test_rupee_axis_labels_share_a_unit_and_tell_ticks_apart():
    labels = fmt.rupee_ticks([100_000.0, 102_500.0, 105_000.0])
    assert labels == [f"{R}1.000 L", f"{R}1.025 L", f"{R}1.050 L"]


def test_percent_axis_labels():
    assert fmt.percent_ticks([-0.2, -0.1, 0.0]) == ["-20%", "-10%", "0%"]
    assert fmt.percent_ticks([-0.005, -0.0025, 0.0]) == ["-0.50%", "-0.25%", "0.00%"]


def test_value_axes_end_on_round_ticks():
    low, high, ticks = charts.value_axis(97_000.0, 112_000.0)
    # 15,000 over five steps is 3,000, rounded up to 5,000.
    assert (low, high) == (95_000.0, 115_000.0)
    assert ticks == [95_000.0, 100_000.0, 105_000.0, 110_000.0, 115_000.0]


def test_a_flat_value_axis_is_widened():
    low, high, _ = charts.value_axis(100.0, 100.0)
    assert low < 100.0 < high
