"""Report text: Indian digit grouping, rupees, percentages, durations and figures by
their unit."""

import math

import pytest

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
        (-0.004, "0.00"),  # a loss below half a paisa shows as zero, unsigned
        (-0.006, "-0.01"),  # 0.6 paisa rounds to one paisa and keeps its sign
        (-0.4, "0"),
    ],
)
def test_a_negative_amount_that_rounds_to_zero_has_no_sign(value, text):
    decimals = 0 if text == "0" else 2
    assert fmt.indian_digits(value, decimals) == text


def test_a_rupee_loss_that_rounds_to_zero_has_no_sign():
    assert fmt.rupees(-0.004) == f"{R}0.00"


def test_rupees_and_percentages():
    assert fmt.rupees(-500.0) == f"-{R}500.00"
    assert fmt.rupees(1_23_456.0, 0) == f"{R}1,23,456"
    assert fmt.percent(0.1234) == "12.34%"
    assert fmt.percent(0.05, 1) == "5.0%"


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
