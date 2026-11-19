"""The histogram of the robustness pages and the optimizer's page (`_robust._charts`), on
hand-built blocks: its axis values stay inside the chart, and its key counts the values
by what they are (runs, or the cross-validation's splits).

The chart is 680 by 300 view-box units with margins of 22 (top), 16 (right), 30 (bottom)
and 52 (left), so its plot is 612 wide and its axis values sit on the baseline 264. Axis
text is 10-pixel mono type: a character is 6 wide.
"""

import re

import numpy as np

from openalgo.backtest._robust._blocks import REAL, HistogramBlock, Mark
from openalgo.backtest._robust._charts import histogram_chart

#: The plot's width and the margins either side of it, view-box units.
PLOT, LEFT, RIGHT = 612.0, 52.0, 16.0
#: The width of a character of axis text.
CHARACTER = 6.0
#: An axis value along the bottom of the chart: x, anchor and text.
AXIS = r"<text x='(-?[\d.]+)' y='264' text-anchor='(\w+)' style='[^']*'>([^<]*)</text>"
#: Net profits up to 1,00,000 rupees: the axis runs from 0 to 1,00,000 in steps of 20,000.
PROFITS = np.array([0.0, 25_000.0, 50_000.0, 1_00_000.0])


def histogram(counted: str = "runs") -> str:
    marks = (Mark(50_000.0, "the best", REAL),)
    return histogram_chart(HistogramBlock("Objectives", "", PROFITS, False, marks, counted))


def spans(chart: str) -> list[tuple[str, float, float]]:
    """(text, left, right) of each axis value along the bottom, from its anchor."""
    found = []
    for x, anchor, text in re.findall(AXIS, chart):
        width = len(text) * CHARACTER
        left = {"middle": float(x) - width / 2, "end": float(x) - width}.get(anchor, float(x))
        found.append((text, left, left + width))
    return found


def test_every_axis_value_stays_inside_the_chart():
    shown = spans(histogram())
    assert [text for text, _, _ in shown] == [
        "0.00",
        "20,000.00",
        "40,000.00",
        "60,000.00",
        "80,000.00",
        "1,00,000.00",
    ]
    for text, left, right in shown:
        assert left >= -LEFT and right <= PLOT + RIGHT, text


def test_the_last_axis_value_ends_at_the_right_end_of_the_plot():
    # "1,00,000.00" is 66 wide: centred on the plot's right end, 612, it would reach 645,
    # past the chart's edge at 628, so it ends at 612 instead; the others stay centred.
    found = re.findall(AXIS, histogram())
    assert found[-1] == ("612.0", "end", "1,00,000.00")
    assert {anchor for _, anchor, _ in found[:-1]} == {"middle"}


def test_a_mark_at_the_right_end_keeps_its_label_inside_the_chart():
    # "the best", 48 wide, marks 1,00,000, the plot's right end: it ends there, as the
    # last axis value does, instead of reaching 636.
    marks = (Mark(1_00_000.0, "the best", REAL),)
    chart = histogram_chart(HistogramBlock("Objectives", "", PROFITS, False, marks))
    assert "<text x='612.0' y='-6' text-anchor='end' " in chart


def test_the_key_counts_the_values_as_runs_by_default():
    assert "Bars: how many of the 4 runs fall in each bin." in histogram()


def test_the_key_counts_the_values_as_what_they_are():
    chart = histogram("splits")
    assert "Bars: how many of the 4 splits fall in each bin." in chart
    assert " runs " not in chart
