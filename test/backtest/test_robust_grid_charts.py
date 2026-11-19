"""The charts of the searching checks (design 17, sections 2.3, 2.16 and 2.17) on hand-built
blocks: the heatmap and the isometric surface of an objective over two parameters, the
fold diagram, and the table with a small equity line a row. Each reads without colour:
the heatmap writes its figures, the surface draws them as heights, the folds tell in
sample from out of sample by fill."""

import re

import numpy as np
import pandas as pd
import pytest
from test_report_tearsheet import Structure

from openalgo.backtest._robust._blocks import FoldsBlock, FoldSpan, GridBlock, SparkTableBlock
from openalgo.backtest._robust._charts_folds import folds_chart, spark_table, sparkline
from openalgo.backtest._robust._charts_grid import heatmap_chart, surface_chart

#: Two parameters: fast 5, 10 and 15 across, slow 20 and 40 down; slow 40 with fast 15
#: never ran.
VALUES = np.array([[1.0, 2.5, -0.5], [0.5, 3.0, np.nan]])


def block(surface: bool = False) -> GridBlock:
    return GridBlock(
        "Objective by parameter",
        "",
        "fast",
        "slow",
        ("5", "10", "15"),
        ("20", "40"),
        VALUES,
        0.0,
        ((1, 1, "chosen"),),
        surface,
    )


def well_formed(html: str) -> bool:
    structure = Structure(html)
    return structure.errors == [] and structure.stack == [] and html.isascii()


def test_the_heatmap_has_a_cell_per_combination_with_its_figure_written():
    chart = heatmap_chart(block())
    assert chart.count("<rect class='heat-cell") == 6
    shown = re.findall(r"class='cell-text'>([^<]*)</text>", chart)
    assert sorted(shown) == sorted(["1.00", "2.50", "-0.50", "0.50", "3.00", ""])
    assert well_formed(chart)


def test_the_heatmap_outlines_the_marked_cell_and_names_it_in_its_key():
    chart = heatmap_chart(block())
    assert chart.count("cell-marked") == 1
    assert "outlined: the chosen setting (fast 10, slow 40)" in chart


def test_a_cell_reads_its_parameters_and_figure_on_hover():
    assert "<title>fast 15, slow 20: -0.50</title>" in heatmap_chart(block())


def test_a_cell_above_break_even_mixes_in_the_gain_colour_and_one_below_the_loss_colour():
    chart = heatmap_chart(block())
    assert "var(--heat-pos) 88%" in chart  # 3.00, the farthest from break-even
    assert "var(--heat-neg)" in chart
    assert "fill:transparent" in chart  # the cell that never ran


def test_the_surface_draws_three_faces_for_each_combination_that_ran():
    chart = surface_chart(block(surface=True))
    assert chart.count("class='surf-top") == 5
    assert chart.count("class='surf-left'") == 5 and chart.count("class='surf-right'") == 5
    assert chart.count("class='surf-base'") == 1
    assert chart.count("surf-top cell-marked") == 1
    assert well_formed(chart)


def test_the_surface_draws_a_taller_column_for_a_larger_figure():
    # A column's top face is lifted by its figure over the break-even plane: compare the
    # top corners of the 3.00 column with the 0.50 one beside it on the plane.
    chart = surface_chart(block(surface=True))
    tops = re.findall(r"<title>(fast \d+, slow \d+): [^<]*</title>(.*?)</g>", chart)
    heights = {}
    for name, faces in tops:
        left = re.search(r"class='surf-left' points='([^']*)'", faces).group(1)
        corners = [tuple(map(float, point.split(","))) for point in left.split()]
        heights[name] = corners[0][1] - corners[3][1]
    assert heights["fast 10, slow 40"] > heights["fast 5, slow 40"] > 0
    assert heights["fast 15, slow 20"] < 0


def test_the_fold_diagram_draws_each_folds_two_windows():
    dates = pd.DatetimeIndex(pd.date_range("2025-01-01", periods=20))
    spans = (FoldSpan("Fold 1", 0, 6, 7, 9), FoldSpan("Fold 2", 3, 9, 10, 12))
    chart = folds_chart(FoldsBlock("Folds", "", dates, spans))
    assert chart.count("class='fold-in'") == 2 and chart.count("class='fold-out'") == 2
    assert "<title>Fold 2, out of sample: 2025-01-11 to 2025-01-13</title>" in chart
    assert ">Fold 1</text>" in chart and well_formed(chart)


def test_a_sparkline_needs_two_points():
    assert sparkline(np.array([5.0])) == ""
    line = sparkline(np.array([0.0, 10.0, -5.0]))
    assert line.count("<polyline class='spark-line'") == 1 and well_formed(line)


def test_the_spark_table_puts_each_rows_line_in_its_column():
    table = spark_table(
        SparkTableBlock(
            "By fold",
            "",
            ("Run", "Net P&L"),
            (("Stitched", "Rs 10"), ("Fold 1", "Rs 4")),
            1,
            "Equity",
            (np.array([0.0, 10.0]), np.array([])),
        )
    )
    assert re.findall(r"<th class='[lr]'>([^<]*)</th>", table) == ["Run", "Equity", "Net P&amp;L"]
    assert table.count("class='bars spark'") == 1
    assert well_formed(table)


@pytest.mark.parametrize("draw", [heatmap_chart, surface_chart])
def test_a_single_row_grid_draws(draw):
    row = GridBlock(
        "One parameter", "", "day", "", ("0", "1", "2"), ("",), np.array([[1.0, 2.0, 3.0]]), 0.0, ()
    )
    assert well_formed(draw(row))


# ----- a large grid ---------------------------------------------------------------------

#: The width of a character of the charts' 10-pixel mono type, pixels (0.6 of an em).
CHARACTER = 6.0
#: The height of a line of that type, pixels.
LINE = 12.0
#: The text of an axis value, and of a figure written in a cell.
AXIS = (
    r"<text x='(-?[\d.]+)' y='(-?[\d.]+)'(?: text-anchor='(\w+)')? "
    r"style='[^']*text-faint\)'>([^<]*)</text>"
)
CELL = (
    r"<rect class='heat-cell[^']*' x='[\d.]+' y='[\d.]+' width='([\d.]+)' height='([\d.]+)'"
    r"[^>]*/><text[^>]*class='cell-text'>([^<]*)</text>"
)


def large(surface: bool = False) -> GridBlock:
    """20 by 20 combinations of a stop and a target as fractions ("0.0125" to "0.25"), the
    net profit of most -20,308.43 rupees, as bt.optimize's default objective gives them,
    and of the first five of the first row 1,234.50, 523.40, 3.25, -1,234.50 and none."""
    values = np.full((20, 20), -20_308.43)
    values[0, :5] = [1_234.5, 523.4, 3.25, -1_234.5, np.nan]
    stops = tuple(f"{0.0125 * (k + 1):g}" for k in range(20))
    return GridBlock("Net profit", "", "stop", "target", stops, stops, values, 0.0, (), surface)


def _spans(labels: list[tuple[float, float, str, str]]) -> list[tuple[float, float, float]]:
    """(left, right, baseline) of each label, from its anchor."""
    found = []
    for x, y, anchor, text in labels:
        width = len(text) * CHARACTER
        left = {"middle": x - width / 2, "end": x - width}.get(anchor, x)
        found.append((left, left + width, y))
    return found


def overlapping(labels: list[tuple[float, float, str, str]]) -> list[tuple[int, int]]:
    """The pairs of labels whose boxes overlap."""
    spans = _spans(labels)
    return [
        (a, b)
        for a in range(len(spans))
        for b in range(a + 1, len(spans))
        if abs(spans[a][2] - spans[b][2]) < LINE
        and spans[a][0] < spans[b][1]
        and spans[b][0] < spans[a][1]
    ]


def axis_labels(chart: str) -> list[tuple[float, float, str, str]]:
    return [(float(x), float(y), a, t) for x, y, a, t in re.findall(AXIS, chart)]


def test_a_large_heatmap_writes_each_figure_whole_short_or_not_at_all():
    chart = heatmap_chart(large())
    cells = re.findall(CELL, chart)
    assert len(cells) == 400
    for width, height, shown in cells:
        assert len(shown) * CHARACTER <= float(width), shown
        assert not shown or float(height) >= LINE, height
    # A cell is about 30 pixels wide, five characters less its padding: 3.25 fits whole,
    # 523.40 is written "523", 1,234.50 "1.2k" and -20,308.43 "-20k"; -1,234.50 needs
    # "-1.2k", five characters, so only its title holds it.
    shown = [text for _, _, text in cells]
    assert shown[:5] == ["1.2k", "523", "3.25", "", ""]
    assert set(shown[5:]) == {"-20k"}


def test_a_large_heatmap_grows_taller_with_its_rows():
    chart = heatmap_chart(large())
    height = float(re.search(r"viewBox='0 0 \d+ ([\d.]+)'", chart).group(1))
    assert height >= 20 * LINE * 1.5


@pytest.mark.parametrize("draw", [heatmap_chart, surface_chart])
def test_the_axis_values_of_a_large_grid_never_overlap(draw):
    labels = axis_labels(draw(large(draw is surface_chart)))
    assert labels
    assert overlapping(labels) == []
