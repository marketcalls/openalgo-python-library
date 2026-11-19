"""An objective over two parameters as inline SVG (`_blocks.GridBlock`), in the style of
`_charts`:

    heatmap   a cell per combination, its objective written in it and its colour mixed
              from the gain or loss colour by how far it is from break-even, against the
              farthest in the grid (the tearsheet's monthly heatmap colours,
              `report._dashboard.heatmaps.heat`); a marked cell (the chosen or the best
              setting) has a bold outline and its label under the grid
    surface   the same grid in an isometric view, original to this package: each
              combination a column standing on the break-even plane, as tall (or as deep
              below it) as its objective is from break-even, coloured as the heatmap;
              drawn from the back to the front, so a nearer column hides a farther one,
              its top face lit and its two front faces shaded; a marked column outlined

Both read without colour: the heatmap writes each figure that fits its cell (short, as
"-20.3k", when the whole one does not, and none in a cell too small, `_chart_text`), the
surface shows it as a height, and every cell and column carries a `<title>` with its
parameters and figure. The heatmap grows taller with its rows, and a crowded axis shows
every few values so that none meets another.
"""

from __future__ import annotations

import math

import numpy as np

from ..report import _format as fmt
from ..report._dashboard.bars import AXIS_TEXT, MUTED_TEXT, Margin, frame
from ..report._dashboard.heatmaps import heat
from ._blocks import GridBlock
from ._chart_text import Label, placed, written
from ._charts import HEIGHT, WIDTH

#: The heatmap's margin: room for the y values on the left and the x values below.
HEATMAP_MARGIN = Margin(14, 12, 40, 64)
#: The height of a heatmap row while the chart grows with its rows, pixels: a written
#: figure's line and some room; past `TALLEST_HEATMAP` the rows grow shorter instead.
ROW_HEIGHT = 18
TALLEST_HEATMAP = 720
#: The smallest distance from break-even the colour scale is stretched to, so a flat
#: grid stays pale.
FLAT = 1e-12
#: The isometric directions: a step along x goes right and down, along y left and down
#: (cos 30 and sin 30 degrees).
ISO_X, ISO_Y = math.cos(math.pi / 6), math.sin(math.pi / 6)
#: The surface's chart height, pixels: taller than the other charts, since an isometric
#: plane takes half as much height as width.
SURFACE_HEIGHT = 420
#: The span from the deepest to the tallest column, a share of the chart's height.
TALLEST_SHARE = 0.36
#: The surface's margin inside the chart, pixels, the room its labels take on the right
#: of the plane, and below it.
SURFACE_PAD = 20
LABEL_ROOM = 48
LABEL_BELOW = 30


def _bound(block: GridBlock) -> float:
    gaps = np.abs(block.values - block.level)
    known = gaps[np.isfinite(gaps)]
    return max(FLAT, float(known.max())) if len(known) else FLAT


def _tip(block: GridBlock, column: int, row: int) -> str:
    value = block.values[row, column]
    return (
        f"{fmt.text(block.x)} {fmt.text(block.columns[column])}, {fmt.text(block.y)} "
        f"{fmt.text(block.rows[row])}: {fmt.number(value)}"
    )


def _marks(block: GridBlock) -> dict[tuple[int, int], str]:
    return {(column, row): label for column, row, label in block.marked}


def heatmap_height(rows: int) -> int:
    """The heatmap's chart height for `rows` rows, pixels (`ROW_HEIGHT`): the other
    charts' height at least, `TALLEST_HEATMAP` at most."""
    margin = HEATMAP_MARGIN
    wanted = margin.top + margin.bottom + rows * ROW_HEIGHT
    return min(max(HEIGHT, wanted), TALLEST_HEATMAP)


def heatmap_chart(block: GridBlock) -> str:
    """The heatmap of `block` with its key."""
    margin = HEATMAP_MARGIN
    count_x, count_y = len(block.columns), len(block.rows)
    chart_height = heatmap_height(count_y)
    width = WIDTH - margin.left - margin.right
    height = chart_height - margin.top - margin.bottom
    cell_w, cell_h = width / count_x, height / count_y
    bound, marks, marked = _bound(block), [], _marks(block)
    for row in range(count_y):
        for column in range(count_x):
            value = block.values[row, column]
            x, y = column * cell_w, (count_y - 1 - row) * cell_h
            css = "heat-cell cell-marked" if (column, row) in marked else "heat-cell"
            shown = written(value, cell_w, cell_h)
            marks.append(
                f"<g><title>{_tip(block, column, row)}</title><rect class='{css}' "
                f"x='{x:.1f}' y='{y:.1f}' width='{cell_w:.1f}' height='{cell_h:.1f}' "
                f"style='fill:{heat(value - block.level, bound)}'/>"
                f"<text x='{x + cell_w / 2:.1f}' y='{y + cell_h / 2 + 3:.1f}' "
                f"text-anchor='middle' class='cell-text'>{shown}</text></g>"
            )
    marks += _heatmap_axes(block, cell_w, cell_h, width, height)
    svg = frame(WIDTH, chart_height, margin, fmt.text(block.title), "".join(marks))
    return f"{svg}<p class='key'>{_key(block)}</p>"


def _heatmap_axes(
    block: GridBlock, cell_w: float, cell_h: float, width: float, height: float
) -> list[str]:
    count_y = len(block.rows)
    across = [
        Label((k + 0.5) * cell_w, height + 14, "middle", value)
        for k, value in enumerate(block.columns)
    ]
    down = [
        Label(-8, (count_y - 1 - k + 0.5) * cell_h + 3, "end", value)
        for k, value in enumerate(block.rows)
    ]
    marks = [_axis_text(label) for label in placed(across) + placed(down)]
    marks.append(
        f"<text x='{width:.1f}' y='{height + 30:.1f}' text-anchor='end' "
        f"style='{MUTED_TEXT}'>{fmt.text(block.x)}</text>"
        f"<text x='-8' y='-4' text-anchor='end' style='{MUTED_TEXT}'>{fmt.text(block.y)}</text>"
    )
    return marks


def _axis_text(label: Label) -> str:
    anchor = "" if label.anchor == "start" else f" text-anchor='{label.anchor}'"
    return (
        f"<text x='{label.x:.1f}' y='{label.y:.1f}'{anchor} style='{AXIS_TEXT}'>"
        f"{fmt.text(label.text)}</text>"
    )


def _key(block: GridBlock) -> str:
    marked = "; ".join(
        f"outlined: the {fmt.text(label)} setting ({fmt.text(block.x)} "
        f"{fmt.text(block.columns[column])}, {fmt.text(block.y)} {fmt.text(block.rows[row])})"
        for column, row, label in block.marked
    )
    level = f"{fmt.number(block.level)}, the value of a run that makes nothing"
    if block.surface:
        text = f"Each column stands above or below {level}, as tall as its figure is from it"
    else:
        text = f"Each cell is coloured by how far its figure is above or below {level}"
    return f"{text}{'; ' + marked if marked else ''}."


def _corner(u: float, v: float, w: float, cell: float) -> tuple[float, float]:
    """The isometric point of grid position (u, v) at height `w` pixels."""
    return (u - v) * ISO_X * cell, (u + v) * ISO_Y * cell - w


def _face(points: list[tuple[float, float]], css: str, fill: str) -> str:
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f"<polygon class='{css}' points='{path}' style='fill:{fill}'/>"


def _column(block: GridBlock, column: int, row: int, cell: float, scale: float) -> str:
    """One combination's column: its two front faces and its top face."""
    value = block.values[row, column]
    if not math.isfinite(value):
        return ""
    w = (value - block.level) * scale
    fill = heat(value - block.level, _bound(block))
    u, v = float(column), float(row)
    top = [_corner(u, v, w, cell), _corner(u + 1, v, w, cell)]
    top += [_corner(u + 1, v + 1, w, cell), _corner(u, v + 1, w, cell)]
    left = [_corner(u, v + 1, 0, cell), _corner(u + 1, v + 1, 0, cell)]
    left += [_corner(u + 1, v + 1, w, cell), _corner(u, v + 1, w, cell)]
    right = [_corner(u + 1, v, 0, cell), _corner(u + 1, v + 1, 0, cell)]
    right += [_corner(u + 1, v + 1, w, cell), _corner(u + 1, v, w, cell)]
    css = "surf-top cell-marked" if (column, row) in _marks(block) else "surf-top"
    return (
        f"<g><title>{_tip(block, column, row)}</title>{_face(left, 'surf-left', fill)}"
        f"{_face(right, 'surf-right', fill)}{_face(top, css, fill)}</g>"
    )


def _extent(block: GridBlock) -> tuple[float, float]:
    """(how far the tallest column rises above break-even, how far the deepest falls
    below it), in the objective's units, zero or more."""
    gaps = block.values - block.level
    known = gaps[np.isfinite(gaps)]
    if not len(known):
        return 0.0, 0.0
    return max(0.0, float(known.max())), max(0.0, -float(known.min()))


def surface_chart(block: GridBlock) -> str:
    """The isometric surface of `block` with its key (module docstring)."""
    count_x, count_y = len(block.columns), len(block.rows)
    rise, fall = _extent(block)
    tall = TALLEST_SHARE * SURFACE_HEIGHT
    scale = tall / max(rise + fall, FLAT)
    room = WIDTH - 2 * SURFACE_PAD - LABEL_ROOM
    across = room / ((count_x + count_y) * ISO_X)
    deep = SURFACE_HEIGHT - 2 * SURFACE_PAD - tall - LABEL_BELOW
    cell = min(across, deep / ((count_x + count_y) * ISO_Y))
    order = sorted(
        ((column, row) for column in range(count_x) for row in range(count_y)),
        key=lambda place: (place[0] + place[1], place[0]),
    )
    base = [_corner(0, 0, 0, cell), _corner(count_x, 0, 0, cell)]
    base += [_corner(count_x, count_y, 0, cell), _corner(0, count_y, 0, cell)]
    plane = " ".join(f"{x:.1f},{y:.1f}" for x, y in base)
    marks = [f"<polygon class='surf-base' points='{plane}'/>"]
    marks += [_column(block, column, row, cell, scale) for column, row in order]
    marks += _surface_axes(block, cell)
    spare = room - (count_x + count_y) * ISO_X * cell
    left = SURFACE_PAD + spare / 2 + count_y * ISO_X * cell
    top = SURFACE_PAD + rise * scale
    margin = Margin(int(round(top)), 0, 0, int(round(left)))
    svg = frame(WIDTH, SURFACE_HEIGHT, margin, fmt.text(block.title), "".join(marks))
    return f"{svg}<p class='key'>{_key(block)}</p>"


def _surface_axes(block: GridBlock, cell: float) -> list[str]:
    """The x values along the front-right edge of the plane and the y values along its
    front-left edge, and the two parameters' names."""
    count_x, count_y = len(block.columns), len(block.rows)
    labels = []
    for k, value in enumerate(block.rows):
        x, y = _corner(count_x, k + 0.5, 0, cell)
        labels.append(Label(x + 6, y + 4, "start", value))
    for k, value in enumerate(block.columns):
        x, y = _corner(k + 0.5, count_y, 0, cell)
        labels.append(Label(x + 4, y + 12, "start", value))
    marks = [_axis_text(label) for label in placed(labels)]
    x_name = _corner(count_x / 2, count_y, 0, cell)
    y_name = _corner(count_x, count_y / 2, 0, cell)
    marks.append(
        f"<text x='{x_name[0] - 8:.1f}' y='{x_name[1] + 26:.1f}' text-anchor='end' "
        f"style='{MUTED_TEXT}'>{fmt.text(block.x)}</text>"
        f"<text x='{y_name[0] + 30:.1f}' y='{y_name[1] + 20:.1f}' "
        f"style='{MUTED_TEXT}'>{fmt.text(block.y)}</text>"
    )
    return marks


def grid_chart(block: GridBlock) -> str:
    """The heatmap or the surface of `block`, as it asks."""
    return surface_chart(block) if block.surface else heatmap_chart(block)


__all__ = ["grid_chart", "heatmap_chart", "surface_chart"]
