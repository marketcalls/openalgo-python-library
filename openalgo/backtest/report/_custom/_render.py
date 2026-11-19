"""How a custom tearsheet's user blocks become the body of a section: each block's data
is computed (a callable called once, with the result being rendered), checked, and drawn
by the dashboard's own code into a card of the page; the charts are in `_charts`, the
tables, tiles, text and fragments here.

Blocks are placed by `width`: a full block takes a row; consecutive half blocks share
the page's two-column grid and consecutive third blocks its three-column grid, wrapping
as the page's own grids do. Every block's HTML is ASCII (`_text.ascii_only`).
"""

import itertools
import math
import re
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pandas as pd

from .. import _format as fmt
from .._dashboard import cards, heatmaps
from .._dashboard.layout import SMALL, card, label, stack
from .._dashboard.trades import PAGER, sort_key
from . import _charts
from ._blocks import Block, prepared
from ._charts import Drawn, category
from ._data import called
from ._fragment import checked
from ._text import ascii_only, escape

#: The grid a run of blocks of one width shares.
GRIDS = {"half": "grid2", "third": "grid3"}
#: What an empty block says, as the page's own charts do.
NO_DATA = "<p class='empty'>No data</p>"
#: A blank line: where a text block starts a new paragraph.
PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


class Placed(NamedTuple):
    """Where a section's blocks are drawn: the run's session dates (the line charts'
    axis), the result their callables receive, the section's name (for errors) and
    the anchor its blocks' anchors start with."""

    sessions: pd.DatetimeIndex
    subject: object
    section: str
    anchor: str


def heatmap(block: Block, frame: pd.DataFrame, placed: Placed, anchor: str) -> Drawn:
    """A table of the rows and columns, each cell in the page's heat colours."""
    if frame.empty:
        return Drawn(NO_DATA)
    bound = heatmaps.heat_bound(frame.to_numpy(dtype=float))

    def shown(value: float) -> str:
        return fmt.percent(value, 1) if block.percent else fmt.number(value)

    head = "".join(f"<th class='r'>{fmt.text(category(c))}</th>" for c in frame.columns)
    rows = []
    for name, values in frame.iterrows():
        cells = "".join(
            f"<td class='r nums' title='{fmt.text(category(name))}, {fmt.text(category(column))}: "
            f"{shown(value)}' "
            f"style='background:{heatmaps.heat(value, bound)}'>"
            f"{shown(value) if math.isfinite(value) else ''}</td>"
            for column, value in values.items()
        )
        rows.append(f"<tr><td class='l nums'>{fmt.text(category(name))}</td>{cells}</tr>")
    table = (
        f"<table class='tbl'><thead><tr><th class='l'></th>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    return Drawn(f"<div class='scroll'>{table}</div>")


#: The kinds of a table column whose cells are numbers: whole numbers only, or any.
NUMBER_KINDS = ("whole", "number")


def _cell(value: object, kind: str, percent: bool) -> tuple[str, object]:
    """A table cell's text and sort value, by its column's kind."""
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return fmt.MISSING, None
    if kind in NUMBER_KINDS:
        number = float(value)
        if percent:
            return fmt.percent(number), number
        return fmt.number(number, 0 if kind == "whole" else 2), number
    if kind == "time":
        return fmt.moment(value), pd.Timestamp(value)
    if kind == "flag":
        return ("yes" if value else "no"), str(bool(value))
    return fmt.text(value), str(value)


def _kind(series: pd.Series) -> str:
    """A column's kind: "flag", "whole" (numbers, every known one whole), "number",
    "time" or "text"."""
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "flag"
    if pd.api.types.is_numeric_dtype(dtype):
        values = series.to_numpy(dtype=float)
        known = values[np.isfinite(values)]
        return "whole" if np.array_equal(known, np.round(known)) else "number"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "time"
    return "text"


def table(block: Block, shown: pd.DataFrame, placed: Placed, anchor: str) -> Drawn:
    """The DataFrame in the page's table style, sortable and paged as asked; `shown`
    holds its index as its first columns already (`_data.table_frame`)."""
    if shown.empty:
        return Drawn(NO_DATA)
    kinds = [_kind(shown[column]) for column in shown.columns]
    sortable, rows = block.option("sortable"), block.option("rows")
    sort = " s" if sortable else ""
    head = "".join(
        f"<th class='{'r' if kind in NUMBER_KINDS else 'l'}{sort}'>{fmt.text(column)}</th>"
        for column, kind in zip(shown.columns, kinds, strict=True)
    )
    body = []
    for i, values in enumerate(shown.itertuples(index=False, name=None)):
        cells = []
        for value, kind in zip(values, kinds, strict=True):
            text, key = _cell(value, kind, block.percent)
            css = "r nums" if kind in NUMBER_KINDS else "l"
            cells.append(f"<td class='{css}' data-v='{sort_key(key)}'>{text}</td>")
        body.append(f"<tr data-i='{i}'>{''.join(cells)}</tr>")
    hooks = (" data-sort" if sortable or rows else "") + (f" data-page='{rows}'" if rows else "")
    markup = (
        f"<div class='scroll'><table class='tbl hover'{hooks}><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>{PAGER if rows else ''}"
    )
    return Drawn(markup)


def _tile_text(value: object, block: Block) -> str:
    if isinstance(value, str):
        return fmt.text(value)
    if fmt.is_missing(value):
        return fmt.MISSING
    if block.percent:
        return cards.text(value, "pct")
    if block.option("rupees"):
        return cards.text(value, "rupees")
    number = float(value)
    return cards.text(number, "int" if number.is_integer() else "num")


def tiles(block: Block, values: dict, placed: Placed, anchor: str) -> Drawn:
    """The page's stat tiles, one per label."""
    if not values:
        return Drawn(NO_DATA)
    cells = "".join(
        f"<div class='stat'><div class='eyebrow trunc' title='{fmt.text(name)}'>"
        f"{fmt.text(name)}</div><div class='stat-value nums'>{_tile_text(value, block)}</div>"
        "</div>"
        for name, value in values.items()
    )
    return Drawn(f"<div class='stats'>{cells}</div>")


def text(block: Block, content: str, placed: Placed, anchor: str) -> Drawn:
    """Paragraphs of plain text, escaped."""
    paragraphs = [part for part in PARAGRAPH_BREAK.split(content.strip()) if part]
    if not paragraphs:
        return Drawn(NO_DATA)
    return Drawn(
        "<div class='prose'>" + "".join(f"<p>{escape(p)}</p>" for p in paragraphs) + "</div>"
    )


def fragment(block: Block, content: str, placed: Placed, anchor: str) -> Drawn:
    """The user's HTML or SVG, checked against the allow list, its ids prefixed."""
    if not content.strip():
        return Drawn(NO_DATA)
    where = f"{block.where} in section {placed.section!r}"
    return Drawn(checked(content, where=where, svg_only=block.kind == "svg", prefix=anchor))


def _chart(draw: Callable) -> Callable[[Block, object, Placed, str], Drawn]:
    def drawn(block: Block, data: object, placed: Placed, anchor: str) -> Drawn:
        return draw(block, data, placed.sessions, anchor)

    return drawn


#: How each kind is drawn.
DRAW: dict[str, Callable[[Block, object, Placed, str], Drawn]] = {
    "line": _chart(_charts.line),
    "bar": _chart(_charts.bar),
    "histogram": _chart(_charts.histogram),
    "scatter": _chart(_charts.scatter),
    "heatmap": heatmap,
    "table": table,
    "tiles": tiles,
    "text": text,
    "html": fragment,
    "svg": fragment,
}


def _card(block: Block, drawn: Drawn) -> str:
    """A drawn block in its card (tiles under their label, as the page's own are)."""
    subtitle = " &#183; ".join(part for part in (escape(block.subtitle), drawn.note) if part)
    if block.kind == "tiles":
        heading = escape(block.title) if block.title else ""
        heading += f" &#183; {subtitle}" if heading and subtitle else ""
        return (label(heading) if heading else "") + drawn.html
    if block.title:
        return card(escape(block.title), subtitle, drawn.html)
    sub = f"<p class='card-sub'>{subtitle}</p>" if subtitle else ""
    return (
        f"<div class='card'><header class='card-head'>{sub}</header>"
        f"<div class='card-body'>{drawn.html}</div></div>"
    )


def draw(block: Block, placed: Placed, anchor: str) -> tuple[str, bool]:
    """One block's card and whether it uses the series colours. Raises ConfigError,
    naming the block and its section, for data it does not take."""
    where = f"{block.where} in section {placed.section!r}"
    data = prepared(block, called(block.data, placed.subject, where), where)
    drawn = DRAW[block.kind](block, data, placed, anchor)
    return ascii_only(_card(block, drawn)), drawn.several


def body(blocks: tuple[Block, ...], placed: Placed) -> tuple[str, bool]:
    """The blocks of a section, placed in rows by width, and whether any uses the
    series colours. Block k's anchor is the section's anchor and k (from 1)."""
    cells = [
        (block.width, *draw(block, placed, f"{placed.anchor}-{k}"))
        for k, block in enumerate(blocks, start=1)
    ]
    rows = []
    for width, run in itertools.groupby(cells, key=lambda cell: cell[0]):
        drawn = [cell[1] for cell in run]
        if width in GRIDS:
            rows.append(f"<div class='{GRIDS[width]}'>{''.join(drawn)}</div>")
        else:
            rows.extend(drawn)
    markup = stack([("" if i == 0 else SMALL, row) for i, row in enumerate(rows)])
    return markup, any(several for _, _, several in cells)


__all__ = ["DRAW", "Placed", "body", "draw"]
