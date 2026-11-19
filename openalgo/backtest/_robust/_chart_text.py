"""Text that fits a chart of the robustness report: a figure written in a cell only when
it fits, short when the whole one does not, and axis values placed so that none meets
another (`_charts_grid`).

Sizes are those of the charts' 10-pixel mono type (`report._dashboard.bars.AXIS_TEXT`, the
`cell-text` style): a character 0.6 of an em wide, a line 12 pixels tall, in the units of
a chart's view box.

A short figure (`short`) keeps `digits` significant digits and names the largest of a
crore, a lakh and a thousand it reaches: -20,308.43 is "-20.3k" to three digits and
"-20k" to two, 1,52,000 is "1.52L", 0.4567 is "0.457". A figure that would need an
exponent has no short form.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..report import _format as fmt

#: The width of a character and the height of a line of 10-pixel mono type, pixels.
CHARACTER, LINE = 6.0, 12.0
#: The room kept between a written figure and its cell's edges, pixels, on each side.
CELL_PAD = 1.0
#: The room kept between two axis values on one line, pixels.
LABEL_GAP = 2.0
#: The suffixes of a short figure, largest first: a crore, a lakh and a thousand.
SHORT_UNITS = ((1e7, "Cr"), (1e5, "L"), (1e3, "k"))


def short(value: float, digits: int) -> str:
    """`value` to `digits` significant digits with its unit (module docstring); "" for
    an undefined figure or one that would need an exponent."""
    if not math.isfinite(value):
        return ""
    rounded = float(f"{value:.{digits}g}")
    unit, suffix = next(
        ((size, name) for size, name in SHORT_UNITS if abs(rounded) >= size), (1.0, "")
    )
    shown = f"{rounded / unit:.{digits}g}"
    return "" if "e" in shown else shown + suffix


def width_of(text: str) -> float:
    """The width of `text` in the charts' mono type, pixels."""
    return len(text) * CHARACTER


def written(value: float, width: float, height: float) -> str:
    """The figure a cell of `width` by `height` pixels writes: the whole figure
    (`fmt.number`), else its short form to three, then two significant digits, the first
    that fits; "" when none fits, the cell is shorter than a line, or the value is
    undefined."""
    if not math.isfinite(value) or height < LINE:
        return ""
    room = width - 2 * CELL_PAD
    for shown in (fmt.number(value), short(value, 3), short(value, 2)):
        if shown and width_of(shown) <= room:
            return shown
    return ""


@dataclass(frozen=True)
class Label:
    """An axis value to draw: its anchor point (`x`, the baseline `y`), its
    `text-anchor` ("start", "middle" or "end") and its text."""

    x: float
    y: float
    anchor: str
    text: str

    def span(self) -> tuple[float, float]:
        """(left, right) of its text, pixels."""
        wide = width_of(self.text)
        left = {"middle": self.x - wide / 2, "end": self.x - wide}.get(self.anchor, self.x)
        return left, left + wide

    def meets(self, other: Label) -> bool:
        """Whether its text and `other`'s would overlap."""
        (a_left, a_right), (b_left, b_right) = self.span(), other.span()
        return (
            abs(self.y - other.y) < LINE
            and a_left < b_right + LABEL_GAP
            and b_left < a_right + LABEL_GAP
        )


def placed(labels: list[Label]) -> list[Label]:
    """The labels to draw, in order: each one that meets none drawn before it, so a
    crowded axis shows every few values, evenly when they are as long."""
    kept: list[Label] = []
    for label in labels:
        if not any(label.meets(other) for other in kept):
            kept.append(label)
    return kept


__all__ = [
    "CELL_PAD",
    "CHARACTER",
    "LABEL_GAP",
    "LINE",
    "SHORT_UNITS",
    "Label",
    "placed",
    "short",
    "width_of",
    "written",
]
