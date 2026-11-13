"""The calendar heatmaps: monthly returns (years down, months across) and weekly
returns of one year at a time.

A cell's colour mixes the gain or loss colour into the neutral one in proportion to the
size of its return against the largest in the map (`HEAT_SHARE` at the largest), so it
follows the page's theme. Without script every year of the weekly map is shown, each
under its heading; with it, a year picker shows one year, the last by default.
"""

import math
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd

from .. import _format as fmt

#: The share of the gain or loss colour in the cell of the largest return.
HEAT_SHARE = 0.88
#: The smallest return the colour scale is stretched to, so a flat map stays pale.
HEAT_FLOOR = 0.0001
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def heat(value: float, bound: float) -> str:
    """The CSS background of a cell with return `value` on a map whose largest return
    is `bound`; transparent for an unknown return."""
    if not math.isfinite(value):
        return "transparent"
    share = round(min(abs(value) / bound, 1.0) * HEAT_SHARE * 100)
    target = "var(--heat-pos)" if value >= 0 else "var(--heat-neg)"
    return f"color-mix(in oklab, {target} {share}%, var(--heat-mid))"


def _bound(values: np.ndarray) -> float:
    known = np.abs(values[np.isfinite(values)])
    return max(HEAT_FLOOR, float(known.max())) if len(known) else HEAT_FLOOR


def monthly(table: pd.DataFrame) -> str:
    """The monthly returns table (rows years, columns 1 to 12, fractions, NaN for a
    month with no session) as a heatmap; a month with no session is blank."""
    bound = _bound(table.to_numpy(dtype=float))
    head = "".join(f"<div class='hm-month'>{name[0]}</div>" for name in MONTHS)
    rows = []
    for year, values in table.iterrows():
        cells = []
        for name, value in zip(MONTHS, values.to_numpy(dtype=float), strict=True):
            known = math.isfinite(value)
            shown = f"{value * 100:.1f}" if known else ""
            cells.append(
                f"<div class='hm-cell nums{'' if known else ' none'}' "
                f"title='{name} {year}: {fmt.percent(value)}' "
                f"style='background:{heat(value, bound)}'>{shown}</div>"
            )
        rows.append(f"<div class='hm-year nums'>{year}</div>{''.join(cells)}")
    return (
        f"<div class='hm-scroll'><div class='hm-grid'><div></div>{head}{''.join(rows)}</div></div>"
    )


class Week(NamedTuple):
    """One week of the weekly map: ISO week number, the week's last day as "Jan 04"
    and its compounded return (a fraction, NaN when unknown)."""

    week: int
    label: str
    value: float


def _weeks(weeks: Sequence[Week]) -> str:
    bound = _bound(np.array([w.value for w in weeks], dtype=float))
    cells = []
    for w in weeks:
        known = math.isfinite(w.value)
        shown = f"{w.value * 100:.1f}" if known else ""
        cells.append(
            f"<div class='wk-cell{'' if known else ' none'}' "
            f"title='Week {w.week} ({w.label}): {fmt.percent(w.value)}' "
            f"style='background:{heat(w.value, bound)}'><span class='wk-value nums'>{shown}</span>"
            f"<span class='wk-label nums'>{w.label}</span></div>"
        )
    return f"<div class='wk-grid'>{''.join(cells)}</div>"


def weekly(by_year: Mapping[int, Sequence[Week]]) -> str:
    """The weekly map, a block per ISO year, each coloured on its own scale; a year
    picker (script) shows the last year first."""
    if not by_year:
        return "<p class='empty'>No data</p>"
    years = sorted(by_year)
    options = "".join(
        f"<option value='{y}'{' selected' if y == years[-1] else ''}>{y}</option>" for y in years
    )
    blocks = []
    for year in years:
        weeks = by_year[year]
        on = " on" if year == years[-1] else ""
        blocks.append(
            f"<div class='wk-year{on}' data-year='{year}'>"
            f"<div class='wk-bar'><div class='eyebrow'>{len(weeks)} weeks &#183; week-ending "
            f"return<span class='wk-head'> &#183; {year}</span></div></div>{_weeks(weeks)}</div>"
        )
    picker = f"<select class='wk-pick js-only nums' aria-label='Select year'>{options}</select>"
    return f"<div class='wk' data-weekly>{picker}{''.join(blocks)}</div>"


__all__ = ["HEAT_SHARE", "MONTHS", "Week", "heat", "monthly", "weekly"]
