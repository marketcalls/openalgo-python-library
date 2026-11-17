"""The trading cost block and the "MAE and MFE" subsection of the Trades section
(`sections_run.trades`).

The trading cost block is a row of tiles for the closed trades: their turnover, charges,
slippage, the trading cost (charges plus slippage), its share of gross P&L and its basis
points of that turnover. The subsection shows, for the closed trades, the excursion
tiles, a histogram of MAE percent and one of MFE percent, and each against the trade's
return as scatter charts, as the mode's parts ask. A trades table without the excursion
columns, or a run with no closed trade, gets one line saying so instead of empty charts.
"""

import pandas as pd

from .._excursion import EXCURSION_COLUMNS
from .._performance import Performance
from . import bars, cards
from .layout import View, card, label
from .scatter import Point, scatter

#: The parts of the subsection, in the order they are drawn.
SUBSECTION_PARTS = ("excursion_stats", "excursion_histograms", "excursion_scatter")
#: What the subsection says instead of charts.
NO_COLUMNS = (
    "No MAE and MFE to show: the trades table has no excursion columns "
    "(mae, mfe, mae_pct, mfe_pct)."
)
NO_CLOSED_TRADE = "No MAE and MFE to show: no trade closed."
#: Percent per unit fraction (the trade list's `_pct` columns are percent).
PERCENT = 100.0
#: (title, subtitle) of the cards of each excursion.
HISTOGRAMS = {
    "MAE": ("MAE Distribution", "Worst open loss of each closed trade, % of capital at work"),
    "MFE": ("MFE Distribution", "Best open profit of each closed trade, % of capital at work"),
}
SCATTERS = {
    "MAE": (
        "MAE vs Return",
        "Each closed trade: its MAE against its net return; circles won, crosses did not",
    ),
    "MFE": (
        "MFE vs Return",
        "Each closed trade: its MFE against its net return; circles won, crosses did not",
    ),
}


def trading_cost(view: View) -> str:
    """The trading cost block, or nothing when the mode does not show it."""
    if not view.has("trade_costs"):
        return ""
    tiles = cards.grid(cards.TRADING_COST, view.performance, None, view.bench_name)
    return label("Trading Cost") + tiles


def _unavailable(performance: Performance, listing: pd.DataFrame) -> str | None:
    """Why the subsection has nothing to draw, or None: no closed trade first (a run
    that never closed one has no excursion columns either), then no columns."""
    if listing.empty:
        return NO_CLOSED_TRADE
    if not all(name in performance.trades.columns for name in EXCURSION_COLUMNS):
        return NO_COLUMNS
    return None


def _points(listing: pd.DataFrame, column: str) -> list[Point]:
    """The closed trades as points of `column` ("mae_pct" or "mfe_pct") against the
    return, fractions."""
    return [
        Point(
            int(number),
            float(row[column]) / PERCENT,
            float(row["return_pct"]) / PERCENT,
            float(row["net_pnl"]) > 0,
        )
        for number, row in listing.iterrows()
    ]


def _pair(cells: list[str]) -> str:
    return f"<div class='grid2'>{''.join(cells)}</div>"


def _charts(view: View, listing: pd.DataFrame) -> list[str]:
    """The histograms and the scatter charts the mode shows."""
    blocks = []
    columns = {"MAE": "mae_pct", "MFE": "mfe_pct"}
    if view.has("excursion_histograms"):
        cells = []
        for name, column in columns.items():
            values = listing[column].to_numpy(dtype=float) / PERCENT
            chart = bars.histogram(values, label=f"{name} distribution", decimals=None)
            cells.append(card(*HISTOGRAMS[name], chart))
        blocks.append(_pair(cells))
    if view.has("excursion_scatter"):
        cells = [
            card(*SCATTERS[name], scatter(_points(listing, column), name=name))
            for name, column in columns.items()
        ]
        blocks.append(_pair(cells))
    return blocks


def mae_mfe(view: View, listing: pd.DataFrame) -> str:
    """The subsection for the closed trades of `listing` (the trade list), or nothing
    when the mode shows none of its parts."""
    if not any(view.has(part) for part in SUBSECTION_PARTS):
        return ""
    head = label("MAE and MFE")
    reason = _unavailable(view.performance, listing)
    if reason is not None:
        return f"<div id='mae-mfe'>{head}<p class='empty'>{reason}</p></div>"
    blocks = []
    if view.has("excursion_stats"):
        blocks.append(cards.grid(cards.EXCURSIONS, view.performance, None, view.bench_name))
    blocks += _charts(view, listing)
    body = "".join(
        block if i == 0 else f"<div class='gap-s'>{block}</div>" for i, block in enumerate(blocks)
    )
    return f"<div id='mae-mfe'>{head}{body}</div>"


__all__ = ["NO_CLOSED_TRADE", "NO_COLUMNS", "SUBSECTION_PARTS", "mae_mfe", "trading_cost"]
