"""The HTML tearsheet: one self-contained page for a strategy or a combined portfolio."""

from collections.abc import Mapping
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd

from ... import __version__
from ..errors import ConfigError
from . import _charts as charts
from . import _sections as sections
from . import _tables as tables
from ._combined import return_correlation
from ._format import text
from ._performance import Performance
from ._style import STYLESHEET

DEFAULT_TITLE = "Backtest"


def tearsheet(
    result_like: object, path: str | PathLike | None = None, *, title: str | None = None
) -> str:
    """Render a performance report as one self-contained HTML page.

    `result_like` is a `Performance` (from `openalgo.backtest.metrics.analyse`), or any
    object with a `performance` attribute holding one. Optional attributes of that
    object are read when present:

        name         the report title (overridden by `title=`)
        assumptions  what the engine assumed, shown as a plain list: a mapping gives
                     "key: value" lines, any other iterable one line per item
        strategies   a mapping {name: Performance} of the strategies of a combined
                     result; with two or more, a comparison and the correlation of their
                     session returns are added

    Sections: headline figures and fact lists, equity with drawdown underneath,
    monthly returns as a heat grid, yearly returns, the distribution of trade net P&L,
    the trade list (first and last rows with a count), costs, the strategy comparison
    (when given), assumptions, and notes with every definition used.

    The page has inline CSS and SVG, no script and no external resource; it follows
    the reader's light or dark setting. With `path` None the HTML is returned;
    otherwise it is written there as UTF-8 and the path is returned as a string.
    Raises ConfigError when no Performance can be found.
    """
    performance, name, assumptions, strategies = _resolve(result_like)
    page = render(
        performance,
        title=title or name or DEFAULT_TITLE,
        assumptions=sections.assumption_lines(assumptions),
        strategies=strategies,
    )
    if path is None:
        return page
    target = Path(path)
    target.write_text(page, encoding="utf-8")
    return str(target)


def _resolve(result_like: object) -> tuple[Performance, str | None, object, Mapping | None]:
    if isinstance(result_like, Performance):
        return result_like, None, None, None
    performance = getattr(result_like, "performance", None)
    if not isinstance(performance, Performance):
        raise ConfigError(
            "tearsheet needs a Performance (from openalgo.backtest.metrics.analyse) or a "
            "result with a .performance attribute holding one"
        )
    name = getattr(result_like, "name", None)
    strategies = getattr(result_like, "strategies", None)
    if not isinstance(strategies, Mapping) or len(strategies) < 2:
        strategies = None
    return performance, name, getattr(result_like, "assumptions", None), strategies


def _closed_net_pnl(trades: pd.DataFrame) -> np.ndarray:
    if trades.empty:
        return np.empty(0)
    closed = trades.loc[~pd.isna(trades["exit_time"]), "net_pnl"]
    return closed.to_numpy(dtype=float)


def _equity_section(performance: Performance) -> str:
    legend = ""
    if performance.benchmark_equity is not None:
        legend = (
            "<div class='legend'><span style='--key: var(--gain)'>Strategy</span>"
            "<span style='--key: var(--bench)'>Benchmark, scaled to the start equity</span></div>"
        )
    lede = (
        "Equity in rupees at every mark. Below it, the fall of the time-weighted wealth "
        "index from its running peak (cash flows excluded)."
    )
    return sections.section(
        "equity", "Equity and drawdown", legend + charts.equity_chart(performance), lede
    )


def _trade_sections(performance: Performance) -> list[str]:
    metrics, trades = performance.metrics, performance.trades
    pnl = _closed_net_pnl(trades)
    no_closed = "<p class='note'>No closed trades.</p>"
    histogram = charts.pnl_histogram(pnl) if len(pnl) else no_closed
    counts = (
        f"{metrics['trades']} closed and {metrics['open_trades']} open trade(s), in entry order."
    )
    listing = tables.trade_list(trades) if len(trades) else "<p class='note'>No trades.</p>"
    return [
        sections.section(
            "distribution",
            "Trade P&amp;L distribution",
            histogram,
            "Net P&amp;L of closed trades, after costs.",
        ),
        sections.section("trades", "Trades", listing, counts),
        sections.section(
            "costs",
            "Costs",
            tables.cost_table(metrics, performance.cost_breakdown),
            "Costs of closed trades.",
        ),
    ]


def render(
    performance: Performance,
    *,
    title: str,
    assumptions: list[str],
    strategies: Mapping[str, Performance] | None,
) -> str:
    """The full HTML page for `performance` (see `tearsheet`)."""
    metrics = performance.metrics
    body = [
        sections.masthead(title, performance),
        sections.section(
            "summary", "Summary", sections.headline(metrics) + sections.facts(metrics)
        ),
        _equity_section(performance),
        sections.section(
            "monthly",
            "Monthly returns",
            tables.monthly_grid(performance),
            "Compounded session returns per calendar month.",
        ),
        sections.section("yearly", "Yearly returns", tables.yearly_bars(performance)),
        *_trade_sections(performance),
    ]
    if strategies is not None:
        correlation = return_correlation(strategies)
        body.append(
            sections.section(
                "strategies", "Strategies", sections.comparison(strategies, correlation)
            )
        )
    body += [
        sections.section("assumptions", "Assumptions", sections.assumption_list(assumptions)),
        sections.section(
            "notes", "Notes and definitions", sections.method_notes(metrics, performance.exchanges)
        ),
        f"<footer>Generated by openalgo {text(__version__)} (openalgo.backtest). Figures come "
        "from the recorded equity and trades; the notes define each one.</footer>",
    ]
    return (
        "<!DOCTYPE html>\n<html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<meta name='color-scheme' content='light dark'>"
        f"<title>{text(title)}: backtest report</title><style>{STYLESHEET}</style></head>"
        f"<body><main>{''.join(body)}</main></body></html>\n"
    )


__all__ = ["DEFAULT_TITLE", "render", "tearsheet"]
