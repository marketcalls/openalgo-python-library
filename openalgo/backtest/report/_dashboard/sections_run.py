"""The sections of the dashboard about the run itself: every metric, the trades and
their charges, the comparison of several strategies, and the notes (how the figures
were computed and what the run assumed)."""

import numpy as np
import pandas as pd

from ..._messages import counted
from ..._result_r import r_summary
from ..._trade_list import trade_list
from .. import _format as fmt
from .._combined import return_correlation
from . import cards, data, tables
from .layout import SMALL, Section, View, card, stack
from .trades import compare, method_notes, r_tiles, trade_table

#: The class of an assumption row that is a warning (its line starts "warning: ").
WARNING = " class='warn'"


def metrics(view: View) -> Section | None:
    """Every metric, the benchmark's beside the strategy's, sortable."""
    if not view.has("metrics_table"):
        return None
    p = view.performance
    table = tables.metrics(
        p,
        view.bench,
        (view.bench_name, view.title),
        comparable=data.COMPARABLE,
        definitions=view.has("definitions"),
    )
    names = ([view.bench_name] if view.bench is not None else []) + [view.title]
    columns = " &#183; ".join(fmt.text(name) for name in names)
    subtitle = f"{len(tables.metric_keys(p))} metrics &#183; {columns}"
    body = f"<div class='card'><div class='card-body'>{table}</div></div>"
    return Section("metrics", "Metrics", subtitle, body)


def _closed_r(view: View) -> np.ndarray:
    """The R-multiples of the closed trades (NaN for one without an initial risk)."""
    trades = view.performance.trades
    if "r_multiple" not in trades.columns:
        return np.empty(0)
    return trades.loc[trades["exit_time"].notna(), "r_multiple"].to_numpy(dtype=float)


def _trade_list_card(view: View) -> str:
    m = view.performance.metrics
    listing = trade_list(view.performance.trades)
    closed = counted(m["trades"], "closed trade")
    subtitle = f"{closed}, sortable; {m['open_trades']} open at the end, not listed"
    body = trade_table(listing) if len(listing) else "<p class='empty'>No closed trades.</p>"
    return card("Trade List", subtitle, body)


def trades(view: View) -> Section | None:
    """The trade tiles, the trade list, the charges by component and, when a closed
    trade has an initial risk, the R-multiples."""
    p = view.performance
    r_values = _closed_r(view)
    has_r = bool(np.isfinite(r_values).any())
    r_card = ""
    if view.has("r_multiples") and has_r:
        r_card = (
            "<div id='r-multiples'>"
            + card(
                "R-Multiples",
                "Net P&amp;L over the initial risk, the loss at the stop the entry set",
                r_tiles(r_summary(r_values)),
            )
            + "</div>"
        )
    blocks = [
        ("", cards.grid(cards.TRADES, p, None, view.bench_name) if view.has("trade_stats") else ""),
        (SMALL, _trade_list_card(view) if view.has("trade_table") else ""),
        (
            SMALL,
            card(
                "Charges",
                "Charges of closed trades by component",
                tables.costs(p.metrics, p.cost_breakdown),
            )
            if view.has("costs")
            else "",
        ),
        (SMALL, r_card),
    ]
    body = stack(blocks)
    return Section("trades", "Trades", "Closed trades &amp; charges", body) if body else None


def strategies(view: View) -> Section | None:
    """Key figures of each strategy of a combined result, the best green and the worst
    red, and the correlation of their session returns."""
    if view.strategies is None or not view.has("strategies"):
        return None
    correlation = return_correlation(view.strategies)
    body = stack(
        [
            (
                "",
                card(
                    "Head to Head",
                    "Best in green, worst in red, per metric",
                    compare(view.strategies),
                ),
            ),
            (SMALL, card("Correlation", "Of session returns", _correlation(correlation))),
        ]
    )
    return Section("strategies", "Strategies", "Side by side", body)


def _correlation(frame: pd.DataFrame) -> str:
    names = list(frame.columns)
    columns = [("", "l")] + [(fmt.text(name), "r") for name in names]
    rows = "".join(
        f"<tr><td class='l'>{fmt.text(row)}</td>"
        + "".join(f"<td class='r nums'>{fmt.number(frame.loc[row, c])}</td>" for c in names)
        + "</tr>"
        for row in names
    )
    table = f"<table class='tbl'>{tables.head(*columns)}<tbody>{rows}</tbody></table>"
    return f"<div class='scroll'>{table}</div>"


def notes(view: View) -> Section | None:
    """How the figures were computed, and the run's assumptions and warnings."""
    method = method_notes(view.performance) if view.has("method") else ""
    blocks = [
        ("", card("How the figures were computed", "", method) if method else ""),
        (SMALL, _assumptions(view.assumptions) if view.has("assumptions") else ""),
    ]
    body = stack(blocks)
    return Section("notes", "Notes", "Method, assumptions &amp; warnings", body) if body else None


def _assumption_row(line: str) -> str:
    """One assumption line as a row: the text before the first ": " as its name (none
    when there is no such separator), the rest as its value; a warning in red."""
    name, separator, value = line.partition(": ")
    if not separator:
        name, value = "", line
    css = WARNING if name == "warning" else ""
    return f"<tr{css}><td class='l'>{fmt.text(name)}</td><td class='l'>{fmt.text(value)}</td></tr>"


def _assumptions(lines: list[str]) -> str:
    if not lines:
        body = "<p class='empty'>No assumptions were recorded with this result.</p>"
    else:
        rows = "".join(_assumption_row(line) for line in lines)
        body = f"<table class='tbl kv'><tbody>{rows}</tbody></table>"
    return card("Assumptions", "What the run assumed, and every warning", body)


__all__ = ["metrics", "notes", "strategies", "trades"]
