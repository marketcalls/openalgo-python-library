"""The dashboard's tables: returns and risk by horizon, yearly returns, the worst
drawdowns, every metric and the costs.

Tables marked sortable (`data-sort`) sort on a click of a column heading in the page's
script: the value to sort by rides on each cell (`data-v`), and a third click restores
the first order. Without script they read in their first order.
"""

import math
from collections.abc import Mapping, Sequence

from .. import _format as fmt
from .._definitions import DEFINITIONS, GROUPS
from .._performance import Performance
from .bars import YearRow
from .cards import tone

#: The horizons of the horizon table, with their labels (the SDK's horizon names).
HORIZONS = (("1Y", "1 Year"), ("3Y", "3 Years"), ("5Y", "5 Years"), ("Since start", "All-time"))
#: Worst drawdowns listed.
WORST_DRAWDOWNS = 10
#: Names of the cost components, as `openalgo.backtest.costs` itemises them.
COST_LABELS = {
    "brokerage": "Brokerage",
    "stt": "STT / CTT",
    "exchange_txn": "Exchange transaction charge",
    "sebi_fee": "SEBI fee",
    "stamp_duty": "Stamp duty",
    "gst": "GST",
    "dp_charge": "DP charge",
}
# Breakdown and trade costs may differ by rounding; a larger gap is shown as a row.
UNITEMISED_TOLERANCE_RUPEES = 0.01


def head(*columns: tuple[str, str]) -> str:
    """A table head of (label, alignment "l" or "r") pairs."""
    cells = "".join(f"<th class='{align}'>{label}</th>" for label, align in columns)
    return f"<thead><tr>{cells}</tr></thead>"


def _cell(text: str, css: str = "r nums") -> str:
    return f"<td class='{css}'>{text}</td>"


#: Units of the metrics that sort by their number; times, text and flags have no sort
#: value, so a sort puts them first (ascending) or last (descending), as undefined.
SORTED_UNITS = frozenset({"fraction", "rupees", "ratio", "count", "days", "seconds", "sessions"})


def _sort_key(key: str, value: object) -> str:
    """The `data-v` of a metric's cell: its number, or empty for an undefined figure or
    one that is not a number."""
    if DEFINITIONS[key].unit not in SORTED_UNITS or fmt.is_missing(value):
        return ""
    number = float(value)
    return repr(number) if math.isfinite(number) else ("1e308" if number > 0 else "-1e308")


def _horizon_cell(value: float, text: str, css: str) -> str:
    """A cell of the horizon table: "N/A", faint, for a horizon longer than the run."""
    return _cell("N/A", "r nums faint") if fmt.is_missing(value) else _cell(text, css)


def horizons(performance: Performance) -> str:
    """CAGR, maximum drawdown and Sharpe over the trailing horizons and since the
    start; "N/A" for a horizon longer than the run."""
    table = performance.horizons
    rows = []
    for key, label in HORIZONS:
        known = key in table.index
        cagr = table.loc[key, "cagr"] if known else math.nan
        depth = table.loc[key, "max_drawdown"] if known else math.nan
        sharpe = table.loc[key, "sharpe"] if known else math.nan
        rows.append(
            f"<tr><td class='l'>{label}</td>"
            + _horizon_cell(cagr, fmt.percent(cagr), f"r nums{tone(cagr)}")
            + _horizon_cell(depth, fmt.percent(depth), "r nums neg" if depth < 0 else "r nums")
            + _horizon_cell(sharpe, fmt.number(sharpe), "r nums")
            + "</tr>"
        )
    columns = (("Horizon", "l"), ("CAGR", "r"), ("Max DD", "r"), ("Sharpe", "r"))
    return f"<table class='tbl'>{head(*columns)}<tbody>{''.join(rows)}</tbody></table>"


def yearly(rows: Sequence[YearRow], with_benchmark: bool) -> str:
    """Each year's return, and the benchmark's beside it."""
    columns = [("Year", "l"), ("Strategy", "r")] + ([("Benchmark", "r")] if with_benchmark else [])
    body = []
    for row in rows:
        cells = _cell(fmt.percent(row.strategy), f"r nums{tone(row.strategy)}")
        if with_benchmark:
            cells += _cell(fmt.percent(row.benchmark), f"r nums{tone(row.benchmark)}")
        body.append(f"<tr><td class='l nums'>{row.year}</td>{cells}</tr>")
    return f"<table class='tbl'>{head(*columns)}<tbody>{''.join(body)}</tbody></table>"


def drawdowns(performance: Performance) -> str:
    """The deepest drawdowns, deepest first: when each started (its peak) and
    recovered ("Not yet" when it has not), its length in calendar days and depth."""
    worst = performance.drawdown_details(WORST_DRAWDOWNS)
    if worst.empty:
        return "<p class='empty'>No drawdowns.</p>"
    body = []
    for _, row in worst.iterrows():
        started = row["start"].strftime("%Y-%m-%d")
        if row["recovered"]:
            recovered = f"<td class='l nums muted'>{row['end'].strftime('%Y-%m-%d')}</td>"
        else:
            last = performance.metrics["end"].strftime("%Y-%m-%d")
            recovered = f"<td class='l nums neg' title='Still underwater as of {last}'>Not yet</td>"
        body.append(
            f"<tr><td class='l nums muted'>{started}</td>{recovered}"
            f"{_cell(fmt.number(row['days'], 0))}{_cell(fmt.percent(row['depth']), 'r nums neg')}"
            "</tr>"
        )
    columns = (("Started", "l"), ("Recovered", "l"), ("Days", "r"), ("Drawdown", "r"))
    return f"<table class='tbl'>{head(*columns)}<tbody>{''.join(body)}</tbody></table>"


def metric_keys(performance: Performance) -> list[str]:
    """Every metric of `performance`, in report order."""
    return [key for keys in GROUPS.values() for key in keys if key in performance.metrics]


def metrics(
    performance: Performance,
    bench: Performance | None,
    names: tuple[str, str],
    *,
    comparable: frozenset[str],
    definitions: bool = False,
) -> str:
    """Every metric with its label (its definition on hover), the benchmark's figure
    when one is given and the metric is one a benchmark has, and the strategy's;
    `names` are the benchmark's and the strategy's column headings. With
    `definitions`, a column states each definition."""
    bench_name, strategy_name = names
    columns = [("Metric", "l")]
    columns += [(fmt.text(bench_name), "r s")] if bench is not None else []
    columns += [(fmt.text(strategy_name), "r s")]
    columns += [("Definition", "l")] if definitions else []
    rows = []
    for i, key in enumerate(metric_keys(performance)):
        entry = DEFINITIONS[key]
        cells = [f"<td class='l' title='{fmt.text(entry.definition)}'>{fmt.text(entry.label)}</td>"]
        if bench is not None:
            value = bench.metrics.get(key) if key in comparable else None
            cells.append(_valued(key, value))
        value = performance.metrics[key]
        cells.append(_valued(key, value))
        if definitions:
            cells.append(f"<td class='l def'>{fmt.text(entry.definition)}</td>")
        rows.append(f"<tr data-i='{i}'>{''.join(cells)}</tr>")
    return (
        f"<div class='scroll'><table class='tbl hover' data-sort>{head(*columns)}"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _valued(key: str, value: object) -> str:
    return f"<td class='r nums' data-v='{_sort_key(key, value)}'>{fmt.figure(key, value)}</td>"


def costs(metrics: Mapping[str, object], breakdown: Mapping[str, float]) -> str:
    """Costs of closed trades by component (when the trades table itemised them), their
    total, and the path from gross to net P&L."""
    total = float(metrics["total_costs"])
    rows = []
    for name, amount in breakdown.items():
        share = amount / total if total else math.nan
        rows.append((COST_LABELS.get(name, fmt.text(name)), fmt.rupees(amount), share))
    unitemised = total - sum(breakdown.values())
    if breakdown and abs(unitemised) > UNITEMISED_TOLERANCE_RUPEES:
        share = unitemised / total if total else math.nan
        rows.append(("Not itemised", fmt.rupees(unitemised), share))
    body = "".join(
        f"<tr><td class='l'>{label}</td>{_cell(amount)}{_cell(fmt.percent(share, 1))}</tr>"
        for label, amount, share in rows
    )
    gross, net = metrics["gross_pnl"], metrics["net_pnl"]
    summary = (
        ("Total charges", fmt.rupees(total), ""),
        ("Gross P&amp;L", fmt.figure("gross_pnl", gross), tone(gross)),
        ("Net P&amp;L", fmt.figure("net_pnl", net), tone(net)),
        (
            "Charges / gross P&amp;L",
            fmt.figure("cost_share_of_gross_pnl", metrics["cost_share_of_gross_pnl"]),
            "",
        ),
    )
    body += "".join(
        f"<tr class='total'><td class='l'>{label}</td>{_cell(value, 'r nums' + css)}<td></td></tr>"
        for label, value, css in summary
    )
    columns = (("Component", "l"), ("Rupees", "r"), ("Share", "r"))
    return f"<table class='tbl'>{head(*columns)}<tbody>{body}</tbody></table>"


__all__ = [
    "COST_LABELS",
    "HORIZONS",
    "WORST_DRAWDOWNS",
    "costs",
    "drawdowns",
    "head",
    "horizons",
    "metric_keys",
    "metrics",
    "yearly",
]
