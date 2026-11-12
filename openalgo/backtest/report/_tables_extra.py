"""The report's analysis tables: worst drawdowns, horizon summary, benchmark comparison,
Monte Carlo percentiles, losing streaks, and every metric grouped for the metrics page."""

from collections.abc import Callable, Mapping

import pandas as pd

from . import _format as fmt
from ._definitions import DEFINITIONS, GROUPS
from ._montecarlo import MonteCarlo


def _table(heads: list[tuple[str, bool]], rows: list[list[str]], css: str = "data") -> str:
    """A scrolling table; each head is (label, numeric), and numeric columns align right."""
    head = "".join(
        f"<th scope='col'{" class='num'" if numeric else ''}>{label}</th>"
        for label, numeric in heads
    )
    body = "".join(
        "<tr>"
        + "".join(
            f"<td{" class='num'" if numeric else ''}>{cell}</td>"
            for cell, (_, numeric) in zip(row, heads, strict=True)
        )
        + "</tr>"
        for row in rows
    )
    return (
        f"<div class='scroll'><table class='{css}'><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def drawdown_table(details: pd.DataFrame) -> str:
    """The worst drawdowns, deepest first (`Performance.drawdown_details`)."""
    if details.empty:
        return "<p class='note'>The equity never fell below a peak.</p>"
    heads = [
        ("#", True),
        ("Start", False),
        ("Trough", False),
        ("Recovery", False),
        ("Depth", True),
        ("Days", True),
        ("To trough", True),
        ("To recover", True),
    ]
    rows = [
        [
            str(rank),
            fmt.moment(row["start"]),
            fmt.moment(row["trough"]),
            fmt.moment(row["end"]) if row["recovered"] else "not recovered",
            fmt.percent(row["depth"]),
            fmt.number(row["days"], 1),
            fmt.number(row["days_to_trough"], 1),
            fmt.number(row["days_to_recover"], 1),
        ]
        for rank, row in details.iterrows()
    ]
    return _table(heads, rows)


def horizon_table(horizons: pd.DataFrame) -> str:
    """The horizon summary (`Performance.horizons`)."""
    heads = [
        ("Horizon", False),
        ("From", False),
        ("Sessions", True),
        ("Return", True),
        ("CAGR", True),
        ("Volatility", True),
        ("Sharpe", True),
        ("Max drawdown", True),
    ]
    with_benchmark = "benchmark_return" in horizons.columns
    if with_benchmark:
        heads += [("Benchmark", True), ("Benchmark sessions", True)]
    rows = []
    for label, row in horizons.iterrows():
        cells = [
            fmt.text(label),
            fmt.day(row["from"]),
            fmt.indian_digits(row["sessions"], 0),
            fmt.percent(row["total_return"], 2, signed=True),
            fmt.percent(row["cagr"]),
            fmt.percent(row["volatility"]),
            fmt.number(row["sharpe"]),
            fmt.percent(row["max_drawdown"]),
        ]
        if with_benchmark:
            cells.append(fmt.percent(row["benchmark_return"], 2, signed=True))
            cells.append(fmt.indian_digits(row["benchmark_sessions"], 0))
        rows.append(cells)
    return _table(heads, rows)


def benchmark_table(comparison: pd.DataFrame) -> str:
    """Strategy against benchmark per year (`Performance.benchmark_comparison`)."""
    heads = [
        ("Year", False),
        ("Strategy", True),
        ("Benchmark", True),
        ("Excess", True),
        ("Beat", False),
    ]
    rows = [
        [
            str(year),
            fmt.percent(row["strategy"], 2, signed=True),
            fmt.percent(row["benchmark"], 2, signed=True),
            fmt.percent(row["excess"], 2, signed=True),
            fmt.MISSING if pd.isna(row["beat"]) else ("yes" if row["beat"] else "no"),
        ]
        for year, row in comparison.iterrows()
    ]
    return _table(heads, rows)


def montecarlo_table(study: MonteCarlo) -> str:
    """Percentiles of each figure over the simulated paths, and the run's own."""
    shown: dict[str, Callable[[float], str]] = {
        "final_return": lambda v: fmt.percent(v, 2, signed=True),
        "cagr": fmt.percent,
        "max_drawdown": fmt.percent,
        "sharpe": fmt.number,
    }
    heads = [("Percentile", False), ("Final return", True), ("CAGR", True)]
    heads += [("Max drawdown", True), ("Sharpe", True)]
    rows = [
        [fmt.percent(level, 0), *(shown[k](row[k]) for k in shown)]
        for level, row in study.percentiles.iterrows()
    ]
    rows.append(["This run", *(shown[k](study.actual[k]) for k in shown)])
    return _table(heads, rows)


def streak_table(streaks: pd.DataFrame) -> str:
    """How many losing streaks of each length, of sessions and of trades."""
    if streaks.empty:
        return "<p class='note'>No losing session and no losing trade.</p>"
    heads = [("Length", True), ("Losing sessions", True), ("Losing trades", True)]
    rows = [
        [str(length), str(int(row["sessions"])), str(int(row["trades"]))]
        for length, row in streaks.iterrows()
    ]
    return _table(heads, rows)


def metrics_list(metrics: Mapping[str, object]) -> str:
    """Every metric present, grouped as `GROUPS`, with its value and definition."""
    parts = []
    for heading, keys in GROUPS.items():
        present = [key for key in keys if key in metrics]
        if not present:
            continue
        rows = [
            [
                fmt.text(DEFINITIONS[key].label),
                fmt.figure(key, metrics[key]),
                fmt.text(DEFINITIONS[key].definition),
            ]
            for key in present
        ]
        parts.append(f"<h3 class='sub'>{fmt.text(heading)}</h3>")
        parts.append(
            _table(
                [("Metric", False), ("Value", True), ("Definition", False)],
                rows,
                css="data metrics",
            )
        )
    return "".join(parts)


__all__ = [
    "benchmark_table",
    "drawdown_table",
    "horizon_table",
    "metrics_list",
    "montecarlo_table",
    "streak_table",
]
