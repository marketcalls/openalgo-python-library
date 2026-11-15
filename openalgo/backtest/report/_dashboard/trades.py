"""The run's own blocks: the trade table (sortable and paged by the page's script), the
R-multiple tiles, the head-to-head table of several strategies and the method notes."""

import math
from collections.abc import Mapping

import pandas as pd

from ..._messages import counted
from .. import _format as fmt
from .._performance import Performance
from .cards import tone

#: Trades on one page of the trade table (every row shows without script).
TRADES_PER_PAGE = 20
#: The figures compared across strategies: label, metrics key, and the direction that is
#: better ("up" or "down").
COMPARED = (
    ("Cumulative Return", "total_return", "up"),
    ("CAGR", "cagr", "up"),
    ("Sharpe", "sharpe", "up"),
    ("Sortino", "sortino", "up"),
    ("Calmar", "calmar", "up"),
    ("Max Drawdown", "max_drawdown", "up"),
    ("Volatility", "volatility", "down"),
    ("Omega", "omega", "up"),
    ("Win Days", "session_win_rate", "up"),
    ("Profit Factor", "profit_factor", "up"),
    ("Ulcer Index", "ulcer_index", "down"),
    ("Serenity Index", "serenity_index", "up"),
)
#: Fraction-valued figures of `COMPARED`, shown as percentages.
_PERCENT_KEYS = {
    "total_return",
    "cagr",
    "max_drawdown",
    "volatility",
    "session_win_rate",
    "ulcer_index",
}


def _key(value: object) -> str:
    """The sort value of a cell: a number, the nanoseconds of a time, or lower-case
    text; empty when missing."""
    if fmt.is_missing(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return str(value.value)
    if isinstance(value, int | float):
        return repr(float(value)) if math.isfinite(value) else ""
    return fmt.text(str(value).lower())


def _td(text: str, value: object, css: str = "r nums") -> str:
    return f"<td class='{css}' data-v='{_key(value)}'>{text}</td>"


def _quantity(units: float) -> str:
    return fmt.number(units, 0 if float(units).is_integer() else 2)


def _row(number: int, trade: pd.Series, with_r: bool) -> str:
    net, ret = float(trade["net_pnl"]), float(trade["return_pct"])
    bars = trade["bars_held"]
    cells = [
        _td(str(number), number),
        _td(fmt.text(trade["symbol"]), trade["symbol"], "l"),
        _td(fmt.text(str(trade["direction"]).capitalize()), trade["direction"], "l"),
        _td(fmt.moment(trade["entry_time"]), trade["entry_time"], "l nums"),
        _td(fmt.moment(trade["exit_time"]), trade["exit_time"], "l nums"),
        _td(_quantity(trade["quantity"]), float(trade["quantity"])),
        _td(fmt.rupees(float(trade["entry_price"])), float(trade["entry_price"])),
        _td(fmt.rupees(float(trade["exit_price"])), float(trade["exit_price"])),
        _td(
            fmt.rupees(float(trade["gross_pnl"])),
            float(trade["gross_pnl"]),
            "r nums" + tone(trade["gross_pnl"]),
        ),
        _td(fmt.rupees(float(trade["charges"])), float(trade["charges"])),
        _td(fmt.rupees(net), net, "r nums" + tone(net)),
        _td(fmt.percent(ret / 100), ret, "r nums" + tone(ret)),
        _td(
            fmt.MISSING if pd.isna(bars) else str(int(bars)), None if pd.isna(bars) else float(bars)
        ),
        _td(fmt.number(float(trade["days_held"]), 1), float(trade["days_held"])),
        _td(fmt.text(trade["exit_reason"]), trade["exit_reason"], "l"),
    ]
    if with_r:
        r = float(trade["r_multiple"])
        cells.append(
            _td(fmt.MISSING if math.isnan(r) else f"{fmt.number(r)}R", r, "r nums" + tone(r))
        )
    return f"<tr data-i='{number}'>{''.join(cells)}</tr>"


def trade_table(listing: pd.DataFrame) -> str:
    """The trade list (`openalgo.backtest._trade_list`) as a sortable table, paged by
    `TRADES_PER_PAGE` rows with script; the R column only when a trade has one."""
    with_r = bool(listing["r_multiple"].notna().any())
    columns = [
        ("#", "r"),
        ("Symbol", "l"),
        ("Side", "l"),
        ("Entry (IST)", "l"),
        ("Exit (IST)", "l"),
        ("Qty", "r"),
        ("Entry Price", "r"),
        ("Exit Price", "r"),
        ("Gross P&amp;L", "r"),
        ("Charges", "r"),
        ("Net P&amp;L", "r"),
        ("Return", "r"),
        ("Bars", "r"),
        ("Days", "r"),
        ("Exit Reason", "l"),
    ] + ([("R", "r")] if with_r else [])
    head = "".join(f"<th class='{align} s'>{label}</th>" for label, align in columns)
    rows = "".join(_row(int(number), trade, with_r) for number, trade in listing.iterrows())
    pager = (
        "<div class='pager js-only'><span class='pager-info nums'></span>"
        "<button type='button' class='btn pg-prev'>Prev</button>"
        "<button type='button' class='btn pg-next'>Next</button></div>"
    )
    return (
        f"<div class='scroll'><table class='tbl hover trades' data-sort "
        f"data-page='{TRADES_PER_PAGE}'><thead><tr>{head}</tr></thead><tbody>{rows}</tbody>"
        f"</table></div>{pager}"
    )


def _r(value: float) -> str:
    return fmt.MISSING if fmt.is_missing(value) else f"{fmt.number(value)}R"


def r_tiles(summary: Mapping[str, object]) -> str:
    """The R-multiple summary of the closed trades (`_result_r.r_summary`) as tiles."""
    tiles = (
        ("Trades with R", f"{summary['trades_with_risk']} of {summary['trades']}", ""),
        ("Mean R", _r(summary["mean_r"]), tone(summary["mean_r"])),
        ("Median R", _r(summary["median_r"]), tone(summary["median_r"])),
        ("Std Dev R", _r(summary["std_r"]), ""),
        ("Best R", _r(summary["best_r"]), tone(summary["best_r"])),
        ("Worst R", _r(summary["worst_r"]), tone(summary["worst_r"])),
        ("Total R", _r(summary["total_r"]), tone(summary["total_r"])),
        ("Above 0R", fmt.percent(summary["positive_share"]), ""),
    )
    cells = "".join(
        f"<div class='stat'><div class='eyebrow trunc'>{label}</div>"
        f"<div class='stat-value nums{css}'>{value}</div></div>"
        for label, value, css in tiles
    )
    return f"<div class='stats'>{cells}</div>"


def _best_and_worst(values: dict[str, float], better: str) -> tuple[str | None, str | None]:
    known = {name: v for name, v in values.items() if not fmt.is_missing(v)}
    if len(known) < 2:
        return None, None
    ranked = sorted(known, key=known.get)
    return (ranked[-1], ranked[0]) if better == "up" else (ranked[0], ranked[-1])


def compare(strategies: Mapping[str, Performance]) -> str:
    """Key figures of each strategy, the best of each row green and the worst red."""
    names = list(strategies)
    columns = "".join(f"<th class='r'>{fmt.text(name)}</th>" for name in names)
    rows = []
    for label, key, better in COMPARED:
        values = {name: strategies[name].metrics.get(key, math.nan) for name in names}
        best, worst = _best_and_worst(values, better)
        cells = []
        for name in names:
            value = values[name]
            shown = fmt.percent(value) if key in _PERCENT_KEYS else fmt.number(value)
            css = " best" if name == best else " neg" if name == worst else ""
            cells.append(f"<td class='r nums{css}'>{shown}</td>")
        rows.append(f"<tr><td class='l'>{label}</td>{''.join(cells)}</tr>")
    return (
        f"<div class='scroll'><table class='tbl'><thead><tr><th class='l'>Metric</th>{columns}"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def method_notes(performance: Performance) -> str:
    """How the figures were computed, in sentences."""
    m = performance.metrics
    notes = [
        "Returns are time weighted: cash flows move the equity but are not returns. Each "
        "IST session date contributes one return, from the last equity point of the date.",
        f"Volatility, Sharpe, Sortino and alpha are annualised with "
        f"{fmt.figure('sessions_per_year', m['sessions_per_year'])} sessions a year: the "
        f"{fmt.text(', '.join(performance.exchanges))} calendar in "
        f"{fmt.text(m['annualisation_years'])}, holiday data "
        f"{fmt.text(m['annualisation_coverage'])}. CAGR uses elapsed calendar time.",
        f"Risk-free rate {fmt.percent(m['risk_free_rate'])} a year, compounded down to "
        f"{fmt.percent(m['risk_free_per_session'], 4)} a session.",
    ]
    if m["annualisation_template_years"]:
        notes.append(
            f"No holiday list for {fmt.text(m['annualisation_template_years'])}: every "
            "weekday was counted as a session, so the annualisation is overstated."
        )
    if m["missing_sessions"]:
        sessions = counted(m["missing_sessions"], "calendar session has", "calendar sessions have")
        notes.append(
            f"{sessions} no equity point; the move of each is folded into the next "
            "session's return."
        )
    if m["ruined"]:
        notes.append(f"The account was ruined on {fmt.moment(m['ruin_time'])}.")
    notes.append(
        "Every drawdown figure is a fall, so it is zero or negative. The benchmark's "
        "figures are measured as the strategy's are, on its known session returns. Each "
        "metric's definition shows on its label; every definition cites its source in "
        "the documentation of openalgo.backtest.metrics."
    )
    paragraphs = "".join(f"<p>{note}</p>" for note in notes)
    return f"<div class='prose'>{paragraphs}</div>"


__all__ = ["COMPARED", "TRADES_PER_PAGE", "compare", "method_notes", "r_tiles", "trade_table"]
