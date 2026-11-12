"""The report's tables: monthly heat grid, yearly returns, trade list and costs."""

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from . import _format as fmt
from ._performance import Performance

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
HEAT_STEPS = 4  # shading steps per arm of the monthly grid
TRADE_ROWS_SHOWN = 10  # rows shown at each end of a long trade list
#: Optional trade columns shown in the list when present, with their headings.
_TRADE_TEXT_COLUMNS = (("symbol", "Symbol"), ("exit_reason", "Exit reason"))
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
_UNITEMISED_TOLERANCE_RUPEES = 0.01


def _heat_class(value: float, scale: float) -> str:
    if math.isnan(value):
        return "empty"
    if value == 0 or scale == 0:
        return "zero"
    step = min(HEAT_STEPS, max(1, math.ceil(abs(value) / scale * HEAT_STEPS)))
    return f"{'g' if value > 0 else 'l'}{step}"


def monthly_grid(performance: Performance) -> str:
    """Years down, months across, each cell shaded by its return (blue gains, red
    losses, the darkest step at the largest monthly move), and the year's return."""
    table, yearly = performance.monthly_returns, performance.yearly_returns
    finite = table.to_numpy(dtype=float)
    scale = float(np.nanmax(np.abs(finite))) if np.isfinite(finite).any() else 0.0
    head = "".join(f"<th scope='col'>{m}</th>" for m in MONTHS)
    rows = []
    for year, values in table.iterrows():
        cells = []
        for month, value in zip(MONTHS, values.to_numpy(dtype=float), strict=True):
            shown = "" if math.isnan(value) else fmt.percent(value, 1)
            tip = f"{month} {year}: {fmt.percent(value, 2, signed=True)}"
            cells.append(f"<td class='{_heat_class(value, scale)}' title='{tip}'>{shown}</td>")
        total = fmt.percent(float(yearly.loc[year]), 1)
        rows.append(
            f"<tr><th scope='row'>{year}</th>{''.join(cells)}<td class='year'>{total}</td></tr>"
        )
    return (
        "<div class='scroll'><table class='heat'><thead><tr><th></th>"
        f"{head}<th scope='col'>Year</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        f"<p class='note'>Darkest shade: {fmt.percent(scale, 1)} in a month.</p>"
    )


def yearly_bars(performance: Performance) -> str:
    """Each calendar year's return with a bar either side of zero."""
    yearly = performance.yearly_returns
    scale = float(np.max(np.abs(yearly.to_numpy(dtype=float)))) if len(yearly) else 0.0
    rows = []
    for year, value in yearly.items():
        width = abs(value) / scale * 100 if scale > 0 else 0.0
        down = f"<i style='width:{width:.2f}%'></i>" if value < 0 else ""
        up = f"<i style='width:{width:.2f}%'></i>" if value > 0 else ""
        halves = f"<span class='down'>{down}</span><span class='up'>{up}</span>"
        track = f"<div class='halves'>{halves}</div>"
        rows.append(
            f"<tr><td>{year}</td><td class='num'>{fmt.percent(value, 2, signed=True)}</td>"
            f"<td class='track'>{track}</td></tr>"
        )
    return (
        "<table class='data bars'><thead><tr><th scope='col'>Year</th>"
        "<th scope='col' class='num'>Return</th><th scope='col'></th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _trade_row(position: int, trade: pd.Series, extra: list[str]) -> str:
    side = "Long" if trade["side"] == "BUY" else "Short"
    cells = [f"<td class='num'>{position}</td>"]
    cells += [f"<td>{fmt.text(trade[name])}</td>" for name in extra]
    cells += [
        f"<td>{side}</td>",
        f"<td>{fmt.moment(trade['entry_time'])}</td>",
        f"<td>{fmt.moment(trade['exit_time'])}</td>",
        f"<td class='num'>{fmt.number(float(trade['quantity']), 0)}</td>",
        f"<td class='num'>{fmt.rupees(float(trade['entry_price']))}</td>",
        f"<td class='num'>{fmt.rupees(float(trade['exit_price']))}</td>",
        f"<td class='num'>{fmt.rupees(float(trade['costs']))}</td>",
        f"<td class='num'>{fmt.rupees(float(trade['net_pnl']))}</td>",
    ]
    return f"<tr>{''.join(cells)}</tr>"


def _shown_times(column: pd.Series) -> pd.Series:
    """Trade times as IST timestamps (int64 UTC nanoseconds or tz-aware input)."""
    if pd.api.types.is_integer_dtype(column.dtype):
        return pd.to_datetime(column, utc=True).dt.tz_convert("Asia/Kolkata")
    return column.dt.tz_convert("Asia/Kolkata")


def trade_list(trades: pd.DataFrame) -> str:
    """Trades in entry order: all of them up to twice `TRADE_ROWS_SHOWN`, else the
    first and last `TRADE_ROWS_SHOWN` with the count of those left out."""
    frame = trades.copy()
    for name in ("entry_time", "exit_time"):
        frame[name] = _shown_times(frame[name])
    frame = frame.sort_values("entry_time", kind="stable").reset_index(drop=True)
    extra = [name for name, _ in _TRADE_TEXT_COLUMNS if name in frame.columns]
    heads = ["#"] + [label for name, label in _TRADE_TEXT_COLUMNS if name in frame.columns]
    heads += ["Side", "Entry", "Exit", "Qty", "Entry price", "Exit price", "Costs", "Net P&amp;L"]
    numeric = {"#", "Qty", "Entry price", "Exit price", "Costs", "Net P&amp;L"}
    head = "".join(
        f"<th scope='col'{" class='num'" if h in numeric else ''}>{h}</th>" for h in heads
    )
    count = len(frame)
    shown = range(count)
    if count > 2 * TRADE_ROWS_SHOWN:
        shown = [*range(TRADE_ROWS_SHOWN), *range(count - TRADE_ROWS_SHOWN, count)]
    rows = []
    for position in shown:
        if position == count - TRADE_ROWS_SHOWN and count > 2 * TRADE_ROWS_SHOWN:
            hidden = count - 2 * TRADE_ROWS_SHOWN
            rows.append(
                f"<tr class='gap'><td colspan='{len(heads)}'>{hidden} trades not shown</td></tr>"
            )
        rows.append(_trade_row(position + 1, frame.loc[position], extra))
    return (
        f"<div class='scroll'><table class='data'><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def cost_table(metrics: Mapping[str, object], breakdown: Mapping[str, float]) -> str:
    """Costs by component (when the trades table itemised them), their total, and the
    path from gross to net P&L."""
    total = float(metrics["total_costs"])
    rows = []
    for name, amount in breakdown.items():
        share = amount / total if total else math.nan
        label = COST_LABELS.get(name, fmt.text(name))
        rows.append((label, fmt.rupees(amount), fmt.percent(share, 1)))
    unitemised = total - sum(breakdown.values())
    if breakdown and abs(unitemised) > _UNITEMISED_TOLERANCE_RUPEES:
        share = unitemised / total if total else math.nan
        rows.append(("Not itemised", fmt.rupees(unitemised), fmt.percent(share, 1)))
    body = "".join(
        f"<tr><td>{label}</td><td class='num'>{amount}</td><td class='num'>{share}</td></tr>"
        for label, amount, share in rows
    )
    summary = (
        ("Total costs", fmt.rupees(total), "total"),
        ("Gross P&amp;L", fmt.figure("gross_pnl", metrics["gross_pnl"]), ""),
        ("Net P&amp;L", fmt.figure("net_pnl", metrics["net_pnl"]), ""),
        (
            "Costs / gross P&amp;L",
            fmt.figure("cost_share_of_gross_pnl", metrics["cost_share_of_gross_pnl"]),
            "",
        ),
    )
    body += "".join(
        f"<tr class='{css}'><td>{label}</td><td class='num'>{value}</td><td></td></tr>"
        for label, value, css in summary
    )
    return (
        "<table class='data'><thead><tr><th scope='col'>Component</th>"
        "<th scope='col' class='num'>Rupees</th><th scope='col' class='num'>Share</th></tr>"
        f"</thead><tbody>{body}</tbody></table>"
    )


__all__ = ["COST_LABELS", "cost_table", "monthly_grid", "trade_list", "yearly_bars"]
