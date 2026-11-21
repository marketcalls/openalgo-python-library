"""The Portfolio section of a portfolio run's tearsheet: four parts that only a portfolio's
page shows, after the Strategies section.

    contribution     each strategy's part in the account (`PortfolioResult.contribution`)
                     and its net P&L month by month, stacked on one bar a month
    account_margin   the account's highest margin and lowest free cash of each session,
                     and each strategy's margin when the account's was highest
    refused          the orders the account refused or ended (`portfolio_tables`)
    net_positions    each instrument two strategies or more held: their units and the net
                     a broker holds, at every change (`portfolio_tables`)

Each part is None when the page's `View.portfolio` is None (every page but a
portfolio's, so no other page changes) or when the page does not show it.
`View.portfolio` is the `PortfolioResult`, read through its public tables. Amounts are
rupees and times IST.

The monthly bars colour each strategy by its place in call order with the series colours
(`--series-1` onward), which a page showing them writes (`draws_series`); with more
strategies than colours, the last colour stands for the rest together.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .. import _format as fmt
from . import cards
from . import timeseries as ts
from .grouped_bars import BarSeries, stacked_chart
from .layout import LARGE, SMALL, TITLES, Section, View, card, label, legend, stack
from .portfolio_tables import net_positions, refused, table
from .sections import BENCHMARK_COLOUR, STRATEGY_COLOUR
from .style import SERIES

#: The section's anchor, and its key in `layout.PARTS`.
ANCHOR = "portfolio"
#: The strategies the monthly bars colour one each; past it, the last colour is the rest.
MAX_SERIES = len(SERIES)
#: Height of the account's margin chart, pixels (the page's drawdown chart's).
MARGIN_PX = 240
#: Percent in a fraction: `contribution`'s share is in percent, the page's format takes a
#: fraction.
PERCENT = 100.0
#: The contribution table: heading, column of `PortfolioResult.contribution`, how its
#: values are shown ("rupees", "percent" for a figure in percent, "int") and whether
#: they are coloured by sign.
CONTRIBUTION = (
    ("Net P&amp;L", "net_pnl_rupees", "rupees", True),
    ("Share", "share_pct", "percent", True),
    ("In Max Drawdown", "drawdown_pnl_rupees", "rupees", True),
    ("Margin at Peak", "peak_margin_rupees", "rupees", False),
    ("Refused", "orders_refused", "int", False),
)


def series_colour(position: int) -> str:
    """The CSS colour of the strategy at `position` (from 0) of the call order."""
    return f"var(--series-{position + 1})"


def draws_series(view: View, sections: Sequence[Section]) -> bool:
    """Whether the page draws the monthly bars, the one part that needs the series
    colours: a portfolio's page whose sections hold the Portfolio section with the
    contribution part."""
    shown = any(section.anchor == ANCHOR for section in sections)
    return view.portfolio is not None and view.has("contribution") and shown


def monthly_pnl(results: Mapping[str, Any]) -> pd.DataFrame:
    """Each strategy's net P&L in rupees by IST calendar month: a row a month (a monthly
    `PeriodIndex`) from the first month of any strategy's equity to the last, a column a
    strategy in call order. A month's P&L is the strategy's last equity of the month
    less its last equity before the month (its capital before its first point); 0 in a
    month with no point of its. Added up, a month's P&L is the account's."""
    lasts = {}
    for name, result in results.items():
        equity = result.equity["equity_rupees"]
        lasts[name] = equity.groupby(equity.index.tz_localize(None).to_period("M")).last()
    months = pd.PeriodIndex(sorted(set().union(*(last.index for last in lasts.values()))), freq="M")
    columns = {}
    for name, result in results.items():
        capital = float(result.manifest["arguments"]["capital"])
        last = lasts[name].reindex(months).ffill().fillna(capital)
        columns[name] = last - last.shift(1, fill_value=capital)
    return pd.DataFrame(columns, index=months, columns=list(results))


def bar_series(months: pd.DataFrame) -> list[BarSeries]:
    """The monthly bars' series: a strategy each in call order, in the series colour of
    its place; with more strategies than `MAX_SERIES`, those from the last colour's place
    on together, named "<count> others"."""
    names = list(months.columns)
    if len(names) > MAX_SERIES:
        rest = names[MAX_SERIES - 1 :]
        together = {f"{len(rest)} others": months[rest].sum(axis=1)}
        months = months[names[: MAX_SERIES - 1]].assign(**together)
    return [
        BarSeries(str(name), months[name].to_numpy(dtype=float), series_colour(k))
        for k, name in enumerate(months.columns)
    ]


def _figure(value: object, shown: str) -> str:
    if shown == "percent":
        return fmt.percent(float(value) / PERCENT)
    return cards.text(value, shown)


def _contribution_table(contribution: pd.DataFrame) -> str:
    columns = [("Strategy", "l"), *[(heading, "r") for heading, *_ in CONTRIBUTION]]
    rows = []
    for name, figures in contribution.iterrows():
        cells = [(fmt.text(name), name, "l")]
        for _, column, shown, toned in CONTRIBUTION:
            value = float(figures[column])
            css = "r nums" + (cards.tone(value) if toned else "")
            cells.append((_figure(value, shown), value, css))
        rows.append(cells)
    return table(columns, rows, paged=False)


def contribution(view: View) -> str | None:
    """Each strategy's part in the account and its net P&L by month, stacked."""
    portfolio = view.portfolio
    if portfolio is None or not view.has("contribution"):
        return None
    figures = portfolio.contribution
    key = (
        "<p class='key'>Net P&amp;L on each strategy's own capital; its share of the "
        "account's; its P&amp;L from the start to the trough of the account's maximum "
        "drawdown; its margin when the account's was highest; its orders the account "
        "refused.</p>"
    )
    months = monthly_pnl(portfolio.results)
    series = bar_series(months)
    chart = stacked_chart(
        [month.strftime("%b %Y") for month in months.index], series, label="Net P&L by month"
    )
    shown = [(one.name, one.colour) for one in series]
    by_month = card(
        "P&amp;L by Month",
        "Each strategy's net P&amp;L in rupees, stacked",
        (legend(shown) if len(months) else "") + chart,
    )
    parts = [
        (
            "",
            card(
                "Contribution",
                "Each strategy's part in the account",
                _contribution_table(figures) + key,
            ),
        ),
        (SMALL, by_month),
    ]
    return stack(parts)


def _by_session(values: pd.Series, sessions: pd.DatetimeIndex, how: str) -> np.ndarray:
    """`values` (by IST time) reduced to one figure a session date by `how` ("max" or
    "min"), on the page's `sessions` (NaN on a session without one)."""
    days = values.index.tz_localize(None).normalize().as_unit("ns")
    reduced = values.groupby(days).agg(how)
    return reduced.reindex(pd.DatetimeIndex(sessions).as_unit("ns")).to_numpy(dtype=float)


def _tile(name: str, value: float, sub: str = "") -> str:
    line = f"<div class='stat-sub nums'>{sub}</div>" if sub else ""
    return (
        f"<div class='stat'><div class='eyebrow trunc' title='{fmt.text(name)}'>"
        f"{fmt.text(name)}</div><div class='stat-value nums'>{cards.text(value, 'rupees')}"
        f"</div>{line}</div>"
    )


def account_margin(view: View) -> str | None:
    """The account's highest margin and lowest free cash of each session as two lines,
    then its highest margin and lowest free cash of the run and each strategy's margin
    when the account's was highest (`PortfolioResult.contribution`)."""
    portfolio = view.portfolio
    if portfolio is None or not view.has("account_margin"):
        return None
    rows = portfolio.account
    sessions = view.performance.session_returns.index
    lines = [
        ts.Line(
            "Highest margin", _by_session(rows["margin_rupees"], sessions, "max"), STRATEGY_COLOUR
        ),
        ts.Line(
            "Lowest free cash",
            _by_session(rows["free_cash_rupees"], sessions, "min"),
            BENCHMARK_COLOUR,
        ),
    ]
    chart = ts.chart(
        sessions,
        lines,
        height=MARGIN_PX,
        percent=False,
        label="The account's margin and free cash",
        ident="account-margin",
        value_labels=ts.number_labels,
    )
    shown = [(line.name, line.color) for line in lines]
    chart_card = card(
        "Account Margin &amp; Free Cash",
        "Each session's highest margin and lowest free cash, rupees",
        legend(shown) + chart,
    )
    if not len(rows):
        return chart_card
    peak, low = rows["margin_rupees"].idxmax(), rows["free_cash_rupees"].idxmin()
    account = [
        _tile("Highest margin", rows.loc[peak, "margin_rupees"], fmt.moment(peak)),
        _tile("Lowest free cash", rows.loc[low, "free_cash_rupees"], fmt.moment(low)),
    ]
    each = portfolio.contribution["peak_margin_rupees"]
    strategies = [_tile(str(name), float(value)) for name, value in each.items()]
    return stack(
        [
            ("", chart_card),
            (LARGE, f"<div class='stats'>{''.join(account)}</div>"),
            (
                SMALL,
                label("Each strategy's margin when the account's was highest")
                + f"<div class='stats'>{''.join(strategies)}</div>",
            ),
        ]
    )


#: The parts of the section in page order, with the gap before each.
IN_ORDER = (("", contribution), (LARGE, account_margin), (LARGE, refused), (LARGE, net_positions))


def portfolio(view: View) -> Section | None:
    """The Portfolio section, or None on a page without a portfolio or showing none of
    its parts."""
    body = stack([(gap, part(view) or "") for gap, part in IN_ORDER])
    if not body:
        return None
    return Section(ANCHOR, TITLES[ANCHOR], "The account and its strategies", body)


__all__ = [
    "ANCHOR",
    "CONTRIBUTION",
    "MAX_SERIES",
    "account_margin",
    "bar_series",
    "contribution",
    "draws_series",
    "monthly_pnl",
    "portfolio",
    "series_colour",
]
