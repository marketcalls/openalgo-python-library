"""The tables of a portfolio run's Portfolio section (`portfolio`): the orders the account
refused or ended, and the net position of each instrument two strategies or more held,
with the sortable table they share.

Each part is None when the page's `View.portfolio` is None (every page but a
portfolio's) or when the page does not show it. `View.portfolio` is the
`PortfolioResult`, read through its public tables (`refused`, `net_positions`,
`strategies`) and its `assumptions`, whose line `REFUSED_NOTE` words an account that
refused nothing in a run's own terms. Times are IST and quantities units.
"""

from collections.abc import Sequence
from typing import Any

import pandas as pd

from ..._messages import counted, spoken_list
from .. import _format as fmt
from .layout import SMALL, View, card, stack
from .trades import PAGER, TRADES_PER_PAGE, sort_key

#: What each reason of `PortfolioResult.refused` means, as its table says it: the
#: account's refusals and the working orders an account rule ended (every reason the
#: fund records, `openalgo._oaindicators.backtest.REFUSALS`).
REASONS = {
    "account_margin": "refused: the account's free cash did not cover its margin",
    "account_loss_gate": "refused: an entry after the account's daily loss limit",
    "account_profit_gate": "refused: an entry after the account's daily profit target",
    "account_trade_limit": "refused: an entry past the account's entries for the day",
    "account_loss": "cancelled at the account's daily loss limit",
    "account_profit": "cancelled at the account's daily profit target",
}
#: The column of a strategy's units in `PortfolioResult.net_positions`.
UNITS_COLUMN = "{name}_units"
#: The key of the note in `assumptions` in which a run words its account's empty refusals
#: itself (a live session's account, by the limits it set: the `SessionResult.portfolio`
#: notes), which this card shows and the page's Assumptions block leaves out; a run
#: without it shows `NOTHING_REFUSED`.
REFUSED_NOTE = "refused"
#: What the card of the account's refusals says when the account refused and ended none.
NOTHING_REFUSED = "The account refused no order and ended none."

#: One cell of a table: its HTML, its sort value and its CSS class.
Cell = tuple[str, object, str]


def table(columns: Sequence[tuple[str, str]], rows: Sequence[Sequence[Cell]], paged: bool) -> str:
    """A table in the page's style that sorts on a click of a heading, as the trade list
    does, and with `paged` shows `TRADES_PER_PAGE` rows at a time with Prev and Next
    (every row reads without script). `columns` are (heading HTML, alignment "l" or
    "r")."""
    head = "".join(f"<th class='{align} s'>{heading}</th>" for heading, align in columns)
    body = "".join(
        f"<tr data-i='{i}'>"
        + "".join(
            f"<td class='{css}' data-v='{sort_key(value)}'>{text}</td>" for text, value, css in row
        )
        + "</tr>"
        for i, row in enumerate(rows)
    )
    hooks = f" data-sort data-page='{TRADES_PER_PAGE}'" if paged else " data-sort"
    return (
        f"<div class='scroll'><table class='tbl hover'{hooks}><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table></div>{PAGER if paged else ''}"
    )


def units(value: float) -> str:
    """Units as a table shows them: whole, or to two places for a part share."""
    return fmt.number(value, 0 if float(value).is_integer() else 2)


def refused(view: View) -> str | None:
    """The orders the account refused or ended (`PortfolioResult.refused`), a row each in
    time order: when (the IST open of the slice), the strategy, its order id (in its own
    engine), the symbol, the side and why."""
    portfolio = view.portfolio
    if portfolio is None or not view.has("refused"):
        return None
    frame = portfolio.refused
    title = "Refused by the Account"
    if not len(frame):
        return card(title, "", f"<p class='empty'>{fmt.text(_nothing_refused(portfolio))}</p>")
    columns = [
        ("Time (IST)", "l"),
        ("Strategy", "l"),
        ("Order", "r"),
        ("Symbol", "l"),
        ("Side", "l"),
        ("Reason", "l"),
    ]
    rows = [
        [
            (fmt.moment(order.time), order.time, "l nums"),
            (fmt.text(order.strategy), order.strategy, "l"),
            (str(order.order_id), int(order.order_id), "r nums"),
            (fmt.text(order.symbol), order.symbol, "l"),
            (fmt.text(order.side), order.side, "l"),
            (fmt.text(REASONS[order.reason]), order.reason, "l"),
        ]
        for order in frame.itertuples(index=False)
    ]
    subtitle = f"{counted(len(frame), 'order')} the account refused or ended, with the times"
    return card(title, subtitle, table(columns, rows, paged=True))


def _nothing_refused(portfolio: Any) -> str:
    """The words of an account that refused and ended no order: the run's own
    (`REFUSED_NOTE`), as a sentence, or `NOTHING_REFUSED`."""
    words = portfolio.assumptions.get(REFUSED_NOTE)
    if not words:
        return NOTHING_REFUSED
    return f"{words[0].upper()}{words[1:]}."


def _opposite_note(rows: pd.DataFrame, members: list[str]) -> str:
    """The note of the first change after which two members held the instrument on
    opposite sides, or "" when they never did."""
    held = rows[[UNITS_COLUMN.format(name=name) for name in members]]
    both = (held > 0).any(axis=1) & (held < 0).any(axis=1)
    if not both.any():
        return ""
    first = rows.loc[both.idxmax(), "time"]
    return (
        f"<p class='key'>Held on opposite sides first from {fmt.moment(first)}: bt.live "
        "refuses the later entry, since a broker nets one position per symbol and "
        "product.</p>"
    )


def _instrument(key: tuple[str, str, str], rows: pd.DataFrame, members: list[str]) -> str:
    """The card of one instrument and product: its members' units and the net at each
    change."""
    symbol, exchange, product = key
    columns = [("Time (IST)", "l"), *[(fmt.text(name), "r") for name in members], ("Net", "r")]
    body = []
    for _, change in rows.iterrows():
        held = [float(change[UNITS_COLUMN.format(name=name)]) for name in members]
        net = float(change["net_units"])
        body.append(
            [
                (fmt.moment(change["time"]), change["time"], "l nums"),
                *[(units(value), value, "r nums") for value in held],
                (units(net), net, "r nums"),
            ]
        )
    title = f"{fmt.text(symbol)} ({fmt.text(exchange)}, {fmt.text(product)})"
    subtitle = (
        f"{fmt.text(spoken_list(members))}: each one's units and the net a broker holds, "
        "at every change"
    )
    return card(title, subtitle, table(columns, body, paged=True) + _opposite_note(rows, members))


def net_positions(view: View) -> str | None:
    """For each instrument and product two strategies or more held (at any time, not
    only together), a card of their units and the net at every change
    (`PortfolioResult.net_positions`), its members in call order."""
    portfolio = view.portfolio
    if portfolio is None or not view.has("net_positions"):
        return None
    frame = portfolio.net_positions
    cards = []
    for key, rows in frame.groupby(["symbol", "exchange", "product"], sort=False):
        members = [
            name
            for name in portfolio.strategies
            if (rows[UNITS_COLUMN.format(name=name)] != 0.0).any()
        ]
        if len(members) > 1:
            cards.append(_instrument(key, rows, members))
    if not cards:
        return card(
            "Net Positions",
            "",
            "<p class='empty'>No instrument was held by two strategies or more.</p>",
        )
    return stack([("" if i == 0 else SMALL, part) for i, part in enumerate(cards)])


__all__ = [
    "NOTHING_REFUSED",
    "REASONS",
    "REFUSED_NOTE",
    "UNITS_COLUMN",
    "net_positions",
    "refused",
    "table",
    "units",
]
