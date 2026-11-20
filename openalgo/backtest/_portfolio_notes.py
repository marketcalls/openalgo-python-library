"""What a portfolio run states in words: its Assumptions lines (the account, the strategies,
the instruments several strategies trade) and its own warnings (opposite sides held on one
instrument, and the days a broker would have netted the charges of several strategies).

Two strategies may trade one instrument in a backtest, on either side: each holds its own
units, blocks its own margin and pays its own charges, as if it traded alone. A broker
nets them: one position per symbol and product, one contract note per day (delivery buys
and sells of one scrip offset into an intraday trade, the DP charge once per scrip and
day of delivery sales, one rounding per note). The Assumptions block says so, and a
warning names each instrument and day where the difference arose. Amounts are rupees.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from ._messages import counted
from ._portfolio_tables import OppositeSides
from ._strategy_args import Account
from .report._format import indian_digits
from .result import Result

#: The product of delivery trades, which a broker's contract note nets across the
#: account within a day.
DELIVERY = "CNC"
#: The Assumptions line of a run with an instrument several strategies trade (18-multi-
#: strategy 3).
SHARED_RULES = (
    "each strategy holds its own units and pays its own charges as if it traded alone; "
    "opposite sides are allowed here, where bt.live refuses the later entry; margin is "
    "blocked per strategy, where a broker nets one position per symbol and product; a "
    "broker's contract note nets one scrip's buys and sells of a day across the account "
    "(delivery that offsets is charged as intraday), charges the DP charge once per scrip "
    "and day and rounds each note once, where each strategy's note is charged here; "
    "positions in different products are different broker positions"
)


def rupees_text(amount: float) -> str:
    """Rupees with Indian digit grouping and no paise: "Rs 8,00,000"."""
    return f"Rs {indian_digits(amount, 0)}"


def spoken_list(names: Sequence[str]) -> str:
    """Names as a sentence says them: "A", "A and B", "A, B and C"."""
    if len(names) < 2:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def account_note(account: Account) -> str:
    """The Assumptions line of the account: what the pool lets one strategy do to another."""
    funds = rupees_text(account.funds)
    if account.pool == "separate":
        note = (
            f"separate allocations of {funds} together: each strategy is limited by its own "
            "capital only, the account is their sum and no strategy's margin can block "
            "another's"
        )
        if account.account_max_daily_loss is not None:
            note += (
                "; with account_max_daily_loss, each strategy's result equals its run alone "
                "only until the account's limit is reached"
            )
        return note
    if account.account_capital is None:
        return (
            f"shared account of {funds}, the sum of the capitals: it refuses an order only "
            "when other strategies' open losses leave too little free cash; give "
            "account_capital to model a smaller account"
        )
    return (
        f"shared account of {funds} for strategies of {rupees_text(sum(account.capitals))} "
        "together: an opening fill needs its margin within the strategy's own free cash and "
        "within the account's, which the other strategies' margin and open losses take"
    )


def _traded(results: Mapping[str, Result]) -> dict[tuple[str, str], list[str]]:
    """By (symbol, exchange), each strategy trading it with its product ("A (MIS)"), in
    call order."""
    traded: dict[tuple[str, str], list[str]] = {}
    for name, result in results.items():
        product = result.manifest["arguments"]["product"]
        for entry in result.manifest["instruments"]:
            key = (entry["symbol"], entry["exchange"])
            traded.setdefault(key, []).append(f"{name} ({product})")
    return traded


def assumptions(account: Account, results: Mapping[str, Result]) -> dict[str, Any]:
    """The Assumptions block of a portfolio run: the strategies, the pool, the account's
    funds and note, its daily loss limit, the instruments several strategies trade (None
    when none is shared) with what a backtest does differently from a broker there, then
    a line per strategy: what it trades, on which interval and base, its product and
    capital."""
    shared = {key: who for key, who in _traded(results).items() if len(who) > 1}
    block: dict[str, Any] = {
        "strategies": ", ".join(results),
        "pool": account.pool,
        "account_capital": account.funds,
        "account": account_note(account),
        "account_max_daily_loss": account.account_max_daily_loss,
        "shared_instruments": "; ".join(
            f"{symbol} ({exchange}): {spoken_list(who)}"
            for (symbol, exchange), who in shared.items()
        )
        or None,
    }
    if shared:
        block["shared_instrument_rules"] = SHARED_RULES
    for name, result in results.items():
        settings = result.manifest["arguments"]
        block[f"strategy {name}"] = (
            f"{', '.join(settings['symbols'])} on {settings['exchange']}, "
            f"{settings['interval']} bars on a {settings['base']} base, "
            f"{settings['product']}, capital {rupees_text(settings['capital'])}"
        )
    return block


def opposite_sides_warning(found: OppositeSides) -> str:
    """The warning of an instrument two strategies held on opposite sides."""
    return (
        f"{found.symbol} ({found.exchange}, {found.product}): {spoken_list(found.long)} held "
        f"it long while {spoken_list(found.short)} held it short, first from "
        f"{found.time:%Y-%m-%d %H:%M} IST; bt.live would have refused the later entry, "
        "since a broker nets one position per symbol and product"
    )


def _delivery_days(results: Mapping[str, Result]) -> pd.DataFrame:
    """Each strategy's delivery (CNC) units bought and sold per scrip and IST date:
    symbol, exchange, date, strategy, bought, sold."""
    frames = []
    for name, result in results.items():
        fills = result.fills
        fills = fills[fills["product"] == DELIVERY]
        exchanges = [entry["exchange"] for entry in result.manifest["instruments"]]
        units = fills["quantity_units"].abs()
        frames.append(
            pd.DataFrame(
                {
                    "symbol": fills["symbol"].to_numpy(),
                    "exchange": [exchanges[index] for index in fills["instrument"]],
                    "date": fills["time"].dt.date.to_numpy(),
                    "strategy": name,
                    "bought": units.where(fills["side"] == "BUY", 0.0).to_numpy(),
                    "sold": units.where(fills["side"] == "SELL", 0.0).to_numpy(),
                }
            )
        )
    days = pd.concat(frames, ignore_index=True)
    keys = ["symbol", "exchange", "date", "strategy"]
    return days.groupby(keys, sort=True)[["bought", "sold"]].sum().reset_index()


def _netting_warning(symbol: str, exchange: str, what: str, dates: list, why: str) -> str:
    more = "" if len(dates) == 1 else f" and {counted(len(dates) - 1, 'more day')}"
    return f"{symbol} ({exchange}) {what} on {dates[0]}{more}: {why}"


def netting_warnings(results: Mapping[str, Result]) -> list[str]:
    """A warning per scrip whose delivery trades a broker would have charged otherwise:
    days on which one strategy bought it on CNC and another sold it on CNC (a contract
    note nets them into an intraday trade), and days on which two strategies sold it from
    holdings (more sold than bought that day: the DP charge is charged once per scrip and
    day). Each names the scrip and its first such day."""
    days = _delivery_days(results)
    crossed: dict[tuple[str, str], list] = {}
    sold_twice: dict[tuple[str, str], list] = {}
    for (symbol, exchange, date), day in days.groupby(["symbol", "exchange", "date"]):
        buyers = set(day.loc[day["bought"] > 0, "strategy"])
        sellers = set(day.loc[day["sold"] > 0, "strategy"])
        if any(buyer != seller for buyer in buyers for seller in sellers):
            crossed.setdefault((symbol, exchange), []).append(date)
        if (day["sold"] > day["bought"]).sum() > 1:
            sold_twice.setdefault((symbol, exchange), []).append(date)
    notes = [
        _netting_warning(
            *key,
            "was bought on CNC by one strategy and sold on CNC by another",
            dates,
            "a broker nets a scrip's buys and sells of a day across the account in one "
            "contract note and charges what offsets as intraday, where each strategy's "
            "delivery trades are charged here",
        )
        for key, dates in crossed.items()
    ]
    notes += [
        _netting_warning(
            *key,
            "was sold from holdings by two strategies",
            dates,
            "a broker charges the DP charge once per scrip and day, where each strategy "
            "is charged one here",
        )
        for key, dates in sold_twice.items()
    ]
    return notes


__all__ = [
    "DELIVERY",
    "SHARED_RULES",
    "account_note",
    "assumptions",
    "netting_warnings",
    "opposite_sides_warning",
    "rupees_text",
    "spoken_list",
]
