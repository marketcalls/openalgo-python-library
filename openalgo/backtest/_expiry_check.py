"""A contract's expiry as OpenAlgo's master contract gives it, set against the reference
tables' (18-multi-strategy 12.2; unit M11): read by a backtest with a client, and by a live
session's preflight (`_live/_preflight_expiry`).

The master contract writes a futures or options contract's expiry as "DD-MMM-YY"
("29-DEC-26"; `EXPIRY_FORMATS` takes the other forms a server or a broker's table may
give). It agrees with the tables when it is the contract's listed expiry (the date in its
symbol) or its last trading day.

**A backtest with a client** (`check_listed_expiries`): for each futures or options
contract the run names that the server still lists (its last trading day on or after the
run's present, in IST), one `client.symbol` call and the same comparison; a disagreement is
a `ReferenceDataWarning` and the run goes on with the tables, which hold the dated history
the master contract does not. An expired contract is not in the master contract, so it is
not asked about; a client without `symbol` (a user's own data source) and a run of cash
equity make no call. A lookup that fails is a warning too: the contract was not checked.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping
from typing import Any

from ._series import ist_time
from .reference import Instrument
from .reference._store import warn

#: How a master contract writes a contract's expiry: OpenAlgo's "29-DEC-26" first, then
#: the forms an older server or a broker's table may give it in.
EXPIRY_FORMATS = ("%d-%b-%y", "%d-%b-%Y", "%Y-%m-%d", "%d%b%y")
#: What the run does about a disagreement, as its warning says.
_GOES_ON = (
    "the run uses the reference tables, which hold the dated history; upgrade openalgo "
    "(pip install -U openalgo), and if it still disagrees, report the table row with the "
    "exchange's circular"
)


def master_expiry(value: Any) -> dt.date | None:
    """The date of an expiry as a master contract writes it (`EXPIRY_FORMATS`), None for
    none or one that cannot be read."""
    if not isinstance(value, str) or not value.strip():
        return None
    for form in EXPIRY_FORMATS:
        try:
            return dt.datetime.strptime(value.strip(), form).date()
        except ValueError:
            continue
    return None


def agrees(contract: Instrument, day: dt.date) -> bool:
    """True when `day` is the contract's listed expiry or its last trading day."""
    return day in (contract.expiry, contract.last_trading_day)


def tables_words(contract: Instrument) -> str:
    """The tables' dates of a contract in words: its listed expiry, and its last trading
    day when that differs."""
    listed, last = contract.expiry, contract.last_trading_day
    return f"listed expiry {listed}" + ("" if last == listed else f", last trading day {last}")


def check_listed_expiries(instruments: Iterable[Instrument], client: Any, present_ns: int) -> None:
    """Warn (`ReferenceDataWarning`) for each listed contract of `instruments` whose expiry
    in `client`'s master contract disagrees with the tables, or cannot be looked up, as of
    `present_ns` (UTC nanoseconds; module docstring)."""
    lookup = getattr(client, "symbol", None)
    if not callable(lookup):
        return
    today = ist_time(present_ns).date()
    for contract in instruments:
        if contract.expiry is None or contract.last_trading_day < today:
            continue
        answer = lookup(symbol=contract.symbol, exchange=contract.exchange)
        note = _disagreement(contract, answer)
        if note is not None:
            warn(note)


def _disagreement(contract: Instrument, answer: Any) -> str | None:
    """What a backtest warns of a contract's master contract `answer`, None when it
    agrees."""
    where = f"{contract.exchange}:{contract.symbol}"
    data = answer.get("data") if isinstance(answer, Mapping) else None
    if not isinstance(data, Mapping):
        return (
            f"{where}: its expiry could not be checked against OpenAlgo's master contract "
            f"({answer!r}); the run uses the reference tables ({tables_words(contract)})"
        )
    day = master_expiry(data.get("expiry"))
    if day is None or agrees(contract, day):
        return None
    return (
        f"{where}: OpenAlgo's master contract gives the expiry {day}, the reference tables "
        f"{tables_words(contract)}: {_GOES_ON}"
    )


__all__ = [
    "EXPIRY_FORMATS",
    "agrees",
    "check_listed_expiries",
    "master_expiry",
    "tables_words",
]
