"""Quantities counted in lots, for the cost helpers' `lots=` argument.

A commodity or currency trader counts in lots, as a broker's calculator does (its
quantity 1 is one lot). The cost engine counts price units, so that price times quantity
is the traded value. One lot of a contract is `lot_size x multiplier` price units on a
trade date, both read from the reference tables a backtest reads (`openalgo.backtest.
reference`): one ALUMINI lot (1 MT quoted per kg) is 1 x 1,000 = 1,000 units, one COPPER
lot is 2,500 x 1 units, one USDINR lot (USD 1,000) is 1 x 1,000 units.

The lot unit itself follows OpenAlgo, which counts quantity in units for every broker and
converts MCX quantity to contracts only inside its Zerodha adapter (server commit
cf27e5f98, 2026-09-22, broker/zerodha/mapping/mcx_contract_size.py); its master contract
carries the real market lot (CRUDEOIL 100, COPPER 2,500, GOLDM 100, ALUMINI 1) and lot 1
for a currency contract. Everywhere else the engine keeps units, and strategies already
take `lots=` on their orders.

An MCX or NSE currency option is counted at the lot and multiplier of its underlying's
futures row (`mcx_contracts`, `currency_contracts`): an option on MCX or CDS has the
contract size of its underlying's future today, and the calculator figures of the COPPER
and USDINR options rest on it. The tables carry no option rows, so a backtest of such an
option still needs a `symbols` row for the exact contract and raises ReferenceDataError
without one.
"""

import datetime as dt
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from ..errors import CostError
from ..reference import default_tables
from ..reference.symbols import EQUITY_EXCHANGES, parse_symbol
from .inputs import common_length, day_to_date, floats, segment_ids, to_days
from .kernel import SEGMENT_TABLE

__all__ = ["INSTRUMENT_CODES", "Contract", "check_segment", "contract_of", "price_units"]

#: The cost model's instrument code of each instrument kind (`costs.segment_for`).
INSTRUMENT_CODES = MappingProxyType({"equity": "EQ", "future": "FUT", "option": "OPT"})
# The instrument code of each brokerage class of the segment table (`costs.segments()`).
_CODE_OF_CLASS = MappingProxyType(
    {"delivery": "EQ", "intraday": "EQ", "futures": "FUT", "options": "OPT"}
)
_KIND_WORDS = MappingProxyType({"EQ": "equity", "FUT": "futures", "OPT": "options"})


@dataclass(frozen=True)
class Contract:
    """The contract `symbol=` and `exchange=` name: what `lots=` counts, and what sets the
    instrument code and underlying of its cost segment when only the product is given."""

    symbol: str
    exchange: str
    instrument: str
    underlying: str | None


def contract_of(symbol: str | None, exchange: str | None) -> Contract | None:
    """The contract an OpenAlgo `symbol` on `exchange` names, or None without a symbol.

    Raises CostError for a symbol without its exchange or one that is not text and for an
    index, and the reference tables' `SymbolError` for a symbol that is not an OpenAlgo
    symbol.
    """
    if symbol is None:
        return None
    if not isinstance(symbol, str) or not isinstance(exchange, str):
        raise CostError(
            "symbol= needs exchange=: the OpenAlgo symbol and the exchange it trades on, "
            f"such as symbol='USDINR29DEC26FUT', exchange='CDS'; got {symbol!r} on {exchange!r}"
        )
    parsed = parse_symbol(symbol, exchange.strip().upper())
    if parsed.kind not in INSTRUMENT_CODES:
        raise CostError(
            f"{parsed.symbol} on {parsed.exchange} is an {parsed.kind}, which is not traded: "
            "give a share, futures or options symbol"
        )
    return Contract(
        parsed.symbol, parsed.exchange, INSTRUMENT_CODES[parsed.kind], parsed.underlying
    )


def check_segment(contract: Contract | None, segment: Any) -> None:
    """Refuse a segment given with `symbol=` that is not the contract's.

    `segment` is a code or an array of codes or ids, as the cost helpers take it. Each
    must price the contract's exchange and kind (equity, futures or options), as its row of
    `costs.segments()` says. Raises CostError for an unknown segment and for one that is
    not the contract's; does nothing without a contract.
    """
    if contract is None:
        return
    size = 1 if isinstance(segment, str) else int(np.size(segment))
    for index in np.unique(segment_ids(segment, size)).tolist():
        row = SEGMENT_TABLE[index]
        code = _CODE_OF_CLASS[row["brokerage_class"]]
        if (row["exchange"], code) == (contract.exchange, contract.instrument):
            continue
        kind = _KIND_WORDS[contract.instrument]
        raise CostError(
            f"segment {row['code']!r} prices {row['exchange']} {_KIND_WORDS[code]}, not "
            f"{contract.symbol}, which is {contract.exchange} {kind}: leave segment= out and "
            f"give product= (the symbol sets the rest), or give a {contract.exchange} {kind} "
            "segment (see costs.segments())"
        )


def _units_per_lot(contract: Contract, day: dt.date) -> float:
    """Price units in one lot of `contract` on the trade date `day`."""
    if contract.exchange in EQUITY_EXCHANGES:
        raise CostError(
            "lots= counts futures and options lots; give an equity's quantity in shares "
            "with quantity="
        )
    tables = default_tables()
    lot_size = tables.lot_size(contract.symbol, contract.exchange, day)
    return lot_size * tables.multiplier(contract.symbol, contract.exchange, day)


def _whole_lots(lots: Any, length: int) -> np.ndarray:
    """`lots` as a float array of `length`, every element a whole number zero or more."""
    counted = floats(lots, length, "lots")
    if not np.all((counted >= 0) & (counted == np.floor(counted))):
        raise CostError(f"lots must be a whole number of lots that is zero or more, got {lots!r}")
    return counted


def price_units(quantity: Any, lots: Any, contract: Contract | None, date: Any) -> Any:
    """The quantity in price units: `quantity` as given, or `lots` of `contract` on each
    trade date in `date` (a scalar for scalar input, else an array).

    Raises CostError unless exactly one of `quantity` and `lots` is given, for `lots`
    without a contract, for part or negative lots and for lots of cash equity; and the
    reference tables' `ReferenceDataError` for a contract they do not cover on a date, the
    error a backtest of that contract raises.
    """
    if (quantity is None) == (lots is None):
        raise CostError(
            "pass either quantity (in price units) or lots (with symbol= and exchange=), "
            "and not both"
        )
    if lots is None:
        return quantity
    if contract is None:
        raise CostError(
            "lots= needs symbol= and exchange=: the contract whose lot size and multiplier "
            "turn lots into price units"
        )
    length = common_length(lots, date)
    days = to_days(date, length)
    per_day = {day: _units_per_lot(contract, day_to_date(day)) for day in np.unique(days).tolist()}
    units = _whole_lots(lots, length) * np.array([per_day[day] for day in days.tolist()])
    return float(units[0]) if np.ndim(lots) == 0 and length == 1 else units
