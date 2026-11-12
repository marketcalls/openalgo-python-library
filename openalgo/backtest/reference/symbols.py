"""OpenAlgo symbol parsing and formatting.

The OpenAlgo symbol string, together with the exchange code, is the primary key of an
instrument. The formats are:

    equity   [Base]                              INFY on NSE, M&M on NSE
    index    [Base]                              NIFTY on NSE_INDEX
    future   [Base][DDMMMYY]FUT                  BANKNIFTY24APR24FUT on NFO
    option   [Base][DDMMMYY][Strike][CE or PE]   NIFTY28NOV2424000CE on NFO
                                                 VEDL25APR24292.5CE on NFO

The exchange decides the family: NSE and BSE symbols are equities, the *_INDEX
exchanges hold indices, and every symbol on a derivatives exchange must parse as a
future or an option. The base may contain digits, hyphens and ampersands
(726GS2033, BAJAJ-AUTO, M&M); the two digits immediately before the month are always
the day. A strike is written without padding, and a decimal strike drops trailing
zeros (292.5, never 292.50).

`format_symbol(...)` is the exact inverse of `parse_symbol(...)` for every symbol in
canonical form, and a symbol that is not in canonical form is rejected with a
`SymbolError` that says what is wrong. A symbol written as a derivative is refused on
a cash or index exchange, so a derivative is never mistaken for a share.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from ..errors import ConfigError

MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
_MONTH_NUMBER = {name: number for number, name in enumerate(MONTHS, start=1)}

EQUITY_EXCHANGES = ("NSE", "BSE")
INDEX_EXCHANGES = ("NSE_INDEX", "BSE_INDEX", "MCX_INDEX", "GLOBAL_INDEX")
# Equity derivatives exchanges: index and stock futures and options, quantity in units
# of the underlying. The built-in lot, freeze, tick and settlement tables cover these.
LISTED_DERIVATIVE_EXCHANGES = ("NFO", "BFO")
DERIVATIVE_EXCHANGES = (*LISTED_DERIVATIVE_EXCHANGES, "MCX", "CDS", "BCD", "NCDEX")

# Exchanges OpenAlgo knows whose symbols this parser does not handle yet.
UNSUPPORTED_EXCHANGES = {
    "NCO": (
        "NSE commodity symbols are documented with a five character expiry (26MAY) "
        "that does not identify a contract; they are not supported"
    ),
    "CRYPTO": "crypto symbols (perpetuals have no expiry) are not supported yet",
}

# Index exchange code to the exchange its derivatives trade on, as the server maps them.
DERIVATIVE_EXCHANGE_OF = {"NSE_INDEX": "NFO", "BSE_INDEX": "BFO", "NSE": "NFO", "BSE": "BFO"}

_DERIVATIVE_RE = re.compile(
    r"^(?P<base>.+?)"
    r"(?P<day>\d{2})(?P<month>" + "|".join(MONTHS) + r")(?P<year>\d{2})"
    r"(?:(?P<fut>FUT)|(?P<strike>\d+(?:\.\d+)?)(?P<option_type>CE|PE))$"
)
_EXPIRY_CODE_RE = re.compile(r"^(\d{2})-?(" + "|".join(MONTHS) + r")-?(\d{2})$")


class SymbolError(ConfigError):
    """A symbol or exchange code is malformed or not supported."""


@dataclass(frozen=True)
class ParsedSymbol:
    """The parts of an OpenAlgo symbol.

    `underlying` is the base of a future or an option and None for an equity or an
    index. `strike` is in rupees. `symbol` is the string exactly as it was given.
    """

    symbol: str
    exchange: str
    kind: str
    underlying: str | None = None
    expiry: _dt.date | None = None
    strike: float | None = None
    option_type: str | None = None

    @property
    def canonical(self) -> str:
        """The symbol rebuilt from its parts."""
        if self.kind in ("equity", "index"):
            return self.symbol
        return format_symbol(self.underlying, self.expiry, self.strike, self.option_type)


def format_expiry(expiry: _dt.date) -> str:
    """A date as the seven character expiry code used inside symbols, for example 28NOV24.

    Raises `SymbolError` when `expiry` is not a date or its year is outside 2000 to 2099.
    """
    if isinstance(expiry, _dt.datetime):
        expiry = expiry.date()
    if not isinstance(expiry, _dt.date):
        raise SymbolError(f"expiry must be a date, got {expiry!r}")
    if not 2000 <= expiry.year <= 2099:
        raise SymbolError(
            f"expiry year {expiry.year} cannot be written with two digits (2000 to 2099)"
        )
    return f"{expiry.day:02d}{MONTHS[expiry.month - 1]}{expiry.year % 100:02d}"


def parse_expiry(code: str) -> _dt.date:
    """An expiry code to a date. Accepts 28NOV24 and the symbol-master form 28-NOV-24.

    Raises `SymbolError` when the code is not of that form or is not a real date.
    """
    if not isinstance(code, str):
        raise SymbolError(f"expiry code must be a string such as '28NOV24', got {code!r}")
    match = _EXPIRY_CODE_RE.match(code.strip().upper())
    if not match:
        raise SymbolError(f"expiry code {code!r} is not of the form DDMMMYY (for example 28NOV24)")
    day, month, year = match.groups()
    try:
        return _dt.date(2000 + int(year), _MONTH_NUMBER[month], int(day))
    except ValueError:
        raise SymbolError(f"expiry code {code!r} is not a real calendar date") from None


def _strike_decimal(strike: int | float | str | Decimal) -> Decimal:
    if isinstance(strike, bool):
        raise SymbolError(f"strike must be a number, got {strike!r}")
    try:
        if isinstance(strike, Decimal):
            value = strike
        elif isinstance(strike, int):
            value = Decimal(strike)
        elif isinstance(strike, float):
            # repr() is the shortest text that reads back as the same float, so
            # 292.5 stays 292.5 and never becomes 292.50000000000003.
            value = Decimal(repr(strike))
        else:
            value = Decimal(str(strike).strip())
    except (InvalidOperation, ValueError):
        raise SymbolError(f"strike {strike!r} is not a number") from None
    if not value.is_finite() or value <= 0:
        raise SymbolError(f"strike must be a positive finite number, got {strike!r}")
    return value


def format_strike(strike: int | float | str | Decimal) -> str:
    """A strike in rupees as it appears in a symbol: no padding, no trailing zeros (292.5).

    Raises `SymbolError` when `strike` is not a positive finite number.
    """
    return format(_strike_decimal(strike).normalize(), "f")


def format_symbol(
    underlying: str,
    expiry: _dt.date | None = None,
    strike: int | float | str | Decimal | None = None,
    option_type: str | None = None,
) -> str:
    """Build an OpenAlgo symbol from its parts.

    `format_symbol("INFY")` is the equity or index symbol,
    `format_symbol("BANKNIFTY", date(2024, 4, 24))` is `BANKNIFTY24APR24FUT` and
    `format_symbol("NIFTY", date(2024, 11, 28), 24000, "CE")` is `NIFTY28NOV2424000CE`.
    `strike` is in rupees. Raises `SymbolError` when the parts do not make a symbol (a
    strike without an option type, an option type other than CE or PE, a bad base).
    """
    _check_base(underlying, "underlying")
    if expiry is None:
        if strike is not None or option_type is not None:
            raise SymbolError("a strike or an option type needs an expiry date")
        return underlying
    code = format_expiry(expiry)
    if strike is None and option_type is None:
        return f"{underlying}{code}FUT"
    if strike is None or option_type is None:
        raise SymbolError("an option needs both a strike and an option type ('CE' or 'PE')")
    if option_type not in ("CE", "PE"):
        raise SymbolError(f"option type must be 'CE' or 'PE', got {option_type!r}")
    return f"{underlying}{code}{format_strike(strike)}{option_type}"


def _check_base(text, what: str) -> None:
    if not isinstance(text, str) or not text:
        raise SymbolError(f"{what} must be a non-empty string, got {text!r}")
    if not text.isascii() or not text.isprintable() or any(ch.isspace() for ch in text):
        raise SymbolError(f"{what} {text!r} contains spaces or characters outside printable ASCII")
    if text != text.upper():
        raise SymbolError(f"{what} {text!r} is not upper case; OpenAlgo symbols are upper case")


def normalise_exchange(exchange: str) -> str:
    """Check an exchange code and return it. Raises `SymbolError` for an unknown code."""
    if not isinstance(exchange, str) or not exchange:
        raise SymbolError(f"exchange must be a string such as 'NSE' or 'NFO', got {exchange!r}")
    if (
        exchange in EQUITY_EXCHANGES
        or exchange in INDEX_EXCHANGES
        or exchange in DERIVATIVE_EXCHANGES
    ):
        return exchange
    if exchange in UNSUPPORTED_EXCHANGES:
        raise SymbolError(f"exchange {exchange!r}: {UNSUPPORTED_EXCHANGES[exchange]}")
    known = ", ".join(EQUITY_EXCHANGES + INDEX_EXCHANGES + DERIVATIVE_EXCHANGES)
    raise SymbolError(f"unknown exchange {exchange!r}; expected one of: {known}")


def derivative_exchange(exchange: str) -> str:
    """The exchange an underlying's derivatives trade on (NSE_INDEX and NSE give NFO).

    A derivatives exchange is returned as it is. Raises `SymbolError` for an unknown code.
    """
    exchange = normalise_exchange(exchange)
    return DERIVATIVE_EXCHANGE_OF.get(exchange, exchange)


def parse_symbol(symbol: str, exchange: str, strict: bool = True) -> ParsedSymbol:
    """Split an OpenAlgo symbol into its parts.

    With `strict` (the default) the symbol must be in canonical form, so that
    `parse_symbol(s, ex).canonical == s` always holds. `strict=False` also accepts a
    strike written with padding or trailing zeros (292.50); `canonical` then differs
    from the given string.

    Raises `SymbolError` for an unknown or unsupported exchange, for a symbol that is
    empty or not upper case, for a symbol written as a derivative on a cash or index
    exchange, and (on a derivatives exchange) for a symbol that is not a future or an
    option with a real expiry date. Whether that date is an expiry of the underlying is
    checked by `ReferenceTables.instrument`, not here.
    """
    exchange = normalise_exchange(exchange)
    _check_base(symbol, "symbol")

    if exchange in EQUITY_EXCHANGES or exchange in INDEX_EXCHANGES:
        if _DERIVATIVE_RE.match(symbol):
            raise SymbolError(
                f"{symbol!r} is written as a derivative ([Base][DDMMMYY]FUT or "
                f"[Base][DDMMMYY][Strike][CE or PE]) but {exchange} lists shares or indices; "
                f"derivatives trade on {', '.join(DERIVATIVE_EXCHANGES)}"
            )
        kind = "equity" if exchange in EQUITY_EXCHANGES else "index"
        return ParsedSymbol(symbol=symbol, exchange=exchange, kind=kind)

    match = _DERIVATIVE_RE.match(symbol)
    if not match:
        raise SymbolError(
            f"{symbol!r} is not a valid {exchange} symbol: expected [Base][DDMMMYY]FUT "
            "(for example BANKNIFTY24APR24FUT) or [Base][DDMMMYY][Strike][CE or PE] "
            "(for example NIFTY28NOV2424000CE)"
        )
    underlying = match.group("base")
    code = match.group("day") + match.group("month") + match.group("year")
    try:
        expiry = parse_expiry(code)
    except SymbolError:
        raise SymbolError(
            f"{symbol!r} has the expiry code {code!r}, which is not a real calendar date"
        ) from None

    if match.group("fut"):
        return ParsedSymbol(
            symbol=symbol, exchange=exchange, kind="future", underlying=underlying, expiry=expiry
        )

    return ParsedSymbol(
        symbol=symbol,
        exchange=exchange,
        kind="option",
        underlying=underlying,
        expiry=expiry,
        strike=_parse_strike(symbol, match.group("strike"), strict),
        option_type=match.group("option_type"),
    )


def _parse_strike(symbol: str, strike_text: str, strict: bool) -> float:
    """The strike in rupees from its text inside `symbol` (canonical text only when strict)."""
    try:
        value = _strike_decimal(strike_text)
    except SymbolError:
        raise SymbolError(
            f"{symbol!r} has the strike {strike_text!r}; a strike must be a positive number"
        ) from None
    canonical = format(value.normalize(), "f")
    if strict and strike_text != canonical:
        raise SymbolError(
            f"{symbol!r} writes the strike as {strike_text!r}; the OpenAlgo form is "
            f"{canonical!r} (no padding and no trailing zeros). Pass strict=False to accept it."
        )
    return float(value)


__all__ = [
    "DERIVATIVE_EXCHANGES",
    "EQUITY_EXCHANGES",
    "INDEX_EXCHANGES",
    "LISTED_DERIVATIVE_EXCHANGES",
    "MONTHS",
    "UNSUPPORTED_EXCHANGES",
    "ParsedSymbol",
    "SymbolError",
    "derivative_exchange",
    "format_expiry",
    "format_strike",
    "format_symbol",
    "normalise_exchange",
    "parse_expiry",
    "parse_symbol",
]
