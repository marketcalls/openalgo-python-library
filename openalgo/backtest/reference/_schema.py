"""Shape of every reference table: its key, its value columns and how a cell is parsed.

This module holds no market facts, only the layout of the CSV files under `data/`.
Every table shares these columns:

    valid_from, valid_to   inclusive ISO dates; an empty valid_to means "until further
                           notice"
    source                 the exchange document the row was taken from
    status                 verified (checked against that document), unverified, or user
                           (a row from a user file)
    note                   free text

Two tables carry a second range next to the validity range (`TableSpec.span`): an
expiry range (inclusive dates) for lot sizes and MCX contracts, and a price slab (lower
bound inclusive, upper bound exclusive) for tick sizes.
"""

import datetime as _dt
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ..errors import ConfigError
from .instruments import DERIVATIVE_SETTLEMENTS

# Anything a caller may pass where a calendar date is expected.
DateLike = _dt.date | _dt.datetime | str

BUILTIN_STATUSES = ("verified", "unverified")
USER_STATUS = "user"

# Position in this tuple equals datetime.date.weekday().
WEEKDAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
# Weekday value of an expiry rule for a period in which the series is not listed.
NO_SERIES = "NONE"
SERIES = ("weekly", "monthly")
UNDERLYING_KINDS = ("index", "stock", "etf")
# What an exact-symbol row may call its underlying; MCX and currency contracts included.
SYMBOL_KINDS = (*UNDERLYING_KINDS, "commodity", "currency")
# Segments the tick-size and settlement rules are keyed by.
SEGMENTS = ("equity", "etf", "index_future", "stock_future", "index_option", "stock_option")

SPAN_DATE = "date"
SPAN_PRICE = "price"
_ISO_DATE_LENGTH = len("2024-11-28")

# The prices an exchange measures an option price band from, and when a stop-limit order
# meets the band (the price_protection table documents each word).
_BAND_BASES = ("reference_price", "trigger_price", "cooling_off_ltp")
_STOP_LIMIT_CHECKS = ("on_trigger", "on_entry")


def as_date(value: DateLike, what: str = "date") -> _dt.date:
    """Return `value` as a calendar date (an IST session date wherever one is meant).

    Accepts a date, a datetime (its date part, with no time zone conversion) and an ISO
    string such as "2024-11-28". Raises `ConfigError` for anything else; `what` names the
    argument in the message.
    """
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    try:
        return _dt.date.fromisoformat(str(value)[:_ISO_DATE_LENGTH])
    except (TypeError, ValueError):
        raise ConfigError(
            f"{what} must be a date or an ISO string such as '2024-11-28', got {value!r}"
        ) from None


def parse_date(text: str) -> _dt.date:
    """An ISO date cell (YYYY-MM-DD). Raises `ValueError` otherwise."""
    return _dt.date.fromisoformat(text)


def _finite(text: str) -> float:
    number = float(text)
    if not math.isfinite(number):
        raise ValueError(f"expected a finite number, got {text!r}")
    return number


def _positive_int(text: str) -> int:
    number = _finite(text)
    if number != int(number) or number < 1:
        raise ValueError(f"expected a whole number, 1 or more, got {text!r}")
    return int(number)


def _positive_float(text: str) -> float:
    number = _finite(text)
    if number <= 0:
        raise ValueError(f"expected a positive number, got {text!r}")
    return number


def parse_price_bound(text: str) -> float:
    """A slab bound in rupees, 0 or more. Raises `ValueError` otherwise."""
    number = _finite(text)
    if number < 0:
        raise ValueError(f"expected a price of 0 or more, got {text!r}")
    return number


def _weekday(text: str) -> str:
    value = text.upper()
    if value != NO_SERIES and value not in WEEKDAYS:
        raise ValueError(f"expected one of {WEEKDAYS} or {NO_SERIES}, got {text!r}")
    return value


def _one_of(allowed: Sequence[str]) -> Callable[[str], str]:
    def parse(text: str) -> str:
        value = text.lower()
        if value not in allowed:
            raise ValueError(f"expected one of {tuple(allowed)}, got {text!r}")
        return value

    return parse


def _text(text: str) -> str:
    return text


@dataclass(frozen=True)
class Column:
    """A value column: its name, the parser of a cell, and whether it may be empty."""

    name: str
    parse: Callable[[str], object]
    required: bool = True


@dataclass(frozen=True)
class TableSpec:
    """Layout of one reference table.

    `key` columns say what a row is about. `lower_key` names the key columns whose
    values are lower case (all others are upper case); `date_keys` names key columns
    that hold an ISO date. `span` is (low column, high column, SPAN_DATE or SPAN_PRICE)
    when a row is limited to a second range. `builtin` is False for a table that has no
    file in the package.
    """

    name: str
    key: tuple[str, ...]
    columns: tuple[Column, ...]
    description: str
    span: tuple[str, str, str] | None = None
    lower_key: tuple[str, ...] = ()
    date_keys: tuple[str, ...] = ()
    builtin: bool = True

    @property
    def header(self) -> tuple[str, ...]:
        """Every column of the table, in file order."""
        span = self.span[:2] if self.span else ()
        values = tuple(column.name for column in self.columns)
        return (*self.key, "valid_from", "valid_to", *span, *values, "source", "status", "note")

    def normalise_key(self, parts: Sequence[object]) -> tuple[str, ...]:
        """Key values in the form the table stores them.

        Raises `ConfigError` on a key of the wrong length and `ValueError` on a date key
        that is not an ISO date.
        """
        if len(parts) != len(self.key):
            raise ConfigError(f"the key of '{self.name}' is {self.key}, got {tuple(parts)!r}")
        normalised = []
        for name, part in zip(self.key, parts, strict=True):
            if name in self.date_keys:
                normalised.append(as_date(part, name).isoformat())
            elif name in self.lower_key:
                normalised.append(str(part).lower())
            else:
                normalised.append(str(part).upper())
        return tuple(normalised)

    def key_of(self, record: Mapping[str, str]) -> tuple[str, ...]:
        """The normalised key of a CSV record."""
        return self.normalise_key([record[name] for name in self.key])


_UNDERLYING_EXCHANGE = ("underlying", "exchange")
# The columns of a contract-terms table (MCX futures, NSE currency futures): one layout,
# so the two tables are read and looked up the same way.
_CONTRACT_COLUMNS = (
    Column("lot_size", _positive_int),
    Column("multiplier", _positive_float),
    Column("tick_size", _positive_float),
    Column("freeze_qty", _positive_int, required=False),
    Column("settlement", _one_of(DERIVATIVE_SETTLEMENTS), required=False),
    Column("trading_unit", _text, required=False),
    Column("quote_unit", _text, required=False),
)

_SPECS = (
    TableSpec(
        name="lot_sizes",
        key=_UNDERLYING_EXCHANGE,
        span=("expiry_from", "expiry_to", SPAN_DATE),
        columns=(Column("lot_size", _positive_int),),
        description=(
            "Market lot in units per underlying. valid_from and valid_to are trade dates; "
            "expiry_from and expiry_to limit the row to contracts expiring in that range"
        ),
    ),
    TableSpec(
        name="underlyings",
        key=_UNDERLYING_EXCHANGE,
        columns=(Column("kind", _one_of(UNDERLYING_KINDS)),),
        description=(
            "What an instrument is: an index, a stock or an ETF. The exchange is the one its "
            "derivatives trade on (NFO, BFO) or, for a cash instrument, NSE or BSE"
        ),
    ),
    TableSpec(
        name="freeze_qty",
        key=_UNDERLYING_EXCHANGE,
        columns=(
            Column("freeze_qty_futures", _positive_int),
            Column("freeze_qty_options", _positive_int),
        ),
        description="Largest quantity in units the exchange accepts in one order, by kind",
    ),
    TableSpec(
        name="strike_steps",
        key=_UNDERLYING_EXCHANGE,
        columns=(
            Column("strike_step", _positive_float),
            Column("scope", _text, required=False),
        ),
        description="Strike interval in rupees of the near contracts per underlying",
    ),
    TableSpec(
        name="expiry_rules",
        key=(*_UNDERLYING_EXCHANGE, "series"),
        lower_key=("series",),
        columns=(Column("weekday", _weekday), Column("listed_from", parse_date, required=False)),
        description=(
            "Expiry weekday per series; the validity range is over expiry dates. weekly: "
            "every such weekday, except in a week that holds the monthly expiry; monthly: "
            "the last such weekday of the month; NONE: the series is not listed. "
            "listed_from: the first trade date on which the row's contracts were listed"
        ),
    ),
    TableSpec(
        name="contract_dates",
        key=(*_UNDERLYING_EXCHANGE, "contract"),
        date_keys=("contract",),
        columns=(
            Column("expiry_date", parse_date),
            Column("last_trading_day", parse_date),
        ),
        description=(
            "Dates of one contract as they stood on a range of trade dates, where the weekday "
            "rule and the calendar do not give them: its listed expiry and its last trading "
            "day. The contract is named by the date its weekday rule gives"
        ),
    ),
    TableSpec(
        name="tick_sizes",
        key=("exchange", "segment"),
        lower_key=("segment",),
        span=("price_from", "price_to", SPAN_PRICE),
        columns=(
            Column("tick_size", _positive_float),
            Column("basis", _text, required=False),
        ),
        description=(
            "Tick size in rupees by price slab (price_from inclusive, price_to exclusive); "
            "basis says which price selects the slab"
        ),
    ),
    TableSpec(
        name="settlement",
        key=("exchange", "segment"),
        lower_key=("segment",),
        columns=(Column("settlement", _one_of(DERIVATIVE_SETTLEMENTS)),),
        description="How a derivative segment settles at expiry: cash or physical delivery",
    ),
    TableSpec(
        name="mcx_contracts",
        key=_UNDERLYING_EXCHANGE,
        span=("expiry_from", "expiry_to", SPAN_DATE),
        columns=_CONTRACT_COLUMNS,
        description=(
            "MCX futures terms by trade date and contract expiry. lot_size is the trading "
            "unit in the unit OpenAlgo counts quantity in (CRUDEOIL 100 barrels, GOLD 1 kg); "
            "multiplier is the rupee value of a one-rupee move in the quoted price for one "
            "unit of quantity"
        ),
    ),
    TableSpec(
        name="currency_contracts",
        key=_UNDERLYING_EXCHANGE,
        span=("expiry_from", "expiry_to", SPAN_DATE),
        columns=_CONTRACT_COLUMNS,
        description=(
            "NSE currency futures terms by trade date and contract expiry. lot_size is the "
            "contract size in units of the base currency, the unit OpenAlgo counts currency "
            "quantity in (USDINR 1,000 dollars, JPYINR 1,00,000 yen); multiplier is the rupee "
            "value of a one-rupee move in the quoted price for one unit of quantity (USDINR 1, "
            "JPYINR 0.01, quoted per 100 yen)"
        ),
    ),
    TableSpec(
        name="symbols",
        key=("symbol", "exchange"),
        columns=(
            Column("lot_size", _positive_int),
            Column("tick_size", _positive_float, required=False),
            Column("freeze_qty", _positive_int, required=False),
            Column("multiplier", _positive_float, required=False),
            Column("underlying_type", _one_of(SYMBOL_KINDS), required=False),
            Column("settlement", _one_of(DERIVATIVE_SETTLEMENTS), required=False),
            Column("last_trading_day", parse_date, required=False),
        ),
        description="Terms of one exact symbol (user files and symbol-master snapshots)",
        builtin=False,
    ),
    TableSpec(
        name="broker_units",
        key=("broker", "segment"),
        lower_key=("broker", "segment"),
        columns=(
            Column("lotsize_field", _text, required=False),
            Column("tick_size_unit", _text, required=False),
            Column("order_quantity_unit", _text, required=False),
            Column("live_status", _text, required=False),
        ),
        description=(
            "What the OpenAlgo server source shows about quantity and tick_size units per "
            "broker. Not confirmed against live brokers"
        ),
    ),
    TableSpec(
        name="price_protection",
        key=("exchange", "segment", "check", "underlying", "days"),
        lower_key=("segment", "check", "days"),
        columns=(
            Column("band_percent", _positive_float),
            Column("band_min_rupees", _positive_float, required=False),
            Column("basis", _one_of(_BAND_BASES)),
            Column("stop_limit_checked", _one_of(_STOP_LIMIT_CHECKS)),
        ),
        description=(
            "Price bands the exchanges apply to the limit price of an option order: the band "
            "is the larger of band_percent of the basis price and band_min_rupees (None: no "
            "minimum). underlying is one underlying or ALL for every underlying without a row "
            "of its own; days is every_day, or expiry_day for a row that applies only on a day "
            "an option on the underlying expires; stop_limit_checked says whether a stop-limit "
            "order meets the band when it triggers or when it is placed and modified"
        ),
    ),
)

TABLES: dict[str, TableSpec] = {spec.name: spec for spec in _SPECS}


def table_spec(table: str) -> TableSpec:
    """The layout of a table. Raises `ConfigError` for an unknown name."""
    try:
        return TABLES[table]
    except KeyError:
        raise ConfigError(
            f"unknown reference table {table!r}; expected one of {sorted(TABLES)}"
        ) from None


__all__ = [
    "BUILTIN_STATUSES",
    "NO_SERIES",
    "SEGMENTS",
    "SERIES",
    "SPAN_DATE",
    "SPAN_PRICE",
    "SYMBOL_KINDS",
    "TABLES",
    "UNDERLYING_KINDS",
    "USER_STATUS",
    "WEEKDAYS",
    "Column",
    "DateLike",
    "TableSpec",
    "as_date",
    "parse_date",
    "parse_price_bound",
    "table_spec",
]
