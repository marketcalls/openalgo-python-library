"""Expiry dates of an underlying's contracts, as they stood on any trade date.

A contract is named by its contract date: the date the weekday rules in
`expiry_rules.csv` give for its week or month (a weekly date in the week of a monthly
date is not a contract). Its listed expiry (the date in its symbol) and its last trading
day come from a `contract_dates` row covering the trade date asked about (a re-dated
contract, a special session, a year without a holiday list); otherwise both are the
contract date, moved back to the previous trading day when the exchange is closed then.

A contract is listed only from its first listing day: the first trade date of its
`contract_dates` rows, and no earlier than the `listed_from` date of its rule. Before
that day no schedule holds it and no symbol names it. How far ahead an exchange lists
contracts is not modelled: a schedule includes later contracts of a listed series.

The calendar is trusted only where it has a holiday list: a "template" date is never
used to place an expiry (`ReferenceDataError` names the exchange and the year), and an
"unverified" one is used with one `ReferenceDataWarning` per call.
"""

import datetime as _dt
from dataclasses import dataclass

from .._protocols import SessionCalendar
from ..errors import CalendarError, ConfigError, ReferenceDataError
from ._rows import RefRow
from ._schema import NO_SERIES, SERIES, WEEKDAYS
from ._store import RowStore, warn

RULES_TABLE = "expiry_rules"
DATES_TABLE = "contract_dates"
ALL_SERIES = "all"
NOT_LISTED = "not listed"  # a contract not yet listed on the trade date
TEMPLATE = "template"
UNVERIFIED = "unverified"

_ONE_DAY = _dt.timedelta(days=1)
_ONE_WEEK = _dt.timedelta(days=7)
# A re-dated contract sits within a week of its contract date; rule dates are read six
# days further, so that a monthly date just outside still removes its week's weekly date.
_WINDOW = _dt.timedelta(days=7)
_WEEK_REST = _dt.timedelta(days=6)
# Longest run of closed days to walk back over; a longer run means a broken calendar.
_MAX_CLOSED_RUN_DAYS = 15
_SATURDAY = 5


@dataclass(frozen=True)
class Expiry:
    """One contract of an underlying, with its dates as in force on one trade date.

    `contract_date` names the contract (the date its weekday rule gives). `expiry` is the
    listed date (in its symbol) and `last_trading_day` the day it stops trading; both
    are IST session dates. `source` and `status` come from the row that gave the dates.
    """

    underlying: str
    exchange: str
    series: str
    contract_date: _dt.date
    expiry: _dt.date
    last_trading_day: _dt.date
    source: str
    status: str


@dataclass(frozen=True)
class Context:
    """One lookup: the tables, the calendar, the trade date (None: the final dates)."""

    store: RowStore
    underlying: str
    exchange: str
    calendar: SessionCalendar
    as_of: _dt.date | None
    unverified: set

    def key(self, series: str) -> tuple[str, str, str]:
        return (self.underlying, self.exchange, series)

    def contract_row(self, contract: _dt.date, rule: RefRow | None = None):
        """The `contract_dates` row of a contract in force on the trade date.

        None when the rule and the calendar give its dates, and NOT_LISTED when the
        contract was not listed yet on the trade date (see the module docstring).
        """
        key = (self.underlying, self.exchange, contract.isoformat())
        if self.as_of is None:
            return self.store.first(DATES_TABLE, key, _dt.date.max)
        rows = self.store.rows(DATES_TABLE, key)
        listed_from = None if rule is None else rule.values.get("listed_from")
        if rows and self.as_of < min(row.valid_from for row in rows):
            return NOT_LISTED
        if listed_from is not None and self.as_of < listed_from:
            return NOT_LISTED
        return self.store.first(DATES_TABLE, key, self.as_of)


def check_calendar(calendar: object) -> None:
    """Raise `ConfigError` unless `calendar` has `session` and `coverage` methods."""
    for name in ("session", "coverage"):
        if not callable(getattr(calendar, name, None)):
            raise ConfigError(
                "expiry dates need a session calendar (an object with session(exchange, day) "
                "and coverage(exchange, day)) to place an expiry on a trading day"
            )


def _template_error(context: Context, day: _dt.date) -> ReferenceDataError:
    return ReferenceDataError(
        f"The calendar has no holiday list for {context.exchange} in {day.year} (coverage "
        f"'template'), so whether {day.isoformat()} is a trading day is not known and no "
        f"expiry of {context.underlying} can be placed by it. Supply a holiday list for "
        f"{day.year} to the calendar, or a 'contract_dates' row for the contract."
    )


def _is_open(context: Context, day: _dt.date) -> bool:
    """Whether the exchange trades on `day`, refusing a date the calendar cannot vouch for."""
    coverage = context.calendar.coverage(context.exchange, day)
    if coverage == TEMPLATE:
        raise _template_error(context, day)
    if coverage == UNVERIFIED:
        context.unverified.add(day)
    return context.calendar.session(context.exchange, day) is not None


def _previous_session(context: Context, day: _dt.date) -> _dt.date:
    current = day
    for _ in range(_MAX_CLOSED_RUN_DAYS):
        if _is_open(context, current):
            return current
        current -= _ONE_DAY
    raise CalendarError(
        f"the calendar has no {context.exchange} session in the {_MAX_CLOSED_RUN_DAYS} days up "
        f"to {day.isoformat()}; cannot place the expiry on a trading day"
    )


def _all_closed_after(context: Context, start: _dt.date, end: _dt.date) -> bool:
    """Whether every day after `start` up to `end` is closed.

    Days the calendar knows decide first: one open day answers False. Otherwise a
    weekday of a "template" year leaves the answer unknown and raises; a weekend day of
    such a year counts as closed.
    """
    unknown = []
    day = start + _ONE_DAY
    while day <= end:
        if context.calendar.coverage(context.exchange, day) == TEMPLATE:
            if day.weekday() < _SATURDAY:
                unknown.append(day)
        elif _is_open(context, day):
            return False
        day += _ONE_DAY
    if unknown:
        raise _template_error(context, unknown[0])
    return True


def _rule_dates(context: Context, first: _dt.date, last: _dt.date) -> dict:
    """Contract dates from `first - 7 days` to `last + 7 days`: {date: (series, rule row)}.

    Raises `ReferenceDataError` when a day from `first` to `last` has no rule row.
    """
    low, high = first - _WINDOW, last + _WINDOW
    found: dict[_dt.date, tuple[str, RefRow]] = {}
    for series in ("monthly", "weekly"):  # a date both series give is the monthly contract
        day = low - _WEEK_REST
        while day <= high + _WEEK_REST:
            row = context.store.first(RULES_TABLE, context.key(series), day)
            if row is None and first <= day <= last:
                raise context.store.missing(RULES_TABLE, context.key(series), day)
            if row is not None and _is_rule_date(day, series, row.values["weekday"]):
                found.setdefault(day, (series, row))
            day += _ONE_DAY
    monthly_weeks = {d.isocalendar()[:2] for d, (s, _) in found.items() if s == "monthly"}
    return {
        d: entry
        for d, entry in found.items()
        if low <= d <= high and (entry[0] == "monthly" or d.isocalendar()[:2] not in monthly_weeks)
    }


def _is_rule_date(day: _dt.date, series: str, weekday: str) -> bool:
    if weekday == NO_SERIES or WEEKDAYS[day.weekday()] != weekday:
        return False
    return series == "weekly" or (day + _ONE_WEEK).month != day.month


def _from_row(context: Context, contract: _dt.date, series: str, row: RefRow) -> Expiry:
    return Expiry(
        context.underlying,
        context.exchange,
        series,
        contract,
        row.values["expiry_date"],
        row.values["last_trading_day"],
        row.source,
        row.status,
    )


def _on(context: Context, contract, series: str, rule: RefRow, day: _dt.date) -> Expiry:
    return Expiry(
        context.underlying, context.exchange, series, contract, day, day, rule.source, rule.status
    )


def _in_range(context: Context, contract, entry, first, last) -> tuple[Expiry, RefRow] | None:
    """The contract's record and the row behind it when its last trading day is in range."""
    series, rule = entry
    row = context.contract_row(contract, rule)
    if row == NOT_LISTED:
        return None
    if row is not None:
        record = _from_row(context, contract, series, row)
        return (record, row) if first <= record.last_trading_day <= last else None
    if contract < first:
        return None  # a date taken from the rule only ever moves back
    if contract > last and not _all_closed_after(context, last, contract):
        return None
    record = _on(context, contract, series, rule, _previous_session(context, contract))
    return (record, rule) if first <= record.last_trading_day <= last else None


def _finish(context: Context, found: list[tuple[Expiry, RefRow]]) -> None:
    """Record the rows used, and warn about unverified calendar dates.

    A row counts as used for the trade date of the lookup or, for the final dates (no
    trade date), for the latest date it gave: its contract date for a rule row, the last
    trading day for a contract row.
    """
    latest: dict[int, tuple[RefRow, _dt.date]] = {}
    for record, row in found:
        day = context.as_of
        if day is None:
            day = record.contract_date if row.table == RULES_TABLE else record.last_trading_day
        if id(row) not in latest or latest[id(row)][1] < day:
            latest[id(row)] = (row, day)
    for row, day in latest.values():
        context.store.note_use(row, day)
    if context.unverified:
        dates = ", ".join(sorted(day.isoformat() for day in context.unverified))
        warn(
            f"reference data: expiry dates of {context.underlying} on {context.exchange} were "
            f"placed using unverified calendar dates ({dates}); confirm the holiday list.",
            stacklevel=5,
        )


def expiry_schedule(
    store: RowStore,
    underlying: str,
    exchange: str,
    first: _dt.date,
    last: _dt.date,
    calendar: SessionCalendar,
    series: str = ALL_SERIES,
    as_of: _dt.date | None = None,
    record_use: bool = True,
) -> list[Expiry]:
    """Contracts whose last trading day falls from `first` to `last`, in date order.

    `as_of` is a trade date: the dates are those in force then; None gives the final
    dates. `exchange` is the derivatives exchange (NFO, BFO) and is what the calendar is
    asked about. `record_use=False` lists the contracts without recording the rows as
    used and without warnings, for a caller that looks ahead past the dates it needs and
    records only the contracts it keeps (by listing each of them again). Raises
    `ConfigError` for a bad argument, `ReferenceDataError` when the rules do not cover a
    day of the range or an answer would rest on a "template" calendar date, and
    `CalendarError` when the calendar has no session to move an expiry back to.
    """
    if series != ALL_SERIES and series not in SERIES:
        raise ConfigError(f"series must be 'all', 'weekly' or 'monthly', got {series!r}")
    check_calendar(calendar)
    if last < first:
        raise ConfigError(f"end {last} is before start {first}")
    context = Context(store, underlying, exchange, calendar, as_of, set())
    found = []
    for contract, entry in sorted(_rule_dates(context, first, last).items()):
        if series in (ALL_SERIES, entry[0]):
            result = _in_range(context, contract, entry, first, last)
            found += [] if result is None else [result]
    if record_use:
        _finish(context, found)
    return sorted((record for record, _ in found), key=lambda item: item.last_trading_day)


def contract_for(
    store: RowStore,
    underlying: str,
    exchange: str,
    expiry: _dt.date,
    on_date: _dt.date,
    calendar: SessionCalendar,
) -> Expiry | None:
    """The contract listed with `expiry` on trade date `on_date`, or None if there was none.

    A `contract_dates` row in force on `on_date` with that listed expiry answers first.
    Otherwise the nearest rule date on or after `expiry` without such a row answers
    when the exchange trades on `expiry` and is closed on every day after it up to that
    rule date. Raises `ReferenceDataError` when the rules do not cover `expiry` or the
    answer would rest on a "template" calendar date.
    """
    check_calendar(calendar)
    context = Context(store, underlying, exchange, calendar, on_date, set())
    contracts = sorted(_rule_dates(context, expiry, expiry).items())
    rows = {contract: context.contract_row(contract, rule) for contract, (_, rule) in contracts}
    match = None
    for contract, (series, _) in contracts:
        row = rows[contract]
        if isinstance(row, RefRow) and row.values["expiry_date"] == expiry:
            match = (_from_row(context, contract, series, row), row)
    derived = [item for item in contracts if rows[item[0]] is None and item[0] >= expiry]
    if match is None and derived:
        contract, (series, rule) = derived[0]
        if _is_open(context, expiry) and _all_closed_after(context, expiry, contract):
            match = (_on(context, contract, series, rule, expiry), rule)
    if match is not None:
        _finish(context, [match])
    return None if match is None else match[0]


def _named_by(store: RowStore, final: Context, contract: _dt.date, expiry: _dt.date) -> bool:
    """Whether `expiry` was the listed expiry of a re-dated contract on some trade date."""
    rows = store.rows(DATES_TABLE, (final.underlying, final.exchange, contract.isoformat()))
    if not rows:
        return False
    listed = {row.values["expiry_date"] for row in rows}
    if final.contract_row(contract) is None:
        listed.add(_previous_session(final, contract))
    return expiry in listed


def alias_for(
    store: RowStore,
    underlying: str,
    exchange: str,
    expiry: _dt.date,
    on_date: _dt.date,
    calendar: SessionCalendar,
) -> Expiry | None:
    """The contract a symbol with listed expiry `expiry` names, with its dates on `on_date`.

    For a symbol `contract_for` does not find: a re-dated contract's other symbols (its
    listed expiry on other trade dates) still name it. None when no contract was ever
    listed with `expiry`, or the one that was had not been listed by `on_date`. Raises
    `ReferenceDataError` when two contracts were listed with `expiry`.
    """
    check_calendar(calendar)
    context = Context(store, underlying, exchange, calendar, on_date, set())
    final = Context(store, underlying, exchange, calendar, None, set())
    contracts = sorted(_rule_dates(context, expiry, expiry).items())
    named = [item for item in contracts if _named_by(store, final, item[0], expiry)]
    if len(named) > 1:
        names = " and ".join(contract.isoformat() for contract, _ in named)
        raise ReferenceDataError(
            f"{expiry.isoformat()} was the listed expiry of two {underlying} contracts on "
            f"{exchange} (contract dates {names}); a symbol with it cannot be resolved"
        )
    if not named:
        return None
    contract, (series, rule) = named[0]
    row = context.contract_row(contract, rule)
    if row == NOT_LISTED:
        return None
    if row is None:
        row, record = (
            rule,
            _on(context, contract, series, rule, _previous_session(context, contract)),
        )
    else:
        record = _from_row(context, contract, series, row)
    _finish(context, [(record, row)])
    return record


def expiry_rule(store: RowStore, underlying: str, exchange: str, day: _dt.date, series: str) -> str:
    """Expiry weekday (MON to FRI, or NONE) of a series for expiry dates around `day`.

    Raises `ConfigError` for an unknown series and `ReferenceDataError` when no rule
    covers the date.
    """
    if series not in SERIES:
        raise ConfigError(f"series must be 'weekly' or 'monthly', got {series!r}")
    row = store.require(RULES_TABLE, (underlying, exchange, series), day)
    return store.note_use(row, day).values["weekday"]


def has_rules(store: RowStore, underlying: str, exchange: str) -> bool:
    """Whether the expiry rules know this underlying at all."""
    return any(store.rows(RULES_TABLE, (underlying, exchange, series)) for series in SERIES)


__all__ = [
    "ALL_SERIES",
    "Expiry",
    "alias_for",
    "contract_for",
    "expiry_rule",
    "expiry_schedule",
    "has_rules",
]
