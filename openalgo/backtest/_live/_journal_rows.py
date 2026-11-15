"""Rows of the live journal (`journal.Journal`): the shared values checked and turned into
table rows and back, the trading day of an instant, and the check that a strategy's state
survives being kept as JSON.

Every value is checked when it is written, so the file holds only what it can give back:
an int column gets a Python int (a NumPy integer is converted, a fraction refused), a
float column a finite float, an enum column the member's value.
"""

from __future__ import annotations

import enum
import json
import math
import numbers
import operator
import sqlite3
from collections.abc import Callable
from dataclasses import fields
from datetime import date, datetime
from typing import Any

from .._series import ist_time
from ..errors import JournalError
from ._types import Mode, OrderIntent, OrderStatus, OrderUpdate, Purpose

# The inserts, filled with the field names of the shared types below. An intent's state
# starts as "recorded" (see `_journal_file`).
_INTENT_TEMPLATE = (
    "INSERT INTO intents ({names}, ledger, day, state, reason, changed_ns) "
    "VALUES ({values}, :ledger, :day, 'recorded', '', :now)"
)
_UPDATE_TEMPLATE = (
    "INSERT INTO updates ({names}, ledger, terminal) VALUES ({values}, :ledger, :terminal)"
)


def whole_number(name: str, value: Any) -> int:
    """`value` as an int (a NumPy integer too). Raises `JournalError` naming `name` for
    None, a bool, a fraction or anything else that is not a whole number."""
    if isinstance(value, bool):
        raise JournalError(f"{name} must be a whole number, not {value!r}")
    try:
        return operator.index(value)
    except TypeError as error:
        raise JournalError(f"{name} must be a whole number, not {value!r}") from error


def finite_number(name: str, value: Any) -> float:
    """`value` as a float. Raises `JournalError` naming `name` for None, a bool, text, NaN,
    an infinity or anything else that is not a finite number."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise JournalError(f"{name} must be a finite float or int, not {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise JournalError(f"{name} must be a finite number, not {value!r}")
    return number


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise JournalError(f"{name} must be text, not {value!r}")
    return str(value)


def _member_of(kind: type[enum.Enum]) -> Callable[[str, Any], Any]:
    def member_value(name: str, value: Any) -> Any:
        if not isinstance(value, kind):
            raise JournalError(f"{name} must be a member of {kind.__name__}, not {value!r}")
        return value.value

    return member_value


# The check of each field, by the type its shared dataclass declares. A field of a type
# missing here fails loudly at import, until the journal learns to keep it.
_WRITERS: dict[str, Callable[[str, Any], Any]] = {
    "str": _text,
    "int": whole_number,
    "float": finite_number,
    "Purpose": _member_of(Purpose),
    "OrderStatus": _member_of(OrderStatus),
    "Mode": _member_of(Mode),
}
_READERS: dict[str, Callable[[Any], Any]] = {
    "Purpose": Purpose,
    "OrderStatus": OrderStatus,
    "Mode": Mode,
}


def _field_types(kind: type) -> tuple[tuple[str, str], ...]:
    # The shared types use postponed annotations, so a field's type is its name.
    found = tuple((item.name, getattr(item.type, "__name__", item.type)) for item in fields(kind))
    unknown = [f"{name}: {type_name}" for name, type_name in found if type_name not in _WRITERS]
    if unknown:
        raise TypeError(f"the journal cannot keep the {kind.__name__} fields {unknown}")
    return found


_FIELDS = {kind: _field_types(kind) for kind in (OrderIntent, OrderUpdate)}


def _insert(template: str, kind: type) -> str:
    names = [name for name, _ in _FIELDS[kind]]
    return template.format(names=", ".join(names), values=", ".join(f":{n}" for n in names))


#: Write a new intent: its fields, then `:ledger`, `:day` and `:now` (UTC nanoseconds).
INSERT_INTENT = _insert(_INTENT_TEMPLATE, OrderIntent)
#: Write an order update: its fields, then `:ledger` and `:terminal`.
INSERT_UPDATE = _insert(_UPDATE_TEMPLATE, OrderUpdate)


def _values(kind: type, record: Any) -> dict[str, Any]:
    return {
        name: _WRITERS[type_name](f"{kind.__name__}.{name}", getattr(record, name))
        for name, type_name in _FIELDS[kind]
    }


def intent_values(intent: OrderIntent) -> dict[str, Any]:
    """The named values of `INSERT_INTENT` from `intent`, with the trading day of its
    `created_ns` as `day`. Raises `JournalError` naming a field of the wrong type."""
    values = _values(OrderIntent, intent)
    return values | {"day": trading_day(values["created_ns"]).isoformat()}


def update_values(update: OrderUpdate) -> dict[str, Any]:
    """The named values of `INSERT_UPDATE` from `update`, with `terminal` from its status.
    Raises `JournalError` naming a field of the wrong type."""
    return _values(OrderUpdate, update) | {"terminal": int(update.status.terminal)}


def _from_row(kind: type, row: sqlite3.Row) -> Any:
    try:
        values = {
            name: _READERS.get(type_name, _same)(row[name]) for name, type_name in _FIELDS[kind]
        }
        return kind(**values)
    except ValueError as error:
        raise JournalError(unreadable(f"an {kind.__name__}", error)) from error


def _same(value: Any) -> Any:
    return value


def intent_from(row: sqlite3.Row) -> OrderIntent:
    """The intent an `intents` row holds. Raises `JournalError` for one it cannot read."""
    return _from_row(OrderIntent, row)


def update_from(row: sqlite3.Row) -> OrderUpdate:
    """The update an `updates` row holds. Raises `JournalError` for one it cannot read."""
    return _from_row(OrderUpdate, row)


def unreadable(what: str, error: Exception) -> str:
    """The message for a row this version cannot read, which only a change made to the
    file by another program can leave."""
    return (
        f"the journal holds {what} this version cannot read ({error}): the file was changed "
        "outside openalgo; move it away and start the session again"
    )


def trading_day(ts_ns: int) -> date:
    """The trading day of an instant in UTC nanoseconds: its IST calendar day. An MIS order
    is made in market hours, so its calendar day is its trading day."""
    return ist_time(ts_ns).date()


def day_key(day: date) -> str:
    """The key of a trading day, `YYYY-MM-DD`. Raises `JournalError` for anything but a
    `datetime.date` (a `datetime` too, whose time would be dropped)."""
    if isinstance(day, datetime) or not isinstance(day, date):
        raise JournalError(f"day must be a datetime.date (the IST trading day), not {day!r}")
    return day.isoformat()


def state_json(state: dict[str, Any], name: str, strategy: str) -> str:
    """A strategy's state (shown to the user as `name`) as JSON. Raises `JournalError`
    naming the first value that would come back from JSON changed, or not at all."""
    if not isinstance(state, dict):
        raise JournalError(
            f"{name} of strategy {strategy!r} must be a dict, not a {type(state).__name__}"
        )
    problem = _json_problem(state, name, ())
    if problem is not None:
        raise JournalError(
            f"{name} of strategy {strategy!r} cannot be saved: {problem}. Keep only str keys "
            "and JSON values in it: dict, list, str, int, finite float, bool and None"
        )
    return json.dumps(state, allow_nan=False)


def state_from(text: str, name: str, strategy: str) -> dict[str, Any]:
    """A saved state read back. Raises `JournalError` for text that is not its JSON."""
    try:
        return json.loads(text)
    except ValueError as error:
        raise JournalError(unreadable(f"{name} of strategy {strategy!r}", error)) from error


def _json_problem(value: Any, where: str, parents: tuple[int, ...]) -> str | None:
    """Why `value` (found at `where`) would not survive a JSON round trip, or None.
    `parents` are the ids of the containers holding it, to catch one that holds itself."""
    if value is None or isinstance(value, str | bool | int):
        return None
    if isinstance(value, float):
        return None if math.isfinite(value) else f"{where} is {value!r}, which JSON cannot hold"
    if not isinstance(value, dict | list):
        return (
            f"{where} has the type {type(value).__name__}, which does not survive a JSON round trip"
        )
    if id(value) in parents:
        return f"{where} holds itself"
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                return f"{where} has the key {key!r}, which JSON turns into a string"
        items = [(f"{where}[{key!r}]", item) for key, item in value.items()]
    else:
        items = [(f"{where}[{index}]", item) for index, item in enumerate(value)]
    for place, item in items:
        problem = _json_problem(item, place, (*parents, id(value)))
        if problem is not None:
            return problem
    return None


__all__ = [
    "INSERT_INTENT",
    "INSERT_UPDATE",
    "day_key",
    "finite_number",
    "intent_from",
    "intent_values",
    "state_from",
    "state_json",
    "trading_day",
    "unreadable",
    "update_from",
    "update_values",
    "whole_number",
]
