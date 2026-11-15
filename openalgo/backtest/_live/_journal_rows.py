"""Rows of the live journal (`journal.Journal`): the shared values as table rows and back,
and the check that a strategy's state survives being kept as JSON."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import fields
from datetime import date, datetime
from typing import Any

from ..errors import JournalError
from ._types import Mode, OrderIntent, OrderStatus, OrderUpdate, Purpose

# The columns of an intent and an update are the fields of the shared types, so a field
# added to them fails loudly (no such column) until the journal's layout gains it.
_INTENT_FIELDS = tuple(item.name for item in fields(OrderIntent))
_UPDATE_FIELDS = tuple(item.name for item in fields(OrderUpdate))

#: Write a new intent: its fields, then `:ledger` and `:now` (UTC nanoseconds).
INSERT_INTENT = (
    f"INSERT INTO intents ({', '.join(_INTENT_FIELDS)}, ledger, state, reason, changed_ns) "
    f"VALUES ({', '.join(':' + name for name in _INTENT_FIELDS)}, :ledger, 'recorded', '', :now)"
)
#: Write an order update: its fields, then `:ledger`.
INSERT_UPDATE = (
    f"INSERT INTO updates ({', '.join(_UPDATE_FIELDS)}, ledger, terminal) "
    f"VALUES ({', '.join(':' + name for name in _UPDATE_FIELDS)}, :ledger, :terminal)"
)


def intent_values(intent: OrderIntent) -> dict[str, Any]:
    """The named values of `INSERT_INTENT` taken from `intent`."""
    values = {name: getattr(intent, name) for name in _INTENT_FIELDS}
    return values | {"purpose": intent.purpose.value}


def intent_from(row: sqlite3.Row) -> OrderIntent:
    """The intent an `intents` row holds."""
    values = {name: row[name] for name in _INTENT_FIELDS}
    return OrderIntent(**values | {"purpose": Purpose(row["purpose"])})


def update_values(update: OrderUpdate) -> dict[str, Any]:
    """The named values of `INSERT_UPDATE` taken from `update`."""
    values = {name: getattr(update, name) for name in _UPDATE_FIELDS}
    status = {"status": update.status.value, "terminal": update.status.terminal}
    return values | status | {"mode": update.mode.value}


def update_from(row: sqlite3.Row) -> OrderUpdate:
    """The order update an `updates` row holds."""
    values = {name: row[name] for name in _UPDATE_FIELDS}
    return OrderUpdate(**values | {"status": OrderStatus(row["status"]), "mode": Mode(row["mode"])})


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
    "intent_from",
    "intent_values",
    "state_json",
    "update_from",
    "update_values",
]
