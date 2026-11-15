"""What the live journal (`journal.Journal`) gives back, and the queries that read it.

Each query takes the open connection, the ledger (a `Mode` value) and, where the answer
is one trading day's, the day's key (`YYYY-MM-DD`): a broker may give an order ID again
on a later day, so an order is found by its ID within a day. Rows are read with the
conversions of `_journal_rows`, which raise `JournalError` for a row this version cannot
read.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date
from typing import Any

from ..errors import JournalError
from ._journal_rows import day_key, intent_from, state_from, unreadable, update_from
from ._types import BoundOrder, Mode, OrderIntent, OrderStatus, OrderUpdate

_UNRESOLVED = (
    "SELECT * FROM intents WHERE ledger = ? AND day = ? AND state IN ('recorded', 'unbound') "
    "ORDER BY seq"
)
_BOUND = "SELECT * FROM intents WHERE ledger = ? AND day = ? AND state = 'bound' ORDER BY seq"
_QUEUED = "SELECT * FROM intents WHERE ledger = ? AND day = ? AND state = 'queued' ORDER BY seq"
_OF_ORDER = "SELECT * FROM updates WHERE ledger = ? AND day = ? AND order_id = ?"
_TERMINAL_UPDATES = f"{_OF_ORDER} AND terminal ORDER BY seq"
_LATEST_UPDATE = f"{_OF_ORDER} ORDER BY seq DESC LIMIT 1"
_UPDATES = f"{_OF_ORDER} ORDER BY seq"
_LEG = "ledger = ? AND day = ? AND strategy = ? AND symbol = ? AND exchange = ?"
_SESSION = "SELECT * FROM sessions WHERE ledger = ? AND day = ?"


@dataclass(frozen=True)
class JournalSession:
    """A trading day's session in the journal's mode: the broker it ran on, when it first
    and last started (UTC nanoseconds) and how many times (more than one after a restart)."""

    day: date
    mode: Mode
    broker: str
    first_started_ns: int
    last_started_ns: int
    starts: int


@dataclass(frozen=True)
class SavedState:
    """A strategy's `self.state` as last saved, and when (UTC nanoseconds)."""

    state: dict[str, Any]
    saved_ns: int


@dataclass(frozen=True)
class JournalSnapshot:
    """What a restart restores, for one trading day `day` in the journal's mode.

    `unbound` holds the intents with no known outcome and `bound_orders` every order of
    the day with its last update, terminal ones included (a fill the broker's position
    book must show), oldest first; `open_orders` are those not yet terminal. `stops` maps
    (strategy, symbol, exchange) to the stop's order ID, `rms_states` the day's risk
    state by strategy, and `user_states` the last `self.state` of each strategy, of
    whatever day. A copy: changing it changes nothing in the journal.
    """

    day: date
    unbound: tuple[OrderIntent, ...]
    open_orders: tuple[BoundOrder, ...]
    bound_orders: tuple[BoundOrder, ...]
    stops: dict[tuple[str, str, str], str]
    rms_states: dict[str, dict[str, Any]]
    user_states: dict[str, SavedState]


def intent_by_id(sql: sqlite3.Connection, ledger: str, intent_id: str) -> sqlite3.Row | None:
    """The `intents` row of an intent ID (its intent, `day`, `state` and `order_id`), or
    None."""
    query = "SELECT * FROM intents WHERE ledger = ? AND intent_id = ?"
    return sql.execute(query, (ledger, intent_id)).fetchone()


def intent_by_order(
    sql: sqlite3.Connection, ledger: str, day: str, order_id: str
) -> sqlite3.Row | None:
    """The `intents` row bound to an order ID on the day, or None."""
    query = "SELECT * FROM intents WHERE ledger = ? AND day = ? AND order_id = ?"
    return sql.execute(query, (ledger, day, order_id)).fetchone()


def bound_day(
    sql: sqlite3.Connection, ledger: str, order_id: str, *, on_or_before: str | None = None
) -> str | None:
    """The latest trading day an order ID was bound on (no later than `on_or_before`
    when given), or None for an order ID the ledger never bound."""
    query = "SELECT day FROM intents WHERE ledger = ? AND order_id = ? AND day <= ?"
    latest = on_or_before if on_or_before is not None else date.max.isoformat()
    row = sql.execute(f"{query} ORDER BY seq DESC", (ledger, order_id, latest)).fetchone()
    return None if row is None else row["day"]


def unbound(sql: sqlite3.Connection, ledger: str, day: str) -> list[OrderIntent]:
    """The day's intents neither bound nor marked not placed, oldest first."""
    return [intent_from(row) for row in sql.execute(_UNRESOLVED, (ledger, day))]


def queued(sql: sqlite3.Connection, ledger: str, day: str) -> list[OrderIntent]:
    """The day's intents semi-auto mode queued and not bound since, oldest first."""
    return [intent_from(row) for row in sql.execute(_QUEUED, (ledger, day))]


def bound_orders(sql: sqlite3.Connection, ledger: str, day: str) -> list[BoundOrder]:
    """The day's bound orders with their last update, oldest first."""
    rows = sql.execute(_BOUND, (ledger, day)).fetchall()
    orders = []
    for row in rows:
        order_id = row["order_id"]
        last = last_update(sql, ledger, day, order_id)
        orders.append(BoundOrder(intent_from(row), order_id, last))
    return orders


def last_update(
    sql: sqlite3.Connection, ledger: str, day: str, order_id: str
) -> OrderUpdate | None:
    """The order's update of the day that counts, or None: the terminal one that stands
    (`_final`), or with none terminal the latest to arrive."""
    key = (ledger, day, order_id)
    terminal = [update_from(row) for row in sql.execute(_TERMINAL_UPDATES, key)]
    if terminal:
        return _final(terminal)
    row = sql.execute(_LATEST_UPDATE, key).fetchone()
    return None if row is None else update_from(row)


def _final(terminal: list[OrderUpdate]) -> OrderUpdate:
    """Of an order's terminal updates, in arrival order, the one that stands: the first,
    later ones being stale, except that a COMPLETE at average price 0 (a poll that saw
    the fill before its price) gives way to a later terminal update with a positive
    average price and as many units filled or more."""
    final = terminal[0]
    for later in terminal[1:]:
        if (
            final.status is OrderStatus.COMPLETE
            and final.average_price == 0
            and later.average_price > 0
            and later.filled_quantity >= final.filled_quantity
        ):
            final = later
    return final


def updates(sql: sqlite3.Connection, ledger: str, day: str, order_id: str) -> list[OrderUpdate]:
    """Every update of an order on the day, in arrival order."""
    return [update_from(row) for row in sql.execute(_UPDATES, (ledger, day, order_id))]


def stop_of(
    sql: sqlite3.Connection, ledger: str, day: str, leg: tuple[str, str, str]
) -> str | None:
    """The order ID of the stop of `leg` (strategy, symbol, exchange) on the day, or None."""
    row = sql.execute(f"SELECT order_id FROM stops WHERE {_LEG}", (ledger, day, *leg)).fetchone()
    return None if row is None else row["order_id"]


def stops(sql: sqlite3.Connection, ledger: str, day: str) -> dict[tuple[str, str, str], str]:
    """The day's stops, by (strategy, symbol, exchange)."""
    rows = sql.execute("SELECT * FROM stops WHERE ledger = ? AND day = ?", (ledger, day))
    return {(row["strategy"], row["symbol"], row["exchange"]): row["order_id"] for row in rows}


def rms_state(
    sql: sqlite3.Connection, ledger: str, strategy: str, day: str
) -> dict[str, Any] | None:
    """A strategy's risk state of the day, or None."""
    query = "SELECT state FROM risk_states WHERE ledger = ? AND strategy = ? AND day = ?"
    row = sql.execute(query, (ledger, strategy, day)).fetchone()
    return None if row is None else state_from(row["state"], "rms_state", strategy)


def rms_states(sql: sqlite3.Connection, ledger: str, day: str) -> dict[str, dict[str, Any]]:
    """The day's risk state of each strategy that saved one."""
    query = "SELECT * FROM risk_states WHERE ledger = ? AND day = ? ORDER BY strategy"
    rows = sql.execute(query, (ledger, day))
    return {row["strategy"]: state_from(row["state"], "rms_state", row["strategy"]) for row in rows}


def user_states(sql: sqlite3.Connection, ledger: str) -> dict[str, SavedState]:
    """The last `self.state` of each strategy that saved one."""
    rows = sql.execute("SELECT * FROM user_states WHERE ledger = ? ORDER BY strategy", (ledger,))
    return {row["strategy"]: _saved(row) for row in rows}


def user_state(sql: sqlite3.Connection, ledger: str, strategy: str) -> SavedState | None:
    """The last `self.state` a strategy saved, or None."""
    query = "SELECT * FROM user_states WHERE ledger = ? AND strategy = ?"
    row = sql.execute(query, (ledger, strategy)).fetchone()
    return None if row is None else _saved(row)


def session(sql: sqlite3.Connection, ledger: str, day: date) -> JournalSession | None:
    """The session of trading day `day` in the ledger, or None if none started. Raises
    `JournalError` when `day` is not a `datetime.date`."""
    row = sql.execute(_SESSION, (ledger, day_key(day))).fetchone()
    return None if row is None else _session_from(row, ledger, day)


def started_session(sql: sqlite3.Connection, ledger: str, day: date) -> JournalSession:
    """The session of trading day `day`, just written by the caller."""
    return _session_from(sql.execute(_SESSION, (ledger, day_key(day))).fetchone(), ledger, day)


def session_days(sql: sqlite3.Connection, ledger: str) -> list[date]:
    """The trading days with a started session in the ledger, oldest first."""
    rows = sql.execute("SELECT day FROM sessions WHERE ledger = ? ORDER BY day", (ledger,))
    try:
        return [date.fromisoformat(row["day"]) for row in rows]
    except ValueError as error:
        raise JournalError(unreadable("a session day", error)) from error


def _session_from(row: sqlite3.Row, ledger: str, day: date) -> JournalSession:
    first, last = row["first_started_ns"], row["last_started_ns"]
    return JournalSession(day, Mode(ledger), row["broker"], first, last, row["starts"])


def _saved(row: sqlite3.Row) -> SavedState:
    return SavedState(state_from(row["state"], "self.state", row["strategy"]), row["saved_ns"])


__all__ = [
    "JournalSession",
    "JournalSnapshot",
    "SavedState",
    "bound_day",
    "bound_orders",
    "intent_by_id",
    "intent_by_order",
    "last_update",
    "queued",
    "rms_state",
    "rms_states",
    "session",
    "session_days",
    "started_session",
    "stop_of",
    "stops",
    "unbound",
    "updates",
    "user_state",
    "user_states",
]
