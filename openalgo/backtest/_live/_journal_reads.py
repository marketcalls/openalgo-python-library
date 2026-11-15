"""What the live journal (`journal.Journal`) gives back, and the queries that read it.

Each query takes the open connection, the ledger (a `Mode` value) and, where the answer
is one trading day's, the day's key (`YYYY-MM-DD`). Rows are read with the conversions
of `_journal_rows`, which raise `JournalError` for a row this version cannot read.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date
from typing import Any

from ._journal_rows import day_key, intent_from, state_from, update_from
from ._types import Mode, OrderIntent, OrderUpdate

_UNRESOLVED = (
    "SELECT * FROM intents WHERE ledger = ? AND day = ? AND state IN ('recorded', 'unbound') "
    "ORDER BY seq"
)
_BOUND = "SELECT * FROM intents WHERE ledger = ? AND day = ? AND state = 'bound' ORDER BY seq"
# A terminal update is final: a later one is stale, so the terminal one comes first.
_LAST_UPDATE = (
    "SELECT * FROM updates WHERE ledger = ? AND order_id = ? "
    "ORDER BY terminal DESC, seq DESC LIMIT 1"
)
_UPDATES = "SELECT * FROM updates WHERE ledger = ? AND order_id = ? ORDER BY seq"
_LEG = "ledger = ? AND day = ? AND strategy = ? AND symbol = ? AND exchange = ?"
_SESSION = "SELECT * FROM sessions WHERE ledger = ? AND day = ?"


@dataclass(frozen=True)
class JournalOrder:
    """A bound order: its intent and the last update it had (None before the first)."""

    order_id: str
    intent: OrderIntent
    last_update: OrderUpdate | None

    @property
    def terminal(self) -> bool:
        """True once the order's last update is complete, rejected or cancelled."""
        return self.last_update is not None and self.last_update.status.terminal


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
    open_orders: tuple[JournalOrder, ...]
    bound_orders: tuple[JournalOrder, ...]
    stops: dict[tuple[str, str, str], str]
    rms_states: dict[str, dict[str, Any]]
    user_states: dict[str, SavedState]


def intent_row(sql: sqlite3.Connection, ledger: str, column: str, value: str) -> Any:
    """The `intents` row whose `column` ("intent_id" or "order_id") is `value`, or None."""
    if column not in ("intent_id", "order_id"):
        raise ValueError(f"intents are found by intent_id or order_id, not {column!r}")
    query = f"SELECT * FROM intents WHERE ledger = ? AND {column} = ?"
    return sql.execute(query, (ledger, value)).fetchone()


def unbound(sql: sqlite3.Connection, ledger: str, day: str) -> list[OrderIntent]:
    """The day's intents neither bound nor marked not placed, oldest first."""
    return [intent_from(row) for row in sql.execute(_UNRESOLVED, (ledger, day))]


def bound_orders(sql: sqlite3.Connection, ledger: str, day: str) -> list[JournalOrder]:
    """The day's bound orders with their last update, oldest first."""
    rows = sql.execute(_BOUND, (ledger, day)).fetchall()
    orders = []
    for row in rows:
        order_id = row["order_id"]
        orders.append(JournalOrder(order_id, intent_from(row), last_update(sql, ledger, order_id)))
    return orders


def last_update(sql: sqlite3.Connection, ledger: str, order_id: str) -> OrderUpdate | None:
    """An order's latest update to arrive, a terminal one ahead of any after it, or None."""
    row = sql.execute(_LAST_UPDATE, (ledger, order_id)).fetchone()
    return None if row is None else update_from(row)


def updates(sql: sqlite3.Connection, ledger: str, order_id: str) -> list[OrderUpdate]:
    """Every update of an order, in arrival order."""
    return [update_from(row) for row in sql.execute(_UPDATES, (ledger, order_id))]


def stop_of(sql: sqlite3.Connection, ledger: str, day: str, leg: tuple[str, str, str]) -> Any:
    """The order ID of the stop of `leg` (strategy, symbol, exchange) on the day, or None."""
    row = sql.execute(f"SELECT order_id FROM stops WHERE {_LEG}", (ledger, day, *leg)).fetchone()
    return None if row is None else row["order_id"]


def stops(sql: sqlite3.Connection, ledger: str, day: str) -> dict[tuple[str, str, str], str]:
    """The day's stops, by (strategy, symbol, exchange)."""
    rows = sql.execute("SELECT * FROM stops WHERE ledger = ? AND day = ?", (ledger, day))
    return {(row["strategy"], row["symbol"], row["exchange"]): row["order_id"] for row in rows}


def rms_states(sql: sqlite3.Connection, ledger: str, day: str) -> dict[str, dict[str, Any]]:
    """The day's risk state of each strategy that saved one."""
    query = "SELECT * FROM risk_states WHERE ledger = ? AND day = ? ORDER BY strategy"
    rows = sql.execute(query, (ledger, day))
    return {row["strategy"]: state_from(row["state"], "rms_state", row["strategy"]) for row in rows}


def rms_state(sql: sqlite3.Connection, ledger: str, strategy: str, day: str) -> Any:
    """A strategy's risk state of the day, or None."""
    query = "SELECT state FROM risk_states WHERE ledger = ? AND strategy = ? AND day = ?"
    row = sql.execute(query, (ledger, strategy, day)).fetchone()
    return None if row is None else state_from(row["state"], "rms_state", strategy)


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


def _session_from(row: sqlite3.Row, ledger: str, day: date) -> JournalSession:
    first, last = row["first_started_ns"], row["last_started_ns"]
    return JournalSession(day, Mode(ledger), row["broker"], first, last, row["starts"])


def _saved(row: sqlite3.Row) -> SavedState:
    return SavedState(state_from(row["state"], "self.state", row["strategy"]), row["saved_ns"])


__all__ = [
    "JournalOrder",
    "JournalSession",
    "JournalSnapshot",
    "SavedState",
    "bound_orders",
    "intent_row",
    "last_update",
    "rms_state",
    "rms_states",
    "session",
    "started_session",
    "stop_of",
    "stops",
    "unbound",
    "updates",
    "user_state",
    "user_states",
]
