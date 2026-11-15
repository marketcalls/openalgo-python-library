"""The live session's journal (08-live-trading sections 6.1 and 7): one SQLite file per
OpenAlgo account, holding what a restart needs.

The order path writes an intent before its HTTP call, then binds it to the broker's order
ID, or marks it unbound when the outcome is unknown (a 502, a timeout, a disconnect) so
that it is resolved against the order book before anything else is sent. Every order
update is kept, in arrival order. The journal also holds each leg's broker-held stop
(D31: one exit owner per leg), the risk rules' state (extremes, trail level, daily
realized P&L) and the strategy's own `self.state`.

Every write is committed, and synced to disk, before its method returns: an intent is
durable before its order is sent. The file (`_journal_file`) uses write-ahead logging
and carries its schema version in `PRAGMA user_version`.

A journal is bound to one account and one broker: an exit sent to another broker would
open a new position instead of closing the old one, so opening it with another broker is
refused. It keeps one ledger per mode: a handle reads and writes the ledger of the mode
it was opened in, so a sandbox session never restores, reconciles or exits a live
position, nor the reverse. Intent IDs are unique across both ledgers.

Tier 1 trades MIS only, so what a restart restores is one IST trading day's: an intent
belongs to the day of its `created_ns`, a stop to its order's day, and the risk state
(the daily realized P&L with it) is kept per day. The day is that of the session started
on the handle (`start_session`), or a `day=` argument. Only `self.state` carries across
days. A handle is used from the thread that opened it, the session's loop.

Times are UTC nanoseconds. Every failure raises `JournalError` saying what to do.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

from ..errors import JournalError
from . import _journal_reads as reads
from ._journal_file import (
    JOURNAL_FILE_NAME,
    LIVE_DIR_VARIABLE,
    SCHEMA_VERSION,
    check_name,
    connect,
    default_journal_path,
)
from ._journal_handle import JournalHandle
from ._journal_reads import JournalOrder, JournalSession, JournalSnapshot, SavedState
from ._journal_rows import (
    INSERT_INTENT,
    INSERT_UPDATE,
    day_key,
    intent_from,
    intent_values,
    state_json,
    update_values,
)
from ._types import Mode, Now, OrderIntent, OrderUpdate

_START_SESSION = (
    "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, 1) ON CONFLICT (ledger, day) "
    "DO UPDATE SET last_started_ns = excluded.last_started_ns, starts = starts + 1"
)


class Journal(JournalHandle):
    """The journal of one account; see the module docstring. Implements `JournalLike`.
    Opened with `Journal.open`; `close()` or a `with` block closes it.

    Lookups of something the journal does not hold return None. Every other failure
    raises `JournalError`, with SQLite's own error chained, and says what to do.
    """

    @classmethod
    def open(
        cls, path: str | os.PathLike[str], *, account: str, broker: str, mode: Mode, now: Now
    ) -> Journal:
        """Create or open the journal at `path` (its folders are created) for `account`,
        bound to `broker`, reading and writing the ledger of `mode`; `now` gives the UTC
        nanoseconds of the times it records.

        Raises `JournalError` when the file cannot be opened, is not a journal, was
        written by a newer version or has an unknown schema version, belongs to another
        account or is bound to another broker.
        """
        if not isinstance(mode, Mode):
            raise JournalError(f"mode must be Mode.SANDBOX or Mode.LIVE, not {mode!r}")
        path = Path(path)
        return cls(connect(path, account, broker), path, mode, broker, now)

    def record_intent(self, intent: OrderIntent) -> None:
        """Write `intent` before its HTTP call, for the trading day of its `created_ns`; it
        is unbound until `bind` or a mark. Raises `JournalError` when its `intent_id` is
        already in the journal or a field has the wrong type (None, NaN, a fraction)."""
        values = intent_values(intent) | {"ledger": self._ledger, "now": self._now_ns()}
        with self._sql("record the intent") as sql:
            known = "SELECT 1 FROM intents WHERE intent_id = ?"
            if sql.execute(known, (values["intent_id"],)).fetchone():
                raise JournalError(
                    f"intent {intent.intent_id!r} is already in the journal: an intent is "
                    "recorded once, before its order is sent, so give each order a new intent_id"
                )
            sql.execute(INSERT_INTENT, values)

    def bind(self, intent_id: str, order_id: str) -> None:
        """Bind an intent to the broker's order ID (binding the same pair again changes
        nothing). Raises `JournalError` when the intent is unknown, is bound to another
        order, or the order is bound to another intent."""
        check_name("order_id", order_id)
        now = self._now_ns()
        with self._sql("bind the intent") as sql:
            bound = self._bound_order(sql, intent_id)
            if bound == order_id:
                return
            if bound is not None:
                raise JournalError(
                    f"intent {intent_id!r} is already bound to order {bound!r}, not "
                    f"{order_id!r}: one intent sends one order, so look for a duplicate in "
                    "the order book"
                )
            owner = reads.intent_row(sql, self._ledger, "order_id", order_id)
            if owner is not None:
                raise JournalError(
                    f"order {order_id!r} is already bound to intent {owner['intent_id']!r}: "
                    f"an order belongs to one intent, so {intent_id!r} needs its own order"
                )
            sql.execute(
                "UPDATE intents SET state = 'bound', order_id = ?, changed_ns = ? "
                "WHERE intent_id = ?",
                (order_id, now, intent_id),
            )

    def mark_unbound(self, intent_id: str, reason: str) -> None:
        """Note an intent whose outcome is unknown, and why; it stays in `unbound()` until
        resolved. Raises `JournalError` when the intent is unknown or already bound."""
        self._resolve(intent_id, "unbound", reason)

    def mark_not_placed(self, intent_id: str, reason: str) -> None:
        """Note an intent whose order never reached the broker (refused, or unbound and not
        in the order book), and why; it leaves `unbound()`. Raises `JournalError` when the
        intent is unknown or already bound."""
        self._resolve(intent_id, "not_placed", reason)

    def unbound(self, *, day: date | None = None) -> list[OrderIntent]:
        """The day's intents neither bound nor marked not placed, oldest first. Raises
        `JournalError` with no `day` before `start_session`."""
        key = self._day_key(day, "list the unbound intents")
        with self._sql("read the unbound intents") as sql:
            return reads.unbound(sql, self._ledger, key)

    def intent(self, intent_id: str) -> OrderIntent | None:
        """The recorded intent, or None."""
        with self._sql("read the intent") as sql:
            row = reads.intent_row(sql, self._ledger, "intent_id", intent_id)
            return None if row is None else intent_from(row)

    def order_id_of(self, intent_id: str) -> str | None:
        """The order ID an intent is bound to, or None while it is not bound (or unknown)."""
        with self._sql("read the order ID") as sql:
            row = reads.intent_row(sql, self._ledger, "intent_id", intent_id)
            return None if row is None else row["order_id"]

    def intent_of(self, order_id: str) -> OrderIntent | None:
        """The intent an order ID is bound to, or None for an order the engine did not send."""
        with self._sql("read the intent of the order") as sql:
            row = reads.intent_row(sql, self._ledger, "order_id", order_id)
            return None if row is None else intent_from(row)

    def record_update(self, update: OrderUpdate) -> None:
        """Keep an order update, after every earlier one; it may arrive before its order is
        bound, or be of an order the engine did not send. Raises `JournalError` for an
        update handled in another mode than the ledger's (a sandbox update must never
        close a live order with the same ID) or a field of the wrong type."""
        values = update_values(update) | {"ledger": self._ledger}
        if values["mode"] != self._ledger:
            raise JournalError(
                f"order {update.order_id!r} was updated in {values['mode']} mode, and this is "
                f"the {self._ledger} ledger: OpenAlgo's mode changed, so the session must "
                "stop and be started again in the mode wanted"
            )
        with self._sql("record the order update") as sql:
            sql.execute(INSERT_UPDATE, values)

    def updates(self, order_id: str) -> list[OrderUpdate]:
        """Every update of an order, in arrival order."""
        with self._sql("read the order updates") as sql:
            return reads.updates(sql, self._ledger, order_id)

    def last_update(self, order_id: str) -> OrderUpdate | None:
        """An order's latest update to arrive, or None. Once a terminal update (complete,
        rejected, cancelled) has arrived, a later one is stale and ignored."""
        with self._sql("read the last order update") as sql:
            return reads.last_update(sql, self._ledger, order_id)

    def bound_orders(self, *, day: date | None = None) -> list[JournalOrder]:
        """Every bound order of the day with its last update, terminal ones included,
        oldest first: what the reconciler matches against the broker's books. Raises
        `JournalError` with no `day` before `start_session`."""
        key = self._day_key(day, "list the orders")
        with self._sql("read the orders") as sql:
            return reads.bound_orders(sql, self._ledger, key)

    def open_orders(self, *, day: date | None = None) -> list[JournalOrder]:
        """The day's bound orders with no terminal update (none at all, or not yet
        terminal), oldest first. Raises `JournalError` with no `day` before `start_session`."""
        return [order for order in self.bound_orders(day=day) if not order.terminal]

    def set_stop(self, strategy: str, symbol: str, exchange: str, order_id: str) -> None:
        """Record the broker-held stop of a leg, for its order's trading day (setting the
        same one again changes nothing). Raises `JournalError` when the order is not bound
        to an intent in this ledger, or the leg has another stop: one exit owner per leg."""
        now = self._now_ns()
        with self._sql("record the stop") as sql:
            owner = reads.intent_row(sql, self._ledger, "order_id", order_id)
            if owner is None:
                raise JournalError(
                    f"stop order {order_id!r} is not bound to an intent in the {self._ledger} "
                    "ledger: record the stop's intent and bind it to the order, then record "
                    "the stop"
                )
            leg = (strategy, symbol, exchange)
            held = reads.stop_of(sql, self._ledger, owner["day"], leg)
            if held == order_id:
                return
            if held is not None:
                raise JournalError(
                    f"strategy {strategy!r} on {exchange}:{symbol} already has stop order "
                    f"{held!r}: a leg has one exit owner, so cancel that stop, wait for the "
                    f"cancel to be confirmed and clear_stop it before recording {order_id!r}"
                )
            row = (self._ledger, owner["day"], *leg, order_id, now)
            sql.execute("INSERT INTO stops VALUES (?, ?, ?, ?, ?, ?, ?)", row)

    def stop_of(
        self, strategy: str, symbol: str, exchange: str, *, day: date | None = None
    ) -> str | None:
        """The order ID of a leg's stop on the day, or None. Raises `JournalError` with no
        `day` before `start_session`."""
        key = self._day_key(day, "read the stop")
        with self._sql("read the stop") as sql:
            return reads.stop_of(sql, self._ledger, key, (strategy, symbol, exchange))

    def clear_stop(
        self, strategy: str, symbol: str, exchange: str, *, day: date | None = None
    ) -> None:
        """Forget a leg's stop of the day (nothing happens when it has none). Raises
        `JournalError` with no `day` before `start_session`."""
        key = self._day_key(day, "clear the stop")
        with self._sql("clear the stop") as sql:
            sql.execute(
                "DELETE FROM stops WHERE ledger = ? AND day = ? AND strategy = ? AND symbol = ? "
                "AND exchange = ?",
                (self._ledger, key, strategy, symbol, exchange),
            )

    def put_rms_state(
        self, strategy: str, state: dict[str, Any], *, day: date | None = None
    ) -> None:
        """Save the day's risk state of a strategy (extremes, trail level, daily realized
        P&L), replacing the last of that day. Raises `JournalError` naming a key whose
        value does not survive a JSON round trip, or with no `day` before `start_session`."""
        key = self._day_key(day, "save the risk state")
        text, now = state_json(state, "rms_state", strategy), self._now_ns()
        with self._sql("save the risk state") as sql:
            row = (self._ledger, strategy, key, text, now)
            sql.execute("INSERT OR REPLACE INTO risk_states VALUES (?, ?, ?, ?, ?)", row)

    def rms_state(self, strategy: str, *, day: date | None = None) -> dict[str, Any] | None:
        """A strategy's risk state of the day, or None: a new day starts with none. Raises
        `JournalError` with no `day` before `start_session`."""
        key = self._day_key(day, "read the risk state")
        with self._sql("read the risk state") as sql:
            return reads.rms_state(sql, self._ledger, strategy, key)

    def put_user_state(self, strategy: str, state: dict[str, Any]) -> None:
        """Save a strategy's `self.state`, replacing the last; it carries across days.
        Raises `JournalError` naming a key whose value does not survive a JSON round trip."""
        text, now = state_json(state, "self.state", strategy), self._now_ns()
        with self._sql("save self.state") as sql:
            row = (self._ledger, strategy, text, now)
            sql.execute("INSERT OR REPLACE INTO user_states VALUES (?, ?, ?, ?)", row)

    def user_state(self, strategy: str) -> SavedState | None:
        """The last `self.state` a strategy saved and when, or None."""
        with self._sql("read self.state") as sql:
            return reads.user_state(sql, self._ledger, strategy)

    def start_session(self, day: date, mode: Mode, broker: str) -> JournalSession:
        """Record that the session of trading day `day` started now, make `day` the
        handle's day, and return the session; a restart on the same day counts one more
        start and keeps the first. Raises `JournalError` when `mode` or `broker` is not
        the one the journal was opened with."""
        if mode is not self._mode or broker != self._broker:
            raise JournalError(
                f"this journal was opened for the {self._ledger} ledger at broker "
                f"{self._broker!r}, not {mode!r} at {broker!r}: open it with the mode and "
                "broker OpenAlgo is in now, then start the session"
            )
        key, now = day_key(day), self._now_ns()
        with self._sql("record the session start") as sql:
            sql.execute(_START_SESSION, (self._ledger, key, broker, now, now))
            started = reads.started_session(sql, self._ledger, day)
        self._session_day = day
        return started

    def session(self, day: date) -> JournalSession | None:
        """The session of trading day `day` in the journal's mode, or None if none started."""
        with self._sql("read the session") as sql:
            return reads.session(sql, self._ledger, day)

    def snapshot(self, *, day: date | None = None) -> JournalSnapshot:
        """What a restart restores for the day: see `JournalSnapshot`. Raises
        `JournalError` with no `day` before `start_session`."""
        key = self._day_key(day, "take a snapshot")
        with self._sql("read the snapshot") as sql:
            orders = reads.bound_orders(sql, self._ledger, key)
            return JournalSnapshot(
                day=date.fromisoformat(key),
                unbound=tuple(reads.unbound(sql, self._ledger, key)),
                open_orders=tuple(order for order in orders if not order.terminal),
                bound_orders=tuple(orders),
                stops=reads.stops(sql, self._ledger, key),
                rms_states=reads.rms_states(sql, self._ledger, key),
                user_states=reads.user_states(sql, self._ledger),
            )

    def _bound_order(self, sql: sqlite3.Connection, intent_id: str) -> str | None:
        row = reads.intent_row(sql, self._ledger, "intent_id", intent_id)
        if row is None:
            raise JournalError(
                f"intent {intent_id!r} is not in the {self._ledger} ledger of the journal: "
                "record it with record_intent before its order is sent"
            )
        return row["order_id"]

    def _resolve(self, intent_id: str, state: str, reason: str) -> None:
        now = self._now_ns()
        with self._sql(f"mark the intent {state}") as sql:
            bound = self._bound_order(sql, intent_id)
            if bound is not None:
                raise JournalError(
                    f"intent {intent_id!r} is bound to order {bound!r}, so it reached the "
                    "broker: resolve it from that order's updates"
                )
            sql.execute(
                "UPDATE intents SET state = ?, reason = ?, changed_ns = ? WHERE intent_id = ?",
                (state, reason, now, intent_id),
            )


__all__ = [
    "JOURNAL_FILE_NAME",
    "LIVE_DIR_VARIABLE",
    "SCHEMA_VERSION",
    "Journal",
    "JournalOrder",
    "JournalSession",
    "JournalSnapshot",
    "SavedState",
    "default_journal_path",
]
