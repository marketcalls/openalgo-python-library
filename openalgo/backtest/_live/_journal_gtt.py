"""The GTTs in the live journal (`journal.Journal`; 20-positional-live 3, the lead's ruling
N2-2): each GTT the session places, bound to the broker's trigger ID, with its terms, every
state it took and when, and the child order it placed when it fired.

A GTT is written before its place call (`record_gtt`), as an intent is, so a restart that
meets an unanswered place finds what to look for in the GTT book; the place's answer binds
it to the broker's trigger ID (`bind_gtt`), and the session notes each later state
(`set_gtt_state`) and each modify the broker confirmed (`set_gtt_terms`). A GTT lives
across days: it is recorded on the session's day and read back on any later one, by its
number (`gtt`), its trigger ID (`gtt_by_trigger`) or while it has not ended (`open_gtts`).
A trigger ID is the broker's GTT's, so it is unique in a ledger. A fired GTT's child is
bound once (`bind_gtt_child`), on the day it fired; an order the session placed itself is
never one.

Every row is in the ledger of the handle's mode. Times are UTC nanoseconds; every failure
raises JournalError saying what to do.
"""

from __future__ import annotations

import sqlite3
from datetime import date

from ..errors import JournalError
from . import _journal_reads as reads
from ._journal_file import check_name
from ._journal_gtt_rows import (
    GTT_PURPOSES,
    GttBinding,
    GttChild,
    GttState,
    GttTerms,
    binding_from,
    check_leg,
    child_from,
    refusal_note,
)
from ._journal_handle import JournalHandle
from ._journal_rows import whole_number
from ._types import Purpose

_INSERT = (
    "INSERT INTO gtts (ledger, day, strategy, gtt_id, request_id, trigger_id, symbol, "
    "exchange, product, purpose, side, trigger_price, limit_price, quantity, state, "
    "recorded_ns, changed_ns) VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
_STATE = "INSERT INTO gtt_states VALUES (?, ?, ?, ?, ?)"
_BY_SEQ = "SELECT * FROM gtts WHERE ledger = ? AND seq = ?"
_BY_TRIGGER = "SELECT * FROM gtts WHERE ledger = ? AND trigger_id = ?"
_OPEN = "SELECT * FROM gtts WHERE ledger = ? AND state NOT IN ({ended}) ORDER BY seq"
_ENDED = tuple(state.value for state in GttState if state.terminal)
_CHILD = "SELECT * FROM gtt_children WHERE ledger = ? AND gtt_seq = ?"
_CHILD_OF_ORDER = "SELECT * FROM gtt_children WHERE ledger = ? AND day = ? AND order_id = ?"
_REFUSED_DAYS = (
    "SELECT DISTINCT g.day FROM gtts AS g JOIN gtt_states AS s "
    "ON s.ledger = g.ledger AND s.gtt_seq = g.seq "
    "WHERE g.ledger = ? AND g.strategy = ? AND g.symbol = ? AND g.exchange = ? "
    "AND g.product = ? AND g.purpose = ? AND g.day < ? AND s.state = ? "
    "AND substr(s.note, 1, ?) = ? "
    "ORDER BY g.day"
)


class JournalGtts(JournalHandle):
    """The journal's part for GTTs (module docstring), mixed into `journal.Journal`."""

    def record_gtt(
        self,
        strategy: str,
        gtt_id: int,
        request_id: int,
        symbol: str,
        exchange: str,
        product: str,
        terms: GttTerms,
        *,
        purpose: Purpose,
    ) -> int:
        """Write a GTT of `strategy` (the kernel's `gtt_id`, placed by the order worker's
        request `request_id`) for `purpose` before its place call, on the session's day, as
        REQUESTED; its number in the journal (`GttBinding.seq`). Raises JournalError for a
        blank name, a product other than CNC or NRML, a purpose other than a leg's stop or
        a strategy's entry, unusable terms (`GttTerms.checked`), ids that are not whole
        numbers, or with no `start_session`."""
        check_leg(strategy, symbol, exchange, product)
        if purpose not in GTT_PURPOSES:
            raise JournalError(f"a GTT is a leg's stop or a strategy's entry, not {purpose!r}")
        terms = terms.checked()
        ids = (whole_number("gtt_id", gtt_id), whole_number("request_id", request_id))
        key, now = self._day_key(None, "record a GTT"), self._now_ns()
        leg = (strategy, *ids, symbol, exchange, product, purpose.value)
        prices = (terms.side, terms.trigger_price, terms.limit_price, terms.quantity)
        with self._sql("record the GTT") as sql:
            row = (self._ledger, key, *leg, *prices, GttState.REQUESTED.value, now, now)
            seq = sql.execute(_INSERT, row).lastrowid
            sql.execute(_STATE, (self._ledger, seq, GttState.REQUESTED.value, "", now))
        return int(seq)

    def bind_gtt(self, seq: int, trigger_id: str) -> None:
        """Bind GTT `seq` to the broker's `trigger_id`, the place's answer: it is PLACED
        (binding the same pair again changes nothing). Raises JournalError for a GTT not in
        the ledger, one bound to another trigger ID or no longer REQUESTED, and a trigger ID
        bound to another GTT."""
        check_name("trigger_id", trigger_id)
        now = self._now_ns()
        with self._sql("bind the GTT") as sql:
            gtt = self._gtt(sql, seq)
            if gtt.trigger_id == trigger_id:
                return
            if gtt.trigger_id is not None or gtt.state is not GttState.REQUESTED:
                raise JournalError(
                    f"GTT {seq} is {gtt.state.value} with trigger {gtt.trigger_id!r}, so it "
                    f"cannot be bound to {trigger_id!r}: a place binds one GTT, once"
                )
            owner = sql.execute(_BY_TRIGGER, (self._ledger, trigger_id)).fetchone()
            if owner is not None:
                raise JournalError(
                    f"trigger {trigger_id!r} is bound to GTT {owner['seq']} already: a "
                    f"broker's GTT belongs to one GTT of the session, so {seq} needs its own"
                )
            sql.execute(
                "UPDATE gtts SET trigger_id = ?, state = ?, changed_ns = ? "
                "WHERE ledger = ? AND seq = ?",
                (trigger_id, GttState.PLACED.value, now, self._ledger, seq),
            )
            sql.execute(_STATE, (self._ledger, seq, GttState.PLACED.value, "", now))

    def set_gtt_state(self, seq: int, state: GttState, *, note: str = "") -> None:
        """Note that GTT `seq` is now in `state`, with why in `note` (a broker's message,
        "cancelled by hand"). Raises JournalError for a GTT not in the ledger, one that has
        ended, PLACED (a place's answer binds it: `bind_gtt`), and any state but NOT_PLACED
        for a GTT whose place has not answered (REQUESTED)."""
        if not isinstance(state, GttState) or not isinstance(note, str):
            raise JournalError(f"a GTT's state is a GttState with a note, not {state!r}")
        now = self._now_ns()
        with self._sql("note the GTT's state") as sql:
            gtt = self._gtt(sql, seq)
            if gtt.state.terminal:
                raise JournalError(
                    f"GTT {seq} is {gtt.state.value}: an ended GTT changes no more; a new "
                    "GTT protects the leg again (record_gtt)"
                )
            if state is GttState.PLACED:
                raise JournalError(
                    f"GTT {seq} is marked placed by its place's answer: bind it to the "
                    "broker's trigger ID with bind_gtt"
                )
            if gtt.state is GttState.REQUESTED and state is not GttState.NOT_PLACED:
                raise JournalError(
                    f"GTT {seq}'s place has not answered: bind it (bind_gtt) or mark it "
                    f"not placed before it is {state.value}"
                )
            sql.execute(
                "UPDATE gtts SET state = ?, changed_ns = ? WHERE ledger = ? AND seq = ?",
                (state.value, now, self._ledger, seq),
            )
            sql.execute(_STATE, (self._ledger, seq, state.value, note, now))

    def set_gtt_terms(self, seq: int, terms: GttTerms) -> None:
        """Note the terms the broker holds for GTT `seq` now, a modify it confirmed (its
        state is the caller's to set). Raises JournalError for a GTT not in the ledger,
        not bound to a trigger yet or ended, and unusable terms."""
        terms, now = terms.checked(), self._now_ns()
        with self._sql("note the GTT's terms") as sql:
            gtt = self._gtt(sql, seq)
            if gtt.trigger_id is None or gtt.state.terminal:
                raise JournalError(
                    f"GTT {seq} is {gtt.state.value}: only a GTT the broker holds is modified"
                )
            sql.execute(
                "UPDATE gtts SET side = ?, trigger_price = ?, limit_price = ?, quantity = ?, "
                "changed_ns = ? WHERE ledger = ? AND seq = ?",
                (
                    terms.side,
                    terms.trigger_price,
                    terms.limit_price,
                    terms.quantity,
                    now,
                    self._ledger,
                    seq,
                ),
            )

    def gtt(self, seq: int) -> GttBinding | None:
        """GTT `seq` of the ledger, of whatever day, or None."""
        with self._sql("read the GTT") as sql:
            row = sql.execute(_BY_SEQ, (self._ledger, seq)).fetchone()
        return None if row is None else binding_from(row)

    def gtt_by_trigger(self, trigger_id: str) -> GttBinding | None:
        """The GTT bound to the broker's `trigger_id` in the ledger, or None."""
        with self._sql("read the GTT of a trigger") as sql:
            row = sql.execute(_BY_TRIGGER, (self._ledger, trigger_id)).fetchone()
        return None if row is None else binding_from(row)

    def open_gtts(self) -> list[GttBinding]:
        """The GTTs of the ledger that have not ended, of every day, oldest first: what a
        start follows (20-positional-live 4.2)."""
        query = _OPEN.format(ended=", ".join("?" * len(_ENDED)))
        with self._sql("read the open GTTs") as sql:
            rows = sql.execute(query, (self._ledger, *_ENDED)).fetchall()
        return [binding_from(row) for row in rows]

    def refused_days(
        self, strategy: str, symbol: str, exchange: str, product: str, purpose: Purpose
    ) -> list[date]:
        """The trading days before the session's on which the broker refused to place a
        GTT of `strategy`'s leg made for `purpose`, oldest first: those whose placement was
        noted not placed with the broker's refusal (`_journal_gtt_rows.refusal_note`).
        Raises JournalError before `start_session`."""
        key = self._day_key(None, "read the GTT refusals of earlier days")
        prefix = refusal_note("")
        leg = (strategy, symbol, exchange, product, purpose.value)
        state = GttState.NOT_PLACED.value
        with self._sql("read the GTT refusals of earlier days") as sql:
            rows = sql.execute(
                _REFUSED_DAYS, (self._ledger, *leg, key, state, len(prefix), prefix)
            ).fetchall()
        return [date.fromisoformat(row["day"]) for row in rows]

    def gtt_states(self, seq: int) -> list[tuple[GttState, str, int]]:
        """Each state GTT `seq` took, with its note and when (UTC nanoseconds), oldest
        first; none for a GTT not in the ledger. Raises JournalError for a row it cannot
        read."""
        query = "SELECT * FROM gtt_states WHERE ledger = ? AND gtt_seq = ? ORDER BY rowid"
        with self._sql("read the GTT's states") as sql:
            rows = sql.execute(query, (self._ledger, seq)).fetchall()
        try:
            return [(GttState(row["state"]), row["note"], row["at_ns"]) for row in rows]
        except ValueError as error:
            raise JournalError(f"the journal holds a GTT state it cannot read ({error})") from error

    def bind_gtt_child(self, seq: int, order_id: str, kernel_order: int) -> None:
        """Bind the child order a fired GTT `seq` placed, broker order `order_id`, followed
        as kernel order `kernel_order`, on the session's day (binding the same again
        changes nothing). Raises JournalError for a GTT not in the ledger or never bound to
        a trigger, one whose child is another order, an order that is another GTT's child
        or that the session placed itself on the day, or with no `start_session`."""
        check_name("order_id", order_id)
        kernel = whole_number("kernel_order", kernel_order)
        key, now = self._day_key(None, "bind a GTT's child"), self._now_ns()
        with self._sql("bind the GTT's child") as sql:
            gtt = self._gtt(sql, seq)
            if gtt.trigger_id is None:
                raise JournalError(f"GTT {seq} never reached the broker, so it has no child")
            child = sql.execute(_CHILD, (self._ledger, seq)).fetchone()
            if child is not None and child["order_id"] == order_id:
                return
            self._check_child(sql, seq, key, order_id, child)
            sql.execute(
                "INSERT INTO gtt_children VALUES (?, ?, ?, ?, ?, ?)",
                (self._ledger, key, seq, order_id, kernel, now),
            )

    def gtt_child(self, seq: int) -> GttChild | None:
        """The child order GTT `seq` placed, or None while none is bound."""
        with self._sql("read the GTT's child") as sql:
            row = sql.execute(_CHILD, (self._ledger, seq)).fetchone()
        return None if row is None else child_from(row)

    def gtt_of_child(self, order_id: str, *, day: date | None = None) -> GttBinding | None:
        """The GTT whose child is broker order `order_id` on the day (the session's, or
        `day`), or None. Raises JournalError with no `day` before `start_session`."""
        key = self._day_key(day, "find the GTT of a child order")
        with self._sql("read the GTT of a child order") as sql:
            child = sql.execute(_CHILD_OF_ORDER, (self._ledger, key, order_id)).fetchone()
            row = None if child is None else self._gtt_row(sql, child["gtt_seq"])
        return None if row is None else binding_from(row)

    def _check_child(
        self,
        sql: sqlite3.Connection,
        seq: int,
        day: str,
        order_id: str,
        child: sqlite3.Row | None,
    ) -> None:
        if child is not None:
            raise JournalError(
                f"GTT {seq}'s child is order {child['order_id']!r}, not {order_id!r}: a SINGLE "
                "GTT places one child"
            )
        other = sql.execute(_CHILD_OF_ORDER, (self._ledger, day, order_id)).fetchone()
        if other is not None:
            raise JournalError(
                f"order {order_id!r} is the child of GTT {other['gtt_seq']} already, not of {seq}"
            )
        if reads.intent_by_order(sql, self._ledger, day, order_id) is not None:
            raise JournalError(
                f"order {order_id!r} was placed by the session itself, so it is no GTT's child: "
                "a child is an order no intent of the journal is bound to"
            )

    def _gtt_row(self, sql: sqlite3.Connection, seq: int) -> sqlite3.Row | None:
        return sql.execute(_BY_SEQ, (self._ledger, seq)).fetchone()

    def _gtt(self, sql: sqlite3.Connection, seq: int) -> GttBinding:
        row = self._gtt_row(sql, seq)
        if row is None:
            raise JournalError(
                f"GTT {seq} is not in the {self._ledger} ledger of the journal: record it "
                "with record_gtt before its place call"
            )
        return binding_from(row)


__all__ = ["JournalGtts"]
