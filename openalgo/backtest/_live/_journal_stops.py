"""The rules for recording a leg's broker-held stop in the live journal (`journal.Journal`):
either the broker holds a leg's stop or the engine does, never both, so a leg has one
stop, and that stop is an order the session sent, as a stop, for that leg, on the
session's trading day.
"""

from __future__ import annotations

import sqlite3

from ..errors import JournalError
from . import _journal_reads as reads
from ._types import Purpose


def needs_recording(
    sql: sqlite3.Connection, ledger: str, day: str, leg: tuple[str, str, str], order_id: str
) -> bool:
    """Whether `order_id` is to be recorded as the stop of `leg` (strategy, symbol,
    exchange) on `day`: False when it already is. Raises `JournalError` when the order is
    not bound on `day` in the ledger, was sent for another leg or not as a stop, or the
    leg already has another stop."""
    owner = reads.intent_by_order(sql, ledger, day, order_id)
    if owner is None:
        raise JournalError(_not_bound(sql, ledger, day, order_id))
    strategy, symbol, exchange = leg
    sent_for = (owner["strategy"], owner["symbol"], owner["exchange"])
    if sent_for != leg:
        raise JournalError(
            f"stop order {order_id!r} was sent for strategy {sent_for[0]!r} on "
            f"{sent_for[2]}:{sent_for[1]}, not strategy {strategy!r} on {exchange}:{symbol}: "
            "record a stop on the leg its order protects"
        )
    if owner["purpose"] != Purpose.STOP.value:
        raise JournalError(
            f"stop order {order_id!r} was sent with purpose {owner['purpose']!r}, not as a "
            "stop: send the stop with Purpose.STOP, then record it"
        )
    held = reads.stop_of(sql, ledger, day, leg)
    if held == order_id:
        return False
    if held is not None:
        raise JournalError(
            f"strategy {strategy!r} on {exchange}:{symbol} already has stop order {held!r}: "
            "a leg has one exit owner, so cancel that stop, wait for the cancel to be "
            f"confirmed and clear_stop it before recording {order_id!r}"
        )
    return True


def _not_bound(sql: sqlite3.Connection, ledger: str, day: str, order_id: str) -> str:
    bound_on = reads.last_day_of_order(sql, ledger, order_id)
    if bound_on is not None:
        return (
            f"stop order {order_id!r} was bound on {bound_on} and the session is of {day}: "
            "start the session for the trading day it trades, then record its stop"
        )
    return (
        f"stop order {order_id!r} is not bound to an intent in the {ledger} ledger on {day}: "
        "record the stop's intent and bind it to the order, then record the stop"
    )


__all__ = ["needs_recording"]
