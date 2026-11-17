"""The reconciler's rules for a restart's difference and for the reads its exits wait for,
on a stand-in session (`_live/reconcile.py`; 16-session-design 5.4 "As built" and 5.5,
unit 12).

A restart's difference of a leg from the broker's books (`Reconciler.differs_at_start`) is
acted on at once while the engine, once restored, holds the journal position the start
judged; when a fill reached the engine after the start read the books (the order worker
follows the day's orders during the warm-up), those books no longer describe the leg, so
nothing is adopted: entries are locked and the leg's exits wait for a fresh read. The
read the safe stop's flatten asks for (`Reconciler.read_before_exits`) is awaited only for
legs a read can still compare.

The stand-in strategy `S` bought 10 SBIN (broker order B1) before the restart; while the
session was down another client sold 6 (M1), so the start read 4 units at the broker
against the journal's 10, a difference the trade book explains. Expected values come from
the scenario.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from openalgo.backtest._live._reconcile_rules import Kind, LegBooks, Trade
from openalgo.backtest._live._types import AlertKind
from openalgo.backtest._live.reconcile import Reconciler

#: The journal position the start judged and the broker's units it read.
JOURNAL_UNITS, BROKER_UNITS = 10, 4
#: The restart's instant, UTC nanoseconds.
NOW_NS = 1_000
BOOKS = LegBooks(
    BROKER_UNITS,
    (Trade("B1", "BUY", 10, 500.0), Trade("M1", "SELL", 6, 498.0)),
    (),
)


class Trader:
    """One strategy's engine as the reconciler sees it at the restart; `fills` are the
    external fills it was given, `locks` its entry locks."""

    name, symbol, exchange = "S", "SBIN", "NSE"
    finished = False

    def __init__(self, units: int) -> None:
        self.units = units
        self.fills: list[tuple[Any, ...]] = []
        self.locks: list[str] = []
        self.engine = SimpleNamespace(apply_external_fill=lambda *args: self.fills.append(args))
        self.venue = SimpleNamespace(bound=dict)

    def open_units(self) -> float:
        return float(self.units)

    def lock(self, reason: str) -> bool:
        if reason in self.locks:
            return False
        self.locks.append(reason)
        return True


class Session:
    """What the reconciler calls on its session, before trading begins (no read is asked
    for yet); `alerts` what it sent."""

    trading, locked = False, False

    def __init__(self, trader: Trader) -> None:
        self.traders = (trader,)
        self.mismatches = 0
        self.alerts: list[tuple[AlertKind, str]] = []
        self.reconcile = Reconciler(self)
        # B1 is the session's own order, sent before the restart (the journal's).
        self.reconcile.start({}, frozenset({"B1"}))

    def now(self) -> int:
        return NOW_NS

    def event(self, kind: str, **fields: Any) -> None:
        del kind, fields

    def alert(self, kind: AlertKind, text: str) -> None:
        self.alerts.append((kind, text))


def started(engine_units: int) -> Session:
    """The start's difference handed to the reconciler, the engine holding `engine_units`
    once restored."""
    trader = Trader(engine_units)
    session = Session(trader)
    session.reconcile.differs_at_start(trader, Kind.EXPLAINED_FEWER, JOURNAL_UNITS, BOOKS)
    return session


@pytest.fixture
def current() -> Session:
    """The engine holds the journal's 10 units once restored."""
    return started(JOURNAL_UNITS)


@pytest.fixture
def stale() -> Session:
    """The leg's stop filled after the start read the books: the engine is flat once
    restored."""
    return started(0)


def test_a_start_difference_the_engine_still_holds_is_adopted_at_once(current):
    [fill] = current.traders[0].fills
    assert (fill[2], fill[3]) == ("SELL", 6.0)


def test_a_start_difference_on_units_the_engine_no_longer_holds_adopts_nothing(stale):
    assert stale.traders[0].fills == []


def test_a_start_difference_on_units_the_engine_no_longer_holds_locks_entries(stale):
    assert stale.traders[0].locks == ["reconciliation"]
    [(kind, text)] = stale.alerts
    assert kind is AlertKind.RECONCILE_MISMATCH
    assert "the engine holds 0 units once restored" in text


def test_a_start_difference_on_units_the_engine_no_longer_holds_waits_for_a_fresh_read(stale):
    # The leg's exits wait for a comparison of a read begun after the restore.
    assert stale.reconcile.guard.awaited_since() == [NOW_NS]
    assert not stale.reconcile.holds(stale.traders[0])


# ----- the read before the exits ------------------------------------------------------


@pytest.mark.parametrize(("finished", "awaited"), [(False, [NOW_NS]), (True, [])])
def test_the_flatten_waits_for_a_read_only_for_a_kernel_session_still_running(finished, awaited):
    # A finished kernel session is never compared again: waiting for its read would ask
    # for the books every few seconds until the session ends.
    trader = Trader(JOURNAL_UNITS)
    trader.finished = finished
    session = Session(trader)
    session.reconcile.read_before_exits()
    assert session.reconcile.guard.awaited_since() == awaited
