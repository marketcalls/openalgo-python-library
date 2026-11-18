"""The invariants of a live session on the fake OpenAlgo (16-session-design 7.1, unit 14),
checked after every session test: `rig` (session_samples) records each fake's order changes
and session runs (`session_trace`) and has the fixture run `assert_session_invariants`
when the test ends. A test that cannot keep one check waives that check alone, with its
reason (`SessionTrace.waive`); every other check still runs.

At every change of an order at the fake, as it happened:

1. **One stop.** At most one stop of the session's (SL or SL-M) works per leg (book,
   strategy, instrument).
2. **Exits within the position.** After each change of one of the session's orders, its
   working orders on the closing side of the instrument (the side opposite its first
   order there) hold no more units than the position on the open side. Units another
   client sold (a sale by hand) while those exits worked are the operator's excess, not
   the session's: they are allowed beside the position until the session's exits fit it
   again (16-session-design 5.3 fits them once its comparisons confirm the sale).
3. **Never the other side.** No fill of the session's orders leaves the instrument's
   position on the other side of the leg, unless another client's fills had taken it
   there already and the session's fill moved it towards flat.

When the last run ends (its close returned, or the test ended after a crash), of that run
(the restarted one after a crash):

4. **No orphan.** Every order of the session's working at the fake is a working kernel
   order of that run (bound to its broker ID), and no stop of it works for a position the
   fake holds flat (but one another client's sales left there, which the close names).
5. **Protected.** Each position the fake holds on the side of a leg of the run that wants
   a stop (its stop level is set) has the session's working stops covering the engine's
   units, confirmed working by the session's stops supervisor (`StopsSupervisor.covered`);
   unless the test leaves the leg unprotected (`SessionTrace.leave_unprotected`: the rows
   F35, F36, F60, the F77 variant and F80; `left_unprotected_by_decision`: a decided end
   of 16-session-design, listed in FAULT_ROWS.md), which then must be named: a row of
   `open_positions` for the leg with no confirmed stop, and an ERROR alert of the run that
   names the strategy. 2 is asserted once more here, over what works at the end.
6. **Books agree.** The engine's units and the fake's are equal per instrument, or the
   run's `reconciliations` or the close's named legs hold the difference (a run that
   crashed has no close: not checked).

Over each run's event stream and the fake's log:

7. **Every Modify and Cancel answered.** A Modify got exactly one of `apply_modify`,
   `apply_modify_refused` or its order's terminal news (none in a run that crashed before
   its close), a modify the exit guard still holds at the close included; a Cancel its
   order's terminal news, `apply_cancel_refused`, or the run's end.
8. **Nothing after a guard lock.** No order call and no book read by the run's clients
   in the fake's log after the entry it held when the run's mode guard locked (by the
   log's order, so a call at the very instant of the lock is one).
9. **Priced fills.** Every fill the kernel took, and every fill of other orders it adopted,
   has a positive price; a fill not flagged estimated is at a price the broker's fills of
   that order hold.

Quantities are signed units, prices rupees, times UTC nanoseconds.
"""

from __future__ import annotations

from typing import Any

import pytest
from fake_openalgo import FakeOpenAlgo
from fake_openalgo.log import CALL
from session_trace import (
    ANSWERED,
    BOOKS_AGREE,
    GUARD_LOCK,
    NO_ORPHAN,
    ONE_STOP,
    OTHER_SIDE,
    PRICED,
    PROTECTED,
    WITHIN,
    Closing,
    OrderChange,
    Run,
    SessionTrace,
    closing_of,
)

from openalgo.backtest._live._types import AlertKind, Mode
from openalgo.backtest._live._venue_requests import ModifyRequest, request_fields

#: The price types of a stop at the broker.
STOP_TYPES = frozenset({"SL", "SL-M"})
#: The order stream's terminal words.
NEWS_ENDED = frozenset({"complete", "cancelled", "rejected"})
#: The calls that send an order or read a book (8).
ORDER_CALLS = frozenset({"placeorder", "modifyorder", "cancelorder"})
BOOK_READS = frozenset({"orderstatus", "orderbook", "tradebook", "positionbook", "funds"})
#: A price the broker's fills hold, within half a paisa.
PRICE_TOLERANCE = 0.005

#: A known defect that breaks invariant 3 (unit 14 round 1, finding 1), until the lead
#: decides: fit the stop at the first comparison that shows an explained sale by hand, or
#: list the exemption in 7.1 and 5.3.
OVERSOLD_AFTER_A_SALE_BY_HAND = (
    "7.1 invariant 3 broken: after another client sells part of the position, the "
    "session's resting stop for the whole position fills and takes the account through "
    "zero (5.3 row 2 cuts the stop only at the second comparison that shows the sale)"
)

Instrument = tuple[Mode, str, str]


class InvariantBroken(AssertionError):
    """An invariant of 16-session-design 7.1 does not hold; the message names it."""


def known_break(defect: str) -> Any:
    """The mark of a test whose session breaks an invariant through `defect`, a known
    defect: it fails with InvariantBroken (raised by `then_the_invariants` at its end)
    until the defect is fixed, and passing then fails it (strict), so the mark goes with
    the fix. A failure of the test's own assertions is no InvariantBroken: it fails."""
    return pytest.mark.xfail(raises=InvariantBroken, strict=True, reason=defect)


def then_the_invariants(trace: SessionTrace) -> None:
    """Check the session invariants now, inside the test rather than when it ends
    (`SessionTrace.check_now`), so that a `known_break` mark sees them."""
    trace.check_now(assert_session_invariants)


def assert_session_invariants(session: SessionTrace, fake: FakeOpenAlgo) -> None:
    """Check every invariant of the module docstring over `session`, the trace of the
    sessions run on `fake`, but those the test waived. Raises InvariantBroken naming the
    first one broken."""
    if not session.runs:
        return
    try:
        _check(session, fake)
    except InvariantBroken:
        raise
    except AssertionError as error:
        raise InvariantBroken(str(error)) from None


def _check(session: SessionTrace, fake: FakeOpenAlgo) -> None:
    waived = session.waived
    mine = session.session_clients()
    last = session.runs[-1]
    # The changes up to the last run's end: what a test does after it (an operator's
    # orders once the session has ended) is not the session's.
    changes = session.changes[: last.ended_at]
    own = [c for c in changes if c.client in mine]
    sides = _sides(own)
    if ONE_STOP not in waived:
        _one_stop_per_leg(changes, mine)
    excess = _within_the_position(
        changes, mine, sides, within=WITHIN not in waived, side=OTHER_SIDE not in waived
    )
    closing = last.closing or _closing_now(last)
    working = _working_at_the_end(own)
    held = {c.instrument: c.position for c in changes}
    if NO_ORPHAN not in waived:
        _no_orphan(last, closing, working, held, excess)
    _at_the_end(session, last, closing, working, held, sides, excess)
    if BOOKS_AGREE not in waived and last.outcome is not None:
        _books_agree(last, closing, held)
    for at, run in enumerate(session.runs):
        if ANSWERED not in waived:
            _modifies_and_cancels_answered(run)
        if GUARD_LOCK not in waived:
            end = run.log_to
            if end is None and at + 1 < len(session.runs):
                end = session.runs[at + 1].log_from
            _nothing_after_a_guard_lock(run, fake, end)
        if PRICED not in waived:
            _priced_fills(run, fake)


# ----- at every change ----------------------------------------------------------------------


def _sides(own: list[OrderChange]) -> dict[Instrument, str]:
    """The open side of each instrument the session traded: its first order's action."""
    sides: dict[Instrument, str] = {}
    for change in own:
        sides.setdefault(change.instrument, change.action)
    return sides


def _one_stop_per_leg(changes: list[OrderChange], mine: frozenset[str]) -> None:
    orders: dict[str, OrderChange] = {}
    for change in changes:
        orders[change.order_id] = change
        if change.client not in mine or change.price_type not in STOP_TYPES:
            continue
        leg = (change.mode, change.strategy, change.symbol, change.exchange)
        stops = [
            o.order_id
            for o in orders.values()
            if o.client in mine and o.working and o.price_type in STOP_TYPES
            and (o.mode, o.strategy, o.symbol, o.exchange) == leg
        ]  # fmt: skip
        assert len(stops) <= 1, f"7.1: two stops of {leg} work at once: {stops} ({change})"


def _within_the_position(
    changes: list[OrderChange],
    mine: frozenset[str],
    sides: dict[Instrument, str],
    *,
    within: bool = True,
    side: bool = True,
) -> dict[Instrument, int]:
    """2 and 3 (module docstring), `within` and `side` saying which is asserted; the units
    of each instrument other clients' sales left beside the session's exits at the end."""
    orders: dict[str, OrderChange] = {}
    excess: dict[Instrument, int] = {}
    for change in changes:
        orders[change.order_id] = change
        instrument, opens = change.instrument, sides.get(change.instrument)
        if opens is None:
            continue
        held = change.position if opens == "BUY" else -change.position
        if change.client not in mine:
            if change.fill_units and change.action != opens:
                excess[instrument] = excess.get(instrument, 0) + change.fill_units
            continue
        closing = sum(
            o.remaining
            for o in orders.values()
            if o.client in mine and o.working and o.instrument == instrument and o.action != opens
        )  # fmt: skip
        if closing <= max(held, 0):
            excess.pop(instrument, None)
        assert not within or closing <= max(held, 0) + excess.get(instrument, 0), (
            f"7.1: the session's working exits of {instrument} hold {closing} units "
            f"against a position of {change.position} ({change})"
        )
        if side and change.fill_units and held < 0:
            moved = change.fill_units if change.action == opens else -change.fill_units
            before = held - moved
            assert before < 0 and moved > 0, (
                f"7.1: a fill of the session's order {change.order_id} left {instrument} at "
                f"{change.position}, the other side of the leg ({change})"
            )
    return excess


# ----- when the test ends -------------------------------------------------------------------


def _closing_now(run: Run) -> dict[str, Closing]:
    """A run with no close (a crash): each strategy's kernel state as it stands now."""
    found = {}
    for trader in getattr(run.session, "traders", ()):
        closing = closing_of(run.session, trader)
        if closing is not None:
            found[trader.name] = closing
    return found


def _working_at_the_end(own: list[OrderChange]) -> list[OrderChange]:
    last: dict[str, OrderChange] = {}
    for change in own:
        last[change.order_id] = change
    return [c for c in last.values() if c.working]


def _no_orphan(
    run: Run,
    closing: dict[str, Closing],
    working: list[OrderChange],
    held: dict[Instrument, int],
    excess: dict[Instrument, int],
) -> None:
    """4; a stop other clients' sales left on a flat position is the operator's to cancel,
    as the close's alert says (`_reconcile_close.NOTHING_TO_CLOSE`)."""
    kernel = {order_id for c in closing.values() for order_id in c.working_order_ids}
    known = _bound_order_ids(run)
    # A working kernel order with no broker ID yet (its placement unanswered) stands for
    # one order at the fake the session cannot name.
    unbound = {name: c.unbound_working for name, c in closing.items()}
    for order in working:
        if order.mode is not run.mode or order.order_id in kernel:
            continue
        ours = closing.get(order.strategy) is not None or order.order_id in known
        if ours and order.order_id not in known and unbound.get(order.strategy, 0) > 0:
            unbound[order.strategy] -= 1
        elif ours:
            raise AssertionError(
                f"7.1: order {order.order_id} works at the fake but is no working kernel "
                f"order of the session: an orphan ({order})"
            )
    for order in working:
        flat = held.get(order.instrument, 0) == 0
        if order.mode is run.mode and order.price_type in STOP_TYPES and flat:
            assert excess.get(order.instrument, 0) > 0, (
                f"7.1: stop {order.order_id} works for a flat position ({order})"
            )


def _at_the_end(
    trace: SessionTrace,
    run: Run,
    closing: dict[str, Closing],
    working: list[OrderChange],
    positions: dict[Instrument, int],
    sides: dict[Instrument, str],
    excess: dict[Instrument, int],
) -> None:
    """2 once more and 5 (module docstring); a leg the test leaves unprotected must need
    it (the fake holds units on its side at the end)."""
    named: set[str] = set()
    for leg in closing.values():
        instrument = (run.mode, leg.symbol, leg.exchange)
        opens = sides.get(instrument)
        position = positions.get(instrument, 0)
        # The units the fake holds on the leg's side (none when the account is flat or on
        # the other side: no stop of the leg's can hold those).
        held = max(position if opens != "SELL" else -position, 0)
        exits = sum(
            o.remaining for o in working if o.instrument == instrument and o.action != opens
        )
        assert WITHIN in trace.waived or exits <= held + excess.get(instrument, 0), (
            f"7.1: {leg.strategy} ends with exits of {exits} units working at the fake "
            f"against a position of {position} {leg.symbol}"
        )
        if PROTECTED in trace.waived or not held:
            continue
        if leg.strategy in trace.unprotected:
            _named_unprotected(run, leg, trace.unprotected[leg.strategy])
            named.add(leg.strategy)
            continue
        if not leg.wants_stop or not leg.units:
            continue
        stops = sum(
            o.remaining for o in working if o.instrument == instrument and o.action != opens
            and o.price_type in STOP_TYPES
        )  # fmt: skip
        needed = min(held, abs(round(leg.units)))
        assert min(stops, leg.stop_covers) >= needed, (
            f"7.1: {leg.strategy} ends with {position} units of {leg.symbol} at the fake, "
            f"stops of {stops} working there and {leg.stop_covers} confirmed by the session, "
            f"for the engine's {leg.units:g}"
        )
    unneeded = sorted(set(trace.unprotected) - named)
    assert PROTECTED in trace.waived or not unneeded, (
        f"7.1: the test leaves {unneeded} unprotected, but the fake holds no units on "
        "their side at the end"
    )


def _named_unprotected(run: Run, leg: Closing, why: str) -> None:
    """A leg the test leaves unprotected (`why`: its row or decided end): `open_positions`
    lists it with no confirmed stop, and an ERROR alert of the run names its strategy."""
    assert run.outcome is not None, (
        f"7.1: {leg.strategy} is left unprotected ({why}), but the run has no outcome to list it in"
    )
    rows = [
        p
        for p in run.outcome.open_positions
        if (p.strategy, p.symbol, p.exchange) == (leg.strategy, leg.symbol, leg.exchange)
    ]
    assert rows and not any(p.stop_confirmed for p in rows), (
        f"7.1: {leg.strategy} is left unprotected ({why}), but open_positions lists {rows}"
    )
    named = [text for kind, text in run.alerts if kind is AlertKind.ERROR and leg.strategy in text]
    assert named, f"7.1: {leg.strategy} is left unprotected ({why}) and no ERROR alert names it"


def _books_agree(run: Run, closing: dict[str, Closing], positions: dict[Instrument, int]) -> None:
    """6 (module docstring)."""
    engine: dict[tuple[str, str], float] = {}
    for leg in closing.values():
        key = (leg.symbol, leg.exchange)
        engine[key] = engine.get(key, 0.0) + leg.units
    outcome = run.outcome
    named = " ".join((*outcome.summary.differing, *outcome.summary.unconfirmed))
    for (symbol, exchange), units in engine.items():
        held = positions.get((run.mode, symbol, exchange), 0)
        if round(units) == held:
            continue
        strategies = {leg.strategy for leg in closing.values() if leg.symbol == symbol}
        rows = [r for r in outcome.reconciliations if r.strategy in strategies]
        said = any(f"{name} {symbol}" in named for name in strategies)
        assert rows or said, (
            f"7.1: the engine ends with {units:g} {symbol} and the fake with {held}, and no "
            "reconciliation holds the difference"
        )


def _bound_order_ids(run: Run) -> set[str]:
    found = set()
    for trader in getattr(run.session, "traders", ()):
        found |= {placed.order_id for _, placed in trader.venue.placed() if placed.order_id}
    return found


# ----- over each run's event stream and the fake's log --------------------------------------


def _modifies_and_cancels_answered(run: Run) -> None:
    """7 (module docstring)."""
    events = run.events
    closed = any(e["kind"] == "closed" for e in events)
    ended = closed or run.outcome is None
    for at, request in enumerate(events):
        kind = request.get("type") if request["kind"] == "request" else None
        if kind not in ("ModifyRequest", "CancelRequest"):
            continue
        later = events[at + 1 :]
        answers = {
            e["applied"]
            for e in later
            if e["kind"] in ("sent", "not_sent", "settled")
            and e.get("request_id") == request["request_id"]
        }
        news = any(
            e["kind"] == "news" and e.get("kernel_order") == request["kernel_order"]
            and e.get("status") in NEWS_ENDED and e.get("strategy") == request["strategy"]
            for e in later
        )  # fmt: skip
        if kind == "ModifyRequest":
            applied = answers & {"modified", "modify_refused"}
            assert len(applied) <= 1, f"7.1: modify {request} answered twice: {applied}"
            assert applied or news or not closed, (
                f"7.1: modify {request} got no apply_modify, no refusal and no terminal news"
            )
        else:
            assert news or "cancel_refused" in answers or ended, (
                f"7.1: cancel {request} got no terminal news and no refusal"
            )
    if not closed:
        return
    # A modify the exit guard held and never handed over has no request line: it is
    # answered only by its order's terminal news.
    for held in run.session.reconcile.guard.left():
        if not isinstance(held, ModifyRequest):
            continue
        fields = request_fields(held)
        news = any(
            e["kind"] == "news" and e.get("kernel_order") == fields["kernel_order"]
            and e.get("status") in NEWS_ENDED and e.get("strategy") == fields["strategy"]
            for e in events
        )  # fmt: skip
        assert news, f"7.1: modify {fields}, held by the exit guard, got no answer"


def _nothing_after_a_guard_lock(run: Run, fake: FakeOpenAlgo, end: int | None) -> None:
    """8: the calls the fake logged after the entry it held at the lock, to the run's end."""
    if run.locked_at is None:
        return
    late = [
        (entry.what, entry.detail.get("client"), entry.ts_ns)
        for entry in fake.log.entries[run.locked_at : end]
        if entry.kind == CALL and entry.detail.get("client") in run.clients
        and entry.what in ORDER_CALLS | BOOK_READS
    ]  # fmt: skip
    assert late == [], f"7.1: calls after the guard lock (log entry {run.locked_at}): {late}"


def _priced_fills(run: Run, fake: FakeOpenAlgo) -> None:
    """9 (module docstring)."""
    for adopted in (e for e in run.events if e["kind"] == "adopted"):
        assert adopted["price"] > 0, f"7.1: the kernel adopted a fill at no price: {adopted}"
    fills = [e for e in run.events if e["kind"] == "fill"]
    broker: dict[str, list[float]] = {}
    for fill in fake.fills(run.mode):
        broker.setdefault(fill.order_id, []).append(fill.price)
    for fill in fills:
        assert fill["price"] > 0, f"7.1: the kernel took a fill at no price: {fill}"
        prices = broker.get(fill["order_id"])
        if fill.get("estimated") or not prices:
            continue
        low, high = min(prices) - PRICE_TOLERANCE, max(prices) + PRICE_TOLERANCE
        assert low <= fill["price"] <= high, (
            f"7.1: fill {fill} is not flagged estimated, but the broker filled the order at "
            f"{prices}"
        )


__all__ = [
    "OVERSOLD_AFTER_A_SALE_BY_HAND",
    "InvariantBroken",
    "assert_session_invariants",
    "known_break",
    "then_the_invariants",
]
