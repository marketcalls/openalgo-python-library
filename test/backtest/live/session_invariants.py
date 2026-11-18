"""The invariants of a live session on the fake OpenAlgo (16-session-design 7.1, unit 14),
checked after every session test: `rig` (session_samples) records each fake's order changes
and session runs (`session_trace`) and has the fixture run `assert_session_invariants`
when the test ends. A test opts out only with its reason (`SessionTrace.opt_out`).

At every change of an order at the fake, as it happened:

1. **One stop.** At most one stop of the session's (SL or SL-M) works per leg (book,
   strategy, instrument).
2. **Exits within the position.** After each change of one of the session's orders, its
   working orders on the closing side of the instrument (the side opposite its first
   order there) hold no more units than the position on the open side.
3. **Never the other side.** No fill of the session's orders leaves the instrument's
   position on the other side of the leg.

   A sale by hand (another client's fill on the closing side) while the session's exits
   work is the operator's, made against the session's own instruction to cancel the stop
   first (`_stops_report`): it can leave a stop larger than the position, which the
   session fits only once its comparisons have confirmed the sale (16-session-design 5.3).
   From such a fill until the session's exits fit the position again, 2 and 3 are not
   asserted for that instrument; one still unfitted at the end must not be claimed
   protected (no confirmed stop in `open_positions`).

When the last run ends (its close returned, or the test ended after a crash), of that run
(the restarted one after a crash):

4. **No orphan.** Every order of the session's working at the fake is a working kernel
   order of that run (bound to its broker ID), and no stop of it works for a position the
   fake holds flat (but one a sale by hand left there, which the close names).
5. **Protected.** Each position the fake holds that a leg of the run wants a stop for
   (its stop level is set) has the session's working stops covering the engine's units,
   confirmed in `open_positions`; unless the test asserts the leg unprotected in
   `open_positions` and the ERROR alert that names it (`SessionTrace.leave_unprotected`:
   the rows F35, F36, F60, the F77 variant and F80, and the merged tests of decided ends
   that leave a leg without a confirmed stop, each saying which in a comment). Such a leg
   is only checked not to be claimed protected, and 2 at the end, 6 and the answer of
   its stop's last modify (7) are not asserted for it.
6. **Books agree.** The engine's units and the fake's are equal per instrument, or the
   run's `reconciliations` or the close's named legs hold the difference (a run that
   crashed has no close, and one stopped by an error or a mode breach reads no books at
   it: neither is checked).

Over each run's event stream:

7. **Every Modify and Cancel answered.** A Modify got exactly one of `apply_modify`,
   `apply_modify_refused` or its order's terminal news (none at a crash that cut it
   short, nor the cut of a stop left unfitted or unprotected at the close); a Cancel its
   order's terminal news, `apply_cancel_refused`, or the run's end.
8. **Nothing after a guard lock.** No order call and no book read by the session's
   clients after the time the guard saw the breach.
9. **Priced fills.** Every fill the kernel took has a positive price; one not flagged
   estimated is at a price the broker's fills of that order hold.

Quantities are signed units, prices rupees, times UTC nanoseconds.
"""

from __future__ import annotations

from fake_openalgo import FakeOpenAlgo
from session_trace import Closing, OrderChange, Run, SessionTrace, closing_of

from openalgo.backtest._live._types import Mode

#: The price types of a stop at the broker.
STOP_TYPES = frozenset({"SL", "SL-M"})
#: The kernel's terminal order states and the order stream's terminal words.
KERNEL_ENDED = frozenset({"Filled", "Canceled", "Expired", "Rejected"})
NEWS_ENDED = frozenset({"complete", "cancelled", "rejected"})
#: The calls that send an order or read a book (8).
ORDER_CALLS = frozenset({"placeorder", "modifyorder", "cancelorder"})
BOOK_READS = frozenset({"orderstatus", "orderbook", "tradebook", "positionbook", "funds"})
#: A price the broker's fills hold, within half a paisa.
PRICE_TOLERANCE = 0.005


def assert_session_invariants(session: SessionTrace, fake: FakeOpenAlgo) -> None:
    """Check every invariant of the module docstring over `session`, the trace of the
    sessions run on `fake`. Raises AssertionError naming the first one broken."""
    if session.opted_out or not session.runs:
        return
    mine = session.session_clients()
    last = session.runs[-1]
    # The changes up to the last run's end: what a test does after it (an operator's
    # orders once the session has ended) is not the session's.
    changes = session.changes[: last.ended_at]
    own = [c for c in changes if c.client in mine]
    sides = _sides(own)
    _one_stop_per_leg(changes, mine)
    by_hand = _within_the_position(changes, mine, sides)
    closing = last.closing or _closing_now(last)
    working = _working_at_the_end(own)
    held = {c.instrument: c.position for c in changes}
    _no_orphan(last, closing, working, held, by_hand)
    _protected(session, last, closing, working, held, sides, by_hand)
    if last.outcome is not None:
        _books_agree(last, closing, held, frozenset(session.unprotected))
    unfitted = {c.strategy for c in own if c.instrument in by_hand}
    excused = frozenset(session.unprotected) | frozenset(unfitted)
    for run in session.runs:
        _modifies_and_cancels_answered(run, excused)
        _nothing_after_a_guard_lock(run, fake)
        _priced_fills(run, fake)


# ----- at every change ----------------------------------------------------------------------


def _sides(own: list[OrderChange]) -> dict[tuple[Mode, str, str], str]:
    """The open side of each instrument the session traded: its first order's action."""
    sides: dict[tuple[Mode, str, str], str] = {}
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
    changes: list[OrderChange], mine: frozenset[str], sides: dict[tuple, str]
) -> set[tuple]:
    """2 and 3 (module docstring); the instruments a sale by hand left unfitted at the end."""
    orders: dict[str, OrderChange] = {}
    by_hand: set[tuple] = set()
    for change in changes:
        orders[change.order_id] = change
        side = sides.get(change.instrument)
        if side is None:
            continue
        if change.client not in mine:
            if change.fill_units and change.action != side:
                by_hand.add(change.instrument)
            continue
        closing = sum(
            o.remaining
            for o in orders.values()
            if o.client in mine and o.working and o.instrument == change.instrument
            and o.action != side
        )  # fmt: skip
        held = change.position if side == "BUY" else -change.position
        if change.instrument in by_hand:
            if held >= 0 and closing <= held:
                by_hand.discard(change.instrument)
            continue
        assert closing <= max(held, 0), (
            f"7.1: the session's working exits of {change.instrument} hold {closing} units "
            f"against a position of {change.position} ({change})"
        )
        assert not change.fill_units or held >= 0, (
            f"7.1: a fill of the session's order {change.order_id} left {change.instrument} "
            f"at {change.position}, the other side of the leg ({change})"
        )
    return by_hand


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
    held: dict[tuple, int],
    by_hand: set[tuple],
) -> None:
    """4; a stop a sale by hand left on a flat position is the operator's to cancel, as
    the close's alert says (`_reconcile_close.NOTHING_TO_CLOSE`)."""
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
            assert order.instrument in by_hand, (
                f"7.1: stop {order.order_id} works for a flat position ({order})"
            )


def _protected(
    trace: SessionTrace,
    run: Run,
    closing: dict[str, Closing],
    working: list[OrderChange],
    positions: dict[tuple, int],
    sides: dict[tuple, str],
    by_hand: set[tuple],
) -> None:
    """5, and 2 once more at the end; a leg a sale by hand left unfitted, or one the test
    leaves unprotected, is only checked not to be claimed protected."""
    for leg in closing.values():
        instrument = (run.mode, leg.symbol, leg.exchange)
        position = positions.get(instrument, 0)
        # The units the fake holds on the leg's side (none when another client's orders
        # took the account to the other side: no stop of the leg's can hold those).
        held = max(position if leg.units > 0 else -position, 0)
        if leg.strategy in trace.unprotected or instrument in by_hand:
            _named_unprotected(run, leg)
            continue
        side = sides.get(instrument)
        exits = sum(o.remaining for o in working if o.instrument == instrument and o.action != side)
        assert exits <= held, (
            f"7.1: {leg.strategy} ends with exits of {exits} units working at the fake "
            f"against a position of {position} {leg.symbol}"
        )
        if not held or not leg.wants_stop or not leg.units:
            continue
        stops = sum(
            o.remaining for o in working if o.instrument == instrument and o.action != side
            and o.price_type in STOP_TYPES
        )  # fmt: skip
        needed = min(held, abs(round(leg.units)))
        assert min(stops, leg.stop_covers) >= needed, (
            f"7.1: {leg.strategy} ends with {position} units of {leg.symbol} at the fake, "
            f"stops of {stops} working there and {leg.stop_covers} confirmed by the session, "
            f"for the engine's {leg.units:g}"
        )


def _named_unprotected(run: Run, leg: Closing) -> None:
    """A leg left unprotected or unfitted: `open_positions` claims no confirmed stop of
    it."""
    if run.outcome is None:
        return
    rows = [p for p in run.outcome.open_positions if p.strategy == leg.strategy]
    assert not any(p.stop_confirmed for p in rows), (
        f"7.1: {leg.strategy} is left unprotected but open_positions says {rows}"
    )


def _books_agree(
    run: Run, closing: dict[str, Closing], positions: dict[tuple, int], excused: frozenset[str]
) -> None:
    """6: a run stopped by an error or a mode breach reads no books at its close (3.5, 6.2)
    and a leg the test leaves unprotected has its own note to assert, so neither is
    checked."""
    if run.outcome.stopped_by in ("error", "mode_breach"):
        return
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
        if strategies & excused:
            continue
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


# ----- over each run's event stream ---------------------------------------------------------


def _modifies_and_cancels_answered(run: Run, excused: frozenset[str]) -> None:
    """7; a strategy the test leaves unprotected (`SessionTrace.leave_unprotected`), or
    whose stop a sale by hand left unfitted, may end with the modify that would have
    fitted its stop unanswered at the close (16-session-design 3.3 step 5 "As built": the
    close names that cut in its ERROR alert)."""
    events = run.events
    ended = any(e["kind"] == "closed" for e in events) or run.outcome is None
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
            cut_short = run.outcome is None or (ended and request["strategy"] in excused)
            assert applied or news or cut_short, (
                f"7.1: modify {request} got no apply_modify, no refusal and no terminal news"
            )
        else:
            assert news or "cancel_refused" in answers or ended, (
                f"7.1: cancel {request} got no terminal news and no refusal"
            )


def _nothing_after_a_guard_lock(run: Run, fake: FakeOpenAlgo) -> None:
    breach = getattr(run.session, "breach", None)
    if breach is None:
        return
    late = [
        call
        for call in fake.log.calls()
        if call.ts_ns > breach.seen_ns and call.detail.get("client") in run.clients
        and call.what in ORDER_CALLS | BOOK_READS
    ]  # fmt: skip
    assert late == [], f"7.1: calls after the guard lock at {breach.seen_ns}: {late}"


def _priced_fills(run: Run, fake: FakeOpenAlgo) -> None:
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


__all__ = ["assert_session_invariants"]
