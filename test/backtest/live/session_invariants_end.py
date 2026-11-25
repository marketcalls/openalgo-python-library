"""The session invariants checked when the last run on the fake OpenAlgo ends
(`session_invariants` 4 to 6; 16-session-design 7.1, unit 14), of that run (the
restarted one after a crash): what works at the fake then, against each strategy's kernel
state as the close stopped it (`Closing`; as it stands, for a run that crashed).

4. **No orphan.** Every order of the session's working at the fake is a working kernel
   order of that run (bound to its broker ID), and no stop of it works for a position the
   fake holds flat, or for a member of an instrument group whose own leg is flat
   (18-multi-strategy I1).
5. **Protected.** Each position the fake holds on the side of a leg that wants a stop is
   covered by the session's working stops of that leg's own strategy (18-multi-strategy
   I7: per member), and by its active GTTs at the fake for a CNC or NRML leg
   (20-positional-live 2: a positional leg's stop is a GTT, `gtt_stops`), confirmed working
   by its stops supervisor,
   unless the test leaves the leg unprotected by a row 7.1 lists or a decided end (then
   named: a row of `open_positions` with no confirmed stop and an ERROR alert naming the
   strategy). 2 is asserted once more here, over what works at the end.
6. **Books agree.** The engine's units and the fake's are equal per instrument, or the
   run's `reconciliations` or the close's named legs hold the difference.

What a sale by hand left (`SaleRoom`) stands at the end only by the test's decided end:
a stop larger than the position, or on the flat one, that was working when the sale
filled and that an ERROR alert names by its order ID (`SessionTrace.kept_stop_by_decision`),
and a stop that filled through zero, its leg then listed at the fake's units with no
confirmed stop and named by an ERROR alert (`SessionTrace.oversold_by_decision`). Each
such declaration must be needed: a decided end no check took is found.

Quantities are signed units, prices rupees, times UTC nanoseconds.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from typing import Any

from session_invariants_changes import STOP_TYPES, Instrument, SaleRoom
from session_trace import (
    NO_ORPHAN,
    OTHER_SIDE,
    PROTECTED,
    WITHIN,
    Closing,
    OrderChange,
    Run,
    SessionTrace,
    closing_of,
)

from openalgo.backtest._live._types import AlertKind


def closing_now(run: Run) -> dict[str, Closing]:
    """A run with no close (a crash): each strategy's kernel state as it stands now."""
    found = {}
    for trader in getattr(run.session, "traders", ()):
        closing = closing_of(run.session, trader)
        if closing is not None:
            found[trader.name] = closing
    return found


def gtt_stops(fake: Any, mode: Any) -> dict[tuple[str, Instrument], int]:
    """The units of the active GTTs at the fake's book of `mode`, by strategy and
    instrument: a CNC or NRML leg's stop (module docstring, 5)."""
    held: dict[tuple[str, Instrument], int] = {}
    for gtt in fake.gtts(mode):
        if gtt.active:
            key = (gtt.strategy, (mode, gtt.symbol, gtt.exchange))
            held[key] = held.get(key, 0) + gtt.quantity
    return held


def working_at_the_end(own: list[OrderChange]) -> list[OrderChange]:
    """The session's orders working at the fake after the last of `own` changes."""
    last: dict[str, OrderChange] = {}
    for change in own:
        last[change.order_id] = change
    return [c for c in last.values() if c.working]


def no_orphan(
    run: Run,
    closing: dict[str, Closing],
    working: list[OrderChange],
    held: dict[Instrument, int],
    rooms: Mapping[Instrument, SaleRoom],
    kept: Mapping[str, str],
) -> set[str]:
    """4 (module docstring); a stop on a position a sale by hand left flat stands only by
    the test's decided end (`kept`), named by an ERROR alert. Returns the strategies whose
    decided end it took so (`at_the_end` judges the declarations no check needed)."""
    named: set[str] = set()
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
        # A flat position of the instrument, or of the member the stop is of
        # (18-multi-strategy I1, per member).
        member = closing.get(order.strategy)
        flat = held.get(order.instrument, 0) == 0 or (member is not None and not member.units)
        if order.mode is run.mode and order.price_type in STOP_TYPES and flat:
            room = rooms.get(order.instrument)
            assert room is not None and room.took(order), (
                f"7.1: stop {order.order_id} works for a flat position ({order})"
            )
            _kept_stop_named(run, order, kept)
            named.add(order.strategy)
    return named


def at_the_end(
    trace: SessionTrace,
    run: Run,
    closing: dict[str, Closing],
    working: list[OrderChange],
    positions: dict[Instrument, int],
    sides: Mapping[Instrument, str],
    rooms: Mapping[Instrument, SaleRoom],
    *,
    kept_named: Set[str] = frozenset(),
    oversold_used: Set[str] = frozenset(),
    gtt_stops: Mapping[tuple[str, Instrument], int] | None = None,
) -> None:
    """2 once more and 5 (module docstring), and the legs a decided end oversold; a leg
    the test leaves unprotected must need it (the fake holds units on its side at the
    end), and so must every decided end the test names: a kept stop one that a check took
    (`kept_named`, the strategies whose stop `no_orphan` took so, and those 2 takes here),
    an oversold leg one whose fill through zero the changes allowed (`oversold_used`) or
    that ends on the other side."""
    named: set[str] = set()
    kept = set(kept_named)
    oversold = set(oversold_used)
    for leg in closing.values():
        instrument = (run.mode, leg.symbol, leg.exchange)
        opens = sides.get(instrument)
        position = positions.get(instrument, 0)
        on_its_side = position if opens != "SELL" else -position
        if on_its_side < 0 and leg.strategy in trace.oversold:
            _oversold_named(run, leg, position, trace.oversold[leg.strategy])
            oversold.add(leg.strategy)
        # The units the fake holds on the leg's side (none when the account is flat or on
        # the other side: no stop of the leg's can hold those).
        held = max(on_its_side, 0)
        exits = [o for o in working if o.instrument == instrument and o.action != opens]
        kept |= _exits_within(trace, run, leg, exits, held, position, rooms.get(instrument))
        if PROTECTED in trace.waived or not held:
            continue
        if leg.strategy in trace.unprotected:
            _named_unprotected(run, leg, trace.unprotected[leg.strategy])
            named.add(leg.strategy)
            continue
        if not leg.wants_stop or not leg.units:
            continue
        # Its own stops (18-multi-strategy I7, per member): another member's stop on the
        # instrument covers that member's units, never this one's.
        own = [o for o in exits if o.strategy == leg.strategy]
        stops = sum(o.remaining for o in own if o.price_type in STOP_TYPES)
        stops += (gtt_stops or {}).get((leg.strategy, instrument), 0)
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
    _declarations_needed(trace, kept, oversold)


def _declarations_needed(trace: SessionTrace, kept: Set[str], oversold: Set[str]) -> None:
    """A decided end the test names that no check took is a declaration it no longer
    needs: found, so none stays silently. A kept stop is judged only while both checks
    that take one run (4 and 2), an oversold leg while 3 runs."""
    unneeded = sorted(set(trace.kept_stops) - kept)
    judged = not {NO_ORPHAN, WITHIN} & set(trace.waived)
    assert not judged or not unneeded, (
        f"7.1: the test names kept_stop_by_decision for {unneeded}, but no stop of theirs "
        "stood beyond the position a sale by hand left"
    )
    unneeded = sorted(set(trace.oversold) - oversold)
    assert OTHER_SIDE in trace.waived or not unneeded, (
        f"7.1: the test names oversold_by_decision for {unneeded}, but no fill of theirs "
        "took the account through zero"
    )


def books_agree(run: Run, closing: dict[str, Closing], positions: dict[Instrument, int]) -> None:
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


def _exits_within(
    trace: SessionTrace,
    run: Run,
    leg: Closing,
    exits: list[OrderChange],
    held: int,
    position: int,
    room: SaleRoom | None,
) -> set[str]:
    """2 over the leg's exits working at the end: beyond what the fake holds on its side
    only by the room a sale by hand left, and each exit standing there named. Returns the
    strategies whose decided end it took so."""
    total = sum(o.remaining for o in exits)
    if WITHIN in trace.waived or total <= held:
        return set()
    assert room is not None and total <= held + room.room(exits), (
        f"7.1: {leg.strategy} ends with exits of {total} units working at the fake "
        f"against a position of {position} {leg.symbol}"
    )
    named = set()
    for order in exits:
        if room.took(order):
            _kept_stop_named(run, order, trace.kept_stops)
            named.add(order.strategy)
    return named


def _kept_stop_named(run: Run, order: OrderChange, kept: Mapping[str, str]) -> None:
    """A stop a sale by hand left beyond the position at the end: the test's decided end
    keeps it (`kept`), and an ERROR alert of the run names its order ID."""
    assert order.strategy in kept, (
        f"7.1: stop {order.order_id} of {order.strategy} works for more than the position "
        f"another client's sale left at the fake ({order}), and the test names no decided "
        "end that keeps it"
    )
    named = [t for kind, t in run.alerts if kind is AlertKind.ERROR and order.order_id in t]
    assert named, (
        f"7.1: stop {order.order_id} of {order.strategy} is kept beyond the position "
        f"({kept[order.strategy]}), and no ERROR alert names it"
    )


def _oversold_named(run: Run, leg: Closing, position: int, why: str) -> None:
    """A leg whose stop a decided end let fill through zero (`why`): `open_positions`
    lists it at the fake's units with no confirmed stop, and an ERROR alert names it."""
    assert run.outcome is not None, (
        f"7.1: {leg.strategy} ends oversold ({why}), but the run has no outcome to list it in"
    )
    rows = [
        (p.units, p.stop_confirmed)
        for p in run.outcome.open_positions
        if (p.strategy, p.symbol, p.exchange) == (leg.strategy, leg.symbol, leg.exchange)
    ]
    assert rows == [(position, False)], (
        f"7.1: {leg.strategy} ends with {position} units of {leg.symbol} at the fake "
        f"({why}), but open_positions lists {rows}"
    )
    named = [text for kind, text in run.alerts if kind is AlertKind.ERROR and leg.strategy in text]
    assert named, f"7.1: {leg.strategy} ends oversold ({why}) and no ERROR alert names it"


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


def _bound_order_ids(run: Run) -> set[str]:
    found = set()
    for trader in getattr(run.session, "traders", ()):
        found |= {placed.order_id for _, placed in trader.venue.placed() if placed.order_id}
    return found


__all__ = ["at_the_end", "books_agree", "closing_now", "no_orphan", "working_at_the_end"]
