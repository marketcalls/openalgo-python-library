"""The driver of an event strategy: the loop over the kernel's decision points.

    pause = engine.advance()            # the kernel runs to a decision point
    at a timer pause (step 2):          # self.at / self.every are due at this open
        the timers' functions           # orders placed here fill at this same open
    at a bar-close pause (step 7):
        reveal the strategy bars        # data up to this slice's close
        callbacks(pause events)         # on_order, on_fill, on_position, on_risk_event
        on_bar(the strategy bars)       # orders placed here queue for the next open
    ... until the pause is "finished"; then on_stop()

Callbacks get every event since the previous step 7 pause before `on_bar` runs, as in
step 7 of 03-architecture 4.2, with the bars of this close already revealed; a timer
pause delivers no events (SPEC-31). Events are dicts with the kernel's keys plus
`symbol` and `time` (the IST open of the slice the event happened in). A strategy bar
with no base bar present is not delivered (SPEC-07): a single-instrument strategy then
gets no `on_bar` call.

Until the strategy's indicators are ready (`ready_at`, `_warm_up`) the pauses only
reveal their bars and deliver their events: no timer runs and `on_bar` is not called.
"""

from __future__ import annotations

from typing import Any

from ._context import RunContext
from ._engine import call
from ._series import ist_time


def _event_payload(ctx: RunContext, event: dict) -> dict:
    payload = dict(event)
    payload["time"] = ist_time(event["slice_ts"])
    if event.get("instrument") is not None:
        payload["symbol"] = ctx.symbol_of(event["instrument"])
    elif event["type"] == "order":
        payload["symbol"] = ctx.symbol_of(ctx.order(event["order_id"])["instrument"])
    return payload


def deliver(strategy: Any, ctx: RunContext, events: list[dict]) -> None:
    """Call the strategy's callbacks for each event, in order of occurrence."""
    for event in events:
        payload = _event_payload(ctx, event)
        if event["type"] == "order":
            strategy.on_order(payload)
            if event["name"] == "Filled":
                strategy.on_fill(payload)
        elif event["type"] == "position":
            strategy.on_position(payload)
        else:
            strategy.on_risk_event(payload)


def _wait(strategy: Any, advance: Any, ctx: RunContext, ready_at: int) -> bool:
    """Reveal the pauses before `ready_at` (UTC nanoseconds) and deliver their events,
    deciding nothing, then decide at the first pause at or after it as the main loop
    would; returns True when the run ended first."""
    while True:
        kind, decided_at, rows, events, timers = call(advance)
        bars = ctx.reveal(decided_at, rows)
        if events:
            deliver(strategy, ctx, events)
        if kind == "finished":
            strategy.on_stop()
            return True
        if decided_at >= ready_at:
            if kind == "timers":
                ctx.timers.fire(timers)
            elif bars:
                single = strategy.symbol is not None or strategy.underlying is not None
                strategy.on_bar(next(iter(bars.values())) if single else bars)
            return False


def drive(strategy: Any, engine: Any, ctx: RunContext, ready_at: int | None = None) -> None:
    """Run the strategy to the end of the grid, deciding from `ready_at` (UTC
    nanoseconds, None from the first pause). Errors of the kernel and of the strategy
    propagate unchanged (a strategy exception stops the run).

    This loop runs once per strategy-bar close, so its own cost is part of every
    callback (NFR-PERF-04): the kernel hands each pause over as a tuple (kind, decision
    instant, bar rows, events, timer ids) and nothing is built that the strategy does
    not receive. The wait for `ready_at` is a loop of its own, so it costs nothing once
    the strategy decides."""
    single = strategy.symbol is not None or strategy.underlying is not None
    advance = engine.advance
    if ready_at is not None and _wait(strategy, advance, ctx, ready_at):
        return
    while True:
        kind, decided_at, rows, events, timers = call(advance)
        if kind == "timers":
            ctx.reveal(decided_at, rows)
            ctx.timers.fire(timers)
            continue
        bars = ctx.reveal(decided_at, rows)
        if events:
            deliver(strategy, ctx, events)
        if kind == "finished":
            strategy.on_stop()
            return
        if bars:
            strategy.on_bar(next(iter(bars.values())) if single else bars)


__all__ = ["deliver", "drive"]
