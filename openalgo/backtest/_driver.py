"""The driver of an event strategy: the loop over the kernel's decision points.

    pause = engine.advance()            # the kernel runs slices to a decision point
    callbacks(pause events)             # on_order, on_fill, on_position, on_risk_event
    on_bar(the strategy bars closing)   # orders placed here queue for the next open
    ... until the pause is "finished"; then on_stop()

Callbacks get every event since the previous decision before `on_bar` runs, as in step 7
of 03-architecture 4.2. Events are dicts with the kernel's keys plus `symbol` and `time`
(the IST open of the slice the event happened in). A strategy bar with no base bar present
is not delivered (SPEC-07): a single-instrument strategy then gets no `on_bar` call.
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


def drive(strategy: Any, engine: Any, ctx: RunContext) -> None:
    """Run the strategy to the end of the grid. Errors of the kernel and of the strategy
    propagate unchanged (a strategy exception stops the run)."""
    single = strategy.symbol is not None
    while True:
        pause = call(engine.advance)
        deliver(strategy, ctx, pause["events"])
        if pause["kind"] == "finished":
            strategy.on_stop()
            return
        if pause["kind"] != "bar_close":
            raise NotImplementedError("timer pauses are not implemented yet (Phase 1 work)")
        bars = ctx.reveal(pause)
        if bars:
            strategy.on_bar(next(iter(bars.values())) if single else bars)


__all__ = ["deliver", "drive"]
