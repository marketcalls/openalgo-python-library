"""The order worker's GTT calls (20-positional-live 3.1, 3.6; the lead's ruling N3a-1):
`placegttorder`, `modifygttorder` and `cancelgttorder` on the orders client, each one order
call under the order rate, each answer told to the server watch (a timeout holds every
send) and checked by the mode guard, and read as what became of the call (`CallResult`).

- **The payload** of a SINGLE GTT: the trigger in the field of its direction
  (`triggerprice_sl` for a trigger below the price, `triggerprice_tg` above it, the other
  0; 3.6), the child's limit as `price` with price type LIMIT, the units as `quantity`, the
  product given. A modify is a full replacement: every field again, with the trigger ID.
- **The answer**: a success with a trigger ID (a modify's or cancel's success needs none)
  is ACCEPTED; a refusal that went no further (a 4xx, an error OpenAlgo answered with 200)
  is REFUSED with the broker's message; a timeout, a dropped connection, a 5xx (the
  server's "due to internal error": the broker may have acted) or a success that names no
  trigger is UNANSWERED; semi-auto's queue in the Action Center is QUEUED. The same reading
  of an error as the order path's (`_order_answers.outcome_of`). Each result says when its
  call was sent and whether the server has finished with it (`server_has_finished`): an
  unanswered call the server may still be working on can still land at the broker, which
  the confirmations wait for (`_gtt_confirm.misses_from_ns`).

Prices are rupees, quantities units. Used by the worker's thread only.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ._answers import server_answer
from ._gtt_types import Direction, GttRequest
from ._order_answers import Outcome, outcome_of, server_has_finished
from ._order_rate import RateBudget
from ._types import GuardLike, Now, Sleep
from .orders import OrderPath

#: OpenAlgo's trigger type of a GTT with one trigger and one child order (P6).
SINGLE = "SINGLE"
#: The price type of a GTT's child order: a LIMIT at the GTT's limit.
CHILD_PRICE_TYPE = "LIMIT"


class CallKind(enum.Enum):
    """What became of one GTT call (module docstring)."""

    ACCEPTED = "accepted"
    REFUSED = "refused"
    UNANSWERED = "unanswered"
    QUEUED = "queued"


@dataclass(frozen=True)
class CallResult:
    """A GTT call's result: its kind, the broker's trigger ID of an accepted place, the
    message of anything else (the broker's words of a refusal), when the call was sent (UTC
    nanoseconds, 0 when not known) and whether the server has finished with it (False only
    for an UNANSWERED call the server may still be working on: a client timeout, a dropped
    connection, a 5xx page)."""

    kind: CallKind
    trigger_id: str = ""
    message: str = ""
    sent_ns: int = 0
    finished: bool = True


def place_fields(request: GttRequest) -> dict[str, Any]:
    """The keyword arguments of `placegttorder` for `request` (module docstring)."""
    below = request.direction is Direction.BELOW
    return {
        "strategy": request.strategy,
        "symbol": request.symbol,
        "action": request.side,
        "exchange": request.exchange,
        "trigger_type": SINGLE,
        "product": request.product,
        "quantity": request.quantity,
        "price_type": CHILD_PRICE_TYPE,
        "price": request.limit_price,
        "triggerprice_sl": request.trigger_price if below else 0.0,
        "triggerprice_tg": 0.0 if below else request.trigger_price,
    }


class GttCalls:
    """The GTT calls through `client` (the session's orders client), counted in `budget`
    (the order rate's window, `_order_throttle`), each answer told to `path`'s server watch
    and checked by `guard`; `sleep` waits for the budget when a call follows another within
    one request (a place again after an unanswered one); `now` gives UTC nanoseconds."""

    def __init__(
        self,
        client: Any,
        *,
        path: OrderPath,
        guard: GuardLike,
        budget: RateBudget,
        sleep: Sleep,
        now: Now,
    ) -> None:
        self._client = client
        self._path = path
        self._guard = guard
        self._budget = budget
        self._sleep = sleep
        self._now = now

    def place(self, request: GttRequest) -> CallResult:
        """Place `request`'s GTT (module docstring)."""
        return self._call(self._client.placegttorder, True, **place_fields(request))

    def modify(self, request: GttRequest, trigger_id: str) -> CallResult:
        """Replace the terms of the broker's GTT `trigger_id` with `request`'s."""
        fields = place_fields(request) | {"trigger_id": trigger_id}
        return self._call(self._client.modifygttorder, False, **fields)

    def cancel(self, strategy: str, trigger_id: str) -> CallResult:
        """Cancel the broker's GTT `trigger_id` of `strategy`."""
        return self._call(
            self._client.cancelgttorder, False, trigger_id=trigger_id, strategy=strategy
        )

    def _call(self, method: Callable[..., Any], wants_trigger: bool, **fields: Any) -> CallResult:
        while (wait := self._budget.wait_seconds()) > 0:
            self._sleep(wait)
        self._budget.take()
        sent_ns = self._now()
        answer = server_answer(method(**fields))
        self._path.note_answer(answer)
        self._guard.check_response(answer)
        return call_result(answer, wants_trigger=wants_trigger, sent_ns=sent_ns)


def call_result(answer: Mapping[str, Any], *, wants_trigger: bool, sent_ns: int = 0) -> CallResult:
    """What `answer` (`server_answer`'s) says became of a GTT call sent at `sent_ns` (UTC
    nanoseconds); `wants_trigger` for a place, whose success must name the broker's
    trigger ID (module docstring)."""
    outcome = outcome_of(answer)
    if outcome.kind is Outcome.QUEUED:
        return CallResult(CallKind.QUEUED, message=outcome.reason, sent_ns=sent_ns)
    if answer.get("status") == "success":
        trigger_id = str(answer.get("trigger_id") or "")
        if wants_trigger and not trigger_id:
            words = "OpenAlgo answered success, no trigger"
            return CallResult(CallKind.UNANSWERED, message=words, sent_ns=sent_ns)
        return CallResult(CallKind.ACCEPTED, trigger_id=trigger_id, sent_ns=sent_ns)
    if outcome.kind is Outcome.REJECTED:
        return CallResult(CallKind.REFUSED, message=outcome.reason, sent_ns=sent_ns)
    finished = server_has_finished(answer)
    return CallResult(
        CallKind.UNANSWERED, message=outcome.reason, sent_ns=sent_ns, finished=finished
    )


__all__ = [
    "CHILD_PRICE_TYPE",
    "SINGLE",
    "CallKind",
    "CallResult",
    "GttCalls",
    "call_result",
    "place_fields",
]
