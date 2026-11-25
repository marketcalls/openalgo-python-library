"""The GTT routes: `placegttorder`, `modifygttorder` and `cancelgttorder`, each a function of
the venue and the request's JSON that returns (body, HTTP status), run under the venue's
lock. SINGLE GTTs only: an OCO is a request the fake does not model.

The order of the checks and the words are the server's: the REST layer's schemas
(`schema.PLACE_GTT`, `MODIFY_GTT`, `CANCEL_GTT`, restx_api/schemas.py, with
`_validate_gtt_place_request`), then the switch's book. The sandbox answers every GTT call
before it looks at the broker, with "mode": "analyze" (services/place_gtt_order_service.py
and its siblings; sandbox/gtt_manager.py: an ID "GTT-YYMMDD-...", a modify or cancel of a
GTT that is not active answered 404 "No active GTT with trigger_id '...'"). Live, a broker
without GTT orders is OpenAlgo's HTTP 501 whatever the login, an expired login the
broker's 403 refusal, and a refusal a test schedules (`venue_gtt.GttVenue.refuse_next_gtt`)
the broker's HTTP 400 with its words, as is a modify or cancel of a GTT that is not
active (`BROKER_GTT_WORDS`: the fake broker's own words). A modify a test has applied in
part changes the trigger only and fails as the server's internal error (HTTP 500).
"""

from __future__ import annotations

from typing import Any

from openalgo.backtest._live._types import Mode

from . import schema
from .answers import EXPIRED_TOKEN, GTT_NOT_SUPPORTED, MODIFY_GTT_FAILED, Unmodelled
from .book import Book
from .gtts import ABOVE, BELOW, CANCELLED, FakeGtt
from .venue import Venue

Answer = tuple[dict[str, Any], int]

#: The fake broker's refusals of a GTT call (HTTP 400).
BROKER_GTT_WORDS = {
    "unknown": "Instrument {exchange}:{symbol} is not tradable at the broker",
    "not_active": "Trigger {trigger_id} is not active and cannot be changed",
    "immutable": "The symbol, exchange and side of trigger {trigger_id} cannot be changed",
}
_SINGLE = "SINGLE"


def place(venue: Venue, payload: dict[str, Any]) -> Answer:
    """`placegttorder`: the schema, then the sandbox or the broker (module docstring)."""
    loaded, problem = _loaded(venue, schema.PLACE_GTT, payload)
    if problem:
        return _refused(venue, problem)
    if venue.mode is Mode.LIVE:
        refusal = _live_refusal(venue)
        if refusal is not None:
            return refusal
    market = venue.market(loaded["symbol"], loaded["exchange"])
    reason = venue.take_gtt_refusal("placegttorder")
    if reason is None and market is None:
        reason = BROKER_GTT_WORDS["unknown"].format(**loaded)
    if reason is not None or market is None:
        return _refused(venue, reason or "")
    book, now = venue.book, venue.clock.now()
    trigger, direction = _trigger(loaded)
    gtt = FakeGtt(
        trigger_id=book.gtt_ids.trigger_id(now),
        strategy=loaded["strategy"],
        symbol=loaded["symbol"],
        exchange=loaded["exchange"],
        action=loaded["action"],
        product=loaded["product"],
        quantity=loaded["quantity"],
        price=loaded["price"],
        trigger=trigger,
        direction=direction,
        created_ns=now,
        last_price=market.last_price or 0.0,
    )
    book.gtts[gtt.trigger_id] = gtt
    return _marked(venue, {"status": "success", "trigger_id": gtt.trigger_id}), 200


def modify(venue: Venue, payload: dict[str, Any]) -> Answer:
    """`modifygttorder`: a full replacement of an active GTT's terms (module docstring)."""
    loaded, problem = _loaded(venue, schema.MODIFY_GTT, payload)
    if problem:
        return _refused(venue, problem)
    if venue.mode is Mode.LIVE:
        refusal = _live_refusal(venue)
        if refusal is not None:
            return refusal
    trigger_id = loaded["trigger_id"]
    gtt = _active(venue.book, trigger_id)
    if gtt is None:
        return _not_active(venue, trigger_id)
    same = (loaded["symbol"], loaded["exchange"], loaded["action"])
    if same != (gtt.symbol, gtt.exchange, gtt.action):
        return _refused(venue, BROKER_GTT_WORDS["immutable"].format(trigger_id=trigger_id))
    reason = venue.take_gtt_refusal("modifygttorder")
    if reason is not None:
        return _refused(venue, reason)
    trigger, direction = _trigger(loaded)
    gtt.updated_ns = venue.clock.now()
    if venue.take_half_modify():
        gtt.trigger, gtt.direction = trigger, direction
        return {"status": "error", "message": MODIFY_GTT_FAILED}, 500
    gtt.trigger, gtt.direction = trigger, direction
    gtt.quantity, gtt.price, gtt.product = loaded["quantity"], loaded["price"], loaded["product"]
    return _marked(venue, {"status": "success", "trigger_id": trigger_id}), 200


def cancel(venue: Venue, payload: dict[str, Any]) -> Answer:
    """`cancelgttorder`: an active GTT ends cancelled (module docstring)."""
    loaded, problem = _loaded(venue, schema.CANCEL_GTT, payload, exclude_unknown=False)
    if problem:
        return _refused(venue, problem)
    if venue.mode is Mode.LIVE:
        refusal = _live_refusal(venue)
        if refusal is not None:
            return refusal
    trigger_id = loaded["trigger_id"]
    gtt = _active(venue.book, trigger_id)
    if gtt is None:
        return _not_active(venue, trigger_id)
    reason = venue.take_gtt_refusal("cancelgttorder")
    if reason is not None:
        return _refused(venue, reason)
    gtt.status, gtt.updated_ns = CANCELLED, venue.clock.now()
    return _marked(venue, {"status": "success", "trigger_id": trigger_id}), 200


def _loaded(
    venue: Venue,
    fields: tuple[schema.Field, ...],
    payload: dict[str, Any],
    *,
    exclude_unknown: bool = True,
) -> tuple[dict[str, Any], str]:
    """The request as the schema loads it, normalised as `_validate_gtt_place_request`
    does for a SINGLE GTT; "" when it passed, else the message. Raises Unmodelled for an
    OCO GTT and for semi-auto mode, which the fake does not model for GTT calls."""
    if venue.semi_auto:
        raise Unmodelled("the fake OpenAlgo does not model GTT calls in semi-auto mode")
    loaded, problem = schema.load(fields, payload, exclude_unknown=exclude_unknown)
    if problem or "trigger_type" not in loaded:
        return loaded, problem
    if loaded["trigger_type"].upper() != _SINGLE:
        raise Unmodelled(f"the fake OpenAlgo models SINGLE GTTs only, not {loaded['trigger_type']}")
    if loaded["triggerprice_sl"] <= 0 and loaded["triggerprice_tg"] <= 0:
        words = "SINGLE GTT requires a positive triggerprice_sl or triggerprice_tg."
        return {}, str({"triggerprice_sl": [words]})
    loaded["action"] = loaded["action"].upper()
    return loaded, ""


def _trigger(loaded: dict[str, Any]) -> tuple[float, str]:
    """The trigger and where it sits: `triggerprice_sl` when above 0 (below the price),
    else `triggerprice_tg` (above it), as the server resolves a SINGLE GTT."""
    if loaded["triggerprice_sl"] > 0:
        return loaded["triggerprice_sl"], BELOW
    return loaded["triggerprice_tg"], ABOVE


def _active(book: Book, trigger_id: str) -> FakeGtt | None:
    gtt = book.gtts.get(trigger_id)
    return gtt if gtt is not None and gtt.active else None


def _live_refusal(venue: Venue) -> Answer | None:
    """OpenAlgo's 501 for a broker without GTT orders, then the broker's 403 for an
    expired login; None when the broker takes the call."""
    if not venue.gtt_supported:
        return {"status": "error", "message": GTT_NOT_SUPPORTED.format(broker=venue.broker)}, 501
    if venue.session_expired:
        return {"status": "error", "message": EXPIRED_TOKEN}, 403
    return None


def _not_active(venue: Venue, trigger_id: str) -> Answer:
    if venue.mode is Mode.SANDBOX:
        words = f"No active GTT with trigger_id '{trigger_id}'"
        return _marked(venue, {"status": "error", "message": words}), 404
    return _refused(venue, BROKER_GTT_WORDS["not_active"].format(trigger_id=trigger_id))


def _refused(venue: Venue, message: str) -> Answer:
    return _marked(venue, {"status": "error", "message": message}), 400


def _marked(venue: Venue, body: dict[str, Any]) -> dict[str, Any]:
    """The body with the sandbox's `"mode": "analyze"` when the switch is on."""
    return {**body, "mode": "analyze"} if venue.mode is Mode.SANDBOX else body


__all__ = ["BROKER_GTT_WORDS", "cancel", "modify", "place"]
