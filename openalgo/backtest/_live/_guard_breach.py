"""The record of why the mode guard locked, and the alert that says it (08-live-trading
section 3). `guard.ModeGuard` builds them; `guard` re-exports `ModeBreach` and `SeenIn`.
"""

from __future__ import annotations

import enum
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from .._calendar_convert import format_ist
from .._products import OVERNIGHT_PRODUCTS
from ._types import Mode


class SeenIn(enum.Enum):
    """Where the guard saw what locked it; the value is how its alert says it."""

    POLL = "the analyzerstatus poll"
    BEFORE_ENTRY = "the analyzerstatus read before an entry"
    RESPONSE = "an order response"
    UPDATE = "an order update"


@dataclass(frozen=True)
class ModeBreach:
    """Why the guard locked.

    `session_mode` is the mode adopted at start. `seen_mode` is the other mode the guard
    saw, or None when it saw semi-auto: an order queued in the Action Center (then
    `pending_order_id` is its id there) or a modify or cancel refused for semi-auto.
    `seen_ns` is when it was seen (UTC nanoseconds) and `seen_in` where.

    `orders_after` are the order IDs whose book is in doubt, in the order they were sent:
    those sent since the switch was last read in the session's mode and not confirmed by
    their own response or update, the order in the response that showed the other mode,
    and any order noted after the lock. `intents_in_doubt` are the intents whose order
    may have reached the server with no order ID (a timeout, an HTTP 502, a lost
    connection) since that read, and any noted after the lock.
    """

    session_mode: Mode
    seen_mode: Mode | None
    seen_ns: int
    seen_in: SeenIn
    orders_after: tuple[str, ...]
    pending_order_id: str | None = None
    intents_in_doubt: tuple[str, ...] = ()

    @property
    def semi_auto(self) -> bool:
        """True when the guard locked on semi-auto rather than on the analyzer switch."""
        return self.seen_mode is None


def breach_text(breach: ModeBreach) -> str:
    """The MODE_BREACH alert for a breach, also the message of its ModeChangedError: what
    was seen, where and when (IST), the orders in doubt, that the session sends nothing
    more, and how to start it again."""
    seen = f"seen in {breach.seen_in.value} at {format_ist(breach.seen_ns)}"
    if breach.seen_mode is None:
        what = (
            f"OpenAlgo is in semi-auto mode ({seen}): it queues new orders in its Action "
            f"Center for approval and refuses to modify or cancel orders."
        )
        if breach.pending_order_id:
            what += (
                f" Pending order {breach.pending_order_id} is waiting for approval and is "
                f"not placed; reject it in the Action Center."
            )
        again = "Set the order mode back to auto on OpenAlgo's API key page, then start it"
    else:
        what = (
            f"OpenAlgo's analyzer switch changed during the session: it started in "
            f"{breach.session_mode.name} mode and {breach.seen_mode.name} mode was {seen}."
        )
        again = "Start it"
    orders = ", ".join(breach.orders_after) if breach.orders_after else "none"
    text = f"{what} Orders sent after the change (since the mode was last confirmed): {orders}."
    if breach.intents_in_doubt:
        text += (
            f" Orders that may have reached the server with no order ID (intents): "
            f"{', '.join(breach.intents_in_doubt)}."
        )
    return f"{text} The session sends nothing more. {again} again in the mode you want."


def breach_alert(
    positions: Sequence[str],
    products: Collection[str] = (),
    *,
    intraday: bool = True,
    dropped: Sequence[str] = (),
) -> str:
    """The session's ERROR alert when the mode guard locked (16-session-design 6.2;
    20-positional-live 10.4): it sends nothing more, each position as last known, in words
    with its stop (`positions`), the orders it now never sends (`dropped`: a pre-open
    decision's, one line each), and what becomes of the positions by the `products` they
    are held in: in a session that runs an MIS strategy (`intraday`), the broker's MIS
    square-off closes the MIS positions (said also when none is open), and the CNC and
    NRML positions stay open overnight, so their stops are to be checked at the broker."""
    held = "; ".join(positions) if positions else "no open position"
    carried = any(product in OVERNIGHT_PRODUCTS for product in products)
    squared = not products or any(product not in OVERNIGHT_PRODUCTS for product in products)
    fates = []
    if intraday and squared:
        fates.append(
            "The broker's MIS square-off closes "
            + ("the MIS positions" if carried else "what is left")
        )
    if carried:
        fates.append(
            "CNC and NRML positions stay open overnight, so check each one's stop at the broker"
        )
    fates.append("stop the session with Ctrl+C")
    told = "; ".join(fates)
    unsent = "".join(f" {line}." for line in dropped)
    return (
        f"OpenAlgo's mode changed: the session sends nothing more. Positions as last known: "
        f"{held}.{unsent} {told[0].upper()}{told[1:]}"
    )


__all__ = ["ModeBreach", "SeenIn", "breach_alert", "breach_text"]
