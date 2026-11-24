"""A closing order rejected on a shared instrument (18-multi-strategy 4.14 as the lead
decided it in 11.0a items 1 and 2, and 4.14a; unit M6).

A broker nets the strategies of an account into one position per instrument and product.
When several strategies of the session trade one, a closing order the broker refuses may
mean it does not hold, on that side, the units the strategy believes it holds (a sale by
hand, a difference not seen yet): sending it again, exiting at market or placing the stop
again would meet the same refusal, or open the other side.

**Which rejections.** In a group of two or more members, any rejection of a closing order
(a leg's stop, a strategy exit, a risk exit, a square-off): its placement refused in
OpenAlgo's answer, or its REJECTED word after acceptance (`takes`). A group of one keeps
the path of a strategy alone exactly (the market exit of R3, the exit retries of
16-session-design 4.8). A placement that never reached the broker (not placed after its
resolution) or one the session withdrew is not a rejection. No rule depends on the
broker's words: the phrase table (`MARGIN_PHRASES`) only chooses the alert's words.

**What the session does** (`reject`): the kernel takes the rejection as
`apply_margin_reject` (KR12): the order ends, a stop is neither exited at market nor
placed again, and nothing closes the position by itself. The strategy's entries are
locked for the rest of the session (`MARGIN_LOCK`, latched); a book read is asked for at once, the
leg's exits held until it is compared (the reconciler's exit guard); one ERROR alert
names the leg, the order, the broker's reason, whether a stop still works, and what to
do. The leg counts as unprotected from the rejection (`unprotected_reason`), and the exit
retries and the bare-leg protector do not start for it (`holds`).

**The gate** (`request_exit`, `protect`). Every automatic exit or protect of a held
position (the kill switch, the safe stop, the stops deadline, the price rule, protect by
state, the square-off) waits for a comparison of its group that agrees, from a book read
begun after the rejection (`agreed`); it is then made, once, and its orders pass the exit
guard; an exit at a LIMIT (the price rule's) is priced again from the latest price then,
since the one it was asked at is stale by then. The callers' alerts say the exit waits
(`HELD_EXIT_WORDS`). From then on the session's own exits go again (the lock stays): at
the strategy's square-off the session asks the kernel for it (`duties`), since the hold
keeps the kernel from making it. A later rejection holds the position again, until
another read agrees.

**Restart.** The hold is not saved (the kernel's is not either), nor is its entry lock
set again (it is not one of `_session_saved.DAY_LOCKS`): after a restart the restart
comparison and the exit guard decide before anything is sent, and the restart's own
protect places a missing stop again. A rejection the restart learns from the journal
(the last save came before it) is held again as it is applied, so it sends no market
exit either.

Times are UTC nanoseconds, quantities units. Used by the loop only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .._engine import call
from ..errors import VenueRefusalError
from ._plan_catch_up import INSTRUMENT
from ._session_exits import EXIT_PURPOSES
from ._session_trader import Trader
from ._stops_cover import CLOSE_REJECTED, REFUSED_FOR_MARGIN
from ._stops_kernel import protect as kernel_protect
from ._types import LOGGER_NAME, AlertKind, OrderIntent, Purpose

logger = logging.getLogger(LOGGER_NAME)

#: The entry lock of a strategy whose closing order was rejected on a shared instrument.
MARGIN_LOCK = "margin_on_close"
#: The orders whose rejection on a shared instrument holds the position.
CLOSING_PURPOSES = frozenset({Purpose.STOP, *EXIT_PURPOSES})
#: What `request_exit` and `protect` answer for a call that waits for an agreeing read.
HELD_FOR_READ = "held_for_read"
#: What the alert of an automatic exit the gate keeps says instead of "exited".
HELD_EXIT_WORDS = (
    "its exit waits until OpenAlgo's books agree with the strategies of the instrument, "
    "since the broker rejected a closing order of the position"
)
#: The square-off the session asks for a position the kernel holds (an exit reason the
#: binding's `request_exit` takes besides its RISK_EXIT_REASONS).
SQUARE_OFF = "square_off"


@dataclass(frozen=True)
class MarginPhrase:
    """Words a broker's rejection reason holds when it refused an order for margin,
    matched without regard to case; where they come from and whether they were checked
    against the source."""

    text: str
    source: str
    verified: bool


#: The words that make the alert say "refused for margin" (module docstring).
MARGIN_PHRASES = (
    MarginPhrase(
        "insufficient funds",
        "OpenAlgo's sandbox fund check (sandbox/fund_manager.py: 'Insufficient funds. "
        "Required: ...')",
        True,
    ),
    MarginPhrase(
        "margin exceeds",
        "a broker's RMS rejection as 18-multi-strategy F118 quotes it; to be checked per "
        "broker in the 2.4.0 sandbox session (open decision 11)",
        False,
    ),
    MarginPhrase(
        "insufficient margin",
        "a broker's RMS rejection; to be checked per broker in the 2.4.0 sandbox session "
        "(open decision 11)",
        False,
    ),
)


def names_margin(reason: str) -> bool:
    """True when a broker's rejection `reason` holds one of `MARGIN_PHRASES`."""
    text = reason.casefold()
    return any(phrase.text in text for phrase in MARGIN_PHRASES)


@dataclass
class Hold:
    """A position held after a closing order's rejection: since when (`since_ns`, UTC
    nanoseconds), whether a read begun after it agreed, whether a call reached the kernel
    after that (its hold then ended), the broker's words named margin (`margin`), and the
    calls the gate kept waiting: the exit's reason and limit (rupees, or None for a market
    exit), and protect."""

    since_ns: int
    margin: bool
    agreed: bool = False
    lifted: bool = False
    exit: tuple[str, float | None] | None = None
    protect: bool = False


class MarginHolds:
    """The held positions of `session`, a `_session_loop.Session` (module docstring)."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._holds: dict[str, Hold] = {}

    def takes(self, trader: Trader, intent: OrderIntent) -> bool:
        """True when a rejection of `trader`'s order placed as `intent` holds its position:
        a closing order on an instrument other strategies of the session trade too."""
        return intent.purpose in CLOSING_PURPOSES and self._session.groups.shared(trader)

    def reject(self, trader: Trader, kernel_order: int, intent: OrderIntent, reason: str) -> bool:
        """The broker rejected `trader`'s closing order `kernel_order` (placed as `intent`)
        for `reason`: the kernel takes it as a margin rejection, then the session acts
        (module docstring). False when the kernel had ended the order already (the race
        of a word that came twice): nothing is done."""
        try:
            call(trader.engine.apply_margin_reject, kernel_order)
        except VenueRefusalError as error:
            logger.debug("%s: o%d not held (%s)", trader.name, kernel_order, error.kind)
            return False
        session, name = self._session, trader.name
        first = name not in self._holds
        self._holds[name] = Hold(session.now(), names_margin(reason))
        session.event("margin_reject", strategy=name, kernel_order=kernel_order, reason=reason)
        if trader.lock(MARGIN_LOCK):
            session.event("lock", strategy=name, reason=MARGIN_LOCK)
        session.reconcile.read_before_exits([name])
        if first:
            session.alert(AlertKind.ERROR, self._words(trader, kernel_order, intent, reason))
        return True

    def holds(self, trader: Trader) -> bool:
        """True from a rejection of `trader`'s closing order until its position is flat:
        no exit retry and no bare-leg protect starts for it."""
        return trader.name in self._holds

    def unprotected_reason(self, trader: Trader) -> str | None:
        """Why a held leg is unprotected (`REFUSED_FOR_MARGIN`, `CLOSE_REJECTED`), None
        for a leg not held."""
        hold = self._holds.get(trader.name)
        if hold is None:
            return None
        return REFUSED_FOR_MARGIN if hold.margin else CLOSE_REJECTED

    def request_exit(
        self, trader: Trader, reason: str, limit_rupees: float | None = None
    ) -> str | None:
        """Ask the kernel to exit `trader`'s position for `reason` (at a LIMIT of
        `limit_rupees`, or at market for None), through the gate (module docstring): what
        the kernel did, `HELD_FOR_READ` while the position waits for an agreeing read, or
        None once the kernel session finished."""
        hold = self._holds.get(trader.name)
        if hold is not None and not hold.agreed:
            if hold.exit is None:
                hold.exit = (reason, limit_rupees)
                self._session.event("held_for_read", strategy=trader.name, call=reason)
            return HELD_FOR_READ
        if trader.finished:
            return None
        if hold is not None:
            hold.lifted = True
        if limit_rupees is None:
            return trader.request_exit(reason)
        exit_ = trader.engine.request_exit
        return str(call(exit_, INSTRUMENT, trader.product, reason, float(limit_rupees)))

    def protect(self, trader: Trader) -> str | None:
        """Place `trader`'s stop again (R7) through the gate (module docstring): what the
        kernel did, `HELD_FOR_READ` while the position waits for an agreeing read, or None
        once the kernel session finished."""
        if self.defers_protect(trader):
            return HELD_FOR_READ
        return kernel_protect(trader)

    def defers_protect(self, trader: Trader) -> bool:
        """The gate of a caller that places `trader`'s stop again itself (protect by
        state): True while the position waits for an agreeing read, the protect then kept
        to be made once one agrees; False when the caller may place it now."""
        hold = self._holds.get(trader.name)
        if hold is not None and not hold.agreed:
            if not hold.protect:
                hold.protect = True
                self._session.event("held_for_read", strategy=trader.name, call="protect")
            return True
        if hold is not None and not trader.finished:
            hold.lifted = True
        return False

    def agreed(self, traders: list[Trader], read_started_ns: int) -> None:
        """A comparison of the group of `traders` agreed, its read begun at
        `read_started_ns`: each held before that read is let go, and the calls the gate
        kept waiting are made, the exit first."""
        for trader in traders:
            hold = self._holds.get(trader.name)
            if hold is None or hold.agreed or hold.since_ns >= read_started_ns:
                continue
            hold.agreed = True
            self._session.event("margin_hold_agreed", strategy=trader.name)
            if hold.exit is not None:
                reason, limit = hold.exit
                if limit is not None:
                    limit = self._session.stops.prices.exit_limit(trader) or limit
                what = self.request_exit(trader, reason, limit)
                self._session.event("exit_after_read", strategy=trader.name, outcome=what)
            if hold.protect:
                what = self.protect(trader)
                self._session.event("protect_after_read", strategy=trader.name, outcome=what)

    def duties(self, now_ns: int) -> None:
        """Forget each hold whose position is flat; ask for the square-off of a position
        the kernel still holds once its strategy's square-off has come (through the
        gate)."""
        for trader in self._session.traders:
            hold = self._holds.get(trader.name)
            if hold is None:
                continue
            if trader.finished or trader.open_units() == 0:
                del self._holds[trader.name]
                continue
            if not hold.lifted and now_ns >= trader.square_off_ns and hold.exit is None:
                what = self.request_exit(trader, SQUARE_OFF)
                self._session.event("square_off_asked", strategy=trader.name, outcome=what)

    def _words(self, trader: Trader, kernel_order: int, intent: OrderIntent, reason: str) -> str:
        """The ERROR alert of a hold (module docstring)."""
        stops = self._session.stops
        covered = stops.covered(trader)
        stop = f"its stop still works for {covered} units" if covered else "it has no stop working"
        refused = "refused for margin" if names_margin(reason) else "rejected"
        return (
            f"{trader.name}: the {intent.purpose.value} order {trader.order_name(kernel_order)} "
            f"of {trader.symbol} {trader.exchange} was {refused} by the broker ({reason}); "
            f"{stop}, and nothing is sent again for it until OpenAlgo's books agree with the "
            f"strategies; {trader.name}'s entries are locked. The broker may not hold these "
            "units: end the session, check the position book and close by hand"
        )


__all__ = [
    "CLOSING_PURPOSES",
    "HELD_EXIT_WORDS",
    "HELD_FOR_READ",
    "MARGIN_LOCK",
    "MARGIN_PHRASES",
    "SQUARE_OFF",
    "Hold",
    "MarginHolds",
    "MarginPhrase",
    "names_margin",
]
