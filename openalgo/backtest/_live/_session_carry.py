"""The CNC and NRML legs a live session carries past its close (20-positional-live 2,
"Day 1, the close", 4.1 and 7.3; the lead's rulings N3b-4 and N3b-5).

- **The close's GTT book read** (`confirmed`): once the latest exchange of a carried leg
  has closed (whether the MIS legs are done or not), the order worker reads the GTT book
  once more with every GTT leg's units (`ReadGtts` with `at_close`, answered as
  `GttBookSeen`). Each carried leg's stop GTT must rest in that read with the trigger and
  the units the session holds it at; one that does not is an ERROR alert naming it. The
  kernel's session has finished at its close and places nothing more, so `protect` is
  asked and answers nothing: the alert says to place the stop by hand before the next open
  (nothing of this version places it after the close; a GTT that ended during the day was
  placed again by the kernel as the follower found it, every 10 s). A finding of the GTT
  follower about a carried leg's stop GTT that comes once the kernel's session has closed
  (a GTT cancelled by hand in the last seconds, found gone) is the same (`after_close`).
  A stop GTT that may have fired (gone from the read, which a book of active GTTs only
  shows of a fired GTT too, or a finding that it fired) is checked first: the alert says
  to check the order book and the position before placing a stop GTT by hand, which
  would sell again beside a child that sold the units (review finding 3 of round 3).
  The kernel still holds such a GTT as resting, so the GTT map notes why it is not
  confirmed (`KernelGtt.unconfirmed`): the stops supervisor, the lines, the result and the
  day-close record never report it as protecting the leg. A finding after the close about
  any other GTT (a strategy's GTT order, a stop GTT of a flat leg) touches no leg: it is
  named in a REJECT alert, an ERROR one when it fired or its terms are not the session's.
  Each finding is named in plain words (`FINDING_WORDS`).
- **The reference for the next day** (`references`): each GTT leg's units at the broker,
  by product (`_broker_units`), read just before the close's GTT book read, with the
  read's time and whether the session's own fills on the leg had settled then; without
  that read (a Ctrl+C), the last reconciliation read's, settled when no order of the leg
  works and no own fill on it came within `_own_fills.POSITION_BOOK_LAG_S` before it. A
  read whose units are not the leg's (one-sided for CNC: the user's own units beyond it
  count for nothing, `_broker_units.strategy_units`) is never settled: the leg's stop GTT
  may have fired, and a fall measured from such a reference would take a GTT that fired
  for one that ended outside the session. The next start takes a settled one as the
  reference of a stop that fires while no session runs.
- **A finding the close leaves open** (`unresolved_at_close`): at every close, a carried
  leg's stop GTT the kernel still holds resting whose latest finding of the GTT follower
  is not RESTING (it went from the book, fired, ended with the leg's units not held, or
  its terms or its fate cannot be told) is not confirmed: an ERROR alert names it once,
  the books to check first, and the lines, the result and the record report it so.
- **The day-close record** (`write`): `_carry_record`.

Times are UTC nanoseconds, quantities signed whole units, prices rupees. Used by the loop
only.
"""

from __future__ import annotations

import logging
from typing import Any

from .._layout import NANOS_PER_SECOND
from ._broker_units import broker_units, strategy_units
from ._carry_record import LegReference, write_day_close
from ._carry_start import REFERENCES_KEY, UNCONFIRMED_KEY
from ._gtt_types import GttBookSeen, GttEvent, GttOutcome, ReadGtts
from ._own_fills import POSITION_BOOK_LAG_S
from ._session_trader import Trader
from ._stops_gtt import PLACE_GTT_BY_HAND, RESTING, overnight, resting_gtts, stop_gtts
from ._stops_kernel import protect
from ._types import LOGGER_NAME, AlertKind, Purpose
from ._venue_gtt_known import KernelGtt
from .inbox import Books

logger = logging.getLogger(LOGGER_NAME)

#: Half a paisa: two trigger prices closer than it are the same price.
_PAISA = 0.005
#: The follower's findings, after the close, of a GTT that holds no carried leg's stop
#: that the operator must check by hand: it fired, or its terms are not the session's.
_CHECK_BY_HAND = frozenset(
    {GttEvent.TRIGGERED, GttEvent.CHILD_MISSED, GttEvent.MISMATCH, GttEvent.OTHER_TERMS}
)
#: The follower's findings after the close in an alert's words (RESTING alerts nothing).
FINDING_WORDS = {
    GttEvent.IN_FLIGHT: "is no longer resting",
    GttEvent.TRIGGERED: "fired",
    GttEvent.CHILD_MISSED: "fired with no child order found",
    GttEvent.ENDED_OUTSIDE: "ended at the broker",
    GttEvent.EXPIRED: "expired at the broker",
    GttEvent.REJECTED: "was rejected by the broker",
    GttEvent.MISMATCH: "fired with a child order that cannot be told from others",
    GttEvent.OTHER_TERMS: "rests with terms other than the session's",
}
#: The findings after the close of a carried leg's stop GTT that may have fired (module
#: docstring): in flight, fired, its child missed or not told apart.
_MAY_HAVE_FIRED = frozenset(
    {GttEvent.IN_FLIGHT, GttEvent.TRIGGERED, GttEvent.CHILD_MISSED, GttEvent.MISMATCH}
)
_LAG_NS = round(POSITION_BOOK_LAG_S * NANOS_PER_SECOND)


class CarriedLegs:
    """The carried legs of `session`, a `_session_loop.Session` (module docstring)."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._asked = False
        self._seen: GttBookSeen | None = None
        self.references: dict[str, LegReference] = {}
        # The strategies whose carried leg an ERROR alert named as not confirmed.
        self._alerted: set[str] = set()
        # The (strategy, kernel GTT, finding) of each other GTT named after the close.
        self._named: set[tuple[str, int, GttEvent]] = set()

    def carried(self) -> list[Trader]:
        """The CNC and NRML strategies holding a position."""
        return [t for t in self._session.traders if overnight(t) and t.open_units() != 0]

    def confirmed(self) -> bool:
        """Whether the close's GTT book read has been judged (module docstring); asks for
        it at the first call. True at once with no leg carried."""
        if not self.carried():
            return True
        if not self._asked:
            self._asked = True
            self._session.hand_over([ReadGtts(at_close=True)])
        return self._seen is not None

    def seen(self, message: GttBookSeen) -> None:
        """Judge the close's GTT book read (module docstring)."""
        if not self._asked:
            return
        self._seen = message
        for trader in self.carried():
            leg = (trader.symbol, trader.exchange, trader.product)
            units = message.units.get(leg)
            if units is not None:
                settled = leg in message.settled and _holds_leg(trader, units)
                self.references[trader.name] = LegReference(units, message.read_ns, settled)
            gtt_id, why, fired = self._unconfirmed(trader, message)
            if why:
                self._not_confirmed(trader, gtt_id, why, fired=fired)

    def after_close(self, trader: Trader, outcome: GttOutcome) -> None:
        """A finding of the GTT follower about `trader`'s GTT once its kernel session has
        closed (module docstring): anything but RESTING leaves the stop GTT of a leg with
        a position not confirmed; any other GTT is named in an alert of its own. A finding
        is about the leg's GTT only when it names both its kernel id and its journal GTT
        (`GttMap.gtt_found`)."""
        if outcome.event is GttEvent.RESTING:
            return
        known = None if trader.gtts is None else trader.gtts.gtt_found(outcome)
        if known is None or known.purpose is not Purpose.STOP or trader.open_units() == 0:
            self._other_after_close(trader, known, outcome)
            return
        why = f"it {FINDING_WORDS[outcome.event]} after the close"
        if outcome.message:
            why += f": {outcome.message}"
        fired = outcome.event in _MAY_HAVE_FIRED
        self._not_confirmed(trader, outcome.gtt_id, why, fired=fired)

    def _other_after_close(
        self, trader: Trader, known: KernelGtt | None, outcome: GttOutcome
    ) -> None:
        """A finding after the close about a GTT that holds no carried leg's stop (a GTT
        order, or a stop GTT of a flat leg), named once per finding: a REJECT alert, or an
        ERROR alert when it fired or its terms are not the session's, since the engine
        follows it no more (a fired GTT's child may open a position it does not hold)."""
        event = outcome.event
        key = (trader.name, outcome.gtt_id, event)
        if key in self._named:
            return
        self._named.add(key)
        kind = "GTT order" if known is not None and known.purpose is not Purpose.STOP else "GTT"
        text = (
            f"{trader.name}: {kind} {outcome.trigger_id} of {trader.symbol} {trader.exchange} "
            f"{FINDING_WORDS[event]} after the session's close: the session follows it no more"
        )
        if event in _CHECK_BY_HAND:
            self._session.alert(AlertKind.ERROR, f"{text}; check the GTT and order books")
        else:
            self._session.alert(AlertKind.REJECT, text)

    def alerted(self, name: str) -> bool:
        """Whether an ERROR alert named strategy `name`'s carried leg as not confirmed."""
        return name in self._alerted

    def books(self, message: Books) -> None:
        """Keep each GTT leg's units of a reconciliation read as its reference, unless the
        close's GTT book read gave one (module docstring)."""
        if self._seen is not None:
            return
        for trader in self._session.traders:
            if not overnight(trader):
                continue
            units = broker_units(
                message.positionbook,
                message.holdings,
                trader.symbol,
                trader.exchange,
                trader.product,
            )
            if units is not None:
                settled = self._settled(trader, message.started_ns) and _holds_leg(trader, units)
                self.references[trader.name] = LegReference(units, message.started_ns, settled)

    def write(self, trader: Trader) -> None:
        """Write `trader`'s day-close record (`_carry_record.write_day_close`) with the
        reference of its leg; its kernel session has stopped."""
        write_day_close(self._session, trader, self.references.get(trader.name))

    def _unconfirmed(self, trader: Trader, message: GttBookSeen) -> tuple[int | None, str, bool]:
        """(the kernel's stop GTT the read does not confirm, None for none; why, "" when
        the read confirms the leg's stop GTT; whether it may have fired: gone from the
        read)."""
        if message.resting is None:
            gtts = resting_gtts(trader)
            gtt_id = int(gtts[-1]["gtt_id"]) if gtts else None
            return gtt_id, "the GTT book could not be read after the close", False
        units = round(abs(trader.open_units()))
        for gtt in stop_gtts(trader):
            known = trader.gtts.gtt(int(gtt["gtt_id"])) if trader.gtts is not None else None
            if gtt["state"] not in RESTING or known is None:
                continue
            gtt_id = int(gtt["gtt_id"])
            row = message.resting.get(known.trigger_id)
            if row is None:
                return gtt_id, "it does not rest in the GTT book", True
            triggers, quantity = row
            if quantity == units == round(gtt["quantity_units"]) and any(
                abs(trigger - gtt["trigger_price"]) < _PAISA for trigger in triggers
            ):
                return None, "", False
            why = (
                f"it rests for {quantity} units at {', '.join(f'{t:.2f}' for t in triggers)}, "
                f"not for {units} at {gtt['trigger_price']:.2f}"
            )
            return gtt_id, why, False
        return None, "no stop GTT of it rests at the broker", False

    def unresolved_at_close(self) -> None:
        """Name each carried leg's stop GTT whose latest finding the close leaves open
        (module docstring), before the close names what it leaves. The kernel's session
        may still run (a Ctrl+C): nothing is asked of it."""
        for trader in self.carried():
            for gtt in resting_gtts(trader):
                known = trader.gtts.gtt(int(gtt["gtt_id"])) if trader.gtts is not None else None
                if known is None or known.finding in (None, GttEvent.RESTING):
                    continue
                why = (
                    f"the GTT follower found that it {FINDING_WORDS[known.finding]}, and the "
                    "session ended before that was resolved"
                )
                self._not_confirmed(trader, int(gtt["gtt_id"]), why, fired=True, closed=False)

    def _not_confirmed(
        self, trader: Trader, gtt_id: int | None, why: str, *, fired: bool, closed: bool = True
    ) -> None:
        """Note on the GTT map that the close did not confirm `trader`'s stop GTT `gtt_id`
        (when known) and why, and alert it once per leg, the books checked first when it
        may have `fired` (module docstring); `closed` when the kernel's session has
        closed: it is then asked to `protect` the leg, and answers nothing."""
        known = None if trader.gtts is None or gtt_id is None else trader.gtts.gtt(gtt_id)
        if known is not None and not known.unconfirmed:
            known.unconfirmed = why
        if trader.name in self._alerted:
            return
        self._alerted.add(trader.name)
        if closed:
            outcome = protect(trader)
            self._session.event("protect", strategy=trader.name, outcome=outcome)
        advice = f": {PLACE_GTT_BY_HAND}"
        if fired:
            advice = (
                "; it may have fired: check the order book and the position first, then "
                f"{PLACE_GTT_BY_HAND}"
            )
        held = (
            f"{trader.name}: {trader.symbol} {trader.exchange} {trader.open_units():g} units "
            "carried overnight"
        )
        if known is None:
            text = f"{held}, but {why}{advice}"
        else:
            text = f"{held}, and its stop GTT {known.trigger_id} is not confirmed ({why}){advice}"
        self._session.alert(AlertKind.ERROR, text)

    def _settled(self, trader: Trader, read_ns: int) -> bool:
        """Whether `trader`'s own fills had settled for a read begun at `read_ns`: no order
        of its leg works, and none of its fills came within the lag before the read."""
        if trader.working():
            return False
        fills = [f.at_ns for f in self._session.fills if f.strategy == trader.name]
        return not fills or max(fills) <= read_ns - _LAG_NS


def _holds_leg(trader: Trader, broker: int) -> bool:
    """Whether the broker's `broker` units are the leg's units (module docstring)."""
    units = round(trader.open_units())
    return strategy_units(trader.product, units, broker) == units


__all__ = ["FINDING_WORDS", "REFERENCES_KEY", "UNCONFIRMED_KEY", "CarriedLegs"]
