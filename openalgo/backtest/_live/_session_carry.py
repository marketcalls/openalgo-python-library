"""The CNC and NRML legs a live session carries past its close (20-positional-live 2,
"Day 1, the close", 4.1 and 7.3; the lead's rulings N3b-4 and N3b-5).

- **The close's GTT book read** (`confirmed`): once the session's exchanges have closed and
  every MIS leg is done, the order worker reads the GTT book once more with every GTT
  leg's units (`ReadGtts` with `at_close`, answered as `GttBookSeen`). Each carried leg's
  stop GTT must rest in that read with the trigger and the units the session holds it at;
  one that does not is an ERROR alert naming it. The kernel's session has finished at its
  close and places nothing more, so `protect` is asked and answers nothing: the alert says
  to place the stop by hand or start the session again before the open (a GTT that ended
  during the day was placed again by the kernel as the follower found it, every 10 s).
- **The reference for the next day** (`references`): each GTT leg's units at the broker,
  by product (`_broker_units`), read just before the close's GTT book read, with the
  read's time and whether the session's own fills on the leg had settled then; without
  that read (a Ctrl+C), the last reconciliation read's, settled when no order of the leg
  works and no own fill on it came within `_own_fills.POSITION_BOOK_LAG_S` before it.
  Unit N5 takes a settled one as the reference of a stop that fires while no session
  runs.
- **The day-close record** (`write`): per strategy that carries a position or a resting
  GTT order, once its kernel session has stopped, the kernel's `carried_state()` with the
  reference under `REFERENCES_KEY` (the kernel's restore takes the record whole), the
  strategy's `self.state` and its version id, on the session's trading day, in the
  ledger of the session's mode (journal L). A strategy that ends flat writes none, but
  one with a record of the same day (an earlier start's Ctrl+C that kept its leg) has it
  replaced by its flat state, so the latest record never names a closed position.

Times are UTC nanoseconds, quantities signed whole units, prices rupees. Used by the loop
only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .._engine import call
from .._layout import NANOS_PER_SECOND
from ..errors import JournalError
from ._broker_units import broker_units
from ._gtt_types import GttBookSeen, ReadGtts
from ._own_fills import POSITION_BOOK_LAG_S
from ._session_trader import Trader
from ._stops_gtt import RESTING, overnight, stop_gtts
from ._stops_kernel import protect
from ._types import LOGGER_NAME, AlertKind
from .inbox import Books

logger = logging.getLogger(LOGGER_NAME)

#: The key of the day-close record that holds each carried leg's reference (module
#: docstring), beside the kernel's carried state.
REFERENCES_KEY = "gtt_references"
#: Half a paisa: two trigger prices closer than it are the same price.
_PAISA = 0.005
_LAG_NS = round(POSITION_BOOK_LAG_S * NANOS_PER_SECOND)


@dataclass(frozen=True)
class LegReference:
    """A GTT leg's units at the broker (signed), when they were read (UTC ns) and whether
    the session's own fills on the leg had settled then (module docstring)."""

    units: int
    read_ns: int
    settled: bool

    def record(self, trader: Trader) -> dict[str, Any]:
        """The reference as the day-close record keeps it."""
        return {
            "symbol": trader.symbol,
            "exchange": trader.exchange,
            "product": trader.product,
            "units": self.units,
            "read_ns": self.read_ns,
            "settled": self.settled,
        }


class CarriedLegs:
    """The carried legs of `session`, a `_session_loop.Session` (module docstring)."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._asked = False
        self._seen: GttBookSeen | None = None
        self.references: dict[str, LegReference] = {}

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
                settled = leg in message.settled
                self.references[trader.name] = LegReference(units, message.read_ns, settled)
            why = self._unconfirmed(trader, message)
            if why:
                self._alert(trader, why)

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
                settled = self._settled(trader, message.started_ns)
                self.references[trader.name] = LegReference(units, message.started_ns, settled)

    def write(self, trader: Trader) -> None:
        """Write `trader`'s day-close record when it carries a position or a resting GTT
        order (module docstring); its kernel session has stopped. A journal that cannot
        take it is a warning of the result."""
        session = self._session
        if not overnight(trader):
            return
        carried = call(trader.engine.carried_state)
        flat = not carried.get("positions") and not carried.get("orders")
        if flat and not self._recorded_today(trader):
            return
        reference = self.references.get(trader.name)
        carried[REFERENCES_KEY] = [] if reference is None else [reference.record(trader)]
        version = trader.part.spec.version.version_id
        state = dict(trader.part.strategy.state)
        try:
            session.parts.journal.put_day_close(
                trader.name, carried, state, version, day=session.parts.spec.day
            )
        except JournalError as error:
            session.warn(f"{trader.name}: the day-close record was not written ({error})")
            return
        session.event("day_close", strategy=trader.name, references=len(carried[REFERENCES_KEY]))

    def _recorded_today(self, trader: Trader) -> bool:
        """Whether the journal holds a day-close record of `trader` of the session's day.
        A journal that cannot be read is a warning of the result (and no record)."""
        session = self._session
        try:
            record = session.parts.journal.day_close(trader.name)
        except JournalError as error:
            session.warn(f"{trader.name}: the day-close record could not be read ({error})")
            return False
        return record is not None and record.day == session.parts.spec.day

    def _unconfirmed(self, trader: Trader, message: GttBookSeen) -> str:
        """Why the read does not confirm `trader`'s stop GTT, or "" when it does."""
        if message.resting is None:
            return "the GTT book could not be read after the close"
        units = round(abs(trader.open_units()))
        for gtt in stop_gtts(trader):
            known = trader.gtts.gtt(int(gtt["gtt_id"])) if trader.gtts is not None else None
            if gtt["state"] not in RESTING or known is None:
                continue
            row = message.resting.get(known.trigger_id)
            if row is None:
                return f"its stop GTT {known.trigger_id} does not rest in the GTT book"
            triggers, quantity = row
            if quantity == units == round(gtt["quantity_units"]) and any(
                abs(trigger - gtt["trigger_price"]) < _PAISA for trigger in triggers
            ):
                return ""
            return (
                f"its stop GTT {known.trigger_id} rests for {quantity} units at "
                f"{', '.join(f'{t:.2f}' for t in triggers)}, not for {units} at "
                f"{gtt['trigger_price']:.2f}"
            )
        return "no stop GTT of it rests at the broker"

    def _alert(self, trader: Trader, why: str) -> None:
        outcome = protect(trader)
        self._session.event("protect", strategy=trader.name, outcome=outcome)
        self._session.alert(
            AlertKind.ERROR,
            f"{trader.name}: {trader.symbol} {trader.exchange} {trader.open_units():g} units "
            f"carried overnight, but {why}: place its stop GTT in OpenAlgo by hand, or start "
            "the session again before the open to have it placed",
        )

    def _settled(self, trader: Trader, read_ns: int) -> bool:
        """Whether `trader`'s own fills had settled for a read begun at `read_ns`: no order
        of its leg works, and none of its fills came within the lag before the read."""
        if trader.working():
            return False
        fills = [f.at_ns for f in self._session.fills if f.strategy == trader.name]
        return not fills or max(fills) <= read_ns - _LAG_NS


__all__ = ["REFERENCES_KEY", "CarriedLegs", "LegReference"]
