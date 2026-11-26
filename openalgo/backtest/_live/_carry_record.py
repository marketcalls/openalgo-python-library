"""The day-close record a live session writes for each CNC and NRML strategy it carries
past its close (20-positional-live 4.1; the lead's rulings N3b-4 and N5a-1, review round 1
of unit N5a), split from `_session_carry` so each module keeps one job.

Per strategy that carries a position or a resting GTT order, once its kernel session has
stopped: the kernel's `carried_state()` with the reference under `REFERENCES_KEY` (the
kernel's restore takes the record whole), the strategy's `self.state` and its version id,
on the session's trading day, in the ledger of the session's mode (journal L), with each
stop GTT the close did not confirm under `UNCONFIRMED_KEY` (its kernel id, trigger ID and
why), each it confirmed resting under `CONFIRMED_KEY` (the next start trusts the reference
of those only, never one it infers from a GTT not named), and the journal's GTT each
carried kernel GTT is bound to under `BINDINGS_KEY`, by which the next day's session takes
them back (`_carry_restore`). A strategy that ends flat writes none, but one with a record
of the same day (an earlier start's Ctrl+C that kept its leg), or one that continued a
record of an earlier day (`_carry_start`), whose record of an earlier day a later save
outdates (`_carry_start`: `CarryAtStart.superseded`, such as one found settled by hand)
or whose carried contract the exchange settled (`_carry_contracts`), has its flat state
written, so the latest record never names a closed position.

Each carried leg's reference (`LegReference`) is the one `_session_carry.CarriedLegs`
took at the close. Times are UTC nanoseconds, quantities signed whole units. Used by the
loop at the close only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._engine import call
from ..errors import JournalError
from ._carry_start import BINDINGS_KEY, CONFIRMED_KEY, REFERENCES_KEY, UNCONFIRMED_KEY
from ._session_trader import Trader
from ._stops_gtt import overnight, resting_gtts, stop_gtts


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


def write_day_close(session: Any, trader: Trader, reference: LegReference | None) -> None:
    """Write `trader`'s day-close record when it carries a position or a resting GTT order
    (module docstring), with its leg's `reference` (None for none); `session` is the
    `_session_loop.Session`, and the trader's kernel session has stopped. A journal that
    cannot take it is a warning of the result."""
    if not overnight(trader):
        return
    carried = call(trader.engine.carried_state)
    flat = not carried.get("positions") and not carried.get("orders")
    carry = session.parts.carry
    continued = carry is not None and (
        carry.of(trader.name) is not None
        or trader.name in carry.expired
        or trader.name in carry.superseded
    )
    if flat and not continued and not _recorded_today(session, trader):
        return
    carried[REFERENCES_KEY] = [] if reference is None else [reference.record(trader)]
    carried[UNCONFIRMED_KEY] = _unconfirmed_gtts(trader)
    carried[CONFIRMED_KEY] = _confirmed_gtts(trader)
    version = trader.part.spec.version.version_id
    state = dict(trader.part.strategy.state)
    try:
        carried[BINDINGS_KEY] = _bindings(session, trader, carried.get("gtts", []))
        session.parts.journal.put_day_close(
            trader.name, carried, state, version, day=session.parts.spec.day
        )
    except JournalError as error:
        session.warn(f"{trader.name}: the day-close record was not written ({error})")
        return
    session.event("day_close", strategy=trader.name, references=len(carried[REFERENCES_KEY]))


def _bindings(session: Any, trader: Trader, gtts: list[dict[str, Any]]) -> list[dict[str, int]]:
    """The journal's GTT each carried kernel GTT of `gtts` is bound to, as the record
    keeps it under `BINDINGS_KEY`: the GTT map's, else (a place under way) the latest
    journal GTT of the day the strategy recorded with that kernel id."""
    day = session.parts.spec.day
    rows = session.parts.journal.day_gtts(trader.name, day=day)
    bindings = []
    for gtt in gtts:
        gtt_id = int(gtt["gtt"])
        known = None if trader.gtts is None else trader.gtts.gtt(gtt_id)
        seqs = [row.seq for row in rows if row.gtt_id == gtt_id and not row.state.terminal]
        seq = None if known is None else known.seq
        seq = max(seqs, default=None) if seq is None else seq
        if seq is not None:
            bindings.append({"gtt": gtt_id, "seq": seq})
    return bindings


def _recorded_today(session: Any, trader: Trader) -> bool:
    """Whether the journal holds a day-close record of `trader` of the session's day.
    A journal that cannot be read is a warning of the result (and no record)."""
    try:
        record = session.parts.journal.day_close(trader.name)
    except JournalError as error:
        session.warn(f"{trader.name}: the day-close record could not be read ({error})")
        return False
    return record is not None and record.day == session.parts.spec.day


def _confirmed_gtts(trader: Trader) -> list[int]:
    """The kernel ids of `trader`'s stop GTTs the close confirmed resting, as the
    day-close record keeps them (module docstring): resting confirmed, and none a
    restart could not tell where it stands."""
    gtts = trader.gtts
    if gtts is None:
        return []
    named = []
    for gtt in resting_gtts(trader):
        known = gtts.gtt(int(gtt["gtt_id"]))
        if known is not None and not known.unsettled:
            named.append(int(gtt["gtt_id"]))
    return named


def _unconfirmed_gtts(trader: Trader) -> list[dict[str, Any]]:
    """Each stop GTT of `trader` the close did not confirm, as the day-close record keeps
    it (module docstring)."""
    gtts = trader.gtts
    if gtts is None:
        return []
    named = []
    for gtt in stop_gtts(trader):
        known = gtts.gtt(int(gtt["gtt_id"]))
        if known is not None and known.unconfirmed:
            gtt_id, trigger_id = int(gtt["gtt_id"]), known.trigger_id
            named.append({"gtt": gtt_id, "trigger_id": trigger_id, "why": known.unconfirmed})
    return named


__all__ = ["LegReference", "write_day_close"]
