"""What a live session continues from earlier trading days (20-positional-live 4.2 and 4.3;
the lead's rulings N5a-1 and N5a-3), read from the journal at the start, before the
preflight, so that every check and refusal of a carry comes before anything is sent.

- **The carry** (`read_carry`): each of the session's strategies whose latest day-close
  record in the ledger of the mode OpenAlgo is in is of an earlier day and carries a
  position, a GTT order or a GTT (a record of a strategy that ended flat carries none).
  With it, the latest day before today on which a session of the strategy started
  (`Journal.last_ran`): a session that ran after the record's day may have changed what
  it carries, so the record's reference of the broker's units then counts no more
  (`Carried.current`, the lead's ruling N5a-1 (3)). The other ledger's latest records are
  read too: their positions are only named in the start banner, never restored
  (`CarryAtStart.other`, 4.2 step 1).
- **The capital** (`capital_refusals`): the engine continues a carried position only with
  the capital it was entered with (`restore_carried_state` refuses another), so a start
  passing another `capital=` is refused before the preflight, naming the figure to pass.
- **Expired contracts** (`expired_instruments`): a carried position on a contract whose
  last trading day is before today was settled by the exchange (section 9). The preflight
  does not look it up in today's master contract, which no longer lists it; the restore
  records its exit at the contract's last close.

Days are IST trading days, amounts rupees, quantities signed whole units.
"""

from __future__ import annotations

import datetime as dt
import pathlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .._futures_contracts import last_day
from ..reference import Instrument
from ._journal_carry import DayClose
from ._preflight_carry import CarriedPosition, carried_positions, held_rupees
from ._preflight_plan import InstrumentFacts, refused
from ._types import Mode, Now
from .journal import Journal

#: The keys of a day-close record the session adds beside the kernel's carried state
#: (`_session_carry`): each carried leg's reference of the broker's units read at the close,
#: each stop GTT the close's read did not confirm, and the journal's GTT (its `seq`) each
#: carried kernel GTT is bound to.
REFERENCES_KEY = "gtt_references"
UNCONFIRMED_KEY = "gtts_not_confirmed"
BINDINGS_KEY = "gtt_bindings"
#: Half a paisa: two capitals closer than it are the same capital.
_PAISA = 0.005


@dataclass(frozen=True)
class Carried:
    """One strategy's carry (module docstring): its latest day-close `record` of an earlier
    day in the session's ledger, and `last_ran`, the latest day before today on which a
    session of it started (None for none)."""

    record: DayClose
    last_ran: dt.date | None

    @property
    def current(self) -> bool:
        """Whether no session of the strategy ran after the record's day: the record then
        holds what the strategy carries, and its reference of the broker's units counts."""
        return self.last_ran is None or self.last_ran <= self.record.day

    @property
    def positions(self) -> tuple[CarriedPosition, ...]:
        """The open positions the record carries. Raises LiveError for a carried state that
        cannot be read."""
        return carried_positions(self.record)


@dataclass(frozen=True)
class CarryAtStart:
    """What a session of trading `day` in `mode` continues (module docstring): `carried`,
    by strategy name in call order, and `other`, the other ledger's latest records of the
    session's strategies that carry a position, by name."""

    day: dt.date
    mode: Mode
    carried: Mapping[str, Carried]
    other: Mapping[str, DayClose]

    @classmethod
    def none(cls, day: dt.date, mode: Mode) -> CarryAtStart:
        """A start that continues nothing."""
        return cls(day, mode, {}, {})

    def of(self, name: str) -> Carried | None:
        """Strategy `name`'s carry, or None."""
        return self.carried.get(name)


def read_carry(journal: Journal, names: Sequence[str], day: dt.date, mode: Mode) -> CarryAtStart:
    """The carry of a session of `day` in `mode` whose strategies are `names` (module
    docstring), read through `journal` (a handle of that mode). Raises JournalError when
    the journal cannot be read, and LiveError for a record whose carried state cannot be
    read."""
    records = journal.day_closes()
    others = journal.other_ledger_day_closes()
    carried = {}
    for name in names:
        record = records.get(name)
        if record is None or record.day >= day or not _carries(record):
            continue
        carried[name] = Carried(record, journal.last_ran(name, day))
    other = {
        name: others[name] for name in names if name in others and carried_positions(others[name])
    }
    return CarryAtStart(day, mode, carried, other)


def capital_refusals(carry: CarryAtStart, capitals: Mapping[str, float]) -> list[str]:
    """A refusal for each carried strategy whose `capitals` entry (rupees, by name) is not
    the capital its record was entered with (module docstring)."""
    refusals = []
    for name, carried in carry.carried.items():
        saved = float(carried.record.carried["account"]["capital_rupees"])
        given = capitals[name]
        if abs(saved - given) >= _PAISA:
            refusals.append(
                f"{name}: what it carries from {carried.record.day} was entered with "
                f"capital={_amount(saved)}, and this start passes capital={_amount(given)}: "
                "the engine continues a carried position only with the capital it was "
                f"entered with, so start it with capital={_amount(saved)}"
            )
    return refusals


def expired_instruments(
    carry: CarryAtStart, instruments: Mapping[str, Sequence[Instrument]]
) -> dict[str, Instrument]:
    """The carried contracts whose last trading day is before the carry's day, by
    strategy name: those of `instruments` (each strategy's, by name) on which its record
    carries a position (module docstring)."""
    expired = {}
    for name, carried in carry.carried.items():
        held = {(p.symbol, p.exchange) for p in carried.positions}
        for instrument in instruments.get(name, ()):
            ends = instrument.expiry is not None and last_day(instrument) < carry.day
            if ends and (instrument.symbol, instrument.exchange) in held:
                expired[name] = instrument
    return expired


def open_carry(
    spec: Any, *, path: pathlib.Path, account: str, broker: str, mode: Mode, now: Now
) -> CarryAtStart:
    """Read the carry of the session `spec` (a `plan.SessionSpec`) from the journal at
    `path` of `account` at `broker`, in the ledger of `mode`, on a handle opened for the
    read and closed after it (module docstring), and refuse a start whose capital is not
    a carried strategy's. Raises PreflightError for such a start, JournalError when the
    journal cannot be read, and LiveError for a record that cannot be read."""
    names = [each.name for each in spec.strategies]
    with Journal.open(path, account=account, broker=broker, mode=mode, now=now) as journal:
        carry = read_carry(journal, names, spec.day, mode)
    refusals = capital_refusals(carry, {each.name: each.capital for each in spec.strategies})
    if refusals:
        raise refused(tuple(refusals))
    return carry


def preflight_terms(carry: CarryAtStart, spec: Any) -> dict[str, Any]:
    """The preflight's terms of a carry (`PreflightPlan.carried` and `expired`): each
    carried position on a contract still trading with the rupees it holds
    (`held_rupees`), and the reference tables' facts of each expired carried contract.
    Raises LiveError for a sold option carried, which this release does not hold
    overnight."""
    instruments = {each.name: each.instruments for each in spec.strategies}
    expired = expired_instruments(carry, instruments)
    held = []
    for name, carried in carry.carried.items():
        by_pair = {(i.symbol, i.exchange): i for i in instruments.get(name, ())}
        for position in carried.positions:
            instrument = by_pair.get((position.symbol, position.exchange))
            if instrument is None or expired.get(name) is instrument:
                continue
            held.append((position, held_rupees(position, instrument)))
    facts = tuple(
        InstrumentFacts(i.symbol, i.exchange, i.lot_size, float(i.tick_size or 0.0), i.freeze_qty)
        for i in expired.values()
    )
    return {"carried": tuple(held), "expired": facts}


@dataclass(frozen=True)
class CarriedReference:
    """A carried GTT's reference from its strategy's day-close record (the lead's ruling
    N5a-1 (3)): the journal's GTT `seq`, its leg (symbol, exchange, product), the leg's
    signed `units` at the broker read at the close at `read_ns` (UTC ns), and whether the
    close's read confirmed the GTT resting (`confirmed`: else the first read of the GTT
    book today judges it)."""

    seq: int
    leg: tuple[str, str, str]
    units: int
    read_ns: int
    confirmed: bool


def gtt_seqs(record: DayClose) -> dict[int, int]:
    """The journal's GTT (`seq`) of each kernel GTT the record carries, by its kernel id at
    the close; one the record binds to none is left out."""
    return {int(row["gtt"]): int(row["seq"]) for row in record.carried.get(BINDINGS_KEY, [])}


def carried_references(journal: Journal, day: dt.date) -> list[CarriedReference]:
    """Each carried GTT's reference (`CarriedReference`), from the latest day-close record,
    of a day before `day`, of every strategy of `journal`'s ledger: only from a record that
    is `Carried.current` (no session of the strategy ran after it) whose reference was
    settled, the lead's ruling N5a-1 (3). Raises JournalError when the journal cannot be
    read."""
    found = []
    for name, record in journal.day_closes().items():
        if record.day >= day or not Carried(record, journal.last_ran(name, day)).current:
            continue
        state = record.carried
        unconfirmed = {int(row["gtt"]) for row in state.get(UNCONFIRMED_KEY, [])}
        legs = {int(gtt["gtt"]): _gtt_leg(state, gtt) for gtt in state.get("gtts", [])}
        for reference in state.get(REFERENCES_KEY, []):
            leg = (reference["symbol"], reference["exchange"], reference["product"])
            if not reference["settled"]:
                continue
            units, read_ns = int(reference["units"]), int(reference["read_ns"])
            for gtt_id, seq in gtt_seqs(record).items():
                if legs.get(gtt_id) == leg:
                    confirmed = gtt_id not in unconfirmed
                    found.append(CarriedReference(seq, leg, units, read_ns, confirmed))
    return found


def _gtt_leg(state: Mapping[str, Any], gtt: Mapping[str, Any]) -> tuple[str, str, str]:
    """The leg (symbol, exchange, product) of a carried state's GTT."""
    terms = gtt["terms"]
    instrument = state["instruments"][int(terms["instrument"])]
    return str(instrument["symbol"]), str(instrument["exchange"]), str(terms["product"])


def _carries(record: DayClose) -> bool:
    """Whether a day-close record carries a position, a GTT order or a GTT."""
    state = record.carried
    positions = any(float(p.get("quantity_units", 0)) for p in state.get("positions", []))
    return positions or bool(state.get("orders")) or bool(state.get("gtts"))


def _amount(rupees: float) -> str:
    """Rupees as `capital=` is passed: whole rupees with no decimals, else to the paisa."""
    return f"{rupees:.0f}" if float(rupees).is_integer() else f"{rupees:.2f}"


__all__ = [
    "BINDINGS_KEY",
    "REFERENCES_KEY",
    "UNCONFIRMED_KEY",
    "CarryAtStart",
    "Carried",
    "CarriedReference",
    "capital_refusals",
    "carried_references",
    "expired_instruments",
    "gtt_seqs",
    "open_carry",
    "preflight_terms",
    "read_carry",
]
