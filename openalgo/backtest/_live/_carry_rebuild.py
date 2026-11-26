"""A strategy's session of an earlier trading day that stopped without its day-close record (a
crash, a power cut, a killed process), rebuilt from its last save and the journal's updates
newer than it, and written as that day's day-close record, marked rebuilt
(20-positional-live 4.2 step 2, review M7; the lead's ruling N5c-2). The kernel's day-close
rules make the record (`Engine.restore_stopped_state`, `stop_session`, `carried_state`), so
the start continues it exactly as a record a close wrote (`_carry_start`).

**The session rebuilt.** A kernel session of the saved day on the instruments the save
names (`_session_saved.KERNEL_INSTRUMENTS_KEY`, the facts that session had) and the
strategy's settings with the save's capital (a start passing another is refused by the
carry's own capital rule, as for a record), with none of that day's bars, takes the save
where it stood.

**The updates newer than the save**, each accounted for: applied, or named in the banner
with why it was not.

- *Orders* the save works and sent: the journal's last word of each (its fills and its end)
  through the order's fill ledger, from the units the save gave the kernel
  (`_fill_ledger.FillLedger`, the one door of fills). An order whose outcome the journal
  does not know (unbound, or waiting in OpenAlgo's Action Center) is named: this start's
  read of the books judges what it did. One never recorded never reached the broker.
- *GTTs*: `_carry_rebuild_gtts`.

**The record.** The kernel's carried state with what a close's record adds: the journal's
GTT each carried GTT is bound to (the save's bindings, and the day's GTTs by kernel id);
the leg's reference of the broker's units from the save's last reconciliation read when it
still holds (settled, no order of the session working when it stopped, the units the
leg's, and nothing of the day's orders filled after the read): with none, a stop GTT gone
from the next start's book is a mismatch, as for any record without one; and each carried
GTT that reference confirms (the journal shows a read of the GTT book found it resting on
the terms it kept since before the reference's read: it had not fired then), the others
named not confirmed (no close read was made: the start's first read of the GTT book judges
them). Under `REBUILT_KEY`: when the session stopped (the latest of its save and the
journal's updates read), when it last saved, how many updates were applied and the words
of those named. The strategy's `self.state` is the last it saved, its version id the one
its session of that day started with. `write` writes it; `Rebuilt.opening` names each order
that opens or adds that still worked when the session stopped, which the start judges
against the books (`_carry_unrecorded`).

**Refused** (`LiveError`, the words naming what and what to do): a save that cannot be read
or names no instruments (one an earlier version wrote), one the kernel refuses (another
instrument than the strategy trades now, a save the kernel cannot read), and an update that
cannot be accounted for.

Days are IST trading days, times UTC nanoseconds, quantities whole units, prices rupees.
Run by the main thread at the start, before the preflight.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .._calendar_convert import to_day
from .._engine import call
from .._market import market_settings
from .._series import ist_time
from ..calendar import Calendar
from ..errors import ConfigError, LiveError
from ._broker_units import strategy_units
from ._carry_rebuild_gtts import GttContext, NamedWords, apply, unsent
from ._carry_start import BINDINGS_KEY, CONFIRMED_KEY, REFERENCES_KEY, UNCONFIRMED_KEY
from ._fill_ledger import FillLedger
from ._journal_gtt_rows import GttBinding, GttState
from ._order_state import ENTRY_PURPOSES
from ._plan_catch_up import INSTRUMENT
from ._plan_engine import session_engine
from ._plan_setup import with_facts
from ._preflight_plan import InstrumentFacts
from ._session_restore_gtts import journal_rows
from ._session_saved import KERNEL_INSTRUMENTS_KEY, Saved, read_saved
from .journal import Journal

#: The key of a rebuilt day-close record's account of its rebuild (module docstring).
REBUILT_KEY = "rebuilt_after_stop"
#: Why a rebuilt record's stop GTTs are not confirmed.
NOT_CONFIRMED = "the session stopped before its close read the GTT book"


@dataclass(frozen=True)
class Rebuilt:
    """A strategy's record rebuilt, not written yet (module docstring): its `name`, the
    trading `day` its session stopped on, when it stopped and when it last saved (UTC
    nanoseconds), the updates applied, the words of each one named, the day-close
    record's carried state (`carried`, with what a close's record adds), the strategy's
    `self.state` and version id, and the words of each order that opens or adds and still
    worked when the session stopped (`opening`: it may have filled after the stop, which
    the journal cannot tell)."""

    name: str
    day: dt.date
    stopped_ns: int
    saved_ns: int
    updates: int
    named: tuple[str, ...]
    carried: Mapping[str, Any] = field(default_factory=dict)
    user_state: Mapping[str, Any] = field(default_factory=dict)
    version_id: str = ""
    opening: tuple[str, ...] = ()

    @property
    def units(self) -> int:
        """The signed units the record carries."""
        return sum(round(float(p["quantity_units"])) for p in self.carried.get("positions", []))

    @property
    def bound(self) -> frozenset[int]:
        """The journal's GTTs (their `seq`) the record binds."""
        return frozenset(int(row["seq"]) for row in self.carried.get(BINDINGS_KEY, []))

    def lines(self) -> list[str]:
        """The banner's lines of it."""
        head = (
            f"{self.name}: its session of {self.day} stopped at {_clock(self.stopped_ns)} IST "
            "without its day-close record; its position is continued from its last save of "
            f"{_clock(self.saved_ns)} and {self.updates} later updates"
        )
        return [head, *(f"{self.name}: {word}" for word in self.named)]


@dataclass
class _Orders:
    """What the order step did: updates applied, words named, the latest journal time
    read, and the kernel orders it gave fills to."""

    applied: int = 0
    named: list[str] = field(default_factory=list)
    latest_ns: int = 0
    handled: set[int] = field(default_factory=set)


def rebuild(
    journal: Journal, spec: Any, day: dt.date, today: dt.date, calendar: Calendar
) -> Rebuilt:
    """Rebuild strategy `spec`'s (a `_plan_strategies.StrategySpec`) session of `day`, which
    stopped without its day-close record, for the start of `today`, from what `journal`
    holds (module docstring); `write` writes its record. Raises LiveError for what is
    refused, and JournalError when the journal cannot be read."""
    name = spec.name
    found = journal.rms_saved(name, day=day)
    if found is None:
        raise LiveError(f"{name}: no save of its session of {day} is in the journal")
    saved = read_saved(found.state, name, spec.product)
    instruments = found.state.get(KERNEL_INSTRUMENTS_KEY)
    if not instruments:
        raise LiveError(
            f"{name}: its save of {day} names no instruments (an earlier version of openalgo "
            "wrote it), so what it holds cannot be rebuilt"
        )
    rows = journal_rows(journal, name, day, saved.bindings)
    views = _gtt_views(_engine(spec, day, saved, instruments, calendar), saved, today, [])
    engine = _engine(spec, day, saved, instruments, calendar)
    pairs = _restore(engine, saved, today, unsent(views, rows), name)
    tick = float(instruments[INSTRUMENT]["tick_size"])
    orders = _orders(engine, saved, pairs, journal, day, tick)
    gtts = apply(engine, views, rows, GttContext(journal, tick, orders.handled))
    if gtts.unaccounted:
        raise LiveError(
            f"{name}: its session of {day} stopped without its day-close record, and what "
            f"the journal holds after its last save cannot be accounted for: "
            f"{'; '.join(gtts.unaccounted)}"
        )
    working = set(call(engine.working_orders))
    opening = _opening(saved, pairs, working)
    call(engine.stop_session)
    carried = dict(call(engine.carried_state))
    _add_record_parts(carried, rows, _reference(found.state, carried, journal, day, working))
    kept = journal.user_state(name)
    versions = [v for v in journal.versions(day=day) if v.strategy == name]
    rebuilt = Rebuilt(
        name,
        day,
        stopped_ns=max(found.saved_ns, orders.latest_ns, gtts.latest_ns),
        saved_ns=found.saved_ns,
        updates=orders.applied + gtts.applied,
        named=(*orders.named, *gtts.left),
        carried=carried,
        user_state={} if kept is None else dict(kept.state),
        version_id=versions[-1].version_id if versions else spec.version.version_id,
        opening=opening,
    )
    carried[REBUILT_KEY] = {
        "stopped_ns": rebuilt.stopped_ns,
        "saved_ns": rebuilt.saved_ns,
        "updates": rebuilt.updates,
        "named": list(rebuilt.named),
    }
    return rebuilt


def write(journal: Journal, rebuilt: Rebuilt) -> None:
    """Write `rebuilt`'s day-close record of its day (module docstring). Raises
    JournalError when the journal cannot write it."""
    journal.put_day_close(
        rebuilt.name,
        dict(rebuilt.carried),
        dict(rebuilt.user_state),
        rebuilt.version_id,
        day=rebuilt.day,
    )


def _opening(saved: Saved, pairs: Mapping[int, int], working: set[int]) -> tuple[str, ...]:
    """The words of each order of the save that opens or adds and still worked when the
    session stopped (`Rebuilt.opening`)."""
    found = []
    for old, new in sorted(pairs.items(), key=lambda pair: pair[1]):
        record = saved.orders.get(old)
        if record is None or new not in working or record.intent.purpose not in ENTRY_PURPOSES:
            continue
        intent = record.intent
        found.append(
            f"its {intent.purpose.value} order {record.order_id or intent.intent_id} "
            f"({intent.action} {intent.quantity} {intent.symbol})"
        )
    return tuple(found)


def _engine(
    spec: Any, day: dt.date, saved: Saved, instruments: Sequence[Mapping[str, Any]], calendar: Any
) -> Any:
    """A kernel session of `day` for `spec` on the saved `instruments`, with the save's
    capital (module docstring). Raises LiveError for a save of another instrument than the
    strategy trades now."""
    capital = float(saved.live_state["account"]["capital_rupees"])
    plan = dataclasses.replace(spec.plan, start=day, end=day, capital=capital)
    facts = [
        InstrumentFacts(
            str(item["symbol"]),
            str(item["exchange"]),
            int(item["lot_size"]),
            float(item["tick_size"]),
            item.get("freeze_qty"),
        )
        for item in instruments
    ]
    try:
        with_tick, _ = with_facts(spec.instruments, facts)
        arguments = spec.arguments
        market = market_settings(
            plan,
            list(with_tick),
            {"costs": arguments["costs"], "slippage": arguments["slippage"], "margin": plan.margin},
            announce=False,
        )
    except ConfigError as error:
        saved_on = ", ".join(f"{i['exchange']}:{i['symbol']}" for i in instruments)
        raise LiveError(
            f"{spec.name}: its save of {day} is of {saved_on}, which it no longer trades as "
            f"saved ({error}): what it holds cannot be rebuilt"
        ) from None
    config = plan.native_config(deliver_events=False) | market.config
    return session_engine(config, instruments, calendar)


def _restore(
    engine: Any, saved: Saved, today: dt.date, never_sent: list[int], name: str
) -> dict[int, int]:
    """Give `engine` the save, standing where it stood; the kernel's (saved id, new id)
    pairs. Raises LiveError naming why the kernel refuses it."""
    try:
        pairs = call(engine.restore_stopped_state, saved.live_state, to_day(today), never_sent)
    except Exception as error:
        raise LiveError(
            f"{name}: the engine refuses its last save of its session of "
            f"{saved.live_state.get('day')} ({error}), so what it holds cannot be rebuilt"
        ) from error
    return {int(old): int(new) for old, new in pairs.items()}


def _gtt_views(probe: Any, saved: Saved, today: dt.date, never_sent: list[int]) -> list[dict]:
    """The kernel's GTTs of the save as `Engine.gtts` dicts, read from `probe`, a session
    given the save with no update (the views the GTT step judges by)."""
    _restore(probe, saved, today, never_sent, "the save")
    return [dict(view) for view in call(probe.gtts, INSTRUMENT, saved.product)]


def _orders(
    engine: Any, saved: Saved, pairs: Mapping[int, int], journal: Journal, day: dt.date, tick: float
) -> _Orders:
    """Give `engine` the journal's last word of each order the save works and sent (module
    docstring)."""
    done = _Orders()
    unbound = {intent.intent_id for intent in journal.unbound(day=day)}
    queued = {each.intent.intent_id for each in journal.queued(day=day)}
    for old, new in sorted(pairs.items(), key=lambda pair: pair[1]):
        record = saved.orders.get(old)
        if record is None or not saved.sent.get(old, False):
            continue
        intent = record.intent
        order_id = record.order_id or journal.order_id_of(intent.intent_id) or ""
        what = f"its order {intent.intent_id} ({intent.action} {intent.quantity} {intent.symbol})"
        if not order_id and intent.intent_id in unbound | queued:
            where = (
                "waited in OpenAlgo's Action Center"
                if intent.intent_id in queued
                else ("had no known outcome")
            )
            done.named.append(
                f"{what} {where} when the session stopped: this start's read of the books "
                "shows what it filled"
            )
        update = journal.last_update(order_id, day=day) if order_id else None
        if update is None:
            continue
        done.latest_ns = max(done.latest_ns, update.ts_recv_ns)
        ledger = FillLedger(
            engine,
            new,
            tick=tick,
            alerts=NamedWords(done.named),
            filled_units=record.filled_units,
            filled_value=record.filled_value,
        )
        result = ledger.apply(update)
        done.handled.add(new)
        if result.increment is not None or result.ended is not None:
            done.applied += 1
    return done


def _reference(
    state: Mapping[str, Any],
    carried: Mapping[str, Any],
    journal: Journal,
    day: dt.date,
    working: set[int],
) -> list[dict[str, Any]]:
    """The leg's reference of the broker's units the rebuilt record keeps (module
    docstring): the save's, when it still holds (no order of the session's still `working`
    when it stopped); else none."""
    references = list(state.get(REFERENCES_KEY) or [])
    if len(references) != 1 or working:
        return []
    reference = dict(references[0])
    if not reference.get("settled"):
        return []
    held = sum(round(float(p["quantity_units"])) for p in carried.get("positions", []))
    broker = int(reference["units"])
    if strategy_units(str(reference["product"]), held, broker) != held:
        return []
    read_ns = int(reference["read_ns"])
    for order in journal.bound_orders(day=day):
        if _filled_after(journal.updates(order.order_id, day=day), read_ns):
            return []
    return [reference]


def _filled_after(words: Sequence[Any], read_ns: int) -> bool:
    """Whether an order's journal `words` (its updates, cumulative) show units filled after
    `read_ns` (UTC nanoseconds): more filled in all of them than in those received by then
    (a later word of a fill made before, a poll's, moves nothing)."""
    before = max((w.filled_quantity for w in words if w.ts_recv_ns <= read_ns), default=0)
    return max((w.filled_quantity for w in words), default=0) > before


def _add_record_parts(
    carried: dict[str, Any], rows: Mapping[int, GttBinding], reference: list[dict[str, Any]]
) -> None:
    """Add to `carried` what a close's record adds (module docstring): the leg's
    `reference`; each carried GTT the reference confirms (`_confirmed`), and each other
    named not confirmed; and the journal's GTT each carried GTT is bound to (`rows`, by
    kernel id)."""
    gtts = [int(gtt["gtt"]) for gtt in carried.get("gtts", []) if int(gtt["gtt"]) in rows]
    read_ns = int(reference[0]["read_ns"]) if reference else None
    confirmed = [gtt for gtt in gtts if _confirmed(rows[gtt], read_ns)]
    carried[REFERENCES_KEY] = reference
    carried[CONFIRMED_KEY] = confirmed
    carried[UNCONFIRMED_KEY] = [
        {"gtt": gtt, "trigger_id": rows[gtt].trigger_id or "", "why": NOT_CONFIRMED}
        for gtt in gtts
        if gtt not in confirmed
    ]
    carried[BINDINGS_KEY] = [{"gtt": gtt, "seq": rows[gtt].seq} for gtt in gtts]


def _confirmed(row: GttBinding, read_ns: int | None) -> bool:
    """Whether the leg's reference read at `read_ns` (None for none) confirms the GTT the
    journal's `row` is: a read of the GTT book showed it resting (ACTIVE) on terms it kept
    from before the reference's read, so it had not fired by then (the reference's units
    hold the leg's, which a fire would have lowered)."""
    resting = row.state is GttState.ACTIVE and row.trigger_id is not None
    return read_ns is not None and resting and row.changed_ns <= read_ns


def _clock(ns: int) -> str:
    """IST wall-clock time of `ns` (UTC nanoseconds), HH:MM:SS."""
    return ist_time(ns).strftime("%H:%M:%S")


__all__ = ["NOT_CONFIRMED", "REBUILT_KEY", "Rebuilt", "rebuild", "write"]
