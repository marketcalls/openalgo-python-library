"""What a strategy holds from an earlier day that no day-close record carries, judged at the
start before the preflight (20-positional-live 4.2 step 2; the lead's ruling N5a-1 and
review round 1 of unit N5a). Split from `_carry_open` so each module keeps one job.

- **An unrecorded save** (`_save`): the strategy's latest session, of an earlier day,
  ended without writing its day-close record (the process ended before the close: a crash,
  a power cut) while its last save of that day, made after the strategy's latest record
  (if any: a record that save outdates is never continued, `_carry_start`), holds a CNC or
  NRML position, a CNC or NRML order working (it may have filled after the save), or a GTT
  that has not ended. An MIS position never counts: the broker squares MIS off at its
  close and nothing of it is carried, so an MIS strategy starts as in 2.4.1 whatever its
  last save holds (20-positional-live 1). A strategy whose outdated record is set aside
  with nothing of the later session left to judge starts flat, the banner saying so
  (`set_aside`), and its close writes its record of the day.
- **A loose GTT**: a GTT of the journal from an earlier day that has not ended and that
  neither the strategy's continued record nor its set-aside one binds (one a session placed
  and never recorded at a close): started afresh beside it, the strategy would trade beside
  a GTT no engine follows.

Such a strategy is judged against the broker's books, read once for every such strategy of
the start (`BookReads`: the position book, the holdings when a CNC leg is judged, the
order and trade books, and the GTT book of every status).

- **Settled by hand**: the broker holds none of any leg involved (each saved CNC or NRML
  position's and order's, each GTT's) in its product, no order of those legs works, none of
  the strategy's GTTs of earlier days rests (ended, expired, rejected, fired, or gone from a
  book that was read), and no stop GTT of its save may have fired by the books
  (`_carry_fired`: gone or triggered with the leg's units moved by exactly its units from
  the save's reference, which 3.3 takes as fired). Each of those journal GTTs is ended
  (`NOT_PLACED` for one whose place was never answered) with a note, so no later start
  follows it, and it starts flat, the banner saying so (`Unrecorded.settled`); its close
  writes its record of the day when an older one is set aside.
- **Rebuilt**: otherwise, a strategy with an unrecorded save has its session rebuilt from
  that save and the journal's updates newer than it, written as that day's day-close
  record (`_carry_rebuild`, the lead's ruling N5c-2), and the start continues it exactly as
  from a record (`_carry_open` reads the carry again): the same judgement against the
  broker's books, the same restore, the same refusals. The banner names it
  (`_carry_rebuild.Rebuilt.lines`).
- **Refused** (PreflightError) where nothing safe can be continued, each naming what may
  still be held and what to do (`CHECK_BY_HAND`): a save the rebuild refuses (a save the
  kernel refuses, one an earlier version wrote, an update that cannot be accounted for), a
  loose GTT (with no save, or one the rebuilt record binds none of), and an order that
  opens or adds still working when the session stopped while the books show the leg other
  than the record holds it. A CNC leg the books show held may hold units of the user's own,
  which the books cannot tell from the strategy's (7.2): the refusal names them and the way
  out that remains, another account name.

A strategy with a save of today (`resumed_today`) is not judged at all, and no set-aside
line is shown for it: an earlier start today passed this judgement (anything loose was
refused, or found settled and ended), and a restart today resumes today's save, whose GTT
bindings take back each GTT of an earlier day it follows (`_session_restore_gtts`). Its
latest record may be of today (a Ctrl+C that kept the leg, or the day's close), which the
carry of earlier days never reads (review round 3 of unit N5a).

Days are IST trading days, quantities signed whole units. Run by the main thread at the
start, before the order worker follows any GTT.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .._products import CNC, OVERNIGHT_PRODUCTS
from ..errors import LiveError
from ._carry_books import BookReads, Leg, books_of
from ._carry_fired import may_have_fired
from ._carry_rebuild import Rebuilt, write
from ._carry_start import CarryAtStart, gtt_seqs
from ._gtt_book import Seen, gtt_rows, seen_of
from ._journal_carry import DayClose
from ._journal_gtt_rows import GttBinding, GttState
from ._types import Purpose
from .journal import Journal

#: What a start refused for what a strategy may still hold from an earlier day asks of the
#: operator.
CHECK_BY_HAND = (
    "check the position and its GTTs in OpenAlgo: close the position and cancel its GTTs "
    "by hand, and the next start finds them closed and begins flat; or start under another "
    "account name (account=) to leave them as they are"
)
#: The kernel's states of a saved GTT that has not ended (the binding's `GTT_STATES`).
_LIVE_GTT_STATES = frozenset({"Requested", "Active", "ModifyPending", "CancelPending", "Triggered"})


@dataclass(frozen=True)
class Unrecorded:
    """What strategy `name` holds from an earlier day that no day-close record carries
    (module docstring): `save_day`, the day of its unrecorded save (None for none),
    `legs`, the legs its save and its GTTs are on, `gtts`, the journal's GTTs of earlier
    days of it that have not ended (every one with an unrecorded save, else the loose
    ones), and `loose`, those of them no record binds."""

    name: str
    save_day: dt.date | None
    legs: frozenset[Leg]
    gtts: tuple[GttBinding, ...]
    loose: tuple[GttBinding, ...]

    def words(self) -> str:
        """Its loose GTTs, in a refusal's words."""
        return f"{self.name}: {_loose_words(self.loose)}"

    def settled(self, today: dt.date) -> str:
        """The banner's line of it once the books show it settled by hand."""
        since = self.save_day or min(gtt.day for gtt in self.gtts)
        legs = ", ".join(f"{s} {e} ({p})" for s, e, p in sorted(self.legs))
        return (
            f"{self.name}: what it held since {since} with no day-close record is closed at "
            f"the broker (none of {legs} held, none of its GTTs resting): it starts flat "
            f"on {today}"
        )


def resumed_today(journal: Journal, names: Sequence[str], day: dt.date) -> frozenset[str]:
    """The strategies of `names` with a save of `day`, which this start resumes and never
    judges (module docstring). Raises JournalError when the journal cannot be read."""
    return frozenset(name for name in names if journal.rms_state(name, day=day) is not None)


def unrecorded(
    journal: Journal,
    names: Sequence[str],
    day: dt.date,
    carry: CarryAtStart,
    *,
    resumed: frozenset[str],
) -> list[Unrecorded]:
    """What each of the strategies `names` holds from before `day` that no record carries
    (module docstring), for those that hold any; those `resumed` today (`resumed_today`)
    are not judged. Raises JournalError when the journal cannot be read."""
    open_gtts = [gtt for gtt in journal.open_gtts() if gtt.day < day]
    found = []
    for name in names:
        if name in resumed:
            continue
        record = _binding_record(journal, carry, name)
        save_day, legs = _save(journal, name, day)
        bound = set() if record is None else set(gtt_seqs(record).values())
        own = tuple(gtt for gtt in open_gtts if gtt.strategy == name)
        loose = tuple(gtt for gtt in own if gtt.seq not in bound)
        if save_day is None and not loose:
            continue
        gtts = own if save_day is not None else loose
        legs |= {(gtt.symbol, gtt.exchange, gtt.product) for gtt in gtts}
        found.append(Unrecorded(name, save_day, frozenset(legs), gtts, loose))
    return found


def _binding_record(journal: Journal, carry: CarryAtStart, name: str) -> DayClose | None:
    """Strategy `name`'s day-close record whose GTT bindings count (module docstring): the
    record the start continues, else the one a later save set aside, else None."""
    carried = carry.of(name)
    if carried is not None:
        return carried.record
    return journal.day_close(name) if name in carry.superseded else None


def judge(
    held: Sequence[Unrecorded],
    books: BookReads,
    journal: Journal,
    carry: CarryAtStart,
    rebuild: Callable[[Unrecorded, dt.date], Rebuilt],
) -> tuple[CarryAtStart, list[str], list[Rebuilt]]:
    """Judge `held` against `books` (module docstring): (the carry with the banner's line
        of each settled strategy, a refusal for each strategy nothing safe can be continued
        of, the strategies whose session was rebuilt by `rebuild` (given the strategy and the day
    of its save) and its record written). A
        settled strategy's journal GTTs are ended through `journal`. Raises JournalError when
        the journal cannot be read or written."""
    settled: list[str] = []
    refusals: list[str] = []
    rebuilt: list[Rebuilt] = []
    for each in held:
        if _closed(each, books) and not may_have_fired(each, books, journal):
            for gtt in each.gtts:
                _end(journal, gtt, carry.day)
            settled.append(each.settled(carry.day))
            continue
        own = _own_units_words(each, books)
        if each.save_day is None:
            refusals.append(f"{each.words()}: {CHECK_BY_HAND}{own}")
            continue
        try:
            done = rebuild(each, each.save_day)
        except LiveError as error:
            refusals.append(f"{error}: {CHECK_BY_HAND}{own}")
            continue
        why = _not_continued(each, done, books)
        if why:
            refusals.append(f"{each.name}: {why}: {CHECK_BY_HAND}{own}")
            continue
        write(journal, done)
        rebuilt.append(done)
    return replace(carry, settled=(*carry.settled, *settled)), refusals, rebuilt


def _not_continued(held: Unrecorded, done: Rebuilt, books: BookReads) -> str:
    """Why the record `done` rebuilt for `held` is not continued, "" when it is (module
    docstring): an order that opens or adds still worked when the session stopped while
    the books show the leg other than the record holds it (its fills after the stop cannot
    be told from units of the user's own), or loose GTTs."""
    loose = [gtt for gtt in held.gtts if gtt.seq not in done.bound]
    if loose:
        return _loose_words(loose)
    if not done.opening:
        return ""
    found = [books_of(books, leg).units for leg in held.legs]
    if all(units == done.units for units in found):
        return ""
    shown = ", ".join("unreadable" if units is None else str(units) for units in found)
    return (
        f"{' and '.join(done.opening)} still worked when its session of {done.day} stopped "
        f"without its day-close record, and the books show {shown} units where its last save "
        f"and the journal's updates hold {done.units}: the start cannot tell what that order "
        "filled after the stop from units of your own"
    )


def set_aside(
    carry: CarryAtStart, held: Sequence[Unrecorded], *, resumed: frozenset[str]
) -> CarryAtStart:
    """The carry with the banner's line of each strategy whose outdated day-close record
    (`CarryAtStart.superseded`) is not continued and whose later save left nothing to
    judge (`held` names none of its saves): it starts flat (module docstring). A strategy
    `resumed` today has none: an earlier start today said so, and it resumes today's save."""
    judged = {each.name for each in held if each.save_day is not None}
    lines = [
        f"{name}: its day-close record of {outdated.record_day} is not continued, since its "
        f"session of {outdated.saved_day} changed what it held and ended without a record of "
        f"its own, and nothing of that session is left to carry: it starts flat on {carry.day}"
        for name, outdated in carry.superseded.items()
        if name not in judged and name not in resumed
    ]
    return replace(carry, settled=(*carry.settled, *lines))


def _save(journal: Journal, name: str, day: dt.date) -> tuple[dt.date | None, set[Leg]]:
    """(the day of strategy `name`'s unrecorded save, the legs of the CNC and NRML
    positions and working orders it holds), or (None, none) when it has none (module
    docstring): its latest session's save, made after its latest day-close record. A
    working order counts: it may have filled after the save."""
    last = journal.last_ran(name, day)
    if last is None:
        return None, set()
    record = journal.day_close(name)
    if record is not None and journal.saved_after(record, day) != last:
        return None, set()
    saved = journal.rms_state(name, day=last)
    state = {} if saved is None else dict(saved.get("live_state", {}))
    instruments = state.get("instruments", [])
    held = [p for p in state.get("positions", []) if float(p["quantity_units"])]
    held += [order["request"] for order in state.get("orders", [])]
    legs = {
        _leg_of(instruments, terms) for terms in held if terms.get("product") in OVERNIGHT_PRODUCTS
    }
    gtts = [g for g in state.get("gtts", []) if g.get("status") in _LIVE_GTT_STATES]
    return (last, legs) if legs or gtts else (None, set())


def _leg_of(instruments: Sequence[Mapping[str, Any]], terms: Mapping[str, Any]) -> Leg:
    """The leg (symbol, exchange, product) of a saved position or order request."""
    instrument = instruments[int(terms["instrument"])]
    return str(instrument["symbol"]), str(instrument["exchange"]), str(terms["product"])


def _closed(held: Unrecorded, books: BookReads) -> bool:
    """Whether the books show everything `held` names closed at the broker (module
    docstring): every book read, no leg held or worked, no GTT resting."""
    read = (books.positionbook, books.orderbook, books.tradebook, books.gttbook)
    rows = None if any(book is None for book in read) else gtt_rows(books.gttbook or {})
    if rows is None:
        return False
    for leg in held.legs:
        found = books_of(books, leg)
        if found.units != 0 or found.rows is None or any(row.working for row in found.rows):
            return False
    return all(_not_resting(gtt, rows) for gtt in held.gtts)


def _own_units_words(held: Unrecorded, books: BookReads) -> str:
    """What a refusal of `held` adds when a CNC leg of it shows units at the broker: they
    may be the user's own (20-positional-live 7.2), which the books cannot tell from the
    strategy's, so the check above never finds such a leg closed; "" for none."""
    if books.positionbook is None or books.orderbook is None or books.tradebook is None:
        return ""
    shown = []
    for leg in sorted(held.legs):
        units = books_of(books, leg).units if leg[2] == CNC else 0
        if units:
            shown.append(f"{units} {leg[0]} {leg[1]} (CNC)")
    if not shown:
        return ""
    return (
        f". The books show {', '.join(shown)} held: if any of them are units of your own, "
        "the start cannot tell them from the strategy's and never finds the leg closed, so "
        "once the strategy's units are sold and its GTTs cancelled, start it under another "
        "account name (account=)"
    )


def _not_resting(gtt: GttBinding, rows: Mapping[str, Any]) -> bool:
    """Whether the GTT book `rows` (by trigger ID) shows journal GTT `gtt` not resting: by
    its trigger ID, or, for one whose place was never answered, no GTT of its instrument
    resting at all."""
    if gtt.trigger_id is None:
        return not any(
            (row.symbol, row.exchange) == (gtt.symbol, gtt.exchange)
            and seen_of(rows, trigger_id).rests
            for trigger_id, row in rows.items()
        )
    seen = seen_of(rows, gtt.trigger_id)
    return not seen.rests and seen is not Seen.UNREADABLE


def _end(journal: Journal, gtt: GttBinding, day: dt.date) -> None:
    """End journal GTT `gtt`, found not resting at the start of `day`'s session."""
    if gtt.trigger_id is None:
        note = f"not resting at the broker at the start of {day}, its place never answered"
        journal.set_gtt_state(gtt.seq, GttState.NOT_PLACED, note=note)
        return
    note = f"not resting at the broker at the start of {day}, its leg closed by hand"
    journal.set_gtt_state(gtt.seq, GttState.ENDED, note=note)


def _loose_words(gtts: Iterable[GttBinding]) -> str:
    """Loose GTTs (module docstring), in a refusal's words."""
    return "; ".join(f"{_gtt_words(gtt)} and no day-close record carries it" for gtt in gtts)


def _gtt_words(gtt: GttBinding) -> str:
    """A GTT of an earlier day that has not ended, in a refusal's words."""
    kind = "stop GTT" if gtt.purpose is Purpose.STOP else "GTT order"
    name = gtt.trigger_id or f"g{gtt.gtt_id} (its place was not answered)"
    leg = f"{gtt.exchange}:{gtt.symbol} ({gtt.product})"
    return f"its {kind} {name} of {gtt.day} on {leg} has not ended"


__all__ = [
    "CHECK_BY_HAND",
    "Unrecorded",
    "judge",
    "resumed_today",
    "set_aside",
    "unrecorded",
]
