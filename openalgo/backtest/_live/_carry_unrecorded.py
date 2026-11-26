"""What a strategy holds from an earlier day that no day-close record carries, judged at the
start before the preflight (20-positional-live 4.2 step 2; the lead's ruling N5a-1 and
review round 1 of unit N5a). Split from `_carry_open` so each module keeps one job.

- **An unrecorded save** (`_save_day`): the strategy's latest session, of an earlier day,
  ended without writing its day-close record (the process ended before the close: a crash,
  a power cut) while its last save of that day holds a CNC or NRML position, or a GTT that
  has not ended. An MIS position never counts: the broker squares MIS off at its close and
  nothing of it is carried, so an MIS strategy starts as in 2.4.1 whatever its last save
  holds (20-positional-live 1).
- **A loose GTT**: a GTT of the journal from an earlier day that has not ended and that no
  carried record binds (one a session placed and never recorded at a close). Started
  afresh beside it, the strategy would trade beside a GTT no engine follows.

Such a strategy is judged against the broker's books, read once for every such strategy of
the start (`BookReads`: the position book, the holdings when a CNC leg is judged, the
order and trade books, and the GTT book of every status). When the broker holds none of
any leg involved (each saved CNC or NRML position's, each GTT's) in its product, no order
of those legs works, and none of the strategy's GTTs of earlier days rests (ended, expired,
rejected, fired, or gone from a book that was read), the operator settled it by hand: each
of those journal GTTs is ended (`NOT_PLACED` for one whose place was never answered) with a
note, so no later start follows it, the strategy's stale carry (a record older than the
unrecorded save) is dropped, and it starts flat, the banner saying so (`Unrecorded.settled`).
Anything else refuses the start (PreflightError) naming what may still be held and what to
do (`CHECK_BY_HAND`).

Continuing such a leg from the last save of its day, with the journal's updates newer than
it (the lead's ruling N5a-1 (5)), needs the engine's state of a day carried to a later
day's session from a mid-session save, which the kernel gives only from a closed session
(`carried_state`): not built in this release.

Days are IST trading days, quantities signed whole units. Run by the main thread at the
start, before the order worker follows any GTT.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .._products import CNC, OVERNIGHT_PRODUCTS
from ._answers import server_answer
from ._carry_start import CarryAtStart, gtt_seqs
from ._gtt_book import Seen, gtt_rows, seen_of
from ._gtt_reads import ALL_GTTS
from ._journal_gtt_rows import GttBinding, GttState
from ._reconcile_rules import leg_books
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
#: The books the judgement reads, by the SDK call that reads each.
_BOOKS = ("positionbook", "orderbook", "tradebook")

#: A leg: (symbol, exchange, product).
Leg = tuple[str, str, str]


@dataclass(frozen=True)
class Unrecorded:
    """What strategy `name` holds from an earlier day that no day-close record carries
    (module docstring): `save_day`, the day of its unrecorded save (None for none),
    `legs`, the legs its save and its GTTs are on, and `gtts`, the journal's GTTs of
    earlier days of it that have not ended: every one with an unrecorded save, else the
    loose ones."""

    name: str
    save_day: dt.date | None
    legs: frozenset[Leg]
    gtts: tuple[GttBinding, ...]
    loose: tuple[GttBinding, ...]

    def words(self) -> str:
        """What it holds, in a refusal's words."""
        held = []
        if self.save_day is not None:
            held.append(
                f"its session of {self.save_day} stopped without its day-close record (the "
                "process ended before the close) while holding a position or a GTT, which "
                "this version continues only from a day-close record"
            )
        held += [_gtt_words(gtt) for gtt in self.loose]
        return f"{self.name}: {'; '.join(held)}"

    def settled(self, today: dt.date) -> str:
        """The banner's line of it once the books show it settled by hand."""
        since = self.save_day or min(gtt.day for gtt in self.gtts)
        legs = ", ".join(f"{s} {e} ({p})" for s, e, p in sorted(self.legs))
        return (
            f"{self.name}: what it held since {since} with no day-close record is closed at "
            f"the broker (none of {legs} held, none of its GTTs resting): it starts flat "
            f"on {today}"
        )


@dataclass(frozen=True)
class BookReads:
    """The broker's books the judgement reads (module docstring), as the SDK answered
    them; a book not read, or not a success, is None."""

    positionbook: Mapping[str, Any] | None
    holdings: Mapping[str, Any] | None
    orderbook: Mapping[str, Any] | None
    tradebook: Mapping[str, Any] | None
    gttbook: Mapping[str, Any] | None


def unrecorded(
    journal: Journal, names: Sequence[str], day: dt.date, carry: CarryAtStart
) -> list[Unrecorded]:
    """What each of the strategies `names` holds from before `day` that no record carries
    (module docstring), for those that hold any. Raises JournalError when the journal
    cannot be read."""
    open_gtts = [gtt for gtt in journal.open_gtts() if gtt.day < day]
    found = []
    for name in names:
        carried = carry.of(name)
        save_day, legs = _save(journal, name, day, carry)
        bound = set() if carried is None else set(gtt_seqs(carried.record).values())
        own = tuple(gtt for gtt in open_gtts if gtt.strategy == name)
        loose = tuple(gtt for gtt in own if gtt.seq not in bound)
        if save_day is None and not loose:
            continue
        gtts = own if save_day is not None else loose
        legs |= {(gtt.symbol, gtt.exchange, gtt.product) for gtt in gtts}
        found.append(Unrecorded(name, save_day, frozenset(legs), gtts, loose))
    return found


def read_books(client: Any, held: Iterable[Unrecorded]) -> BookReads:
    """The broker's books for judging `held` (module docstring), read through `client`
    (the session's data client); only reads."""
    answers = {name: _answer(client, name) for name in _BOOKS}
    cnc = any(leg[2] == CNC for each in held for leg in each.legs)
    holdings = _answer(client, "holdings") if cnc else None
    gttbook = _answer(client, "gttorderbook", status=ALL_GTTS)
    return BookReads(holdings=holdings, gttbook=gttbook, **answers)


def judge(
    held: Sequence[Unrecorded], books: BookReads, journal: Journal, carry: CarryAtStart
) -> tuple[CarryAtStart, list[str]]:
    """Judge `held` against `books` (module docstring): (the carry without the settled
    strategies' stale records and with the banner's line of each settled strategy, a
    refusal for each other). A settled strategy's journal GTTs are ended through `journal`.
    Raises JournalError when the journal cannot write them."""
    settled: list[str] = []
    refusals: list[str] = []
    dropped: set[str] = set()
    for each in held:
        if not _closed(each, books):
            refusals.append(f"{each.words()}: {CHECK_BY_HAND}")
            continue
        for gtt in each.gtts:
            _end(journal, gtt, carry.day)
        settled.append(each.settled(carry.day))
        if each.save_day is not None:
            dropped.add(each.name)
    kept = {name: c for name, c in carry.carried.items() if name not in dropped}
    return replace(carry, carried=kept, settled=(*carry.settled, *settled)), refusals


def _save(
    journal: Journal, name: str, day: dt.date, carry: CarryAtStart
) -> tuple[dt.date | None, set[Leg]]:
    """(the day of strategy `name`'s unrecorded save, the legs of the CNC and NRML
    positions it holds), or (None, none) when it has none (module docstring)."""
    last = journal.last_ran(name, day)
    carried = carry.of(name)
    if last is None or (carried is not None and carried.record.day >= last):
        return None, set()
    record = journal.day_close(name)
    if record is not None and record.day >= last:
        return None, set()
    saved = journal.rms_state(name, day=last)
    state = {} if saved is None else dict(saved.get("live_state", {}))
    instruments = state.get("instruments", [])
    legs = {
        _position_leg(instruments, position)
        for position in state.get("positions", [])
        if position.get("product") in OVERNIGHT_PRODUCTS and float(position["quantity_units"])
    }
    gtts = [g for g in state.get("gtts", []) if g.get("status") in _LIVE_GTT_STATES]
    return (last, legs) if legs or gtts else (None, set())


def _position_leg(instruments: Sequence[Mapping[str, Any]], position: Mapping[str, Any]) -> Leg:
    """The leg (symbol, exchange, product) of a saved position."""
    instrument = instruments[int(position["instrument"])]
    return str(instrument["symbol"]), str(instrument["exchange"]), str(position["product"])


def _closed(held: Unrecorded, books: BookReads) -> bool:
    """Whether the books show everything `held` names closed at the broker (module
    docstring): every book read, no leg held or worked, no GTT resting."""
    if books.positionbook is None or books.orderbook is None or books.tradebook is None:
        return False
    rows = None if books.gttbook is None else gtt_rows(books.gttbook)
    if rows is None:
        return False
    for symbol, exchange, product in held.legs:
        leg = leg_books(
            books.positionbook,
            books.tradebook,
            books.orderbook,
            symbol,
            exchange,
            product=product,
            holdings=books.holdings,
        )
        if leg.units != 0 or leg.rows is None or any(row.working for row in leg.rows):
            return False
    return all(_not_resting(gtt, rows) for gtt in held.gtts)


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


def _answer(client: Any, name: str, **fields: Any) -> Mapping[str, Any] | None:
    """The SDK's answer to read `name`, or None when it is not a success."""
    answer = server_answer(getattr(client, name)(**fields))
    return answer if answer.get("status") == "success" else None


def _gtt_words(gtt: GttBinding) -> str:
    """A GTT of an earlier day that has not ended and that no record carries."""
    kind = "stop GTT" if gtt.purpose is Purpose.STOP else "GTT order"
    name = gtt.trigger_id or f"g{gtt.gtt_id} (its place was not answered)"
    return (
        f"its {kind} {name} of {gtt.day} on {gtt.exchange}:{gtt.symbol} ({gtt.product}) has "
        "not ended and no day-close record carries it"
    )


__all__ = ["CHECK_BY_HAND", "BookReads", "Unrecorded", "judge", "read_books", "unrecorded"]
