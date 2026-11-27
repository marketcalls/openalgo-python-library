"""The start's read of what a session continues from earlier days (`open_carry`, on a
journal handle opened for it before the preflight; `_carry_start`), and what it cannot
continue, refused before anything is sent (20-positional-live 4.2 step 2; the lead's
rulings N5a-1 and N5c-2): a capital other than a carried strategy's
(`_carry_start.capital_refusals`), what a strategy holds from an earlier day that no
day-close record carries (with a record a later save outdates, which is set aside), unless
the broker's books show it closed by hand, the strategy has a save of today, which a
restart resumes, or its session that stopped without its record is rebuilt from its last
save and written as that day's record (`_carry_unrecorded`, `_carry_rebuild`: the carry
is then read again, so the rebuilt record is continued as any record), and a carried
contract the strategy no longer trades that has not expired (`_carry_contracts`: one that
expired was settled by the exchange, and the strategy starts flat).

Each refusal names what is held and what to do. Days are IST trading days. Run by the
main thread at the start, before the preflight.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import pathlib
from typing import Any

from ..calendar import default_calendar
from ._carry_books import read_books
from ._carry_contracts import settle_expired
from ._carry_rebuild import Rebuilt, rebuild
from ._carry_start import CarryAtStart, capital_refusals, read_carry
from ._carry_unrecorded import Unrecorded, judge, resumed_today, set_aside, unrecorded
from ._preflight_plan import refused
from ._types import Mode, Now
from .journal import Journal


def open_carry(
    spec: Any,
    *,
    path: pathlib.Path,
    account: str,
    broker: str,
    mode: Mode,
    now: Now,
    client: Any,
) -> CarryAtStart:
    """Read the carry of the session `spec` (a `plan.SessionSpec`) from the journal at
    `path` of `account` at `broker`, in the ledger of `mode`, on a handle opened for the
    read and closed after it (module docstring), and refuse a start whose capital is not a
    carried strategy's or that holds what no record carries and the books of `client` (the
    session's data client, read only then) do not show closed and the rebuild cannot
    continue. Raises PreflightError for such a start, JournalError when the journal cannot
    be read or written, and LiveError for a record that cannot be read."""
    names = [each.name for each in spec.strategies]
    refusals: list[str] = []
    with Journal.open(path, account=account, broker=broker, mode=mode, now=now) as journal:
        carry = read_carry(journal, names, spec.day, mode)
        resumed = resumed_today(journal, names, spec.day)
        held = unrecorded(journal, names, spec.day, carry, resumed=resumed)
        if held:
            books = read_books(client, [leg for each in held for leg in each.legs])
            carry, refusals, done = judge(held, books, journal, carry, _rebuild(journal, spec))
            if done:
                carry = _read_again(journal, spec, carry, done)
        carry = set_aside(carry, held, resumed=resumed)
        carry, moved = settle_expired(journal, carry, spec.strategies)
    refusals += moved
    capitals = {each.name: each.capital for each in spec.strategies}
    refusals = capital_refusals(carry, capitals) + refusals
    if refusals:
        raise refused(tuple(refusals))
    return carry


def _rebuild(journal: Journal, spec: Any) -> Any:
    """The rebuild of a strategy of `spec` (`_carry_rebuild.rebuild`) for the judgement."""
    by_name = {each.name: each for each in spec.strategies}
    calendar = default_calendar()

    def rebuild_one(held: Unrecorded, day: dt.date) -> Rebuilt:
        return rebuild(journal, by_name[held.name], day, spec.day, calendar)

    return rebuild_one


def _read_again(
    journal: Journal, spec: Any, carry: CarryAtStart, done: list[Rebuilt]
) -> CarryAtStart:
    """The carry read again once the records of `done` are written, with the banner's lines
    the judgement gave (`carry.settled`) and each rebuilt strategy's."""
    names = [each.name for each in spec.strategies]
    fresh = read_carry(journal, names, spec.day, carry.mode)
    lines = [line for each in done for line in each.lines()]
    return dataclasses.replace(fresh, settled=(*carry.settled, *lines))


__all__ = ["open_carry"]
