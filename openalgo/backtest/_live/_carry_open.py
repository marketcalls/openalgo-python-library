"""The start's read of what a session continues from earlier days (`open_carry`, on a
journal handle opened for it before the preflight; `_carry_start`), and what it cannot
continue, refused before anything is sent (20-positional-live 4.2 step 2; the lead's
ruling N5a-1): a capital other than a carried strategy's (`_carry_start.capital_refusals`),
what a strategy holds from an earlier day that no day-close record carries (with a record
a later save outdates, which is set aside), unless the broker's books show it closed by
hand (`_carry_unrecorded`), and a carried contract the
strategy no longer trades that has not expired (`_carry_contracts`: one that expired was
settled by the exchange, and the strategy starts flat).

Each refusal names what is held and what to do. Days are IST trading days. Run by the
main thread at the start, before the preflight.
"""

from __future__ import annotations

import pathlib
from typing import Any

from ._carry_contracts import settle_expired
from ._carry_start import CarryAtStart, capital_refusals, read_carry
from ._carry_unrecorded import judge, read_books, set_aside, unrecorded
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
    session's data client, read only then) do not show closed. Raises PreflightError for
    such a start, JournalError when the journal cannot be read, and LiveError for a record
    that cannot be read."""
    names = [each.name for each in spec.strategies]
    refusals: list[str] = []
    with Journal.open(path, account=account, broker=broker, mode=mode, now=now) as journal:
        carry = read_carry(journal, names, spec.day, mode)
        held = unrecorded(journal, names, spec.day, carry)
        if held:
            carry, refusals = judge(held, read_books(client, held), journal, carry)
        carry = set_aside(carry, held)
        carry, moved = settle_expired(journal, carry, spec.strategies)
    refusals += moved
    capitals = {each.name: each.capital for each in spec.strategies}
    refusals = capital_refusals(carry, capitals) + refusals
    if refusals:
        raise refused(tuple(refusals))
    return carry


__all__ = ["open_carry"]
