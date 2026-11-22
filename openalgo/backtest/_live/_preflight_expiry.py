"""The preflight's check of each futures and options contract's expiry against OpenAlgo's
master contract (`preflight`; 18-multi-strategy 12.2; unit M11).

The preflight reads each instrument from OpenAlgo's master contract already (`symbol`,
`_preflight_server`); its `expiry` ("DD-MMM-YY", "29-DEC-26") is set against the reference
tables' contract, the `Instrument` the session resolved, by the rule a backtest uses too
(`_expiry_check`): the master contract agrees when it gives the listed expiry (the date in
the symbol) or the last trading day. A
disagreement refuses the start, naming both dates: the expiry day's square-off, the rolls
and the settlement rest on the tables' dates, so the tables or the master contract are
wrong and the session must not trade on them. An expiry missing from the answer, or one
that cannot be read, is a warning, not a refusal. A session of cash equity only has no
contract to check, and the check makes no call of its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .._expiry_check import agrees, master_expiry, tables_words
from ..reference import Instrument
from ._preflight_plan import WARNING, Check, InstrumentFacts

#: What to do about a disagreement, as its refusal says.
_ADVICE = (
    "the expiry day's square-off, the rolls and the settlement rest on these dates: upgrade "
    "openalgo (pip install -U openalgo), and if it still disagrees, report the reference "
    "table row with the exchange's circular; or download the broker's master contract "
    "again in OpenAlgo"
)


def expiry_checks(contracts: Sequence[Instrument], facts: Sequence[InstrumentFacts]) -> list[Check]:
    """One check per contract of `contracts` (the session's futures and options, as the
    reference tables resolved them) that OpenAlgo found (`facts`, the preflight's
    instruments): passed when OpenAlgo's expiry is the contract's listed expiry or its last
    trading day, failed when it is another day, and passed with a warning when OpenAlgo
    gives none or one that cannot be read (module docstring). A contract several
    strategies trade is checked once."""
    found = {(fact.symbol, fact.exchange): fact for fact in facts}
    named = {(contract.symbol, contract.exchange): contract for contract in contracts}
    checks = []
    for (symbol, exchange), contract in named.items():
        fact = found.get((symbol, exchange))
        if fact is not None:
            checks.append(_check(contract, fact.expiry))
    return checks


def _check(contract: Instrument, given: Any) -> Check:
    name = f"expiry:{contract.exchange}:{contract.symbol}"
    where, tables = f"{contract.exchange}:{contract.symbol}", tables_words(contract)
    day = master_expiry(given)
    if day is None:
        what = "no expiry" if given in (None, "") else f"the expiry {given!r}, which cannot be read"
        return Check(
            name,
            True,
            f"{WARNING}OpenAlgo's master contract gives {where} {what}: the session uses the "
            f"reference tables' dates ({tables}); check them against the exchange",
        )
    if agrees(contract, day):
        return Check(
            name, True, f"{where}: OpenAlgo's master contract agrees with the tables ({tables})"
        )
    return Check(
        name,
        False,
        f"{where}: OpenAlgo's master contract gives the expiry {day}, the reference tables "
        f"{tables}: {_ADVICE}",
    )


__all__ = ["expiry_checks"]
