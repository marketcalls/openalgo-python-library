"""Why the history of an expired contract cannot be loaded, and what to do instead.

A contract leaves the master contract the OpenAlgo server downloads from the broker
once it has expired, and brokers stop serving its history, so the server answers a
history request for it as for a symbol it never knew (SymbolNotFoundError, raised by the
data layer). That answer tells a user to check the symbol's spelling, which is no help
for a futures run that starts before the last expiry. When the symbol is a contract of
the run whose expiry is before the run's present (`now=`, the clock by default), the
error says that the contract expired, on which date, and what to do.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from ._series import ist_time
from .errors import SymbolNotFoundError
from .reference import Instrument


def _expired(
    instruments: list[Instrument], symbol: str | None, present_ns: int
) -> Instrument | None:
    """The run's contract named `symbol` when it expired before the IST date of
    `present_ns` (UTC nanoseconds), else None."""
    today = ist_time(present_ns).date()
    for instrument in instruments:
        if instrument.symbol == symbol and instrument.expiry is not None:
            return instrument if instrument.expiry < today else None
    return None


def _message(contract: Instrument) -> str:
    expiry = contract.expiry.isoformat()
    return (
        f"{contract.exchange}:{contract.symbol}: the contract expired on {expiry} and the "
        "OpenAlgo server no longer lists it: brokers stop serving an expired contract's "
        f"history. Start the run after {expiry}, or run on stored bars: data= a dict by "
        f"contract symbol holding {contract.symbol} and every other contract the run trades"
    )


@contextmanager
def expired_contracts(instruments: list[Instrument], present_ns: int) -> Iterator[None]:
    """Run the block, turning a SymbolNotFoundError for one of `instruments` that had
    expired before the IST date of `present_ns` (UTC nanoseconds) into a
    SymbolNotFoundError that says so (the server's answer is its cause). Any other error
    passes unchanged."""
    try:
        yield
    except SymbolNotFoundError as exc:
        contract = _expired(instruments, exc.symbol, present_ns)
        if contract is None:
            raise
        raise SymbolNotFoundError(_message(contract), exc.symbol, exc.exchange) from exc


__all__ = ["expired_contracts"]
