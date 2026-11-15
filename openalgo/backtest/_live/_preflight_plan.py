"""The values of a live session's preflight (`preflight`): what the session asks it to
check (`PreflightPlan`), the outcome of one check (`Check`), the facts OpenAlgo gives of
an instrument (`InstrumentFacts`) and what a passed preflight found (`PreflightReport`).

Amounts and prices are rupees, quantities units (never lots). Every value is frozen.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any

from ..errors import ConfigError
from ._types import AlertsLike, Mode

#: How the message of a check that passed, with something the user should know, starts.
WARNING = "warning: "


def positive_amount(value: Any) -> float | None:
    """`value` as a float when it is a finite real number above zero (a bool is not one),
    else None."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    amount = float(value)
    return amount if math.isfinite(amount) and amount > 0 else None


@dataclass(frozen=True, kw_only=True)
class PreflightPlan:
    """What a live session asks the preflight to check. Keyword arguments only.

    `instruments` are the (symbol, exchange) pairs the session trades, written as
    OpenAlgo writes them (("SBIN", "NSE")); a list is kept as a tuple. `interval` is the
    strategy's bar interval as OpenAlgo writes it ("5m"), `product` its order product.
    `capital`, `max_daily_loss` and `account_max_daily_loss` are rupees as the user gave
    them, None when not given: the preflight checks them, so each problem is reported
    with the others. `mode` is the mode the mode guard read when the session started
    (D21), and `alerts` the session's alert channel, None when it has none.

    Raises ConfigError when `instruments` is empty or holds anything but pairs of
    non-blank strings, or when `mode` is not a `Mode`.
    """

    instruments: tuple[tuple[str, str], ...]
    interval: str
    product: str
    capital: float | None
    max_daily_loss: float | None
    account_max_daily_loss: float | None = None
    mode: Mode
    alerts: AlertsLike | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "instruments", _instrument_pairs(self.instruments))
        if not isinstance(self.mode, Mode):
            raise ConfigError(
                f"mode is the mode the mode guard read at the session's start, Mode.SANDBOX "
                f"or Mode.LIVE, not {self.mode!r}"
            )


@dataclass(frozen=True)
class Check:
    """The outcome of one preflight check. `name` says which check it is ("funds",
    "symbol:NSE:SBIN"), `ok` whether it passed, and `message` what it found: for a
    failure what to fix, for a pass what was verified. A pass the user should hear about
    (no alert channel in SANDBOX mode, say) has a message that starts with `WARNING`."""

    name: str
    ok: bool
    message: str


@dataclass(frozen=True)
class InstrumentFacts:
    """What OpenAlgo's master contract gives of an instrument: `lot_size` in units (1 for
    cash equity), `tick_size` in rupees, and `freeze_quantity`, the most units one order
    may carry on the exchange, None when OpenAlgo knows no such limit for it."""

    symbol: str
    exchange: str
    lot_size: int
    tick_size: float
    freeze_quantity: int | None


@dataclass(frozen=True)
class PreflightReport:
    """What a passed preflight found: every check in the order it ran, the `broker`
    OpenAlgo is logged in to (as `ping` names it), `funds_available`, the available cash
    in rupees OpenAlgo reported (the sandbox's own funds in SANDBOX mode), and the facts
    of each instrument, in the plan's order."""

    checks: tuple[Check, ...]
    broker: str
    funds_available: float
    instruments: tuple[InstrumentFacts, ...]

    @property
    def warnings(self) -> tuple[str, ...]:
        """The messages of the checks that passed with a warning, in order."""
        return tuple(check.message for check in self.checks if check.message.startswith(WARNING))


def _instrument_pairs(instruments: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(instruments, list | tuple) or not instruments:
        raise ConfigError(
            "instruments must be a non-empty list of (symbol, exchange) pairs, such as "
            f"[('SBIN', 'NSE')], not {instruments!r}"
        )
    pairs = []
    for pair in instruments:
        named = isinstance(pair, list | tuple) and len(pair) == 2
        if not (named and all(isinstance(part, str) and part.strip() for part in pair)):
            raise ConfigError(
                "each instrument is a (symbol, exchange) pair of strings, such as "
                f"('SBIN', 'NSE'), not {pair!r}"
            )
        pairs.append((pair[0], pair[1]))
    return tuple(pairs)


__all__ = [
    "WARNING",
    "Check",
    "InstrumentFacts",
    "PreflightPlan",
    "PreflightReport",
    "positive_amount",
]
