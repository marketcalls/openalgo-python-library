"""The values of a live session's preflight (`preflight`): what the session asks it to
check (`PreflightPlan`), the outcome of one check (`Check`), the facts OpenAlgo gives of
an instrument (`InstrumentFacts`), every check of one run, passed or not
(`PreflightOutcome`), and what a passed preflight found (`PreflightReport`).

Amounts and prices are rupees, quantities units (never lots). Every value is frozen.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any

from .._risk_settings import GTT_STOPS
from ..errors import ConfigError, PreflightError
from ..reference import Instrument
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
    strategy's bar interval as OpenAlgo writes it ("5m"), and `intervals` those of every
    strategy of the session (a list is kept as a tuple): give either or both, and each
    interval named is checked once (`bar_intervals`). `product` is the order product.
    `capital`, `max_daily_loss` and `account_max_daily_loss` are rupees as the user gave
    them, None when not given: the preflight checks them, so each problem is reported
    with the others. `mode` is the mode the mode guard read when the session started
    (D21), and `alerts` the session's alert channel, None when it has none.
    `tested_alerts` is the alerts check of an earlier preflight of the same session,
    given when the preflight runs again while the broker's login is awaited: that check
    is reported again and no second test alert is sent. `contracts` are the session's
    futures and options as the reference tables resolved them, whose expiry is checked
    against OpenAlgo's master contract (`_preflight_expiry`; a list is kept as a tuple).
    `gtt` is True for a session with a CNC or NRML strategy, whose stops rest at the broker
    as GTTs: the GTT capability check runs (`_preflight_gtt`), with `positional_stops`, how
    the session holds those stops overnight ("gtt", the default, or "rearm_daily" when
    every CNC and NRML strategy of the session chose it).

    Raises ConfigError when `instruments` is empty or holds anything but pairs of
    non-blank strings, when neither `interval` nor `intervals` names an interval or
    `intervals` is not a list, when `mode` is not a `Mode`, and when `tested_alerts` is
    not a `Check` of the alert channel.
    """

    instruments: tuple[tuple[str, str], ...]
    interval: str | None = None
    intervals: tuple[str, ...] = ()
    product: str
    capital: float | None
    max_daily_loss: float | None
    account_max_daily_loss: float | None = None
    mode: Mode
    alerts: AlertsLike | None = None
    tested_alerts: Check | None = None
    contracts: tuple[Instrument, ...] = ()
    gtt: bool = False
    positional_stops: str = GTT_STOPS

    def __post_init__(self) -> None:
        object.__setattr__(self, "instruments", _instrument_pairs(self.instruments))
        object.__setattr__(self, "contracts", tuple(self.contracts))
        object.__setattr__(self, "intervals", _intervals(self.interval, self.intervals))
        if not isinstance(self.mode, Mode):
            raise ConfigError(
                f"mode is the mode the mode guard read at the session's start, Mode.SANDBOX "
                f"or Mode.LIVE, not {self.mode!r}"
            )
        tested = self.tested_alerts
        if tested is not None and not (isinstance(tested, Check) and tested.name == "alerts"):
            raise ConfigError(
                "tested_alerts is the alerts check of an earlier preflight of this session "
                f"(the Check named 'alerts' in its outcome), not {tested!r}"
            )

    @property
    def bar_intervals(self) -> tuple[Any, ...]:
        """Every interval the plan names, `interval` first, each once, as given (the
        interval check says how to write one that is wrong)."""
        named = [] if self.interval is None else [self.interval]
        for interval in self.intervals:
            if interval not in named:
                named.append(interval)
        return tuple(named)


@dataclass(frozen=True)
class Check:
    """The outcome of one preflight check. `name` says which check it is ("funds",
    "symbol:NSE:SBIN"), `ok` whether it passed, and `message` what it found: for a
    failure what to fix, for a pass what was verified. A pass the user should hear about
    (no alert channel in SANDBOX mode, say) has a message that starts with `WARNING`.
    `login` marks a failure that logging in to the broker in OpenAlgo may fix: OpenAlgo
    refuses the API key of a session it has ended, and some brokers answer an expired
    login with no funds or no quote."""

    name: str
    ok: bool
    message: str
    login: bool = False


@dataclass(frozen=True)
class InstrumentFacts:
    """What OpenAlgo's master contract gives of an instrument: `lot_size` in units (1 for
    cash equity), `tick_size` in rupees, `freeze_quantity`, the most units one order
    may carry on the exchange, None when OpenAlgo knows no such limit for it, and
    `expiry`, a contract's expiry as the answer wrote it ("29-DEC-26"; "" or None for
    cash equity, or an answer without one)."""

    symbol: str
    exchange: str
    lot_size: int
    tick_size: float
    freeze_quantity: int | None
    expiry: Any = None


@dataclass(frozen=True)
class PreflightReport:
    """What a passed preflight found: every check in the order it ran, the `broker`
    OpenAlgo is logged in to (as `ping` names it), `funds_available`, the available cash
    in rupees OpenAlgo reported (the sandbox's own funds in SANDBOX mode), the facts
    of each instrument, in the plan's order, and the later days on which OpenAlgo's
    holidays and the package calendar disagree, in words (`calendar_disagrees`,
    `_preflight_calendar`)."""

    checks: tuple[Check, ...]
    broker: str
    funds_available: float
    instruments: tuple[InstrumentFacts, ...]
    calendar_disagrees: tuple[str, ...] = ()

    @property
    def warnings(self) -> tuple[str, ...]:
        """The messages of the checks that passed with a warning, in order."""
        return tuple(check.message for check in self.checks if check.message.startswith(WARNING))


@dataclass(frozen=True)
class PreflightOutcome:
    """Every check of one preflight run in the order it ran, passed or failed, with what
    the server checks found: the `broker` ping named and the available cash in rupees
    (each None when not answered), the facts of each instrument found, and the days on
    which OpenAlgo's holidays and the package calendar disagree, in words."""

    checks: tuple[Check, ...]
    broker: str | None
    funds_available: float | None
    instruments: tuple[InstrumentFacts, ...]
    calendar_disagrees: tuple[str, ...] = ()

    @property
    def failures(self) -> tuple[str, ...]:
        """The messages of the checks that failed, in order."""
        return tuple(check.message for check in self.checks if not check.ok)

    @property
    def waits_for_login(self) -> bool:
        """True when checks failed and each of them is a login failure (`Check.login`):
        a session started before the open runs the preflight again until the broker's
        login works, and stops at once on any other failure."""
        failed = [check for check in self.checks if not check.ok]
        return bool(failed) and all(check.login for check in failed)

    def check(self, name: str) -> Check | None:
        """The check named `name`, or None when it did not run."""
        return next((check for check in self.checks if check.name == name), None)

    def report(self) -> PreflightReport:
        """The report of a run in which every check passed. Raises PreflightError when
        any failed, its `failures` holding the message of each in order."""
        broker, available = self.broker, self.funds_available
        # A broker or funds missing fails the broker session check, so both are known
        # whenever every check passed.
        if self.failures or broker is None or available is None:
            raise PreflightError(_failed_text(self.failures), self.failures)
        return PreflightReport(
            self.checks, broker, available, self.instruments, self.calendar_disagrees
        )


def refused(failures: tuple[str, ...]) -> PreflightError:
    """The PreflightError of a start that `failures` (each a failed check's message)
    refuse."""
    return PreflightError(_failed_text(failures), failures)


def _failed_text(failures: tuple[str, ...]) -> str:
    count = f"{len(failures)} preflight check{'' if len(failures) == 1 else 's'} failed"
    lines = "\n".join(f"- {message}" for message in failures)
    return f"the live session cannot start: {count}. Fix each, then start it again:\n{lines}"


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


def _intervals(interval: Any, intervals: Any) -> tuple[Any, ...]:
    if not isinstance(intervals, list | tuple):
        raise ConfigError(
            "intervals is a list of the strategies' bar intervals as OpenAlgo writes them, "
            f"such as ['5m', '15m'], not {intervals!r}"
        )
    if interval is None and not intervals:
        raise ConfigError(
            "name the bar interval to check: interval='5m' for the strategy, or "
            "intervals=['5m', '15m'] for every strategy of the session"
        )
    return tuple(intervals)


__all__ = [
    "WARNING",
    "Check",
    "InstrumentFacts",
    "PreflightOutcome",
    "PreflightPlan",
    "PreflightReport",
    "positive_amount",
    "refused",
]
