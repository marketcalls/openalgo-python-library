"""The daily summary of a live session (08-live-trading section 8), as plain text in the
style of `Result.summary()`:

    Session of 2026-11-16 (IST): SANDBOX mode, broker zerodha
    Strategies  : ema_cross, orb
    Trades      : 4 closed
    Gross P&L   : Rs 1,250.00
    Charges     : Rs 182.40 (modelled)
    Net P&L     : Rs 1,067.60
    Slippage    : Rs 35.00 worse than the backtest
    Rejected    : 1 order
    Unprotected : 0 s without a stop at the broker
    Reconcile   : 0 mismatches
    End state   : flat

An "Alerts : 2 not delivered" line comes before the end state when an alert of the
session did not reach OpenAlgo. A leg whose OpenAlgo books differed from the engine at the
close with no later read to confirm it (16-session-design 5.4 "As built", round 7) is
counted and named on the Reconcile and End state lines, "(at the close: s1 SBIN NSE -4
units in OpenAlgo's books, the engine's 0)" after "1 mismatch" and after "not flat: a
position is still open", or after "flat" when OpenAlgo's books show every such leg flat
(round 8: "s1 SBIN NSE no position in OpenAlgo's books, the engine's 10"). A leg no read
judged after a fill of its own orders landed during a read is named there too, as not
confirmed, and is no mismatch (round 9: "s1 SBIN NSE the engine's 0 units, not confirmed,
since ...").

Rupees are written as `Result.summary()` writes them (Indian digit grouping, a loss as
"Rs -500.00"). `alerts` re-exports both names.
"""

from __future__ import annotations

import datetime as dt
import math
import numbers
from dataclasses import dataclass

from .._summary import _rupees
from ..errors import ConfigError
from ..report._format import MISSING
from ._types import Mode

# Rupee text has two decimals (paise); a slippage that rounds to zero there is none.
_PAISE_DECIMALS = 2
# The fields that count something: whole numbers of at least zero.
_COUNTS = ("closed_trades", "rejected_orders", "reconcile_mismatches", "undelivered_alerts")


@dataclass(frozen=True)
class SessionSummary:
    """The figures of one live session's trading day.

    `day` is the IST date and `mode` the mode the session ran in; `broker` is the broker
    OpenAlgo traded through and `strategies` the session's strategy names.
    `closed_trades` counts the round trips closed in the session. `gross_pnl` and
    `charges` are rupees: the P&L of the session's fills before charges, and the charges
    the engine's cost model puts on them (the broker's trade book carries none).
    `slippage` is rupees the live fills cost beyond the fills the backtest models for
    the same orders (negative when live did better), or None when no backtest was
    compared. `rejected_orders` counts the orders the broker or OpenAlgo refused,
    `unprotected_seconds` the seconds a position had no stop at the broker (shown
    rounded up, so any exposure shows), and
    `reconcile_mismatches` the reconciliations whose broker position differed from the
    engine's. `undelivered_alerts` counts the session's alerts that did not reach
    OpenAlgo, failed or dropped (`Alerts.undelivered`). `flat` is True when the session
    ended with no open position. `differing` names, in words, each leg whose OpenAlgo
    books differed from the engine at the close with no later read to confirm it; each
    is a mismatch, and a position that may be open unless the words say OpenAlgo's books
    show it flat (a session can then end `flat` with it named). `unconfirmed` names, in
    words, each leg no read judged after a fill of its own orders landed during a read,
    with no difference to name: not a mismatch, named beside them.

    Raises ConfigError when a rupee figure or the seconds unprotected is not a finite
    number (None is allowed for `slippage` only), when the seconds unprotected are
    negative, when a count is not a whole number of at least zero, when a leg of
    `differing` or `unconfirmed` is not named in words, or when `differing` names more legs
    than the mismatches counted.
    """

    day: dt.date
    mode: Mode
    broker: str
    strategies: tuple[str, ...]
    closed_trades: int
    gross_pnl: float
    charges: float
    slippage: float | None
    rejected_orders: int
    unprotected_seconds: float
    reconcile_mismatches: int
    undelivered_alerts: int
    flat: bool
    differing: tuple[str, ...] = ()
    unconfirmed: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _check_figure("gross_pnl", self.gross_pnl)
        _check_figure("charges", self.charges)
        if self.slippage is not None:
            _check_figure("slippage", self.slippage)
        _check_figure("unprotected_seconds", self.unprotected_seconds)
        if self.unprotected_seconds < 0:
            raise ConfigError(
                f"the session summary's unprotected_seconds cannot be negative, got "
                f"{self.unprotected_seconds!r}: it counts seconds a position had no stop"
            )
        for name in _COUNTS:
            _check_count(name, getattr(self, name))
        _check_differing(self.differing, self.reconcile_mismatches)
        _check_legs("unconfirmed", self.unconfirmed)

    @property
    def net_pnl(self) -> float:
        """Rupees: the gross P&L less the modelled charges."""
        return self.gross_pnl - self.charges


def summary_text(summary: SessionSummary) -> str:
    """The summary of the module docstring: a title line, then one line per figure with
    the colons aligned, the alerts line only when an alert was not delivered. Plain
    text, no emoji."""
    named = (*summary.differing, *summary.unconfirmed)
    at_close = f" (at the close: {'; '.join(named)})" if named else ""
    lines = [
        ("Strategies", ", ".join(summary.strategies) or "none"),
        ("Trades", f"{summary.closed_trades} closed"),
        ("Gross P&L", _rupees(summary.gross_pnl)),
        ("Charges", f"{_rupees(summary.charges)} (modelled)"),
        ("Net P&L", _rupees(summary.net_pnl)),
        ("Slippage", _slippage(summary.slippage)),
        ("Rejected", _count(summary.rejected_orders, "order")),
        ("Unprotected", f"{math.ceil(summary.unprotected_seconds)} s without a stop at the broker"),
        (
            "Reconcile",
            _count(summary.reconcile_mismatches, "mismatch", "mismatches") + at_close,
        ),
    ]
    if summary.undelivered_alerts:
        lines.append(("Alerts", f"{summary.undelivered_alerts} not delivered"))
    end = ("flat" if summary.flat else "not flat: a position is still open") + at_close
    lines.append(("End state", end))
    width = max(len(label) for label, _ in lines)
    title = (
        f"Session of {summary.day.isoformat()} (IST): {summary.mode.name} mode, "
        f"broker {summary.broker}"
    )
    return "\n".join([title, *(f"{label:<{width}} : {value}" for label, value in lines)])


def _check_figure(name: str, value: object) -> None:
    """Raise ConfigError unless `value` is a finite number (a bool is not a figure)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ConfigError(
            f"the session summary's {name} must be a finite number, got {value!r}: a figure "
            f"the session could not measure is a fault to fix, not a total"
        )


def _check_count(name: str, value: object) -> None:
    """Raise ConfigError unless `value` is a whole number of at least zero."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ConfigError(
            f"the session summary's {name} must be a whole number of at least zero, got {value!r}"
        )


def _check_differing(differing: object, mismatches: int) -> None:
    """Raise ConfigError unless `differing` is a tuple of legs in words, each counted
    among the `mismatches`."""
    _check_legs("differing", differing)
    if len(differing) > mismatches:
        raise ConfigError(
            f"the session summary counts {mismatches} reconcile mismatches but names "
            f"{len(differing)} legs that differ at the close: it counts at least one "
            "mismatch for each leg"
        )


def _check_legs(name: str, legs: object) -> None:
    """Raise ConfigError unless `legs` is a tuple of legs, each named in words."""
    if not isinstance(legs, tuple) or not all(isinstance(words, str) and words for words in legs):
        raise ConfigError(f"the session summary's {name} names each leg in words, got {legs!r}")


def _slippage(slippage: float | None) -> str:
    if slippage is None:
        return f"{MISSING} (no backtest to compare with)"
    shown = round(slippage, _PAISE_DECIMALS)
    if shown == 0:
        return "none against the backtest"
    if shown > 0:
        return f"{_rupees(slippage)} worse than the backtest"
    return f"{_rupees(-slippage)} better than the backtest"


def _count(count: int, one: str, many: str | None = None) -> str:
    """ "1 order", "2 orders"; `many` for a plural that is not `one` plus "s"."""
    return f"{count} {one if count == 1 else many or one + 's'}"


__all__ = ["SessionSummary", "summary_text"]
