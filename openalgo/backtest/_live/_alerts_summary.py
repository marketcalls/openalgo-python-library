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
session did not reach OpenAlgo. Rupees are written as `Result.summary()` writes them
(Indian digit grouping, a loss as "Rs -500.00"). `alerts` re-exports both names.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from .._summary import _rupees
from ..report._format import MISSING
from ._types import Mode

# Rupee text has two decimals (paise); a slippage that rounds to zero there is none.
_PAISE_DECIMALS = 2


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
    `unprotected_seconds` the seconds a position had no stop at the broker, and
    `reconcile_mismatches` the reconciliations whose broker position differed from the
    engine's. `undelivered_alerts` counts the session's alerts that did not reach
    OpenAlgo, failed or dropped (`Alerts.undelivered`). `flat` is True when the session
    ended with no open position.
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

    @property
    def net_pnl(self) -> float:
        """Rupees: the gross P&L less the modelled charges."""
        return self.gross_pnl - self.charges


def summary_text(summary: SessionSummary) -> str:
    """The summary of the module docstring: a title line, then one line per figure with
    the colons aligned, the alerts line only when an alert was not delivered. Plain
    text, no emoji."""
    lines = [
        ("Strategies", ", ".join(summary.strategies) or "none"),
        ("Trades", f"{summary.closed_trades} closed"),
        ("Gross P&L", _rupees(summary.gross_pnl)),
        ("Charges", f"{_rupees(summary.charges)} (modelled)"),
        ("Net P&L", _rupees(summary.net_pnl)),
        ("Slippage", _slippage(summary.slippage)),
        ("Rejected", _count(summary.rejected_orders, "order")),
        ("Unprotected", f"{round(summary.unprotected_seconds)} s without a stop at the broker"),
        ("Reconcile", _count(summary.reconcile_mismatches, "mismatch", "mismatches")),
    ]
    if summary.undelivered_alerts:
        lines.append(("Alerts", f"{summary.undelivered_alerts} not delivered"))
    lines.append(("End state", "flat" if summary.flat else "not flat: a position is still open"))
    width = max(len(label) for label, _ in lines)
    title = (
        f"Session of {summary.day.isoformat()} (IST): {summary.mode.name} mode, "
        f"broker {summary.broker}"
    )
    return "\n".join([title, *(f"{label:<{width}} : {value}" for label, value in lines)])


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
