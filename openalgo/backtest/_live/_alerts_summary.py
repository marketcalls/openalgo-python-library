"""The daily summary of a live session (08-live-trading section 8), as plain text in the
style of `Result.summary()`:

    Session of 2026-11-16 (IST): SANDBOX mode, broker zerodha
    Strategies  : ema_cross (version 1f3a9c0b7d2e), orb (version 0c4d5e6f7a8b)
    Trades      : 4 closed
    Gross P&L   : Rs 1,250.00
    Charges     : Rs 182.40 (modelled)
    Net P&L     : Rs 1,067.60
    Slippage    : Rs 35.00 worse than the backtest
    Stop fills  : Rs 4.50 worse than the backtest
    Rejected    : 1 order
    Unprotected : 0 s without a stop at the broker
    Reconcile   : 0 mismatches
    End state   : flat

Each strategy is named with its version id (18-multi-strategy 12.1), when the session gave
them. The Charges line reads "Rs 182.40 (modelled per fill; Rs 181.00 on the contract
notes)" when the strategies' contract notes, which round STT and stamp duty to whole rupees,
charged another figure at the paisa (`notes_charges`, given at the day's first start only).
A "Limits : SbinTrend at most 3 trades a day; account trade cap 4 trades a day" line
follows the strategies when the session set daily limits (18-multi-strategy 2.5a). An
"Alerts : 2 not delivered" line comes before the end state when an alert of the
session did not reach OpenAlgo. The Slippage line adds "(1 MARKET fill not compared: no
bar of its decision)" when the session had no bar to set a MARKET fill against, and reads
"not compared (2 MARKET fills: no bar of their decisions)" when it had none for any. The
Stop fills line, the same for the fills of the stops (18-multi-strategy 12.3), is printed
only when a stop filled; a stop fill it leaves out is counted with "see fill_comparison",
whose row gives the reason (no bar reached its trigger, a repriced stop, a bar that opened
beyond a stop-limit's limit). The
Strategies line names after "; earlier today:" the
strategies an earlier start of the day ran and this one does not, whose figures the day's
include. A leg whose OpenAlgo books differed from the engine at the
close with no later read to confirm it (16-session-design 5.4 "As built", round 7) is
counted and named on the Reconcile and End state lines, "(at the close: s1 SBIN NSE -4
units in OpenAlgo's books, the engine's 0)" after "1 mismatch" and after "not flat: a
position is still open", or after "flat" when OpenAlgo's books show every such leg flat
(round 8: "s1 SBIN NSE no position in OpenAlgo's books, the engine's 10"). A leg no read
judged after a fill of its own orders landed during a read is named there too, as not
confirmed, and is no mismatch (round 9: "s1 SBIN NSE the engine's 0 units, not confirmed,
since ..."). An instrument several strategies of the session share has a "Shared" line
after the Reconcile line (18-multi-strategy 4.11): "SBIN NSE (A and B): OpenAlgo's net 90
by the last read, their sum 90: they agree"; a session with none prints none. A CNC or NRML
leg the session leaves open has a "Carried" line before the end state (20-positional-live
4.1): "CncHold SBIN NSE 10 units entered on 2026-11-16 at 500.00, stop 497.00: stop GTT
261116000001 confirmed at the broker, trigger 497.00 for 10 units".

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
from ._types import Mode

# Rupee text has two decimals (paise); a slippage that rounds to zero there is none.
_PAISE_DECIMALS = 2
# The fields that count something: whole numbers of at least zero.
_COUNTS = (
    "closed_trades",
    "rejected_orders",
    "reconcile_mismatches",
    "undelivered_alerts",
    "slippage_not_compared",
    "stop_slippage_not_compared",
)


@dataclass(frozen=True)
class SessionSummary:
    """The figures of one live session's trading day.

    `day` is the IST date and `mode` the mode the session ran in; `broker` is the broker
    OpenAlgo traded through and `strategies` the session's strategy names.
    `closed_trades` counts the round trips closed in the session. `gross_pnl` and
    `charges` are rupees: the P&L of the session's fills before charges, and the charges
    the engine's cost model puts on them (the broker's trade book carries none), each
    fill's rounded to the paisa, so they add up across the day's restarts. A contract note
    rounds its lines once, so a strategy's `bt.Result` charges its fills the notes'
    charges: they differ from these by the notes' rounding, which
    `Result.contract_notes` names (its `adjustment` column). `notes_charges` is rupees
    the strategies' contract notes charged, what their results charge, given when their
    results cover the day's fills (the day's first start) and None otherwise: after a
    restart the results cover their start's fills only, while these figures cover the
    day's.
    `slippage` is rupees the live MARKET fills cost beyond the fills the backtest models
    for the same orders, each at the open of the bar that opens at its decision with the
    run's modelled slippage (negative when live did better), or None when not one of the
    day's MARKET fills could be compared; `slippage_not_compared` counts the MARKET fills
    it could not set against a bar; `stop_slippage` and `stop_slippage_not_compared` are
    the same for the fills of the legs' stops, each at its trigger (or its bar's open when
    the bar opened through it) with the run's slippage, `stop_slippage` None when no
    stop's fill was compared. `rejected_orders` counts the orders the broker or
    OpenAlgo refused, `unprotected_seconds` the seconds a position had no stop at the
    broker (shown rounded up, so any exposure shows), and
    `reconcile_mismatches` the reconciliations whose broker position differed from the
    engine's. `undelivered_alerts` counts the session's alerts that did not reach
    OpenAlgo, failed or dropped (`Alerts.undelivered`). `flat` is True when the session
    ended with no open position. `differing` names, in words, each leg whose OpenAlgo
    books differed from the engine at the close with no later read to confirm it; each
    is a mismatch, and a position that may be open unless the words say OpenAlgo's books
    show it flat (a session can then end `flat` with it named). `unconfirmed` names, in
    words, each leg no read judged after a fill of its own orders landed during a read,
    with no difference to name: not a mismatch, named beside them. `earlier` names, in
    words, each strategy an earlier start of the day ran and this one does not, whose
    figures the day's include (with the position its last save held, when it held one).
    `shared` gives, in words, each instrument several strategies of the session share:
    OpenAlgo's net, the members' sum and whether they agree. `limits` names, in words,
    the daily limits the session set (each strategy's and the account's), none when it
    set none. `versions` are the strategies' version ids, in the order of `strategies`
    (none: not given). `carried` names, in words, each CNC or NRML leg the session leaves
    open with its stop GTT and its status (`_carry_close`), none when it carries none.

    Raises ConfigError when a rupee figure or the seconds unprotected is not a finite
    number (None is allowed for `slippage`, `stop_slippage` and `notes_charges`), when
    the seconds unprotected are negative, when a count is not a whole number of at least
    zero, when a leg of `differing`, `unconfirmed`, `shared` or `limits` is not named in
    words, when `differing` names more legs than the mismatches counted, or when
    `versions` are given and are not one id in words per strategy.
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
    earlier: tuple[str, ...] = ()
    slippage_not_compared: int = 0
    shared: tuple[str, ...] = ()
    limits: tuple[str, ...] = ()
    versions: tuple[str, ...] = ()
    stop_slippage: float | None = None
    stop_slippage_not_compared: int = 0
    notes_charges: float | None = None
    carried: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _check_figure("gross_pnl", self.gross_pnl)
        _check_figure("charges", self.charges)
        if self.notes_charges is not None:
            _check_figure("notes_charges", self.notes_charges)
        if self.slippage is not None:
            _check_figure("slippage", self.slippage)
        if self.stop_slippage is not None:
            _check_figure("stop_slippage", self.stop_slippage)
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
        _check_legs("earlier", self.earlier)
        _check_legs("shared", self.shared)
        _check_legs("limits", self.limits)
        _check_legs("versions", self.versions)
        _check_legs("carried", self.carried)
        if self.versions and len(self.versions) != len(self.strategies):
            raise ConfigError(
                f"the session summary names {len(self.strategies)} strategies and gives "
                f"{len(self.versions)} version ids: give one per strategy, or none"
            )

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
    earlier = f"; earlier today: {'; '.join(summary.earlier)}" if summary.earlier else ""
    lines = [("Strategies", (", ".join(_named(summary)) or "none") + earlier)]
    if summary.limits:
        lines.append(("Limits", "; ".join(summary.limits)))
    lines += [
        ("Trades", f"{summary.closed_trades} closed"),
        ("Gross P&L", _rupees(summary.gross_pnl)),
        ("Charges", _charges(summary)),
        ("Net P&L", _rupees(summary.net_pnl)),
        ("Slippage", _slippage(summary.slippage, summary.slippage_not_compared)),
        *_stop_line(summary),
        ("Rejected", _count(summary.rejected_orders, "order")),
        ("Unprotected", f"{math.ceil(summary.unprotected_seconds)} s without a stop at the broker"),
        (
            "Reconcile",
            _count(summary.reconcile_mismatches, "mismatch", "mismatches") + at_close,
        ),
        *(("Shared", line) for line in summary.shared),
    ]
    if summary.undelivered_alerts:
        lines.append(("Alerts", f"{summary.undelivered_alerts} not delivered"))
    lines += [("Carried", line) for line in summary.carried]
    end = ("flat" if summary.flat else "not flat: a position is still open") + at_close
    lines.append(("End state", end))
    width = max(len(label) for label, _ in lines)
    title = (
        f"Session of {summary.day.isoformat()} (IST): {summary.mode.name} mode, "
        f"broker {summary.broker}"
    )
    return "\n".join([title, *(f"{label:<{width}} : {value}" for label, value in lines)])


def _charges(summary: SessionSummary) -> str:
    """The Charges line (module docstring): the modelled charges, and the contract notes'
    beside them when the summary has them and they differ at the paisa."""
    modelled = _rupees(summary.charges)
    notes = summary.notes_charges
    if notes is None or _rupees(notes) == modelled:
        return f"{modelled} (modelled)"
    return f"{modelled} (modelled per fill; {_rupees(notes)} on the contract notes)"


def _named(summary: SessionSummary) -> list[str]:
    """The strategies as the Strategies line names them: each with its version id, when
    the summary has them."""
    if not summary.versions:
        return list(summary.strategies)
    pairs = zip(summary.strategies, summary.versions, strict=True)
    return [f"{name} (version {version})" for name, version in pairs]


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


def _slippage(slippage: float | None, not_compared: int) -> str:
    """The Slippage line (module docstring): the rupees against the backtest, or "not
    compared" when no fill was, with the note of the MARKET fills left out."""
    if slippage is None:
        return "not compared" + _not_compared(not_compared, words="")
    shown = round(slippage, _PAISE_DECIMALS)
    if shown == 0:
        figure = "none against the backtest"
    elif shown > 0:
        figure = f"{_rupees(slippage)} worse than the backtest"
    else:
        figure = f"{_rupees(-slippage)} better than the backtest"
    return figure + _not_compared(not_compared, words=" not compared")


# Where the reason a stop fill was not compared is given: each row of
# `SessionResult.fill_comparison` names its own, and they differ from fill to fill.
_SEE_ROWS = "see fill_comparison"


def _stop_line(summary: SessionSummary) -> list[tuple[str, str]]:
    """The Stop fills line (module docstring), none when no stop filled."""
    missed = summary.stop_slippage_not_compared
    if summary.stop_slippage is None and not missed:
        return []
    if summary.stop_slippage is None:
        return [("Stop fills", f"not compared ({_count(missed, 'stop fill')}: {_SEE_ROWS})")]
    shown = round(summary.stop_slippage, _PAISE_DECIMALS)
    if shown == 0:
        figure = "none against the backtest"
    elif shown > 0:
        figure = f"{_rupees(summary.stop_slippage)} worse than the backtest"
    else:
        figure = f"{_rupees(-summary.stop_slippage)} better than the backtest"
    note = f" ({_count(missed, 'stop fill')} not compared: {_SEE_ROWS})" if missed else ""
    return [("Stop fills", figure + note)]


def _not_compared(count: int, *, words: str) -> str:
    """The note of the MARKET fills the slippage leaves out, `words` after their count;
    "" for none."""
    if not count:
        return ""
    decisions = "its decision" if count == 1 else "their decisions"
    return f" ({_count(count, 'MARKET fill')}{words}: no bar of {decisions})"


def _count(count: int, one: str, many: str | None = None) -> str:
    """ "1 order", "2 orders"; `many` for a plural that is not `one` plus "s"."""
    return f"{count} {one if count == 1 else many or one + 's'}"


__all__ = ["SessionSummary", "summary_text"]
