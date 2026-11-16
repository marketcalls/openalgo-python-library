"""The start banner of a live session (16-session-design 3.1 step 13; unit 8): the lines
the operator reads before the countdown, the mode first. Amounts are rupees, times IST.
"""

from __future__ import annotations

from typing import Any

from .._calendar_convert import format_ist
from ..calendar import default_calendar
from ._session_trader import ist_instant
from ._types import Mode

#: The banner's lines a SESSION alert carries: the mode, the funds and the first
#: strategy's limits.
BANNER_ALERT_LINES = 4


def banner_lines(session: Any) -> list[str]:
    """The start banner of `session` (a `_session_loop.Session` whose strategies are
    set): the guard's mode line first ("OpenAlgo is in LIVE mode: orders go to
    zerodha"), then the funds, each strategy's instrument, interval, capital, daily loss
    limit, entry cut-off and square-off, each instrument's freeze cap and a warning that
    an entry without `stop=` has no stop at the broker (allowed, decided 2026-11-16), the
    account's limit, the cost model, the sandbox's difference, and the warnings."""
    parts = session.parts
    spec, report = parts.spec, parts.report
    lines = [parts.banner, f"Funds available: Rs {report.funds_available:,.2f}"]
    facts = {(fact.symbol, fact.exchange): fact for fact in report.instruments}
    calendar = default_calendar()
    for each, trader in zip(spec.strategies, session.traders, strict=True):
        times = calendar.intraday_times(each.exchange, spec.day)
        until = times.entry_until
        if each.plan.entry_until is not None:
            until = ist_instant(spec.day, each.plan.entry_until)
        lines.append(
            f"{each.name}: {each.exchange}:{each.symbol} every {each.interval}, capital "
            f"Rs {each.capital:,.2f}, daily loss limit Rs {each.max_daily_loss:,.2f}, "
            f"entries until {_clock(until)} IST, square-off {_clock(trader.square_off_ns)} IST"
        )
        fact = facts.get((each.symbol, each.exchange))
        freeze = None if fact is None else fact.freeze_quantity
        lines.append(f"{each.symbol}: freeze quantity {freeze if freeze else 'none'}")
        lines.append(
            f"Warning: {each.name}: an entry without stop= has no stop at the broker until a "
            "trail or breakeven sets one (allowed; the result names such positions)"
        )
    if spec.arguments.account_max_daily_loss is not None:
        limit = spec.arguments.account_max_daily_loss
        lines.append(f"Account daily loss limit: Rs {limit:,.2f}")
    costs = spec.strategies[0].plan.costs
    lines.append(f"Charges modelled with: {costs or 'no cost model'}")
    if parts.mode is Mode.SANDBOX:
        lines.append(
            "SANDBOX: OpenAlgo simulates the fills at its own prices; nothing reaches the broker"
        )
    lines += [f"Warning: {note}" for note in session.warnings]
    return lines


def _clock(ts_ns: int) -> str:
    """ "HH:MM" of an instant, IST."""
    return format_ist(ts_ns)[-9:-4]


__all__ = ["BANNER_ALERT_LINES", "banner_lines"]
