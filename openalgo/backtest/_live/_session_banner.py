"""The start banner of a live session (16-session-design 3.1 step 13; units 8 and 13): the
lines the operator reads before the countdown, the mode first. Amounts are rupees, times
IST.

Whether an entry carries a stop is known only when it is placed (`stop=` is an argument of
each `self.buy` or `self.sell`), so the banner states the rule once (`STOPS_LINE`), and the
session warns at the first position actually opened with no stop at the broker
(`stops.StopsSupervisor`), on the console and in the result (decided 2026-11-16: allowed,
with a warning).
"""

from __future__ import annotations

from typing import Any

from .._calendar_convert import format_ist
from ..calendar import default_calendar
from ._groups import shared_lines
from ._session_restore import restart_lines
from ._session_trader import ist_instant
from ._shared_alerts import warning_line
from ._types import Mode

#: The banner's lines a SESSION alert carries: the mode, the funds and the first
#: strategy's limits.
BANNER_ALERT_LINES = 4
#: The banner's line on broker-held stops (module docstring).
STOPS_LINE = (
    "Stops: an entry placed with stop= gets its stop at the broker as it fills; one placed "
    "without has none until a trail or breakeven sets one (allowed), and the session warns "
    "at the first such position"
)


def banner_lines(session: Any) -> list[str]:
    """The start banner of `session` (a `_session_loop.Session` whose strategies are
    set): the guard's mode line first ("OpenAlgo is in LIVE mode: orders go to
    zerodha"), at a restart the start's number and each resumed leg with its stop, then
    the funds, each strategy's instrument, interval, capital, daily loss limit, entry
    cut-off and square-off, each instrument's freeze cap (once), each instrument several
    strategies share with its members and the rule on trading by hand (18-multi-strategy
    4.11), the rule of broker-held stops (`STOPS_LINE`), the account's limit, the cost
    model, the sandbox's difference, and the warnings, each once with one prefix."""
    parts = session.parts
    spec, report = parts.spec, parts.report
    lines = [parts.banner, *restart_lines(session)]
    lines.append(f"Funds available: Rs {report.funds_available:,.2f}")
    facts = {(fact.symbol, fact.exchange): fact for fact in report.instruments}
    calendar = default_calendar()
    shown: set[tuple[str, str]] = set()
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
        if (each.symbol, each.exchange) not in shown:
            shown.add((each.symbol, each.exchange))
            fact = facts.get((each.symbol, each.exchange))
            freeze = None if fact is None else fact.freeze_quantity
            lines.append(f"{each.symbol}: freeze quantity {freeze if freeze else 'none'}")
    lines += shared_lines(session.groups.groups)
    lines.append(STOPS_LINE)
    if spec.arguments.account_max_daily_loss is not None:
        limit = spec.arguments.account_max_daily_loss
        lines.append(f"Account daily loss limit: Rs {limit:,.2f}")
    costs = spec.strategies[0].plan.costs
    lines.append(f"Charges modelled with: {costs or 'no cost model'}")
    if parts.mode is Mode.SANDBOX:
        lines.append(
            "SANDBOX: OpenAlgo simulates the fills at its own prices; nothing reaches the broker"
        )
    lines += [warning_line(note) for note in session.warnings]
    return lines


def _clock(ts_ns: int) -> str:
    """ "HH:MM" of an instant, IST."""
    return format_ist(ts_ns)[-9:-4]


__all__ = ["BANNER_ALERT_LINES", "STOPS_LINE", "banner_lines"]
