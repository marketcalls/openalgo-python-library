"""The start banner of a live session (16-session-design 3.1 step 13; units 8 and 13): the
lines the operator reads before the countdown, the mode first. Amounts are rupees, written
with Indian digit grouping as every report writes them (`report._format.indian_digits`),
times IST.

Whether an entry carries a stop is known only when it is placed (`stop=` is an argument of
each `self.buy` or `self.sell`), so the banner states the rule once (`STOPS_LINE`), and the
session warns at the first position actually opened with no stop at the broker
(`stops.StopsSupervisor`), on the console and in the result (decided 2026-11-16: allowed,
with a warning).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .._calendar_convert import format_ist
from ..calendar import default_calendar
from ..report._format import indian_digits
from ._groups import shared_lines
from ._preflight_gtt import SANDBOX_GTT_LINE
from ._session_restore import restart_lines
from ._session_trader import ist_instant
from ._shared_alerts import warning_line
from ._stops_gtt import overnight
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


#: A CNC or NRML strategy's square-off in its banner line: none (20-positional-live 7.3).
CARRIED_WORDS = "no square-off: the position is carried overnight"
#: The banner's line on a CNC or NRML position's stop (20-positional-live 2, P6).
POSITIONAL_STOPS_LINE = (
    "CNC and NRML: each position's stop rests at the broker as a SINGLE GTT, kept overnight; "
    "its targets are held by the session while it runs"
)
#: What Ctrl+C does with CNC and NRML positions, by `on_stop` (20-positional-live 10.2).
ON_STOP_LINES = {
    None: (
        "Ctrl+C: exits the MIS positions and keeps the CNC and NRML positions open with "
        "their GTT stops (on_stop default; on_stop='flatten' exits them all)"
    ),
    "flatten": "Ctrl+C: exits every position, CNC and NRML too (on_stop='flatten')",
    "keep": "Ctrl+C: keeps every position open with its stop at the broker (on_stop='keep')",
}


def banner_lines(session: Any) -> list[str]:
    """The start banner of `session` (a `_session_loop.Session` whose strategies are
    set): the guard's mode line first ("OpenAlgo is in LIVE mode: orders go to
    zerodha"), at a restart the start's number and each resumed leg with its stop, then
    the funds, each strategy's instrument, interval, capital, daily loss limit, entry
    cut-off and square-off, and its daily limits when set (18-multi-strategy 2.5a), each
    instrument's freeze cap (once), each strategy's version id, a version changed since
    the day's earlier start and a source not hashed (12.1, `_session_versions`), each
    instrument several strategies share with its
    members and the rule on trading by hand (4.11), the rule of broker-held stops
    (`STOPS_LINE`), with a CNC or NRML strategy how its stop is held and what Ctrl+C does
    (`positional_lines`), the account's limits, the order rate, the cost model, the sandbox's
    difference, and the warnings, each once with one prefix."""
    parts = session.parts
    spec, report = parts.spec, parts.report
    lines = [parts.banner, *restart_lines(session)]
    lines.append(f"Funds available: Rs {indian_digits(report.funds_available)}")
    facts = {(fact.symbol, fact.exchange): fact for fact in report.instruments}
    calendar = default_calendar()
    shown: set[tuple[str, str]] = set()
    for each, trader in zip(spec.strategies, session.traders, strict=True):
        times = calendar.intraday_times(each.exchange, spec.day)
        until = times.entry_until
        if each.plan.entry_until is not None:
            until = ist_instant(spec.day, each.plan.entry_until)
        # A CNC or NRML leg is carried, never squared off (20-positional-live 7.3).
        square_off = (
            CARRIED_WORDS if overnight(trader) else f"square-off {_clock(trader.square_off_ns)} IST"
        )
        lines.append(
            f"{each.name}: {each.exchange}:{each.symbol} every {each.interval}, capital "
            f"Rs {indian_digits(each.capital)}, daily loss limit "
            f"Rs {indian_digits(each.max_daily_loss)}, "
            f"entries until {_clock(until)} IST, {square_off}"
        )
        limits = daily_limit_words(each.plan.risk)
        if limits:
            lines.append(f"{each.name} daily limits: {limits}")
        if (each.symbol, each.exchange) not in shown:
            shown.add((each.symbol, each.exchange))
            fact = facts.get((each.symbol, each.exchange))
            freeze = None if fact is None else fact.freeze_quantity
            lines.append(f"{each.symbol}: freeze quantity {freeze if freeze else 'none'}")
    lines += session.versions.banner_lines()
    lines += shared_lines(session.groups.groups)
    lines.append(STOPS_LINE)
    lines += positional_lines(spec, parts.mode)
    if spec.arguments.account_max_daily_loss is not None:
        limit = spec.arguments.account_max_daily_loss
        lines.append(f"Account daily loss limit: Rs {indian_digits(limit)}")
    lines += account_limit_lines(spec.arguments)
    lines.append(
        f"Order rate: at most {spec.arguments.max_orders_per_second} order calls a second "
        "(place, modify and cancel together); stops and exits go first"
    )
    costs = spec.strategies[0].plan.costs
    lines.append(f"Charges modelled with: {costs or 'no cost model'}")
    if parts.mode is Mode.SANDBOX:
        lines.append(
            "SANDBOX: OpenAlgo simulates the fills at its own prices; nothing reaches the broker"
        )
    lines += [warning_line(note) for note in session.warnings]
    return lines


def positional_lines(spec: Any, mode: Mode) -> list[str]:
    """The banner's lines of a session with a CNC or NRML strategy (20-positional-live 2,
    5 and 10.2): how their stops are held, in `mode` SANDBOX that the live broker's GTT
    support is checked in LIVE mode only (`SANDBOX_GTT_LINE`), and what Ctrl+C does with
    their positions by `on_stop` (`ON_STOP_LINES`); none for an MIS-only session, whose
    banner is as before."""
    if not spec.gtt_legs:
        return []
    sandbox = [SANDBOX_GTT_LINE] if mode is Mode.SANDBOX else []
    return [POSITIONAL_STOPS_LINE, *sandbox, ON_STOP_LINES[spec.arguments.on_stop]]


def daily_limit_words(risk: Mapping[str, Any]) -> str:
    """A strategy's daily limits that are set, in words ("at most 3 trades a day, profit
    target Rs 5,000.00, orders of at most Rs 2,00,000.00 each"); "" for none
    (18-multi-strategy 2.5a). `risk` is its resolved risk settings (`RunPlan.risk`)."""
    words = []
    if risk["max_trades_per_day"] is not None:
        words.append(f"at most {risk['max_trades_per_day']} trades a day")
    if risk["daily_profit_target"] is not None:
        words.append(f"profit target Rs {indian_digits(risk['daily_profit_target'])}")
    if risk["max_order_value"] is not None:
        words.append(f"orders of at most Rs {indian_digits(risk['max_order_value'])} each")
    return ", ".join(words)


def account_limit_lines(arguments: Any) -> list[str]:
    """The banner's lines of the account's daily trade cap and profit target that are
    set (`_plan_args.LiveArguments`)."""
    lines = []
    if arguments.account_max_trades_per_day is not None:
        lines.append(f"Account trade cap: {arguments.account_max_trades_per_day} trades a day")
    if arguments.account_daily_profit_target is not None:
        target = arguments.account_daily_profit_target
        lines.append(f"Account daily profit target: Rs {indian_digits(target)}")
    return lines


def limit_lines(session: Any) -> tuple[str, ...]:
    """The daily limits the session set, in words, for its daily summary: each strategy's
    that are set, then the account's ("account trade cap 4 trades a day, daily profit
    target Rs 3,000.00"); none when it set none."""
    spec = session.parts.spec
    lines = [
        f"{each.name} {words}"
        for each in spec.strategies
        if (words := daily_limit_words(each.plan.risk))
    ]
    arguments = spec.arguments
    account = []
    if arguments.account_max_trades_per_day is not None:
        account.append(f"trade cap {arguments.account_max_trades_per_day} trades a day")
    if arguments.account_daily_profit_target is not None:
        target = arguments.account_daily_profit_target
        account.append(f"daily profit target Rs {indian_digits(target)}")
    if account:
        lines.append("account " + ", ".join(account))
    return tuple(lines)


def _clock(ts_ns: int) -> str:
    """ "HH:MM" of an instant, IST."""
    return format_ist(ts_ns)[-9:-4]


__all__ = [
    "BANNER_ALERT_LINES",
    "CARRIED_WORDS",
    "ON_STOP_LINES",
    "POSITIONAL_STOPS_LINE",
    "STOPS_LINE",
    "account_limit_lines",
    "banner_lines",
    "daily_limit_words",
    "limit_lines",
    "positional_lines",
]
