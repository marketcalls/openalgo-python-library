"""Resolving the settings of a run: the documented defaults, the strategy's class
attributes and the run arguments, into one `RunPlan` the kernel receives explicitly.

Precedence (lead ruling): a strategy class attribute wins over the run argument of the
same name, which wins over the documented default. The documented defaults are those of
03-architecture 4.1 and 04-execution 2: product MIS for intraday intervals, CNC for daily
equity and NRML for daily derivatives; entry window and square-off from the calendar's
dated table (MIS only, SPEC-32); fill at the next open; whole shares and lots (the venue
rule, decision D26); Zerodha charges; the per-segment slippage. Every value used is
listed in the Assumptions block.

The plan is `_run_plan.RunPlan`; the run's dates are read by `_run_dates` and its risk
settings, in the units of 14-strategy-api, by `_risk_settings`, which refuses the combined
rules of a run until Phase 3 builds them; no other setting is accepted and then ignored
(`_setting_checks`). The futures margin, `margin=`, is read by `_futures_margin`.
"""

from __future__ import annotations

from typing import Any

from . import slippage as slippage_module
from ._futures import check_roll
from ._futures_margin import parse_margin
from ._intraday_assumptions import TIMES as INTRADAY_TIMES
from ._intraday_assumptions import intraday_times
from ._layout import parse_interval
from ._perturbation import FILL_MODELS, delay, delay_assumption
from ._risk_settings import choice, refuse_combined_rules, risk_settings
from ._run_dates import instant, run_dates
from ._run_plan import RunPlan
from ._scale_in import describe as describe_scale_limits
from ._setting_checks import class_leverage, refuse_unusable, unused_settings
from ._size_controls import describe as describe_size_controls
from ._sizing_assumption import sizing_assumption
from .data._sessions import resolve_now
from .errors import ConfigError

DERIVATIVE_EXCHANGES = ("NFO", "BFO", "MCX", "CDS", "BCD")
FILL_MODES = ("next_open", "close", *FILL_MODELS)
QUANTITY_MODES = ("fractional", "venue")
# Run settings a strategy class may override (lead ruling on several strategies).
CLASS_OVERRIDES = (
    "max_daily_loss",
    "entry_from",
    "entry_until",
    "square_off",
    "drawdown_halt",
    "cooldown",
)


def _slippage(value: Any) -> tuple[str, float, int]:
    """The run-wide slippage form; a table (the default) is resolved per instrument
    by `_market`. Raises ConfigError for an unknown form."""
    resolved = slippage_module.resolve(value)
    if isinstance(resolved, slippage_module.SlippageTable):
        return ("segment_default", 0.0, 0)
    return resolved.to_native()


def _product(strategy: type, product: str | None, interval: str, exchange: str) -> str:
    chosen = strategy.product or product
    if chosen is not None:
        return choice(chosen, ("CNC", "MIS", "NRML"), "product")
    if parse_interval(interval)[0] == "intraday":
        return "MIS"
    return "NRML" if exchange in DERIVATIVE_EXCHANGES else "CNC"


def _base(base: str | None, interval: str) -> str:
    """The base interval before data is seen: `base=`, daily for daily or longer
    strategies, else the strategy interval, which `_base_interval.choose_base` replaces
    with 1 minute when 1-minute data covers the run (03-architecture 4.1)."""
    if base is not None:
        parse_interval(base)
        return base
    if parse_interval(interval)[0] != "intraday":
        return "D"
    return interval


def resolve(strategy: type, args: dict[str, Any]) -> RunPlan:
    """The plan of a run of `strategy` with the run arguments `args` (the keyword
    arguments of `bt.backtest`). Raises ConfigError for unusable values and
    NotImplementedError for settings of later work."""
    symbols = strategy.traded_symbols()
    if not strategy.exchange or not strategy.interval:
        raise ConfigError(f"{strategy.__name__}: set exchange and interval on the class")
    parse_interval(strategy.interval)
    capital = args["capital"]
    if isinstance(capital, bool) or not isinstance(capital, int | float) or capital <= 0:
        raise ConfigError("capital must be a positive number of rupees")
    refuse_combined_rules(strategy)
    refuse_unusable(strategy, args)
    warnings: list[str] = []
    if args["costs"] is None:
        warnings.append("costs=None: no charges are modelled; results overstate P&L")
    overrides = {
        name: getattr(strategy, name) if getattr(strategy, name) is not None else args[name]
        for name in CLASS_OVERRIDES
    }
    start, end = run_dates(strategy, args)
    product = _product(strategy, args["product"], strategy.interval, strategy.exchange)
    warnings += unused_settings(strategy, args, product, overrides)
    return RunPlan(
        symbols=symbols,
        exchange=strategy.exchange,
        interval=strategy.interval,
        base=_base(args["base"], strategy.interval),
        product=product,
        start=start,
        end=end,
        first_slice=instant(args["start"]),
        last_slice=instant(args["end"]),
        present=resolve_now(args["now"]),
        capital=float(capital),
        fill=choice(args["fill"], FILL_MODES, "fill"),
        quantity_mode=choice(args["quantity_mode"], QUANTITY_MODES, "quantity_mode"),
        costs=args["costs"],
        slippage=_slippage(args["slippage"]),
        risk=risk_settings(strategy, args, overrides),
        entry_from=overrides["entry_from"],
        entry_until=overrides["entry_until"],
        square_off=overrides["square_off"],
        leverage=class_leverage(strategy),
        warnings=tuple(warnings),
        roll=check_roll(args["roll"]),
        margin=parse_margin(args["margin"]),
        delay_entries=delay(args.get("delay_entries", 0), "delay_entries"),
        delay_exits=delay(args.get("delay_exits", 0), "delay_exits"),
    )


def assumptions(plan: RunPlan, calendar: Any) -> dict[str, Any]:
    """The Assumptions block: what the run used. Entry window and square-off apply to
    MIS only (SPEC-32); they are stated as they applied over the whole run
    (`_intraday_assumptions`)."""
    block: dict[str, Any] = {"base": plan.base, "product": plan.product}
    times: dict[str, str | None] = dict.fromkeys(INTRADAY_TIMES)
    if plan.product == "MIS":
        given = {name: getattr(plan, name) for name in INTRADAY_TIMES}
        times = intraday_times(calendar, plan.exchange, plan.start, plan.end, given)
    block.update(times)
    block.update(
        fill=plan.fill,
        quantity_mode=plan.quantity_mode,
        sizing=sizing_assumption(plan.quantity_mode),
        costs=plan.costs,
        slippage=plan.slippage[0],
        capital=plan.capital,
    )
    block.update(_risk_assumptions(plan))
    block.update(delay_assumption(plan))
    return block


def _risk_assumptions(plan: RunPlan) -> dict[str, Any]:
    """The strategy and session rules of the run, and where their exits fill: at the
    next base-bar open, or on a daily base at the close of the day that breached
    (SPEC-17)."""
    risk = plan.risk
    block = {
        "max_daily_loss": risk["max_daily_loss"],
        "risk_basis": risk["risk_basis"],
        "after_risk_exit": risk["after_risk_exit"],
        "drawdown_halt": risk["drawdown_halt"],
        "cooldown": risk["cooldown_bars"],
        "max_positions": risk["max_positions"],
        "max_hold": risk["max_hold_bars"],
        "leverage": plan.leverage,
        **describe_scale_limits(risk),
        **describe_size_controls(risk),
    }
    if risk["max_daily_loss"] is not None or risk["drawdown_halt"] is not None:
        block["risk_exit_fill"] = (
            "at the close of the day that breached, flagged"
            if plan.base == "D"
            else "at the next base-bar open"
        )
    return block


__all__ = ["CLASS_OVERRIDES", "RunPlan", "assumptions", "resolve"]
