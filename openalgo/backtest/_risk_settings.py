"""The risk settings of a run, in the units of 14-strategy-api: `max_daily_loss` in rupees
(a number or `bt.rupees`), `drawdown_halt` in percent (`bt.pct(x)` or a number of
percent), `cooldown`, `max_hold` and `max_positions` as whole numbers (base bars,
positions), the scale-in limits as `_scale_in` reads them, and `reinvest` and
`size_caps` as `_size_controls` reads them. The daily limits of 18-multi-strategy 2.5a:
`max_trades_per_day` a whole number of entries a session day, `daily_profit_target` and
`max_order_value` rupees (a number or `bt.rupees`); each off when None. How a CNC or NRML
position's stop is held, `positional_stops` (design 20-positional-live section 6): "gtt"
(the default, one SINGLE GTT resting overnight) or "rearm_daily" (a DAY stop order placed
again at each session's open), whose run carries one warning for the trader
(`rearm_warning`). The combined rules of a run are refused until Phase 3 builds them.
"""

from __future__ import annotations

from typing import Any

from ._messages import not_supported
from ._scale_in import scale_limits
from ._size_controls import size_controls
from .errors import ConfigError
from .units import Distance

#: The risk bases of the daily loss limit: P&L before charges ("gross") or after them
#: ("net").
GROSS_BASIS = "gross"
NET_BASIS = "net"
RISK_BASES = (GROSS_BASIS, NET_BASIS)
AFTER_RISK_EXIT = ("lock_session", "allow_reentry")
#: How a CNC or NRML position's stop is held: one SINGLE GTT (the default), or a DAY stop
#: order placed again at each session's open (rust/oa_bt/src/engine/rearm.rs).
GTT_STOPS = "gtt"
REARM_DAILY = "rearm_daily"
POSITIONAL_STOPS = (GTT_STOPS, REARM_DAILY)
# Combined rules of a run, class attributes for strategies that open single positions
# (14-strategy-api 2.1); refused until Phase 3 builds them.
COMBINED_RULES = ("combined_stop", "combined_target", "lock_profit", "trail_sl_to_entry")


def choice(value: Any, allowed: tuple[str, ...], name: str) -> str:
    """`value` when it is one of `allowed`. Raises ConfigError otherwise."""
    if value not in allowed:
        raise ConfigError(f"{name} takes one of {allowed}, got {value!r}")
    return value


def rupees_amount(value: Any, name: str) -> float | None:
    """None, or a positive amount of rupees (a number or bt.rupees(x)). Raises
    ConfigError otherwise."""
    if value is None:
        return None
    if isinstance(value, Distance) and value.unit == "rupees":
        return value.value
    if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
        raise ConfigError(f"{name} takes a positive amount of rupees, got {value!r}")
    return float(value)


def _percent(value: Any, name: str) -> float | None:
    """None, or a percent above 0 and below 100 (bt.pct(x) or a bare number of percent)."""
    if value is None:
        return None
    if isinstance(value, Distance) and value.unit == "pct":
        value = value.value
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value < 100:
        raise ConfigError(f"{name} takes bt.pct(x) with x above 0 and below 100, got {value!r}")
    return float(value)


def _count(value: Any, name: str, unit: str) -> int | None:
    """None, or a positive whole number of `unit`."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{name} takes a positive whole number of {unit}, got {value!r}")
    return value


def _bars(value: Any, name: str) -> int | None:
    """None, or a positive whole number of base bars. A time ("30m") is not built yet."""
    if isinstance(value, str):
        raise not_supported(
            f"{name} as a time ({value!r})", f"give {name} as a whole number of base bars"
        )
    return _count(value, name, "base bars")


def rearm_warning(product: str) -> str:
    """The warning of a run with `positional_stops="rearm_daily"`, for a trader, whose
    product is `product`."""
    note = (
        "positional_stops='rearm_daily': each CNC or NRML position's stop is a DAY stop "
        "order placed again at every session's open, and nothing protects the position from "
        "a session's close to the next open. This backtest assumes the engine was started "
        "before the open on every trading day a position was held; live, start it before "
        "the open on each of those days"
    )
    if product == "MIS":
        note += "; this run's product is MIS, so it applies only to orders placed as CNC or NRML"
    return note


def refuse_combined_rules(strategy: type) -> None:
    """Combined rules of a run (05-risk-management 4) are Phase 3; never ignore them.
    Raises NotImplementedError for one the strategy declares."""
    for name in COMBINED_RULES:
        if getattr(strategy, name, None) not in (None, False):
            raise not_supported(
                f"{name} (a combined rule over the run's positions)",
                "give each order its own stop=, target= or trail=, and the run max_daily_loss=",
            )


def risk_settings(
    strategy: type, args: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """The risk settings of a run of `strategy` (module docstring), from the run
    arguments `args` and the class attributes that override them (`overrides`). Raises
    ConfigError for unusable values and NotImplementedError for a time given as a count
    of bars."""
    return {
        "max_daily_loss": rupees_amount(overrides["max_daily_loss"], "max_daily_loss"),
        "max_positions": _count(args["max_positions"], "max_positions", "positions"),
        "drawdown_halt": _percent(overrides["drawdown_halt"], "drawdown_halt"),
        "cooldown_bars": _bars(overrides["cooldown"], "cooldown"),
        "max_trades_per_day": _count(
            overrides["max_trades_per_day"], "max_trades_per_day", "entries a day"
        ),
        "daily_profit_target": rupees_amount(
            overrides["daily_profit_target"], "daily_profit_target"
        ),
        "max_order_value": rupees_amount(overrides["max_order_value"], "max_order_value"),
        "max_hold_bars": _bars(getattr(strategy, "max_hold", None), "max_hold"),
        "risk_basis": choice(args["risk_basis"], RISK_BASES, "risk_basis"),
        "after_risk_exit": choice(args["after_risk_exit"], AFTER_RISK_EXIT, "after_risk_exit"),
        "positional_stops": choice(
            args.get("positional_stops", GTT_STOPS), POSITIONAL_STOPS, "positional_stops"
        ),
        "round_levels": bool(args["round_levels"]),
        "strict_config": bool(args["strict_config"]),
        **scale_limits(strategy),
        **size_controls(strategy),
    }


__all__ = [
    "AFTER_RISK_EXIT",
    "COMBINED_RULES",
    "GROSS_BASIS",
    "GTT_STOPS",
    "NET_BASIS",
    "POSITIONAL_STOPS",
    "REARM_DAILY",
    "RISK_BASES",
    "choice",
    "rearm_warning",
    "refuse_combined_rules",
    "risk_settings",
    "rupees_amount",
]
