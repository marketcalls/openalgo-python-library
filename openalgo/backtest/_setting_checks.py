"""No setting of a run is accepted and then ignored (the audit of the strategy's class
attributes and the arguments of `bt.backtest`).

Every setting either works, raises NotImplementedError naming the feature, raises
ConfigError when it cannot be used, or, when it is used only by some orders or some
strategies, is named in the result's warnings with the reason it did nothing here:

* `leverage` (class attribute): the default of every order, overridden by `leverage=`
  on an order (14-strategy-api 2.4); a number of 1 or more.
* `state`: the strategy's own dict, `self.state`, filled while the run goes; a class
  attribute of that name would be replaced by an empty dict, so it is refused.
* `after_risk_exit`: governs only the exits of the combined rules (14-strategy-api 3),
  which are planned for Phase 3, so `"allow_reentry"` is not built yet.
* `pyramiding` on a strategy with `on_bar`: read by condition strategies only; an
  `on_bar` strategy places its own adds, which the scale-in limits cap.
* `client` with `data=`: the bars come from `data=`, so the client is not used.
* `entry_from`, `entry_until`, `square_off` on a run whose product is not MIS: they
  apply to MIS orders only (SPEC-32).
* `margin=` on a run that trades no futures, or naming underlyings it does not trade as
  futures: it sets the margin of futures only (reported by `_futures_margin`).
"""

from __future__ import annotations

from typing import Any

from ._messages import not_supported
from .errors import ConfigError

INTRADAY_TIMES = ("entry_from", "entry_until", "square_off")
# The values of `after_risk_exit` that only the combined rules (Phase 3) would act on.
AFTER_RISK_EXIT_NOT_BUILT = ("allow_reentry",)
INTRADAY_PRODUCT = "MIS"


def class_leverage(strategy: type) -> float | None:
    """The strategy's `leverage` class attribute as a number, or None. Raises
    ConfigError for anything but a number of 1 or more."""
    value = getattr(strategy, "leverage", None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not value >= 1:
        raise ConfigError(
            f"leverage takes a number of 1 or more (5 asks for 5x intraday leverage), got {value!r}"
        )
    return float(value)


def refuse_unusable(strategy: type, args: dict[str, Any]) -> None:
    """Raise for a setting that could only be ignored: ConfigError for a class attribute
    `state`, NotImplementedError for `after_risk_exit="allow_reentry"`."""
    if getattr(strategy, "state", None) is not None:
        raise ConfigError(
            f"{strategy.__name__}: state is the strategy's own dict (self.state), filled in "
            "init or on_bar; remove the class attribute state"
        )
    if args["after_risk_exit"] in AFTER_RISK_EXIT_NOT_BUILT:
        raise not_supported(
            f"after_risk_exit={args['after_risk_exit']!r}, which governs only the exits of "
            "the combined rules (overall_sl, lock_profit)",
            "leave after_risk_exit at its default; the daily loss limit and the drawdown "
            "halt have their own rules",
        )


def unused_settings(
    strategy: type, args: dict[str, Any], product: str, times: dict[str, str | None]
) -> list[str]:
    """Warnings for the settings this run does not use, each with the reason. `times`
    holds the resolved `entry_from`, `entry_until` and `square_off`."""
    from .strategy import Strategy

    notes = []
    if strategy.pyramiding and strategy.on_bar is not Strategy.on_bar:
        notes.append(
            f"pyramiding={strategy.pyramiding!r} is read by condition strategies only "
            "(entry_when, short_when); this strategy's on_bar places its own adds, which "
            "the scale-in limits cap (max_adds, max_position, min_spacing, max_risk)"
        )
    if args.get("client") is not None and args["data"] is not None:
        notes.append("client= is not used: the bars come from data=")
    if product != INTRADAY_PRODUCT:
        for name in INTRADAY_TIMES:
            value = times[name]
            if value is not None:
                notes.append(
                    f"{name}={value!r} applies to MIS orders only; this run's "
                    f"product is {product}, so it applies only to orders placed with "
                    "product='MIS'"
                )
    return notes


__all__ = ["class_leverage", "refuse_unusable", "unused_settings"]
