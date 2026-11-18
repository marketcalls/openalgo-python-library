"""What a run changes for a robustness check (design 17, sections 2.7, 2.12, 2.14 and
2.15), as `bt.backtest` reads it and the native config carries it to the kernel's
`Perturbation`: the fill models beside the next open (`fill=`), the strategy bars a
decided order is held (`delay_entries=`, `delay_exits=`), and the robustness reruns' own
settings, which no run argument sets (`RunPlan.bar_shift`, `RunPlan.exit_scales`).

An ordinary run changes nothing, so its native config, its manifest and its assumptions
are the ones it always had: each function here leaves out what is at its default.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .errors import ConfigError

#: The fill models besides the next open (design 17, section 2.15): another price of the
#: next base bar, and its average with the bar after it (the "_2" forms).
FILL_MODELS = (
    "next_close",
    "next_mid",
    "next_ohlc4",
    "next_close_2",
    "next_mid_2",
    "next_ohlc4_2",
)
#: Plan settings the robustness reruns set and no run argument does (`RunPlan.bar_shift`,
#: `RunPlan.exit_scales`).
RERUN_SETTINGS = ("bar_shift", "exit_scales")
#: The longest delay, in strategy bars: the kernel counts a delay in an unsigned 32-bit
#: whole number (`engine::config::Perturbation`).
LONGEST_DELAY = 2**32 - 1


def delay(value: Any, name: str) -> int:
    """A delay in strategy bars, the run argument `name`: a whole number from 0 to
    `LONGEST_DELAY`. Raises ConfigError for anything else, naming what it takes."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{name} takes a whole number of strategy bars, 0 or more, got {value!r}")
    if value > LONGEST_DELAY:
        raise ConfigError(
            f"{name} takes at most {LONGEST_DELAY} strategy bars, got {value}: a delay "
            "longer than the run's strategy bars holds every order past the run's last bar"
        )
    return value


def perturbation(plan: Any) -> dict[str, Any]:
    """What `plan` (a `RunPlan`) changes for a robustness check, as the native config's
    keys: none for an ordinary run, so its config is the one it always was."""
    changes = {
        "delay_entries": plan.delay_entries,
        "delay_exits": plan.delay_exits,
        "bar_shift": plan.bar_shift,
        "exit_scales": list(plan.exit_scales),
    }
    return {key: value for key, value in changes.items() if value}


def manifest_settings(plan: Any) -> dict[str, Any]:
    """Every setting of `plan` (a `RunPlan`), for the manifest's arguments, but the
    robustness reruns' own (`RERUN_SETTINGS`) at their defaults: no run argument sets
    them, so an ordinary run's manifest does not name them."""
    settings = {field.name: getattr(plan, field.name) for field in dataclasses.fields(plan)}
    for name in RERUN_SETTINGS:
        if not settings[name]:
            del settings[name]
    return settings


def delay_assumption(plan: Any) -> dict[str, str]:
    """The Assumptions line of a run that holds its decided orders, under "delay"; none
    for a run that does not, so every other run's block is as it was."""
    if not (plan.delay_entries or plan.delay_exits):
        return {}
    return {
        "delay": (
            f"entries held {plan.delay_entries} and exits {plan.delay_exits} strategy bars "
            "before they can fill; the engine's own exits and the legs' levels are not held"
        )
    }


__all__ = [
    "FILL_MODELS",
    "LONGEST_DELAY",
    "RERUN_SETTINGS",
    "delay",
    "delay_assumption",
    "manifest_settings",
    "perturbation",
]
