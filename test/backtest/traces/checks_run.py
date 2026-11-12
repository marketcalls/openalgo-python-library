"""Consistency rules for the run-level parts of a case: defaults, assumptions, stale marks.

Every rule takes a case and returns its violations; each violation names the fixture,
the slice (where there is one) and the rule that is broken.
"""

from __future__ import annotations

from typing import Any

from . import ledger
from .bars import bars_by_slice, rupees
from .timegrid import Case, interval_step, parse_ist

Violations = list[str]

# Defaults the documents state for run settings, with the place that states each.
DEFAULT_SETTINGS: dict[str, Any] = {
    "fill": "next_open",  # 04-execution 4.2
    "risk_basis": "gross",  # 14-strategy-api 3
    "after_risk_exit": "lock_session",  # 14-strategy-api 3
    "round_levels": True,  # 05-risk-management 8
}
# Default intraday times of day (IST), 04-execution section 2. They are dated: NSE and BSE
# cash move to 14:50 and 15:00 on 2026-08-03 (decision TIMES-1); NFO and BFO keep 15:00
# and 15:10. Each row: exchanges, first trade date it applies to, the times.
DEFAULT_INTRADAY_TIMES = (
    (("NSE", "BSE"), "2026-08-03", {"entry_until": "14:50", "square_off": "15:00"}),
    (("NSE", "BSE", "NFO", "BFO"), "0001-01-01", {"entry_until": "15:00", "square_off": "15:10"}),
)
INTRADAY_TIME_KEYS = ("entry_until", "square_off")
# Default base interval, 03-architecture 4.1: 1 minute for an intraday strategy whose
# 1-minute data covers the run, daily for a daily strategy.
DEFAULT_INTRADAY_BASE = "1m"
DAILY_INTERVAL = "D"
# Default product, 10-simple-path section 2: MIS for an intraday interval; for a daily
# interval CNC for equity and NRML for F&O.
DEFAULT_PRODUCT_INTRADAY = "MIS"
DEFAULT_PRODUCT_DAILY = {"equity": "CNC", "future": "NRML"}
# Default roll, 04-execution section 7: one session before expiry.
DEFAULT_ROLL_SESSIONS_BEFORE_EXPIRY = 1


def _is_daily(case: Case) -> bool:
    return interval_step(case["settings"]["strategy_interval"]) is None


def _intraday_default(exchange: str, trade_date: str) -> dict[str, str]:
    """The documented intraday times for an exchange on a trade date (YYYY-MM-DD).

    Raises KeyError when the documents give no default for the exchange.
    """
    for exchanges, valid_from, times in DEFAULT_INTRADAY_TIMES:
        if exchange in exchanges and trade_date >= valid_from:
            return times
    raise KeyError(exchange)


def _documented_value(case: Case, key: str) -> tuple[Any, Any]:
    """(the documented default, the value the case uses) for one setting.

    Raises KeyError when the documents state no default for the setting.
    """
    settings = case["settings"]
    instrument = case["instruments"][0]
    if key in DEFAULT_SETTINGS:
        return DEFAULT_SETTINGS[key], settings[key]
    if key in INTRADAY_TIME_KEYS:
        return _intraday_default(instrument["exchange"], settings["first_slice_ist"][:10])[
            key
        ], settings[key]
    if key == "base":
        default = DAILY_INTERVAL if _is_daily(case) else DEFAULT_INTRADAY_BASE
        return default, settings["base"]
    if key == "product":
        daily_product = DEFAULT_PRODUCT_DAILY[instrument["type"]]
        default = daily_product if _is_daily(case) else DEFAULT_PRODUCT_INTRADAY
        return default, instrument["product"]
    if key == "roll" and settings["roll"] is not None:
        return DEFAULT_ROLL_SESSIONS_BEFORE_EXPIRY, settings["roll"]["sessions_before_expiry"]
    raise KeyError(key)


def documented_defaults_carry_the_documented_values(case: Case) -> Violations:
    """Rule: a setting listed in documented_defaults has the value the documents give as
    the default."""
    problems = []
    for key in case["settings"]["documented_defaults"]:
        try:
            documented, used = _documented_value(case, key)
        except KeyError:
            problems.append(f"{case['id']} settings: the documents state no default for {key!r}")
            continue
        if documented != used:
            problems.append(
                f"{case['id']} settings: {key} is {used!r} but the documented default is "
                f"{documented!r}"
            )
    return problems


def assumptions_repeat_the_resolved_settings(case: Case) -> Violations:
    """Rule: the expected assumptions are the base interval, product and intraday times
    the run was given."""
    settings = case["settings"]
    resolved = {
        "base": settings["base"],
        "product": case["instruments"][0]["product"],
        "entry_until": settings["entry_until"],
        "square_off": settings["square_off"],
    }
    stated = case["expected"]["assumptions"]
    if stated == resolved:
        return []
    return [f"{case['id']} assumptions: stated {stated}, the run settings give {resolved}"]


def stale_marks_cover_open_positions_without_a_bar(case: Case) -> Violations:
    """Rule: a stale mark is listed exactly where an open position has no bar, at the
    instrument's last close before that slice."""
    bars = bars_by_slice(case)
    required = set()
    for gap in case["missing_slices"]:
        instrument, slice_ist = gap["instrument"], gap["slice_ist"]
        if ledger.position_after(case, instrument, slice_ist).quantity_units == 0:
            continue
        earlier = [
            bar
            for time_ist, bar in bars[instrument].items()
            if parse_ist(time_ist) < parse_ist(slice_ist)
        ]
        required.add((slice_ist, instrument, earlier[-1].close))
    stated = {
        (mark["slice_ist"], mark["instrument"], rupees(mark["mark_price_rupees"]))
        for mark in case["expected"]["stale_marks"]
    }
    problems = [
        f"{case['id']} {item[0]}: {item[1]} is open without a bar and needs a stale mark "
        f"at {item[2]}"
        for item in sorted(required - stated)
    ]
    problems += [
        f"{case['id']} {item[0]}: stale mark of {item[1]} at {item[2]} matches no open position "
        "without a bar"
        for item in sorted(stated - required)
    ]
    return problems
