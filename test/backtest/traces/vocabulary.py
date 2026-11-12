"""The field names and value vocabularies of the trace fixture format.

Data only. `schema` checks the files against these tables and `kernel_hook` uses them
to decide which kernel events are compared. Names carry their unit: `_ist` is an IST
timestamp, `_units` a quantity in units (never lots), `_rupees` a price or an amount.
"""

from __future__ import annotations

FORMAT = "openalgo.backtest.trace/1"
SPEC_REVIEW_FORMAT = "openalgo.backtest.spec-review/1"
BAR_COLUMNS = ["time_ist", "open", "high", "low", "close", "volume"]

# Key added to an expectation that depends on an open item of the specification review.
DEPENDS_ON = "depends_on"

KEYS: dict[str, tuple[str, ...]] = {
    "file": ("format", "id", "roadmap_trace", "title", "phase", "cases"),
    "case": (
        "id",
        "title",
        "depends_on",
        "notes",
        "instruments",
        "calendar",
        "settings",
        "bars",
        "missing_slices",
        "strategy",
        "expected",
        "derivation",
    ),
    "instrument": (
        "symbol",
        "exchange",
        "type",
        "tick_size_rupees",
        "lot_size_units",
        "product",
        "expiry_date",
        "underlying",
    ),
    "session": ("date", "open_ist", "close_ist"),
    "settings": (
        "strategy_interval",
        "base",
        "entry_from",
        "entry_until",
        "square_off",
        "fill",
        "slippage",
        "costs",
        "capital",
        "quantity_mode",
        "round_levels",
        "max_daily_loss",
        "risk_basis",
        "after_risk_exit",
        "roll",
        "first_slice_ist",
        "last_slice_ist",
        "documented_defaults",
    ),
    "bars": ("interval", "derived_from_case", "columns", "rows"),
    "missing_slice": ("instrument", "slice_ist"),
    "strategy": ("class_attributes", "conditions", "decisions"),
    "conditions": (
        "bar_opens_ist",
        "pyramiding",
        "entry_when",
        "exit_when",
        "short_when",
        "cover_when",
    ),
    "decision": (
        "bar_open_ist",
        "decision_time_ist",
        "decision_slice_ist",
        "if_position",
        "call",
        "python",
    ),
    "call": ("method", "arguments"),
    "expected": (
        "assumptions",
        "strategy_bars",
        "decisions",
        "fills",
        "levels",
        "events",
        "stale_marks",
        "final_positions",
        "gross_pnl",
    ),
    "assumptions": ("base", "product", "entry_until", "square_off"),
    "strategy_bar": ("bar_open_ist", "open", "high", "low", "close"),
    "expected_decision": ("decision", "position_condition_met", "order_id", "outcome"),
    "fill": (
        "slice_ist",
        "step",
        "instrument",
        "side",
        "quantity_units",
        "price_rupees",
        "fill_at",
        "reason",
        "order_id",
        "target_index",
        "flags",
    ),
    "level_row": (
        "slice_ist",
        "instrument",
        "position_quantity_units",
        "average_price_rupees",
        "stop",
        "targets",
        "remainder_quantity_units",
        "trail",
        "breakeven",
    ),
    "stop": ("level_rupees", "quantity_units"),
    "target": ("index", "level_rupees", "quantity_units", "state"),
    "trail": ("armed", "favourable_extreme_rupees"),
    "breakeven": ("armed", "trigger_price_rupees", "level_rupees"),
    "order_event": (
        "slice_ist",
        "step",
        "type",
        "name",
        "order_id",
        "quantity_units",
        "price_rupees",
        "reject_code",
    ),
    "position_event": (
        "slice_ist",
        "step",
        "type",
        "name",
        "instrument",
        "position_quantity_units",
    ),
    "risk_event": (
        "slice_ist",
        "step",
        "type",
        "name",
        "scope",
        "instrument",
        "level_rupees",
        "target_index",
        "observed_rupees",
    ),
    "stale_mark": ("slice_ist", "instrument", "mark_price_rupees"),
    "final_position": ("instrument", "quantity_units", "average_price_rupees"),
    "gross_pnl": ("realized_rupees", "unrealized_rupees", "total_rupees"),
    "derivation_step": ("n", "slice_ist", "step", "text", "cites"),
    "review_item": (
        "id",
        "kind",
        "severity",
        "status",
        "title",
        "documents",
        "question",
        "readings",
        "recommended",
        "decision",
        "decided_on",
        "reason",
        "proposed_wording",
        "cases",
    ),
}

SIDES = ("BUY", "SELL")
FILL_AT = ("open", "level", "close")
FILL_FLAGS = ("filled_at_last_close",)

# Reasons a fill adds to or moves a position. The documents define reasons for exits
# only; these three are the fixture's own (specification review SPEC-22).
OPENING_REASONS = ("entry", "add", "roll")
# Exit reasons, 05-risk-management section 7.
EXIT_REASONS = (
    "leg_sl",
    "leg_target",
    "time_exit",
    "overall_sl",
    "overall_target",
    "lock_profit",
    "daily_loss_limit",
    "scheduler",
    "manual",
    "eod",
    "drawdown_halt",
    "margin_liquidation",
    "expiry",
    "signal",
)
FILL_REASONS = OPENING_REASONS + EXIT_REASONS

IF_POSITION = ("any", "flat", "long", "short")
ORDER_METHODS = ("buy", "sell", "exit", "reduce", "target")
DECISION_OUTCOMES = ("filled", "rejected", "no_order")
TARGET_STATES = ("working", "filled")
REJECT_CODES = ("entry_window", "daily_loss_gate")
RISK_SCOPES = ("leg", "session")

# The sequence of 03-architecture section 4.2 has eight steps. The four that produce
# fills or order events in the fixtures are named here.
FIRST_STEP, LAST_STEP = 1, 8
STEP_FILLS_AT_OPEN = 2
STEP_LEVELS = 3
STEP_AGGREGATE_RULES = 6
STEP_CALLBACKS = 7

# Event names compared with the kernel, by event type. A kernel event with any other
# name (for example Accepted) is dropped before the comparison (SPEC-21).
EVENT_NAMES: dict[str, tuple[str, ...]] = {
    "order": ("Submitted", "Filled", "Rejected", "Canceled", "Expired"),
    "position": ("PositionOpened", "PositionChanged", "PositionClosed"),
    "risk": (
        "leg_sl_hit",
        "leg_target_hit",
        "leg_trail_armed",
        "leg_breakeven_armed",
        "daily_loss_limit_hit",
    ),
}
EVENT_KEYS = {"order": "order_event", "position": "position_event", "risk": "risk_event"}

REVIEW_KINDS = ("contradiction", "ambiguity", "gap", "clarification")
REVIEW_SEVERITIES = ("high", "medium", "low")
# "open" until the maintainer decides; then "resolved", with the reading chosen in
# `decision` and the ISO date in `decided_on` (both empty strings while open).
REVIEW_STATUSES = ("open", "resolved")
