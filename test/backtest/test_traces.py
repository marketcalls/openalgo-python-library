"""Worked traces of the kernel rules, as fixtures the kernel is built against.

The rules of 03-architecture section 4 and the contract of 14-strategy-api are pinned by
small hand-derived runs (09-roadmap section 2). Each run is a fixture: fixed bars, a
scripted strategy, and the fills, levels and events the documents require, with a
derivation that cites the rule behind every number. This module checks, without a
kernel, that every fixture is well formed and consistent with itself, and runs the
kernel against the fixtures.

Fixture format (`traces/trace-NN-*.json`, format id "openalgo.backtest.trace/1")
------------------------------------------------------------------------------
One file per roadmap trace: `id`, `roadmap_trace`, `title`, `phase` and `cases`. A case
is one complete run; the cases of a file are variants of that trace (another base
interval, the short side, one specification point). Every case of every trace has
exactly the keys below (`traces/vocabulary.py` is the authoritative list). Names carry
the unit: `_ist` is an ISO 8601 timestamp in IST (+05:30), `_units` a quantity in units
(never lots), `_rupees` a price or an amount. A slice is one slot of the base-interval
grid and is named by the open of its base bar; a daily bar carries the session open of
its date.

    id, title, notes   what the case shows and why its numbers were chosen
    depends_on         ids of the specification points (open or decided) the case relies on
    instruments        symbol, exchange, type, tick_size_rupees, lot_size_units, product,
                       expiry_date, underlying; listed in registration order. These
                       facts override any built-in reference table
    calendar           per exchange, the sessions of the run: date, open_ist, close_ist
    settings           arguments of the run, under their engine names: strategy_interval,
                       base, entry_from, entry_until, square_off (times of day, IST),
                       fill, slippage, costs, capital and max_daily_loss (rupees),
                       quantity_mode, round_levels, risk_basis, after_risk_exit, roll
                       (the contracts traded in order, and sessions_before_expiry);
                       first_slice_ist and last_slice_ist bound the grid; and
                       documented_defaults names the settings whose value is the
                       documented default, so a runner may leave them out
    bars               interval (always the base interval), columns and rows per
                       instrument (time_ist, open, high, low, close in rupees, volume),
                       or derived_from_case: the id of a case in the same file whose rows
                       are aggregated to this interval by the loader, so two variants of
                       one market cannot drift apart
    missing_slices     every grid slot for which an instrument deliberately has no bar
    strategy           class_attributes; decisions; and conditions (null unless the case
                       is a condition strategy). A decision reads: at the close of the
                       strategy bar bar_open_ist (decision_time_ist, which is step 7 of
                       the base slice decision_slice_ist), if the position is if_position
                       (any, flat, long, short), make `call` (method and arguments named
                       as in 14-strategy-api; a {"bt": name, "args": [...]} object is a
                       helper such as bt.points(4)). `python` is the same call as source
                       text. conditions holds one boolean per strategy bar for entry_when
                       and exit_when; the decisions of such a case are its equivalent
                       on_bar (14-strategy-api 4.1) and both forms must give `expected`
    expected           assumptions      resolved base, product, entry_until, square_off
                       strategy_bars    the bars handed to on_bar, as they were when used
                                        (for a rolled future, the contract held that day)
                       decisions        per decision: position_condition_met, order_id,
                                        outcome (filled, rejected, no_order)
                       fills            slice_ist, step (1 to 8, the step of 03 4.2 that
                                        produced it), instrument, side, quantity_units,
                                        price_rupees, fill_at (open, level, close), reason
                                        (entry, add, roll or an exit reason of 05 section
                                        7), order_id, target_index, flags
                                        (filled_at_last_close, gtt_child_unfilled), gtt_id
                                        (the GTT whose exit the fill is, else null)
                       levels           per instrument, a row at the end of every slice in
                                        which its position, stop, targets, trail state or
                                        breakeven state changed. A level written at the
                                        end of slice t is in force from slice t + 1
                       events           order, position, risk and gtt events in order of
                                        occurrence (names in vocabulary.EVENT_NAMES)
                       stale_marks      open positions marked at a last close for want of
                                        a bar
                       final_positions  per instrument: quantity_units, average_price_rupees
                       gross_pnl        realized_rupees, unrealized_rupees, total_rupees
    derivation         numbered steps (slice_ist, step, text, cites) that walk the
                       sequence slice by slice and justify every expected number

Positional products (04-execution 4.4). On a CNC or NRML position the stop and target are
held as a GTT: placed with the fill that opens the position, modified by a trail or
breakeven, fired by a bar that reaches a trigger, after which its child LIMIT fills or is
cancelled unfilled and the engine exits at the next bar open. A gtt event has the keys
gtt_id, instrument, quantity_units, stop_trigger_rupees, target_trigger_rupees,
fired_leg, child_limit_rupees and price_rupees; which of them are set depends on its name
(vocabulary.GTT_EVENT_FIELDS). The level timeline keeps a stop whose GTT has fired at its
level until the position closes; the gtt events say whether its child filled.

An expectation that depends on a point the documents do not settle carries
`depends_on` with the ids of those points; `traces/spec_review.json` states each point,
its readings, the recommended one, the decision once taken and the wording proposed for
the documents. Warnings and costs are not part of the format: every trace runs at zero
cost and zero slippage. Cases of one file that are different markets carry their own
rows even where their opening minutes coincide; nothing requires those rows to stay equal.

The kernel hook is described in `traces/kernel_hook.py`: `traces/kernel_runner.py` runs a
case through `bt.backtest`, and a case that needs a part of the kernel not built yet is
skipped with a reason that names the part.

Each consistency rule is one test, parametrised over the cases and named after the rule
function it runs (`case_rule`); the rules live in `traces/checks_*.py`.
"""

import copy
from collections.abc import Callable
from pathlib import Path

import pytest
from traces import (
    checks_bars,
    checks_events,
    checks_fills,
    checks_gtt,
    checks_gtt_children,
    checks_levels,
    checks_ratchets,
    checks_review,
    checks_risk_events,
    checks_run,
    checks_strategy,
    checks_text,
    checks_triggers,
    kernel_hook,
)
from traces.fixture_loader import load_cases, load_spec_review, load_trace_files
from traces.schema import validate_case_inputs, validate_file
from traces.schema_expected import validate_expected

# Roadmap traces (09-roadmap section 2) that belong to Phase 1.
PHASE_1_TRACES = (1, 2, 3, 5, 6, 8, 11, 12)

TRACE_FILES = load_trace_files()
CASES = load_cases()
SPEC_REVIEW = load_spec_review()
TRACES_DIR = Path(__file__).parent / "traces"
PLAIN_TEXT_FILES = [
    Path(__file__),
    *sorted(path for path in TRACES_DIR.iterdir() if path.is_file()),
]
KERNEL_RUNS = [(case, form) for case in CASES for form in kernel_hook.strategy_forms(case)]

each_case = pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])


def assert_rule_holds(violations: list[str]) -> None:
    """Fail with every violation of one rule, one per line."""
    assert not violations, "\n".join(violations)


def case_rule(rule: Callable[[dict], list[str]]) -> Callable[[dict], None]:
    """A test that runs one consistency rule on every case; it carries the rule's name."""

    @each_case
    def test(case: dict) -> None:
        assert_rule_holds(rule(case))

    test.__name__ = test.__qualname__ = f"test_{rule.__name__}"
    test.__doc__ = rule.__doc__
    return test


# ---------------------------------------------------------------- files and schema


@pytest.mark.parametrize("trace_file", TRACE_FILES, ids=lambda trace_file: trace_file["id"])
def test_trace_file_header_is_well_formed(trace_file):
    assert_rule_holds(validate_file(trace_file, trace_file["id"]))


def test_every_phase_1_trace_has_a_fixture():
    present = sorted(trace_file["roadmap_trace"] for trace_file in TRACE_FILES)
    missing = [number for number in PHASE_1_TRACES if number not in present]
    assert not missing, f"no fixture for roadmap trace(s) {missing}"


@pytest.mark.parametrize("path", PLAIN_TEXT_FILES, ids=lambda path: path.name)
def test_file_is_plain_ascii(path):
    assert_rule_holds(checks_text.file_is_plain_ascii(path))


def test_case_ids_are_unique_across_files():
    identifiers = [case["id"] for case in CASES]
    assert len(set(identifiers)) == len(identifiers), identifiers


test_case_inputs_follow_the_schema = case_rule(validate_case_inputs)
test_expected_section_follows_the_schema = case_rule(validate_expected)
test_every_derivation_step_cites_a_rule = case_rule(checks_text.every_derivation_step_cites_a_rule)

# ---------------------------------------------------------------- run settings

test_documented_defaults_carry_the_documented_values = case_rule(
    checks_run.documented_defaults_carry_the_documented_values
)
test_assumptions_repeat_the_resolved_settings = case_rule(
    checks_run.assumptions_repeat_the_resolved_settings
)
test_stale_marks_cover_open_positions_without_a_bar = case_rule(
    checks_run.stale_marks_cover_open_positions_without_a_bar
)

# ---------------------------------------------------------------- bars

test_bars_are_valid_ohlc = case_rule(checks_bars.bars_are_valid_ohlc)
test_bars_are_in_time_order = case_rule(checks_bars.bars_are_in_time_order)
test_bars_are_on_the_declared_grid = case_rule(checks_bars.bars_are_on_the_declared_grid)
test_missing_slices_are_declared = case_rule(checks_bars.missing_slices_are_declared)
test_bar_prices_are_on_the_tick_grid = case_rule(checks_bars.bar_prices_are_on_the_tick_grid)

# ---------------------------------------------------------------- fills

test_fills_are_in_sequence_order = case_rule(checks_fills.fills_are_in_sequence_order)
test_fill_prices_lie_inside_their_bar = case_rule(checks_fills.fill_prices_lie_inside_their_bar)
test_fill_points_agree_with_steps = case_rule(checks_fills.fill_points_agree_with_steps)
test_fill_prices_are_on_the_tick_grid = case_rule(checks_fills.fill_prices_are_on_the_tick_grid)
test_fill_quantities_are_whole_lots = case_rule(checks_fills.fill_quantities_are_whole_lots)
test_fills_net_to_the_final_positions = case_rule(checks_fills.fills_net_to_the_final_positions)
test_gross_pnl_matches_the_fills = case_rule(checks_fills.gross_pnl_matches_the_fills)

# ---------------------------------------------------------------- level timeline

test_level_rows_are_in_slice_order = case_rule(checks_levels.level_rows_are_in_slice_order)
test_level_rows_follow_the_fills = case_rule(checks_levels.level_rows_follow_the_fills)
test_every_position_change_has_a_level_row = case_rule(
    checks_levels.every_position_change_has_a_level_row
)
test_stops_never_loosen = case_rule(checks_levels.stops_never_loosen)
test_working_exit_quantity_never_exceeds_the_position = case_rule(
    checks_levels.working_exit_quantity_never_exceeds_the_position
)
test_flat_rows_carry_no_levels = case_rule(checks_levels.flat_rows_carry_no_levels)
test_levels_are_on_the_tick_grid = case_rule(checks_levels.levels_are_on_the_tick_grid)
test_favourable_extremes_match_the_bars = case_rule(
    checks_ratchets.favourable_extremes_match_the_bars
)
test_armed_breakeven_is_reflected_in_the_stop = case_rule(
    checks_ratchets.armed_breakeven_is_reflected_in_the_stop
)

# ---------------------------------------------------------------- fills against levels

test_no_level_in_force_is_missed = case_rule(checks_triggers.no_level_in_force_is_missed)
test_level_fills_are_priced_from_the_level_in_force = case_rule(
    checks_triggers.level_fills_are_priced_from_the_level_in_force
)
test_roll_shifts_levels_by_the_spread = case_rule(checks_triggers.roll_shifts_levels_by_the_spread)

# ---------------------------------------------------------------- GTT orders (positional)

test_intraday_positions_hold_no_gtts = case_rule(checks_gtt.intraday_positions_hold_no_gtts)
test_gtt_life_cycles_are_complete = case_rule(checks_gtt.gtt_life_cycles_are_complete)
test_gtt_modifications_follow_the_rules = case_rule(checks_gtt.gtt_modifications_follow_the_rules)
test_gtts_are_placed_with_the_fill_that_opens_the_position = case_rule(
    checks_gtt.gtts_are_placed_with_the_fill_that_opens_the_position
)
test_positional_levels_are_held_by_gtts = case_rule(checks_gtt.positional_levels_are_held_by_gtts)
test_engine_exits_cancel_the_gtt_first = case_rule(checks_gtt.engine_exits_cancel_the_gtt_first)
test_gtt_children_carry_the_buffer = case_rule(checks_gtt_children.gtt_children_carry_the_buffer)
test_gtt_children_fill_by_the_limit_rule = case_rule(
    checks_gtt_children.gtt_children_fill_by_the_limit_rule
)

# ---------------------------------------------------------------- strategy script

test_python_text_matches_the_call = case_rule(checks_strategy.python_text_matches_the_call)
test_decisions_sit_on_strategy_bar_closes = case_rule(
    checks_strategy.decisions_sit_on_strategy_bar_closes
)
test_position_conditions_follow_the_fills = case_rule(
    checks_strategy.position_conditions_follow_the_fills
)
test_filled_orders_fill_at_the_next_open = case_rule(
    checks_strategy.filled_orders_fill_at_the_next_open
)
test_strategy_bars_are_the_aggregate_of_the_base_bars = case_rule(
    checks_strategy.strategy_bars_are_the_aggregate_of_the_base_bars
)
test_condition_arrays_equal_the_scripted_decisions = case_rule(
    checks_strategy.condition_arrays_equal_the_scripted_decisions
)

# ---------------------------------------------------------------- events

test_events_are_in_sequence_order = case_rule(checks_events.events_are_in_sequence_order)
test_position_events_follow_the_fills = case_rule(checks_events.position_events_follow_the_fills)
test_order_events_match_the_decisions = case_rule(checks_events.order_events_match_the_decisions)
test_rule_exits_have_matching_risk_events = case_rule(
    checks_risk_events.rule_exits_have_matching_risk_events
)
test_hit_events_report_the_level_in_force = case_rule(
    checks_risk_events.hit_events_report_the_level_in_force
)
test_arming_events_match_the_timeline = case_rule(
    checks_risk_events.arming_events_match_the_timeline
)
test_daily_loss_events_report_the_run_mark_to_market = case_rule(
    checks_risk_events.daily_loss_events_report_the_run_mark_to_market
)

# ---------------------------------------------------------------- specification review


def test_spec_review_items_are_complete():
    assert_rule_holds(checks_review.review_items_are_complete(SPEC_REVIEW))


@each_case
def test_case_dependencies_are_registered(case):
    assert_rule_holds(checks_review.case_dependencies_are_registered(case, SPEC_REVIEW))


test_expectation_dependencies_are_declared_on_the_case = case_rule(
    checks_review.expectation_dependencies_are_declared_on_the_case
)


def test_spec_review_lists_the_cases_that_depend_on_each_item():
    assert_rule_holds(
        checks_review.review_lists_the_cases_that_depend_on_each_item(SPEC_REVIEW, CASES)
    )


# ---------------------------------------------------------------- kernel


@each_case
def test_comparison_accepts_the_expected_result_itself(case):
    result = copy.deepcopy(case["expected"])
    result["events"].append({"type": "order", "name": "Accepted", "order_id": "o1"})
    for form in kernel_hook.strategy_forms(case):
        assert_rule_holds(kernel_hook.compare_with_expected(case, result, form))


@each_case
def test_comparison_reports_a_changed_fill_price(case):
    result = copy.deepcopy(case["expected"])
    result["fills"][0]["price_rupees"] += 0.05
    differences = kernel_hook.compare_with_expected(case, result, kernel_hook.SCRIPT_FORM)
    assert len(differences) == 1 and "price_rupees" in differences[0], differences


@pytest.mark.parametrize(
    ("case", "strategy_form"),
    KERNEL_RUNS,
    ids=lambda value: value if isinstance(value, str) else value["id"],
)
def test_kernel_reproduces_the_trace(case, strategy_form):
    runner = kernel_hook.find_kernel_runner()
    if runner is None:
        pytest.skip(kernel_hook.KERNEL_MISSING_REASON)
    result = runner(copy.deepcopy(case), strategy_form)
    assert_rule_holds(kernel_hook.compare_with_expected(case, result, strategy_form))
