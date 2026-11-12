"""Trace 11a in its two forms (14-strategy-api 4.1): the condition arrays run by the
Rust signal driver and the equivalent scripted `on_bar` give one result, not only the
sections the fixture compares: every fill, level row, event (all names), stale mark,
final position and the gross P&L."""

import copy

from traces import kernel_runner
from traces.fixture_loader import load_cases
from traces.kernel_hook import CONDITIONS_FORM, SCRIPT_FORM

CASE_ID = "trace-11a-conditions-and-on-bar"


def run_form(form: str) -> dict:
    case = next(case for case in load_cases() if case["id"] == CASE_ID)
    result = kernel_runner.run_trace_case(copy.deepcopy(case), form)
    # Scripted decisions exist only in the script form.
    result.pop("decisions")
    return result


def test_trace_11a_gives_one_result_in_both_forms():
    assert run_form(CONDITIONS_FORM) == run_form(SCRIPT_FORM)
