"""The two evaluation paths of a condition strategy on the markets of the worked traces
(the Phase 1 exit criterion: the fast path is bit-identical to the general path on the
fixture set).

For every trace case a condition strategy can trade, and for seeded condition arrays on
its strategy bars, the Rust signal driver (`entry_when` and the other conditions) and the
literal `on_bar` of 14-strategy-api 4.1 run by the Python event driver give the same
result bit for bit, with the default broker's charges and the default slippage on. Each
case keeps its own calendar, bars, instrument, product and settings; only its decisions
are replaced by the arrays, so the paths meet the entry windows, square-offs, missing
bars, daily CNC bars and GTT-held levels of the traces.

The futures cases (trace 8) are left out: a condition strategy on a futures underlying
is not built yet and is refused with NotImplementedError (`_pipeline.execute`).
"""

import numpy as np
import pytest
from condition_runs import CONDITION_NAMES, assert_same_result, conditions, literal_on_bar
from traces import kernel_runner
from traces.fixture_loader import load_cases
from traces.timegrid import parse_ist

import openalgo.backtest as bt

pytestmark = pytest.mark.filterwarnings("ignore::openalgo.backtest.costs.UnverifiedScheduleWarning")

CHARGES_AND_SLIPPAGE = {"costs": "zerodha", "slippage": "default"}
SEEDS = range(6)
# The share of bars on which a drawn condition is true.
TRUE_SHARE = 0.4
ORDER_ARGUMENTS = (
    {"qty": 10},
    {"qty": 10, "stop": bt.pct(0.5), "target": bt.pct(1)},
    {"qty": 10, "trail": bt.trail_pct(0.5)},
)
CASES = [
    case
    for case in load_cases()
    if case["settings"]["roll"] is None and len(case["instruments"]) == 1
]


def run_case(case: dict, strategy: type):
    """`strategy` on the case's market and settings, with charges and slippage on."""
    settings = case["settings"]
    return bt.backtest(
        strategy,
        data=kernel_runner._frames(case),
        instrument=kernel_runner._instruments(case),
        calendar=kernel_runner._calendar(case),
        start=parse_ist(settings["first_slice_ist"]),
        end=parse_ist(settings["last_slice_ist"]),
        product=case["instruments"][0]["product"],
        **CHARGES_AND_SLIPPAGE,
        **{name: settings[name] for name in kernel_runner.FORWARDED_SETTINGS},
    )


def drawn(case: dict, seed: int) -> tuple[dict, int, dict]:
    """Condition arrays (one value per strategy bar of the case), pyramiding and order
    arguments, from a fixed seed."""
    rng = np.random.default_rng(seed)
    bars = len(case["expected"]["strategy_bars"])
    arrays = {name: rng.random(bars) < TRUE_SHARE for name in CONDITION_NAMES}
    return arrays, int(rng.integers(0, 3)), ORDER_ARGUMENTS[seed % len(ORDER_ARGUMENTS)]


def test_every_single_instrument_trace_case_is_compared():
    futures = {case["id"] for case in load_cases()} - {case["id"] for case in CASES}
    assert futures == {"trace-08a-trail-across-roll", "trace-08b-daily-loss-across-roll"}


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_conditions_and_the_literal_on_bar_agree_on_the_trace_markets(case, seed):
    arrays, pyramiding, order_args = drawn(case, seed)
    attributes = case["strategy"]["class_attributes"]
    by_conditions = run_case(case, conditions(arrays, pyramiding, order_args, attributes))
    by_on_bar = run_case(case, literal_on_bar(arrays, pyramiding, order_args, attributes))
    assert_same_result(by_conditions, by_on_bar)
