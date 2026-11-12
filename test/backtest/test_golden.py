"""Golden fixtures: a fixed strategy set with its equity, trades and metrics pinned bit for
bit (NFR-COR-02), identical across runs and interpreters on one platform and within a
documented tolerance across platforms (NFR-DET-01).

The runs are in `golden_runs` (an `on_bar` MIS intraday strategy, a condition CNC
strategy, a futures strategy across a roll, a GTT positional strategy) and the pins in
`golden/<run>.json`. The pins are regression pins: they were produced once by the
implementation, after each run's trades and a few equity rows were checked by hand
against an independent computation (the crossings from the closes, each fill price from
the next open or the level with its slippage, the stop and target levels and the first
bar that reaches them, the roll spread from the two opens). A difference therefore means
the engine's behaviour changed. When that change is meant, regenerate the pins on
purpose, on the reference platform, and review the diff:

    python test/backtest/golden/regenerate.py --write

No test writes a pin.

Platforms: on the reference platform (Windows x86_64) every number must have the same
bits as its pin. Elsewhere every number must match within CROSS_PLATFORM_TOLERANCE,
relative or absolute; integers, strings and flags must be equal.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from golden_runs import RUNS, on_reference_platform, read_pins, snapshot

HERE = Path(__file__).resolve().parent

# The kernel works on integer price units and posts the ledger in paise, so the equity
# and trades tables use only additions and multiplications, which IEEE 754 makes exact
# and the same everywhere. The metrics also call the platform's math library (logarithms,
# exponentials, powers for annualised and distribution figures), whose last bit differs
# between operating systems; such a 1-ulp difference grows to about 1e-14 relative in a
# ratio, or more near zero. 1e-9 (relative, or absolute near zero) is far above that
# noise and far below any real change: one paisa on equity of Rs 10,00,000 is 1e-8.
CROSS_PLATFORM_TOLERANCE = 1e-9


def same_number(pinned: float, current: float, exact: bool) -> bool:
    """Whether two floats agree: the same bits when `exact`, else within the tolerance
    (NaN matches NaN, an infinity only itself)."""
    if exact:
        return float(pinned).hex() == float(current).hex()
    if math.isnan(pinned) or math.isnan(current):
        return math.isnan(pinned) and math.isnan(current)
    if math.isinf(pinned) or math.isinf(current):
        return pinned == current
    return math.isclose(
        pinned, current, rel_tol=CROSS_PLATFORM_TOLERANCE, abs_tol=CROSS_PLATFORM_TOLERANCE
    )


def differences(pinned: Any, current: Any, exact: bool, path: str = "") -> list[str]:
    """Every place where `current` does not match `pinned`, as readable paths."""
    if isinstance(pinned, dict) and isinstance(current, dict):
        found = []
        if pinned.keys() != current.keys():
            found.append(f"{path}: keys {sorted(pinned.keys() ^ current.keys())} differ")
        for key in pinned.keys() & current.keys():
            found += differences(pinned[key], current[key], exact, f"{path}.{key}")
        return found
    if isinstance(pinned, list) and isinstance(current, list):
        if len(pinned) != len(current):
            return [f"{path}: {len(pinned)} pinned values, {len(current)} now"]
        return [
            difference
            for index, (old, new) in enumerate(zip(pinned, current, strict=True))
            for difference in differences(old, new, exact, f"{path}[{index}]")
        ]
    if isinstance(pinned, float) or isinstance(current, float):
        both_numbers = all(isinstance(value, int | float) for value in (pinned, current))
        agree = both_numbers and same_number(pinned, current, exact)
    else:
        agree = pinned == current and type(pinned) is type(current)
    return [] if agree else [f"{path}: pinned {pinned!r}, now {current!r}"]


@pytest.fixture(scope="module")
def snapshots() -> dict[str, dict[str, Any]]:
    """Each golden run's snapshot, read back through JSON as the pins were."""
    return {name: json.loads(json.dumps(snapshot(run()))) for name, run in RUNS.items()}


@pytest.mark.parametrize("name", list(RUNS))
def test_the_golden_run_matches_its_pins(name, snapshots):
    found = differences(read_pins(name), snapshots[name], exact=on_reference_platform())
    assert not found, f"{len(found)} differences, first: {found[:5]}"


@pytest.mark.parametrize("name", list(RUNS))
def test_the_golden_run_has_trades_to_pin(name):
    # A pin of a run that never trades would pin nothing of the engine.
    assert len(read_pins(name)["trades"]["exit_ts"]) >= 5


@pytest.mark.parametrize("name", list(RUNS))
def test_a_run_repeated_in_one_process_is_bit_identical(name, snapshots):
    again = json.loads(json.dumps(snapshot(RUNS[name]())))
    assert differences(snapshots[name], again, exact=True) == []


def test_the_runs_in_a_fresh_interpreter_with_another_hash_seed_are_bit_identical(snapshots):
    # No result may depend on the order of a hash-ordered container.
    code = (
        "import json, sys\n"
        f"sys.path[:0] = [{str(HERE.parents[1])!r}, {str(HERE)!r}]\n"
        "from golden_runs import RUNS, snapshot\n"
        "print(json.dumps({name: snapshot(run()) for name, run in RUNS.items()}))\n"
    )
    environment = {**os.environ, "PYTHONHASHSEED": "20261112"}
    finished = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code],
        capture_output=True,
        text=True,
        env=environment,
        timeout=600,
    )
    assert finished.returncode == 0, finished.stderr
    fresh = json.loads(finished.stdout.strip().splitlines()[-1])
    assert differences(snapshots, fresh, exact=True) == []


@pytest.mark.parametrize(
    ("pinned", "current", "exact", "agree"),
    [
        (1_000_000.0, 1_000_000.0, True, True),
        (0.0, -0.0, True, False),  # the same value, other bits
        (0.1 + 0.2, 0.3, True, False),  # one ulp apart
        (0.1 + 0.2, 0.3, False, True),
        (1_000_000.0, 1_000_000.01, False, False),  # a paisa on Rs 10,00,000
        (2e-10, -3e-10, False, True),  # noise around zero
        (math.nan, math.nan, False, True),
        (math.inf, 1e308, False, False),
    ],
)
def test_numbers_compare_bit_for_bit_or_within_the_cross_platform_tolerance(
    pinned, current, exact, agree
):
    assert same_number(pinned, current, exact) is agree
