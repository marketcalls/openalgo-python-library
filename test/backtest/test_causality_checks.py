"""The run-time causality checks of indicators (03-architecture 5): truncation on the
run's bars, the window-stability rule, the warm-up measured on the engine's reference
bars, and checking once per key.

Each case gives the checks a computation written here in NumPy, so the expected outcome
follows from its definition: a 3-bar centred average reads the next bar; a 20-bar
rolling sum is settled by any window of at least 20 bars, so the first window the rule
tries that holds 20 bars, 32 (it doubles from 16), settles it exactly; a running count
depends on where the series starts, so no window settles it.
"""

import numpy as np
import pytest

from openalgo.backtest._causality_checks import ANCHORED, IndicatorChecks
from openalgo.backtest._warm_up_anchor import PeriodAnchor
from openalgo.backtest._warm_up_measure import (
    NEVER_SETTLES,
    OWN_VALUES,
    WINDOW_TOLERANCE,
    ReferenceCall,
    settle_length,
)
from openalgo.backtest.errors import ConfigError

BARS = 200
PRICES = 100.0 + np.sin(np.arange(BARS) / 7.0)
LABEL = "probe(SBIN.close)"


def rolling_sum(values: np.ndarray, length: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    sums = np.convolve(values, np.ones(length), mode="valid")
    out[length - 1 :] = sums
    return out


def causal_sum(start: int, stop: int) -> list[np.ndarray]:
    return [rolling_sum(PRICES[start:stop], 20)]


def centred_average(start: int, stop: int) -> list[np.ndarray]:
    part = PRICES[start:stop]
    out = np.full(len(part), np.nan)
    out[1:-1] = (part[:-2] + part[1:-1] + part[2:]) / 3.0
    return [out]


def running_count(start: int, stop: int) -> list[np.ndarray]:
    return [np.arange(1, stop - start + 1, dtype=float)]


def sum_of_closes(bars) -> list[np.ndarray]:
    """The 20-bar rolling sum on any bars: what the reference bars measure."""
    return [rolling_sum(np.asarray(bars["close"]), 20)]


def count_of_bars(bars) -> list[np.ndarray]:
    return [np.arange(1, len(bars["close"]) + 1, dtype=float)]


def check(compute, key="k", anchor=None, checks=None, reference=sum_of_closes):
    checks = checks or IndicatorChecks("sha256:strategy")
    recorder = checks.for_run()
    measured = None if reference is None else ReferenceCall(reference)
    recorder.check(key, LABEL, compute, compute(0, BARS), anchor, reference=measured)
    return recorder


def settled(compute) -> int | None:
    return settle_length(compute, compute(0, BARS), BARS, WINDOW_TOLERANCE).bars


def test_an_indicator_that_reads_the_next_bar_fails_the_truncation_check():
    with pytest.raises(ConfigError, match=r"probe\(SBIN.close\) looks ahead"):
        check(centred_average)


def test_the_window_rule_settles_a_20_bar_sum_in_its_first_window_of_20_bars_or_more():
    # Windows of 16 bars leave a 20-bar sum unknown (NaN); 32 bars settle it exactly.
    assert settled(causal_sum) == 32


def test_the_window_rule_finds_no_window_for_a_running_count():
    assert settled(running_count) is None


def test_the_window_rule_finds_no_window_for_an_indicator_longer_than_every_window():
    def needs_190_bars(start, stop):
        if stop - start < 190:
            raise ValueError("Period (190) cannot be greater than data length")
        return causal_sum(start, stop)

    # Windows of 16 to 128 bars are all shorter than 190; 256 is longer than the series.
    assert settled(needs_190_bars) is None


def test_the_warm_up_is_measured_on_the_reference_bars_not_on_the_runs():
    # The run's computation is a running count, which no window settles; the warm-up is
    # that of the indicator on the reference bars, here the 20-bar sum's 32 bars.
    (outcome,) = check(running_count).outcomes
    assert (outcome.warm_up, outcome.note) == (32, "")


def test_a_causal_indicator_is_described_with_its_cuts_and_its_warm_up():
    assert check(causal_sum).assumption() == (
        "window tolerance 1e-06 (relative); probe(SBIN.close): truncation passed at 4 "
        "cuts, warm-up 32 bars"
    )


def test_an_indicator_no_window_settles_on_the_reference_bars_has_no_warm_up():
    (outcome,) = check(causal_sum, reference=count_of_bars).outcomes
    assert (outcome.warm_up, outcome.note) == (None, NEVER_SETTLES)


def test_an_indicator_over_values_of_the_users_own_has_no_warm_up():
    (outcome,) = check(causal_sum, reference=None).outcomes
    assert (outcome.warm_up, outcome.note) == (None, OWN_VALUES)


def test_an_anchored_indicator_is_replayed_from_the_start_of_its_anchor_period():
    month = PeriodAnchor("Month")
    (outcome,) = check(causal_sum, anchor=month).outcomes
    assert (outcome.warm_up, outcome.note, outcome.anchor) == (None, ANCHORED, month)
    assert outcome.describe().endswith("warm-up since the start of its Month")


def test_an_indicator_that_reads_the_period_before_is_replayed_from_its_start():
    (outcome,) = check(causal_sum, anchor=PeriodAnchor("Week", periods_before=1)).outcomes
    assert outcome.describe().endswith("warm-up since the start of the Week before its own")


def test_a_key_is_checked_once():
    calls = []

    def counted(start, stop):
        calls.append((start, stop))
        return causal_sum(start, stop)

    checks = IndicatorChecks("sha256:strategy")
    check(counted, checks=checks)
    first = len(calls)
    check(counted, checks=checks)
    assert first > 0 and len(calls) == first + 1  # only the whole-series call of the second


def test_a_key_checked_elsewhere_skips_its_truncation_check_but_takes_its_warm_up():
    # The look-ahead of the centred average goes unnoticed (an earlier run of the sweep
    # checked the key), but its warm-up is still taken: each run waits for it.
    known = IndicatorChecks("sha256:strategy").keys | {"sha256:strategy:k"}
    checks = IndicatorChecks("sha256:strategy", frozenset(known))
    assert check(centred_average, checks=checks).assumption() == (
        "window tolerance 1e-06 (relative); probe(SBIN.close): truncation checked by an "
        "earlier run of the sweep, warm-up 32 bars"
    )


def test_the_warm_up_is_measured_once_per_label():
    measured = []

    def reference(bars):
        measured.append(len(bars["close"]))
        return sum_of_closes(bars)

    checks = IndicatorChecks("sha256:strategy")
    check(causal_sum, key="a", checks=checks, reference=reference)
    once = len(measured)
    check(causal_sum, key="b", checks=checks, reference=reference)
    assert once > 0 and len(measured) == once


def test_the_truncation_is_checked_once_per_key_whatever_the_label():
    calls = []

    def counted(start, stop):
        calls.append((start, stop))
        return causal_sum(start, stop)

    checks = IndicatorChecks("sha256:strategy").for_run()
    whole = counted(0, BARS)
    reference = ReferenceCall(sum_of_closes)
    checks.check("k", "probe(SBIN.close, 20)", counted, whole, None, reference=reference)
    after_first = len(calls)
    checks.check("k", "probe(SBIN.close, 30)", counted, whole, None, reference=reference)
    # The cuts at bars 120, 140, 160 and 180 of 200 ran for the first label only.
    cuts = {(0, cut) for cut in (120, 140, 160, 180)}
    assert cuts <= set(calls[:after_first])
    assert len(calls) == after_first
    assert checks.assumption().count("warm-up 32 bars") == 2


def test_a_cut_too_short_for_the_indicator_is_not_counted():
    def needs_150_bars(start, stop):
        if stop - start < 150:
            raise ValueError("Period (150) cannot be greater than data length")
        return causal_sum(start, stop)

    # Cuts at 120, 140, 160 and 180: two are long enough.
    assert "truncation passed at 2 cuts" in check(needs_150_bars).assumption()


def test_an_indicator_no_cut_is_long_enough_for_is_reported_as_not_checked():
    def needs_190_bars(start, stop):
        if stop - start < 190:
            raise ValueError("Period (190) cannot be greater than data length")
        return causal_sum(start, stop)

    assert check(needs_190_bars).assumption() == (
        "window tolerance 1e-06 (relative); probe(SBIN.close): truncation not checked (no "
        "cut leaves the bars it needs), warm-up 32 bars"
    )
