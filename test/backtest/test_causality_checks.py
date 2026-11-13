"""The run-time causality checks of indicators (03-architecture 5): truncation, window
stability and checking once per key.

Each case gives the checks a `compute(start, stop)` written here in NumPy, so the
expected outcome follows from its definition: a 3-bar centred average reads the next
bar; a 20-bar rolling sum is settled by any window of at least 20 bars; a running count
depends on where the series starts.
"""

import numpy as np
import pytest

from openalgo.backtest._causality_checks import IndicatorChecks
from openalgo.backtest.errors import ConfigError

BARS = 200
PRICES = 100.0 + np.sin(np.arange(BARS) / 7.0)


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


def check(compute, key="k", anchored=False, checks=None):
    checks = checks or IndicatorChecks("sha256:strategy")
    checks.check(key, "probe(SBIN.close)", compute, compute(0, BARS), anchored)
    return checks


def test_an_indicator_that_reads_the_next_bar_fails_the_truncation_check():
    with pytest.raises(ConfigError, match=r"probe\(SBIN.close\) looks ahead"):
        check(centred_average)


def test_a_causal_indicator_passes_and_its_warm_up_is_the_first_window_that_settles():
    # Windows of 16 bars leave a 20-bar sum unknown (NaN); 32 bars settle it exactly.
    checks = check(causal_sum)
    assert checks.assumption() == (
        "window tolerance 1e-06 (relative); probe(SBIN.close): truncation passed at 4 "
        "cuts, warm-up 32 bars"
    )


def test_an_indicator_no_window_settles_needs_its_whole_history():
    checks = check(running_count)
    assert "needs the whole history" in checks.assumption()


def test_an_anchored_indicator_is_replayed_from_its_anchor():
    checks = check(causal_sum, anchored=True)
    assert "anchored, replayed from its anchor" in checks.assumption()


def test_a_key_is_checked_once():
    calls = []

    def counted(start, stop):
        calls.append((start, stop))
        return causal_sum(start, stop)

    checks = check(counted)
    first = len(calls)
    check(counted, checks=checks)
    assert first > 0 and len(calls) == first + 1  # only the whole-series call of the second


def test_a_key_checked_elsewhere_skips_its_truncation_check_but_measures_its_warm_up():
    # The look-ahead of the centred average goes unnoticed (an earlier run of the sweep
    # checked the key), but its warm-up is still measured: each run waits for it.
    known = IndicatorChecks("sha256:strategy").keys | {"sha256:strategy:k"}
    checks = check(centred_average, checks=IndicatorChecks("sha256:strategy", frozenset(known)))
    assert checks.assumption() == (
        "window tolerance 1e-06 (relative); probe(SBIN.close): truncation checked by an "
        "earlier run of the sweep, no window settles within the tolerance: needs the "
        "whole history"
    )


def test_the_warm_up_is_measured_for_each_label_and_the_truncation_once_per_key():
    calls = []

    def counted(start, stop):
        calls.append((start, stop))
        return causal_sum(start, stop)

    checks = IndicatorChecks("sha256:strategy")
    whole = counted(0, BARS)
    checks.check("k", "probe(SBIN.close, 20)", counted, whole, False)
    after_first = len(calls)
    checks.check("k", "probe(SBIN.close, 30)", counted, whole, False)
    # The second label runs only the window check: none of the truncation cuts (bars
    # 120, 140, 160 and 180 of 200) is computed again.
    cuts = {(0, cut) for cut in (120, 140, 160, 180)}
    assert cuts <= set(calls[:after_first])
    assert not cuts & set(calls[after_first:])
    assert len(calls) > after_first
    assert checks.assumption().count("warm-up 32 bars") == 2


def test_a_cut_too_short_for_the_indicator_is_not_counted():
    def needs_150_bars(start, stop):
        if stop - start < 150:
            raise ValueError("Period (150) cannot be greater than data length")
        return causal_sum(start, stop)

    checks = check(needs_150_bars)
    # Cuts at 120, 140, 160 and 180: two are long enough.
    assert "truncation passed at 2 cuts" in checks.assumption()


def test_an_indicator_no_cut_is_long_enough_for_is_reported_as_not_checked():
    def needs_190_bars(start, stop):
        if stop - start < 190:
            raise ValueError("Period (190) cannot be greater than data length")
        return causal_sum(start, stop)

    # Cuts at 120, 140, 160 and 180: none is long enough, and no window of 16 to 128 bars.
    assert check(needs_190_bars).assumption() == (
        "window tolerance 1e-06 (relative); probe(SBIN.close): truncation not checked (no "
        "cut leaves the bars it needs), no window settles within the tolerance: needs the "
        "whole history"
    )
