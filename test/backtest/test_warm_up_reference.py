"""The reference bars warm-ups are measured on (`_warm_up_reference`): made from integers
by the splitmix64 mixer, the same bit for bit everywhere, holding the trends, ranges,
reversals and gaps of real prices; and the warm-ups of the indicators in common use
measured on them (`_warm_up_measure`).
"""

import hashlib

import numpy as np
import pandas as pd
import pytest

from openalgo import ta
from openalgo.backtest._warm_up_measure import ReferenceCall, ReferenceNeeds
from openalgo.backtest._warm_up_reference import REFERENCE_BARS, _mixed, reference_columns

# The first three outputs of splitmix64 from the seed 0, as published with the mixer.
SPLITMIX64_FROM_ZERO = [16294208416658607535, 7960286522194355700, 487617019471545679]
# A regression pin: the SHA-256 of the reference columns (little-endian int64 times and
# float64 values, in the order below). The bars are made from integers only, so this is
# the digest on every platform and NumPy version; it changes only if the formula does.
COLUMN_ORDER = ("ts_open", "ts_close", "open", "high", "low", "close", "volume")
REFERENCE_DIGEST = "af49c10cc7aec1cf7ce038bcb2a13f312c4be430c9a349da147a32d67993a555"


def digest(columns: dict[str, np.ndarray]) -> str:
    hasher = hashlib.sha256()
    for name in COLUMN_ORDER:
        layout = "<i8" if columns[name].dtype.kind == "i" else "<f8"
        hasher.update(np.ascontiguousarray(columns[name]).astype(layout).tobytes())
    return hasher.hexdigest()


def test_the_mixer_gives_the_published_splitmix64_outputs():
    assert [int(value) for value in _mixed(np.arange(3, dtype=np.uint64))] == SPLITMIX64_FROM_ZERO


def test_the_reference_bars_are_the_same_bit_for_bit():
    assert digest(reference_columns()) == REFERENCE_DIGEST


def test_every_bar_is_a_well_formed_bar_of_whole_paise():
    bars = reference_columns()
    assert len(bars["close"]) == REFERENCE_BARS
    assert (bars["high"] >= np.maximum(bars["open"], bars["close"])).all()
    assert (bars["low"] <= np.minimum(bars["open"], bars["close"])).all()
    assert (bars["low"] >= 25.0).all()  # half the floor of Rs 50
    for name in ("open", "high", "low", "close"):
        # Each price is a whole number of paise divided by 100, the nearest double.
        assert np.array_equal(np.round(bars[name] * 100) / 100, bars[name])
    # 1 to 3 lakh units a bar, three times that on a jump.
    assert ((bars["volume"] >= 100_000) & (bars["volume"] <= 900_000)).all()


def test_the_bars_open_with_a_gap_of_two_percent_or_more_about_one_bar_in_48():
    bars = reference_columns()
    before = np.concatenate(([1_000.0], bars["close"][:-1]))
    gaps = np.abs(bars["open"] / before - 1) >= 0.0199
    assert REFERENCE_BARS / 60 < gaps.sum() < REFERENCE_BARS / 40


def test_the_bars_hold_a_trend_of_hundreds_of_bars():
    # A stretch of 500 bars or more in which the close never falls 10 percent below its
    # highest close since the stretch began.
    closes = reference_columns()["close"]
    longest, first, peak = 0, 0, closes[0]
    for bar, close in enumerate(closes):
        peak = max(peak, close)
        if close < 0.9 * peak:
            first, peak = bar, close
        longest = max(longest, bar - first)
    assert longest >= 500


def test_the_close_moves_like_a_share_from_large_to_small():
    # Moves of up to 0.5 to 5 percent a bar, uniform, by regime: a spread of 1.5 to 2.5
    # percent over all the bars.
    closes = reference_columns()["close"]
    spread = float(np.std(np.diff(np.log(closes))))
    assert 0.015 < spread < 0.025


def series(bars, name):
    return pd.Series(bars[name], index=bars["times"])


def outputs(returned):
    parts = returned if isinstance(returned, tuple | list) else (returned,)
    return [np.asarray(part, dtype=float) for part in parts]


# The warm-ups of the indicators in common use, as measured on the reference bars (a pin:
# the reference bars are fixed). A simple average of n bars is exact in any window of n
# bars or more, so its warm-up is the first window of the doubling rule (16, 32, 64, ...)
# that holds n bars; an EMA of n bars keeps (1 - 2 / (n + 1)) of its seed's gap per bar,
# so it needs about 8.5 n bars to bring a gap of a few percent within 1e-6, and more
# after a jump, rounded up to the next window. Whether these are long enough on other
# prices is test_warm_up_battery.py.
COMMON = [
    ("ema 20", lambda b: outputs(ta.ema(series(b, "close"), 20)), 256),
    ("ema 50", lambda b: outputs(ta.ema(series(b, "close"), 50)), 512),
    ("ema 200", lambda b: outputs(ta.ema(series(b, "close"), 200)), 2048),
    ("sma 3", lambda b: outputs(ta.sma(series(b, "close"), 3)), 16),
    ("sma 20", lambda b: outputs(ta.sma(series(b, "close"), 20)), 32),
    ("sma 50", lambda b: outputs(ta.sma(series(b, "close"), 50)), 64),
    ("rsi 14", lambda b: outputs(ta.rsi(series(b, "close"), 14)), 256),
    ("macd 12/26/9", lambda b: outputs(ta.macd(series(b, "close"), 12, 26, 9)), 512),
    (
        "supertrend 10/3",
        lambda b: outputs(
            ta.supertrend(series(b, "high"), series(b, "low"), series(b, "close"), 10, 3)
        ),
        512,
    ),
    ("psar", lambda b: outputs(ta.psar(series(b, "high"), series(b, "low"))), 128),
]


@pytest.mark.parametrize(("label", "compute", "bars"), COMMON, ids=[c[0] for c in COMMON])
def test_the_warm_up_of_an_indicator_in_common_use(label, compute, bars):
    assert ReferenceNeeds().need(label, ReferenceCall(compute)).bars == bars


def test_a_running_sum_has_no_warm_up():
    def balance(bars):
        return outputs(ta.obv(series(bars, "close"), series(bars, "volume")))

    assert ReferenceNeeds().need("obv", ReferenceCall(balance)).bars is None


def test_a_warm_up_is_measured_once_per_label():
    calls = []

    def counted(bars):
        calls.append(len(bars["close"]))
        return outputs(ta.sma(series(bars, "close"), 3))

    needs = ReferenceNeeds()
    needs.need("sma 3", ReferenceCall(counted))
    once = len(calls)
    needs.need("sma 3", ReferenceCall(counted))
    assert len(calls) == once


def test_warm_ups_measured_elsewhere_are_not_measured_again():
    calls = []

    def counted(bars):
        calls.append(len(bars["close"]))
        return outputs(ta.sma(series(bars, "close"), 3))

    first = ReferenceNeeds()
    first.need("sma 3", ReferenceCall(counted))
    calls.clear()
    second = ReferenceNeeds(measured=first.measured())
    assert second.need("sma 3", ReferenceCall(counted)).bars == 16
    assert calls == []
