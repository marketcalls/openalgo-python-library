"""The spliced series of a futures strategy, back-adjusted as known at each decision
point (04-execution 7; SPEC-30; LEAD-04, LEAD-16; FR-DATA-16).

Two contracts with daily strategy bars on 2024-09-23 to 26. The near contract's bars
close at 100.00, 102.00 and 104.00 on the 23rd, 24th and 25th (opens 99.00, 101.00,
103.00); the next contract's open at 109.00 and 111.00 and close at 110.00 and 112.00 on
the 25th and 26th. The series switches at the 25th's open, where the spread of the two
opens is 109.00 - 103.00 = 6.00. The spliced closes are 100, 102, 110, 112 as each bar
was used, and 106, 108, 110, 112 once the switch is known.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from openalgo import ta
from openalgo.backtest._causality_checks import IndicatorChecks
from openalgo.backtest._indicators import register
from openalgo.backtest._series import BarStore
from openalgo.backtest._spliced import SplicedStore
from openalgo.backtest.errors import ConfigError

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
SPREAD = 6.0


def _at(day: int, hour: int, minute: int) -> int:
    return pd.Timestamp(dt.datetime(2024, 9, day, hour, minute, tzinfo=IST)).value


def _store(symbol: str, days: list[int], opens: list[float], closes: list[float]) -> BarStore:
    series = {
        "ts_open": np.array([_at(d, 9, 15) for d in days], dtype=np.int64),
        "ts_close": np.array([_at(d, 15, 30) for d in days], dtype=np.int64),
        "open": np.array(opens, dtype=float),
        "high": np.array(closes, dtype=float) + 1.0,
        "low": np.array(opens, dtype=float) - 1.0,
        "close": np.array(closes, dtype=float),
        "volume": np.full(len(days), 1_000.0),
    }
    return BarStore(symbol, "D", series)


def _series() -> tuple[SplicedStore, list[BarStore]]:
    near = _store("NIFTY26SEP24FUT", [23, 24, 25], [99.0, 101.0, 103.0], [100.0, 102.0, 104.0])
    following = _store("NIFTY31OCT24FUT", [25, 26], [109.0, 111.0], [110.0, 112.0])
    switch = {"slice_ts": _at(25, 9, 15), "spread_rupees": SPREAD}
    return SplicedStore("NIFTY", "D", [near, following], [switch]), [near, following]


def _decide(store: SplicedStore, sources: list[BarStore], day: int) -> None:
    """The decision at the close of `day`: reveal the contracts' bars of that day, move
    the series to that instant and reveal its bar."""
    revealed = []
    for position, source in enumerate(sources):
        opens = source.column("ts_open")
        row = int(np.searchsorted(opens, _at(day, 9, 15)))
        if row < len(opens) and opens[row] == _at(day, 9, 15):
            values = [float(source.column(name)[row]) for name in ("open", "high", "low", "close")]
            opened = int(opens[row])
            bar = source.reveal((position, opened, _at(day, 15, 30), *values, 1_000.0))
            revealed.append((position, row, bar))
    store.advance(_at(day, 15, 30))
    for position, row, bar in revealed:
        store.reveal_from(position, row, bar)


def test_before_the_switch_the_series_is_the_near_contracts_bars():
    store, sources = _series()
    for day in (23, 24):
        _decide(store, sources, day)
    assert list(store.view("close").values) == [100.0, 102.0]


def test_at_the_switch_the_history_is_re_adjusted_by_the_spread():
    store, sources = _series()
    for day in (23, 24, 25):
        _decide(store, sources, day)
    assert list(store.view("close").values) == [106.0, 108.0, 110.0]
    assert list(store.view("open").values) == [105.0, 107.0, 109.0]


def test_volume_is_never_adjusted():
    store, sources = _series()
    for day in (23, 24, 25):
        _decide(store, sources, day)
    assert list(store.view("volume").values) == [1_000.0] * 3


def test_a_view_built_before_the_switch_reads_the_re_adjusted_history_after_it():
    store, sources = _series()
    close = store.view("close")
    previous = close.shift(1)
    _decide(store, sources, 23)
    _decide(store, sources, 24)
    assert (close[-1], previous[-1]) == (102.0, 100.0)
    _decide(store, sources, 25)
    # [-1] and [-2] are on one basis: 110.00 against the adjusted 108.00.
    assert (close[-1], close[-2], previous[-1]) == (110.0, 108.0, 108.0)


def test_a_ratio_is_recomputed_on_the_re_adjusted_history():
    # close / open of the 24th: 102 / 101 before the switch, 108 / 107 after it.
    store, sources = _series()
    ratio = store.view("close") / store.view("open")
    _decide(store, sources, 23)
    _decide(store, sources, 24)
    assert ratio[-1] == pytest.approx(102.0 / 101.0)
    _decide(store, sources, 25)
    assert ratio[-2] == pytest.approx(108.0 / 107.0)


def test_the_bar_handed_to_on_bar_is_the_contracts_own_bar():
    store, sources = _series()
    for day in (23, 24):
        _decide(store, sources, day)
    following = sources[1]
    bar = following.reveal((1, _at(25, 9, 15), _at(25, 15, 30), 109.0, 111.0, 108.0, 110.0, 1e3))
    store.advance(_at(25, 15, 30))
    assert store.reveal_from(1, 0, bar) is bar
    assert (bar.symbol, bar.open, bar.close) == ("NIFTY31OCT24FUT", 109.0, 110.0)


def test_an_indicator_reads_the_history_as_re_adjusted_at_the_decision_point():
    # sma(close, 2): 101.00 on the 24th; after the switch (106 + 108) / 2 = 107.00 on the
    # 24th and (108 + 110) / 2 = 109.00 on the 25th.
    store, sources = _series()
    sma = register(IndicatorChecks("test"), ta.sma, (store.view("close"), 2), {})
    _decide(store, sources, 23)
    _decide(store, sources, 24)
    assert sma[-1] == 101.0
    _decide(store, sources, 25)
    assert (sma[-2], sma[-1]) == (107.0, 109.0)


def test_an_indicator_does_not_jump_at_the_roll():
    # roc(close, 1) on the 25th: (110 - 108) / 108 = 1.85 percent on the adjusted series,
    # not the (110 - 102) / 102 = 7.84 percent jump of the raw contracts.
    store, sources = _series()
    roc = register(IndicatorChecks("test"), ta.roc, (store.view("close"), 1), {})
    for day in (23, 24, 25):
        _decide(store, sources, day)
    assert roc[-1] == pytest.approx(2.0 / 108.0 * 100.0)


class _Call:
    """An indicator call as `_indicators` describes one: function, inputs, parameters."""

    def __init__(self, func, inputs, label):
        self.func, self.inputs, self.params, self.label = func, inputs, {}, label


def test_the_values_checked_point_in_time_are_those_used_at_each_close():
    # sma(close, 2) as used: 101.00 on the 24th (before the switch), then 109.00 and
    # 111.00 on the 25th and 26th (after it); a computation truncated at the 25th
    # reproduces the value used then.
    store, _ = _series()
    adapter = store.indicator_adapter(
        _Call(ta.sma, (store.view("close"), 2), "sma"), ["value"], ("value",)
    )
    whole = adapter.whole[0]
    assert np.isnan(whole[0]) and list(whole[1:]) == [101.0, 109.0, 111.0]
    assert list(adapter.compute(0, 3)[0][1:]) == [101.0, 109.0]


def test_a_value_that_reads_a_later_bar_fails_the_point_in_time_check():
    # The next bar's close is not known at a bar's close: truncated at the cut, the
    # value differs from the one the whole series gives, so the check rejects it.
    store, _ = _series()
    ahead = _Call(lambda close: close.shift(-1), (store.view("close"),), "ahead")
    adapter = store.indicator_adapter(ahead, ["value"], ("value",))
    with pytest.raises(ConfigError, match="looks ahead"):
        IndicatorChecks("test").check("ahead", "ahead", adapter.compute, adapter.whole, False)
