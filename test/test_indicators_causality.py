"""
Causality of the `ta` indicator library.

An output is causal when its value on bar t uses only bars up to and including
t. The registry in `openalgo/indicators/causality.py` records that for every
public `ta` function and each of its outputs. This file is what makes the
registry trustworthy: no row is taken on faith.

- Truncation sweep. Each function is computed on a full series and again on the
  same series cut after every single bar the function accepts. A causal output
  must return the same values up to the cut. The measured result has to equal
  the registry, output by output, in both directions: a causal row must pass
  and a non-causal row must actually be caught.
- Replaced-future sweep. The bars after the cut are swapped for different
  prices. A causal output must not move on any bar up to the cut.
- Coverage guard. Every public `ta` function must have a row, so a new
  indicator cannot be added without being classified and then swept here.
- Teeth. A deliberately wrong indicator (a VWAP that restarts one bar early)
  must be caught, so a sweep that could not fail is noticed.

The sweeps run on two markets (complete bars, and bars with NaN holes), at the
default settings and at shortened periods, on the compiled Rust core ("rust",
skipped when the extension is not installed) and on the NumPy fallback
("numpy").

Run: .venv\\Scripts\\python.exe -m pytest test/test_indicators_causality.py -q
"""

import functools
import inspect
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from openalgo import ta
from openalgo.indicators import TechnicalAnalysis, causality
from openalgo.indicators._anchors import anchor_starts
from openalgo.indicators.hybrid import ZigZag

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

BARS_PER_DAY = 25  # 15-minute bars, 09:15 to 15:15
FIRST_DAY = "2025-03-27"  # a Thursday
# Eleven sessions: two week ends, the end of March and the end of a quarter.
TRADING_DAYS = 11
BARS = TRADING_DAYS * BARS_PER_DAY
# The longest default periods in the library are 252 and 255 bars.
LONGEST_DEFAULT_PERIOD = 255
# Bars of the short-period sweep: Thursday, Friday, Monday 31 March, Tuesday 1 April.
SHORT_BARS = 4 * BARS_PER_DAY
# Prices move in ticks, so equal highs, lows and closes are common.
TICK = 0.25
# Share of bars that trade nothing.
ZERO_VOLUME_SHARE = 0.06
# The holed market starts with unknown bars and has more of them further in.
LEADING_NAN_BARS = 3
FIRST_HOLE_BAR = 30
HOLES_PER_COLUMN = 5
WHOLE_NAN_BAR = 61  # every column unknown on the same bar
# Cuts of the replaced-future sweep, spread over the lengths a function accepts.
SWAP_CUTS = 12
# Two outputs agree when they differ by less than this, relative and absolute.
TOLERANCE = 1e-9
# Bars the default Ichimoku lagging span looks ahead: displacement (26) - 1.
ICHIMOKU_LOOKAHEAD = 25


@dataclass(frozen=True)
class Market:
    """Bars the sweeps run on: one array per column, and the bar times."""

    columns: dict
    index: pd.DatetimeIndex


def _bar_times():
    """15-minute bars from 09:15 India time on consecutive business days."""
    sessions = [
        pd.date_range(
            day + pd.Timedelta(hours=9, minutes=15),
            periods=BARS_PER_DAY,
            freq="15min",
            tz="Asia/Kolkata",
        )
        for day in pd.bdate_range(FIRST_DAY, periods=TRADING_DAYS)
    ]
    index = sessions[0]
    for session in sessions[1:]:
        index = index.append(session)
    return index


def _columns(market, seed):
    """Tick-rounded OHLCV with zero-volume bars, plus a second, unrelated close."""
    open_, high, low, close, volume = market(BARS, seed)
    other = market(BARS, seed + 1)[3] / 2.0
    rng = np.random.default_rng(seed)
    volume = np.where(rng.random(BARS) < ZERO_VOLUME_SHARE, 0.0, volume)
    columns = {"o": open_, "h": high, "l": low, "c": close, "c2": other}
    columns = {key: np.round(values / TICK) * TICK for key, values in columns.items()}
    return {**columns, "v": volume}


def _with_holes(columns, seed):
    """A copy with a NaN warm-up, NaN bars per column and one bar unknown everywhere."""
    rng = np.random.default_rng(seed)
    holed = {}
    for key, values in columns.items():
        values = values.copy()
        values[:LEADING_NAN_BARS] = np.nan
        values[rng.integers(FIRST_HOLE_BAR, BARS, HOLES_PER_COLUMN)] = np.nan
        values[WHOLE_NAN_BAR] = np.nan
        holed[key] = values
    return holed


@pytest.fixture(scope="module")
def markets(market):
    """The markets of the sweeps, and for each the market its future is replaced with."""
    index = _bar_times()
    plain, other = _columns(market, seed=11), _columns(market, seed=29)
    return {
        "plain": Market(plain, index),
        "holes": Market(_with_holes(plain, seed=5), index),
        "plain future": Market(other, index),
        "holes future": Market(_with_holes(other, seed=6), index),
    }


SWEPT_MARKETS = ["plain", "holes"]

# ---------------------------------------------------------------------------
# How to call each function
# ---------------------------------------------------------------------------

# Series parameters, filled from the market columns by parameter name.
SERIES_ARGS = {
    "open_prices": "o",
    "high": "h",
    "low": "l",
    "close": "c",
    "volume": "v",
    "data": "c",
    "data1": "c",
    "data2": "c2",
    "asset": "c",
    "market": "c2",
    "series1": "c",
    "series2": "c2",
    "array": "c",
}
# Boolean-like signal parameters of the utilities.
SIGNAL_ARGS = {
    "primary": lambda columns: (columns["c"] > columns["o"]).astype(float),
    "expr": lambda columns: (columns["c"] > columns["o"]).astype(float),
    "secondary": lambda columns: (columns["c"] < columns["o"]).astype(float),
}
# Required scalar parameters that have no default in the signature.
REQUIRED = {"period": 14, "length": 14}
REQUIRED_SHORT = {"period": 5, "length": 3}
# The short-period sweep divides every integer default by this, within these bounds.
SHORTEN_BY = 3
SHORTEST_PERIOD = 2
LONGEST_SHORT_PERIOD = 20
# Keyword value that stands for the bar times.
TIMESTAMPS = "ts"
# How a call refuses an input it cannot work on. Too few bars is a ValueError
# everywhere; ta.volosc also refuses, with a RuntimeError, a series that has
# traded nothing so far (its check_volume_validity setting).
TOO_SHORT = (ValueError,)
REFUSALS = {"volosc": (ValueError, RuntimeError)}


class Col(str):
    """A keyword value naming a market column, cut like the series arguments."""


@dataclass(frozen=True)
class Case:
    """One way of calling a `ta` function in the sweeps."""

    name: str  # ta function
    label: str = ""  # test id; the function name when empty
    kwargs: dict = field(default_factory=dict, hash=False, compare=False)
    as_series: bool = False  # pass pandas Series that carry the bar times
    bars: int = BARS  # how much of the market the sweep uses

    @property
    def id(self):
        return self.label or self.name


def _column(market, key, bars, as_series):
    if key == TIMESTAMPS:
        return market.index[:bars]
    values = market.columns[key][:bars]
    return pd.Series(values, index=market.index[:bars]) if as_series else values


def call(case, market, bars):
    """Outputs of `case` on the first `bars` bars of `market`, as float arrays."""
    fn = getattr(ta, case.name)
    parameters = inspect.signature(fn).parameters
    unknown = set(case.kwargs) - set(parameters)
    assert not unknown, f"case {case.id}: ta.{case.name} has no parameter {sorted(unknown)}"
    args = {}
    for param in parameters.values():
        if param.name in case.kwargs:
            value = case.kwargs[param.name]
            if isinstance(value, Col):
                value = _column(market, value, bars, case.as_series)
            args[param.name] = value
        elif param.default is not inspect.Parameter.empty:
            continue
        elif param.name in SERIES_ARGS:
            args[param.name] = _column(market, SERIES_ARGS[param.name], bars, case.as_series)
        elif param.name in SIGNAL_ARGS:
            args[param.name] = SIGNAL_ARGS[param.name](market.columns)[:bars]
        elif param.name in REQUIRED:
            args[param.name] = REQUIRED[param.name]
        else:
            pytest.fail(
                f"ta.{case.name} has a required parameter {param.name!r} that the "
                "causality sweep cannot fill. Add it to SERIES_ARGS, SIGNAL_ARGS or "
                "REQUIRED in test/test_indicators_causality.py."
            )
    return as_arrays(fn(**args))


def as_arrays(result):
    parts = result if isinstance(result, tuple) else (result,)
    return [np.asarray(part, dtype=np.float64) for part in parts]


def short_periods(name):
    """Keyword arguments that shorten every integer setting of ta.<name>."""
    kwargs = {}
    for param in inspect.signature(getattr(ta, name)).parameters.values():
        default = param.default
        if param.name in REQUIRED_SHORT and default is inspect.Parameter.empty:
            kwargs[param.name] = REQUIRED_SHORT[param.name]
        elif isinstance(default, int) and not isinstance(default, bool):
            shorter = min(default // SHORTEN_BY, LONGEST_SHORT_PERIOD)
            kwargs[param.name] = max(SHORTEST_PERIOD, shorter)
    return kwargs


# The table the sweeps run over: one default call per registered function, the
# calls whose behaviour is set by a flag, a mode or an anchor, and every
# function again with short periods on a short series.
DEFAULT_CASES = [Case(name) for name in causality.indicator_names()]
VARIANT_CASES = [
    Case("dpo", "dpo[centered]", {"is_centered": True}),
    Case("alligator", "alligator[shifts]", {"jaw_shift": 3, "teeth_shift": 2, "lips_shift": 1}),
    Case("ulcerindex", "ulcerindex[signal]", {"return_signal": True}),
    Case("ulcerindex", "ulcerindex[ema signal]", {"return_signal": True, "signal_type": "EMA"}),
    Case("variance", "variance[components]", {"return_components": True}),
    Case("variance", "variance[log returns]", {"mode": "LR", "return_components": True}),
    Case("obv_smoothed", "obv_smoothed[SMA]", {"ma_type": "SMA"}),
    Case("obv_smoothed", "obv_smoothed[bands]", {"ma_type": "SMA + Bollinger Bands"}),
    Case("obv_smoothed", "obv_smoothed[EMA]", {"ma_type": "EMA"}),
    Case("obv_smoothed", "obv_smoothed[RMA]", {"ma_type": "SMMA (RMA)"}),
    Case("obv_smoothed", "obv_smoothed[WMA]", {"ma_type": "WMA"}),
    Case("obv_smoothed", "obv_smoothed[VWMA]", {"ma_type": "VWMA"}),
    Case("psar", "psar[trend]", {"return_trend": True}),
    Case("pvi_with_signal", "pvi_with_signal[SMA]", {"signal_type": "SMA"}),
    Case("ma_envelopes", "ma_envelopes[EMA]", {"ma_type": "EMA"}),
    Case("po", "po[EMA]", {"ma_type": "EMA"}),
    Case("apo", "apo[EMA]", {"ma_type": "EMA"}),
    Case("median_bands", "median_bands[source]", {"source": Col("c")}),
    # Anchored forms need bar times: Series that carry them, or timestamps=.
    Case("vwap", "vwap[session]", {}, as_series=True),
    Case("vwap", "vwap[week]", {"anchor": "Week"}, as_series=True),
    Case("vwap", "vwap[month]", {"anchor": "Month"}, as_series=True),
    Case("vwap", "vwap[quarter]", {"anchor": "Quarter"}, as_series=True),
    Case("vwap", "vwap[4H]", {"anchor": "4H"}, as_series=True),
    Case("vwap", "vwap[1H]", {"anchor": "1H"}, as_series=True),
    Case("vwap", "vwap[30m]", {"anchor": "30m"}, as_series=True),
    Case("vwap", "vwap[no anchor]", {"anchor": None}, as_series=True),
    Case("vwap", "vwap[timestamps]", {"timestamps": Col(TIMESTAMPS)}),
    Case("pivot_points", "pivot_points[D]", {"anchor": "D"}, as_series=True),
    Case("pivot_points", "pivot_points[week]", {"anchor": "Week"}, as_series=True),
    Case("pivot_points", "pivot_points[month]", {"anchor": "Month"}, as_series=True),
    Case("pivot_points", "pivot_points[1H]", {"anchor": "1H"}, as_series=True),
    Case(
        "pivot_points",
        "pivot_points[timestamps]",
        {"anchor": "D", "timestamps": Col(TIMESTAMPS)},
    ),
]
SHORT_CASES = [
    Case(name, f"{name}[short]", short_periods(name), bars=SHORT_BARS)
    for name in causality.indicator_names()
]
CASES = DEFAULT_CASES + VARIANT_CASES + SHORT_CASES
CASE_IDS = [case.id for case in CASES]

# Functions whose default call returns only the first of their listed outputs,
# with the variant that returns all of them.
OPTIONAL_OUTPUTS = {
    "ulcerindex": "ulcerindex[signal]",
    "variance": "variance[components]",
    "obv_smoothed": "obv_smoothed[bands]",
    "psar": "psar[trend]",
}


# ---------------------------------------------------------------------------
# Measuring causality
# ---------------------------------------------------------------------------


def _first_difference(reference, candidate):
    """Index of the first bar where two equal-length outputs disagree, else None."""
    ref_nan, cand_nan = np.isnan(reference), np.isnan(candidate)
    with np.errstate(invalid="ignore"):
        close = np.isclose(reference, candidate, rtol=TOLERANCE, atol=TOLERANCE)
    differs = (ref_nan != cand_nan) | (~close & ~ref_nan & ~cand_nan)
    hits = np.flatnonzero(differs)
    return int(hits[0]) if hits.size else None


def shortest_input(compute, bars, refusals=TOO_SHORT):
    """Fewest bars `compute` accepts. Shorter inputs are refused with one of `refusals`."""

    def accepts(length):
        try:
            compute(length)
        except refusals:
            return False
        return True

    assert accepts(bars), f"the call refuses {bars} bars"
    low, high = 2, bars
    while low < high:
        middle = (low + high) // 2
        if accepts(middle):
            high = middle
        else:
            low = middle + 1
    return low


def truncation_violations(compute, bars, refusals=TOO_SHORT):
    """For each output: None when causal, else a description of the first violation.

    compute:  bars -> list of output arrays, on the first `bars` bars of a market.
    refusals: the errors with which `compute` refuses an input that is too short.
    The series is cut after every bar from the shortest input the call accepts
    up to one bar short of the full series, so the warm-up and seeding code is
    exercised as well as the steady state, and a look-ahead that shows on a
    single cut (the bar before a session starts, say) is not stepped over.
    """
    full = compute(bars)
    verdict = [None] * len(full)
    for length in range(shortest_input(compute, bars, refusals), bars):
        part = compute(length)
        assert len(part) == len(full), "the number of outputs changes with the length"
        for k, (whole, short) in enumerate(zip(full, part, strict=True)):
            assert short.shape == (length,), f"output {k}: {short.shape[0]} values, {length} bars"
            if verdict[k] is not None:
                continue
            bar = _first_difference(whole[:length], short)
            if bar is not None:
                verdict[k] = (
                    f"bar {bar} is {whole[bar]!r} on the full series but "
                    f"{short[bar]!r} when the series ends at bar {length - 1}"
                )
    return verdict


_TRUNCATION_RESULTS = {}


def measure_by_truncation(case, market_name, markets, backend_name):
    """`truncation_violations` of a case on a market, computed once per backend."""
    key = (case.id, market_name, backend_name)
    if key not in _TRUNCATION_RESULTS:
        compute = functools.partial(call, case, markets[market_name])
        refusals = REFUSALS.get(case.name, TOO_SHORT)
        _TRUNCATION_RESULTS[key] = truncation_violations(compute, case.bars, refusals)
    return _TRUNCATION_RESULTS[key]


def _spliced(market, future, cut):
    """`market` up to and including bar `cut`, `future` after it."""
    columns = {
        key: np.concatenate([market.columns[key][: cut + 1], future.columns[key][cut + 1 :]])
        for key in market.columns
    }
    return Market(columns, market.index)


def measure_by_replaced_future(case, market, future):
    """For each output: None when no bar up to the cut moves, else the first that does."""
    bars = case.bars
    reference = call(case, market, bars)
    verdict = [None] * len(reference)
    refusals = REFUSALS.get(case.name, TOO_SHORT)
    first = shortest_input(functools.partial(call, case, market), bars, refusals)
    cuts = np.unique(np.linspace(first, bars - 2, SWAP_CUTS).astype(int))
    for cut in cuts.tolist():
        altered = call(case, _spliced(market, future, cut), bars)
        for k, (ref, alt) in enumerate(zip(reference, altered, strict=True)):
            if verdict[k] is not None:
                continue
            bar = _first_difference(ref[: cut + 1], alt[: cut + 1])
            if bar is not None:
                verdict[k] = (
                    f"bar {bar} changes from {ref[bar]!r} to {alt[bar]!r} "
                    f"when only the bars after {cut} are replaced"
                )
    return verdict


def _check_against_registry(case, verdict, method):
    names = causality.outputs(case.name)
    assert len(verdict) <= len(names), (
        f"ta.{case.name} returned {len(verdict)} outputs but the registry lists {names}"
    )
    for k, violation in enumerate(verdict):
        registered_causal = causality.is_causal(case.name, k)
        if registered_causal and violation is not None:
            pytest.fail(
                f"{case.id} output {names[k]!r} is registered as causal but the "
                f"{method} sweep shows look-ahead: {violation}. Fix the indicator, or "
                "mark the output non-causal in openalgo/indicators/causality.py."
            )
        if not registered_causal and violation is None:
            pytest.fail(
                f"{case.id} output {names[k]!r} is registered as non-causal but the "
                f"{method} sweep found no look-ahead. Remove it from the row's "
                "non_causal in openalgo/indicators/causality.py."
            )


# ---------------------------------------------------------------------------
# The library-wide sweeps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("market_name", SWEPT_MARKETS)
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_truncation_matches_the_registry(case, market_name, markets, backend):
    """Cutting the series after any bar never changes a causal output."""
    verdict = measure_by_truncation(case, market_name, markets, backend)
    _check_against_registry(case, verdict, "truncation")


@pytest.mark.parametrize("market_name", SWEPT_MARKETS)
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_replacing_the_future_matches_the_registry(case, market_name, markets, backend):
    """Different later bars never change a causal output."""
    verdict = measure_by_replaced_future(
        case, markets[market_name], markets[f"{market_name} future"]
    )
    _check_against_registry(case, verdict, "replaced-future")


@pytest.mark.parametrize("market_name", SWEPT_MARKETS)
def test_truncation_finds_exactly_the_registered_look_ahead_outputs(market_name, markets, backend):
    found = {}
    for case in DEFAULT_CASES:
        names = causality.outputs(case.name)
        verdict = measure_by_truncation(case, market_name, markets, backend)
        looks_ahead = tuple(names[k] for k, violation in enumerate(verdict) if violation)
        if looks_ahead:
            found[case.name] = looks_ahead
    assert found == causality.non_causal()


@pytest.mark.parametrize("case", DEFAULT_CASES + VARIANT_CASES, ids=lambda case: case.id)
def test_the_sweep_compares_numbers_on_every_output(case, markets, backend):
    """An output that is NaN on every bar would pass any sweep."""
    for name, values in zip(
        causality.outputs(case.name), call(case, markets["plain"], BARS), strict=False
    ):
        assert np.isfinite(values).any(), f"{case.id} output {name!r} has no value on any bar"


# ---------------------------------------------------------------------------
# Teeth: a wrong indicator must be caught
# ---------------------------------------------------------------------------


def vwap_restarting_one_bar_early(market, bars):
    """A session VWAP that restarts on the last bar of a day, not on the first.

    Knowing that a bar is the last of its day takes the date of the next bar,
    so this reads one bar ahead, and only on the bar before each session.
    """
    starts = anchor_starts(market.index[:bars], "Session")
    one_bar_early = np.append(starts[1:], False)
    columns = [market.columns[key][:bars] for key in ("h", "l", "c", "v")]
    return as_arrays(ta.vwap(*columns, session_starts=one_bar_early))


def test_truncation_catches_an_anchor_that_starts_one_bar_early(markets, backend):
    compute = functools.partial(vwap_restarting_one_bar_early, markets["plain"])
    (violation,) = truncation_violations(compute, BARS)
    assert violation is not None


def test_truncation_passes_the_same_vwap_with_the_anchor_on_the_right_bar(markets, backend):
    market = markets["plain"]

    def session_vwap(bars):
        starts = anchor_starts(market.index[:bars], "Session")
        columns = [market.columns[key][:bars] for key in ("h", "l", "c", "v")]
        return as_arrays(ta.vwap(*columns, session_starts=starts))

    assert truncation_violations(session_vwap, BARS) == [None]


# ---------------------------------------------------------------------------
# Coverage guard: no public ta function without a row
# ---------------------------------------------------------------------------


def public_functions(cls):
    """Names of the public callables of a TechnicalAnalysis class."""
    return {name for name in dir(cls) if not name.startswith("_") and callable(getattr(cls, name))}


def unclassified(cls):
    """Public functions of `cls` that have no registry row."""
    return public_functions(cls) - set(causality.indicator_names())


def test_every_public_ta_function_has_a_registry_row():
    missing = unclassified(TechnicalAnalysis)
    assert not missing, (
        f"ta functions without a causality row: {sorted(missing)}. Add each to REGISTRY "
        "in openalgo/indicators/causality.py; the sweeps in this file then check it."
    )


def test_registry_has_no_row_for_a_function_ta_does_not_have():
    stale = set(causality.indicator_names()) - public_functions(TechnicalAnalysis)
    assert not stale, f"causality rows for functions ta no longer has: {sorted(stale)}"


def test_an_unregistered_indicator_is_reported():
    class Extended(TechnicalAnalysis):
        def shiny_new_average(self, data, period):
            return self.sma(data, period)

    assert unclassified(Extended) == {"shiny_new_average"}


def test_every_registered_function_is_swept_at_default_and_at_short_periods():
    registered = set(causality.indicator_names())
    assert {case.name for case in DEFAULT_CASES} == registered
    assert {case.name for case in SHORT_CASES} == registered
    assert len(set(CASE_IDS)) == len(CASE_IDS), "duplicate case ids"


def test_the_series_is_long_enough_for_the_longest_default_period():
    assert BARS > LONGEST_DEFAULT_PERIOD


@pytest.mark.parametrize("name", causality.indicator_names())
def test_output_names_match_the_returned_tuple(name, markets):
    names = causality.outputs(name)
    assert len(set(names)) == len(names), f"{name}: duplicate output names {names}"
    returned = len(call(Case(name), markets["plain"], BARS))
    expected = 1 if name in OPTIONAL_OUTPUTS else len(names)
    assert returned == expected, f"ta.{name} returns {returned} outputs, the row lists {names}"


@pytest.mark.parametrize("name, label", sorted(OPTIONAL_OUTPUTS.items()))
def test_optional_outputs_are_all_named(name, label, markets):
    case = next(case for case in VARIANT_CASES if case.id == label)
    assert case.name == name
    assert len(call(case, markets["plain"], BARS)) == len(causality.outputs(name))


# ---------------------------------------------------------------------------
# The known look-ahead outputs, shown directly
# ---------------------------------------------------------------------------


def test_ichimoku_lagging_span_holds_the_close_of_a_later_bar(markets):
    """lagging_span[t] is close[t + displacement - 1]."""
    columns = markets["plain"].columns
    lagging = np.asarray(ta.ichimoku(columns["h"], columns["l"], columns["c"])[4])
    known = BARS - ICHIMOKU_LOOKAHEAD
    np.testing.assert_array_equal(lagging[:known], columns["c"][ICHIMOKU_LOOKAHEAD:])
    assert np.isnan(lagging[known:]).all()


def test_a_fractal_flag_appears_only_after_later_bars(markets):
    periods = 2
    high, low = markets["plain"].columns["h"], markets["plain"].columns["l"]
    up, _ = ta.fractals(high, low, periods)
    centre = int(np.flatnonzero(np.asarray(up))[5])
    # One bar short of the `periods` later bars the flag needs.
    bars = centre + periods
    early_up, _ = ta.fractals(high[:bars], low[:bars], periods)
    assert not np.asarray(early_up)[centre]


def test_zigzag_backfills_swings_after_later_bars(markets):
    columns = markets["plain"].columns

    def zigzag(bars):
        return as_arrays(
            ZigZag().calculate(columns["h"][:bars], columns["l"][:bars], columns["c"][:bars])
        )

    (violation,) = truncation_violations(zigzag, BARS)
    assert violation is not None, "ZigZag passed the truncation sweep; its row is wrong"
    assert causality.is_causal("zigzag") is False


# ---------------------------------------------------------------------------
# The registry API (openalgo.indicators.causality)
# ---------------------------------------------------------------------------


def test_is_causal_for_a_whole_function_needs_every_output_causal():
    assert causality.is_causal("rsi") is True
    assert causality.is_causal("macd") is True
    assert causality.is_causal("ichimoku") is False
    assert causality.is_causal("fractals") is False


def test_is_causal_for_one_output_by_name_or_position():
    assert causality.is_causal("ichimoku", "lagging_span") is False
    assert causality.is_causal("ichimoku", 4) is False
    assert causality.is_causal("ichimoku", -1) is False
    assert [causality.is_causal("ichimoku", k) for k in range(4)] == [True] * 4
    assert causality.is_causal("ichimoku", "leading_span_a") is True
    assert causality.is_causal("rsi", "value") is True


def test_is_causal_accepts_the_ta_function_itself():
    assert causality.is_causal(ta.sma) is True
    assert causality.is_causal(ta.ichimoku) is False
    assert causality.is_causal(ta.ichimoku, "base_line") is True
    assert causality.is_causal(functools.partial(ta.fractals, periods=3)) is False


@pytest.mark.parametrize("name", causality.indicator_names())
def test_every_ta_function_resolves_to_its_own_row(name):
    assert causality.causality(getattr(ta, name))["name"] == name


def test_a_method_of_an_unrelated_class_with_the_same_names_is_rejected():
    class TechnicalAnalysis:  # same class name and method name as the real one
        def sma(self, data, period):
            return data

    with pytest.raises(KeyError):
        causality.is_causal(TechnicalAnalysis().sma)


def test_an_overridden_method_of_a_subclass_is_rejected():
    class Custom(TechnicalAnalysis):
        def sma(self, data, period):
            return np.asarray(data)[::-1]  # anything: it is not the library's sma

    with pytest.raises(KeyError):
        causality.is_causal(Custom().sma)


def test_a_method_of_another_instance_is_rejected():
    """Only functions of the `ta` object itself are recognised; pass the name otherwise."""
    with pytest.raises(KeyError):
        causality.is_causal(TechnicalAnalysis().sma)


def test_non_causal_lists_every_look_ahead_output():
    assert causality.non_causal() == {
        "fractals": ("fractal_up", "fractal_down"),
        "ichimoku": ("lagging_span",),
    }


def test_outputs_names_the_returned_values_in_order():
    assert causality.outputs("macd") == ("macd", "signal", "histogram")
    assert causality.outputs(ta.supertrend) == ("supertrend", "direction")
    assert causality.outputs("sma") == ("value",)


@pytest.mark.parametrize("unknown", ["not_an_indicator", "is_causal", len, print])
def test_an_unknown_function_is_rejected_not_assumed_causal(unknown):
    with pytest.raises(KeyError):
        causality.is_causal(unknown)


@pytest.mark.parametrize(
    "output, error",
    [("no_such_output", KeyError), (3, IndexError), (1.5, TypeError), (True, TypeError)],
)
def test_an_unknown_output_is_rejected(output, error):
    with pytest.raises(error):
        causality.is_causal("macd", output)


def test_causality_record_is_plain_data():
    assert causality.causality("sma") == {
        "name": "sma",
        "on_ta": True,
        "outputs": ("value",),
        "causal": True,
        "non_causal": {},
        "notes": None,
    }


def test_causality_record_gives_the_reason_an_output_looks_ahead():
    record = causality.causality("ichimoku")
    assert record["causal"] is False
    assert list(record["non_causal"]) == ["lagging_span"]
    assert "displacement" in record["non_causal"]["lagging_span"]


def test_pivot_points_record_explains_the_same_bar_levels():
    record = causality.causality("pivot_points")
    assert record["causal"] is True
    assert "anchor" in record["notes"]


def test_causality_without_a_name_returns_every_row():
    everything = causality.causality()
    assert set(everything) == set(causality.REGISTRY)
    assert {name for name, record in everything.items() if not record["causal"]} == {
        "fractals",
        "ichimoku",
        "zigzag",
    }


def test_zigzag_class_has_a_row_but_is_not_a_ta_function():
    assert causality.causality("ZigZag")["on_ta"] is False
    assert "zigzag" not in causality.indicator_names()
    assert "zigzag" not in causality.non_causal()


def test_ta_carries_no_causality_helpers():
    """The API is module-level: `dir(ta)` lists indicators and nothing else."""
    listed = {name for name in dir(ta) if not name.startswith("_")}
    assert listed == set(causality.indicator_names())
