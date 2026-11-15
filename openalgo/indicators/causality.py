"""
Causality registry for the `ta` indicator library.

An output is causal when its value on bar t is computed only from bars up to and
including t. A causal value never changes once its bar has closed, so it can be
used as a signal on that bar. A non-causal output is drawn on bar t but decided
by later bars (a displaced plot, a swing point confirmed after the fact). It is
fine on a chart and wrong in a backtest: a strategy reading it on bar t is
reading the future.

This module is one table, `REGISTRY`, with a row per public `ta` function, and
the functions that read it:

    from openalgo import ta
    from openalgo.indicators import causality

    causality.is_causal("rsi")                       # True
    causality.is_causal("ichimoku")                  # False: one output looks ahead
    causality.is_causal("ichimoku", "lagging_span")  # False
    causality.is_causal("ichimoku", 0)               # True (conversion line)
    causality.is_causal(ta.fractals)                 # False, the function works too
    causality.non_causal()   # {'fractals': ('fractal_up', 'fractal_down'),
                             #  'ichimoku': ('lagging_span',)}
    causality.outputs("macd")                        # ('macd', 'signal', 'histogram')

The table is not taken on trust. `test/test_indicators_causality.py` runs a
truncation test over the whole library: each function is computed on a full
series and again on the series cut after bar t, and a causal output must return
the same values up to t. The test fails when an output behaves differently from
its row, in either direction, and when a public `ta` function has no row. A new
indicator therefore cannot be added without being classified and checked.

Two things a True does not say:

- Causal is not "settled from the first bar". Recursive indicators (EMA, RSI,
  ATR) are causal but need warm-up bars before their values stop depending on
  where the series started.
- A causal value on bar t is known when bar t closes, not before.
"""

import functools
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

# The real `ta` object. A function is recognised by identity against it, so
# this module must stay out of the package's own imports.
from openalgo.indicators import ta as _ta

__all__ = [
    "REGISTRY",
    "Anchor",
    "Indicator",
    "causality",
    "indicator_names",
    "is_causal",
    "non_causal",
    "outputs",
]

_NO_LOOKAHEAD: Mapping[str, str] = MappingProxyType({})


@dataclass(frozen=True)
class Anchor:
    """How an anchored indicator restarts on calendar periods (`_anchors`).

    An anchored value is replayed from the start of a period (a session, a week, a
    month), not computed over a rolling window of bars, so how many bars it needs
    depends on the calendar.

    argument:       the parameter that names the anchor period. A call that leaves
                    it None runs over the whole input as one period.
    periods_before: the whole anchor periods before the current one that a value
                    reads: 0 when it reads the bars of its own period since that
                    began (a session VWAP), 1 when it reads the previous period
                    (floor pivots drawn from the previous day, week or month).
    overrides:      the parameters that, when given, supply the bar times or the
                    period starts in place of the input's own dates.
    """

    argument: str = "anchor"
    periods_before: int = 0
    overrides: tuple[str, ...] = ("timestamps",)


@dataclass(frozen=True)
class Indicator:
    """One row of the registry.

    outputs:    names of the returned values, in the order the function returns
                them. A function with optional outputs lists all of them; its
                default call returns the leading ones.
    non_causal: the outputs that look ahead, as {output name: reason}. Every
                output not named here is causal.
    notes:      what a user of a causal output still has to know, if anything.
    on_ta:      False for an indicator class that has no `ta` function.
    anchor:     how the indicator restarts on calendar periods, None for one that
                does not (`Anchor`).
    """

    outputs: tuple[str, ...] = ("value",)
    non_causal: Mapping[str, str] = field(default=_NO_LOOKAHEAD)
    notes: str | None = None
    on_ta: bool = True
    anchor: Anchor | None = None


# Single-output, causal: the common case.
_PLAIN = Indicator()
_BANDS = Indicator(outputs=("upper", "middle", "lower"))

_FRACTAL_REASON = (
    "marked on the centre bar and decided by the bars after it: `periods` later "
    "bars, up to `periods` + 4 when highs or lows tie"
)

REGISTRY: Mapping[str, Indicator] = MappingProxyType(
    {
        # ---- trend
        "sma": _PLAIN,
        "ema": _PLAIN,
        "wma": _PLAIN,
        "dema": _PLAIN,
        "tema": _PLAIN,
        "hma": _PLAIN,
        "vwma": _PLAIN,
        "alma": _PLAIN,
        "kama": _PLAIN,
        "zlema": _PLAIN,
        "t3": _PLAIN,
        "frama": _PLAIN,
        "trima": _PLAIN,
        "mcginley": _PLAIN,
        "vidya": _PLAIN,
        "supertrend": Indicator(outputs=("supertrend", "direction")),
        "ichimoku": Indicator(
            outputs=(
                "conversion_line",
                "base_line",
                "leading_span_a",
                "leading_span_b",
                "lagging_span",
            ),
            non_causal={
                "lagging_span": (
                    "the close plotted displacement - 1 bars back: the value on bar t "
                    "is the close of bar t + displacement - 1"
                ),
            },
        ),
        "alligator": Indicator(outputs=("jaw", "teeth", "lips")),
        "ma_envelopes": _BANDS,
        "ckstop": Indicator(outputs=("stop_long", "stop_short")),
        # ---- momentum
        "rsi": _PLAIN,
        "macd": Indicator(outputs=("macd", "signal", "histogram")),
        "stochastic": Indicator(outputs=("k", "d")),
        "cci": _PLAIN,
        "williams_r": _PLAIN,
        "bop": _PLAIN,
        "elderray": Indicator(outputs=("bull_power", "bear_power")),
        "fisher": Indicator(outputs=("fisher", "trigger")),
        "crsi": _PLAIN,
        # ---- volatility
        "atr": _PLAIN,
        "bbands": _BANDS,
        "keltner": _BANDS,
        "donchian": _BANDS,
        "chaikin": _PLAIN,
        "natr": _PLAIN,
        "ultimate_oscillator": _PLAIN,
        "true_range": _PLAIN,
        "massindex": _PLAIN,
        "bbpercent": _PLAIN,
        "bbwidth": _PLAIN,
        "chandelier_exit": Indicator(outputs=("long_exit", "short_exit")),
        "hv": _PLAIN,
        "ulcerindex": Indicator(outputs=("ulcer_index", "signal")),  # signal: return_signal
        "starc": _BANDS,
        "rvi_volatility": _PLAIN,
        # ---- volume
        "obv": _PLAIN,
        "obv_smoothed": Indicator(  # bands: the Bollinger ma_type
            outputs=("obv_smoothed", "bb_upper", "bb_lower")
        ),
        "vwap": Indicator(
            notes=(
                "Anchored: the value restarts at every anchor period and depends on "
                "all bars since the anchor, so replay it from the anchor, not from a "
                "rolling window. The anchor needs bar times; without them the whole "
                "input is one period."
            ),
            anchor=Anchor(overrides=("timestamps", "session_starts")),
        ),
        "mfi": _PLAIN,
        "adl": _PLAIN,
        "cmf": _PLAIN,
        "emv": _PLAIN,
        "force_index": _PLAIN,
        "nvi": _PLAIN,
        "nvi_with_ema": Indicator(outputs=("nvi", "nvi_ema")),
        "pvi": _PLAIN,
        "pvi_with_signal": Indicator(outputs=("pvi", "signal")),
        "volosc": _PLAIN,
        "vroc": _PLAIN,
        "kvo": Indicator(outputs=("kvo", "trigger")),
        "pvt": _PLAIN,
        "rvol": _PLAIN,
        "cho": _PLAIN,
        # ---- oscillators
        "cmo": _PLAIN,
        "trix": _PLAIN,
        "uo_oscillator": _PLAIN,
        "awesome_oscillator": _PLAIN,
        "accelerator_oscillator": _PLAIN,
        "ppo": Indicator(outputs=("ppo", "signal", "histogram")),
        "po": _PLAIN,
        "dpo": _PLAIN,
        "aroon_oscillator": _PLAIN,
        "stochrsi": Indicator(outputs=("k", "d")),
        "rvi": Indicator(outputs=("rvi", "signal")),
        "chop": _PLAIN,
        "kst": Indicator(outputs=("kst", "signal")),
        "tsi": Indicator(outputs=("tsi", "signal")),
        "vi": Indicator(outputs=("vi_plus", "vi_minus")),
        "stc": _PLAIN,
        "gator_oscillator": Indicator(outputs=("upper", "lower")),
        "coppock": _PLAIN,
        # ---- statistics
        "linreg": _PLAIN,
        "lrslope": _PLAIN,
        "correlation": _PLAIN,
        "beta": _PLAIN,
        "variance": Indicator(  # all five: return_components
            outputs=("variance", "ema_variance", "zscore", "ema_zscore", "stdev")
        ),
        "tsf": _PLAIN,
        "median": _PLAIN,
        "median_bands": Indicator(outputs=("median", "upper", "lower", "median_ema")),
        "mode": _PLAIN,
        # ---- hybrid
        "adx": Indicator(outputs=("plus_di", "minus_di", "adx")),
        "dmi": Indicator(outputs=("plus_di", "minus_di")),
        "aroon": Indicator(outputs=("aroon_up", "aroon_down")),
        "pivot_points": Indicator(
            outputs=("pivot", "r1", "s1", "r2", "s2", "r3", "s3"),
            notes=(
                "Without an anchor the levels on bar t are built from bar t's own "
                "high, low and close: they are the levels for bar t + 1 and are known "
                "only when bar t closes. Pass anchor='D' (or 'Week', 'Month') with bar times "
                "to get the previous period's levels on every bar."
            ),
            anchor=Anchor(periods_before=1),
        ),
        "psar": Indicator(outputs=("sar", "trend")),  # trend: return_trend
        "fractals": Indicator(
            outputs=("fractal_up", "fractal_down"),
            non_causal={"fractal_up": _FRACTAL_REASON, "fractal_down": _FRACTAL_REASON},
        ),
        "rwi": Indicator(outputs=("rwi_high", "rwi_low")),
        # ---- classic single-formula indicators (classic.py)
        "mom": _PLAIN,
        "rocp": _PLAIN,
        "rocr": _PLAIN,
        "rocr100": _PLAIN,
        "midpoint": _PLAIN,
        "midprice": _PLAIN,
        "apo": _PLAIN,
        "avgprice": _PLAIN,
        "medprice": _PLAIN,
        "typprice": _PLAIN,
        "wclprice": _PLAIN,
        "plus_dm": _PLAIN,
        "minus_dm": _PLAIN,
        "dx": _PLAIN,
        "adxr": _PLAIN,
        "stochf": Indicator(outputs=("fastk", "fastd")),
        "linregangle": _PLAIN,
        "linregintercept": _PLAIN,
        # ---- utilities
        "crossover": _PLAIN,
        "crossunder": _PLAIN,
        "cross": _PLAIN,
        "highest": _PLAIN,
        "lowest": _PLAIN,
        "change": _PLAIN,
        "roc": _PLAIN,
        "stdev": _PLAIN,
        "rising": _PLAIN,
        "falling": _PLAIN,
        "exrem": _PLAIN,
        "flip": _PLAIN,
        "valuewhen": _PLAIN,
        # ---- indicator classes without a `ta` function
        "zigzag": Indicator(
            outputs=("zigzag",),
            non_causal={
                "zigzag": (
                    "a swing is written on the bar where it happened once a later bar "
                    "has moved `deviation` percent away, so past bars are filled in "
                    "afterwards"
                ),
            },
            on_ta=False,
        ),
    }
)


def _is_ta_function(target, name: str) -> bool:
    """True when `target` is the method `name` of the real `ta` object.

    Decided by identity, not by name: a method of another object, of a subclass
    or of an unrelated class that happens to be called the same is not it.
    """
    return getattr(target, "__self__", None) is _ta and getattr(
        target, "__func__", None
    ) is getattr(type(_ta), name, None)


def _key(name) -> str:
    """Registry key for an indicator name or for a function of `ta`.

    Raises KeyError when there is no row: an unknown function is unclassified,
    never assumed causal.
    """
    target = name
    while isinstance(target, functools.partial):
        target = target.func
    if isinstance(target, str):
        for candidate in (target.strip(), target.strip().lower()):
            if candidate in REGISTRY:
                return candidate
    elif callable(target):
        method = getattr(target, "__name__", "")
        if method in REGISTRY and _is_ta_function(target, method):
            return method
    else:
        raise TypeError(f"expected an indicator name or a ta function, got {type(name).__name__}")
    raise KeyError(
        f"{name!r} is not an openalgo.ta indicator, so it has no causality "
        "classification. Check a custom function with a truncation test."
    )


def _output_name(key: str, output) -> str:
    """Name of one output of `key`, given its name or its position."""
    names = REGISTRY[key].outputs
    if isinstance(output, str):
        if output not in names:
            raise KeyError(f"{key} has no output {output!r}; its outputs are {names}")
        return output
    if isinstance(output, int) and not isinstance(output, bool):
        if not -len(names) <= output < len(names):
            raise IndexError(f"{key} has {len(names)} output(s) {names}; no index {output}")
        return names[output]
    raise TypeError(f"output must be a name or an index, got {type(output).__name__}")


def indicator_names() -> tuple[str, ...]:
    """Sorted names of the `ta` functions that have a row in the registry."""
    return tuple(sorted(name for name, row in REGISTRY.items() if row.on_ta))


def outputs(name) -> tuple[str, ...]:
    """Names of the values an indicator returns, in the order it returns them.

    name: a `ta` function name, or the function itself.
    Returns ('value',) for a single-output function. A function with optional
    outputs (ulcerindex, variance, obv_smoothed, psar) lists all of them.
    Raises KeyError for a name without a row.
    """
    return REGISTRY[_key(name)].outputs


def is_causal(name, output=None) -> bool:
    """True when the output on bar t uses only bars up to and including t.

    name:   a `ta` function name ("ichimoku"), or the function itself
            (ta.ichimoku, also wrapped in functools.partial).
    output: one output, by name or by position in the returned tuple. None asks
            about the function as a whole: True only when every output is causal.
    Raises KeyError for a name without a row or an output the function does not
    have, IndexError for a position out of range.
    """
    key = _key(name)
    lookahead = REGISTRY[key].non_causal
    if output is None:
        return not lookahead
    return _output_name(key, output) not in lookahead


def non_causal() -> dict[str, tuple[str, ...]]:
    """Every `ta` output that looks ahead, as {function name: (output name, ...)}.

    Sorted by function name, outputs in return order. Indicator classes that
    have no `ta` function are left out; `causality()` lists them.
    """
    return {
        name: tuple(out for out in REGISTRY[name].outputs if out in REGISTRY[name].non_causal)
        for name in indicator_names()
        if REGISTRY[name].non_causal
    }


def _record(key: str) -> dict:
    row = REGISTRY[key]
    anchor = row.anchor
    return {
        "name": key,
        "on_ta": row.on_ta,
        "outputs": row.outputs,
        "causal": not row.non_causal,
        "non_causal": dict(row.non_causal),
        "notes": row.notes,
        "anchor": None
        if anchor is None
        else {
            "argument": anchor.argument,
            "periods_before": anchor.periods_before,
            "overrides": anchor.overrides,
        },
    }


def causality(name=None) -> dict:
    """The registry as plain data.

    name: a `ta` function name or the function itself, or None for everything.
    Returns one record {"name", "on_ta", "outputs", "causal",
    "non_causal": {output name: reason}, "notes", "anchor"}, or {name: record}
    for every row, including indicator classes that have no `ta` function.
    "anchor" is None, or {"argument", "periods_before", "overrides"} for an
    indicator that restarts on calendar periods (`Anchor`).
    """
    if name is not None:
        return _record(_key(name))
    return {key: _record(key) for key in sorted(REGISTRY)}
