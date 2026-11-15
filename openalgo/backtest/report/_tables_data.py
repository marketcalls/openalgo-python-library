"""The tables of a `Performance` as DataFrames: drawdown episodes and the worst of
them, losing streaks, the benchmark comparison and returns compounded by period."""

import numbers

import numpy as np
import pandas as pd

from ..errors import ConfigError
from . import _periods as periods

#: Columns of `Performance.drawdown_episodes`.
EPISODE_COLUMNS = (
    "start",
    "trough",
    "end",
    "recovered",
    "depth",
    "days",
    "days_to_trough",
    "days_to_recover",
)
#: The periods a benchmark comparison groups by, with their pandas period codes.
COMPARISON_PERIODS = {"year": "Y", "month": "M"}


def drawdown_episodes(episodes: list[tuple], ts: np.ndarray, start_ns: int) -> pd.DataFrame:
    """Episodes from the kernel's tuples (peak or None, trough, recovery or None, depth,
    duration in nanoseconds) as `Performance.drawdown_episodes` (see there). `ts` are the
    equity points' UTC nanoseconds and `start_ns` the start of the run, where a fall
    from the opening is timed."""
    rows = []
    for peak, trough, recovery, depth, duration in episodes:
        begin = start_ns if peak is None else int(ts[peak])
        rows.append(
            {
                "start": periods.ist_instant(begin),
                "trough": periods.ist_instant(int(ts[trough])),
                "end": periods.ist_instant(None if recovery is None else int(ts[recovery])),
                "recovered": recovery is not None,
                "depth": float(depth),
                "days": duration / periods.NANOS_PER_DAY,
                "days_to_trough": (int(ts[trough]) - begin) / periods.NANOS_PER_DAY,
                "days_to_recover": np.nan
                if recovery is None
                else (int(ts[recovery]) - int(ts[trough])) / periods.NANOS_PER_DAY,
            }
        )
    frame = pd.DataFrame(rows, columns=list(EPISODE_COLUMNS))
    for name in ("start", "trough", "end"):
        frame[name] = pd.to_datetime(frame[name], utc=True).dt.tz_convert(periods.IST)
    return frame.astype({"recovered": bool})


def worst_drawdowns(episodes: pd.DataFrame, top: int) -> pd.DataFrame:
    """The `top` deepest episodes, deepest first and the earlier on a tie, numbered
    from 1. Raises ConfigError for a `top` that is not a positive whole number."""
    whole = isinstance(top, numbers.Integral) and not isinstance(top, bool)
    if not whole or top < 1:
        raise ConfigError(f"top is how many drawdowns to list, a positive whole number: {top!r}")
    if episodes.empty:
        return episodes.copy()
    ranked = episodes.sort_values(["depth", "start"], kind="stable").head(top)
    ranked = ranked.reset_index(drop=True)
    ranked.index = pd.RangeIndex(1, len(ranked) + 1, name="rank")
    return ranked


def losing_streaks(session_streaks: list[int], trade_streaks: list[int]) -> pd.DataFrame:
    """How many losing streaks of each length there were, of sessions and of trades,
    indexed by length (1 to the longest)."""
    longest = max([*session_streaks, *trade_streaks], default=0)
    lengths = pd.RangeIndex(1, longest + 1, name="length")

    def counts(streaks: list[int]) -> pd.Series:
        return pd.Series(streaks, dtype="int64").value_counts().reindex(lengths, fill_value=0)

    return pd.DataFrame({"sessions": counts(session_streaks), "trades": counts(trade_streaks)})


def compound(returns: pd.Series, code: str | None) -> pd.Series:
    """Session returns compounded per period (`code` a pandas period code, None for no
    grouping), indexed by period. A period with an unknown return is NaN."""
    if code is None:
        return returns
    growth = 1.0 + returns
    keys = returns.index.to_period(code)
    product = growth.groupby(keys).prod(min_count=1)
    unknown = growth.isna().groupby(keys).any()
    return product.where(~unknown) - 1.0


def benchmark_comparison(
    returns: pd.Series, benchmark: pd.Series | None, period: str
) -> pd.DataFrame:
    """Implementation of `Performance.benchmark_comparison` (documented there)."""
    if benchmark is None:
        raise ConfigError("benchmark_comparison needs a Performance analysed with a benchmark")
    if period not in COMPARISON_PERIODS:
        raise ConfigError(f"period is one of {sorted(COMPARISON_PERIODS)}, got {period!r}")
    code = COMPARISON_PERIODS[period]
    keys = benchmark.index.to_period(code)
    bench = benchmark.groupby(keys).agg(lambda span: periods.benchmark_growth(span.to_numpy()))
    frame = pd.DataFrame({"strategy": compound(returns, code), "benchmark": bench})
    frame["excess"] = frame["strategy"] - frame["benchmark"]
    # Unknown (NA) where the benchmark return is unknown, rather than a False.
    frame["beat"] = (frame["excess"] > 0).astype("boolean").mask(frame["excess"].isna())
    if period == "year":
        # Years as plain integers, as in `yearly_returns`.
        frame.index = frame.index.year
    frame.index.name = period
    return frame


__all__ = [
    "COMPARISON_PERIODS",
    "EPISODE_COLUMNS",
    "benchmark_comparison",
    "compound",
    "drawdown_episodes",
    "losing_streaks",
    "worst_drawdowns",
]
