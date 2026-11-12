"""Strategies on fixed allocations: the combined equity and their return correlation."""

import math

import pandas as pd
import pytest

from openalgo.backtest import metrics
from openalgo.backtest.errors import ConfigError

IST = "Asia/Kolkata"


def at(day: str, clock: str = "15:30") -> pd.Timestamp:
    return pd.Timestamp(f"{day} {clock}", tz=IST)


def series(points: list[tuple[pd.Timestamp, float]]) -> pd.Series:
    return pd.Series([v for _, v in points], index=[t for t, _ in points])


NFO = series([(at("2025-06-02", "09:15"), 100.0), (at("2025-06-03"), 110.0)])
MCX = series(
    [(at("2025-06-02", "09:00"), 50.0), (at("2025-06-03"), 40.0), (at("2025-06-03", "23:30"), 45.0)]
)


def test_the_combined_equity_is_the_sum_on_the_union_of_timestamps():
    combined = metrics.combine_equity({"nfo": NFO, "mcx": MCX})
    assert list(combined.index) == [
        at("2025-06-02", "09:00"),
        at("2025-06-02", "09:15"),
        at("2025-06-03"),
        at("2025-06-03", "23:30"),
    ]
    # NFO before its first mark is its allocation, 100; it holds 110 after 3 June.
    assert combined.to_numpy() == pytest.approx([150.0, 150.0, 150.0, 155.0])


def test_the_combined_return_is_measured_on_the_total_capital():
    combined = metrics.combine_equity({"nfo": NFO, "mcx": MCX})
    m = metrics.analyse(combined, exchanges=["NFO", "MCX"]).metrics
    assert m["total_return"] == pytest.approx(155 / 150 - 1)
    assert m["start_equity"] == 150.0


def test_combining_nothing_is_refused():
    with pytest.raises(ConfigError):
        metrics.combine_equity({})


# NFO first marks at 09:30 after earning 4 rupees on its capital of 100 in its first slice.
NFO_MARKED = series([(at("2025-06-02", "09:30"), 104.0), (at("2025-06-03"), 110.0)])
CAPITALS = {"nfo": 100.0, "mcx": 50.0}


def test_with_capitals_a_strategy_holds_its_capital_before_its_first_point():
    combined = metrics.combine_equity({"nfo": NFO_MARKED, "mcx": MCX}, opening_wealth=CAPITALS)
    assert list(combined.index) == [
        at("2025-06-02", "09:00"),
        at("2025-06-02", "09:30"),
        at("2025-06-03"),
        at("2025-06-03", "23:30"),
    ]
    # NFO is its capital, 100, until its first mark at 09:30 (not 104 before it happened).
    assert combined.to_numpy() == pytest.approx([150.0, 154.0, 150.0, 155.0])


def test_the_combined_opening_wealth_is_the_sum_of_the_capitals():
    combined = metrics.combine_equity({"nfo": NFO_MARKED, "mcx": MCX}, opening_wealth=CAPITALS)
    result = metrics.analyse(
        combined, exchanges=["NFO", "MCX"], opening_wealth=sum(CAPITALS.values())
    )
    assert result.metrics["start_equity"] == 150.0
    assert result.metrics["total_return"] == pytest.approx(155 / 150 - 1)
    # 2 June ends at 154; 3 June ends at the 23:30 MCX mark, 155.
    assert result.session_returns.to_numpy() == pytest.approx([154 / 150 - 1, 155 / 154 - 1])


def test_capitals_must_name_every_strategy():
    with pytest.raises(ConfigError, match="mcx"):
        metrics.combine_equity({"nfo": NFO_MARKED, "mcx": MCX}, opening_wealth={"nfo": 100.0})


def daily(returns: list[float]) -> pd.Series:
    closes = pd.bdate_range("2025-06-02", periods=len(returns), tz=IST)
    return pd.Series(returns, index=closes.tz_localize(None).normalize().rename("date"))


def test_correlation_of_session_returns():
    a = daily([0.01, -0.02, 0.03])
    b = daily([0.02, -0.04, 0.06])
    c = daily([-0.01, 0.02, -0.03])
    matrix = metrics.return_correlation({"a": a, "b": b, "c": c})
    assert matrix.loc["a", "b"] == pytest.approx(1.0)
    assert matrix.loc["a", "c"] == pytest.approx(-1.0)
    assert matrix.loc["b", "b"] == pytest.approx(1.0)


def test_a_session_without_a_point_counts_as_no_move():
    a = daily([0.01, -0.02, 0.03])
    b = daily([0.01, -0.02])
    matrix = metrics.return_correlation({"a": a, "b": b})
    # b reads 0.01, -0.02, 0 on the union of dates.
    x, y = [0.01, -0.02, 0.03], [0.01, -0.02, 0.0]
    mx, my = sum(x) / 3, sum(y) / 3
    cov = sum((p - mx) * (q - my) for p, q in zip(x, y, strict=True))
    var = sum((p - mx) ** 2 for p in x) * sum((q - my) ** 2 for q in y)
    assert matrix.loc["a", "b"] == pytest.approx(cov / math.sqrt(var))


def test_correlation_accepts_performances():
    p = metrics.analyse(NFO, exchanges="NFO")
    q = metrics.analyse(MCX, exchanges="MCX")
    matrix = metrics.return_correlation({"nfo": p, "mcx": q})
    assert list(matrix.columns) == ["nfo", "mcx"]


def test_correlation_needs_two_strategies():
    with pytest.raises(ConfigError):
        metrics.return_correlation({"a": daily([0.01, 0.02])})
