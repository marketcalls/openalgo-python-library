"""What `metrics.analyse` refuses, and the definitions table behind its keys."""

import pandas as pd
import pytest

from openalgo.backtest import metrics
from openalgo.backtest.errors import ConfigError
from openalgo.backtest.report._definitions import UNITS

IST = "Asia/Kolkata"


def at(day: str, clock: str = "15:30") -> pd.Timestamp:
    return pd.Timestamp(f"{day} {clock}", tz=IST)


GOOD = pd.Series(
    [100.0, 101.0, 102.0], index=[at("2025-06-02", "09:15"), at("2025-06-02"), at("2025-06-03")]
)


@pytest.mark.parametrize(
    ("equity", "message"),
    [
        (GOOD.to_numpy(), "pandas Series"),
        (GOOD.iloc[:1], "at least 2"),
        (GOOD.iloc[[0, 2, 1]], "do not increase at row 2"),
        (pd.Series([100.0, float("nan"), 1.0], index=GOOD.index), "row 1"),
        (pd.Series([0.0, 1.0, 2.0], index=GOOD.index), "positive starting capital"),
        (pd.Series(GOOD.to_numpy(), index=GOOD.index.tz_localize(None)), "time zone"),
        (pd.Series(["a", "b", "c"], index=GOOD.index), "numbers"),
    ],
)
def test_unusable_equity_is_refused_with_a_reason(equity, message):
    with pytest.raises(ConfigError, match=message):
        metrics.analyse(equity, exchanges="NSE")


def test_a_cash_flow_at_the_start_is_refused():
    flow = pd.Series([10.0], index=[at("2025-06-02", "09:15")])
    with pytest.raises(ConfigError, match="starting equity"):
        metrics.analyse(GOOD, exchanges="NSE", cash_flows=flow)


def test_a_withdrawal_that_empties_the_account_is_refused():
    equity = pd.Series([100.0, 110.0, 0.0], index=GOOD.index)
    flow = pd.Series([-110.0], index=[at("2025-06-03")])
    with pytest.raises(ConfigError, match="no capital"):
        metrics.analyse(equity, exchanges="NSE", cash_flows=flow)


def test_every_key_is_defined_with_a_known_unit():
    bench = pd.Series([10.0, 10.1, 10.2], index=GOOD.index)
    keys = metrics.analyse(GOOD, exchanges="NSE", benchmark=bench).metrics.keys()
    assert set(keys) == set(metrics.DEFINITIONS)
    assert all(d.unit in UNITS and d.definition for d in metrics.DEFINITIONS.values())


def test_the_metrics_mapping_is_read_only():
    result = metrics.analyse(GOOD, exchanges="NSE")
    with pytest.raises(TypeError):
        result.metrics["sharpe"] = 1.0  # type: ignore[index]


def test_the_module_help_lists_every_definition():
    assert all(f"    {key} (" in metrics.__doc__ for key in metrics.DEFINITIONS)
