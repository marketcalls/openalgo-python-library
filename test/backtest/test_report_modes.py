"""Report modes: full, basic, metrics only (also as a DataFrame) and charts only, on
`openalgo.backtest.report` and on `Result`. Every mode is one self-contained,
well-formed page: no script, no external URL, light and dark, phone width."""

import math

import numpy as np
import pandas as pd
import pytest
from accounting_runs import round_trip
from test_report_tearsheet import Structure, section_ids

from openalgo.backtest import metrics, report
from openalgo.backtest.errors import ConfigError
from openalgo.backtest.metrics import DEFINITIONS, GROUPS

IST = "Asia/Kolkata"
FORBIDDEN = ("<script", "http://", "https://", "@import", "url(", "<link", "<iframe", "<img")
MODE_SECTIONS = {
    "basic": [
        "summary",
        "equity",
        "monthly",
        "yearly",
        "drawdowns",
        "trades",
        "costs",
        "assumptions",
        "notes",
    ],
    "metrics": ["metrics", "assumptions"],
    "charts": [
        "equity",
        "cumulative",
        "daily",
        "return-distribution",
        "rolling",
        "montecarlo",
        "distribution",
    ],
}


@pytest.fixture(scope="module")
def performance() -> metrics.Performance:
    closes = pd.bdate_range("2025-01-01", "2025-09-30", tz=IST) + pd.Timedelta(hours=15, minutes=30)
    rng = np.random.default_rng(5)
    equity = pd.Series(
        1_00_000 * np.cumprod(1 + rng.normal(0.0004, 0.01, len(closes))), index=closes
    )
    return metrics.analyse(equity, exchanges="NSE", opening_wealth=1_00_000, rolling_window=40)


@pytest.mark.parametrize("mode", list(report.MODES))
def test_every_mode_is_one_self_contained_well_formed_page(performance, mode):
    page = report.tearsheet(performance, mode=mode)
    assert page.startswith("<!DOCTYPE html>") and page.isascii()
    lower = page.lower()
    assert not [word for word in FORBIDDEN if word in lower]
    structure = Structure(page)
    assert structure.errors == [] and structure.stack == []
    assert "@media (prefers-color-scheme: dark)" in page and "width=device-width" in page


@pytest.mark.parametrize("mode", list(MODE_SECTIONS))
def test_each_mode_has_its_sections_in_order(performance, mode):
    assert section_ids(report.tearsheet(performance, mode=mode)) == MODE_SECTIONS[mode]


def test_the_charts_mode_draws_every_chart_and_no_table(performance):
    page = report.tearsheet(performance, mode="charts")
    assert "<table" not in page
    # Equity and drawdown, cumulative, bars, box plot, four rolling statistics, the band.
    assert Structure(page).tags["svg"] == 10


def test_the_metrics_mode_lists_every_metric_with_its_definition(performance):
    page = report.tearsheet(performance, mode="metrics")
    assert "<svg" not in page
    for key in performance.metrics:
        assert DEFINITIONS[key].label.replace("&", "&amp;") in page, key


def test_the_cumulative_return_can_be_drawn_on_a_log_scale(performance):
    assert "Cumulative return, log scale" in report.tearsheet(performance, log_scale=True)
    assert "Cumulative return, log scale" not in report.tearsheet(performance)


def test_an_unknown_mode_is_refused(performance):
    with pytest.raises(ConfigError, match="mode"):
        report.tearsheet(performance, mode="summary")


def test_the_metrics_table_is_every_metric_in_report_order(performance):
    table = report.metrics_table(performance)
    expected = [key for keys in GROUPS.values() for key in keys if key in performance.metrics]
    assert list(table.index) == expected
    assert list(table.columns) == ["group", "label", "value", "unit", "definition"]
    for key in table.index:
        value, shown = performance.metrics[key], table["value"][key]
        same = value == shown or (
            isinstance(value, float) and math.isnan(value) and math.isnan(shown)
        )
        assert same, key
        assert table["unit"][key] == DEFINITIONS[key].unit


# ----- on a Result ------------------------------------------------------------------


@pytest.fixture(scope="module")
def result():
    """The intraday round trip of 400 shares on the default costs and slippage."""
    return round_trip({"qty": 400})


@pytest.mark.parametrize("mode", list(report.MODES))
def test_a_result_renders_every_mode(result, mode):
    page = result.tearsheet(mode=mode)
    assert Structure(page).errors == []
    assert "RoundTrip" in page


def test_a_result_gives_its_metrics_as_a_dataframe(result):
    table = result.metrics_table()
    assert table.loc["trades", "value"] == result.metrics["trades"]
    assert table.loc["sharpe", "group"] == "Returns and risk"


def test_a_result_runs_a_monte_carlo_study(result):
    first = result.monte_carlo(simulations=20, seed=4)
    again = result.monte_carlo(simulations=20, seed=4)
    pd.testing.assert_frame_equal(first.runs, again.runs)
