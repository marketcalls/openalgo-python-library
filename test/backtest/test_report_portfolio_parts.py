"""The Portfolio section's four parts and the per-strategy overfitting rows of a portfolio's
tearsheet, each on hand-made inputs (18-multi-strategy 6; 19-custom-tearsheets 3).

The parts read a portfolio through its public tables only, so a stand-in with those
tables (`stand_in`) gives inputs whose every figure is worked by hand here. Each part is
None on any other page (`View.portfolio` None), so no single strategy's page changes.
"""

from __future__ import annotations

import dataclasses
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from custom_pages import judged_quick_checks, small_performance

from openalgo.backtest import metrics
from openalgo.backtest._native import native
from openalgo.backtest._robust._quick import quick_checks
from openalgo.backtest.report._dashboard import grouped_bars
from openalgo.backtest.report._dashboard import portfolio as section
from openalgo.backtest.report._dashboard import portfolio_tables as tables
from openalgo.backtest.report._dashboard.layout import MODE_PARTS, PARTS, View
from openalgo.backtest.report._dashboard.overfitting import (
    PORTFOLIO_SCOPE,
    by_strategy,
    overfitting,
)
from openalgo.backtest.report._format import text as escaped

IST = "Asia/Kolkata"
PORTFOLIO_PARTS = PARTS["portfolio"]


def at(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=IST)


def strategy(capital: float, equity: dict[str, float]) -> SimpleNamespace:
    """A strategy's result as the parts read it: its equity and its capital."""
    index = pd.DatetimeIndex([at(when) for when in equity], name="time")
    frame = pd.DataFrame({"equity_rupees": list(equity.values())}, index=index)
    return SimpleNamespace(equity=frame, manifest={"arguments": {"capital": capital}})


def stand_in(**tables_given) -> SimpleNamespace:
    """A portfolio with the tables given, the others empty."""
    empty_account = pd.DataFrame(
        {"margin_rupees": [], "free_cash_rupees": []}, index=pd.DatetimeIndex([], tz=IST)
    )
    given = {
        "strategies": ("A", "B"),
        "results": {"A": strategy(100_000, {}), "B": strategy(100_000, {})},
        "account": empty_account,
        "contribution": pd.DataFrame(
            {
                "net_pnl_rupees": [0.0, 0.0],
                "share_pct": [np.nan, np.nan],
                "drawdown_pnl_rupees": [np.nan, np.nan],
                "peak_margin_rupees": [0.0, 0.0],
                "orders_refused": [0, 0],
            },
            index=pd.Index(["A", "B"], name="strategy"),
        ),
        "refused": pd.DataFrame(
            columns=["strategy", "order_id", "symbol", "side", "time", "reason"]
        ),
        "net_positions": pd.DataFrame(
            columns=["time", "symbol", "exchange", "product", "net_units", "A_units", "B_units"]
        ),
        "assumptions": {},
    }
    return SimpleNamespace(**{**given, **tables_given})


def view_of(portfolio: object, parts=MODE_PARTS["full"], performance=None) -> View:
    return View(
        performance=performance if performance is not None else small_performance(),
        bench=None,
        bench_name="Benchmark",
        title="Portfolio",
        assumptions=[],
        strategies=None,
        log_scale=False,
        parts=frozenset(parts),
        portfolio=portfolio,
    )


# ----- only a portfolio's page shows them ---------------------------------------------------


PART_FUNCTIONS = {
    "contribution": section.contribution,
    "account_margin": section.account_margin,
    "refused": tables.refused,
    "net_positions": tables.net_positions,
}


@pytest.mark.parametrize("part", PORTFOLIO_PARTS)
def test_every_portfolio_part_is_none_without_a_portfolio(part):
    assert PART_FUNCTIONS[part](view_of(None)) is None


def test_the_portfolio_section_is_none_without_a_portfolio():
    assert section.portfolio(view_of(None)) is None


@pytest.mark.parametrize("part", PORTFOLIO_PARTS)
def test_a_part_the_page_does_not_show_is_none(part):
    others = MODE_PARTS["full"] - {part}
    assert PART_FUNCTIONS[part](view_of(stand_in(), parts=others)) is None


def test_the_section_holds_its_parts_in_order_under_its_title():
    found = section.portfolio(view_of(stand_in()))
    assert (found.anchor, found.title) == ("portfolio", "Portfolio")
    titles = re.findall(r"<h3 class='serif'>([^<]+)</h3>", found.body)
    assert titles == [
        "Contribution",
        "P&amp;L by Month",
        "Account Margin &amp; Free Cash",
        "Refused by the Account",
        "Net Positions",
    ]


def test_only_the_full_mode_shows_the_portfolio_parts():
    for mode, parts in MODE_PARTS.items():
        assert set(PORTFOLIO_PARTS) <= parts if mode == "full" else not set(PORTFOLIO_PARTS) & parts


# ----- contribution -------------------------------------------------------------------------


def test_each_strategy_monthly_pnl_runs_from_its_capital_to_its_last_point_of_each_month():
    # A (capital 1,00,000): 1,01,000 at the end of June, 99,500 mid July, 1,03,000 in
    # August: +1,000, -1,500 and +3,500. B (50,000) has one point, 51,000 on 31 July:
    # nothing in June, +1,000 in July and nothing in August, where it has no point.
    results = {
        "A": strategy(
            100_000,
            {
                "2025-06-30 15:30": 101_000.0,
                "2025-07-15 15:30": 99_500.0,
                "2025-08-29 15:30": 103_000.0,
            },
        ),
        "B": strategy(50_000, {"2025-07-31 15:30": 51_000.0}),
    }
    months = section.monthly_pnl(results)
    assert [month.strftime("%b %Y") for month in months.index] == [
        "Jun 2025",
        "Jul 2025",
        "Aug 2025",
    ]
    assert months["A"].tolist() == [1_000.0, -1_500.0, 3_500.0]
    assert months["B"].tolist() == [0.0, 1_000.0, 0.0]


def test_with_more_strategies_than_colours_the_last_colour_is_the_rest_together():
    names = [f"S{k}" for k in range(1, 8)]
    months = pd.DataFrame([[float(k) for k in range(1, 8)]], columns=names)
    series = section.bar_series(months)
    assert [one.name for one in series] == ["S1", "S2", "S3", "S4", "S5", "2 others"]
    assert series[-1].values.tolist() == [6.0 + 7.0]
    assert [one.colour for one in series] == [f"var(--series-{k})" for k in range(1, 7)]


def test_a_stack_puts_gains_on_gains_and_losses_under_zero():
    # One month: A +3 and C +2 above zero (C on A), B -1 below it.
    series = [
        grouped_bars.BarSeries("A", np.array([3.0]), "var(--series-1)"),
        grouped_bars.BarSeries("B", np.array([-1.0]), "var(--series-2)"),
        grouped_bars.BarSeries("C", np.array([2.0]), "var(--series-3)"),
    ]
    chart = grouped_bars.stacked_chart(["Jul 2025"], series, label="P&L")
    rects = {
        title.split(" &#183; ")[1].split(":")[0]: (float(y), float(height))
        for y, height, title in re.findall(
            r"<rect x='[^']+' y='([^']+)' width='[^']+' height='([^']+)'[^>]*><title>([^<]+)",
            chart,
        )
    }
    a_top, a_height = rects["A"]
    assert rects["C"][0] + rects["C"][1] == pytest.approx(a_top)  # C sits on A
    assert rects["B"][0] == pytest.approx(a_top + a_height)  # B hangs from zero
    assert rects["C"][1] / a_height == pytest.approx(2 / 3)
    assert "Jul 2025 &#183; B: -1.00" in chart
    assert "aria-label='P&amp;L'" in chart


def test_a_stack_of_nothing_says_no_data():
    nothing = [grouped_bars.BarSeries("A", np.array([np.nan]), "red")]
    assert grouped_bars.stacked_chart(["Jul 2025"], nothing, label="x") == (
        "<p class='empty'>No data</p>"
    )


def test_the_contribution_table_shows_rupees_shares_in_percent_and_counts():
    contribution = pd.DataFrame(
        {
            "net_pnl_rupees": [1_500.0, -500.0],
            "share_pct": [150.0, -50.0],
            "drawdown_pnl_rupees": [np.nan, np.nan],
            "peak_margin_rupees": [60_000.0, 0.0],
            "orders_refused": [0, 2],
        },
        index=pd.Index(["A", "B"], name="strategy"),
    )
    html = section.contribution(view_of(stand_in(contribution=contribution)))
    rows = re.findall(r"<tr data-i='\d'>(.*?)</tr>", html)
    cells = [re.findall(r"<td class='([^']+)' data-v='[^']*'>([^<]*)</td>", row) for row in rows]
    assert cells == [
        [
            ("l", "A"),
            ("r nums pos", "&#8377;1,500"),
            ("r nums pos", "150.00%"),
            ("r nums", "n/a"),
            ("r nums", "&#8377;60,000"),
            ("r nums", "0"),
        ],
        [
            ("l", "B"),
            ("r nums neg", "-&#8377;500.00"),
            ("r nums neg", "-50.00%"),
            ("r nums", "n/a"),
            ("r nums", "&#8377;0.00"),
            ("r nums", "2"),
        ],
    ]


def test_the_monthly_bars_name_each_strategy_in_a_legend():
    results = {
        "A": strategy(100_000, {"2025-07-15 15:30": 101_000.0}),
        "B": strategy(100_000, {"2025-07-15 15:30": 99_000.0}),
    }
    html = section.contribution(view_of(stand_in(results=results)))
    assert "background:var(--series-1)'></span>A</span>" in html
    assert "background:var(--series-2)'></span>B</span>" in html
    assert "Jul 2025 &#183; A: 1,000.00" in html and "Jul 2025 &#183; B: -1,000.00" in html


def test_the_series_colours_are_needed_only_where_the_monthly_bars_are_drawn():
    shown = [section.portfolio(view_of(stand_in()))]
    assert section.draws_series(view_of(stand_in()), shown)
    assert not section.draws_series(view_of(None), shown)
    assert not section.draws_series(view_of(stand_in()), [])
    without = MODE_PARTS["full"] - {"contribution"}
    assert not section.draws_series(view_of(stand_in(), parts=without), shown)


# ----- the account's margin -------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_sessions():
    # The account on 1 and 2 July 2025. 1 July: margin 0, 30,000, 0 and free cash
    # 1,00,000, 70,000, 1,00,000; 2 July: margin 50,000 then 0, free cash 45,000 then
    # 95,000. Each session's highest margin: 30,000 and 50,000; lowest free cash: 70,000
    # and 45,000. The account's highest margin is 50,000 at 2 July 10:00, its lowest free
    # cash 45,000 then.
    times = ["2025-07-01 09:16", "2025-07-01 10:00", "2025-07-01 15:30"]
    times += ["2025-07-02 10:00", "2025-07-02 15:30"]
    account = pd.DataFrame(
        {
            "equity_rupees": [100_000.0, 100_000.0, 100_000.0, 95_000.0, 95_000.0],
            "free_cash_rupees": [100_000.0, 70_000.0, 100_000.0, 45_000.0, 95_000.0],
            "margin_rupees": [0.0, 30_000.0, 0.0, 50_000.0, 0.0],
        },
        index=pd.DatetimeIndex([at(when) for when in times], name="time"),
    )
    performance = metrics.analyse(
        account["equity_rupees"].rename("equity"), exchanges="NSE", opening_wealth=100_000.0
    )
    contribution = stand_in().contribution.assign(peak_margin_rupees=[50_000.0, 0.0])
    portfolio = stand_in(account=account, contribution=contribution)
    return section.account_margin(view_of(portfolio, performance=performance))


def test_the_margin_chart_draws_each_session_highest_margin_and_lowest_free_cash(two_sessions):
    assert "data-v0='30000,50000'" in two_sessions
    assert "data-v1='70000,45000'" in two_sessions
    assert "background:var(--text)'></span>Highest margin</span>" in two_sessions
    assert "background:var(--benchmark)'></span>Lowest free cash</span>" in two_sessions


def test_the_margin_tiles_give_the_account_extremes_and_each_strategy_margin_then(two_sessions):
    values = re.findall(r"<div class='stat-value nums'>([^<]+)</div>", two_sessions)
    assert values == ["&#8377;50,000", "&#8377;45,000", "&#8377;50,000", "&#8377;0.00"]
    assert two_sessions.count("02 Jul 2025 10:00") == 2


# ----- refused --------------------------------------------------------------------------------


def test_every_reason_the_fund_records_has_words_in_the_refused_table():
    assert set(tables.REASONS) == set(native.REFUSALS)


def test_the_refused_table_lists_each_order_with_its_time_and_reason():
    refused = pd.DataFrame(
        {
            "strategy": ["B", "A"],
            "order_id": [1, 3],
            "symbol": ["SBIN", "INFY"],
            "side": ["BUY", "SELL"],
            "time": [at("2025-07-01 09:16"), at("2025-07-01 11:00")],
            "reason": ["account_margin", "account_loss"],
        }
    )
    html = tables.refused(view_of(stand_in(refused=refused)))
    rows = re.findall(r"<tr data-i='\d'>(.*?)</tr>", html)
    cells = [re.findall(r">([^<]*)</td>", row) for row in rows]
    assert cells == [
        ["01 Jul 2025 09:16", "B", "1", "SBIN", "BUY", escaped(tables.REASONS["account_margin"])],
        ["01 Jul 2025 11:00", "A", "3", "INFY", "SELL", escaped(tables.REASONS["account_loss"])],
    ]
    assert "2 orders the account refused or ended" in html


def test_an_account_that_refused_nothing_says_so():
    assert "The account refused no order and ended none." in tables.refused(view_of(stand_in()))


# ----- net positions ----------------------------------------------------------------------------


def net_rows(rows: list[tuple]) -> pd.DataFrame:
    columns = ["time", "symbol", "exchange", "product", "A_units", "B_units"]
    frame = pd.DataFrame(rows, columns=columns)
    frame["net_units"] = frame["A_units"] + frame["B_units"]
    return frame


def test_an_instrument_two_strategies_held_shows_their_units_and_the_net():
    # SBIN MIS: A long 100 at 09:16, B short 40 at 09:20 (opposite sides), both flat at
    # 15:10. INFY MIS is held by A alone and SBIN CNC by B alone: no card for either.
    frame = net_rows(
        [
            (at("2025-07-01 09:16"), "SBIN", "NSE", "MIS", 100.0, 0.0),
            (at("2025-07-01 09:17"), "INFY", "NSE", "MIS", 10.0, 0.0),
            (at("2025-07-01 09:20"), "SBIN", "NSE", "MIS", 100.0, -40.0),
            (at("2025-07-01 09:21"), "SBIN", "NSE", "CNC", 0.0, 5.0),
            (at("2025-07-01 15:10"), "SBIN", "NSE", "MIS", 0.0, 0.0),
        ]
    )
    html = tables.net_positions(view_of(stand_in(net_positions=frame)))
    assert html.count("<div class='card'>") == 1
    assert "<h3 class='serif'>SBIN (NSE, MIS)</h3>" in html
    rows = re.findall(r"<tr data-i='\d'>(.*?)</tr>", html)
    assert [re.findall(r">([^<]*)</td>", row) for row in rows] == [
        ["01 Jul 2025 09:16", "100", "0", "100"],
        ["01 Jul 2025 09:20", "100", "-40", "60"],
        ["01 Jul 2025 15:10", "0", "0", "0"],
    ]
    assert "Held on opposite sides first from 01 Jul 2025 09:20" in html


def test_no_instrument_held_by_two_strategies_says_so():
    frame = net_rows([(at("2025-07-01 09:16"), "SBIN", "NSE", "MIS", 100.0, 0.0)])
    html = tables.net_positions(view_of(stand_in(net_positions=frame)))
    assert "No instrument was held by two strategies or more." in html


# ----- the overfitting checks of each strategy --------------------------------------------------


def test_a_portfolio_page_shows_each_strategy_quick_checks_in_a_row_of_its_own():
    # The small performance has 8 closed trades: too few to judge, every check not judged.
    quick = quick_checks(small_performance())
    view = dataclasses.replace(view_of(stand_in()), overfitting={"A": quick, "B": None})
    found = overfitting(view)
    body = found.body
    rows = re.findall(r"<tbody>(.*)</tbody>", body)[0].split("</tr>")[:-1]
    assert len(rows) == 2
    assert "<td class='l'>A</td><td class='r nums'>8 of 30 needed</td>" in rows[0]
    assert rows[0].count("not judged") == 4
    assert "<td class='l'>B</td><td class='r nums'>n/a</td>" in rows[1]
    assert escaped(PORTFOLIO_SCOPE) in body
    pointer = (
        "<p class='key'>result.robustness() on a run of the strategy alone with bt.backtest "
        "runs the full set of checks, with charts.</p>"
    )
    assert pointer in body
    assert found.subtitle == "Quick checks of each strategy's trade list, 2 strategies"


def test_the_rows_are_the_strategies_in_call_order():
    section_found = by_strategy({"B": None, "A": None})
    names = re.findall(r"<tr><td class='l'>([^<]+)</td>", section_found.body)
    assert names == ["B", "A"]


def test_a_judged_strategy_row_shows_each_verdict_with_its_figure_and_rule_on_hover():
    # 40 closed trades, enough to judge: the count alone, then the four checks in report
    # order, each verdict coloured (passed green, failed red, informational muted).
    found = by_strategy({"A": judged_quick_checks()})
    row = re.findall(r"<tbody>(.*)</tbody>", found.body)[0]
    assert row == (
        "<tr><td class='l'>A</td><td class='r nums'>40</td>"
        "<td class='l' title='Sharpe 1.10 out of sample; rule: at least half the Sharpe in "
        "sample'><span class='verdict passed'>passed</span></td>"
        "<td class='l' title='drawdown -8.0% at the 95th; rule: read only'>"
        "<span class='verdict quiet'>informational</span></td>"
        "<td class='l' title='return 2.0% at the 5th; rule: above zero'>"
        "<span class='verdict passed'>passed</span></td>"
        "<td class='l' title='p = 0.20; rule: p below 0.05'>"
        "<span class='verdict failed'>failed</span></td></tr>"
    )
    assert "From each strategy's trade list, no rerun: 200 runs each, seed 7" in found.body
