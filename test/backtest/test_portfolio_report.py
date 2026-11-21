"""The reports of a portfolio run (`bt.backtest` given a list): `PortfolioResult.summary()`
and the combined tearsheet, `PortfolioResult.tearsheet()` (18-multi-strategy 1.2 and 6;
19-custom-tearsheets 3).

The runs are `portfolio_samples`' scripts on in-process bars of NSE sessions of July 2025,
cash equity at 1x margin, no charges and no slippage; nothing contacts a server. Every
expected figure is worked by hand from the scripts and the bars in the comment above it.
"""

from __future__ import annotations

import datetime as dt
import re
import warnings

import pandas as pd
import pytest
from custom_pages import element_ids, header_links, section_anchors, section_of
from portfolio_samples import DAYS, Scripted, flat_bars, no_costs, stepped_bars
from strategy_samples import CAPITAL as STRATEGY_CAPITAL
from test_portfolio_run import THREE, THREE_RUN

import openalgo.backtest as bt
from openalgo.backtest import metrics
from openalgo.backtest import report as rp
from openalgo.backtest._portfolio_report import PER_STRATEGY
from openalgo.backtest.report._dashboard.portfolio_tables import REASONS
from openalgo.backtest.report._format import text as escaped

LAKH = 100_000
#: The sections of a combined page of two strategies with trades, in page order.
COMBINED_SECTIONS = [
    "overview",
    "performance",
    "risk",
    "monthly",
    "distribution",
    "metrics",
    "trades",
    "overfitting",
    "strategies",
    "portfolio",
    "notes",
]


def stylesheets(page: str) -> list[str]:
    return re.findall(r"<style>(.*?)</style>", page, flags=re.S)


def cells(html: str) -> list[list[str]]:
    """The text of each cell of each body row of the tables in `html`."""
    rows = re.findall(r"<tr data-i='\d+'>(.*?)</tr>", html)
    return [re.findall(r">([^<]*)</td>", row) for row in rows]


# ----- runs ----------------------------------------------------------------------------------


class First(Scripted):
    qty = 600


class Second(Scripted):
    qty = 600


@pytest.fixture(scope="module")
def contention():
    # Two strategies of Rs 1,00,000 on an account of Rs 1,00,000, SBIN at 100.00 all day;
    # both buy 600 at the 09:16 decision. First fills at the 09:16 open (60,000 of
    # margin), leaving the account 40,000 of free cash, too little for Second, which the
    # account refuses (account_margin). First holds to the 15:10 square-off.
    data = {"SBIN": flat_bars(100.0)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bt.backtest(
            [First, Second], data=data, capital=LAKH, account_capital=LAKH, **no_costs()
        )


class Long(Scripted):
    pass


class Short(Scripted):
    side, wait = "SELL", 1


@pytest.fixture(scope="module")
def opposite():
    # Long buys 100 at the 09:16 open; Short skips one decision and sells 100 at the 09:17
    # open; both are squared off at the 15:10 open. SBIN is 100.00 all day.
    return bt.backtest([Long, Short], data={"SBIN": flat_bars(100.0)}, capital=LAKH, **no_costs())


class LongDaily(bt.Strategy):
    """Buys 100 SBIN at 09:16 every session (a timer: it fills at that open)."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.at("09:16", lambda: self.buy(qty=100))


class ShortDaily(LongDaily):
    """Sells 100 SBIN at 09:17 every session."""

    def init(self):
        self.at("09:17", lambda: self.sell(qty=100))


@pytest.fixture(scope="module")
def two_days():
    # 1 July: SBIN 100.00, then 110.00 from the bar opening at 10:55; 2 July: 100.00, then
    # 95.00. Every position is squared off at 15:10. LongDaily makes 100 x 10.00 = +1,000
    # then 100 x -5.00 = -500, ShortDaily -1,000 then +500: in July, +500 and -500.
    data = {
        "SBIN": pd.concat(
            [
                stepped_bars(100.0, 110.0, 100, day=DAYS[0]),
                stepped_bars(100.0, 95.0, 100, day=DAYS[1]),
            ]
        )
    }
    return bt.backtest([LongDaily, ShortDaily], data=data, capital=LAKH, **no_costs())


class Faller(Scripted):
    qty = 600


class Watcher(Scripted):
    entries = 0


@pytest.fixture(scope="module")
def loss():
    # Faller buys 600 SBIN at the 09:16 open at 100.00; SBIN is 95.00 from the bar opening
    # at 09:25 to the 15:10 square-off: -3,000. Watcher never trades.
    return bt.backtest(
        [Faller, Watcher], data={"SBIN": stepped_bars(100.0, 95.0, 10)}, capital=LAKH, **no_costs()
    )


# ----- the names --------------------------------------------------------------------------------


def test_results_maps_each_strategy_name_to_its_result_in_call_order(loss):
    assert loss.strategies == ("Faller", "Watcher")
    assert list(loss.results) == ["Faller", "Watcher"]
    assert all(isinstance(result, bt.Result) for result in loss.results.values())
    with pytest.raises(TypeError):
        loss.results["Other"] = loss.results["Faller"]


# ----- the account's equity ----------------------------------------------------------------------


def test_the_account_equity_of_a_separate_pool_is_the_combined_equity_of_its_strategies():
    # Fixed allocations: the account is the strategies' equities added up on the union of
    # their times, as metrics.combine_equity adds them with each strategy's capital.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pf = bt.backtest(THREE, pool="separate", **THREE_RUN)
    equities = {name: result.equity["equity_rupees"] for name, result in pf.results.items()}
    capitals = dict.fromkeys(pf.results, STRATEGY_CAPITAL)
    combined = metrics.combine_equity(equities, opening_wealth=capitals)
    assert pf.equity.index.tolist() == combined.index.tolist()
    assert pf.equity.tolist() == pytest.approx(combined.tolist(), abs=0.005)


# ----- summary() ---------------------------------------------------------------------------------


def test_the_summary_gives_the_account_figures_and_each_strategy_overfitting_line(loss):
    # The account of Rs 2,00,000 ends at 1,97,000: -3,000, a return of -1.50 percent, and
    # a maximum drawdown of -1.50 percent from its opening. One closed trade, a loss.
    # Faller closed 1 trade and Watcher none: both too few for the quick checks, and none
    # are run on the account's trades together.
    assert loss.summary().splitlines() == [
        "Net profit   : Rs -3,000.00",
        "Return       : -1.50%",
        "Max drawdown : -1.50%",
        "Trades       : 1 (0% won)",
        f"Overfitting  : {PER_STRATEGY}",
        "  Faller     : too few trades to judge (1 closed, 30 needed)",
        "  Watcher    : too few trades to judge (0 closed, 30 needed)",
    ]


class StrategyWithALongName(Scripted):
    entries = 0


def test_a_long_strategy_name_moves_the_colons_of_the_strategy_lines_only():
    pf = bt.backtest(
        [StrategyWithALongName, Watcher],
        data={"SBIN": flat_bars(100.0)},
        capital=LAKH,
        **no_costs(),
    )
    lines = pf.summary().splitlines()
    assert lines[4].index(":") == 13
    assert [line.index(":") for line in lines[5:]] == [2 + 21 + 1] * 2


def test_the_summary_gives_the_benchmark_of_the_first_strategy():
    # Closes given for 30 June (100) and 1 July (103): a return of 3.00 percent over the
    # run's one session.
    closes = pd.Series(
        [100.0, 103.0], index=pd.DatetimeIndex([dt.date(2025, 6, 30), dt.date(2025, 7, 1)])
    )
    pf = bt.backtest(
        [Watcher, Faller],
        data={"SBIN": flat_bars(100.0)},
        capital=LAKH,
        benchmark=closes,
        **no_costs(),
    )
    assert pf.summary().splitlines()[-1] == "Benchmark    : 3.00% return (given closes)"


# ----- the combined page --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_days_page(two_days):
    return two_days.tearsheet()


def test_the_combined_page_has_the_strategies_and_portfolio_sections_before_the_notes(
    two_days_page,
):
    assert section_anchors(two_days_page) == COMBINED_SECTIONS
    assert header_links(two_days_page) == COMBINED_SECTIONS
    ids = element_ids(two_days_page)
    assert len(ids) == len(set(ids))


def test_the_combined_page_is_titled_by_its_strategies(two_days_page):
    assert "<h1 class='serif'>Portfolio: LongDaily and ShortDaily</h1>" in two_days_page


def test_the_combined_page_writes_the_series_colours_once_after_its_own_style(two_days_page):
    sheets = stylesheets(two_days_page)
    assert len(sheets) == 2
    assert sheets[1].startswith(":root{--series-1:var(--text);--series-2:var(--benchmark);")


def test_the_monthly_bars_stack_each_strategy_pnl(two_days_page):
    portfolio = section_of(two_days_page, "portfolio")
    assert "Jul 2025 &#183; LongDaily: 500.00" in portfolio
    assert "Jul 2025 &#183; ShortDaily: -500.00" in portfolio
    assert "aria-label='Net P&amp;L by month'" in portfolio


def test_the_contribution_table_gives_each_strategy_net_pnl(two_days_page):
    portfolio = section_of(two_days_page, "portfolio")
    table = portfolio[: portfolio.index("P&amp;L by Month")]
    assert [row[:2] for row in cells(table)] == [
        ["LongDaily", "&#8377;500.00"],
        ["ShortDaily", "-&#8377;500.00"],
    ]


def test_the_notes_hold_the_portfolio_assumptions_and_warnings(two_days, two_days_page):
    notes = section_of(two_days_page, "notes")
    assert escaped(two_days.assumptions["account"]) in notes
    assert escaped(two_days.warnings[0]) in notes


def test_the_refused_part_lists_the_order_the_account_refused(contention):
    refused = section_of(contention.tearsheet(), "portfolio")
    refused = refused[refused.index("Refused by the Account") :]
    assert cells(refused)[0] == [
        "01 Jul 2025 09:16",
        "Second",
        "1",
        "SBIN",
        "BUY",
        escaped(REASONS["account_margin"]),
    ]


def test_the_margin_part_gives_the_account_extremes_and_each_strategy_margin(contention):
    # First's 600 at 100.00 block 60,000 from the 09:16 open: the account's margin is
    # highest, and its free cash lowest (40,000), first at the 09:17 round. Second holds
    # nothing.
    page = section_of(contention.tearsheet(), "portfolio")
    margin = page[page.index("Account Margin") : page.index("Refused by the Account")]
    values = re.findall(r"<div class='stat-value nums'>([^<]+)</div>", margin)
    assert values == ["&#8377;60,000", "&#8377;40,000", "&#8377;60,000", "&#8377;0.00"]
    assert margin.count("01 Jul 2025 09:17") == 2


def test_the_net_position_part_shows_the_shared_symbol_at_every_change(opposite):
    page = section_of(opposite.tearsheet(), "portfolio")
    net = page[page.index("<h3 class='serif'>SBIN (NSE, MIS)</h3>") :]
    assert cells(net) == [
        ["01 Jul 2025 09:16", "100", "0", "100"],
        ["01 Jul 2025 09:17", "100", "-100", "0"],
        ["01 Jul 2025 15:10", "0", "0", "0"],
    ]
    assert "Held on opposite sides first from 01 Jul 2025 09:17" in net


def test_the_overfitting_section_has_a_row_per_strategy_and_none_for_the_account(two_days_page):
    found = section_of(two_days_page, "overfitting")
    names = re.findall(r"<tr><td class='l'>([^<]+)</td>", found)
    assert names == ["LongDaily", "ShortDaily"]
    assert "Quick checks by strategy" in found
    assert "Quick checks of the trade list" not in found


def test_a_strategy_page_inside_a_portfolio_is_a_single_strategy_page(two_days):
    page = two_days.strategy("LongDaily").tearsheet()
    assert "portfolio" not in section_anchors(page)
    assert len(stylesheets(page)) == 1
    assert "Quick checks of the trade list" in section_of(page, "overfitting")


def test_the_basic_mode_leaves_out_the_strategies_and_portfolio_sections(two_days):
    page = two_days.tearsheet(mode="basic")
    anchors = section_anchors(page)
    assert "portfolio" not in anchors and "strategies" not in anchors
    assert "Quick checks by strategy" in section_of(page, "overfitting")
    assert len(stylesheets(page)) == 1


def test_the_page_is_written_to_a_path(two_days, two_days_page, tmp_path):
    target = tmp_path / "portfolio.html"
    assert two_days.tearsheet(target) == str(target)
    assert target.read_text(encoding="utf-8") == two_days_page


# ----- custom layouts on a portfolio --------------------------------------------------------------


def test_a_layout_of_the_full_base_alone_is_the_default_combined_page(two_days, two_days_page):
    assert two_days.tearsheet(layout=rp.Layout()) == two_days_page


def test_a_layout_hides_the_portfolio_section_and_its_series_colours(two_days):
    page = two_days.tearsheet(layout=rp.Layout(hide=("portfolio",)))
    assert "portfolio" not in section_anchors(page)
    assert len(stylesheets(page)) == 1


def test_a_layout_block_receives_the_portfolio(two_days):
    received = []

    def by_strategy(pf):
        received.append(pf)
        return pd.DataFrame({n: r.equity["equity_rupees"] for n, r in pf.results.items()})

    layout = rp.Layout(sections=(rp.section("Equity", rp.line(by_strategy, title="By strategy")),))
    page = two_days.tearsheet(layout=layout)
    assert received == [two_days]
    assert "u-equity" in section_anchors(page)
