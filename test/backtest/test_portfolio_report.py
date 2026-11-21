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
from custom_pages import (
    element_ids,
    header_links,
    judged_quick_checks,
    section_anchors,
    section_of,
)
from portfolio_samples import DAYS, Scripted, flat_bars, no_costs, stepped_bars
from strategy_samples import CAPITAL as STRATEGY_CAPITAL
from strategy_samples import daily_bars
from test_portfolio_run import THREE, THREE_RUN, DailyInfy, MinuteSbin

import openalgo.backtest as bt
from openalgo.backtest import metrics
from openalgo.backtest import report as rp
from openalgo.backtest._messages import ALONE_POINTER
from openalgo.backtest._portfolio_report import PER_STRATEGY
from openalgo.backtest._result_analysis import AnalysisMethods
from openalgo.backtest._robust._quick import POINTER as QUICK_POINTER
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


def test_on_one_base_grid_a_separate_pool_account_is_the_combined_equity_of_its_strategies():
    # Fixed allocations of strategies that all close their slices on one 1-minute grid: the
    # account is the strategies' equities added up on the union of their times, as
    # metrics.combine_equity adds them with each strategy's capital.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pf = bt.backtest(THREE, pool="separate", **THREE_RUN)
    equities = {name: result.equity["equity_rupees"] for name, result in pf.results.items()}
    capitals = dict.fromkeys(pf.results, STRATEGY_CAPITAL)
    combined = metrics.combine_equity(equities, opening_wealth=capitals)
    assert pf.equity.index.tolist() == combined.index.tolist()
    assert pf.equity.tolist() == pytest.approx(combined.tolist(), abs=0.005)


def test_beside_a_daily_strategy_the_account_counts_its_open_fill_before_its_own_equity_does():
    # MinuteSbin (1-minute bars) and DailyInfy (daily bars) on separate allocations of
    # Rs 1,00,000, with the default charges and no slippage, 1 to 4 July. DailyInfy's own
    # equity moves only at its 15:30 closes, while the account takes its fills from the
    # 09:15 open of their session (18-multi-strategy 2.3): the fill's charges paid, to the
    # paisa, and the units at the fill price. So the account equals metrics.combine_equity
    # at every 15:30 close, and its rounds from 09:16 to 15:29 differ from it by:
    # - 1 and 3 July: nothing, DailyInfy does not fill;
    # - 2 July: its buy of 10 INFY at the 101.00 open, nothing held before, so the charges
    #   only: STT 1.01, exchange 0.031007, SEBI 0.00101, stamp duty 0.1515 and GST 18% of
    #   0.032017 (0.0057631), 1.1992801 or -1.20 to the paisa;
    # - 4 July: its sale of the 10 at the 103.00 open, held at the 3 July close of 102.05:
    #   10 x 0.95 = 9.50 of price, less STT 1.03, exchange 0.031621, SEBI 0.00103, GST
    #   0.0058772 and the DP charge of 15.34 (16.4085282), -6.9085282 or -6.91.
    minutes = pd.concat([flat_bars(100.0 + number, day=day) for number, day in enumerate(DAYS)])
    data = {"SBIN": minutes, "INFY": daily_bars(DAYS)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pf = bt.backtest(
            [MinuteSbin, DailyInfy], pool="separate", data=data, capital=LAKH, slippage=0
        )
    equities = {name: result.equity["equity_rupees"] for name, result in pf.results.items()}
    combined = metrics.combine_equity(equities, opening_wealth=dict.fromkeys(pf.results, LAKH))
    assert pf.equity.index.tolist() == combined.index.tolist()
    gap = pd.Series(pf.equity.to_numpy() - combined.to_numpy(), index=pf.equity.index)
    inside = {DAYS[0]: 0.0, DAYS[1]: -1.20, DAYS[2]: 0.0, DAYS[3]: -6.91}
    for day, expected in inside.items():
        rounds = gap[gap.index.date == day]
        assert rounds.index[-1].time() == dt.time(15, 30)
        assert rounds.iloc[-1] == pytest.approx(0.0, abs=1e-6)
        assert rounds.iloc[:-1].to_numpy() == pytest.approx(expected, abs=1e-6)


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


class Penny(Scripted):
    qty = 20


def test_the_summary_of_an_account_that_lost_a_rupee_reads_a_return_and_drawdown_of_zero():
    # Penny buys 20 SBIN at the 09:16 open at 100.00; SBIN is 99.95 (one tick lower) from
    # the bar opening at 09:25 to the 15:10 square-off: 20 x -0.05 = -1.00 on an account of
    # Rs 2,00,000, a return and a maximum drawdown of -0.0005 percent, which read 0.00%
    # with no minus sign.
    pf = bt.backtest(
        [Penny, Watcher], data={"SBIN": stepped_bars(100.0, 99.95, 10)}, capital=LAKH, **no_costs()
    )
    assert pf.metrics["max_drawdown"] < 0
    assert pf.summary().splitlines()[:4] == [
        "Net profit   : Rs -1.00",
        "Return       : 0.00%",
        "Max drawdown : 0.00%",
        "Trades       : 1 (0% won)",
    ]


def test_the_repr_reads_a_net_pnl_below_half_a_paisa_without_a_minus_sign():
    # Penny buys and squares off 20 SBIN at 100.00 all day: the account ends at its funds of
    # Rs 2,00,000. The same records with funds 0.004 rupees higher end 0.004 below them,
    # less than half a paisa, which reads "0.00" as every rupee figure of a report does.
    pf = bt.backtest([Penny, Watcher], data={"SBIN": flat_bars(100.0)}, capital=LAKH, **no_costs())
    records = dict(pf._records) | {"account_capital": pf._records["account_capital"] + 0.004}
    shifted = bt.PortfolioResult(pf._account, pf.results, records)
    assert repr(shifted).splitlines()[2] == "  net P&L: 0.00"


class Big(Scripted):
    qty = 10_000


def test_the_repr_reads_the_net_pnl_in_indian_digits():
    # Big buys 10,000 SBIN at the 09:16 open at 100.00; SBIN is 120.00 from the bar opening
    # at 09:25 to the 15:10 square-off: 10,000 x 20.00 = Rs 2,00,000.
    data = {"SBIN": stepped_bars(100.0, 120.0, 10)}
    pf = bt.backtest([Big, Watcher], data=data, capital=10 * LAKH, **no_costs())
    assert repr(pf).splitlines()[2] == "  net P&L: 2,00,000.00"


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


@pytest.fixture
def judged_loss():
    # The loss run, with Faller's quick checks those of `judged_quick_checks` (40 closed
    # trades, variance failed), set where its Result caches its own.
    pf = bt.backtest(
        [Faller, Watcher], data={"SBIN": stepped_bars(100.0, 95.0, 10)}, capital=LAKH, **no_costs()
    )
    pf.strategy("Faller").__dict__["_quick"] = judged_quick_checks()
    return pf


def test_the_summary_line_of_a_judged_strategy_counts_its_verdicts_and_names_the_failed_one(
    judged_loss,
):
    assert judged_loss.summary().splitlines()[5:] == [
        "  Faller     : quick checks: 2 passed, 1 failed (variance), 1 informational",
        "  Watcher    : too few trades to judge (0 closed, 30 needed)",
    ]


def test_a_failed_quick_check_of_a_strategy_points_to_a_run_of_it_alone(judged_loss):
    # A strategy of a portfolio cannot call robustness(), so the portfolio's warnings, and
    # the Notes of its page, point to a run of the strategy alone; the strategy's own
    # Result keeps the single run's pointer.
    failed = [line for line in judged_loss.warnings if "Overfitting check failed" in line]
    assert failed == [
        "Faller: Overfitting check failed, variance: p = 0.20, where the rule is p below "
        "0.05. The edge is fragile or small next to the noise of the run's own trades; "
        "result.robustness() on a run of the strategy alone with bt.backtest runs the full "
        "set of checks, with charts"
    ]
    assert escaped(failed[0]) in section_of(judged_loss.tearsheet(), "notes")
    own = judged_loss.strategy("Faller").warnings[-1]
    assert own.endswith("own trades; result.robustness() runs the full set of checks, with charts")


def test_a_members_warnings_differ_from_its_own_only_in_the_pointer(judged_loss):
    # The portfolio names each line of a strategy's warnings by the strategy; the line of a
    # failed quick check points to a run of the strategy alone where its own result
    # points to its robustness().
    faller = judged_loss.strategy("Faller")
    named = [line for line in judged_loss.warnings if line.startswith("Faller: ")]
    own = [line.removeprefix("Faller: ") for line in named]
    assert len(own) == len(faller.warnings)
    assert [line.replace(ALONE_POINTER, QUICK_POINTER) for line in own] == faller.warnings


def test_a_line_the_warnings_gain_reaches_the_portfolio_named_by_its_strategy(
    judged_loss, monkeypatch
):
    # One composition makes a Result's warnings, the Notes of its page and its lines in a
    # portfolio: a kind of line it gains is in each of them.
    composed = AnalysisMethods._all_warnings
    later = "a later kind of warning"
    monkeypatch.setattr(
        AnalysisMethods,
        "_all_warnings",
        lambda result, pointer=None: [*composed(result, pointer), later],
    )
    assert judged_loss.strategy("Faller").warnings[-1] == later
    assert f"Faller: {later}" in judged_loss.warnings
    assert escaped(later) in section_of(judged_loss.strategy("Faller").tearsheet(), "notes")


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


def trade_list_of(page: str) -> str:
    """The trade list table of a page's Trades section."""
    trades = section_of(page, "trades")
    listing = trades[trades.index("Trade List") :]
    return listing[: listing.index("</table>")]


def test_the_combined_trade_list_names_the_strategy_of_each_row(two_days_page):
    # Each strategy numbers its own trades from 1, so a row is told by its strategy and
    # its number together.
    listing = trade_list_of(two_days_page)
    heads = "<th class='r s'>#</th><th class='l s'>Strategy</th><th class='l s'>Symbol</th>"
    assert heads in listing
    assert sorted(row[:4] for row in cells(listing)) == [
        ["1", "LongDaily", "SBIN", "Long"],
        ["1", "ShortDaily", "SBIN", "Short"],
        ["2", "LongDaily", "SBIN", "Long"],
        ["2", "ShortDaily", "SBIN", "Short"],
    ]


def test_the_combined_trade_list_is_in_the_order_the_trades_closed(two_days_page):
    # LongDaily and ShortDaily each close a trade at the 15:10 square-off of 1 July and
    # of 2 July: in closing order, those of one instant in call order, each row marked
    # by its place in the list (a sort's ties fall back on it).
    listing = trade_list_of(two_days_page)
    places = re.findall(r"<tr data-i='(\d+)'>", listing)
    assert places == ["1", "2", "3", "4"]
    assert [row[:2] for row in cells(listing)] == [
        ["1", "LongDaily"],
        ["1", "ShortDaily"],
        ["2", "LongDaily"],
        ["2", "ShortDaily"],
    ]


def excursion_titles(page: str, name: str) -> list[str]:
    """The hover readouts of the marks of the `name` ("MAE" or "MFE") scatter of a page's
    Trades section, in the order they are drawn."""
    trades = section_of(page, "trades")
    chart = trades[trades.index(f"aria-label='{name} against return'") :]
    chart = chart[: chart.index("</svg>")]
    return re.findall(r"<title>([^<]*)</title>", chart)


def test_each_mark_of_the_excursion_scatters_names_its_strategy_and_trade(two_days_page):
    # Each strategy numbers its own trades from 1, so a mark is told by its strategy and
    # its number together, as a row of the combined trade list is. 100 units on 10,000 of
    # capital at work: LongDaily makes +10.00% on 1 July (never below its entry, MAE 0,
    # MFE +10.00%) and -5.00% on 2 July (MAE -5.00%, MFE 0); ShortDaily the mirror.
    assert sorted(excursion_titles(two_days_page, "MAE")) == [
        "LongDaily trade 1: MAE 0.00%, return 10.00%",
        "LongDaily trade 2: MAE -5.00%, return -5.00%",
        "ShortDaily trade 1: MAE -10.00%, return -10.00%",
        "ShortDaily trade 2: MAE 0.00%, return 5.00%",
    ]
    assert sorted(excursion_titles(two_days_page, "MFE")) == [
        "LongDaily trade 1: MFE 10.00%, return 10.00%",
        "LongDaily trade 2: MFE 0.00%, return -5.00%",
        "ShortDaily trade 1: MFE 0.00%, return -10.00%",
        "ShortDaily trade 2: MFE 5.00%, return 5.00%",
    ]


def test_a_strategy_page_inside_a_portfolio_numbers_its_marks_by_trade_alone(two_days):
    page = two_days.strategy("LongDaily").tearsheet()
    assert excursion_titles(page, "MAE") == [
        "Trade 1: MAE 0.00%, return 10.00%",
        "Trade 2: MAE -5.00%, return -5.00%",
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


def test_the_report_tearsheet_of_a_portfolio_is_its_combined_page(two_days, two_days_page):
    assert rp.tearsheet(two_days) == two_days_page


def test_the_report_tearsheet_of_a_portfolio_takes_a_title(two_days):
    assert "<h1 class='serif'>Two sides</h1>" in rp.tearsheet(two_days, title="Two sides")


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
