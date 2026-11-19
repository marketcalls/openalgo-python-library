"""The Trades section of the tearsheet: the trading cost block, the "MAE and MFE"
subsection (tiles, histograms, scatter charts), the trade list's MAE and MFE, the
Charges table's slippage and trading cost, money tiles in paise below Rs 1,000, and
what each report mode shows of them.

Nine closed trades of `test_report_tearsheet.sample_trades`, Rs 5,000 at work each (10
units at 500.00): net P&L 100, -110, 120, -130, 140, -150, 160, -170, 180 and gross P&L
20 more (120, -90, 140, -110, 160, -130, 180, -150, 200, together 320), charges 20.00 and
slippage 2.00 each. MAE is the gross P&L capped at 0 less 25 (-25 for a winner, -115,
-135, -155, -175 for the others, together -705) and MFE the gross P&L floored at 0 plus
30 (150, 170, 190, 210, 230 for the winners, 30 for the others, together 1,070).
"""

import re

import numpy as np
import pandas as pd
import pytest
from test_metrics_excursion import analysed as analysed_with_an_open_trade
from test_metrics_excursion import trades_table as three_closed_and_one_open
from test_report_tearsheet import Structure, at, sample_trades, section_of, tile

from openalgo.backtest import metrics, report
from openalgo.backtest.report._dashboard.excursions import NO_CLOSED_TRADE, NO_COLUMNS

IST = "Asia/Kolkata"
CAPITAL_AT_WORK = 5_000.0
RUPEE = "&#8377;"


def closes() -> pd.DatetimeIndex:
    days = pd.bdate_range("2025-06-02", "2025-08-29", tz=IST)
    return days + pd.Timedelta(hours=15, minutes=30)


def excursion_trades() -> pd.DataFrame:
    trades = sample_trades(9, closes())
    gross = trades["gross_pnl"]
    trades["mae"] = gross.clip(upper=0.0) - 25.0
    trades["mfe"] = gross.clip(lower=0.0) + 30.0
    trades["mae_pct"] = trades["mae"] / CAPITAL_AT_WORK * 100
    trades["mfe_pct"] = trades["mfe"] / CAPITAL_AT_WORK * 100
    trades["slippage"] = 2.0
    return trades


def analysed(trades: pd.DataFrame | None) -> metrics.Performance:
    index = closes()
    rng = np.random.default_rng(9)
    values = 1_00_000 * np.cumprod(1 + rng.normal(0.0004, 0.008, len(index)))
    equity = pd.concat(
        [pd.Series([1_00_000.0], index=[at("2025-06-02", "09:15")]), pd.Series(values, index=index)]
    )
    return metrics.analyse(equity, exchanges="NSE", trades=trades, rolling_window=20)


@pytest.fixture(scope="module")
def page() -> str:
    return report.tearsheet(analysed(excursion_trades()))


def svg(page: str, label: str) -> str:
    """The inline SVG chart named `label`."""
    start = page.index(f"aria-label='{label}'")
    return page[start : page.index("</svg>", start)]


@pytest.mark.parametrize(
    ("label", "shown"),
    [
        # -705 / 9 = -78.33 of 5,000: -1.57 percent.
        ("Avg MAE", "-1.57%"),
        # 1,070 / 9 = 118.89 of 5,000: 2.38 percent.
        ("Avg MFE", "2.38%"),
        ("Worst MAE", f"-{RUPEE}175.00"),
        ("Best MFE", f"{RUPEE}230.00"),
        # 2.3778 / 1.5667.
        ("Edge Ratio", "1.52"),
        # The winners' gross 800 over their MFE 950.
        ("Profit Kept", "84.21%"),
    ],
)
def test_the_mae_and_mfe_tiles_show_the_excursion_figures(page, label, shown):
    assert tile(section_of(page, "trades"), label)[0] == shown


@pytest.mark.parametrize(
    ("label", "shown"),
    [
        # Entries 9 x 5,000 and exits 45,000 + 320, every trade closed.
        ("Turnover", f"{RUPEE}90,320"),
        ("Slippage", f"{RUPEE}18.00"),
        # Charges 180.00 and slippage 18.00.
        ("Trading Cost", f"{RUPEE}198.00"),
        # 198 of the gross 320.
        ("Cost / Gross P&amp;L", "61.88%"),
        # 198 / 90,320 x 10,000.
        ("Cost / Turnover", "21.9 bps"),
    ],
)
def test_the_trading_cost_block_shows_turnover_slippage_and_the_cost(page, label, shown):
    assert tile(section_of(page, "trades"), label)[0] == shown


def test_the_turnover_tile_is_the_closed_trades_the_cost_is_measured_against():
    # test_metrics_excursion's table: three closed trades traded 6,190.00 and one still
    # open was entered for 1,000.00; the trading cost 28.50 is 46.0 basis points of
    # 6,190.00 (of 7,190.00 it would be 39.6).
    page = report.tearsheet(analysed_with_an_open_trade(three_closed_and_one_open()))
    trades = section_of(page, "trades")
    assert tile(trades, "Turnover") == (f"{RUPEE}6,190", "of closed trades")
    assert tile(trades, "Cost / Turnover")[0] == "46.0 bps"


def test_the_trades_section_shows_the_charges_once_and_the_gross_pnl_beside_the_net(page):
    trades = section_of(page, "trades")
    assert trades.count(">Charges</div><div class='stat-value") == 1
    # Gross P&L 320.00 of the nine trades, then net 140.00 after charges of 180.00.
    assert tile(trades, "Gross P&amp;L") == (f"{RUPEE}320.00", "before charges")
    assert tile(trades, "Net P&amp;L")[0] == f"{RUPEE}140.00"


def test_the_subsection_draws_a_histogram_and_a_scatter_chart_of_each_excursion(page):
    trades = section_of(page, "trades")
    assert "<div id='mae-mfe'>" in trades and ">MAE and MFE</div>" in trades
    for label in ("MAE distribution", "MFE distribution", "MAE against return"):
        assert trades.count(f"aria-label='{label}'") == 1, label
    assert trades.count("aria-label='MFE against return'") == 1


@pytest.mark.parametrize("label", ["MAE against return", "MFE against return"])
def test_winners_are_circles_and_the_other_trades_crosses(page, label):
    chart = svg(page, label)
    # Five winners and four losers, and one of each mark in the legend.
    assert chart.count("<circle class='win'") == 5 + 1
    assert chart.count("<path class='loss'") == 4 + 1
    assert ">Win</text>" in chart and ">Loss or flat</text>" in chart


def test_a_scatter_mark_reads_its_trade_on_hover(page):
    # Trade 2: MAE -115 of 5,000 is -2.30 percent; net -110 over its entry value of
    # 5,000 is -2.20 percent.
    assert "<title>Trade 2: MAE -2.30%, return -2.20%</title>" in svg(page, "MAE against return")


def test_the_trade_list_shows_each_trades_mae_and_mfe(page):
    trades = section_of(page, "trades")
    assert "<th class='r s'>MAE</th><th class='r s'>MFE</th>" in trades
    assert f"title='-2.30% of capital at work'>-{RUPEE}115.00</td>" in trades


@pytest.mark.parametrize(
    ("label", "tallest"),
    [
        # The five winners' MAE is -25 of 5,000, -0.50 percent, all in one bin.
        ("MAE distribution", 5),
        # The four other trades' MFE is 30 of 5,000, 0.60 percent, all in one bin.
        ("MFE distribution", 4),
    ],
)
def test_each_excursion_histogram_has_a_count_axis(page, label, tallest):
    chart = svg(page, label)
    counts = re.findall(r"text-anchor='end' style='[^']*'>(\d+)</text>", chart)
    assert counts == [str(count) for count in range(tallest + 1)]
    assert ">trades</text>" in chart


def test_the_trade_list_keeps_every_column_and_wraps_a_time_between_date_and_time(page):
    trades = section_of(page, "trades")
    table = trades[trades.index("<table class='tbl hover trades'") :]
    head = table[: table.index("</thead>")]
    assert re.findall(r"<th class='[lr] s'>([^<]+)</th>", head) == [
        "#",
        "Symbol",
        "Side",
        "Entry (IST)",
        "Exit (IST)",
        "Qty",
        "Entry Price",
        "Exit Price",
        "Gross P&amp;L",
        "Charges",
        "Net P&amp;L",
        "Return",
        "MAE",
        "MFE",
        "Bars",
        "Days",
        "Exit Reason",
    ]
    # An entry and an exit time break only between the date and the time of day, so a
    # table wider than its card narrows its two time columns, not its figures.
    times = re.findall(r"<td class='l nums when' data-v='\d+'>([^<]+)</td>", table)
    assert len(times) == 2 * 9
    assert all(re.fullmatch(r"\d\d&#160;\w{3}&#160;\d{4} \d\d:\d\d", time) for time in times)
    assert ".trades td.when{white-space:normal}" in page


def test_the_charges_table_has_the_slippage_and_the_trading_cost_under_the_total(page):
    labels = re.findall(r"<tr class='total'><td class='l'>([^<]+)</td>", page)
    assert labels[:3] == ["Total charges", "Slippage (in fill prices)", "Trading cost"]


def test_money_tiles_show_paise_under_a_thousand_rupees():
    # One share: gross -1.20, charges 3.60, net -4.80; whole rupees from Rs 1,000 on.
    trades = sample_trades(1, closes())
    trades[["gross_pnl", "costs", "net_pnl"]] = [-1.2, 3.6, -4.8]
    trades["exit_price"] = 499.88
    page = report.tearsheet(analysed(trades))
    trade_tiles = section_of(page, "trades")
    assert tile(trade_tiles, "Gross P&amp;L")[0] == f"-{RUPEE}1.20"
    assert tile(trade_tiles, "Charges")[0] == f"{RUPEE}3.60"
    assert tile(trade_tiles, "Net P&amp;L")[0] == f"-{RUPEE}4.80"
    assert tile(trade_tiles, "Turnover")[0] == f"{RUPEE}9,999"


def test_without_excursion_columns_the_subsection_says_so_in_one_line():
    page = report.tearsheet(analysed(sample_trades(9, closes())))
    trades = section_of(page, "trades")
    assert f"<p class='empty'>{NO_COLUMNS}</p>" in trades
    assert "aria-label='MAE distribution'" not in trades
    assert "<th class='r s'>MAE</th>" not in trades
    assert tile(trades, "Slippage")[0] == "n/a"


def test_without_a_closed_trade_the_subsection_says_so_in_one_line():
    trades = section_of(report.tearsheet(analysed(None)), "trades")
    assert f"<p class='empty'>{NO_CLOSED_TRADE}</p>" in trades


@pytest.mark.parametrize(
    ("mode", "shown", "hidden"),
    [
        ("full", ("Avg MAE", "MAE distribution", "MAE against return", "Trading Cost"), ()),
        ("basic", ("MAE distribution", "Trading Cost"), ("Avg MAE", "MAE against return")),
        ("charts", ("MAE distribution", "MAE against return"), ("Avg MAE", "Trading Cost")),
        ("metrics", ("Average MAE",), ("MAE distribution", "Trading Cost</div>")),
    ],
)
def test_each_mode_shows_its_parts(mode, shown, hidden):
    page = report.tearsheet(analysed(excursion_trades()), mode=mode)
    for text in shown:
        assert text in page, (mode, text)
    for text in hidden:
        assert text not in page, (mode, text)


def test_the_page_stays_well_formed_ascii(page):
    structure = Structure(page)
    assert structure.errors == [] and structure.stack == [] and page.isascii()
