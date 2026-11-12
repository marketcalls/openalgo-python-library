"""The HTML tearsheet: self-contained, every section present, tags properly nested."""

from collections import Counter
from html.parser import HTMLParser
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openalgo.backtest import metrics, report
from openalgo.backtest.errors import ConfigError

IST = "Asia/Kolkata"
SECTIONS = (
    "summary",
    "equity",
    "monthly",
    "yearly",
    "distribution",
    "trades",
    "costs",
    "assumptions",
    "notes",
)
# Elements that never have a closing tag in HTML.
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "wbr"}


class Structure(HTMLParser):
    """Records tags, attributes and every nesting error of a page."""

    def __init__(self, page: str):
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.tags: Counter = Counter()
        self.attributes: list[tuple[str, str | None]] = []
        self.feed(page)
        self.close()

    def handle_starttag(self, tag, attrs):
        self.tags[tag] += 1
        self.attributes.extend(attrs)
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.tags[tag] += 1
        self.attributes.extend(attrs)

    def handle_endtag(self, tag):
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        else:
            self.errors.append(f"</{tag}> closes {self.stack[-1:] or 'nothing'}")


def at(day: str, clock: str = "15:30") -> pd.Timestamp:
    return pd.Timestamp(f"{day} {clock}", tz=IST)


def sample_trades(count: int, closes: pd.DatetimeIndex) -> pd.DataFrame:
    net = np.array([(-1) ** i * (100.0 + 10 * i) for i in range(count)])
    return pd.DataFrame(
        {
            "symbol": "SBIN",
            "entry_time": closes[:count],
            "exit_time": closes[1 : count + 1],
            "entry_price": 500.0,
            "exit_price": 500.0 + (net + 20.0) / 10,
            "quantity": 10.0,
            "side": ["BUY", "SELL"] * (count // 2) + ["BUY"] * (count % 2),
            "gross_pnl": net + 20.0,
            "net_pnl": net,
            "costs": 20.0,
            "brokerage": 15.0,
            "stt": 5.0,
        }
    )


@pytest.fixture(scope="module")
def performance() -> metrics.Performance:
    closes = pd.bdate_range("2025-06-02", "2025-09-30", tz=IST) + pd.Timedelta(hours=15, minutes=30)
    rng = np.random.default_rng(3)
    values = 1_00_000 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(closes)))
    equity = pd.concat(
        [
            pd.Series([1_00_000.0], index=[at("2025-06-02", "09:15")]),
            pd.Series(values, index=closes),
        ]
    )
    bench = pd.Series(24_000 * np.cumprod(1 + rng.normal(0.0003, 0.008, len(closes))), index=closes)
    trades = sample_trades(25, closes)
    return metrics.analyse(
        equity, exchanges="NSE", trades=trades, benchmark=bench, risk_free_rate=0.065
    )


@pytest.fixture(scope="module")
def page(performance) -> str:
    return report.tearsheet(performance)


def test_the_page_is_returned_as_a_string_without_a_path(page):
    assert page.startswith("<!DOCTYPE html>")


def test_the_page_is_written_to_a_path_and_the_path_returned(performance, tmp_path, page):
    target = tmp_path / "report.html"
    assert report.tearsheet(performance, target) == str(target)
    assert target.read_text(encoding="utf-8") == page


def test_the_page_fetches_nothing_and_runs_nothing(page):
    lower = page.lower()
    for forbidden in (
        "<script",
        "http://",
        "https://",
        "@import",
        "url(",
        "<link",
        "<iframe",
        "<img",
    ):
        assert forbidden not in lower, forbidden
    names = {name for name, _ in Structure(page).attributes}
    assert not names & {"src", "href", "onload", "onclick", "onmouseover"}


def test_the_page_contains_every_section(page):
    structure = Structure(page)
    ids = {value for name, value in structure.attributes if name == "id"}
    assert set(SECTIONS) <= ids
    assert structure.tags["section"] == len(SECTIONS)


def test_every_tag_is_properly_nested_and_closed(page):
    structure = Structure(page)
    assert structure.errors == []
    assert structure.stack == []
    assert (structure.tags["html"], structure.tags["head"], structure.tags["body"]) == (1, 1, 1)


def test_the_page_follows_light_and_dark_settings_and_phone_width(page):
    assert "@media (prefers-color-scheme: dark)" in page
    assert "width=device-width" in page
    assert "color-scheme" in page


def test_the_page_is_plain_ascii_with_character_entities(page):
    assert page.isascii()


def test_charts_are_inline_svg_and_the_histogram_marks_gains_and_losses(page):
    structure = Structure(page)
    assert structure.tags["svg"] == 2
    assert "class='bin loss'" in page and "class='bin gain'" in page


def test_the_trade_list_shows_the_first_and_last_rows_with_a_count(page):
    trades = page[page.index("<section id='trades'") :]
    trades = trades[: trades.index("</section>")]
    assert trades.count("<tr>") == 1 + 20
    assert "5 trades not shown" in trades
    assert "<td class='num'>1</td>" in trades and "<td class='num'>25</td>" in trades
    assert "<td class='num'>11</td>" not in trades


def test_costs_are_broken_down_by_component(page):
    costs = page[page.index("<section id='costs'") :]
    assert "Brokerage" in costs and "STT / CTT" in costs
    # 25 trades at Rs 15 and Rs 5.
    assert "&#8377;375.00" in costs and "&#8377;125.00" in costs


def test_assumptions_and_name_come_from_a_result(performance):
    result = SimpleNamespace(
        performance=performance,
        name="Pairs <A & B>",
        assumptions={"fill": "next_open", "slippage": "1 tick"},
    )
    page = report.tearsheet(result)
    assert "<li>fill: next_open</li>" in page and "<li>slippage: 1 tick</li>" in page
    assert "Pairs &lt;A &amp; B&gt;" in page
    assert Structure(page).tags["a"] == 0


def test_a_title_argument_overrides_the_name(performance):
    result = SimpleNamespace(performance=performance, name="Engine name")
    assert "<h1>Chosen title</h1>" in report.tearsheet(result, title="Chosen title")


def test_several_strategies_add_a_comparison_and_correlation(performance):
    result = SimpleNamespace(
        performance=performance, strategies={"one": performance, "two": performance}
    )
    page = report.tearsheet(result)
    assert "id='strategies'" in page
    assert "Correlation of session returns" in page
    assert Structure(page).errors == []


def test_an_object_without_a_performance_is_refused():
    with pytest.raises(ConfigError, match="Performance"):
        report.tearsheet(SimpleNamespace(name="x"))


def edge_cases() -> dict[str, pd.Series]:
    start = at("2025-06-02", "09:15")
    closes = pd.bdate_range("2025-06-02", periods=5, tz=IST) + pd.Timedelta(hours=15, minutes=30)

    def curve(*values: float) -> pd.Series:
        return pd.Series(values, index=[start, *closes[: len(values) - 1]])

    return {
        "flat": curve(100.0, 100.0, 100.0, 100.0),
        "ruined": curve(100.0, 50.0, 0.0, 10.0),
        "single session": curve(100.0, 101.0),
    }


@pytest.mark.parametrize("name", list(edge_cases()))
def test_edge_cases_render_a_complete_page(name):
    page = report.tearsheet(metrics.analyse(edge_cases()[name], exchanges="NSE"))
    structure = Structure(page)
    assert structure.errors == [] and structure.stack == []
    assert "No closed trades." in page and "No trades." in page
    ids = {value for key, value in structure.attributes if key == "id"}
    assert set(SECTIONS) <= ids


def test_a_ruined_account_is_said_in_the_notes():
    page = report.tearsheet(metrics.analyse(edge_cases()["ruined"], exchanges="NSE"))
    assert "The account was ruined on 03 Jun 2025 15:30." in page


def test_an_open_trade_is_listed_without_an_exit():
    closes = pd.bdate_range("2025-06-02", periods=4, tz=IST) + pd.Timedelta(hours=15, minutes=30)
    equity = pd.Series([100.0, 101.0, 102.0, 103.0], index=closes)
    trades = sample_trades(1, closes)
    trades.loc[0, "exit_time"] = pd.NaT
    trades.loc[0, ["exit_price", "gross_pnl", "net_pnl", "costs"]] = np.nan
    page = report.tearsheet(metrics.analyse(equity, exchanges="NSE", trades=trades))
    assert "0 closed and 1 open trade(s)" in page
    assert Structure(page).errors == []
