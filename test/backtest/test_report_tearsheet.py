"""The HTML tearsheet, a dashboard: self-contained, its sections in order, its figures
those of the metrics, well formed, and complete as static HTML with script off."""

import hashlib
import re
from collections import Counter
from html.parser import HTMLParser
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openalgo.backtest import metrics, report
from openalgo.backtest.errors import ConfigError
from openalgo.backtest.report._dashboard import data

IST = "Asia/Kolkata"
# Every section of the full report, in page order.
SECTIONS = (
    "overview",
    "performance",
    "risk",
    "monthly",
    "distribution",
    "metrics",
    "trades",
    "notes",
)
# Elements that never have a closing tag in HTML.
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "wbr"}
# The first 16 hex digits of the SHA-256 of names that must never appear in the page:
# chart, table, styling and analytics libraries and trading platforms. Kept as hashes so
# that this file does not spell them out.
FORBIDDEN_NAME_HASHES = frozenset(
    {
        "b6456c2513abc1c5",
        "310206690a5a7b57",
        "f451a61749c611ba",
        "082fca0433a6e39a",
        "da8ea8c9f72bd6bb",
        "46b3369be76a889a",
        "8b0580d3326507f3",
        "275976081ce1abf6",
        "cb88b41d223c7f04",
        "44c8cf4ce0611762",
        "bb8e94b352aa28a8",
        "d276630b7ec4a6c2",
        "04c1acc0cd687a47",
        "2b15fcd19d49b750",
        "ee54b596e975ec7c",
        "5536a646767eeca5",
        "db1530a0d085c987",
        "f84035fd958129fc",
        "aeb6dd081c371827",
        "b20aef8327cbd7ad",
        "8e7e3b7ba0b890bb",
        "4d93dddb39519a93",
        "1fcf716d138462ac",
        "c888fa0d640bdf48",
        "e134c033e88fbff4",
        "6a97fe4adfb3dc73",
        "6dbc69e4be68f9bd",
        "00862f08ca12c5b0",
        "af8599bd9665d318",
        "365107e5b93c9539",
        "248f10eab2bf15fd",
        "917bd93adcff5b14",
    }
)


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
    # Four months of sessions: a 20-session window gives every rolling chart values.
    return metrics.analyse(
        equity,
        exchanges="NSE",
        trades=trades,
        benchmark=bench,
        risk_free_rate=0.065,
        rolling_window=20,
    )


@pytest.fixture(scope="module")
def page(performance) -> str:
    return report.tearsheet(performance)


def section_ids(page: str) -> list[str]:
    """The ids of the page's numbered sections, in page order."""
    return re.findall(r"<section id='([a-z-]+)'>", page)


def section_of(page: str, anchor: str) -> str:
    part = page[page.index(f"<section id='{anchor}'") :]
    return part[: part.index("</section>")]


def without_script(page: str) -> str:
    """The page as a browser with script turned off reads it."""
    return re.sub(r"<script>.*?</script>", "", page, flags=re.S)


def tile(page: str, label: str) -> tuple[str, str]:
    """The value and subline of the first tile labelled `label`."""
    found = re.search(
        rf"title='[^']*'>{re.escape(label)}</div><div class='stat-value nums[^']*'>([^<]*)</div>"
        r"(?:<div class='stat-sub nums'>([^<]*)</div>)?",
        page,
    )
    assert found, label
    return found.group(1), found.group(2) or ""


def test_the_page_is_returned_as_a_string_without_a_path(page):
    assert page.startswith("<!DOCTYPE html>")


def test_the_page_is_written_to_a_path_and_the_path_returned(performance, tmp_path, page):
    target = tmp_path / "report.html"
    assert report.tearsheet(performance, target) == str(target)
    assert target.read_text(encoding="utf-8") == page


def test_the_page_contains_every_section_in_order(page):
    assert section_ids(page) == list(SECTIONS)


def test_the_header_links_to_every_section_in_order(page):
    links = re.findall(r"<a href='#([a-z-]+)'", page)
    assert links == list(SECTIONS)


def test_sections_are_numbered_in_page_order(page):
    numbers = re.findall(r"<span class='sec-num nums'>(\d+)</span>", page)
    assert numbers == [f"{i:02d}" for i in range(1, len(SECTIONS) + 1)]


def test_the_page_fetches_nothing(page):
    lower = page.lower()
    for forbidden in ("http://", "https://", "@import", "<link", "<iframe", "<img", "src="):
        assert forbidden not in lower, forbidden
    # The only url() is a gradient fill defined in the page itself.
    assert re.findall(r"url\(([^)]*)\)", lower)
    assert all(target.startswith("#") for target in re.findall(r"url\(([^)]*)\)", lower))
    hrefs = [value for name, value in Structure(page).attributes if name == "href"]
    assert hrefs and all(value.startswith("#") for value in hrefs)


def test_the_page_names_no_third_party_product(page):
    tokens = set(re.findall(r"[a-z0-9][a-z0-9.\-]*", page.lower()))
    tokens |= {part for token in tokens for part in token.split("-")}
    hashed = {hashlib.sha256(token.encode()).hexdigest()[:16]: token for token in tokens}
    assert not set(hashed) & FORBIDDEN_NAME_HASHES, [
        hashed[h] for h in FORBIDDEN_NAME_HASHES & set(hashed)
    ]


def test_every_tag_is_properly_nested_and_closed(page):
    structure = Structure(page)
    assert structure.errors == [] and structure.stack == []
    assert (structure.tags["html"], structure.tags["head"], structure.tags["body"]) == (1, 1, 1)


def test_the_page_is_plain_ascii_with_character_entities(page):
    assert page.isascii()


def test_the_page_follows_light_and_dark_settings_and_phone_width(page):
    assert "@media (prefers-color-scheme: dark)" in page
    assert ":root[data-theme=dark]" in page
    assert "width=device-width" in page and "@media print" in page


def test_the_overview_tiles_show_the_metrics(performance, page):
    m = performance.metrics
    assert tile(page, "Cumulative Return")[0] == f"{m['total_return'] * 100:.2f}%"
    assert tile(page, "CAGR")[0] == f"{m['cagr'] * 100:.2f}%"
    assert tile(page, "Sharpe")[0] == f"{m['sharpe']:.2f}"
    assert tile(page, "Max Drawdown")[0] == f"{m['max_drawdown'] * 100:.2f}%"
    assert tile(page, "Time in Market")[0] == f"{m['exposure'] * 100:.2f}%"


def test_a_tile_shows_the_benchmarks_figure_under_the_strategys(performance, page):
    # The benchmark measured as the strategy is compounds the same known returns.
    bench = data.benchmark_performance(performance)
    assert bench.metrics["total_return"] == pytest.approx(
        performance.metrics["benchmark_total_return"]
    )
    expected = f"Benchmark {bench.metrics['total_return'] * 100:.2f}%"
    assert tile(page, "Cumulative Return")[1] == expected


def test_a_trade_figure_has_no_benchmark_subline(page):
    assert tile(page, "Time in Market")[1] == "exposure"


def test_the_benchmark_tiles_show_the_metrics(performance, page):
    m = performance.metrics
    assert tile(page, "Beta")[0] == f"{m['beta']:.2f}"
    assert tile(page, "Correlation")[0] == f"{m['correlation']:.2f}"


def test_the_metrics_table_lists_every_metric_with_the_strategys_value(performance, page):
    table = section_of(page, "metrics")
    keys = [k for k in report.metrics_table(performance).index]
    assert table.count("<tr data-i=") == len(keys)
    assert f">{performance.metrics['sharpe']:.2f}</td>" in table


def test_charts_are_inline_svg(page):
    # Cumulative and three rolling statistics; drawdown and rolling beta; yearly bars and
    # streaks; histogram and box plot.
    assert section_of(page, "performance").count("<svg") == 4
    assert section_of(page, "risk").count("<svg") == 3
    assert section_of(page, "monthly").count("<svg") == 1
    assert section_of(page, "distribution").count("<svg") == 2


def test_the_cumulative_chart_carries_its_values_for_the_crosshair(performance, page):
    chart = section_of(page, "performance")
    assert "data-v0='" in chart and "data-v1='" in chart  # the strategy and the benchmark
    dates = re.search(r"<body data-dates='([^']*)'", page).group(1).split(",")
    assert len(dates) == len(performance.session_returns)


def test_the_page_reads_in_full_without_script(performance, page):
    static = without_script(page)
    assert section_ids(static) == list(SECTIONS)
    trades = section_of(static, "trades")
    assert trades.count("<tr data-i=") == performance.metrics["trades"]
    assert "class='paged'" not in static  # paging is applied by script only
    weekly = static.count("class='wk-year")
    assert weekly == len(data.weekly(performance.session_returns))


def test_only_script_hides_rows_weeks_and_controls(page):
    css = page[page.index("<style>") : page.index("</style>")]
    assert ".js .tbl tr.paged{display:none}" in css
    assert ".js .wk-year:not(.on){display:none}" in css
    assert "html:not(.js) .js-only{display:none!important}" in css
    head = page[: page.index("</head>")]
    assert "d.classList.add('js')" in head


def test_costs_are_broken_down_by_component(page):
    costs = section_of(page, "trades")
    assert "Brokerage" in costs and "STT / CTT" in costs
    # 25 trades at Rs 15 and Rs 5.
    assert "&#8377;375.00" in costs and "&#8377;125.00" in costs


def test_assumptions_name_and_benchmark_come_from_a_result(performance):
    result = SimpleNamespace(
        performance=performance,
        name="Pairs <A & B>",
        assumptions={"fill": "next_open", "slippage": "1 tick"},
        benchmark_name="NIFTY",
    )
    page = report.tearsheet(result)
    assert "<td class='l'>fill</td><td class='l'>next_open</td>" in page
    assert "Pairs &lt;A &amp; B&gt;" in page
    assert "vs NIFTY" in page and "vs Benchmark &#183; NIFTY" in page


def test_a_warning_is_shown_as_one(performance):
    result = SimpleNamespace(performance=performance, assumptions=["warning: no charges"])
    page = report.tearsheet(result)
    assert "<tr class='warn'><td class='l'>warning</td><td class='l'>no charges</td></tr>" in page


def test_a_title_argument_overrides_the_name(performance):
    result = SimpleNamespace(performance=performance, name="Engine name")
    assert "<h1 class='serif'>Chosen title</h1>" in report.tearsheet(result, title="Chosen title")


def test_several_strategies_add_a_comparison_and_correlation(performance):
    result = SimpleNamespace(
        performance=performance, strategies={"one": performance, "two": performance}
    )
    page = report.tearsheet(result)
    assert section_ids(page) == [*SECTIONS[:-1], "strategies", "notes"]
    assert "Head to Head" in page and "Of session returns" in page
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
    assert "No closed trades." in page
    assert section_ids(page) == list(SECTIONS)


def test_a_run_shorter_than_the_rolling_window_says_so():
    page = report.tearsheet(metrics.analyse(edge_cases()["flat"], exchanges="NSE"))
    assert "the run has fewer sessions than one" in section_of(page, "performance")


def test_a_ruined_account_is_said_in_the_notes():
    page = report.tearsheet(metrics.analyse(edge_cases()["ruined"], exchanges="NSE"))
    assert "The account was ruined on 03 Jun 2025 15:30." in page
