"""Custom tearsheets, the user blocks: what each kind takes and how it is drawn, the
checks when a block is made and when its data arrives, and the line charts' alignment
to the run's sessions (design 19, section 1.2). Expected values are worked by hand
from the inputs of each test."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
from custom_pages import IST, section_of, small_performance

from openalgo.backtest import report as rp
from openalgo.backtest.errors import ConfigError
from openalgo.backtest.report._custom._charts import on_sessions

NO_DATA = "<p class='empty'>No data</p>"


@pytest.fixture(scope="module")
def performance():
    return small_performance()


@pytest.fixture(scope="module")
def sessions(performance) -> pd.DatetimeIndex:
    return performance.session_returns.index


def drawn(performance, *blocks: rp.custom.Block) -> str:
    """The HTML of a custom section "Blocks" holding `blocks`."""
    layout = rp.Layout(sections=(rp.section("Blocks", *blocks),))
    return section_of(rp.tearsheet(performance, layout=layout), "u-blocks")


def page(performance, *blocks: rp.custom.Block) -> str:
    return rp.tearsheet(performance, layout=rp.Layout(sections=(rp.section("Blocks", *blocks),)))


# ----- a line on the run's sessions ---------------------------------------------------------

DAYS = pd.DatetimeIndex(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06"])


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        # The last value of a date counts for it; a date with none keeps the last known.
        ({"2025-01-01 10:00": 1.0, "2025-01-01 15:00": 2.0, "2025-01-03 09:30": 5.0}, [2, 2, 5, 5]),
        # A series that starts later is blank before its first value.
        ({"2025-01-02": 3.0}, [np.nan, 3, 3, 3]),
        # A missing value is not a value: the last known one stands.
        ({"2025-01-01": 1.0, "2025-01-02": np.nan}, [1, 1, 1, 1]),
        # Points out of order are read in time order.
        ({"2025-01-03": 5.0, "2025-01-01": 1.0}, [1, 1, 5, 5]),
    ],
)
def test_a_line_takes_the_last_known_value_on_or_before_each_session(points, expected):
    series = pd.Series(list(points.values()), index=pd.DatetimeIndex(list(points)))
    np.testing.assert_array_equal(on_sessions(series, DAYS), np.array(expected, dtype=float))


def test_a_time_with_a_zone_counts_for_its_ist_date():
    # 20:00 UTC on 1 January is 01:30 IST on 2 January.
    series = pd.Series([7.0], index=pd.DatetimeIndex(["2025-01-01 20:00"]).tz_localize("UTC"))
    np.testing.assert_array_equal(on_sessions(series, DAYS), [np.nan, 7, 7, 7])


def test_a_line_of_one_series_is_drawn_in_the_strategys_colour(performance, sessions):
    series = pd.Series([1.0, 2.0], index=[sessions[0], sessions[2]])
    html = drawn(performance, rp.line(series, title="Steps"))
    assert "data-c0='var(--text)'" in html and "data-lines='1'" in html
    assert re.search(r"data-v0='1,1,2,2,2,", html)


def test_a_line_of_several_series_takes_the_series_colours_and_a_legend(performance, sessions):
    frame = pd.DataFrame({"fast": 1.0, "slow": 2.0}, index=sessions)
    full = page(performance, rp.line(frame, title="Two"))
    html = section_of(full, "u-blocks")
    assert "data-c0='var(--series-1)'" in html and "data-c1='var(--series-2)'" in html
    assert "background:var(--series-1)'></span>fast" in html
    assert "--series-1:var(--text);--series-2:var(--benchmark);" in full


def test_a_page_without_a_chart_of_several_series_writes_no_series_colours(performance):
    full = page(performance, rp.line(lambda p: p.session_returns, title="One"))
    assert "--series-1" not in full and full.count("<style>") == 1


@pytest.mark.parametrize("columns", [1, 6])
def test_a_line_draws_up_to_six_series(performance, sessions, columns):
    frame = pd.DataFrame({f"c{k}": float(k) for k in range(columns)}, index=sessions)
    assert f"data-lines='{columns}'" in drawn(performance, rp.line(frame, title="Many"))


def test_a_seventh_series_is_refused():
    frame = pd.DataFrame({f"c{k}": [1.0] for k in range(7)}, index=pd.DatetimeIndex(["2025-01-01"]))
    with pytest.raises(ConfigError, match="at most 6 series"):
        rp.line(frame, title="Too many")


@pytest.mark.parametrize(
    "series",
    [
        pd.Series(dtype=float),
        pd.Series([np.nan, np.nan], index=pd.DatetimeIndex(["2025-06-02", "2025-06-03"])),
        # One value on the last session is one point: a line needs two sessions.
        pd.Series([1.0], index=pd.DatetimeIndex(["2025-09-30"])),
    ],
)
def test_a_line_with_fewer_than_two_points_says_no_data(performance, series):
    assert NO_DATA in drawn(performance, rp.line(series, title="Thin"))


def test_a_line_on_an_index_that_is_not_dates_is_refused():
    with pytest.raises(ConfigError, match="index is dates or datetimes"):
        rp.line(pd.Series([1.0, 2.0]), title="Numbered")


def test_a_line_of_numbers_labels_its_axis_in_indian_notation(performance, sessions):
    series = pd.Series(np.linspace(0, 12_00_000, len(sessions)), index=sessions)
    html = drawn(performance, rp.line(series, title="Rupees"))
    assert "<span style='top:" in html and ">10,00,000</span>" in html


# ----- bars ------------------------------------------------------------------------------


def test_a_bar_chart_draws_a_bar_per_category(performance):
    html = drawn(performance, rp.bar(pd.Series({"Mon": 1.0, "Tue": 2.0, "Wed": 3.0}), title="B"))
    assert html.count("<rect ") == 3 and "style='fill:var(--text)'" in html
    assert "<title>Tue &#183; B: 2.00</title>" in html


def test_a_zero_line_is_drawn_only_when_the_values_change_sign(performance):
    signed = drawn(performance, rp.bar(pd.Series({"a": 1.0, "b": -2.0}), title="S"))
    unsigned = drawn(performance, rp.bar(pd.Series({"a": 1.0, "b": 2.0}), title="U"))
    assert "class='base'" in signed and "class='base'" not in unsigned


def test_grouped_bars_take_the_series_colours_and_a_legend(performance):
    frame = pd.DataFrame({"long": [1.0, 2.0], "short": [-1.0, 0.5]}, index=["Mon", "Tue"])
    html = drawn(performance, rp.bar(frame, title="G"))
    assert html.count("<rect ") == 4
    assert "fill:var(--series-2)" in html and "</span>short</span>" in html


def test_bars_in_rows_put_each_category_on_a_row(performance):
    html = drawn(performance, rp.bar(pd.Series({"a": 1.0, "b": 2.0}), title="H", horizontal=True))
    # Two rows of 26 pixels between a top margin of 10 and a bottom margin of 26.
    assert "viewBox='0 0 680 88'" in html


def test_a_dated_category_reads_as_its_date(performance):
    series = pd.Series([1.0], index=pd.DatetimeIndex(["2025-01-01"]))
    assert ">2025-01-01</text>" in drawn(performance, rp.bar(series, title="D"))


@pytest.mark.parametrize("series", [pd.Series(dtype=float), pd.Series({"a": np.nan, "b": np.nan})])
def test_bars_with_no_value_say_no_data(performance, series):
    assert NO_DATA in drawn(performance, rp.bar(series, title="Empty"))


# ----- histograms ------------------------------------------------------------------------


def test_missing_values_of_a_histogram_are_counted_in_its_subtitle(performance):
    values = pd.Series([0.01, 0.02, np.nan, np.nan])
    html = drawn(performance, rp.histogram(values, title="H", subtitle="Returns"))
    assert "<p class='card-sub'>Returns &#183; 2 missing values left out</p>" in html


def test_a_histogram_of_plain_numbers_has_no_percent_labels(performance):
    html = drawn(performance, rp.histogram(np.array([1.0, 2.0, 3.0, 40.0]), title="Bars held"))
    axis = re.findall(r"text-anchor='middle' style='[^']*'>([^<]*)</text>", html)
    assert axis and not any("%" in label for label in axis)


def test_a_histogram_of_fractions_shown_as_percent_labels_percent(performance):
    html = drawn(performance, rp.histogram([0.01, 0.03, -0.02], title="R", percent=True))
    assert "%</text>" in html


@pytest.mark.parametrize("values", [[], [np.nan]])
def test_a_histogram_with_no_value_says_no_data(performance, values):
    assert NO_DATA in drawn(performance, rp.histogram(np.array(values, dtype=float), title="E"))


# ----- scatter ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "days_held": [1.0, 2.0, 3.0, np.nan],
            "return_pct": [0.5, -1.0, 2.0, 1.0],
            "side": ["long", "short", "long", "long"],
        }
    )


def test_a_scatter_draws_a_point_per_row_with_both_values(performance, trades):
    html = drawn(performance, rp.scatter(trades, x="days_held", y="return_pct", title="S"))
    assert html.count("<circle ") == 3
    assert "1 point without both values left out" in html
    assert "<title>days_held 2.00, return_pct -1.00</title>" in html


def test_a_scatters_groups_take_the_series_colours_and_a_legend(performance, trades):
    block = rp.scatter(trades, x="days_held", y="return_pct", title="S", group="side")
    html = drawn(performance, block)
    assert "fill:var(--series-1)" in html and "fill:var(--series-2)" in html
    assert "</span>long</span>" in html and "<title>short: days_held 2.00" in html


def test_a_seventh_group_is_refused():
    frame = pd.DataFrame({"x": range(7), "y": range(7), "g": list("abcdefg")})
    with pytest.raises(ConfigError, match="at most 6 groups"):
        rp.scatter(frame, x="x", y="y", title="S", group="g")


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"x": "nope", "y": "return_pct"}, "x names column 'nope'"),
        ({"x": "days_held", "y": "side"}, "column 'y' is not numbers"),
        ({"x": "days_held", "y": "return_pct", "group": "nope"}, "group names column 'nope'"),
    ],
)
def test_a_scatter_of_columns_that_cannot_be_drawn_is_refused(trades, arguments, message):
    with pytest.raises(ConfigError, match=message):
        rp.scatter(trades, title="S", **arguments)


# ----- heatmaps --------------------------------------------------------------------------


def test_a_heatmaps_cells_take_the_heat_colour_by_size_against_the_largest(performance):
    frame = pd.DataFrame({"a": [0.1, -0.2], "b": [np.nan, 0.0]}, index=["x", "y"])
    html = drawn(performance, rp.heatmap(frame, title="Heat", percent=True))
    # The bound is 0.2: 0.1 is half of it, 44 percent of the gain colour (88 at the bound).
    assert "background:color-mix(in oklab, var(--heat-pos) 44%, var(--heat-mid))'>10.0%" in html
    assert "background:color-mix(in oklab, var(--heat-neg) 88%, var(--heat-mid))'>-20.0%" in html
    assert "title='x, b: n/a' style='background:transparent'></td>" in html


def test_a_heatmap_of_text_is_refused():
    with pytest.raises(ConfigError, match="column 'a' is not numbers"):
        rp.heatmap(pd.DataFrame({"a": ["x"]}), title="Heat")


# ----- tables ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def listing() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price": [1234567.5, 10.0],
            "qty": [1500000, 3],
            "when": [pd.Timestamp("2026-11-12 09:15", tz=IST), pd.NaT],
            "name": ["A&B", None],
        }
    )


def test_a_table_shows_numbers_in_indian_notation_and_times_as_the_page_does(performance, listing):
    html = drawn(performance, rp.table(listing, title="T"))
    cells = re.findall(r"<td class='[^']*' data-v='[^']*'>([^<]*)</td>", html)
    assert cells == [
        "12,34,567.50",
        "15,00,000",
        "12 Nov 2026 09:15",
        "A&amp;B",
        "10.00",
        "3",
        "n/a",
        "n/a",
    ]


def test_a_tables_cells_carry_their_sort_values(performance, listing):
    html = drawn(performance, rp.table(listing, title="T"))
    first_row = html[html.index("<tr data-i='0'>") : html.index("<tr data-i='1'>")]
    stamp = pd.Timestamp("2026-11-12 09:15", tz=IST).value
    assert re.findall(r"data-v='([^']*)'", first_row) == [
        "1234567.5",
        "1500000.0",
        str(stamp),
        "a&amp;b",
    ]


def test_a_table_is_sortable_and_paged_as_asked(performance, listing):
    paged = drawn(performance, rp.table(listing, title="T", rows=1))
    assert "data-sort data-page='1'" in paged
    assert "<button type='button' class='btn pg-next'>Next</button>" in paged
    fixed = drawn(performance, rp.table(listing, title="T", sortable=False))
    assert "data-sort" not in fixed and " s'>" not in fixed


def test_a_named_index_is_the_first_column_and_the_plain_one_is_not(performance):
    named = pd.DataFrame({"v": [1.0]}, index=pd.Index(["a"], name="key"))
    assert ">key</th>" in drawn(performance, rp.table(named, title="T"))
    plain = drawn(performance, rp.table(pd.DataFrame({"v": [1.0]}), title="T"))
    assert plain.count("<th ") == 1


def test_a_table_of_fractions_shown_as_percent(performance):
    html = drawn(performance, rp.table(pd.DataFrame({"r": [0.1234]}), title="T", percent=True))
    assert ">12.34%</td>" in html


# ----- tiles and text --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "options", "shown"),
    [
        ({"Win rate": 0.5}, {"percent": True}, ["50.00%"]),
        ({"Charges": 3.6, "Net": 123456.0}, {"rupees": True}, ["&#8377;3.60", "&#8377;1,23,456"]),
        ({"Trades": 12, "Ratio": 1.5, "Note": "ok <b>", "Gone": None}, {}, None),
    ],
)
def test_tiles_show_numbers_as_the_page_does_and_text_as_it_is(performance, values, options, shown):
    html = drawn(performance, rp.tiles(values, title="Tiles", **options))
    found = re.findall(r"<div class='stat-value nums'>([^<]*)</div>", html)
    assert found == (shown or ["12", "1.50", "ok &lt;b&gt;", "n/a"])
    assert "<div class='eyebrow grid-label'>Tiles</div>" in html


@pytest.mark.parametrize("value", [True, [1, 2]])
def test_a_tile_of_anything_but_a_number_or_text_is_refused(value):
    with pytest.raises(ConfigError, match="a tile shows a number or text"):
        rp.tiles({"x": value})


def test_text_is_escaped_into_paragraphs(performance):
    html = drawn(performance, rp.text("Para one.\n\nPara two & <b> \u20b9 5."))
    assert (
        "<div class='prose'><p>Para one.</p><p>Para two &amp; &lt;b&gt; &#8377; 5.</p></div>"
        in html
    )


def test_empty_text_says_no_data(performance):
    assert NO_DATA in drawn(performance, rp.text("  "))


# ----- widths ----------------------------------------------------------------------------


def test_consecutive_half_blocks_share_the_pages_two_column_grid(performance):
    html = drawn(
        performance,
        rp.text("a", title="A", width="half"),
        rp.text("b", title="B", width="half"),
        rp.text("c", title="C"),
        rp.text("d", title="D", width="third"),
    )
    assert re.search(r"<div class='grid2'><div class='card'>.*A</h3>.*B</h3>.*</div>", html)
    assert html.index("grid2") < html.index("C</h3>") < html.index("grid3")


# ----- checks when a block is made, and when its data arrives ----------------------------

SERIES = pd.Series([1.0], index=pd.DatetimeIndex(["2025-01-01"]))
TWICE = pd.DataFrame([[1.0, 2.0]], columns=["a", "a"])


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: rp.line(SERIES, title=""), "title is text that is not empty"),
        (lambda: rp.line(SERIES, title=None), "needs a title"),
        (lambda: rp.line(SERIES, title="t", width="wide"), "width is one of"),
        (lambda: rp.line(SERIES, title="t", percent=1), "percent is True or False"),
        (lambda: rp.line(SERIES, title="t", log_scale=True), "log_scale draws growth"),
        (lambda: rp.line("text", title="t"), "data is a pandas Series or DataFrame"),
        (lambda: rp.bar(SERIES, title="t", horizontal="yes"), "horizontal is True or False"),
        (lambda: rp.histogram(SERIES, title="t", bins=0), "bins is a whole number"),
        (lambda: rp.histogram(["a"], title="t"), "not numbers"),
        (lambda: rp.table(pd.DataFrame(), title="t", rows=0), "rows is a whole number"),
        (lambda: rp.tiles({}, rupees=True, percent=True), "rupees or as percent"),
        (lambda: rp.text(5), "data is text"),
        (lambda: rp.table(TWICE, title="t"), "column names repeat"),
        (lambda: rp.line(TWICE.set_axis(SERIES.index), title="t"), "column names repeat"),
        (lambda: rp.html(b"<b>x</b>"), "data is text"),
    ],
)
def test_a_block_that_cannot_be_drawn_is_refused_when_made(make, message):
    with pytest.raises(ConfigError, match=message):
        make()


def test_a_data_function_that_raises_is_refused_naming_the_block_and_its_section(performance):
    block = rp.line(lambda result: result.nope, title="Broken")
    with pytest.raises(ConfigError, match="line block 'Broken' in section 'Blocks'") as raised:
        page(performance, block)
    assert isinstance(raised.value.__cause__, AttributeError)


def test_data_a_function_returns_is_checked_when_the_page_is_rendered(performance):
    block = rp.line(lambda result: "not a series", title="Wrong")
    with pytest.raises(ConfigError, match="in section 'Blocks': data is a pandas Series"):
        page(performance, block)
