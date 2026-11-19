"""Custom tearsheets, the theme and the page as a whole: a theme's colours are checked by
a pattern and written in a second stylesheet after the page's own, in the three places
the page writes its tokens; a page that asks for nothing custom has no second
stylesheet; and a custom page with every kind of block stays one offline file whose
only scripts are the page's own (design 19, sections 1.3, 2 and 4)."""

from __future__ import annotations

import re

import pandas as pd
import pytest
from custom_pages import small_performance
from test_report_tearsheet import Structure

from openalgo.backtest import report as rp
from openalgo.backtest.errors import ConfigError
from openalgo.backtest.report._dashboard import style

DARK_MEDIA = "@media (prefers-color-scheme: dark){:root:not([data-theme=light]){"
DARK_SET = ":root[data-theme=dark]{"


@pytest.fixture(scope="module")
def performance():
    return small_performance()


def extra_style(page: str) -> str:
    """The page's second stylesheet, or "" without one."""
    sheets = re.findall(r"<style>(.*?)</style>", page, flags=re.S)
    return sheets[1] if len(sheets) > 1 else ""


def several_series(performance) -> rp.custom.CustomSection:
    frame = pd.DataFrame({"a": 1.0, "b": 2.0}, index=performance.session_returns.index)
    return rp.section("Two", rp.line(frame, title="Two"))


# ----- colours -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "colour",
    [
        "#0b7a3e",
        "#FFF",
        "rgb(11, 122, 62)",
        "rgba(0,0,0,.5)",
        "hsl(142 72% 33%)",
        "hsl(142deg 72% 33% / .5)",
        "Teal",
        "rebeccapurple",
        "transparent",
        "currentColor",
    ],
)
def test_a_colour_in_a_listed_form_is_accepted(colour):
    assert rp.Theme(light={"accent": colour}).light == {"accent": colour}


@pytest.mark.parametrize(
    "colour",
    [
        "red;}",
        "red;}body{display:none",
        "url(x.png)",
        "#12345",
        "#gggggg",
        "gren",
        "rgb(1,2,3);",
        "rgb(1,2,3) red",
        "expression(alert(1))",
        "var(--text)",
        "",
        5,
    ],
)
def test_a_colour_in_any_other_form_is_refused(colour):
    with pytest.raises(ConfigError, match="is not a colour"):
        rp.Theme(dark={"bg": colour})


def test_an_unknown_token_is_refused_naming_the_tokens():
    with pytest.raises(ConfigError, match="'pnl' is not a token of the page; the tokens are bg,"):
        rp.Theme(light={"pnl": "red"})


@pytest.mark.parametrize(
    ("series", "message"),
    [
        (["red"] * 7, "at most 6 colours"),
        ("red", "a list of colours"),
        ([("red", "blue", "green")], "a \\(light, dark\\) pair"),
        ([("red", "bleu")], "is not a colour"),
    ],
)
def test_series_colours_that_cannot_be_written_are_refused(series, message):
    with pytest.raises(ConfigError, match=message):
        rp.Theme(series=series)


# ----- the second stylesheet -----------------------------------------------------------------


def test_the_theme_comes_after_the_pages_stylesheet_in_its_three_places(performance):
    theme = rp.Theme(light={"pnl-pos": "#0b7a3e"}, dark={"pnl-pos": "#4ade80"})
    page = rp.tearsheet(performance, layout=rp.Layout(theme=theme))
    expected = (
        ":root{--pnl-pos:#0b7a3e;}"
        f"{DARK_MEDIA}--pnl-pos:#4ade80;}}}}"
        f"{DARK_SET}--pnl-pos:#4ade80;}}"
    )
    assert f"</style><style>{expected}</style></head>" in page


def test_the_theme_uses_the_selectors_the_pages_own_stylesheet_uses():
    sheet = style.stylesheet()
    assert sheet.startswith(":root{") and DARK_MEDIA in sheet and DARK_SET in sheet


def test_a_light_theme_alone_writes_the_root_only(performance):
    page = rp.tearsheet(performance, layout=rp.Layout(theme=rp.Theme(light={"bg": "white"})))
    assert extra_style(page) == ":root{--bg:white;}"


def test_tokens_are_written_in_the_pages_order(performance):
    theme = rp.Theme(light={"benchmark": "#111111", "bg": "#222222"})
    page = rp.tearsheet(performance, layout=rp.Layout(theme=theme))
    assert extra_style(page) == ":root{--bg:#222222;--benchmark:#111111;}"


def test_a_page_that_asks_for_nothing_custom_has_no_second_stylesheet(performance):
    sections = (rp.section("Words", rp.text("x")),)
    page = rp.tearsheet(performance, layout=rp.Layout(hide=("monthly",), sections=sections))
    assert page.count("<style>") == 1


def test_series_colours_are_written_for_a_chart_of_several_series(performance):
    page = rp.tearsheet(performance, layout=rp.Layout(sections=(several_series(performance),)))
    light = "--series-1:var(--text);--series-2:var(--benchmark);--series-3:hsl(32 95% 44%);"
    dark = "--series-1:var(--text);--series-2:var(--benchmark);--series-3:hsl(38 92% 60%);"
    sheet = extra_style(page)
    assert sheet.startswith(f":root{{{light}") and f"{DARK_SET}{dark}" in sheet


def test_a_themes_series_colours_come_first_and_the_pages_follow(performance):
    theme = rp.Theme(series=("#111111", ("#222222", "#333333")))
    layout = rp.Layout(theme=theme, sections=(several_series(performance),))
    sheet = extra_style(rp.tearsheet(performance, layout=layout))
    assert ":root{--series-1:#111111;--series-2:#222222;--series-3:hsl(32 95% 44%);" in sheet
    assert f"{DARK_SET}--series-1:#111111;--series-2:#333333;--series-3:hsl(38 92% 60%);" in sheet


def test_a_theme_without_a_chart_of_several_series_writes_no_series_colours(performance):
    theme = rp.Theme(light={"bg": "white"}, series=("red",))
    assert "--series" not in extra_style(rp.tearsheet(performance, layout=rp.Layout(theme=theme)))


# ----- the page as a whole -----------------------------------------------------------------


@pytest.fixture(scope="module")
def every_kind(performance) -> str:
    dates = performance.session_returns.index
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0.5, -1.0, 2.0], "g": ["a", "b", "a"]})
    blocks = (
        rp.line(pd.DataFrame({"a": 1.0, "b": 2.0}, index=dates), title="Line"),
        rp.bar(pd.Series({"Mon": 1.0, "Tue": -2.0}), title="Bar", width="half"),
        rp.histogram(performance.session_returns, title="Histogram", width="half", percent=True),
        rp.scatter(frame, x="x", y="y", group="g", title="Scatter", width="third"),
        rp.heatmap(frame[["x", "y"]], title="Heatmap", width="third"),
        rp.table(frame, title="Table", rows=2, width="third"),
        rp.tiles({"Trades": 3, "Note": "ok"}, title="Tiles"),
        rp.text("Plain words."),
        rp.html("<p>A chart: <img src='data:image/png;base64,iVBORw0KGgo=' alt='c'></p>"),
        rp.svg(
            "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 2 2'><defs>"
            "<linearGradient id='g'><stop offset='0' stop-color='red'/></linearGradient></defs>"
            "<circle cx='1' cy='1' r='1' fill='url(#g)'/></svg>"
        ),
    )
    layout = rp.Layout(
        theme=rp.Theme(light={"accent": "#123456"}), sections=(rp.section("All", *blocks),)
    )
    return rp.tearsheet(performance, layout=layout)


def test_a_custom_page_refers_to_nothing_outside_itself(every_kind):
    assert "http:" not in every_kind and "https:" not in every_kind
    assert all(link.startswith("#") for link in re.findall(r"href='([^']*)'", every_kind))
    assert all(src.startswith("data:image/") for src in re.findall(r"src='([^']*)'", every_kind))
    assert all(ref.startswith("url(#") for ref in re.findall(r"url\([^)]*\)", every_kind))


def test_a_custom_pages_only_scripts_are_the_pages_own(performance, every_kind):
    scripts = re.findall(r"<script>(.*?)</script>", every_kind, flags=re.S)
    assert scripts == re.findall(r"<script>(.*?)</script>", rp.tearsheet(performance), flags=re.S)
    assert len(scripts) == 2


def test_a_custom_page_is_whole_and_ascii(every_kind):
    structure = Structure(every_kind)
    assert structure.errors == [] and structure.stack == []
    assert every_kind.isascii()
