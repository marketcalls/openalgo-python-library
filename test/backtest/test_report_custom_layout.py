"""Custom tearsheets, the layout: which built-in parts and sections a page shows, where
custom sections go, the header's links and numbers, printing, and the checks a layout
passes when it is made (design 19, sections 1.1, 1.2 and 2)."""

from __future__ import annotations

import pickle

import pandas as pd
import pytest
import tearsheet_runs
from custom_pages import (
    element_ids,
    header_links,
    section_anchors,
    section_numbers,
    section_of,
    small_performance,
)

import openalgo.backtest as bt
from openalgo.backtest import report as rp
from openalgo.backtest.errors import ConfigError

FULL = [
    "overview",
    "performance",
    "risk",
    "monthly",
    "distribution",
    "metrics",
    "trades",
    "overfitting",
    "notes",
]


@pytest.fixture(scope="module")
def performance():
    return small_performance()


def note(words: str) -> rp.custom.CustomSection:
    """A custom section of one text block, named `words`."""
    return rp.section(words, rp.text(f"{words} body"))


def page_of(performance, **layout) -> str:
    return rp.tearsheet(performance, layout=rp.Layout(**layout))


# ----- parts and sections ----------------------------------------------------------------


def test_hiding_a_section_leaves_it_out(performance):
    assert section_anchors(page_of(performance, hide=("monthly",))) == [
        name for name in FULL if name != "monthly"
    ]


def test_hiding_a_part_leaves_its_section_without_it(performance):
    page = page_of(performance, hide=("weekly_heatmap",))
    assert "monthly" in section_anchors(page)
    weekly = "<div class='wk' data-weekly>"
    assert weekly not in page and weekly in rp.tearsheet(performance)


def test_hiding_every_part_of_a_section_leaves_the_section_out(performance):
    page = page_of(performance, hide=("distribution_stats", "histogram", "box_plot"))
    assert "distribution" not in section_anchors(page)


def test_showing_a_part_the_base_leaves_out_adds_it(performance):
    page = page_of(performance, show=("definitions",))
    assert "<td class='l def'>" in page
    assert "<td class='l def'>" not in rp.tearsheet(performance)


def test_a_part_shown_on_another_base_brings_its_section_in_page_order(performance):
    page = page_of(performance, base="metrics", show=("histogram",))
    assert section_anchors(page) == ["distribution", "metrics", "notes"]


def test_a_section_with_nothing_to_show_is_left_out_as_on_the_default_page(performance):
    # A single strategy has no Strategies section; ordering it is not an error.
    page = page_of(performance, order=("strategies",))
    assert section_anchors(page) == FULL


# ----- where custom sections go ------------------------------------------------------------


def test_a_custom_section_goes_before_notes_by_default(performance):
    page = page_of(performance, sections=(note("Exposure"),))
    assert section_anchors(page) == [*FULL[:-1], "u-exposure", "notes"]


def test_a_custom_section_goes_after_the_section_it_names(performance):
    page = page_of(performance, sections=(rp.section("Exposure", rp.text("x"), after="risk"),))
    assert section_anchors(page)[2:5] == ["risk", "u-exposure", "monthly"]


def test_sections_after_one_section_keep_the_layouts_order(performance):
    sections = (
        rp.section("First", rp.text("x"), after="risk"),
        rp.section("Second", rp.text("x"), after="risk"),
    )
    page = page_of(performance, sections=sections)
    assert section_anchors(page)[2:6] == ["risk", "u-first", "u-second", "monthly"]


def test_a_custom_section_may_follow_a_custom_section_listed_before_it(performance):
    sections = (
        rp.section("First", rp.text("x"), after="risk"),
        rp.section("Second", rp.text("x"), after="First"),
    )
    page = page_of(performance, sections=sections)
    assert section_anchors(page)[2:6] == ["risk", "u-first", "u-second", "monthly"]


def test_order_puts_the_named_sections_first_and_keeps_the_rest_in_place(performance):
    page = page_of(performance, sections=(note("Exposure"),), order=("Notes", "Exposure", "risk"))
    expected = ["notes", "u-exposure", "risk"]
    expected += [name for name in FULL if name not in ("notes", "risk")]
    assert section_anchors(page) == expected


def test_without_notes_a_custom_section_goes_last(performance):
    page = page_of(performance, hide=("notes",), sections=(note("Exposure"),))
    assert section_anchors(page)[-1] == "u-exposure"


# ----- additions to a built-in section --------------------------------------------------------


def test_a_section_named_as_a_built_in_one_adds_its_blocks_at_its_end(performance):
    page = page_of(performance, sections=(rp.section("Trades", rp.text("Added here")),))
    assert section_anchors(page) == FULL
    trades = section_of(page, "trades")
    assert trades.index("Trade List") < trades.index("Added here")
    assert "<div class='gap-l'><div class='card'>" in trades[trades.index("Trade List") :]


def test_an_addition_to_a_section_absent_on_the_page_forms_that_section(performance):
    page = page_of(performance, sections=(rp.section("Strategies", rp.text("Alone")),))
    assert section_anchors(page) == [*FULL[:-1], "strategies", "notes"]
    assert "<h2 class='serif'>Strategies</h2>" in section_of(page, "strategies")


def test_a_section_formed_by_an_addition_takes_the_built_in_title_and_print_choice(performance):
    sections = (rp.section("strategies", rp.text("screen only"), printed=False),)
    page = page_of(performance, sections=sections)
    strategies = section_of(page, "strategies")
    assert strategies.startswith("<section id='strategies' class='no-print'>")
    assert "<h2 class='serif'>Strategies</h2>" in strategies
    assert "<a href='#strategies'>Strategies</a>" in page
    assert "<div class='no-print'>" not in strategies


# ----- navigation --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def busy_page(performance) -> str:
    sections = (
        rp.section("Exposure", rp.line(pd.Series(dtype=float), title="Empty"), after="overview"),
        rp.section("Overview 2", rp.text("x")),
        rp.section("risk", rp.text("added to risk")),
        rp.section("u", rp.text("x"), after="Exposure"),
    )
    return page_of(performance, hide=("monthly",), sections=sections)


def test_every_header_link_resolves_to_one_section(busy_page):
    anchors = section_anchors(busy_page)
    assert header_links(busy_page) == anchors
    assert all(element_ids(busy_page).count(anchor) == 1 for anchor in anchors)


def test_element_ids_are_unique(busy_page):
    ids = element_ids(busy_page)
    assert len(ids) == len(set(ids))


def test_sections_are_numbered_consecutively_in_page_order(busy_page):
    assert section_numbers(busy_page) == list(range(1, len(section_anchors(busy_page)) + 1))


def test_a_custom_anchor_never_takes_a_built_in_anchor(busy_page):
    custom = [a for a in section_anchors(busy_page) if a not in rp.PARTS]
    assert custom == ["u-exposure", "u-u", "u-overview-2"]


def test_a_fragment_id_never_takes_another_custom_sections_anchor(performance):
    # Section "S" (anchor u-s) holds a fragment whose element has the id "x"; section
    # "s 1 x" has the anchor u-s-1-x. Prefixed with its block's anchor, the fragment's id
    # must not be that anchor too, or the header link and the scroll spy find it first.
    sections = (rp.section("S", rp.html("<div id='x'>a</div>")), rp.section("s 1 x", rp.text("b")))
    ids = element_ids(page_of(performance, sections=sections))
    assert ids.count("u-s-1-x") == 1
    assert len(ids) == len(set(ids))


def test_a_charts_own_ids_never_take_another_custom_sections_anchor(performance):
    # An area line draws a gradient whose id comes from its block's anchor (block 1 of
    # section "S", series 0); section "s 1 0" has the anchor u-s-1-0.
    area = rp.line(performance.session_returns, title="Area", area=True)
    sections = (rp.section("S", area), rp.section("s 1 0", rp.text("b")))
    ids = element_ids(page_of(performance, sections=sections))
    assert ids.count("u-s-1-0") == 1
    assert len(ids) == len(set(ids))


# ----- printing and repeatability ----------------------------------------------------------


def test_a_section_left_out_of_print_carries_the_pages_no_print_class(performance):
    page = page_of(performance, sections=(rp.section("Exposure", rp.text("x"), printed=False),))
    assert "<section id='u-exposure' class='no-print'>" in page


def test_blocks_added_to_a_built_in_section_can_be_left_out_of_print(performance):
    sections = (rp.section("risk", rp.text("screen only"), printed=False),)
    risk = section_of(page_of(performance, sections=sections), "risk")
    assert "<div class='no-print'><div class='card'>" in risk


def test_custom_sections_carry_the_pages_card_and_section_classes(performance):
    page = page_of(performance, sections=(rp.section("Exposure", rp.text("x", title="T")),))
    body = section_of(page, "u-exposure")
    assert "<div class='sec-head'>" in body and "<div class='card'>" in body


def test_the_same_layout_rendered_twice_gives_the_same_bytes(performance):
    layout = rp.Layout(sections=(note("Exposure"),), hide=("monthly",))
    first = rp.tearsheet(performance, layout=layout)
    assert rp.tearsheet(performance, layout=layout) == first


def _returns(performance) -> pd.Series:
    return performance.session_returns


def test_a_layout_is_plain_data_that_pickles_with_its_functions(performance):
    layout = rp.Layout(sections=(rp.section("Returns", rp.line(_returns, title="R")),))
    again = pickle.loads(pickle.dumps(layout))
    assert again.sections[0].blocks[0].data is _returns
    assert rp.tearsheet(performance, layout=again) == rp.tearsheet(performance, layout=layout)


def test_parts_lists_the_built_in_sections_in_page_order_read_only():
    assert list(rp.PARTS) == [*FULL[:-1], "strategies", "portfolio", "notes"]
    with pytest.raises(TypeError):
        rp.PARTS["mine"] = ("x",)


def test_the_portfolio_section_holds_the_four_parts_only_a_portfolio_shows():
    assert rp.PARTS["portfolio"] == ("contribution", "account_margin", "refused", "net_positions")


def test_a_layout_holding_a_theme_hashes_by_the_theme_identity():
    # A theme is equal only to itself, as a block is: its mappings can change after it is
    # made, so they cannot be hashed.
    theme = rp.Theme(light={"bg": "red"})
    layout = rp.Layout(theme=theme)
    assert hash(layout) == hash(rp.Layout(theme=theme))
    assert theme != rp.Theme(light={"bg": "red"})


# ----- the result the blocks receive ---------------------------------------------------------


def test_a_results_blocks_receive_the_result_once():
    received = []

    def returns(result):
        received.append(result)
        return result.equity["equity_rupees"]

    result = tearsheet_runs.equity()
    layout = rp.Layout(sections=(rp.section("Equity", rp.line(returns, title="Equity")),))
    result.tearsheet(layout=layout)
    assert len(received) == 1 and received[0] is result and isinstance(result, bt.Result)


def test_report_tearsheets_blocks_receive_the_object_handed_to_it(performance):
    received = []

    def returns(given):
        received.append(given)
        return given.session_returns

    rp.tearsheet(
        performance, layout=rp.Layout(sections=(rp.section("R", rp.line(returns, title="R")),))
    )
    assert received == [performance]


# ----- refusals ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"base": "weekly"}, "base is one of"),
        ({"base": ["full"]}, "base is one of"),
        ({"hide": ("nope",)}, "neither a built-in section nor a part"),
        ({"show": ("distribution",)}, "show takes part names"),
        ({"show": ("nope",)}, "is not a part"),
        ({"hide": ("monthly",), "show": ("weekly_heatmap",)}, "both hides and shows"),
        ({"hide": ("histogram",), "show": ("histogram",)}, "both hides and shows"),
        ({"hide": 3}, "hide is a name or a tuple of names"),
        ({"sections": (note("Exposure"), note("exposure"))}, "two sections named"),
        ({"sections": (note("P&L"), note("P L"))}, "would share the link"),
        ({"sections": (note("!!!"),)}, "needs a letter or a digit"),
        ({"sections": (rp.section("X", rp.text("x"), after="nowhere"),)}, "after names"),
        (
            {
                "sections": (
                    rp.section("First", rp.text("x"), after="Second"),
                    rp.section("Second", rp.text("x")),
                )
            },
            "after names",
        ),
        (
            {"hide": ("monthly",), "sections": (rp.section("X", rp.text("x"), after="monthly"),)},
            "after names",
        ),
        ({"hide": ("monthly",), "sections": (note("Monthly"),)}, "which the layout hides"),
        ({"sections": (rp.section("Trades", rp.text("x"), after="risk"),)}, "no after"),
        ({"sections": (rp.section("Trades", rp.text("x"), subtitle="s"),)}, "no subtitle"),
        ({"order": ("nope",)}, "not a section the layout shows"),
        ({"order": ("risk", "Risk")}, "twice"),
        ({"hide": ("monthly",), "order": ("monthly",)}, "not a section the layout shows"),
        ({"sections": ("Exposure",)}, "sections is a tuple of sections"),
        ({"theme": "dark"}, "theme is a Theme"),
    ],
)
def test_a_layout_that_cannot_be_drawn_is_refused_when_made(arguments, message):
    with pytest.raises(ConfigError, match=message):
        rp.Layout(**arguments)


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: rp.section("Empty"), "needs at least one block"),
        (lambda: rp.section("X", "text"), "is not a block"),
        (lambda: rp.section("  ", rp.text("x")), "name is text"),
        (lambda: rp.section("X", rp.text("x"), printed="no"), "printed is True or False"),
        (lambda: rp.section("X", rp.text("x"), after=3), "after names a section"),
    ],
)
def test_a_section_that_cannot_be_drawn_is_refused_when_made(make, message):
    with pytest.raises(ConfigError, match=message):
        make()


def test_a_layout_with_a_mode_is_refused_naming_its_base(performance):
    with pytest.raises(ConfigError, match=r"Layout\(base='basic'\)"):
        rp.tearsheet(performance, mode="basic", layout=rp.Layout())


def test_a_layout_that_is_not_a_layout_is_refused(performance):
    with pytest.raises(ConfigError, match="layout is a report.custom.Layout"):
        rp.tearsheet(performance, layout={"hide": ("monthly",)})


def test_a_single_name_is_read_as_one_name():
    assert rp.Layout(hide="monthly").hide == ("monthly",)
