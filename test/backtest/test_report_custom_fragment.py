"""Custom tearsheets, the escape hatch: an `html()` or `svg()` fragment is checked against
the allow list of elements and attributes (design 19, section 1.4, with the lead's
settlement of 2026-11-20), written back balanced and escaped, and its ids prefixed with
its block's anchor and its references rewritten to match."""

from __future__ import annotations

import pytest
from custom_pages import section_of, small_performance

from openalgo.backtest import report as rp
from openalgo.backtest.errors import ConfigError
from openalgo.backtest.report._custom import _allowed as allowed
from openalgo.backtest.report._custom._fragment import checked

WHERE = "html block 'F' in section 'S'"


def html(fragment: str, prefix: str = "p") -> str:
    return checked(fragment, where=WHERE, svg_only=False, prefix=prefix)


def svg(fragment: str, prefix: str = "p") -> str:
    return checked(fragment, where=WHERE, svg_only=True, prefix=prefix)


# ----- what the allow list lets through -----------------------------------------------------

#: The listed elements that need an attribute to be allowed at all.
NEEDS = {"img": " src='data:image/png;base64,iVBORw0KGgo='", "a": " href='#top'"}


@pytest.mark.parametrize("element", sorted(allowed.HTML_ELEMENTS))
def test_every_listed_html_element_is_allowed(element):
    attributes = NEEDS.get(element, "")
    if element in allowed.VOID_ELEMENTS:
        assert html(f"<{element}{attributes}>") == f"<{element}{attributes}>"
    else:
        expected = f"<{element}{attributes}>x</{element}>"
        assert html(expected) == expected


@pytest.mark.parametrize("element", allowed.SVG_ELEMENTS)
def test_every_listed_svg_element_is_allowed_inside_an_svg(element):
    fragment = f"<svg><{element}></{element}></svg>"
    expected = fragment if element != "svg" else "<svg><svg></svg></svg>"
    assert svg(fragment.lower()) == expected


@pytest.mark.parametrize("attribute", allowed.ATTRIBUTES)
def test_every_listed_attribute_is_allowed(attribute):
    written = svg(f"<svg><g {attribute.lower()}='v1'></g></svg>")
    value = "p-v1" if attribute == "id" else "v1"
    assert written == f"<svg><g {attribute}='{value}'></g></svg>"


@pytest.mark.parametrize("attribute", ["aria-label", "data-row", "data-anything-at-all"])
def test_aria_and_data_attributes_are_allowed(attribute):
    assert html(f"<span {attribute}='x'>y</span>") == f"<span {attribute}='x'>y</span>"


@pytest.mark.parametrize(
    "src",
    [
        "data:image/png;base64,iVBORw0KGgo=",
        "data:image/jpeg;base64,/9j/4AAQ",
        "data:image/svg+xml;charset=utf-8,%3Csvg%3E%3C/svg%3E",
    ],
)
def test_an_image_is_allowed_inline_as_png_jpeg_or_svg(src):
    assert html(f"<img src='{src}' alt='chart'>") == f"<img src='{src}' alt='chart'>"


def test_a_link_inside_the_page_is_kept():
    assert html("<a href='#trades'>Trades</a>") == "<a href='#trades'>Trades</a>"


# ----- what it refuses, naming it -----------------------------------------------------------


@pytest.mark.parametrize(
    ("fragment", "named"),
    [
        ("<script>alert(1)</script>", "<script>"),
        ("<style>b{color:red}</style>", "<style>"),
        ("<iframe></iframe>", "<iframe>"),
        ("<object></object>", "<object>"),
        ("<link rel='x'>", "<link>"),
        ("<form></form>", "<form>"),
        ("<marquee></marquee>", "<marquee>"),
        ("<img src='https://example.org/a.png'>", "src on <img>"),
        ("<img src='data:image/gif;base64,R0lGOD'>", "src on <img>"),
        ("<img src='x.png'>", "src on <img>"),
        ("<a href='https://example.org'>x</a>", "href='https://example.org' on <a>"),
        ("<a href='javascript:alert(1)'>x</a>", "on <a>"),
        ("<a target='_blank' href='#x'>x</a>", "'target'"),
        ("<div onclick='go()'></div>", "'onclick'"),
        ("<svg onload='go()'></svg>", "'onload'"),
        ("<div style='color:red'></div>", "'style'"),
        ("<div foo='1'></div>", "'foo'"),
        ("<span src='data:image/png;base64,iVBO'></span>", "'src'"),
        ("<table data-sort><tbody></tbody></table>", "'data-sort'"),
        ("<table data-page='5'><tbody></tbody></table>", "'data-page'"),
        ("<svg><foreignObject></foreignObject></svg>", "<foreignobject>"),
        ("<svg><image href='a.png'></image></svg>", "<image>"),
        ("<svg><div></div></svg>", "<div> is not on the allow list of elements inside an <svg>"),
        ("<svg><use href='https://example.org/s.svg#a'/></svg>", "on <use>"),
        ("<svg><use href='#missing'/></svg>", "#missing"),
        ("<svg><rect fill='url(https://example.org/p)'/></svg>", "url(#name)"),
        ("<svg><rect fill='url(#nope)'/></svg>", "#nope"),
        ("<p id='a' title='url(#a)'></p>", "url(#name)"),
        ("<div><span></div>", "</div> does not close <span>"),
        ("<div>", "<div> is not closed"),
        ("</div>", "</div> does not close nothing"),
        ("<br></br>", "</br> does not close nothing"),
        ("<div id='a'></div><p id='a'></p>", "id 'a' is given twice"),
        ("<div id='1a'></div>", "an id is a letter"),
        ("<svg xmlns='http://example.org/ns'></svg>", "unknown namespace"),
        ("<![CDATA[x]]>", "is not allowed"),
    ],
)
def test_anything_off_the_list_is_refused_naming_it_and_the_block(fragment, named):
    with pytest.raises(ConfigError) as refused:
        html(fragment)
    assert named in str(refused.value) and str(refused.value).startswith(WHERE)


@pytest.mark.parametrize(
    ("fragment", "message"),
    [
        ("<div></div>", "found <div>"),
        ("<b>x</b>", "found <b>"),
        ("chart <svg></svg>", "found text outside them"),
    ],
)
def test_an_svg_block_holds_svg_elements_only(fragment, message):
    with pytest.raises(ConfigError, match=message):
        svg(fragment)


# ----- ids and references ---------------------------------------------------------------------


def test_ids_are_prefixed_with_the_blocks_anchor_and_references_rewritten():
    fragment = (
        "<svg><defs><linearGradient id='g'><stop offset='0' stop-color='red'/></linearGradient>"
        "</defs><rect fill='url(#g)' stroke='url( #g )'/><use href='#g'/>"
        "<use xlink:href='#g'/></svg>"
    )
    assert svg(fragment, prefix="u-s-1") == (
        "<svg><defs><linearGradient id='u-s-1-g'><stop offset='0' stop-color='red'/>"
        "</linearGradient></defs><rect fill='url(#u-s-1-g)' stroke='url(#u-s-1-g)'/>"
        "<use href='#u-s-1-g'/><use xlink:href='#u-s-1-g'/></svg>"
    )


def test_ids_named_by_aria_attributes_are_rewritten_and_others_kept():
    written = svg("<svg aria-labelledby='t other'><title id='t'>T</title></svg>", prefix="u-s-1")
    assert written == "<svg aria-labelledby='u-s-1-t other'><title id='u-s-1-t'>T</title></svg>"


def test_a_link_to_an_element_of_the_fragment_follows_its_prefix():
    written = html("<p id='n'>note</p><a href='#n'>to the note</a>", prefix="u-s-1")
    assert written == "<p id='u-s-1-n'>note</p><a href='#u-s-1-n'>to the note</a>"


def test_a_plain_svg_chart_is_inserted_with_only_its_ids_prefixed():
    chart = (
        "<svg viewBox='0 0 100 50' width='100' height='50'><g transform='translate(5,5)'>"
        "<rect x='0' y='0' width='40' height='30' fill='#0b7a3e'/>"
        "<line x1='0' y1='30' x2='90' y2='30' stroke='currentColor' stroke-width='1'/>"
        "<text x='45' y='45' text-anchor='middle' font-size='10'>Q1 &amp; Q2</text></g></svg>"
    )
    assert svg(chart) == chart


# ----- how it is written back -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("fragment", "written"),
    [
        # Names in their listed spelling, whatever case they were written in.
        (
            "<SVG VIEWBOX='0 0 1 1'><LINEARGRADIENT id='g'></LINEARGRADIENT></SVG>",
            "<svg viewBox='0 0 1 1'><linearGradient id='p-g'></linearGradient></svg>",
        ),
        # A standalone document's declarations and namespaces are implied by the page.
        (
            "<?xml version='1.0' encoding='UTF-8'?><!DOCTYPE svg PUBLIC '-//W3C//DTD SVG 1.1//EN'"
            " 'svg11.dtd'><!-- by hand --><svg xmlns='http://www.w3.org/2000/svg' "
            "xmlns:xlink='http://www.w3.org/1999/xlink'><g/></svg>",
            "<svg><g/></svg>",
        ),
        # An empty HTML element is written whole; a void one has no end tag.
        ("<div/><br/><hr>", "<div></div><br><hr>"),
        # Text and values escaped, every character outside ASCII as a reference.
        ("<p title='a \"b\" &lt;c&gt;'>\u20b9 5 &amp; more</p>", None),
    ],
)
def test_a_fragment_is_written_back_from_what_was_checked(fragment, written):
    expected = written or "<p title='a &quot;b&quot; &lt;c&gt;'>&#8377; 5 &amp; more</p>"
    out = html(fragment) if not fragment.startswith(("<SVG", "<?xml")) else svg(fragment)
    assert out == expected


# ----- in a page ------------------------------------------------------------------------------


def test_html_and_svg_blocks_sit_in_cards_with_their_ids_prefixed():
    layout = rp.Layout(
        sections=(
            rp.section(
                "Hand notes",
                rp.html("<p id='n'>Seen <b>here</b></p>", title="Hand"),
                rp.svg("<svg viewBox='0 0 2 2'><circle id='c' cx='1' cy='1' r='1'/></svg>"),
            ),
        )
    )
    hand = section_of(rp.tearsheet(small_performance(), layout=layout), "u-hand-notes")
    assert "<h3 class='serif'>Hand</h3>" in hand
    assert "<div class='card-body'><p id='u-hand-notes-1-n'>Seen <b>here</b></p></div>" in hand
    assert "<circle id='u-hand-notes-2-c' cx='1' cy='1' r='1'/>" in hand


def test_a_fragment_given_as_a_value_is_checked_when_the_block_is_made():
    with pytest.raises(ConfigError, match="untitled html block: the element <script>"):
        rp.html("<script>x</script>")
