"""Custom tearsheets, the escape hatch: an `html()` or `svg()` fragment is checked against
the allow list of elements and attributes (design 19, section 1.4, with the lead's
settlement of 2026-11-20), written back balanced and escaped, and its ids prefixed with
its block's anchor and its references rewritten to match."""

from __future__ import annotations

import re

import pytest
from custom_pages import section_of, small_performance

from openalgo.backtest import report as rp
from openalgo.backtest.errors import ConfigError
from openalgo.backtest.report._custom import _allowed as allowed
from openalgo.backtest.report._custom._fragment import checked
from openalgo.backtest.report._dashboard import script

WHERE = "html block 'F' in section 'S'"


def html(fragment: str, prefix: str = "p") -> str:
    return checked(fragment, where=WHERE, svg_only=False, prefix=prefix)


def svg(fragment: str, prefix: str = "p") -> str:
    return checked(fragment, where=WHERE, svg_only=True, prefix=prefix)


# ----- the lists, as design 19 section 1.4 and the lead's 11.0a.6 of design 18 give them ------

#: HTML elements: the design's 33, then img and a (11.0a.6).
HTML_ELEMENTS = (
    "div", "span", "p", "br", "hr", "b", "strong", "i", "em", "small", "sub", "sup", "code",
    "pre", "ul", "ol", "li", "dl", "dt", "dd", "h3", "h4", "table", "caption", "thead", "tbody",
    "tfoot", "tr", "th", "td", "figure", "figcaption", "blockquote", "img", "a",
)  # fmt: skip
SVG_ELEMENTS = (
    "svg", "g", "defs", "title", "desc", "path", "rect", "circle", "ellipse", "line",
    "polyline", "polygon", "text", "tspan", "linearGradient", "radialGradient", "stop",
    "clipPath", "mask", "marker", "pattern", "use",
)  # fmt: skip
#: Attributes allowed on any listed element; aria-* and data-* are families, and xmlns a
#: namespace declaration (checked and left out of the page).
ATTRIBUTES = (
    "class", "title", "lang", "dir", "colspan", "rowspan", "scope", "id", "viewBox", "width",
    "height", "x", "y", "x1", "y1", "x2", "y2", "cx", "cy", "r", "rx", "ry", "d", "points",
    "transform", "fill", "fill-opacity", "fill-rule", "stroke", "stroke-width",
    "stroke-opacity", "stroke-dasharray", "stroke-linecap", "stroke-linejoin", "opacity",
    "font-size", "font-family", "font-weight", "text-anchor", "dominant-baseline", "offset",
    "stop-color", "stop-opacity", "gradientUnits", "gradientTransform", "clip-path", "mask",
    "marker-start", "marker-mid", "marker-end", "markerWidth", "markerHeight", "refX", "refY",
    "orient", "patternUnits", "preserveAspectRatio",
)  # fmt: skip
#: Attributes allowed on one element only: href on use (design) and on a (11.0a.6), and
#: src with its text alternative alt on img (11.0a.6).
ELEMENT_ATTRIBUTES = {"use": ("href", "xlink:href"), "a": ("href",), "img": ("src", "alt")}


def test_the_allow_lists_are_the_designs():
    assert (len(HTML_ELEMENTS), len(SVG_ELEMENTS)) == (35, 22)
    assert frozenset(HTML_ELEMENTS) == allowed.HTML_ELEMENTS
    assert tuple(allowed.SVG_ELEMENTS) == SVG_ELEMENTS
    assert sorted(ATTRIBUTES) == sorted(allowed.ATTRIBUTES)
    assert allowed.ATTRIBUTE_PREFIXES == ("aria-", "data-")
    assert ELEMENT_ATTRIBUTES == allowed.ELEMENT_ATTRIBUTES
    assert {"xmlns", "xmlns:xlink"} == set(allowed.NAMESPACES)


# ----- what the allow list lets through -----------------------------------------------------

#: The listed elements that need an attribute to be allowed at all.
NEEDS = {"img": " src='data:image/png;base64,iVBORw0KGgo='", "a": " href='#top'"}
#: The listed elements that have no end tag.
VOID = ("br", "hr", "img")


@pytest.mark.parametrize("element", HTML_ELEMENTS)
def test_every_listed_html_element_is_allowed(element):
    attributes = NEEDS.get(element, "")
    if element in VOID:
        assert html(f"<{element}{attributes}>") == f"<{element}{attributes}>"
    else:
        expected = f"<{element}{attributes}>x</{element}>"
        assert html(expected) == expected


@pytest.mark.parametrize("element", SVG_ELEMENTS)
def test_every_listed_svg_element_is_allowed_inside_an_svg(element):
    fragment = f"<svg><{element}></{element}></svg>"
    expected = fragment if element != "svg" else "<svg><svg></svg></svg>"
    assert svg(fragment.lower()) == expected


@pytest.mark.parametrize("attribute", ATTRIBUTES)
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


def test_a_font_family_may_quote_its_names():
    written = svg("<svg><text font-family=\"'Chart Sans', sans-serif\">x</text></svg>")
    assert written == "<svg><text font-family='&#x27;Chart Sans&#x27;, sans-serif'>x</text></svg>"


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
        # A CSS escape in a value the browser reads as CSS could spell url( unseen.
        (r"<svg><rect mask='u\72 l(https://example.org/m.svg#m)'/></svg>", "a backslash"),
        (r"<svg><rect fill='\75rl(https://example.org/p)'/></svg>", "a backslash"),
        # A quoted string names a resource in a CSS image function without url(.
        ("<svg><rect mask=\"image-set('https://example.org/m.png' 1x)\"/></svg>", "a quote"),
        # The hooks the page's script reads, which would run its code on the fragment.
        ("<figure class='ts'></figure>", "class 'ts'"),
        ("<div class='card ts-x'></div>", "class 'ts-x'"),
        ("<figure data-lines='1'></figure>", "'data-lines'"),
        ("<span data-v0='1'></span>", "'data-v0'"),
        ("<span data-c12='red'></span>", "'data-c12'"),
        ("<div data-weekly></div>", "'data-weekly'"),
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


#: What the page's script reads: an attribute it gets (a family when it adds a number to
#: the name) or selects by, and a class it selects by or tests.
READ_ATTRIBUTE = re.compile(r"getAttribute\('(data-[a-z-]+)'(\+k)?|\[(data-[a-z-]+)\]")
SELECTED_CLASS = re.compile(
    r"querySelector(?:All)?\('[a-z]*\.([a-z-]+)|classList\.contains\('([a-z-]+)'"
)


def script_hooks() -> tuple[set[str], set[str]]:
    """The data attributes and classes the page's script reads, from its own text."""
    text = script.HEAD + script.script(1)
    attributes = {
        (got + "0" if family else got) or selected
        for got, family, selected in READ_ATTRIBUTE.findall(text)
    }
    classes = {selected or tested for selected, tested in SELECTED_CLASS.findall(text)}
    return attributes, classes


def test_every_hook_the_pages_script_reads_is_refused_in_a_fragment():
    attributes, classes = script_hooks()
    assert {"data-sort", "data-lines", "data-v0", "data-c0", "data-weekly"} <= attributes
    assert {"ts", "ts-x", "nav", "s"} <= classes
    for attribute in sorted(attributes):
        with pytest.raises(ConfigError, match=attribute):
            html(f"<span {attribute}='1'></span>")
    for name in sorted(classes):
        with pytest.raises(ConfigError, match=f"class '{name}'"):
            html(f"<span class='box {name}'></span>")


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
