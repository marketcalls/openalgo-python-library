"""The allow list of the escape hatch (`report.custom.html` and `report.custom.svg`): the
elements and attributes a fragment may use. Anything not listed is refused, so a page
stays one offline file whose only script is its own (design 19, section 1.4, and the
lead's settlement of 2026-11-20: no `style` attribute, `img` only with a `data:` image,
`a` only with an in-page link).

Every entry is a plain name, written as SVG spells it (`viewBox`, `linearGradient`); a
fragment's names are matched without regard to case, as a browser's HTML parser reads
them, and written back in the listed spelling.
"""

#: HTML elements, outside an `<svg>`.
HTML_ELEMENTS = frozenset(
    {
        "div",
        "span",
        "p",
        "br",
        "hr",
        "b",
        "strong",
        "i",
        "em",
        "small",
        "sub",
        "sup",
        "code",
        "pre",
        "ul",
        "ol",
        "li",
        "dl",
        "dt",
        "dd",
        "h3",
        "h4",
        "table",
        "caption",
        "thead",
        "tbody",
        "tfoot",
        "tr",
        "th",
        "td",
        "figure",
        "figcaption",
        "blockquote",
        "img",
        "a",
    }
)
#: HTML elements that have no end tag.
VOID_ELEMENTS = frozenset({"br", "hr", "img"})
#: SVG elements, inside an `<svg>` (which opens the SVG context).
SVG_ELEMENTS = (
    "svg",
    "g",
    "defs",
    "title",
    "desc",
    "path",
    "rect",
    "circle",
    "ellipse",
    "line",
    "polyline",
    "polygon",
    "text",
    "tspan",
    "linearGradient",
    "radialGradient",
    "stop",
    "clipPath",
    "mask",
    "marker",
    "pattern",
    "use",
)
#: Attributes allowed on any listed element.
ATTRIBUTES = (
    "class",
    "title",
    "lang",
    "dir",
    "colspan",
    "rowspan",
    "scope",
    "id",
    "viewBox",
    "width",
    "height",
    "x",
    "y",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    "r",
    "rx",
    "ry",
    "d",
    "points",
    "transform",
    "fill",
    "fill-opacity",
    "fill-rule",
    "stroke",
    "stroke-width",
    "stroke-opacity",
    "stroke-dasharray",
    "stroke-linecap",
    "stroke-linejoin",
    "opacity",
    "font-size",
    "font-family",
    "font-weight",
    "text-anchor",
    "dominant-baseline",
    "offset",
    "stop-color",
    "stop-opacity",
    "gradientUnits",
    "gradientTransform",
    "clip-path",
    "mask",
    "marker-start",
    "marker-mid",
    "marker-end",
    "markerWidth",
    "markerHeight",
    "refX",
    "refY",
    "orient",
    "patternUnits",
    "preserveAspectRatio",
)
#: Families of attributes allowed on any listed element, by the start of their name.
ATTRIBUTE_PREFIXES = ("aria-", "data-")
#: Data attributes the page's own script reads as hooks (a sortable or paged table); a
#: fragment that set them could break the script for the whole page.
RESERVED_DATA_ATTRIBUTES = frozenset({"data-sort", "data-page"})
#: Attributes allowed only on some elements: a link inside the page, and an image given
#: inline with its text alternative.
ELEMENT_ATTRIBUTES = {
    "use": ("href", "xlink:href"),
    "a": ("href",),
    "img": ("src", "alt"),
}
#: Attributes whose value may be a reference `url(#name)` to an element of the fragment.
URL_ATTRIBUTES = frozenset(
    {"fill", "stroke", "clip-path", "mask", "marker-start", "marker-mid", "marker-end"}
)
#: Attributes whose value is a list of ids of the page.
ID_LIST_ATTRIBUTES = frozenset({"aria-labelledby", "aria-describedby"})
#: Namespace declarations of a standalone SVG document, with the only values they take.
#: They are implied inside an HTML page, so they are checked and left out of it.
NAMESPACES = {
    "xmlns": "http://www.w3.org/2000/svg",
    "xmlns:xlink": "http://www.w3.org/1999/xlink",
}
#: The image types an `img` may carry inline, as a `data:` URI.
IMAGE_TYPES = ("image/png", "image/jpeg", "image/svg+xml")

__all__ = [
    "ATTRIBUTES",
    "ATTRIBUTE_PREFIXES",
    "ELEMENT_ATTRIBUTES",
    "HTML_ELEMENTS",
    "ID_LIST_ATTRIBUTES",
    "IMAGE_TYPES",
    "NAMESPACES",
    "RESERVED_DATA_ATTRIBUTES",
    "SVG_ELEMENTS",
    "URL_ATTRIBUTES",
    "VOID_ELEMENTS",
]
