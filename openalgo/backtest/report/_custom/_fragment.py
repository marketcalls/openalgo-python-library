"""The escape hatch's check: a fragment of HTML or SVG a user wrote for `html()` or
`svg()` is parsed with the standard library's HTML parser and every element and
attribute is checked against the allow list (`_allowed`); anything not on it is refused
with a ConfigError that names it and the block.

What passes is written back from what was parsed, so the page holds exactly what was
checked: names in their listed spelling, every attribute value quoted and escaped, text
escaped, and every non-ASCII character as a character reference. The fragment's `id`s
are prefixed with its block's anchor (so two blocks cannot clash with each other or
with the page) and its references to them are rewritten to match: `href="#name"` on
`use` and `a`, `url(#name)` in `fill`, `stroke`, `clip-path`, `mask` and `marker-*`,
and the ids listed in `aria-labelledby` and `aria-describedby`.

A value the browser reads as CSS holds no backslash and no quote (font names aside), so
`url(` cannot be spelt past the check; and the fragment holds none of the hooks the
page's script finds its own elements by (`_allowed`), so the script never runs on it.

Its tags must balance, so it cannot break the page around it; an empty element may be
written `<name/>`. Comments, an XML declaration and a document type declaration carry
no content and are left out, as are the namespace declarations of a standalone SVG
document, which an HTML page implies. The check keeps the page offline, with one script
and whole; it is not a sanitiser for untrusted input: the fragment is the user's own.
"""

import html
import re
from html.parser import HTMLParser
from typing import NamedTuple

from ...errors import ConfigError
from . import _allowed as allowed
from ._text import ascii_only, escape

_SVG_NAMES = {name.lower(): name for name in allowed.SVG_ELEMENTS}
_ATTRIBUTE_NAMES = {name.lower(): name for name in allowed.ATTRIBUTES}
_ELEMENT_ATTRIBUTE_NAMES = {
    element: {name.lower(): name for name in names}
    for element, names in allowed.ELEMENT_ATTRIBUTES.items()
}
_ID = r"[A-Za-z_][A-Za-z0-9_.-]*"
#: An id a fragment may give: a letter or underscore, then letters, digits, `_`, `.`, `-`.
ID = re.compile(_ID)
#: A reference to an element of the fragment, as an attribute holds it.
REFERENCE = re.compile(rf"#({_ID})")
URL_REFERENCE = re.compile(rf"\s*url\(\s*#({_ID})\s*\)\s*", re.IGNORECASE)
#: A link inside the page: `#` and an anchor (empty for the top of the page).
IN_PAGE_LINK = re.compile(r"#[A-Za-z0-9_.:-]*")
#: An inline image: a `data:` URI of one of the listed types, with optional parameters.
DATA_IMAGE = re.compile(
    rf"data:({'|'.join(re.escape(kind) for kind in allowed.IMAGE_TYPES)})"
    r"(;[A-Za-z0-9=._+-]+)*,.*",
    re.DOTALL,
)


class _Token(NamedTuple):
    """One piece of the fragment: a start tag (its listed name, checked attributes,
    whether written empty, whether an SVG element), an end tag, or text."""

    kind: str
    name: str
    attributes: tuple[tuple[str, str], ...] = ()
    empty: bool = False
    svg: bool = False


class _Reader(HTMLParser):
    """Parses a fragment, refusing what the allow list does not hold."""

    def __init__(self, where: str, svg_only: bool) -> None:
        super().__init__(convert_charrefs=True)
        self.where, self.svg_only = where, svg_only
        self.tokens: list[_Token] = []
        self.open: list[str] = []
        self.ids: list[str] = []
        self.references: list[str] = []

    def refuse(self, what: str) -> None:
        raise ConfigError(f"{self.where}: {what}")

    def in_svg(self) -> bool:
        return "svg" in self.open

    def element(self, tag: str) -> str:
        """The listed name of element `tag` where it stands, or a refusal."""
        if self.in_svg():
            if tag not in _SVG_NAMES:
                self.refuse(f"<{tag}> is not on the allow list of elements inside an <svg>")
            return _SVG_NAMES[tag]
        if tag == "svg":
            return tag
        if self.svg_only:
            self.refuse(f"an SVG fragment is one or more <svg> elements, found <{tag}>")
        if tag not in allowed.HTML_ELEMENTS:
            self.refuse(f"the element <{tag}> is not on the allow list")
        return tag

    def handle_starttag(self, tag: str, attrs: list) -> None:
        self.start(tag, attrs, empty=False)

    def handle_startendtag(self, tag: str, attrs: list) -> None:
        self.start(tag, attrs, empty=True)

    def start(self, tag: str, attrs: list, empty: bool) -> None:
        name = self.element(tag)
        svg = name == "svg" or self.in_svg()
        checked = tuple(
            pair for key, value in attrs if (pair := self.attribute(name, key, value)) is not None
        )
        empty = empty or (not svg and name in allowed.VOID_ELEMENTS)
        self.tokens.append(_Token("start", name, checked, empty, svg))
        if not empty:
            self.open.append(name)

    def handle_endtag(self, tag: str) -> None:
        if not self.open or self.open[-1].lower() != tag:
            closes = f"<{self.open[-1]}>" if self.open else "nothing"
            self.refuse(f"</{tag}> does not close {closes}: the tags must balance")
        self.tokens.append(_Token("end", self.open.pop()))

    def handle_data(self, data: str) -> None:
        if self.svg_only and not self.open and data.strip():
            self.refuse("an SVG fragment is one or more <svg> elements, found text outside them")
        self.tokens.append(_Token("text", data))

    def unknown_decl(self, data: str) -> None:
        self.refuse(f"<![{data[:20]}...]> is not allowed")

    def attribute(self, element: str, key: str, value: str | None) -> tuple[str, str] | None:
        """The attribute's listed name and value, None for a namespace declaration (left
        out), or a refusal."""
        value = "" if value is None else value
        if key in allowed.NAMESPACES:
            if value != allowed.NAMESPACES[key]:
                self.refuse(f"{key} on <{element}> declares an unknown namespace {value!r}")
            return None
        specific = _ELEMENT_ATTRIBUTE_NAMES.get(element, {})
        if key in specific:
            return specific[key], self.specific(element, specific[key], value)
        if key in _ATTRIBUTE_NAMES:
            name = _ATTRIBUTE_NAMES[key]
        elif key.startswith(allowed.ATTRIBUTE_PREFIXES):
            if script_hook(key):
                self.refuse(f"the attribute {key!r} on <{element}> is a hook of the page's script")
            name = key
        else:
            self.refuse(f"the attribute {key!r} on <{element}> is not on the allow list")
        self.general(element, name, value)
        return name, value

    def general(self, element: str, name: str, value: str) -> None:
        """Checks an attribute allowed on any element: no class the page's script
        selects by, a value read as CSS without a backslash or a quote, an id's form,
        and a `url(...)` only as a reference inside the fragment."""
        if name == "class":
            hooks = [part for part in value.split() if part.lower() in allowed.RESERVED_CLASSES]
            if hooks:
                self.refuse(
                    f"class {hooks[0]!r} on <{element}> is a hook of the page's script; give "
                    "the element another class"
                )
        if name in allowed.CSS_VALUE_ATTRIBUTES:
            self.css_value(element, name, value)
        if name == "id":
            if not ID.fullmatch(value):
                self.refuse(
                    f"id {value!r} on <{element}>: an id is a letter or '_' followed by "
                    "letters, digits, '_', '.' or '-'"
                )
            if value in self.ids:
                self.refuse(f"id {value!r} is given twice: an id names one element")
            self.ids.append(value)
        if "url(" in value.lower():
            match = URL_REFERENCE.fullmatch(value) if name in allowed.URL_ATTRIBUTES else None
            if match is None:
                self.refuse(
                    f"{name}={value!r} on <{element}>: a reference is url(#name) to an element "
                    "of the fragment, in fill, stroke, clip-path, mask or marker-*"
                )
            self.references.append(match.group(1))

    def css_value(self, element: str, name: str, value: str) -> None:
        """Checks a value the browser may read as CSS (`_allowed.CSS_VALUE_ATTRIBUTES`)."""
        if "\\" in value:
            self.refuse(
                f"{name}={value!r} on <{element}>: a backslash is not allowed in a value read "
                "as CSS, where an escape could spell url( unseen"
            )
        if name != allowed.QUOTED_NAMES and ("'" in value or '"' in value):
            self.refuse(
                f"{name}={value!r} on <{element}>: a quote is not allowed in a value read as "
                f"CSS (only {allowed.QUOTED_NAMES} quotes its names)"
            )

    def specific(self, element: str, name: str, value: str) -> str:
        """Checks an attribute allowed only on `element`: a link inside the page or the
        fragment, or an inline image."""
        if element == "img":
            if name == "src" and not DATA_IMAGE.fullmatch(value):
                self.refuse(
                    f"src on <img> is a data: URI of type {', '.join(allowed.IMAGE_TYPES)}: "
                    "an image is carried inside the page"
                )
            return value
        if element == "use":
            match = REFERENCE.fullmatch(value)
            if match is None:
                self.refuse(f"{name}={value!r} on <use>: it points to #name inside the fragment")
            self.references.append(match.group(1))
            return value
        if not IN_PAGE_LINK.fullmatch(value):
            self.refuse(f"href={value!r} on <a>: a link stays inside the page, as href='#...'")
        return value


def script_hook(name: str) -> bool:
    """Whether data attribute `name` is one the page's script reads: a listed name, or
    a listed family's name followed by a number (`_allowed`)."""
    if name in allowed.RESERVED_DATA_ATTRIBUTES:
        return True
    return any(
        name.startswith(family) and name[len(family) :].isdigit()
        for family in allowed.RESERVED_DATA_FAMILIES
    )


def _rewritten(name: str, value: str, ids: frozenset[str], prefix: str) -> str:
    """An attribute's value with the fragment's ids prefixed."""
    if name == "id":
        return f"{prefix}-{value}"
    if name in ("href", "xlink:href"):
        match = REFERENCE.fullmatch(value)
        return f"#{prefix}-{match.group(1)}" if match and match.group(1) in ids else value
    if name in allowed.URL_ATTRIBUTES and (match := URL_REFERENCE.fullmatch(value)):
        return f"url(#{prefix}-{match.group(1)})"
    if name in allowed.ID_LIST_ATTRIBUTES:
        return " ".join(f"{prefix}-{part}" if part in ids else part for part in value.split())
    return value


def _written(token: _Token, ids: frozenset[str], prefix: str) -> str:
    if token.kind == "text":
        return ascii_only(html.escape(token.name, quote=False))
    if token.kind == "end":
        return f"</{token.name}>"
    attributes = "".join(
        f" {name}='{escape(_rewritten(name, value, ids, prefix))}'"
        for name, value in token.attributes
    )
    if not token.empty:
        return f"<{token.name}{attributes}>"
    if token.svg:
        return f"<{token.name}{attributes}/>"
    if token.name in allowed.VOID_ELEMENTS:
        return f"<{token.name}{attributes}>"
    return f"<{token.name}{attributes}></{token.name}>"


def checked(fragment: str, *, where: str, svg_only: bool, prefix: str) -> str:
    """The fragment as the page holds it (module docstring). `where` names the block in
    every refusal, `svg_only` asks for SVG elements only (`svg()`), and `prefix` is the
    block's anchor. Raises ConfigError for anything off the allow list, unbalanced tags,
    a duplicate or malformed id, or a reference to an element the fragment lacks."""
    reader = _Reader(where, svg_only)
    reader.feed(fragment)
    reader.close()
    if reader.open:
        reader.refuse(f"<{reader.open[-1]}> is not closed: the tags must balance")
    ids = frozenset(reader.ids)
    for name in reader.references:
        if name not in ids:
            reader.refuse(f"a reference points to #{name}, which no element of the fragment has")
    return "".join(_written(token, ids, prefix) for token in reader.tokens)


__all__ = ["DATA_IMAGE", "ID", "IN_PAGE_LINK", "checked", "script_hook"]
