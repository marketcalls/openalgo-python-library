"""A custom tearsheet's own text in the page: a user's names, titles and table cells are
HTML-escaped and every character outside ASCII is written as a character reference, so
the page stays ASCII whatever the text holds."""

import html


def escape(value: object) -> str:
    """`value` as text for an HTML element or a quoted attribute: `&`, `<`, `>` and both
    quotes escaped, and every non-ASCII character as `&#N;`."""
    return ascii_only(html.escape(str(value), quote=True))


def ascii_only(markup: str) -> str:
    """`markup` (already HTML) with every non-ASCII character as `&#N;`."""
    return markup.encode("ascii", "xmlcharrefreplace").decode("ascii")


__all__ = ["ascii_only", "escape"]
