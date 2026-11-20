"""The colours of a custom tearsheet (`report.custom.Theme`): the checks of a theme's
token and series colours, and the page's second stylesheet that writes them.

A colour is written into a style sheet, so it must be one of the forms a pattern can
check: `#rgb`, `#rrggbb`, `rgb(...)` or `rgba(...)`, `hsl(...)` or `hsla(...)` holding
numbers (each with a percent or angle unit, or none) or `none`, apart by commas, slashes
or spaces, or a CSS colour name; text such as `red;}` never passes. The second
stylesheet comes after the page's own and writes the given tokens in the three places
the page's writes them (the root, the dark media query for a root not set light, and a
root set dark), so the theme toggle and the reader's setting work as on the default
page. It is written only when a page asks for it: a
theme, or a chart of several series (the series tokens).
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

from ...errors import ConfigError
from .._dashboard import style
from ._colour_names import COLOUR_KEYWORDS, CSS_COLOUR_NAMES

#: The most series one user chart draws, and the series colours a theme sets.
MAX_SERIES = len(style.SERIES)
#: The page's token names, in the order its stylesheet writes them.
TOKENS = tuple(style.LIGHT)
HEX_COLOUR = re.compile(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})")
#: A colour function's argument: a number (sign, decimals, exponent) with a percent or
#: angle unit or none, or the keyword `none` (a missing component).
_ARGUMENT = r"(?:[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?(?:%|deg|rad|grad|turn)?|none)"
FUNCTION_COLOUR = re.compile(
    rf"(?:rgba?|hsla?)\(\s*{_ARGUMENT}(?:(?:\s*[,/]\s*|\s+){_ARGUMENT})*\s*\)"
)


def colour(value: object, where: str) -> str:
    """`value` as a checked CSS colour; ConfigError naming `where` otherwise."""
    if isinstance(value, str):
        text = value.strip()
        lower = text.lower()
        known = lower in CSS_COLOUR_NAMES or lower in COLOUR_KEYWORDS
        if known or HEX_COLOUR.fullmatch(text) or FUNCTION_COLOUR.fullmatch(lower):
            return text
    raise ConfigError(
        f"{where}: {value!r} is not a colour; give #rgb, #rrggbb, rgb(...), hsl(...) or a "
        "CSS colour name"
    )


def token_colours(colours: object, which: str) -> dict[str, str]:
    """The checked {token: colour} of a theme's `which` ("light" or "dark") mapping, in
    the page's token order; {} for None. Raises ConfigError for anything but a mapping,
    an unknown token (naming the tokens) or a colour that does not pass."""
    if colours is None:
        return {}
    if not isinstance(colours, Mapping):
        raise ConfigError(f"Theme {which} maps the page's token names to colours")
    unknown = [name for name in colours if name not in TOKENS]
    if unknown:
        raise ConfigError(
            f"Theme {which}: {unknown[0]!r} is not a token of the page; the tokens are "
            f"{', '.join(TOKENS)}"
        )
    return {
        name: colour(colours[name], f"Theme {which} {name}") for name in TOKENS if name in colours
    }


def series_colours(colours: object) -> tuple[tuple[str, str], ...]:
    """The checked (light, dark) pairs of a theme's series colours, () for None: each
    entry one colour for both themes or a (light, dark) pair, at most `MAX_SERIES`."""
    if colours is None:
        return ()
    if isinstance(colours, str) or not isinstance(colours, Sequence):
        raise ConfigError("Theme series is a list of colours, each one colour or a (light, dark)")
    if len(colours) > MAX_SERIES:
        raise ConfigError(f"Theme series sets at most {MAX_SERIES} colours, got {len(colours)}")
    pairs = []
    for number, entry in enumerate(colours, start=1):
        where = f"Theme series {number}"
        if isinstance(entry, str):
            shared = colour(entry, where)
            pairs.append((shared, shared))
        elif isinstance(entry, Sequence) and len(entry) == 2:
            pairs.append((colour(entry[0], f"{where} light"), colour(entry[1], f"{where} dark")))
        else:
            raise ConfigError(f"{where}: give one colour, or a (light, dark) pair of colours")
    return tuple(pairs)


def checked(theme: Any) -> tuple[dict[str, str], dict[str, str], tuple[tuple[str, str], ...]]:
    """A `report.custom.Theme`'s light and dark token colours and its series pairs,
    checked again as a page is rendered (all empty for None). A theme checks them when it
    is made, but its mappings can be changed afterwards, and only checked text may reach
    the stylesheet. Raises ConfigError as `token_colours` and `series_colours` do."""
    if theme is None:
        return {}, {}, ()
    return (
        token_colours(theme.light, "light"),
        token_colours(theme.dark, "dark"),
        series_colours(theme.series),
    )


def series_tokens(theme_series: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    """The six (light, dark) series colours of a page: the theme's first, then the
    page's own (`style.SERIES`)."""
    return tuple(theme_series) + style.SERIES[len(theme_series) :]


def _declarations(tokens: Mapping[str, str]) -> str:
    return "".join(f"--{name}:{value};" for name, value in tokens.items())


def stylesheet(
    light: Mapping[str, str],
    dark: Mapping[str, str],
    series: Sequence[tuple[str, str]] | None,
) -> str:
    """The page's second stylesheet: the `light` and `dark` token colours and, when a
    block draws several series, the `series` colours as `--series-1` onward; "" when
    there is nothing to write."""
    light, dark = dict(light), dict(dark)
    for number, (light_colour, dark_colour) in enumerate(series or (), start=1):
        light[f"series-{number}"] = light_colour
        dark[f"series-{number}"] = dark_colour
    rules = f":root{{{_declarations(light)}}}" if light else ""
    if dark:
        tokens = _declarations(dark)
        rules += (
            f"@media (prefers-color-scheme: dark){{:root:not([data-theme=light]){{{tokens}}}}}"
            f":root[data-theme=dark]{{{tokens}}}"
        )
    return rules


__all__ = [
    "MAX_SERIES",
    "TOKENS",
    "checked",
    "colour",
    "series_colours",
    "series_tokens",
    "stylesheet",
    "token_colours",
]
