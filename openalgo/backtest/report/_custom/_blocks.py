"""The user blocks of a custom tearsheet: `Block`, the functions that make one (`line`,
`bar`, `histogram`, `scatter`, `heatmap`, `table`, `tiles`, `text`, `html`, `svg`;
`report.custom` exports them), and the check of a block's data (`prepared`).

A block's `data` is a value, or a callable that takes the result being rendered (a
`Result`, or the object handed to `report.tearsheet`) and returns one, so one layout
serves every run. Data given as a value is checked when the block is made; data a
callable returns, when the page is rendered. Every refusal is a ConfigError naming the
block, and at render its section too.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ...errors import ConfigError
from . import _data
from ._fragment import checked

#: Where a block sits in its section's grid: a whole row, half of one, or a third.
WIDTHS = ("full", "half", "third")
#: Bins asked of a user histogram by default.
DEFAULT_BINS = 30
#: The kinds whose data is a fragment checked against the allow list.
FRAGMENTS = ("html", "svg")

Data = Any | Callable[[Any], Any]


@dataclass(frozen=True)
class Block:
    """One user block: its kind, its data (a value or a callable of the result), title,
    subtitle, width ("full", "half" or "third"), whether its values are fractions shown
    as percent, and the options of its kind as (name, value) pairs. Made by the block
    functions of `report.custom`, which check every field."""

    kind: str
    data: Data
    title: str | None
    subtitle: str = ""
    width: str = "full"
    percent: bool = False
    options: tuple[tuple[str, Any], ...] = ()

    def option(self, name: str, default: Any = None) -> Any:
        """The value of option `name`, or `default` when the block has none."""
        return dict(self.options).get(name, default)

    @property
    def where(self) -> str:
        """The block as an error names it: its kind and title."""
        return named(self.kind, self.title)


def named(kind: str, title: object) -> str:
    """A block as an error names it: "line block 'Margin'", or "untitled text block"."""
    return f"{kind} block {title!r}" if title else f"untitled {kind} block"


def prepared(block: Block, data: object, where: str) -> object:
    """`data` checked for `block`'s kind and put in the shape its drawing takes (`_data`):
    (a DataFrame, whether several series) for a line (indexed by dates) or a bar,
    floats for a histogram, `_data.Points` for a scatter, a DataFrame for a heatmap or a
    table, {label: value} for tiles, text for the rest (a fragment checked against the
    allow list, `_fragment`)."""
    kind = block.kind
    if kind in ("line", "bar"):
        frame, several = _data.series_frame(data, where, block.title or kind)
        return (_data.dated(frame, where) if kind == "line" else frame), several
    if kind == "histogram":
        return _data.numbers(data, where)
    if kind == "scatter":
        x, y, group = (block.option(name) for name in ("x", "y", "group"))
        return _data.points(data, where, x, y, group)
    if kind == "heatmap":
        return _data.heat(data, where)
    if kind == "table":
        return _data.frame_of(data, where)
    if kind == "tiles":
        return _data.tile_values(data, where)
    content = _data.text_of(data, where)
    if kind in FRAGMENTS:
        # The check does not depend on the anchor the page later prefixes ids with.
        checked(content, where=where, svg_only=kind == "svg", prefix=kind)
    return content


def _flag(value: object, name: str, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where}: {name} is True or False, got {value!r}")
    return value


def _count(value: object, name: str, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{where}: {name} is a whole number of 1 or more, got {value!r}")
    return value


def make(
    kind: str,
    data: Data,
    *,
    title: str | None,
    subtitle: str = "",
    width: str = "full",
    percent: bool = False,
    titled: bool = True,
    **options: Any,
) -> Block:
    """A checked block of `kind`; `titled` says whether the kind needs a title. Raises
    ConfigError for an empty or missing title, an unknown width, a flag that is not a
    bool, or data given as a value that the kind does not take."""
    where = named(kind, title)
    if title is not None and (not isinstance(title, str) or not title.strip()):
        raise ConfigError(f"{where}: the title is text that is not empty, or None")
    if titled and title is None:
        raise ConfigError(f"a {kind} block needs a title")
    if not isinstance(subtitle, str):
        raise ConfigError(f"{where}: the subtitle is text, got {subtitle!r}")
    if width not in WIDTHS:
        raise ConfigError(f"{where}: width is one of {WIDTHS}, got {width!r}")
    percent = _flag(percent, "percent", where)
    block = Block(kind, data, title, subtitle, width, percent, tuple(options.items()))
    if not callable(data):
        prepared(block, data, where)
    return block


def line(
    data: Data,
    *,
    title: str,
    subtitle: str = "",
    width: str = "full",
    percent: bool = False,
    area: bool = False,
    zero_line: bool = False,
    log_scale: bool = False,
) -> Block:
    """A time-series chart: a Series, or a DataFrame with one line per column (at most
    `MAX_SERIES`), indexed by dates or datetimes. It is drawn on the run's sessions, as
    the page's own charts are: one point per session, the last known value on or before
    the session's IST date (a series that starts later is blank before it), with the
    page's axes, crosshair and hover readout. A Series is drawn in the strategy's colour,
    the columns of a DataFrame in the series colours with a legend. `area` fills under
    each line, `zero_line` draws a dashed line at zero, `log_scale` draws growth on a log
    scale (returns as fractions, so it needs `percent=True`). Raises ConfigError for an
    index that is not dates, a column that is not numbers or too many columns."""
    where = named("line", title)
    for name, value in (("area", area), ("zero_line", zero_line), ("log_scale", log_scale)):
        _flag(value, name, where)
    if log_scale and not percent:
        raise ConfigError(f"{where}: log_scale draws growth, so give returns with percent=True")
    return make(
        "line",
        data,
        title=title,
        subtitle=subtitle,
        width=width,
        percent=percent,
        area=area,
        zero_line=zero_line,
        log_scale=log_scale,
    )


def bar(
    data: Data,
    *,
    title: str,
    subtitle: str = "",
    width: str = "full",
    percent: bool = False,
    horizontal: bool = False,
) -> Block:
    """A bar chart: a Series (category to value) in the strategy's colour, or a
    DataFrame of grouped bars, one per column in the series colours with a legend (at
    most `MAX_SERIES` columns). The categories are the index, in its order. A zero line
    is drawn when the values change sign. `horizontal` draws a row per category."""
    _flag(horizontal, "horizontal", named("bar", title))
    return make(
        "bar",
        data,
        title=title,
        subtitle=subtitle,
        width=width,
        percent=percent,
        horizontal=horizontal,
    )


def histogram(
    data: Data,
    *,
    title: str,
    subtitle: str = "",
    width: str = "full",
    percent: bool = False,
    bins: int = DEFAULT_BINS,
) -> Block:
    """A histogram of a Series or an array of numbers in about `bins` round-edged bins
    (the page's histogram: values at or above zero green, below red, a dashed line at
    the mean, a count axis). NaN values are left out and counted in the subtitle."""
    _count(bins, "bins", named("histogram", title))
    return make(
        "histogram", data, title=title, subtitle=subtitle, width=width, percent=percent, bins=bins
    )


def scatter(
    data: Data,
    *,
    x: str,
    y: str,
    title: str,
    subtitle: str = "",
    width: str = "full",
    percent: bool = False,
    group: str | None = None,
) -> Block:
    """A scatter chart of a DataFrame's column `x` (across) against column `y` (up),
    both numbers; `group` names a column whose values get the series colours and a
    legend (at most `MAX_SERIES` groups). Points without both numbers are left out and
    counted in the subtitle. Raises ConfigError for a column the data does not have."""
    return make(
        "scatter",
        data,
        title=title,
        subtitle=subtitle,
        width=width,
        percent=percent,
        x=x,
        y=y,
        group=group,
    )


def heatmap(
    data: Data, *, title: str, subtitle: str = "", width: str = "full", percent: bool = False
) -> Block:
    """A heatmap of a DataFrame of numbers, rows by columns, in a table whose cells take
    the page's heat colours: gains green and losses red, in proportion to their size
    against the largest in the map (a bound symmetric around zero, as the monthly
    heatmap's)."""
    return make("heatmap", data, title=title, subtitle=subtitle, width=width, percent=percent)


def table(
    data: Data,
    *,
    title: str,
    subtitle: str = "",
    width: str = "full",
    percent: bool = False,
    rows: int | None = None,
    sortable: bool = True,
) -> Block:
    """A DataFrame in the page's table style: numbers in Indian notation (fractions as
    percent with `percent`), times as "12 Nov 2026 09:15", text as it is; its index as
    the first column unless it is the plain 0, 1, 2 ... index. `sortable` lets a click
    on a heading sort it (the page's script), `rows` shows that many rows a page with
    Prev and Next (every row prints and reads without script)."""
    where = named("table", title)
    _flag(sortable, "sortable", where)
    if rows is not None:
        _count(rows, "rows", where)
    return make(
        "table",
        data,
        title=title,
        subtitle=subtitle,
        width=width,
        percent=percent,
        rows=rows,
        sortable=sortable,
    )


def tiles(
    data: Data,
    *,
    title: str | None = None,
    subtitle: str = "",
    width: str = "full",
    percent: bool = False,
    rupees: bool = False,
) -> Block:
    """Stat tiles, the page's: a mapping (or Series) of label to value, a number or text
    shown as it is. A number reads in Indian notation, as percent with `percent` (a
    fraction), or as rupees with `rupees`, as the page shows money (paise below Rs
    1,000); None or NaN reads "n/a". With a title, it is the small label over the
    tiles."""
    where = named("tiles", title)
    if _flag(rupees, "rupees", where) and percent:
        raise ConfigError(f"{where}: a tile is shown as rupees or as percent, not both")
    return make(
        "tiles",
        data,
        title=title,
        subtitle=subtitle,
        width=width,
        percent=percent,
        titled=False,
        rupees=rupees,
    )


def text(data: Data, *, title: str | None = None, subtitle: str = "", width: str = "full") -> Block:
    """Plain text in the page's note style, escaped; a blank line starts a paragraph."""
    return make("text", data, title=title, subtitle=subtitle, width=width, titled=False)


def html(data: Data, *, title: str | None = None, subtitle: str = "", width: str = "full") -> Block:
    """An HTML fragment of the user's, inserted in a card after the allow list's check
    (`report._custom._fragment`; its elements and attributes are in the docs): no
    script, no style, nothing fetched, links only inside the page, images only inline
    as `data:` PNG, JPEG or SVG, and balanced tags. Its ids are prefixed with the
    block's anchor. Raises ConfigError naming what is not allowed."""
    return make("html", data, title=title, subtitle=subtitle, width=width, titled=False)


def svg(data: Data, *, title: str | None = None, subtitle: str = "", width: str = "full") -> Block:
    """An SVG document or element of the user's, inserted in a card after the same
    check as `html` (SVG elements only); a standalone document's XML declaration, type
    declaration, comments and namespace declarations are left out, as an HTML page
    implies them."""
    return make("svg", data, title=title, subtitle=subtitle, width=width, titled=False)


__all__ = [
    "DEFAULT_BINS",
    "WIDTHS",
    "Block",
    "bar",
    "heatmap",
    "histogram",
    "html",
    "line",
    "make",
    "named",
    "prepared",
    "scatter",
    "table",
    "text",
    "tiles",
    "svg",
]
