"""Custom tearsheets: choose which built-in sections and charts a tearsheet shows, add
sections and charts of your own, and set the page's colours. The built-in tearsheet
stays the default; a layout is used only when it is passed.

    from openalgo.backtest import report as rp

    layout = rp.Layout(
        base="full",                                       # the built-in mode it starts from
        hide=("rolling_beta", "weekly_heatmap", "monthly"),   # parts or whole sections
        show=("definitions",),                             # parts the base leaves out
        theme=rp.Theme(light={"pnl-pos": "#0b7a3e"}, dark={"pnl-pos": "#4ade80"}),
        sections=(
            rp.section(
                "Exposure",
                rp.line(lambda r: r.equity["margin_rupees"], title="Margin blocked"),
                rp.histogram(lambda r: r.trade_list()["bars_held"], title="Bars held"),
                after="risk",
            ),
            rp.section(                                    # a built-in section's name:
                "trades",                                  # its blocks go at its end
                rp.scatter(lambda r: r.trade_list(), x="days_held", y="return_pct",
                           title="Return against holding time"),
            ),
        ),
    )
    result.tearsheet("custom.html", layout=layout)

A block's data is a pandas object (or text, for `text`, `html` and `svg`), or a callable
that takes the result being rendered (a `Result`, or the object handed to
`report.tearsheet`) and returns one, so one layout serves every run. User charts are
drawn by the tearsheet's own chart code, in its look; the page stays one offline file.

Names: `Layout`, `Theme`, `section` (and the `CustomSection` it returns), the blocks
`line`, `bar`, `histogram`, `scatter`, `heatmap`, `table`, `tiles`, `text`, `html` and
`svg` (each returns a `Block`), `PARTS` (the built-in sections and their parts) and
`MAX_SERIES` (the most series one chart draws).
"""

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from types import MappingProxyType

from ..errors import ConfigError
from ._custom import _layout, _theme
from ._custom._blocks import (
    Block,
    bar,
    heatmap,
    histogram,
    html,
    line,
    scatter,
    svg,
    table,
    text,
    tiles,
)
from ._dashboard import layout as _dashboard_layout

#: The built-in sections of the tearsheet, in page order, each with its parts (the names
#: `Layout`'s `hide` and `show` take).
PARTS = MappingProxyType(_dashboard_layout.PARTS)
#: The most series one user chart draws (columns of a line or bar chart, groups of a
#: scatter chart), and the most series colours a `Theme` sets.
MAX_SERIES = _theme.MAX_SERIES


@dataclass(frozen=True)
class Theme:
    """The page's colours. `light` and `dark` map the page's token names to colours for
    its light and dark themes (a token not given keeps the page's value): bg, panel,
    panel-2, border, border-strong, grid, text, text-muted, text-faint, accent,
    accent-ink, label, pnl-pos, pnl-neg, heat-neg, heat-mid, heat-pos, benchmark.
    `series` sets up to `MAX_SERIES` colours of user charts with several series, each
    one colour for both themes or a (light, dark) pair; the rest keep the page's (the
    strategy's and the benchmark's colours, then four more).

    A colour is `#rgb`, `#rrggbb`, `rgb(...)`, `hsl(...)` or a CSS colour name. Raises
    ConfigError for an unknown token or a colour in any other form, which is never
    written into the page."""

    light: Mapping[str, str] | None = None
    dark: Mapping[str, str] | None = None
    series: Sequence[str | tuple[str, str]] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "light", _theme.token_colours(self.light, "light"))
        object.__setattr__(self, "dark", _theme.token_colours(self.dark, "dark"))
        object.__setattr__(self, "series", _theme.series_colours(self.series))


@dataclass(frozen=True)
class CustomSection:
    """A section of a custom tearsheet (made by `section`): its name, its blocks, its
    subtitle, the section it follows, and whether Export PDF prints it."""

    name: str
    blocks: tuple[Block, ...]
    subtitle: str = ""
    after: str | None = None
    printed: bool = True


def section(
    name: str,
    *blocks: Block,
    subtitle: str = "",
    after: str | None = None,
    printed: bool = True,
) -> CustomSection:
    """A section of `blocks`, in order. `name` is its heading and its link in the
    page's header; named as a built-in section (a key of `PARTS`, in any case), it adds
    its blocks at the end of that section instead, and then takes no `subtitle` or
    `after`. `after` names the section it follows (a built-in section or a custom one
    listed before it; by default it goes before Notes). `printed=False` leaves it out of
    Export PDF. Raises ConfigError for an empty name, no block, or a block not made by
    the block functions."""
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("a section's name is text that is not empty")
    if not blocks:
        raise ConfigError(f"section {name!r} needs at least one block")
    for block in blocks:
        if not isinstance(block, Block):
            raise ConfigError(
                f"section {name!r}: {block!r} is not a block; make one with line, bar, "
                "histogram, scatter, heatmap, table, tiles, text, html or svg"
            )
    if not isinstance(subtitle, str):
        raise ConfigError(f"section {name!r}: the subtitle is text, got {subtitle!r}")
    if after is not None and not isinstance(after, str):
        raise ConfigError(f"section {name!r}: after names a section, got {after!r}")
    if not isinstance(printed, bool):
        raise ConfigError(f"section {name!r}: printed is True or False, got {printed!r}")
    return CustomSection(name, tuple(blocks), subtitle, after, printed)


@dataclass(frozen=True)
class Layout:
    """A custom tearsheet: plain data, reusable across results (and picklable when its
    callables are). Pass it as `layout=` to `Result.tearsheet` or `report.tearsheet`.

    base       the built-in mode it starts from, one of "full", "basic", "metrics" and
               "charts": its parts are that mode's
    hide       built-in part names, or section names (all their parts), to leave out
               (`PARTS` lists both)
    show       built-in part names to add to the base
    order      section names (built-in and custom) in the order the page shows them;
               sections not named keep their places after the named ones. Default: the
               built-in order, each custom section after the section its `after` names
    sections   custom sections (`section`); one named as a built-in section adds its
               blocks at that section's end
    theme      a `Theme`, or None for the page's own colours

    `Layout()` is today's full page and `Layout(base=m)` today's page in mode `m`, byte
    for byte. Every section shown gets a number and a link in the header, in page order;
    a section with nothing to show is left out, as on the default page. Raises
    ConfigError, naming the valid choices, for an unknown base, part or section, a name
    hidden and shown at once, two custom sections of one name, an `after` or `order`
    naming no section the layout shows, or a custom section adding to a hidden
    section."""

    base: str = "full"
    _: KW_ONLY
    hide: tuple[str, ...] = ()
    show: tuple[str, ...] = ()
    order: tuple[str, ...] | None = None
    sections: tuple[CustomSection, ...] = ()
    theme: Theme | None = None

    def __post_init__(self) -> None:
        hide, show = _layout.names(self.hide, "hide"), _layout.names(self.show, "show")
        order = None if self.order is None else _layout.names(self.order, "order")
        sections = self.sections
        if not isinstance(sections, list | tuple) or not all(
            isinstance(one, CustomSection) for one in sections
        ):
            raise ConfigError("Layout sections is a tuple of sections made by section(...)")
        if self.theme is not None and not isinstance(self.theme, Theme):
            raise ConfigError(f"Layout theme is a Theme or None, got {self.theme!r}")
        for name, value in (("hide", hide), ("show", show), ("order", order)):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "sections", tuple(sections))
        plans = [_layout.Plan(s.name, s.after, s.subtitle, s.printed) for s in sections]
        _layout.check(self.base, hide, show, order, plans)


__all__ = [
    "MAX_SERIES",
    "PARTS",
    "Block",
    "CustomSection",
    "Layout",
    "Theme",
    "bar",
    "heatmap",
    "histogram",
    "html",
    "line",
    "scatter",
    "section",
    "svg",
    "table",
    "text",
    "tiles",
]
