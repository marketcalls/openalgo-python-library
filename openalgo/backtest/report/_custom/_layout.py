"""Resolution of a custom tearsheet's layout (`report.custom.Layout`): the checks a layout
passes when it is made, which built-in parts it shows, the order of its sections, and
the page it renders.

    parts      the parts of the base mode (`MODE_PARTS[base]`), less those hidden (a
               hidden section hides all its parts), plus those shown
    sections   the built-in sections in page order, each built from the `View` with
               those parts and left out when it shows nothing, as on the default page;
               a custom section goes after the section its `after` names (by default
               before Notes; several after one section keep the layout's order), then
               the sections `order` names come first, in its order, and the rest keep
               their places after them
    additions  a custom section named as a built-in section (its key in `PARTS`, in any
               case) adds its blocks at the end of that section, after a large gap; when
               the page leaves that section out (Strategies on a single strategy, Trades
               on a run without trades) the blocks form it, under its own title, printed
               as the custom section asks
    anchors    a custom section's anchor is `u-` and its name made a slug (`u-exposure`),
               so it never takes a built-in anchor; its blocks' anchors follow it after
               a "_" (`u-exposure_1`), which no slug holds, so no id a block writes is
               another section's anchor

The page is the default page's shell (`page.render`) with these sections and, when a
theme or a chart of several series asks for one, a second stylesheet (`_theme`); a
portfolio's Portfolio section counts as such a chart (its monthly bars by strategy), so a
theme's series colours colour its strategies.
"""

import dataclasses
import re
from collections.abc import Sequence
from typing import Any, NamedTuple

from ...errors import ConfigError
from .._dashboard import page
from .._dashboard import portfolio as portfolio_section
from .._dashboard.layout import LARGE, MODE_PARTS, PARTS, TITLES, Section, View, stack
from . import _render, _theme
from ._text import escape

#: The built-in sections, in page order.
BUILT_IN = tuple(PARTS)
#: The section each built-in part belongs to.
SECTION_OF = {part: name for name, parts in PARTS.items() for part in parts}
#: A custom section with no `after` goes before this section (at the end without it).
DEFAULT_BEFORE = "notes"
#: The start of a custom section's anchor; no built-in anchor starts so.
ANCHOR_PREFIX = "u-"
SLUG_BREAK = re.compile(r"[^a-z0-9]+")


class Plan(NamedTuple):
    """What the checks read of a custom section: its name, `after`, subtitle and
    whether it is printed."""

    name: str
    after: str | None
    subtitle: str
    printed: bool


def key(name: str) -> str:
    """A section's name as the layout matches it: trimmed, lower case."""
    return name.strip().lower()


def anchor(name: str) -> str:
    """A custom section's anchor: `u-` and its name as a slug."""
    return ANCHOR_PREFIX + SLUG_BREAK.sub("-", key(name)).strip("-")


def names(value: object, what: str) -> tuple[str, ...]:
    """`hide`, `show` or `order` as a tuple of names: one name, or a list or tuple of
    them. Raises ConfigError for anything else."""
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple) and all(isinstance(name, str) for name in value):
        return tuple(value)
    raise ConfigError(f"Layout {what} is a name or a tuple of names, got {value!r}")


def _listing() -> str:
    return "; ".join(f"{section}: {', '.join(parts)}" for section, parts in PARTS.items())


def parts_of(base: str, hide: Sequence[str], show: Sequence[str]) -> frozenset[str]:
    """The built-in parts a layout shows (module docstring)."""
    hidden = {part for name in hide for part in PARTS.get(name, (name,))}
    return frozenset((MODE_PARTS[base] - hidden) | set(show))


def shown_sections(parts: frozenset[str]) -> list[str]:
    """The built-in sections that show one of `parts`, in page order."""
    return [name for name in BUILT_IN if any(part in parts for part in PARTS[name])]


def _check_names(hide: Sequence[str], show: Sequence[str]) -> None:
    for name in hide:
        if name not in SECTION_OF and name not in PARTS:
            raise ConfigError(
                f"Layout hide: {name!r} is neither a built-in section nor a part; the sections "
                f"and their parts are {_listing()}"
            )
    for name in show:
        if name in PARTS and name not in SECTION_OF:
            raise ConfigError(
                f"Layout show takes part names: {name!r} is a section, whose parts are "
                f"{', '.join(PARTS[name])}"
            )
        if name not in SECTION_OF:
            raise ConfigError(f"Layout show: {name!r} is not a part; the parts are {_listing()}")
        if name in hide or SECTION_OF[name] in hide:
            raise ConfigError(f"Layout both hides and shows {name!r}")


def _check_section(plan: Plan, shown: list[str], earlier: set[str]) -> None:
    """One custom section: an addition to a shown built-in section, or a new section
    whose `after` names a shown section or a custom section listed before it."""
    name = key(plan.name)
    if name in PARTS:
        if name not in shown:
            raise ConfigError(
                f"section {plan.name!r} adds to the built-in section {name!r}, which the layout "
                f"hides: show one of its parts ({', '.join(PARTS[name])}) or rename the section"
            )
        if plan.after is not None or plan.subtitle:
            raise ConfigError(
                f"section {plan.name!r} adds its blocks to the built-in section {name!r}, which "
                "keeps its own place and subtitle: give it no after and no subtitle"
            )
        return
    if anchor(plan.name) == ANCHOR_PREFIX:
        raise ConfigError(f"section name {plan.name!r} needs a letter or a digit for its link")
    if plan.after is None or key(plan.after) in shown or key(plan.after) in earlier:
        return
    raise ConfigError(
        f"section {plan.name!r}: after names {plan.after!r}, which is neither a section the "
        f"layout shows ({', '.join(shown)}) nor a custom section listed before it"
    )


def _check_sections(plans: Sequence[Plan], shown: list[str]) -> None:
    seen: set[str] = set()
    anchors: dict[str, str] = {}
    for plan in plans:
        name = key(plan.name)
        if name in seen:
            raise ConfigError(f"Layout has two sections named {plan.name!r}")
        _check_section(plan, shown, seen - set(PARTS))
        seen.add(name)
        if name in PARTS:
            continue
        link = anchor(plan.name)
        if link in anchors:
            raise ConfigError(
                f"sections {anchors[link]!r} and {plan.name!r} would share the link #{link}: "
                "rename one"
            )
        anchors[link] = plan.name


def _check_order(order: Sequence[str] | None, shown: list[str], custom: set[str]) -> None:
    seen: set[str] = set()
    for name in order or ():
        if key(name) in seen:
            raise ConfigError(f"Layout order names {name!r} twice")
        if key(name) not in shown and key(name) not in custom:
            raise ConfigError(
                f"Layout order: {name!r} is not a section the layout shows; it shows "
                f"{', '.join(shown)} and its custom sections"
            )
        seen.add(key(name))


def check(
    base: str,
    hide: Sequence[str],
    show: Sequence[str],
    order: Sequence[str] | None,
    plans: Sequence[Plan],
) -> None:
    """The checks of a layout when it is made (`report.custom.Layout`). Raises
    ConfigError, naming the valid choices, for an unknown base, part or section, a name
    hidden and shown at once, two custom sections of one name or link, an `after` or
    `order` naming no section the layout shows, or additions to a hidden section."""
    if not isinstance(base, str) or base not in MODE_PARTS:
        raise ConfigError(f"Layout base is one of {tuple(MODE_PARTS)}, got {base!r}")
    _check_names(hide, show)
    shown = shown_sections(parts_of(base, hide, show))
    _check_sections(plans, shown)
    custom = {key(plan.name) for plan in plans if key(plan.name) not in PARTS}
    _check_order(order, shown, custom)


def page_order(layout: Any) -> list[str]:
    """The keys of the layout's sections in page order (module docstring), before the
    sections the page leaves out for want of anything to show."""
    order = shown_sections(parts_of(layout.base, layout.hide, layout.show))
    last_after: dict[str, str] = {}
    for custom in layout.sections:
        name = key(custom.name)
        if name in PARTS:
            continue
        if custom.after is None:
            at = order.index(DEFAULT_BEFORE) if DEFAULT_BEFORE in order else len(order)
        else:
            target = key(custom.after)
            at = order.index(last_after.get(target, target)) + 1
            last_after[target] = name
        order.insert(at, name)
    named = [key(name) for name in layout.order or ()]
    return named + [name for name in order if name not in named]


def _section(name: str, custom: Any, built: Section | None, placed: _render.Placed) -> tuple:
    """A section with a custom section's blocks, and whether they use series colours."""
    markup, several = _render.body(custom.blocks, placed)
    if name not in PARTS:
        title, subtitle = escape(custom.name), escape(custom.subtitle)
        return Section(placed.anchor, title, subtitle, markup, custom.printed), several
    if built is None:
        return Section(name, TITLES[name], "", markup, custom.printed), several
    if not custom.printed:
        markup = f"<div class='no-print'>{markup}</div>"
    body = stack([("", built.body), (LARGE, markup)])
    return dataclasses.replace(built, body=body), several


def render(layout: Any, view: View, subject: object, version: str) -> str:
    """The custom page of `layout` for `view`; `subject` is what the blocks' callables
    receive and `version` the SDK's. Raises ConfigError, naming the block and its
    section, for a block whose data cannot be drawn."""
    built = {section.anchor: section for section in page.built(view)}
    customs = {key(custom.name): custom for custom in layout.sections}
    sessions = view.performance.session_returns.index
    sections, several = [], False
    for name in page_order(layout):
        custom = customs.get(name)
        if custom is None:
            if name in built:
                sections.append(built[name])
            continue
        placed = _render.Placed(sessions, subject, custom.name, anchor(custom.name))
        section, uses = _section(name, custom, built.get(name), placed)
        sections.append(section)
        several = several or uses
    light, dark, theme_series = _theme.checked(layout.theme)
    several = several or portfolio_section.draws_series(view, sections)
    series = _theme.series_tokens(theme_series) if several else None
    extra = _theme.stylesheet(light, dark, series)
    return page.render(view, version, sections, extra_style=extra)


__all__ = [
    "ANCHOR_PREFIX",
    "BUILT_IN",
    "Plan",
    "anchor",
    "check",
    "key",
    "names",
    "page_order",
    "parts_of",
    "render",
    "shown_sections",
]
