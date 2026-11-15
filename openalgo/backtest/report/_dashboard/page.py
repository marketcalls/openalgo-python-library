"""The page: the head, the sticky header (mark, section links, Export PDF and the theme
toggle), the title line, the numbered sections the mode shows and the footer."""

from .. import _format as fmt
from . import script, sections, sections_run, style
from .layout import View, section
from .timeseries import MIN_TIME_GAP_PX

#: The sections in page order; each returns None when the mode shows none of it.
SECTIONS = (
    sections.overview,
    sections.performance,
    sections.risk,
    sections.monthly,
    sections.distribution,
    sections_run.metrics,
    sections_run.trades,
    sections_run.strategies,
    sections_run.notes,
)
#: The class of the section link marked as in view (the first, until script runs).
ACTIVE = " class='on'"
#: The mark: a rising line in a ring, drawn in the text colour.
MARK = (
    "<div class='mark'><span class='mark-ring'><svg width='18' height='18' viewBox='0 0 24 24' "
    "fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' "
    "stroke-linejoin='round' aria-hidden='true'><path d='M3 17l5-6 4 3 5-8 4 5'/></svg></span>"
    "<div><div class='serif mark-name'>OpenAlgo</div><div class='eyebrow'>openalgo.in</div>"
    "</div></div>"
)
_ICON = (
    "width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' "
    "stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round' aria-hidden='true'"
)
#: The theme toggle: a moon to switch to dark, a sun to switch to light.
TOGGLE = (
    "<button type='button' class='theme js-only' aria-label='Switch theme' title='Theme'>"
    f"<svg class='moon' {_ICON}><path d='M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z'/></svg>"
    f"<svg class='sun' {_ICON}><circle cx='12' cy='12' r='4'/><path d='M12 2v2M12 20v2M4.93 "
    "4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-"
    "1.41'/></svg></button>"
)


def _header(nav: str) -> str:
    return (
        f"<header class='top no-print'><div class='top-in'>{MARK}<nav class='nav'>{nav}</nav>"
        "<div class='tools'><button type='button' class='btn print js-only'>Export PDF</button>"
        f"{TOGGLE}</div></div></header>"
    )


def _title(view: View) -> str:
    p = view.performance
    dates = p.session_returns.index
    span = f"{dates[0]:%Y-%m-%d} &#8594; {dates[-1]:%Y-%m-%d}" if len(dates) else ""
    sessions = int(p.metrics["sessions"])
    facts = [span, f"{fmt.indian_digits(sessions, 0)} {'session' if sessions == 1 else 'sessions'}"]
    if p.benchmark_returns is not None:
        facts.append(f"vs {fmt.text(view.bench_name)}")
    meta = " &#183; ".join(fact for fact in facts if fact)
    return (
        f"<div class='title'><h1 class='serif'>{fmt.text(view.title)}</h1>"
        f"<span class='nums'>{meta}</span></div>"
    )


def _footer(version: str) -> str:
    return (
        "<footer class='foot no-print'><span class='serif'>OpenAlgo</span>"
        f"<span class='nums'>{fmt.text(version)}</span><span>&#183; openalgo.backtest</span>"
        "<span class='end'>openalgo.in</span></footer>"
    )


def render(view: View, version: str) -> str:
    """The whole page for `view`; `version` is the SDK's, shown in the footer."""
    parts = [part for part in (build(view) for build in SECTIONS) if part is not None]
    nav = "".join(
        f"<a href='#{part.anchor}'{ACTIVE if i == 0 else ''}>{part.title}</a>"
        for i, part in enumerate(parts)
    )
    body = "".join(section(i + 1, part) for i, part in enumerate(parts))
    dates = ",".join(f"{d:%Y-%m-%d}" for d in view.performance.session_returns.index)
    return (
        "<!DOCTYPE html>\n<html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<meta name='color-scheme' content='light dark'>"
        f"<title>{fmt.text(view.title)}: backtest tearsheet</title>"
        f"<script>{script.HEAD}</script><style>{style.stylesheet()}</style></head>"
        f"<body data-dates='{dates}'>{_header(nav)}<main>{_title(view)}{body}"
        f"{_footer(version)}</main><script>{script.script(MIN_TIME_GAP_PX)}</script>"
        "</body></html>\n"
    )


__all__ = ["SECTIONS", "render"]
