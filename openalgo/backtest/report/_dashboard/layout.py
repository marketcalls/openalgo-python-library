"""The dashboard's building blocks: what a page shows (`View`), numbered sections,
cards and grid labels.

A report mode is a set of parts; a section shows the parts of the mode it holds and is
left out when it holds none. Sections are numbered in page order among those shown.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from .._performance import Performance

#: Every part of a page, by section, in page order.
PARTS = {
    "overview": ("overview",),
    "performance": ("cumulative", "rolling", "horizons", "periods"),
    "risk": (
        "drawdown_chart",
        "risk_stats",
        "benchmark_stats",
        "rolling_beta",
        "worst_drawdowns",
        "streaks",
    ),
    "monthly": (
        "monthly_heatmap",
        "weekly_heatmap",
        "monthly_stats",
        "yearly_bars",
        "yearly_table",
    ),
    "distribution": ("distribution_stats", "histogram", "box_plot"),
    "metrics": ("metrics_table", "definitions"),
    "trades": ("trade_stats", "trade_table", "costs", "r_multiples"),
    "strategies": ("strategies",),
    "notes": ("method", "assumptions"),
}
_EVERY = frozenset(part for parts in PARTS.values() for part in parts)
#: The parts of each report mode.
MODE_PARTS = {
    "full": _EVERY - {"definitions"},
    "basic": frozenset(
        {
            "overview",
            "cumulative",
            "periods",
            "drawdown_chart",
            "worst_drawdowns",
            "monthly_heatmap",
            "yearly_bars",
            "yearly_table",
            "trade_stats",
            "trade_table",
            "costs",
            "method",
            "assumptions",
        }
    ),
    "metrics": frozenset({"metrics_table", "definitions", "assumptions"}),
    "charts": frozenset(
        {
            "cumulative",
            "rolling",
            "drawdown_chart",
            "rolling_beta",
            "streaks",
            "monthly_heatmap",
            "weekly_heatmap",
            "yearly_bars",
            "histogram",
            "box_plot",
        }
    ),
}


@dataclass(frozen=True)
class View:
    """What one page shows: the run's performance, the benchmark measured as the
    strategy is (None without one) and its name, the title, the assumption lines, the
    strategies of a combined result (None unless two or more), whether the cumulative
    return is on a log scale, and the parts of the report mode."""

    performance: Performance
    bench: Performance | None
    bench_name: str
    title: str
    assumptions: list[str]
    strategies: Mapping[str, Performance] | None
    log_scale: bool
    parts: frozenset[str]

    def has(self, part: str) -> bool:
        """Whether the page shows `part`."""
        return part in self.parts


@dataclass(frozen=True)
class Section:
    """A section's anchor, title, subtitle and body, before it is numbered."""

    anchor: str
    title: str
    subtitle: str
    body: str


def section(number: int, part: Section) -> str:
    """A numbered section: "01  Overview" over a hairline, the subtitle on the right."""
    subtitle = f"<span class='sec-sub'>{part.subtitle}</span>" if part.subtitle else ""
    return (
        f"<section id='{part.anchor}'><div class='sec-head'><span class='sec-num nums'>"
        f"{number:02d}</span><h2 class='serif'>{part.title}</h2>{subtitle}</div>"
        f"{part.body}</section>"
    )


def card(title: str, subtitle: str, body: str) -> str:
    """A card with a title and an optional subtitle over its body."""
    sub = f"<p class='card-sub'>{subtitle}</p>" if subtitle else ""
    head = f"<header class='card-head'><h3 class='serif'>{title}</h3>{sub}</header>"
    return f"<div class='card'>{head}<div class='card-body'>{body}</div></div>"


def label(text: str) -> str:
    """The small label over a grid of tiles."""
    return f"<div class='eyebrow grid-label'>{text}</div>"


#: The gaps between blocks of a section: a small one between cards of one kind, a large
#: one before a block of another kind (pixels, as the reference dashboard spaces them).
SMALL, LARGE = "gap-s", "gap-l"


def stack(blocks: list[tuple[str, str]]) -> str:
    """(gap, block) pairs one under the other, each block after the first preceded by
    its gap (`SMALL` or `LARGE`); empty blocks are dropped."""
    shown = [(gap, block) for gap, block in blocks if block]
    return "".join(
        block if i == 0 else f"<div class='{gap}'>{block}</div>"
        for i, (gap, block) in enumerate(shown)
    )


__all__ = [
    "LARGE",
    "MODE_PARTS",
    "PARTS",
    "SMALL",
    "Section",
    "View",
    "card",
    "label",
    "section",
    "stack",
]
