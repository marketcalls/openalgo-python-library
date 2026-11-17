"""The HTML robustness report (design 17, section 3): one self-contained page in the
tearsheet's look, built on its page shell (`report._dashboard.page.document`), styles
and cards. Inline SVG, no fetch, light and dark, phone width, readable without colour.

Sections, numbered in order: the Scorecard (how many checks passed, failed and were not
judged, the wording rule, then every listed check with its question, figure, rule and
verdict in report order), a section per check that ran (its verdict, figure, rule and
note, then its charts and summary), and the Notes (how to read the verdicts, the
settings and seed of every check, the plan, and the run's assumptions and warnings).
"""

from __future__ import annotations

from collections.abc import Mapping

from ... import __version__
from ..report import _format as fmt
from ..report._dashboard.layout import SMALL, Section, card, label, stack
from ..report._dashboard.page import document
from ..report._dashboard.sections_run import assumptions_card
from . import _charts
from ._blocks import FanBlock, HistogramBlock, SessionsBlock, TilesBlock
from ._registry import UNITS
from ._report import RunFacts
from ._request import Request
from ._scorecard import Row, lead, table, verdict_badge
from ._verdict import NOT_BUILT, WORDING_RULE, CheckResult

#: What the Notes say about the verdicts beyond the wording rule.
READING = (
    "Informational: the check shows its figures and judges nothing until its limit is "
    "given in its settings.",
    "Not built: the check comes with a later version; the scorecard names what brings it.",
    "Every check draws from its own random stream, seeded with the report's seed plus the "
    "check's number, so the same seed gives the same report on every platform.",
)


def anchor(name: str) -> str:
    """The id of a check's section: its name with hyphens."""
    return name.replace("_", "-")


def _heading(facts: RunFacts, request: Request) -> str:
    span = ""
    if facts.first_session is not None and facts.last_session is not None:
        span = f"{facts.first_session:%Y-%m-%d} &#8594; {facts.last_session:%Y-%m-%d}"
    facts_shown = [
        span,
        f"{fmt.indian_digits(facts.sessions, 0)} sessions",
        f"{fmt.indian_digits(facts.closed, 0)} closed trades",
        f"seed {request.seed}",
    ]
    meta = " &#183; ".join(part for part in facts_shown if part)
    return (
        f"<div class='title'><h1 class='serif'>{fmt.text(facts.strategy)}</h1>"
        f"<span class='nums'>{meta}</span></div>"
    )


def _scorecard(request: Request, results: Mapping[str, CheckResult], counts: dict) -> Section:
    rows = []
    for check in request.listed:
        if check.built:
            r = results[check.name]
            rows.append(
                Row(
                    check.number,
                    check.title,
                    check.question,
                    r.figure,
                    r.rule,
                    r.verdict,
                    anchor(check.name),
                )
            )
        else:
            rule = f"comes with robustness unit {check.unit}, {UNITS[check.unit]}"
            rows.append(Row(check.number, check.title, check.question, "", rule, NOT_BUILT))
    body = card("Every check", "In report order", lead(counts) + table(rows))
    return Section("scorecard", "Scorecard", "Each check, its rule and its verdict", body)


def _tiles(block: TilesBlock) -> str:
    tiles = "".join(
        f"<div class='stat'><div class='eyebrow trunc'>{fmt.text(tile.label)}</div>"
        f"<div class='stat-value nums {tile.tone}' title='{fmt.text(tile.value)}'>"
        f"{fmt.text(tile.value)}</div><div class='stat-sub nums'>{fmt.text(tile.sub)}</div></div>"
        for tile in block.tiles
    )
    return f"{label(fmt.text(block.title))}<div class='stats'>{tiles}</div>"


def _block(block: object) -> str:
    if isinstance(block, TilesBlock):
        return _tiles(block)
    if isinstance(block, FanBlock):
        chart = _charts.fan_chart(block)
    elif isinstance(block, HistogramBlock):
        chart = _charts.histogram_chart(block)
    elif isinstance(block, SessionsBlock):
        chart = _charts.sessions_chart(block)
    else:
        raise TypeError(f"no chart for {type(block).__name__}")
    return card(fmt.text(block.title), fmt.text(block.subtitle), chart)


def _row(blocks: tuple) -> str:
    """A row of blocks: charts side by side, tiles one under the other."""
    if len(blocks) == 2 and not any(isinstance(b, TilesBlock) for b in blocks):
        return f"<div class='grid2'>{''.join(_block(b) for b in blocks)}</div>"
    return "".join(
        _block(b) if i == 0 else f"<div class='gap-s'>{_block(b)}</div>"
        for i, b in enumerate(blocks)
    )


def _settings_text(result: CheckResult) -> str:
    shown = ", ".join(f"{name} {value}" for name, value in result.settings.items())
    return f"{shown}; seed {result.seed}" if result.seed is not None else shown


def _check_section(result: CheckResult, blocks: tuple) -> Section:
    verdict = (
        f"<p class='lead'>{verdict_badge(result.verdict)} &#183; {fmt.text(result.figure)}</p>"
        f"<p class='lead'>Rule: {fmt.text(result.rule)}</p>"
        f"<p class='key'>{fmt.text(result.note)}</p>"
        f"<p class='key'>Settings: {fmt.text(_settings_text(result))}</p>"
    )
    parts = [("", card("Verdict", "", verdict))]
    parts += [(SMALL, _row(row)) for row in blocks]
    return Section(
        anchor(result.name), fmt.text(result.title), fmt.text(result.question), stack(parts)
    )


def _kv(lines: list[tuple[str, str]]) -> str:
    rows = "".join(
        f"<tr><td class='l'>{fmt.text(name)}</td><td class='l'>{fmt.text(value)}</td></tr>"
        for name, value in lines
    )
    return f"<table class='tbl kv'><tbody>{rows}</tbody></table>"


def _notes(
    facts: RunFacts, request: Request, results: Mapping[str, CheckResult], plan: str
) -> Section:
    reading = "".join(f"<p>{fmt.text(line)}</p>" for line in (WORDING_RULE, *READING))
    objective = request.objective
    name = objective if isinstance(objective, str) else getattr(objective, "__name__", "custom")
    asked = "the CPU count" if request.workers is None else str(request.workers)
    general = [
        ("objective", name),
        ("runs", fmt.indian_digits(request.runs, 0)),
        ("seed", str(request.seed)),
        ("workers", f"{asked}; none started, as every check reads the trade list"),
        ("plan", plan),
    ]
    general += [(result.name, _settings_text(result)) for result in results.values()]
    body = stack(
        [
            ("", card("Reading the verdicts", "", f"<div class='prose'>{reading}</div>")),
            (SMALL, card("Settings and seeds", "Of the report and of each check", _kv(general))),
            (SMALL, assumptions_card(list(facts.lines))),
        ]
    )
    return Section("notes", "Notes", "Verdicts, settings, seeds &amp; the run", body)


def render(
    facts: RunFacts,
    request: Request,
    results: tuple[CheckResult, ...],
    blocks: Mapping[str, tuple],
    counts: dict[str, int],
    plan: str,
) -> str:
    """The whole page (module docstring); `blocks` holds what each check's section
    draws, by check name."""
    by_name = {result.name: result for result in results}
    parts = [
        _scorecard(request, by_name, counts),
        *(_check_section(result, blocks[result.name]) for result in results),
        _notes(facts, request, by_name, plan),
    ]
    title = f"{facts.strategy}: robustness report"
    return document(title, _heading(facts, request), parts, __version__, "")


__all__ = ["anchor", "render"]
