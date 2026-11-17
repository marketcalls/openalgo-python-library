"""The scorecard: one row per check (its number, title, question, figure, rule and
verdict) and the line above it that counts the verdicts and states the wording rule. The
robustness report opens with it, and the tearsheet's "Overfitting checks" section shows
the quick checks in it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ..report import _format as fmt
from ..report._dashboard.tables import head
from ._verdict import FAILED, PASSED, VERDICTS, WORDING_RULE


@dataclass(frozen=True)
class Row:
    """One check of a scorecard, as text; `anchor` is the id of its section in the same
    page, or "" for none."""

    number: int
    title: str
    question: str
    figure: str
    rule: str
    verdict: str
    anchor: str = ""


def verdict_badge(verdict: str) -> str:
    """The verdict as a word, coloured green when passed, red when failed, muted
    otherwise."""
    css = {PASSED: "passed", FAILED: "failed"}.get(verdict, "quiet")
    return f"<span class='verdict {css}'>{fmt.text(verdict)}</span>"


def counts_line(counts: Mapping[str, int]) -> str:
    """How many checks had each verdict, in `VERDICTS` order, leaving out the ones none
    had: "2 passed, 1 failed, 13 not built"."""
    shown = [f"{counts[verdict]} {verdict}" for verdict in VERDICTS if counts.get(verdict)]
    return ", ".join(shown) or "no check"


def lead(counts: Mapping[str, int]) -> str:
    """The scorecard's top line: the counts and the wording rule, as HTML."""
    return (
        f"<p class='lead'><b>{fmt.text(counts_line(counts))}.</b></p>"
        f"<p class='lead muted'>{fmt.text(WORDING_RULE)}</p>"
    )


def table(rows: Iterable[Row]) -> str:
    """The scorecard table; a row's title links to its section when it has one."""
    body = []
    for row in rows:
        title = fmt.text(row.title)
        if row.anchor:
            title = f"<a href='#{row.anchor}'>{title}</a>"
        body.append(
            f"<tr><td class='r nums'>{row.number}</td><td class='l'>{title}</td>"
            f"<td class='l wrap'>{fmt.text(row.question)}</td>"
            f"<td class='l wrap'>{fmt.text(row.figure)}</td>"
            f"<td class='l wrap'>{fmt.text(row.rule)}</td>"
            f"<td class='l'>{verdict_badge(row.verdict)}</td></tr>"
        )
    columns = (
        ("#", "r"),
        ("Check", "l"),
        ("Question", "l"),
        ("Figure", "l"),
        ("Rule", "l"),
        ("Verdict", "l"),
    )
    return (
        f"<div class='scroll'><table class='tbl hover'>{head(*columns)}"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


__all__ = ["Row", "counts_line", "lead", "table", "verdict_badge"]
