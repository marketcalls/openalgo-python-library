"""Rules for the written parts of a fixture: its derivation and its plain-text style."""

from __future__ import annotations

from pathlib import Path

from .timegrid import Case

Violations = list[str]

# What a derivation step may cite: a PRD document by its file-name prefix (followed by a
# section), the reference note on the OpenAlgo /strategy module, or a named golden vector
# of the OpenAlgo server's risk core (test/risk/vectors.json in the server repository).
CITABLE_SOURCES = (
    "03-architecture ",
    "04-execution ",
    "05-risk-management ",
    "06-data-layer ",
    "08-live-trading ",
    "10-simple-path ",
    "12-position-sizing ",
    "14-strategy-api ",
    "strategy-module ",
    "vectors: ",
)


def every_derivation_step_cites_a_rule(case: Case) -> Violations:
    """Rule: each derivation step cites at least one rule, by document and section."""
    problems = []
    for step in case["derivation"]:
        where = f"{case['id']} {step['slice_ist']}: derivation step {step['n']}"
        if not step["cites"]:
            problems.append(f"{where} cites no rule")
        for cite in step["cites"]:
            if not cite.startswith(CITABLE_SOURCES):
                problems.append(f"{where} cites {cite!r}, which names no known document")
    return problems


def file_is_plain_ascii(path: Path) -> Violations:
    """Rule: a file holds ASCII only (so no long dashes, typographic quotes or symbols)."""
    data = path.read_bytes()
    if data.isascii():
        return []
    position = next(index for index, byte in enumerate(data) if byte > 127)
    line = data.count(b"\n", 0, position) + 1
    return [f"{path.name} line {line}: the file contains a non-ASCII character"]
