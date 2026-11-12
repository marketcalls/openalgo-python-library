"""Consistency rules between the fixtures and the specification review.

`spec_review.json` lists every point the documents do not settle. An expectation that
depends on such a point carries `depends_on` with the item's id, and the case lists every
id any of its expectations (or its inputs) depends on. These rules keep the two in step.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from typing import Any

from . import vocabulary as vocab
from .schema import each, member_violation
from .timegrid import Case

Violations = list[str]
Review = dict[str, Any]

TEXT_FIELDS = ("title", "question", "recommended", "reason", "proposed_wording")


def review_items_are_complete(review: Review) -> Violations:
    """Rule: every review item has all fields, a known kind and severity, at least two
    readings, recommends one of its readings, and, once resolved, names the reading
    decided."""
    name = "spec_review.json"
    problems = member_violation(review.get("format"), (vocab.SPEC_REVIEW_FORMAT,), "format", name)
    problems += each(review["items"], "review_item", f"{name} items")
    if problems:
        return problems
    identifiers = [item["id"] for item in review["items"]]
    if len(set(identifiers)) != len(identifiers):
        problems.append(f"{name}: item ids are not unique")
    for item in review["items"]:
        where = f"{name} {item['id']}"
        problems += member_violation(item["kind"], vocab.REVIEW_KINDS, "kind", where)
        problems += member_violation(item["severity"], vocab.REVIEW_SEVERITIES, "severity", where)
        for field in TEXT_FIELDS:
            if not str(item[field]).strip():
                problems.append(f"{where}: {field} is empty")
        if not item["documents"] or len(item["readings"]) < 2:
            problems.append(f"{where}: needs the documents involved and at least two readings")
        if not any(reading.startswith(f"{item['recommended']}.") for reading in item["readings"]):
            problems.append(f"{where}: recommended must be the letter of one of the readings")
        problems += _decision_violations(item, where)
    return problems


def _decision_violations(item: dict[str, Any], where: str) -> Violations:
    """A resolved item names one of its readings and the ISO date it was decided on; an
    open item names neither."""
    problems = member_violation(item["status"], vocab.REVIEW_STATUSES, "status", where)
    letters = {reading.split(".", 1)[0] for reading in item["readings"]}
    if item["status"] == "resolved":
        if item["decision"] not in letters:
            problems.append(f"{where}: decision must be the letter of one of the readings")
        if not _is_iso_date(item["decided_on"]):
            problems.append(f"{where}: decided_on must be an ISO date (YYYY-MM-DD)")
    elif item["decision"] or item["decided_on"]:
        problems.append(f"{where}: an open item has no decision and no decided_on")
    return problems


def _is_iso_date(text: str) -> bool:
    try:
        date.fromisoformat(text)
    except (TypeError, ValueError):
        return False
    return len(text) == 10


def _tags(node: Any) -> Iterator[str]:
    """Every depends_on id found anywhere inside an object."""
    if isinstance(node, dict):
        yield from node.get(vocab.DEPENDS_ON, [])
        for value in node.values():
            yield from _tags(value)
    elif isinstance(node, list):
        for value in node:
            yield from _tags(value)


def case_dependencies_are_registered(case: Case, review: Review) -> Violations:
    """Rule: every id in a case's depends_on is an item of the specification review."""
    registered = {item["id"] for item in review["items"]}
    return [
        f"{case['id']}: depends_on names {identifier}, which is not in spec_review.json"
        for identifier in case["depends_on"]
        if identifier not in registered
    ]


def expectation_dependencies_are_declared_on_the_case(case: Case) -> Violations:
    """Rule: an id tagged on an expectation is also listed in the case's depends_on."""
    declared = set(case["depends_on"])
    return [
        f"{case['id']}: an expectation depends on {identifier}, which the case does not list"
        for identifier in sorted(set(_tags(case["expected"])) - declared)
    ]


def review_lists_the_cases_that_depend_on_each_item(
    review: Review, cases: list[Case]
) -> Violations:
    """Rule: the cases named by a review item are exactly the cases that depend on it."""
    problems = []
    for item in review["items"]:
        using = sorted(case["id"] for case in cases if item["id"] in case["depends_on"])
        if sorted(item["cases"]) != using:
            problems.append(
                f"spec_review.json {item['id']}: lists {sorted(item['cases'])}, "
                f"the fixtures that depend on it are {using}"
            )
    return problems
