"""The strategies' versions at a live session's start (18-multi-strategy 12.1; unit M11): the
trader's own record of which code and which parameters placed each order.

At each start, for each strategy in call order, the session builds the run manifest a
backtest of the day builds (`_manifest.build_manifest`: the engine version, the strategy's
module, name, source hash and parameter values, every resolved argument, the instruments,
the warm-up history's partition hashes, the reference table and cost schedule versions; a
client is recorded only as given, never its API key) and writes it with the strategy's
version id to journal L (`Journal.record_version`). Every intent the strategy's venue map
records carries the id (`_venue_requests.intent_fields`); the `strategy` field OpenAlgo
receives stays the plain class name, which the order book's attribution, the reconciler
and the restart match orders on.

A restart whose version id of a strategy differs from the day's latest earlier start of it
is allowed: the start banner says so, naming each parameter whose value differs between
the two starts' run manifests (`_version_id.moved_parameters`; not one whose text alone
differs, a set's or an object's known by its memory address), and the event
`version_changed` records both ids, so the record shows the trader changed the code or the
parameters (or the strategy rebound a value on its class as it ran). The banner names every
strategy's version id, each strategy whose source could not be hashed (a class built at
run time) with what was hashed instead (its compiled code under the Python version, or its
name when its code could not be read at all), and each value hashed by its type alone (a
parameter, or a class attribute, default argument or callable of a class hashed by its
compiled code; `_version_id`), with the record of the start's versions
(`_version_id.StartVersions`).
Times are UTC nanoseconds.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .._manifest import build_manifest
from .._pipeline_result import manifest_arguments
from ..reference import default_tables
from ._plan_catch_up import StrategyPlan
from ._result_build import prepared_run
from ._version_id import (
    Started,
    StartVersions,
    given_parameters,
    moved_parameters,
    parameter_values,
)

#: The event of a strategy whose version differs from the day's earlier start's.
VERSION_CHANGED = "version_changed"


def start_manifest(part: StrategyPlan) -> dict[str, Any]:
    """The run manifest of one strategy's part of the session plan at its start (module
    docstring), as JSON types."""
    spec, setup = part.spec, part.setup
    prepared = prepared_run(part)
    arguments = manifest_arguments(prepared, dict(spec.arguments), setup.market)
    manifest = build_manifest(
        spec.strategy,
        parameter_values(spec.strategy, spec.params),
        arguments,
        [instrument.to_dict() for instrument in setup.instruments],
        setup.history.partitions(),
        default_tables().version,
        setup.market.cost_schedules,
    )
    return manifest.to_dict()


def record_versions(session: Any) -> StartVersions:
    """Write each strategy's version and run manifest of this start to `session`'s journal
    L for its day, in call order, and the event `VERSION_CHANGED` for each whose version
    differs from the day's latest earlier start of it (module docstring). Raises
    JournalError when the journal cannot write them."""
    parts = session.parts
    journal, day = parts.journal, parts.spec.day
    earlier = _latest(journal.versions(day=day))
    started = []
    for part in parts.plan.strategies:
        spec = part.spec
        manifest = start_manifest(part)
        before = earlier.get(spec.name)
        values = given_parameters(spec.strategy, spec.params)
        one = Started(
            spec.name,
            spec.version.version_id,
            spec.version.source_hashed,
            manifest,
            None if before is None else before.version_id,
            spec.version.by_type,
            spec.version.compiled_under,
            () if before is None else moved_parameters(before.manifest, manifest, values),
        )
        journal.record_version(one.name, one.version_id, one.manifest, day=day)
        if one.changed:
            changed = {"strategy": one.name, "earlier": one.earlier, "now": one.version_id}
            session.event(VERSION_CHANGED, **changed)
        started.append(one)
    return StartVersions(tuple(started))


def _latest(rows: Any) -> Mapping[str, Any]:
    """Each strategy's row (`StartVersion`: its version id and run manifest) at the latest
    start of the journal's `rows` (oldest first)."""
    return {row.strategy: row for row in rows}


__all__ = ["VERSION_CHANGED", "record_versions", "start_manifest"]
