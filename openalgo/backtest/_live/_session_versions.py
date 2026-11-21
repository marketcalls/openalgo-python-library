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
is allowed: the start banner says so and the event `version_changed` records both ids, so
the record shows the trader changed the code or the parameters. The banner names every
strategy's version id, and each strategy whose source could not be hashed (a class built
at run time). Times are UTC nanoseconds.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._manifest import build_manifest
from .._pipeline_result import manifest_arguments
from ..reference import default_tables
from ._plan_catch_up import StrategyPlan
from ._result_build import prepared_run
from ._version_id import parameter_values

#: The event of a strategy whose version differs from the day's earlier start's.
VERSION_CHANGED = "version_changed"


@dataclass(frozen=True)
class Started:
    """One strategy's version at this start: its name, `version_id`, whether its source was
    hashed, its run `manifest` (JSON types) and the version id of the day's latest earlier
    start of it (`earlier`, None at its first start of the day)."""

    name: str
    version_id: str
    source_hashed: bool
    manifest: dict[str, Any]
    earlier: str | None = None

    @property
    def changed(self) -> bool:
        """True when an earlier start of the day ran the strategy at another version."""
        return self.earlier is not None and self.earlier != self.version_id


@dataclass(frozen=True)
class StartVersions:
    """Every strategy's version at this start, in call order (`started`)."""

    started: tuple[Started, ...] = ()

    @property
    def manifests(self) -> dict[str, dict[str, Any]]:
        """Each strategy's run manifest of this start, by name in call order."""
        return {one.name: one.manifest for one in self.started}

    @property
    def ids(self) -> dict[str, str]:
        """Each strategy's version id, by name in call order."""
        return {one.name: one.version_id for one in self.started}

    def banner_lines(self) -> list[str]:
        """The start banner's lines (module docstring): the version ids, then each
        strategy whose version changed since the day's earlier start, then each whose source
        was not hashed."""
        if not self.started:
            return []
        ids = ", ".join(f"{one.name} {one.version_id}" for one in self.started)
        lines = [f"Versions: {ids}"]
        lines += [
            f"{one.name}: its version changed since the day's earlier start, from "
            f"{one.earlier} to {one.version_id} (its code or its parameters)"
            for one in self.started
            if one.changed
        ]
        lines += [
            f"{one.name}: its source was not hashed (a class built at run time): its version "
            "id hashes its name and parameters only"
            for one in self.started
            if not one.source_hashed
        ]
        return lines


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
        one = Started(
            spec.name,
            spec.version.version_id,
            spec.version.source_hashed,
            start_manifest(part),
            earlier.get(spec.name),
        )
        journal.record_version(one.name, one.version_id, one.manifest, day=day)
        if one.changed:
            changed = {"strategy": one.name, "earlier": one.earlier, "now": one.version_id}
            session.event(VERSION_CHANGED, **changed)
        started.append(one)
    return StartVersions(tuple(started))


def _latest(rows: Any) -> Mapping[str, str]:
    """Each strategy's version id at the latest start of the journal's `rows` (oldest
    first)."""
    return {row.strategy: row.version_id for row in rows}


__all__ = ["VERSION_CHANGED", "StartVersions", "Started", "record_versions", "start_manifest"]
