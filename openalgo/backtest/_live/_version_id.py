"""A strategy's version id (18-multi-strategy 12.1): which code and which parameters placed
an order, kept on every intent of the journal and in each start's manifest.

The id is the first `VERSION_ID_LENGTH` hexadecimal characters of the SHA-256 of a
canonical JSON text (keys sorted, no spaces, ASCII; `_version_values.canonical_json`) of
the strategy's source hash (`_manifest.strategy_source_hash`: its class's source and its
own base classes') and its parameter values (the class's parameters with those given over
them), so the same code with the same parameters gives the same id on every day and every
restart, and a change to either gives a new one. A class whose source cannot be read (one
defined in a notebook cell or built at run time) hashes its compiled code in place of the
source hash (`_version_code.compiled_code`, which depends on the Python version, as the
start banner says); one whose code cannot be read at all (every callable it holds is a
builtin or an object called as a method, none a Python function) hashes its qualified
name, and the banner says its source was not hashed. The id is the trader's own record of
what placed an order; the exchange's algo identifier is the broker's to attach.

The canonical text is the same in every process (`_version_values.canonical_value`): a
set's items are sorted (a set's own order depends on the process's hash seed), and an
object that has no text of its value (its text names its memory address, which differs
from run to run) is hashed by that text without the address, its type alone; the start
banner names such a parameter, and such a value of a class hashed by its compiled code,
where a value Python cannot hash (one the strategy may change as it runs, such as a list)
is also known by its type alone (`_version_code`), since a change to it does not change the
id. The banner's lines and each start's record of the versions are `StartVersions`
(written by `_session_versions`).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._manifest import plain, strategy_source_hash
from ._version_code import compiled_code, python_version
from ._version_values import canonical_json, canonical_value

#: How many hexadecimal characters of the SHA-256 a version id keeps: 48 bits, enough
#: that two versions of one strategy never share one in practice, short enough to read.
VERSION_ID_LENGTH = 12


@dataclass(frozen=True)
class StrategyVersion:
    """A strategy's version (module docstring): its `version_id`, whether its source was
    hashed (`source_hashed`: False for a class built at run time), the values hashed by
    their type alone (`by_type`: objects with no fixed text of their value; its
    parameters, and when its compiled code was hashed its class attributes, default
    arguments and callables, as `_version_code` names them), and the Python its compiled
    code was hashed under when its source was not (`compiled_under`, such as "CPython
    3.12"; None when its source was hashed or its code could not be read)."""

    version_id: str
    source_hashed: bool
    by_type: tuple[str, ...] = ()
    compiled_under: str | None = None


def parameter_values(strategy: type, params: Mapping[str, Any] | None) -> dict[str, Any]:
    """The strategy's parameter values: its class's (`Strategy.parameters`) with `params`
    over them, as JSON types."""
    return plain(strategy.parameters() | dict(params or {}))


def version_of(strategy: type, params: Mapping[str, Any] | None) -> StrategyVersion:
    """The version of `strategy` run with `params` (module docstring)."""
    source = strategy_source_hash(strategy)
    compiled = None if source is not None else compiled_code(strategy)
    values = strategy.parameters() | dict(params or {})
    canonical = {str(name): canonical_value(value) for name, value in values.items()}
    code = strategy.__qualname__ if compiled is None else compiled.code_hash
    hashed = {
        "parameters": {name: text for name, (text, _) in canonical.items()},
        "source": source or code,
    }
    digest = hashlib.sha256(canonical_json(hashed).encode("utf-8")).hexdigest()
    by_type = [name for name, (_, typed) in canonical.items() if typed]
    under = None
    if compiled is not None:
        by_type += compiled.by_type
        under = python_version()
    version = digest[:VERSION_ID_LENGTH]
    return StrategyVersion(version, source is not None, tuple(dict.fromkeys(by_type)), under)


@dataclass(frozen=True)
class Started:
    """One strategy's version at this start: its name, `version_id`, whether its source was
    hashed, its run `manifest` (JSON types), the version id of the day's latest earlier
    start of it (`earlier`, None at its first start of the day), the values hashed by
    their type alone (`by_type`) and the Python its compiled code was hashed under
    (`compiled_under`, as `StrategyVersion`'s)."""

    name: str
    version_id: str
    source_hashed: bool
    manifest: dict[str, Any]
    earlier: str | None = None
    by_type: tuple[str, ...] = ()
    compiled_under: str | None = None

    @property
    def changed(self) -> bool:
        """True when an earlier start of the day ran the strategy at another version."""
        return self.earlier is not None and self.earlier != self.version_id


@dataclass(frozen=True)
class StartVersions:
    """Every strategy's version at this start, in call order (`started`); none before the
    start records them (`_session_versions.record_versions`)."""

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
        """The start banner's lines (`_session_versions`): the version ids, then each
        strategy whose version changed since the day's earlier start, each whose source
        was not hashed (and what was hashed instead), and each with values hashed by their
        type alone (`by_type`)."""
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
        lines += [_not_hashed_line(one) for one in self.started if not one.source_hashed]
        lines += [
            f"{one.name}: {', '.join(one.by_type)} has no fixed text of its value (an object "
            "shown by its memory address, or one that can change as the strategy runs, such as "
            "a list, a dict or a DataFrame): its version id hashes its type only, so a change to "
            "its value does not change the id"
            for one in self.started
            if one.by_type
        ]
        return lines


def _not_hashed_line(one: Started) -> str:
    """The banner's line of a strategy whose source was not hashed: what its version id
    hashes instead, its compiled code or, when that could not be read either, its name."""
    if one.compiled_under is None:
        return (
            f"{one.name}: its source was not hashed (a class built at run time): its version "
            "id hashes its name and parameters only"
        )
    return (
        f"{one.name}: its source could not be read (a class built at run time): its version "
        f"id hashes its compiled code (its functions and class attributes) under "
        f"{one.compiled_under} and its parameters, so the same code under another Python "
        "version gets another id"
    )


__all__ = [
    "VERSION_ID_LENGTH",
    "StartVersions",
    "Started",
    "StrategyVersion",
    "parameter_values",
    "version_of",
]
