"""A strategy's version id (18-multi-strategy 12.1): which code and which parameters placed
an order, kept on every intent of the journal and in each start's manifest.

The id is the first `VERSION_ID_LENGTH` hexadecimal characters of the SHA-256 of a
canonical JSON text (keys sorted, no spaces, ASCII) of the strategy's source hash
(`_manifest.strategy_source_hash`: its class's source and its own base classes') and its
parameter values (the class's parameters with those given over them), so the same code
with the same parameters gives the same id on every day and every restart, and a change to
either gives a new one. A class with no source (one built at run time) hashes its
qualified name in place of the source hash; the start banner says its source was not
hashed. The id is the trader's own record of what placed an order; the exchange's algo
identifier is the broker's to attach.

The canonical text is the same in every process: a set's items are sorted (a set's own
order depends on the process's hash seed), and an object that has no text of its value
(its text names its memory address, which differs from run to run) is hashed by that text
without the address, its type alone; the start banner names such a parameter, since a
change to its value does not change the id. The banner's lines and each start's record of
the versions are `StartVersions` (written by `_session_versions`).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._manifest import plain, strategy_source_hash

#: How many hexadecimal characters of the SHA-256 a version id keeps: 48 bits, enough
#: that two versions of one strategy never share one in practice, short enough to read.
VERSION_ID_LENGTH = 12
#: The memory address in the text of an object with no text of its own ("<Model object
#: at 0x...>"): it differs from run to run, so the id leaves it out.
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


@dataclass(frozen=True)
class StrategyVersion:
    """A strategy's version (module docstring): its `version_id`, whether its source was
    hashed (`source_hashed`: False for a class built at run time), and the parameters
    hashed by their type alone (`by_type`: objects with no text of their value)."""

    version_id: str
    source_hashed: bool
    by_type: tuple[str, ...] = ()


def parameter_values(strategy: type, params: Mapping[str, Any] | None) -> dict[str, Any]:
    """The strategy's parameter values: its class's (`Strategy.parameters`) with `params`
    over them, as JSON types."""
    return plain(strategy.parameters() | dict(params or {}))


def version_of(strategy: type, params: Mapping[str, Any] | None) -> StrategyVersion:
    """The version of `strategy` run with `params` (module docstring)."""
    source = strategy_source_hash(strategy)
    values = strategy.parameters() | dict(params or {})
    canonical = {str(name): _canonical(value) for name, value in values.items()}
    hashed = {
        "parameters": {name: text for name, (text, _) in canonical.items()},
        "source": source or strategy.__qualname__,
    }
    digest = hashlib.sha256(_json(hashed).encode("utf-8")).hexdigest()
    by_type = tuple(name for name, (_, typed) in canonical.items() if typed)
    return StrategyVersion(digest[:VERSION_ID_LENGTH], source is not None, by_type)


def _canonical(value: Any) -> tuple[Any, bool]:
    """`value` as JSON types that are the same in every process (module docstring), and
    whether a part of it was known by its type alone."""
    if isinstance(value, Mapping):
        parts = {str(key): _canonical(item) for key, item in value.items()}
        return {key: text for key, (text, _) in parts.items()}, any(
            typed for _, typed in parts.values()
        )
    if isinstance(value, list | tuple | set | frozenset):
        items = [_canonical(item) for item in value]
        texts = [text for text, _ in items]
        if isinstance(value, set | frozenset):
            texts.sort(key=_json)
        return texts, any(typed for _, typed in items)
    text = plain(value)
    if isinstance(text, str) and not isinstance(value, str):
        bare = _ADDRESS.sub("", text)
        return bare, bare != text
    return text, False


def _json(value: Any) -> str:
    """The canonical JSON text of `value`: keys sorted, no spaces, ASCII."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class Started:
    """One strategy's version at this start: its name, `version_id`, whether its source was
    hashed, its run `manifest` (JSON types), the version id of the day's latest earlier
    start of it (`earlier`, None at its first start of the day) and the parameters hashed
    by their type alone (`by_type`)."""

    name: str
    version_id: str
    source_hashed: bool
    manifest: dict[str, Any]
    earlier: str | None = None
    by_type: tuple[str, ...] = ()

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
        was not hashed, and each with parameters hashed by their type alone."""
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
        lines += [
            f"{one.name}: {', '.join(one.by_type)} has no text of its value (an object shown "
            "by its memory address): its version id hashes its type only, so a change to its "
            "value does not change the id"
            for one in self.started
            if one.by_type
        ]
        return lines


__all__ = [
    "VERSION_ID_LENGTH",
    "StartVersions",
    "Started",
    "StrategyVersion",
    "parameter_values",
    "version_of",
]
