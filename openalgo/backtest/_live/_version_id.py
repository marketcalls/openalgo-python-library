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
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._manifest import plain, strategy_source_hash

#: How many hexadecimal characters of the SHA-256 a version id keeps: 48 bits, enough
#: that two versions of one strategy never share one in practice, short enough to read.
VERSION_ID_LENGTH = 12


@dataclass(frozen=True)
class StrategyVersion:
    """A strategy's version (module docstring): its `version_id`, and whether its source
    was hashed (`source_hashed`: False for a class built at run time)."""

    version_id: str
    source_hashed: bool


def parameter_values(strategy: type, params: Mapping[str, Any] | None) -> dict[str, Any]:
    """The strategy's parameter values: its class's (`Strategy.parameters`) with `params`
    over them, as JSON types."""
    return plain(strategy.parameters() | dict(params or {}))


def version_of(strategy: type, params: Mapping[str, Any] | None) -> StrategyVersion:
    """The version of `strategy` run with `params` (module docstring)."""
    source = strategy_source_hash(strategy)
    hashed = {
        "parameters": parameter_values(strategy, params),
        "source": source or strategy.__qualname__,
    }
    canonical = json.dumps(
        hashed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return StrategyVersion(digest[:VERSION_ID_LENGTH], source is not None)


__all__ = ["VERSION_ID_LENGTH", "StrategyVersion", "parameter_values", "version_of"]
