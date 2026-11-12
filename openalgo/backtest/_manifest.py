"""The run manifest: what is needed to reproduce a result or to tell that it can no
longer be reproduced (14-strategy-api 6).

It records the engine version, the strategy (module, name, a hash of its source and its
parameter values), every argument after defaults were resolved, the instruments in
registration order (their terms as used), the data partition hashes of every
instrument, the reference-table and cost-schedule versions, the random seed (always
None: nothing in a backtest is random) and the platform. No API key or other secret is
ever written: a client is recorded only as having been given.

`RunManifest.to_dict()` gives plain JSON types and `from_dict` reads them back, so a
manifest written with `json.dumps` reads back equal.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import inspect
import json
import platform
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._native import native

# Raised by `inspect.getsource` for a class whose source is not in a file.
_NO_SOURCE = (OSError, TypeError)


def strategy_source_hash(strategy: type) -> str | None:
    """SHA-256 of the source of a strategy class and of its own base classes up to
    `bt.Strategy`, or None when a source is not available (a class built at run time)."""
    from .strategy import Strategy

    hasher = hashlib.sha256()
    for klass in strategy.__mro__:
        if klass is Strategy:
            break
        try:
            hasher.update(inspect.getsource(klass).encode("utf-8"))
        except _NO_SOURCE:
            return None
    return f"sha256:{hasher.hexdigest()}"


def _plain(value: Any) -> Any:
    """A JSON-ready copy of a value: dates as ISO strings, tuples as lists."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return repr(value)


def _platform() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


@dataclass(frozen=True)
class RunManifest:
    """One run's manifest; see the module docstring. Every field holds JSON types."""

    engine_version: str
    strategy: dict[str, Any]
    arguments: dict[str, Any]
    instruments: list[dict[str, Any]]
    data: dict[str, dict[str, str]]
    reference_tables: str | None
    cost_schedules: list[str]
    random_seed: None
    platform: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """The manifest as a dict of JSON types."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> RunManifest:
        """Read a manifest written by `to_dict`. Raises KeyError for a missing field."""
        return cls(**{field.name: values[field.name] for field in dataclasses.fields(cls)})

    def to_json(self) -> str:
        """The manifest as JSON text, keys sorted."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, text: str) -> RunManifest:
        """Read a manifest written by `to_json`."""
        return cls.from_dict(json.loads(text))


def build_manifest(
    strategy: type,
    parameters: dict[str, Any],
    arguments: dict[str, Any],
    instruments: list[dict[str, Any]],
    data: dict[str, dict[str, str]],
    reference_tables: str | None,
    cost_schedules: list[str],
) -> RunManifest:
    """The manifest of a run of `strategy` with resolved `parameters` and `arguments`,
    its `instruments` as `Instrument.to_dict()` (anything not a JSON type is written as
    its repr)."""
    described = {
        "module": strategy.__module__,
        "name": strategy.__qualname__,
        "source_hash": strategy_source_hash(strategy),
        "parameters": _plain(parameters),
    }
    return RunManifest(
        engine_version=native.engine_version(),
        strategy=described,
        arguments=_plain(arguments),
        instruments=_plain(instruments),
        data=_plain(data),
        reference_tables=reference_tables,
        cost_schedules=cost_schedules,
        random_seed=None,
        platform=_platform(),
    )


__all__ = ["RunManifest", "build_manifest", "strategy_source_hash"]
