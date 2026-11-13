"""`Result.rerun(**overrides)`: the same backtest again, with some arguments changed.

A rerun calls `bt.backtest` with the strategy class, the client and every argument of
the original call, then applies the overrides: a name of a `bt.backtest` argument
replaces that argument (`quantity_mode="fractional"`), a name of a strategy parameter
replaces that parameter (`fast=20`). With no overrides the result is the original one
bit for bit, because the kernel is deterministic and the data are the same; if the data
the source returns have changed since (other partition hashes in the manifest), the
rerun's warnings say so first, and likewise when only the benchmark's closes changed.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from ._benchmark import benchmark_partitions
from .errors import ConfigError

# Arguments that decide which bars a run reads.
DATA_ARGUMENTS = ("start", "end", "data", "source", "tz", "base", "calendar", "now")


@dataclass(frozen=True)
class RunRecord:
    """What a result needs to run again: the strategy class, the client and the keyword
    arguments of the original `bt.backtest` call."""

    strategy: type
    client: Any
    arguments: dict[str, Any]

    def rerun(self, overrides: dict[str, Any], manifest: dict[str, Any]) -> Any:
        """Run again with `overrides` (see the module docstring). Raises ConfigError for
        a name that is neither an argument nor a parameter, and whatever the run
        raises."""
        from .run import backtest

        names = set(inspect.signature(backtest).parameters) - {"strategy", "client"}
        parameters = self.strategy.parameters()
        arguments = {name: self.arguments[name] for name in names}
        changes = dict(overrides)
        client = changes.pop("client", self.client)
        # `params=` replaces the original mapping; a parameter by name changes one entry.
        params = dict(changes.pop("params", arguments.get("params")) or {})
        for name, value in changes.items():
            if name in names:
                arguments[name] = value
            elif name in parameters:
                params[name] = value
            else:
                raise ConfigError(
                    f"rerun: {name!r} is neither an argument of bt.backtest nor a parameter "
                    f"of {self.strategy.__name__} ({sorted(parameters)})"
                )
        arguments["params"] = params or None
        result = backtest(self.strategy, client, **arguments)
        same_data = not any(name in overrides for name in (*DATA_ARGUMENTS, "client"))
        other_closes = benchmark_partitions(result.manifest) != benchmark_partitions(manifest)
        if same_data and "benchmark" not in overrides and other_closes:
            result._warnings.insert(
                0,
                "The benchmark's data differ from those of the original run (other partition "
                "hashes): its benchmark figures are not a reproduction of it",
            )
        if same_data and result.manifest["data"] != manifest["data"]:
            result._warnings.insert(
                0,
                "The data differ from those of the original run (other partition hashes): "
                "this rerun is not a reproduction of it",
            )
        return result


__all__ = ["RunRecord"]
