"""`Result.rerun(**overrides)`: the same backtest again, with some arguments changed.

A rerun calls `bt.backtest` with the strategy class, the client and every argument of
the original call, then applies the overrides: a name of a `bt.backtest` argument
replaces that argument (`quantity_mode="venue"`), a name of a strategy parameter
replaces that parameter (`fast=20`). With no overrides the result is the original one
bit for bit, because the kernel is deterministic and the data are the same; if the data
the source returns have changed since (other partition hashes in the manifest), the
rerun's warnings say so first.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from .errors import ConfigError

# Arguments that decide which bars a run reads.
DATA_ARGUMENTS = ("start", "end", "data", "source", "tz", "base", "calendar")


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
        params = dict(arguments.get("params") or {})
        changes = dict(overrides)
        client = changes.pop("client", self.client)
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
        if same_data and result.manifest["data"] != manifest["data"]:
            result._warnings.insert(
                0,
                "The data differ from those of the original run (other partition hashes): "
                "this rerun is not a reproduction of it",
            )
        return result


__all__ = ["RunRecord"]
