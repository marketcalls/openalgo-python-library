"""The worker processes of `bt.optimize` (03-architecture 5).

Process pools use the spawn start method, the only one Windows has, everywhere: a
worker starts a fresh interpreter and imports what it runs. The strategy class (and an
objective function) therefore reach a worker by module and name, so they must be
importable from a module. A class defined in a notebook, an interactive session or
inside a function is not; it is sent by value with cloudpickle when cloudpickle is
already installed, and otherwise the sweep stops with a ConfigError that says what to
do. Nothing is installed or imported speculatively.

The combinations are split into one contiguous chunk per worker, in grid order, and the
rows come back in that order, so a sweep's table does not depend on the number of
workers.
"""

from __future__ import annotations

import importlib.util
import multiprocessing
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

from ._causality_checks import IndicatorChecks
from ._objective import evaluate
from ._pipeline import PreparedRun, execute
from .errors import ConfigError

START_METHOD = "spawn"


def _importable(target: Any) -> bool:
    """True when a worker can import `target` by its module and qualified name."""
    module = sys.modules.get(getattr(target, "__module__", ""))
    qualname = getattr(target, "__qualname__", "")
    if module is None or "<locals>" in qualname:
        return False
    if module.__name__ == "__main__" and not getattr(module, "__file__", None):
        return False
    found: Any = module
    for part in qualname.split("."):
        found = getattr(found, part, None)
    return found is target


def portable(target: Any, what: str) -> tuple[str, Any]:
    """How `target` travels to a worker: ("ref", target) when importable, or
    ("cloudpickle", bytes) when cloudpickle is installed. Raises ConfigError otherwise."""
    if not callable(target) or _importable(target):
        return ("ref", target)
    if importlib.util.find_spec("cloudpickle") is None:
        name = getattr(target, "__qualname__", repr(target))
        raise ConfigError(
            f"{what} {name} is defined in a notebook, an interactive session or a function, "
            "so worker processes cannot import it. Move it into a module (a .py file) and "
            "import it from there, install cloudpickle, or pass workers=1"
        )
    import cloudpickle

    return ("cloudpickle", cloudpickle.dumps(target))


def _restore(shipped: tuple[str, Any]) -> Any:
    kind, value = shipped
    return pickle.loads(value) if kind == "cloudpickle" else value


@dataclass(frozen=True)
class Chunk:
    """One worker's share of a sweep: everything it needs, all picklable."""

    strategy: tuple[str, Any]
    objective: tuple[str, Any]
    prepared: PreparedRun
    arguments: dict[str, Any]
    combinations: list[dict[str, Any]]
    strategy_hash: str
    checked: frozenset[str]


def run_combinations(
    strategy: type,
    objective: Any,
    prepared: PreparedRun,
    arguments: dict[str, Any],
    combinations: list[dict[str, Any]],
    checks: IndicatorChecks,
) -> list[dict[str, float]]:
    """The objective and key metrics of each combination, in order."""
    rows = []
    for params in combinations:
        result = execute(strategy, prepared, {**arguments, "params": params}, checks=checks)
        rows.append(evaluate(result, prepared.plan.exchange, objective))
    return rows


def run_chunk(chunk: Chunk) -> list[dict[str, float]]:
    """A worker's entry point: run one chunk."""
    checks = IndicatorChecks(chunk.strategy_hash, chunk.checked)
    return run_combinations(
        _restore(chunk.strategy),
        _restore(chunk.objective),
        chunk.prepared,
        chunk.arguments,
        chunk.combinations,
        checks,
    )


def split(combinations: list[dict[str, Any]], parts: int) -> list[list[dict[str, Any]]]:
    """Contiguous chunks of near-equal size, in order."""
    size, extra = divmod(len(combinations), parts)
    chunks, start = [], 0
    for index in range(parts):
        stop = start + size + (1 if index < extra else 0)
        chunks.append(combinations[start:stop])
        start = stop
    return [chunk for chunk in chunks if chunk]


def run_in_pool(chunks: list[Chunk], workers: int) -> list[dict[str, float]]:
    """Run the chunks in `workers` spawned processes; the rows in chunk order."""
    context = multiprocessing.get_context(START_METHOD)
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        return [row for rows in pool.map(run_chunk, chunks) for row in rows]


__all__ = ["Chunk", "portable", "run_combinations", "run_in_pool", "split"]
