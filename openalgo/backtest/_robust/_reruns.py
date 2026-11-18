"""The reruns of the bar checks: families of runs on generated bars, run in this process or
in worker processes on `bt.optimize`'s pool (`_sweep_pool`), each kept once.

A family (`Family`) says what its runs read and run: the run's bars as they are,
permuted (`_generate_permuted`) or noisy (`_generate_noisy`) from a stream, and the run's
strategy or random entries drawn from a stream (`_random_entries`); with run arguments
changed (`arguments`: a delay, a fill model, the dates of a window), strategy parameters
changed (`params`: a combination of a search), settings of the run's plan changed
(`plan`: strategy bars that end earlier), or each run's exits drawn (`exits`,
`_exit_draws`); and whether its rows keep the run's equity and trades (`detail`). Run
r of a family is a pure function of the run's bars, the family and r, so it is the same
in any process and for any number of workers, and two checks that ask for the same
family and run (versus random's strategy on permuted bars, which are the permutation
test's runs) share it.

Workers: `workers` processes (the CPU count by default), as `bt.optimize` starts them,
or this process when there is one run to do, when the strategy was made by
`bt.from_arrays` or `bt.from_signals`, or when the run's arguments cannot be sent to
another process (a calendar or cost model of its own); `Workers.why` says which. The
runs of a family are split into one contiguous batch per worker and come back in order,
and so are the families of a search, one run each (`Reruns.firsts`). The pool starts
when the first family needs it and stops when the report is done; a runner of other
bars made by `Reruns.sibling` (the searching checks', `_search_setup`) shares it, so a
report never holds two pools.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .._arrays import made_by
from .._sweep_pool import START_METHOD, portable, split
from ..errors import DataError
from ._exit_draws import ExitDraws
from ._generate_noisy import Noise, noisy, ranges
from ._generate_permuted import permuted
from ._random_entries import EntryPlan, random_strategy, schedule
from ._rerun_one import Known, RerunRow, failed_row, measured_row, rerun
from ._run_bars import RunBars

REAL, PERMUTED, NOISY = "real", "permuted", "noisy"


@dataclass(frozen=True)
class Family:
    """What a family of reruns reads and runs (module docstring): `bars` REAL, PERMUTED or
    NOISY, generated from the stream `bars_seed` with `noise`; the run's strategy, or
    random entries from the stream `entries_seed` when it is not None; `arguments`,
    `params` and `plan`, (name, value) pairs of run arguments, strategy parameters and
    plan settings every run changes; `exits`, what each run's exits are drawn from (None
    for none); `detail`, whether its rows keep the run's equity and trades
    (`_rerun_one.RunDetail`)."""

    bars: str
    bars_seed: int = 0
    noise: Noise | None = None
    entries_seed: int | None = None
    arguments: tuple[tuple[str, Any], ...] = ()
    plan: tuple[tuple[str, Any], ...] = ()
    exits: ExitDraws | None = None
    params: tuple[tuple[str, Any], ...] = ()
    detail: bool = False

    def changes(self, run: int, arguments: dict[str, Any]) -> tuple[dict[str, Any], dict]:
        """(the run arguments, the plan settings) of run `run`, from the run's own
        `arguments`."""
        changed = {**arguments, **dict(self.arguments)}
        if self.params:
            changed["params"] = {**(changed.get("params") or {}), **dict(self.params)}
        plan = dict(self.plan)
        if self.exits is not None:
            params, scales = self.exits.draw(run)
            changed["params"] = {**(changed.get("params") or {}), **params}
            plan["exit_scales"] = scales
        return changed, plan

    def frames(self, source: RunBars, run: int, atrs: Sequence[np.ndarray] | None) -> list:
        """The bars of run `run` (canonical layout, by symbol in registration order).
        Raises DataError when the run's instruments cannot be permuted together."""
        if self.bars == PERMUTED:
            first = source.first_run_ns
            return permuted(source.frames, source.symbols, source.ticks, first, self.bars_seed, run)
        if self.bars == NOISY:
            noise = self.noise or Noise()
            return noisy(source.frames, source.ticks, noise, self.bars_seed, run, atrs)
        return list(source.frames)


@dataclass(frozen=True)
class Later:
    """Runs whose family a check knows only after a search has run (the combination a
    fold chose, `check`'s `what`): the plan counts them, but they share no run and time
    no estimate."""

    check: str
    what: str


@dataclass(frozen=True, eq=False)
class Setup:
    """What every rerun of a report shares, all of it able to travel to a worker: the
    strategy and the objective (`_sweep_pool.portable` forms), the reruns' arguments, the
    run's bars, the indicator checks known, the run's session dates and the fan's steps
    of them, and what random entries are drawn from (None when no family draws them)."""

    strategy: tuple[str, Any]
    objective: tuple[str, Any]
    arguments: dict[str, Any]
    bars: RunBars
    known: Known
    dates: pd.DatetimeIndex
    steps: np.ndarray
    entries: EntryPlan | None


def _restore(shipped: tuple[str, Any]) -> Any:
    kind, value = shipped
    return pickle.loads(value) if kind == "cloudpickle" else value


def run_batch(setup: Setup, family: Family, runs: Sequence[int]) -> list[RerunRow]:
    """The rows of runs `runs` of `family`, in order (module docstring); a worker's entry
    point. Raises what a rerun raises, but for a DataError, which is kept as the run's
    failure: its bars could not be made (a generator's requirement), or the data checks
    refused them."""
    strategy, objective = _restore(setup.strategy), _restore(setup.objective)
    atrs = None
    if family.bars == NOISY:
        period = (family.noise or Noise()).period
        atrs = [ranges(frame, period) for frame in setup.bars.frames]
    rows = []
    for run in runs:
        try:
            frames = family.frames(setup.bars, run, atrs)
        except DataError as error:
            rows.append(failed_row(f"its bars could not be made: {error}", len(setup.steps)))
            continue
        chosen = strategy
        if family.entries_seed is not None and setup.entries is not None:
            chosen = random_strategy(strategy, schedule(setup.entries, family.entries_seed, run))
        arguments, plan = family.changes(run, setup.arguments)
        try:
            result, _ = rerun(chosen, arguments, setup.bars.symbols, frames, setup.known, plan=plan)
        except DataError as error:
            rows.append(failed_row(f"the data checks refused its bars: {error}", len(setup.steps)))
            continue
        rows.append(measured_row(result, objective, setup.dates, setup.steps, family.detail))
    return rows


def _batch(job: tuple[Setup, Family, tuple[int, ...]]) -> list[RerunRow]:
    return run_batch(*job)


def _firsts_batch(job: tuple[Setup, tuple[Family, ...]]) -> list[RerunRow]:
    """Run 0 of each family of a batch, in order; a worker's entry point."""
    setup, families = job
    return [row for family in families for row in run_batch(setup, family, [0])]


@dataclass(frozen=True)
class Workers:
    """How many processes run the reruns, and why this process alone does ("" when the
    count is the choice)."""

    count: int
    why: str


def choose_workers(requested: int | None, total: int, strategy: type, arguments: dict) -> Workers:
    """The worker processes for `total` reruns (module docstring)."""
    count = min(requested or os.cpu_count() or 1, max(total, 1))
    if count == 1:
        return Workers(1, "")
    maker = made_by(strategy)
    if maker is not None:
        return Workers(1, f"a strategy made by {maker} is built in this process")
    try:
        pickle.dumps(arguments)
    except (pickle.PicklingError, TypeError, AttributeError):
        return Workers(1, "the run's own calendar or cost model cannot be sent to workers")
    return Workers(count, "")


def shipped(target: Any, what: str, workers: Workers) -> tuple[str, Any]:
    """How `target` (the strategy, the objective) reaches the reruns: by reference in this
    process, and as `_sweep_pool.portable` sends it to worker processes. Raises
    ConfigError, as `bt.optimize` does, for one that workers cannot import."""
    return portable(target, what) if workers.count > 1 else ("ref", target)


class Reruns:
    """The reruns of one report (module docstring): `rows(family, runs)` gives the rows of
    those runs of `family`, running the ones not run yet. Use it as a context manager so
    the worker processes stop when the report is done."""

    def __init__(self, setup: Setup, workers: Workers) -> None:
        self._setup = setup
        self._workers = workers
        self._kept: dict[tuple[Family, int], RerunRow] = {}
        # One slot holding the pool, shared with every sibling (module docstring).
        self._pool: list[ProcessPoolExecutor | None] = [None]

    def sibling(self, setup: Setup) -> Reruns:
        """A runner of the reruns of `setup` (other bars, other arguments) that shares
        this runner's worker processes, and stops them when this one closes."""
        other = Reruns(setup, self._workers)
        other._pool = self._pool
        return other

    @property
    def setup(self) -> Setup:
        """What every rerun shares."""
        return self._setup

    @property
    def done(self) -> int:
        """The reruns run so far."""
        return len(self._kept)

    def rows(self, family: Family, runs: Sequence[int]) -> list[RerunRow]:
        """The rows of `runs` of `family`, in order."""
        missing = [run for run in runs if (family, run) not in self._kept]
        if missing:
            for run, row in zip(missing, self._run(family, missing), strict=True):
                self._kept[(family, run)] = row
        return [self._kept[(family, run)] for run in runs]

    def firsts(self, families: Sequence[Family]) -> list[RerunRow]:
        """The rows of run 0 of each of `families`, in order: the combinations of a
        search, the families not run yet split into one batch per worker."""
        missing = list(dict.fromkeys(f for f in families if (f, 0) not in self._kept))
        if missing:
            for family, row in zip(missing, self._run_firsts(missing), strict=True):
                self._kept[(family, 0)] = row
        return [self._kept[(family, 0)] for family in families]

    def _run_firsts(self, families: list[Family]) -> list[RerunRow]:
        count = min(self._workers.count, len(families))
        if count <= 1:
            return _firsts_batch((self._setup, tuple(families)))
        jobs = [(self._setup, tuple(part)) for part in split(families, count)]
        return [row for rows in self._executor().map(_firsts_batch, jobs) for row in rows]

    def _run(self, family: Family, runs: list[int]) -> list[RerunRow]:
        count = min(self._workers.count, len(runs))
        if count <= 1:
            return run_batch(self._setup, family, runs)
        jobs = [(self._setup, family, tuple(part)) for part in split(runs, count)]
        return [row for rows in self._executor().map(_batch, jobs) for row in rows]

    def _executor(self) -> ProcessPoolExecutor:
        if self._pool[0] is None:
            context = multiprocessing.get_context(START_METHOD)
            self._pool[0] = ProcessPoolExecutor(max_workers=self._workers.count, mp_context=context)
        return self._pool[0]

    def close(self) -> None:
        """Stop the worker processes, if any started (a sibling's included)."""
        if self._pool[0] is not None:
            self._pool[0].shutdown()
            self._pool[0] = None

    def __enter__(self) -> Reruns:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = [
    "NOISY",
    "PERMUTED",
    "REAL",
    "Family",
    "Later",
    "Reruns",
    "Setup",
    "Workers",
    "choose_workers",
    "run_batch",
    "shipped",
]
