"""What the checks that rerun the backtest plan (`planned`), and the plan's words.

What a report plans is what each check says it will rerun (its `reruns`, from the same
reasons as its run: a check not judged before any rerun, for too few trades, no edge,
decisions fixed in advance or an objective it cannot measure, reruns nothing): versus
random's runs of each family that can run, noise and permutation theirs, stress its
candidates, other markets one a symbol, the searching checks a run of each combination
(and each window) and the runs of their choices, which are known only once their search
has run (`_reruns.Later`); less the runs two checks share: a family is rerun once up to
the most runs any check asks of it (`_bar_common`: the permutation test's permuted paths,
which versus random and stress read, and the noise check's noisy copies, which stress
reads at the default noise; the parameter neighbourhood's and the noise-robust
optimization's runs of the same grid). The searching checks' runs read their own bars
(`_search_setup`), so they share runs with each other only, never with a bar check. When
no check will rerun, the plan says so, and each check's note says why.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._conditions import EVENT_PATH
from . import _random
from . import _words as words
from ._registry import SEARCHING
from ._request import Request
from ._reruns import REAL, Family

#: Reruns on the event path above which the plan suggests the condition form, as
#: `bt.optimize` does for a large grid (`_optimize.LARGE_EVENT_GRID`).
LARGE_EVENT_RERUNS = 1000
#: The plan's suggestion for that many reruns on the event path.
CONDITION_FORM = (
    " That many reruns suit the condition form (entry_when, exit_when), which runs "
    "without a Python call per bar."
)
#: What the searching checks' reruns read, in words.
SEARCHED = "the run's own bars over parameter combinations or windows of the run"


@dataclass(frozen=True)
class Plan:
    """What the checks of a report will rerun (module docstring): the reruns each plans,
    by name (`counts`), how many of them another bar check shares (`shared`), the bar
    checks' families, in the order the checks list them (`families`), whether any check
    reruns other symbols (`others`), the searching checks' families (`searched`) and how
    many of their runs another searching check shares (`search_shared`)."""

    counts: Mapping[str, int]
    shared: int
    families: tuple[Family, ...]
    others: bool = False
    searched: tuple[Family, ...] = ()
    search_shared: int = 0

    @property
    def total(self) -> int:
        """The reruns the checks will run, each shared one once."""
        return sum(self.counts.values()) - self.shared - self.search_shared


def planned(request: Request, preview: Any, search: Any) -> Plan:
    """The `Plan` of `request` on the run: each bar check's `reruns` on `preview` (a
    `BarRun` with no runner yet; None when no bar check runs) and each searching check's
    on `search` (a `SearchRun` with no runner yet; None when none runs)."""
    counts: dict[str, int] = {}
    most: dict[tuple[bool, Any], int] = {}
    for check in request.reruns:
        families = check.reruns(
            search if check.searches else preview,
            request.settings[check.name],
            runs=request.runs_of(check),
            seed=_random.check_seed(request.seed, check.number),
            objective=request.objective,
        )
        counts[check.name] = sum(runs for _, runs in families)
        for family, runs in families:
            key = (check.searches, family)
            most[key] = max(most.get(key, 0), runs)

    def kept(searched: bool) -> tuple[Family, ...]:
        return tuple(
            family
            for (searches, family), runs in most.items()
            if searches == searched and isinstance(family, Family) and runs
        )

    def shared(searched: bool) -> int:
        asked = sum(count for name, count in counts.items() if (name in SEARCHING) == searched)
        run = sum(runs for (searches, _), runs in most.items() if searches == searched)
        return asked - run

    others = any(isinstance(family, str) and runs for (_, family), runs in most.items())
    return Plan(counts, shared(False), kept(False), others, kept(True), shared(True))


def changed_only(family: Family) -> bool:
    """Whether `family` reruns the strategy on the run's own bars with a setting changed
    (`_bar_setup`), rather than on generated bars or with random entries."""
    return family.bars == REAL and family.entries_seed is None


def plan_words(plan: Plan | None, same: bool, reference: float, real: float) -> str:
    """The plan's sentence on the reruns (deterministic: no time, no worker count); no
    `plan` when no check reruns the backtest. `same` says whether the first rerun on the
    run's own bars gave the run's objective, `reference` against `real`."""
    if plan is None:
        return (
            "They read the run's trade list, so the backtest is not rerun and no worker "
            "process starts."
        )
    if not plan.total:
        return (
            "The trade-list checks read the run's trade list; the bar checks will not rerun "
            "the backtest, since each is not judged before any rerun, as its note says."
            + _changed(same, reference, real)
        )
    parts = [f"{name} {words.number(count, 0)}" for name, count in plan.counts.items() if count]
    total = words.number(plan.total, 0)
    text = (
        f"The trade-list checks read the run's trade list; {_who(plan)} rerun the backtest "
        f"{total} times or fewer on {_where(plan)} ({', '.join(parts)}), "
        "each an ordinary backtest of the same kernel."
    )
    if plan.shared:
        text += (
            f" The checks share {words.number(plan.shared, 0)} of the runs listed, which read "
            "the same generated bars, so the total counts each once."
        )
    if plan.search_shared:
        text += (
            f" The searching checks share {words.number(plan.search_shared, 0)} of their runs, "
            "the same combinations on the same bars, so the total counts each once."
        )
    return text + _changed(same, reference, real)


def _who(plan: Plan) -> str:
    """Which checks rerun, in words."""
    searching = any(count for name, count in plan.counts.items() if name in SEARCHING)
    bars = any(count for name, count in plan.counts.items() if name not in SEARCHING)
    if searching and bars:
        return "the bar and searching checks"
    return "the searching checks" if searching else "the bar checks"


def _where(plan: Plan) -> str:
    """What the reruns of `plan` read, in words."""
    searched = any(count for name, count in plan.counts.items() if name in SEARCHING)
    if not plan.families and not plan.others:
        return SEARCHED
    changed = [changed_only(family) for family in plan.families]
    if not any(changed):
        text = "generated bars or other symbols"
    elif all(changed) and not plan.others:
        text = "the run's own bars, each with one setting changed"
    else:
        text = "generated bars, other symbols or the run's own bars with one setting changed"
    return f"{text}, or {SEARCHED}" if searched else text


def _changed(same: bool, reference: float, real: float) -> str:
    """What the plan says when the first rerun did not give the run's objective."""
    if same:
        return ""
    return (
        f" A rerun on the run's own bars gave {reference!r} where the run gave {real!r}: the "
        "source's bars changed since the run, and every rerun is measured against the run."
    )


def timing_words(plan: Plan, seconds: float, generated: bool, workers: Any, path: str) -> str:
    """The plan's estimate of the reruns, reported before them and not kept: `seconds`
    the time of the timed rerun, on generated bars when `generated`, in `workers`, on the
    evaluation `path`."""
    total = plan.total
    if not total:
        return f" A rerun on the run's own bars took {seconds:.2f} s; no other will run."
    where = f"{workers.count} worker processes" if workers.count > 1 else "this process"
    why = f" ({workers.why})" if workers.why else ""
    estimate = total * seconds / max(workers.count, 1)
    timed = (
        "A rerun on generated bars took {:.2f} s, making its bars and measuring it included"
        if generated
        else "A rerun on the run's own bars took {:.2f} s"
    )
    text = (
        f" {timed.format(seconds)}; in {where}{why} the reruns should take about {estimate:.1f} s."
    )
    if path == EVENT_PATH and total > LARGE_EVENT_RERUNS:
        text += CONDITION_FORM
    return text


__all__ = [
    "CONDITION_FORM",
    "LARGE_EVENT_RERUNS",
    "SEARCHED",
    "Plan",
    "changed_only",
    "plan_words",
    "planned",
    "timing_words",
]
