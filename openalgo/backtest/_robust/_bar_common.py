"""What the bar checks share (design 17, sections 2.4 to 2.9): what they read of the run
(`BarRun`), the generated bars several checks read, the objective's break-even, the
figures of a set of reruns, and their fan and histogram.

`BarRun` is the run as the bar checks see it: its trade list (`_trades.TradeRun`), its
`Result`, the objective's value on it, its bars and the reruns' runner (None while the
report plans, before the runner exists), the report's seed, the symbols of `others=` and
what random entries are drawn from.

A strategy made by `bt.from_arrays` decides from arrays fixed in advance, which never read
generated bars: on them it trades at the same bars, so a check on generated bars could
judge only its fills, whatever the arrays were fitted to, and leaves it not judged
(`fixed_unjudged`, one reason for every such check).

Shared bars: the permutation test's permuted paths (`permuted_family`) are also versus
random's strategy on permuted bars and synthetic stress's permuted candidates, and the
noise check's noisy copies at its default settings (`noisy_family`) are also stress's
noisy candidates, so a report that runs several of them reruns each path once.

A rerun is profitable when its net profit is at least half a paisa
(`_trade_common.BREAK_EVEN_RUPEES`, as the trade-list checks count a profitable path). A
rerun's objective is measured when the rerun ran (its generated bars were made and not
refused) and the objective is a number; the checks judge the measured ones and say how
many were not.

Break-even: the ratio checks (versus random's margin, noise's share of the objective) read
an objective from the value of a run that makes nothing: 0 for sharpe, sortino, calmar,
net_profit, net_pnl, total_return, cagr and expectancy, 1 for profit_factor. Another
metric has none, nor has a function of a `Result` (what it gives for a run that makes
nothing is not known: the end equity, for one, gives the capital), and those checks are
not judged under it (`unmeasured`).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .._arrays import made_by
from ..errors import DataError
from . import _random
from . import _words as words
from ._blocks import REAL, FanBlock, HistogramBlock, Mark
from ._generate_noisy import Noise
from ._ohlc import same_grid
from ._part_objective import BREAK_EVEN as PART_BREAK_EVEN
from ._part_objective import NO_BREAK_EVEN, objective_name
from ._paths import Fan, bands
from ._random_entries import EntryPlan
from ._rerun_one import RerunRow
from ._reruns import NOISY, PERMUTED, Family, Reruns
from ._run_bars import RunBars
from ._trade_common import BREAK_EVEN_RUPEES
from ._trades import TradeRun
from ._verdict import Outcome, not_judged

#: The number of the permutation test: the stream its permuted bars come from, which
#: versus random and synthetic stress share (`permuted_family`).
PERMUTATION_NUMBER = 5
#: The number of the noise check: the stream its noisy copies come from, which synthetic
#: stress shares (`noisy_family`).
NOISE_NUMBER = 6
#: Why a function objective leaves the ratio checks not judged (module docstring).
FUNCTION_BREAK_EVEN = (
    "what a function of a Result gives for a run that makes nothing is not known (the end "
    "equity, for one, gives the capital)"
)
#: Break-even values of the objectives that have one (module docstring): those a part of
#: the trades is measured by, and the measures of a whole run's profit.
BREAK_EVEN = {
    **PART_BREAK_EVEN,
    "calmar": 0.0,
    "net_pnl": 0.0,
    "total_return": 0.0,
    "cagr": 0.0,
}
#: The runs a fan draws thin (design 17, section 2).
SAMPLED = 100


@dataclass(frozen=True, eq=False)
class BarRun:
    """The run as the bar checks read it (module docstring): `real` the objective's value
    on it (NaN when undefined), `reruns` the runner (None while the report plans), `seed`
    the report's seed, `others` the (symbol, exchange) pairs of `others=`, `client` the
    run's client, `entries` what random entries are drawn from (None when no check draws
    them)."""

    trades: TradeRun
    result: Any
    real: float
    bars: RunBars
    reruns: Reruns | None
    seed: int
    others: tuple[tuple[str, str], ...]
    client: Any
    entries: EntryPlan | None = None


def permuted_family(report_seed: int) -> Family:
    """The permutation test's permuted bars of the report seeded `report_seed`."""
    return Family(PERMUTED, _random.check_seed(report_seed, PERMUTATION_NUMBER))


def noisy_family(report_seed: int, noise: Noise) -> Family:
    """The noise check's noisy copies with `noise` of the report seeded `report_seed`."""
    return Family(NOISY, _random.check_seed(report_seed, NOISE_NUMBER), noise)


def grid_problem(bars: RunBars) -> str:
    """Why no permuted path of `bars` can be made ("" when one can): several symbols are
    permuted together only when their bars are at the same times (`_ohlc.same_grid`)."""
    try:
        same_grid(bars.frames, bars.symbols)
    except DataError as error:
        return str(error)
    return ""


#: Why a check on generated bars cannot judge a strategy made by `bt.from_arrays`
#: (module docstring), the one reason every such check gives.
FIXED_ARRAYS = (
    "its decisions are arrays fixed in advance (bt.from_arrays), which never read the "
    "generated bars, so on them it trades at the same bars and only its fills move"
)


def fixed_decisions(run: BarRun) -> bool:
    """Whether the run's strategy decides from arrays fixed in advance (`FIXED_ARRAYS`)."""
    return made_by(run.result._record.strategy) == "bt.from_arrays"


def fixed_unjudged(run: BarRun, title: str, rule: str) -> Outcome | None:
    """The "not judged" outcome of check `title` (as a sentence starts it, "Noise") with
    pass rule `rule` for a run whose strategy decides from arrays fixed in advance, else
    None."""
    if not fixed_decisions(run):
        return None
    return not_judged(rule, f"{title} cannot judge this strategy: {FIXED_ARRAYS}.")


def break_even(objective: Any) -> float | None:
    """The break-even value of `objective` (module docstring), None when it has none."""
    if not isinstance(objective, str):
        return None
    return BREAK_EVEN.get(objective)


def unmeasured(objective: Any) -> str:
    """Why a ratio check cannot judge `objective`, which has no break-even value."""
    name = objective_name(objective)
    if isinstance(objective, str):
        reason = NO_BREAK_EVEN.get(name, "it has no value that says a run made nothing")
    else:
        reason = FUNCTION_BREAK_EVEN
    return (
        f"The objective {name} cannot judge this check: {reason}; give objective= one of "
        f"{', '.join(BREAK_EVEN)} to judge it."
    )


def undefined(name: str, compared: str) -> str:
    """Why a check is not judged when the run's objective `name` is undefined: there is
    no figure for the `compared` runs to be measured against."""
    return (
        f"The run's {name} is undefined (a ratio of session returns needs returns that "
        f"vary, over two sessions or more), so the {compared} runs have no figure to be "
        "measured against."
    )


def below_break_even(name: str, real: float, level: float) -> str:
    """Why a check is not judged when the run's objective is not above break-even."""
    return (
        f"No edge to judge: the run's {name} is {words.number(real)}, not above "
        f"{level:g}, the value of a run that makes nothing."
    )


@dataclass(frozen=True, eq=False)
class RerunSet:
    """Rows of reruns as arrays: `objective` (NaN when not measured), `net_profit`,
    `max_drawdown`, `trades`, `largest_win`, `largest_loss`, `win_streak`, `loss_streak`,
    `paths` (runs, points) and `failures`, the reason of each refused run ("" for one that
    ran)."""

    objective: np.ndarray
    net_profit: np.ndarray
    max_drawdown: np.ndarray
    trades: np.ndarray
    largest_win: np.ndarray
    largest_loss: np.ndarray
    win_streak: np.ndarray
    loss_streak: np.ndarray
    paths: np.ndarray
    failures: tuple[str, ...]

    @property
    def measured(self) -> np.ndarray:
        """Whether each run ran and its objective is a number."""
        ran = np.array([not failure for failure in self.failures], dtype=bool)
        return ran & np.isfinite(self.objective)

    @property
    def profitable(self) -> np.ndarray:
        """Whether each run made at least half a paisa (module docstring)."""
        return self.net_profit >= BREAK_EVEN_RUPEES

    @property
    def ran(self) -> np.ndarray:
        """Whether each run ran (its bars were not refused)."""
        return np.array([not failure for failure in self.failures], dtype=bool)

    def failure_note(self) -> str:
        """How many runs were left out, and the first reason (their bars could not be
        made, or the data checks refused them); "" for none."""
        refused = [failure for failure in self.failures if failure]
        if not refused:
            return ""
        return (
            f" {words.whole(len(refused), 'run was', 'runs were')} left out (first: {refused[0]})."
        )

    def table(self) -> pd.DataFrame:
        """A row per run: its figures, and why it was refused."""
        frame = pd.DataFrame(
            {
                "objective": self.objective,
                "net_profit": self.net_profit,
                "max_drawdown": self.max_drawdown,
                "trades": self.trades,
                "largest_win": self.largest_win,
                "largest_loss": self.largest_loss,
                "win_streak": self.win_streak,
                "loss_streak": self.loss_streak,
                "refused": list(self.failures),
            }
        )
        frame.index.name = "run"
        return frame


def rerun_set(rows: Sequence[RerunRow]) -> RerunSet:
    """The `RerunSet` of `rows`, in order."""

    def column(name: str, kind: Any = float) -> np.ndarray:
        return np.array([getattr(row, name) for row in rows], dtype=kind)

    return RerunSet(
        column("objective"),
        column("net_profit"),
        column("max_drawdown"),
        column("trades", np.int64),
        column("largest_win"),
        column("largest_loss"),
        column("win_streak", np.int64),
        column("loss_streak", np.int64),
        np.vstack([row.path for row in rows]),
        tuple(row.failure for row in rows),
    )


def joined(parts: Sequence[RerunSet]) -> RerunSet:
    """The runs of `parts`, part after part, as one set."""
    values = {}
    for column in dataclasses.fields(RerunSet):
        pieces = [getattr(part, column.name) for part in parts]
        if column.name == "paths":
            values[column.name] = np.vstack(pieces)
        elif column.name == "failures":
            values[column.name] = tuple(failure for piece in pieces for failure in piece)
        else:
            values[column.name] = np.concatenate(pieces)
    return RerunSet(**values)


def real_path(trades: TradeRun, steps: np.ndarray) -> np.ndarray:
    """The run's own cumulative return at `steps` of its sessions (point 0 the start)."""
    return np.concatenate(([0.0], trades.cumulative))[steps]


def session_fan(title: str, subtitle: str, set_: RerunSet, run: BarRun) -> FanBlock:
    """The fan of the runs that ran, by session, with the run's own equity bold."""
    setup = run.reruns.setup
    paths = set_.paths[set_.ran]
    if not len(paths):
        paths = np.zeros((1, len(setup.steps)))
    fan = Fan(
        setup.steps, bands(paths), paths[:SAMPLED], real_path(run.trades, setup.steps), len(paths)
    )
    dates = pd.DatetimeIndex(setup.dates)
    return FanBlock(title, subtitle, fan, dates, "the run's own equity")


def objective_histogram(
    title: str, subtitle: str, values: np.ndarray, real: float, marks: tuple[Mark, ...] = ()
) -> HistogramBlock:
    """The histogram of the runs' objectives with the run's own marked."""
    shown = values[np.isfinite(values)]
    return HistogramBlock(title, subtitle, shown, False, (Mark(real, "the run", REAL), *marks))


def finite(value: float) -> bool:
    """Whether `value` is a number (not NaN, not infinite)."""
    return value is not None and math.isfinite(value)


__all__ = [
    "BREAK_EVEN",
    "FIXED_ARRAYS",
    "FUNCTION_BREAK_EVEN",
    "NOISE_NUMBER",
    "PERMUTATION_NUMBER",
    "BarRun",
    "RerunSet",
    "below_break_even",
    "break_even",
    "finite",
    "fixed_decisions",
    "fixed_unjudged",
    "grid_problem",
    "joined",
    "noisy_family",
    "objective_histogram",
    "permuted_family",
    "real_path",
    "rerun_set",
    "session_fan",
    "undefined",
    "unmeasured",
]
