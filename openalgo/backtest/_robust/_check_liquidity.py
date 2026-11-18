"""Liquidity (design 17, section 2.15): does it survive worse fill prices?

The strategy reruns on its own bars under each fill model of `bt.backtest(fill=)` (the
kernel's `engine::fill_model`) besides the run's own: the next base bar's close
("next_close"), its midpoint ("next_mid"), the average of its open, high, low and close
("next_ohlc4"), and each of them averaged with the bar after it ("next_close_2",
"next_mid_2", "next_ohlc4_2"); a run made on another model reruns on the next open
("next_open") too, in place of its own (`models`). A model prices the strategy's market
orders, after the bar's stops and targets, with the run's slippage; its resting orders,
GTTs, the legs' stops and targets and the engine's own exits keep their prices, and the
note says how many of the run's fills the models priced (`priced_fills`).

Figures: each model's objective (and its ratio to the run's from the objective's
break-even, when it has one) and its maximum drawdown against the run's. Pass: every
model profitable (net profit of at least half a paisa) and no model's maximum drawdown
deeper than `drawdown` (1.5) times the run's.

Not judged when the run did not make money or has too few closed trades, when no fill
of the run was a market order of the strategy (every rerun would be the run itself), or
when no model ran.

Charts: the equity of each model, the next bar's prices and the two-bar averages in a
group each, with the run's bold; a table by model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .._perturbation import FILL_MODELS
from . import _words as words
from ._bar_common import BarRun, RerunSet
from ._blocks import Tile, TilesBlock
from ._part_objective import objective_name
from ._reruns import REAL, Family
from ._setting import Setting, between
from ._trade_common import BREAK_EVEN_RUPEES, ENOUGH_TRADES
from ._variant_common import (
    labelled_table,
    ratios,
    run_unjudged,
    variant_set,
    variants_chart,
    variants_table,
)
from ._verdict import FAILED, PASSED, Outcome, not_judged

#: The default fill, which a run made on a model is checked against.
NEXT_OPEN = "next_open"
#: The order type of the strategy's orders the fill models price (`Result.orders`).
MARKET = "MARKET"

#: What each model fills at, in words.
MEANINGS = {
    NEXT_OPEN: "the next bar's open",
    "next_close": "the next bar's close",
    "next_mid": "the next bar's midpoint",
    "next_ohlc4": "the next bar's average price",
    "next_close_2": "the close of the next two bars",
    "next_mid_2": "the midpoint of the next two bars",
    "next_ohlc4_2": "the average price of the next two bars",
}

SETTINGS = (
    ENOUGH_TRADES,
    Setting(
        "drawdown",
        1.5,
        between(1.0, 100.0, low_open=False, high_open=False),
        "a number from 1 to 100",
        "how many times the run's maximum drawdown a model's may reach",
    ),
)


def rule(settings: Mapping[str, Any], objective: Any) -> str:
    """The pass rule, in words."""
    return (
        "every fill model profitable, and no model's maximum drawdown deeper than "
        f"{settings['drawdown']:g} times the run's"
    )


def _family(model: str) -> Family:
    return Family(REAL, arguments=(("fill", model),))


def models(subject: BarRun) -> tuple[str, ...]:
    """The fill models the run is checked under: every model but its own fill, the next
    open first when the run was made on another (module docstring)."""
    own = subject.result._record.arguments["fill"]
    return tuple(model for model in (NEXT_OPEN, *FILL_MODELS) if model != own)


def priced_fills(result: Any) -> tuple[int, int]:
    """(the run's fills of the strategy's market orders, which a fill model prices; all
    its fills)."""
    orders, fills = result.orders, result.fills
    market = orders.loc[orders["order_type"] == MARKET, "order_id"]
    return int(fills["order_id"].isin(market).sum()), len(fills)


def unjudged(subject: BarRun, settings: Mapping[str, Any], objective: Any) -> Outcome | None:
    """The "not judged" outcome of the module docstring that needs no rerun, else None:
    the one source of whether the check reruns (`reruns`)."""
    text = rule(settings, objective)
    short = run_unjudged(subject, settings, text)
    if short is None and priced_fills(subject.result)[0] == 0:
        short = not_judged(
            text,
            "Nothing to judge: no fill of the run was a market order of the strategy, the "
            "only orders the fill models price; its resting orders, GTTs, the legs' stops "
            "and targets and the engine's own exits keep their prices under every model, "
            "so each rerun would be the run itself.",
        )
    return short


def reruns(
    subject: BarRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> list[tuple[Family, int]]:
    """(family, runs) of what the check reruns: one run of each model, none when it is
    not judged before any rerun (`unjudged`)."""
    if unjudged(subject, settings, objective) is not None:
        return []
    return [(_family(model), 1) for model in models(subject)]


def run(
    subject: BarRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> Outcome:
    """The check of the module docstring."""
    text = rule(settings, objective)
    short = unjudged(subject, settings, objective)
    if short is not None:
        return short
    made = models(subject)
    set_ = variant_set(subject, [_family(model) for model in made])
    ran = set_.ran
    if not ran.any():
        return not_judged(text, "No fill model ran." + set_.failure_note())
    name = objective_name(objective)
    ratio = ratios(set_.objective, subject.real, objective)
    real_drawdown = float(subject.result.metrics["max_drawdown"])
    depth = _depths(set_.max_drawdown, real_drawdown)
    summary = _summary(set_, depth, real_drawdown)
    passed = summary["profitable_share"] == 1.0 and summary["deepest"] <= settings["drawdown"]
    figure = (
        f"{words.percent(summary['profitable_share'])} of "
        f"{words.whole(int(summary['models']), 'fill model')} profitable; the deepest "
        f"maximum drawdown is {words.number(summary['deepest'])} times the run's "
        f"{words.percent(real_drawdown)}"
    )
    priced, every = priced_fills(subject.result)
    note = (
        "Each run filled the strategy's market orders at another price of the bar after "
        "the decision, with the run's slippage, after that bar's stops and targets. The "
        f"fill models priced {priced} of the run's {every} fills; the others (resting "
        "orders, GTTs, the legs' stops and targets, the engine's own exits) kept their "
        "prices." + set_.failure_note()
    )
    table = labelled_table(set_, ratio, model=made)
    table.insert(table.columns.get_loc("max_drawdown") + 1, "drawdown_times", depth)
    labels = [MEANINGS[model] for model in made]
    blocks = _blocks(subject, (set_, made), ratio, labels, summary, name)
    return Outcome(PASSED if passed else FAILED, figure, text, note, table, summary, blocks)


def _depths(drawdowns: np.ndarray, real: float) -> np.ndarray:
    """Each maximum drawdown as a multiple of the run's (both zero or negative): 0 for
    none, infinite for a fall where the run had none."""
    drawdowns = np.asarray(drawdowns, dtype=float)
    if real < 0:
        return drawdowns / real
    return np.where(drawdowns < 0, np.inf, 0.0)


def _summary(set_: RerunSet, depth: np.ndarray, real_drawdown: float) -> dict[str, float]:
    ran = set_.ran
    return {
        "models": float(ran.sum()),
        "profitable_share": float((set_.net_profit[ran] >= BREAK_EVEN_RUPEES).mean()),
        "deepest": float(np.max(depth[ran])),
        "run_max_drawdown": real_drawdown,
        "worst_net_profit": float(np.min(set_.net_profit[ran])),
    }


def _blocks(
    subject: BarRun,
    ran: tuple[RerunSet, tuple[str, ...]],
    ratio: np.ndarray,
    labels: list[str],
    summary: Mapping[str, float],
    name: str,
) -> tuple:
    set_, made = ran
    one_bar = [k for k, model in enumerate(made) if not model.endswith("_2")]
    two_bars = [k for k, model in enumerate(made) if model.endswith("_2")]
    chart = variants_chart(
        "Equity under other fill prices",
        "The next bar's prices and the two-bar averages; the run bold",
        [("the next bar", one_bar), ("two-bar averages", two_bars)],
        set_,
        subject,
    )
    table = variants_table(
        "By fill model", "Where each run's orders filled", labels, set_, ratio, name
    )
    tiles = TilesBlock(
        "Summary",
        "Over the fill models",
        (
            Tile("Profitable", words.percent(summary["profitable_share"]), "of the models"),
            Tile("Deepest drawdown", words.number(summary["deepest"]), "times the run's"),
            Tile(
                "Run's drawdown",
                words.percent(summary["run_max_drawdown"]),
                "maximum",
                "neg",
            ),
            Tile("Worst net P&L", words.rupees(summary["worst_net_profit"]), "of a model"),
        ),
    )
    return ((chart,), (table,), (tiles,))


__all__ = ["MEANINGS", "SETTINGS", "models", "priced_fills", "reruns", "rule", "run", "unjudged"]
