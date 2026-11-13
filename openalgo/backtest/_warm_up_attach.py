"""The history before `start`, joined to a run's series once `init` has run (decision
D25; the rule is in `_warm_up`).

`init` runs once, on the run's own bars, with no history: what it registers says what
history its indicators need (`_warm_up_probe.history_needs`), and only then is that
history known (`_warm_up_history`). It is joined in front of each symbol's series here:
the strategy bars (`BarStore.rebase`), the base bars every other interval is built from
(`RunData.base_bars`) and the series of the other intervals `init` read
(`self.data[symbol, "W"]`, built again from those base bars). Every view `init` made is
then computed again by its recipe (`_views.Recipe`) on the joined bars: series,
indicators, expressions and conditions alike, so that their values are those of one
computation over the history and the run together, as if the history had been there
from the start. A view without a recipe (an indicator over values of the user's own,
which cover the run's bars) keeps its values on the run's bars and is unknown (NaN, or
false for a condition) on the history's.

Nothing is decided before this: the driver begins after it. A futures series has no
history, so nothing is joined.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from ._series import FIELDS, TIMES, SeriesStore

COLUMNS = (*TIMES, *FIELDS)


def _padded(values: np.ndarray, added: int) -> np.ndarray:
    """`values` after `added` unknown values: false for a condition, else NaN."""
    if values.dtype == np.bool_:
        return np.concatenate((np.zeros(added, dtype=bool), values))
    return np.concatenate((np.full(added, np.nan), values.astype(float)))


def _recompute(store: SeriesStore, added: int) -> None:
    """Compute again every live view made on `store` while `init` declared, on its
    joined bars, `added` of them new at the front; then forget them."""
    fields = store.fields()
    for reference in store.views:
        view = reference()
        if view is None:
            continue
        if view._recipe is not None:
            view._values = np.asarray(view._recipe(fields))
        else:
            view._values = _padded(view._values, added)
    store.views.clear()


def attach(context: Any, history: Any) -> None:
    """Join `history` (`_warm_up_history.RunHistory`) to the series of `context` (the
    run's `RunContext`, after `init`, before the driver) and compute the views of `init`
    again (module docstring)."""
    strategy_bars = history.strategy_bars()
    for symbol, store in context.stores.items():
        before = strategy_bars.get(symbol)
        added = 0 if before is None else len(before["ts_open"])
        if added:
            joined = {name: np.concatenate((before[name], store.column(name))) for name in COLUMNS}
            store.rebase(joined, added)
            _recompute(store, added)
    run = context.run
    base_bars = {symbol: history.with_base(symbol, bars) for symbol, bars in run.base_bars.items()}
    context.run = dataclasses.replace(run, base_bars=base_bars)
    for (symbol, interval), store in context.timeframes.items():
        # Imported here, as the run context does: a run on its own interval needs none.
        from ._timeframes import timeframe_store

        rebuilt = timeframe_store(
            symbol,
            run.exchange,
            interval,
            run.base,
            base_bars[symbol],
            run.calendar,
            lambda: context.decided_at,
        )
        added = len(rebuilt.column("ts_open")) - len(store.column("ts_open"))
        if added:
            store.rebase({name: rebuilt.column(name) for name in COLUMNS}, 0)
            _recompute(store, added)


__all__ = ["attach"]
