"""One strategy's setup for a live session's day, as a backtest of that day builds it
(16-session-design 3.1 step 11; 08-live-trading section 5 rule 6; unit 7). Part of
`plan`.

The steps are those of `_pipeline.prepare` and `execute` for a backtest of today with
`source="api"`, a 1-minute base and `now` the present, so history before today and
today's closed 1-minute candles are loaded and every setting and assumption is the
backtest's. The live terms differ:

- **The instruments' facts.** The lot size, tick size and freeze quantity the preflight
  read from OpenAlgo (`preflight.InstrumentFacts`) replace the reference tables' for the
  session (`with_facts`); a value that differs is a warning. The market settings
  (charges, slippage, per-instrument facts) are built on the replaced instruments.
- **Today's bars.** Today's closed candles (`read_today`, the data layer's load a
  backtest makes of its own bars) may be none, before the open or in its first minute;
  a backtest stops on that, a session does not. They are not the kernel's bars: the
  session's engine is built with none, and the plan's catch-up adds them slice by slice.
- **No base-interval probe and no price-slab ticks**: the base is 1 minute (1.1), and
  OpenAlgo's tick replaces any tick the reference tables leave to a price slab.

Times are UTC nanoseconds, prices rupees, quantities units (never lots).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import pandas as pd

from .._market import MarketSettings, market_settings
from .._settings import RunPlan
from .._warm_up_history import RunHistory, empty_base
from .._warm_up_load import load_history
from .._warm_up_source import HistorySource
from ..calendar import Calendar
from ..data import load_bars
from ..errors import ConfigError, NoDataError
from ..reference import Instrument
from ._plan_strategies import StrategySpec
from ._preflight_plan import InstrumentFacts
from .data_worker import CANDLE_INTERVAL, HISTORY_SOURCE

#: Where the facts that replace the reference tables' come from, as an instrument's
#: provenance records it.
FACTS_SOURCE = "OpenAlgo symbol (preflight)"
FACTS_STATUS = "live"
#: The instrument fields the preflight's facts replace, with the name of the fact.
FACT_FIELDS = (
    ("lot_size", "lot_size"),
    ("tick_size", "tick_size"),
    ("freeze_qty", "freeze_quantity"),
)
#: The arguments of a backtest that say where its history is read from
#: (`_warm_up_source.HistorySource`): the broker, no data of the user's.
_HISTORY_ARGUMENTS = MappingProxyType(
    {"data": None, "source": HISTORY_SOURCE, "tz": None, "corporate_actions": None}
)


@dataclass(frozen=True)
class StrategySetup:
    """One strategy's setup for the day (module docstring): its settings (`plan`, the
    present of the build), its instruments with OpenAlgo's facts, the market settings
    built on them, the native engine's config, the history kept before today
    (`_warm_up_history.RunHistory`), today's closed 1-minute bars as read at `read_ns`
    (the bars layout, possibly none) and the warnings, in order."""

    plan: RunPlan
    instruments: tuple[Instrument, ...]
    market: MarketSettings
    config: dict[str, Any]
    history: RunHistory
    today: pd.DataFrame
    read_ns: int
    warnings: tuple[str, ...]


def with_facts(
    instruments: Sequence[Instrument], facts: Sequence[InstrumentFacts]
) -> tuple[tuple[Instrument, ...], tuple[str, ...]]:
    """(the instruments with the preflight's lot size, tick size and freeze quantity in
    place of the reference tables', a warning for each value that differed). Raises
    ConfigError for an instrument the facts do not cover (the preflight found it, so the
    session passes its report's facts) and for facts an instrument cannot hold."""
    by_key = {(fact.symbol, fact.exchange): fact for fact in facts}
    replaced, notes = [], []
    for instrument in instruments:
        fact = by_key.get((instrument.symbol, instrument.exchange))
        if fact is None:
            raise ConfigError(
                f"no preflight facts of {instrument.exchange}:{instrument.symbol}: pass the "
                "instrument facts of the preflight's report"
            )
        values = {field: getattr(fact, name) for field, name in FACT_FIELDS}
        for field, value in values.items():
            if getattr(instrument, field) != value:
                notes.append(
                    f"{instrument.symbol}: OpenAlgo's {field} {value} replaces "
                    f"{getattr(instrument, field)} of the reference tables for the session"
                )
        provenance = tuple((field, FACTS_SOURCE, FACTS_STATUS) for field in values)
        replaced.append(
            dataclasses.replace(instrument, **values, generic=False, provenance=provenance)
        )
    return tuple(replaced), tuple(notes)


def read_today(
    spec: StrategySpec, client: Any, calendar: Calendar, read_ns: int
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """(today's closed 1-minute bars of the strategy's symbol, the bars layout, none when
    the source has none yet; the load's warnings), read through `client` (the session's
    data client) as a backtest of today loads its own bars (`_base_interval.choose_base`
    with its base given), the candles that had not closed at `read_ns` (UTC nanoseconds)
    dropped. Raises the data layer's errors but NoDataError."""
    plan = spec.plan
    try:
        result = load_bars(
            spec.symbol,
            spec.exchange,
            CANDLE_INTERVAL,
            plan.start,
            plan.end,
            source=HISTORY_SOURCE,
            client=client,
            calendar=calendar,
            now=read_ns,
        )
    except NoDataError:
        return empty_base(), ()
    return result.bars, tuple(result.meta.warnings)


def strategy_setup(
    spec: StrategySpec,
    client: Any,
    facts: Sequence[InstrumentFacts],
    present_ns: int,
    calendar: Calendar,
) -> StrategySetup:
    """The setup of `spec` for its day at the present `present_ns` (UTC nanoseconds),
    reading through `client` (module docstring).

    Raises ConfigError for facts that do not cover an instrument and for settings the
    market cannot use, CostError for a cost model that cannot be used, ServerError when
    the OpenAlgo server cannot answer history at all (`_warm_up_load`), and the data
    layer's other errors for today's bars."""
    plan = dataclasses.replace(spec.plan, present=present_ns)
    instruments, notes = with_facts(spec.instruments, facts)
    arguments = spec.arguments
    market_args = {"costs": arguments["costs"], "slippage": arguments["slippage"]}
    market = market_settings(plan, list(instruments), market_args | {"margin": plan.margin})
    today, read_notes = read_today(spec, client, calendar, present_ns)
    origins = {spec.symbol: (HISTORY_SOURCE, None)}
    source = HistorySource(_HISTORY_ARGUMENTS, client, calendar, origins)
    loaded = load_history(plan, source, [spec.needs])
    history = loaded.for_run(spec.needs, plan, calendar, {spec.symbol: today})
    config = plan.native_config(deliver_events=spec.strategy.wants_events()) | market.config
    return StrategySetup(
        plan=plan,
        instruments=instruments,
        market=market,
        config=config,
        history=history,
        today=today,
        read_ns=present_ns,
        warnings=(*plan.warnings, *read_notes, *market.warnings, *history.warnings, *notes),
    )


__all__ = [
    "FACTS_SOURCE",
    "FACT_FIELDS",
    "StrategySetup",
    "read_today",
    "strategy_setup",
    "with_facts",
]
