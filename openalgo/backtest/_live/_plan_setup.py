"""One strategy's setup for a live session's day, as a backtest of that day builds it
(16-session-design 3.1 step 11; 08-live-trading section 5 rule 6; unit 7). Part of
`plan`.

The steps are those of `_pipeline.prepare` and `execute` for a backtest of today with
`source="api"`, a 1-minute base and `now` the present, so history before today and
today's closed 1-minute candles are loaded and every setting and assumption is the
backtest's. The live terms differ:

- **The instruments' facts.** The lot size, tick size and freeze quantity the preflight
  read from OpenAlgo (`preflight.InstrumentFacts`) replace the reference tables' for the
  session (`with_facts`), and so does their provenance; a value that differs is a
  warning. A freeze quantity OpenAlgo has none of (a stock install configures them for
  NFO only) leaves the tables' in place, with a warning: the kernel denies an order
  above it and splits exits into orders of at most it, which the exchange requires.
  The market settings (charges, slippage, per-instrument facts) are built on the
  replaced instruments.
- **Today's bars.** Today's closed candles (`read_today`, the data layer's load a
  backtest makes of its own bars) may be none; a backtest stops on that, a session does
  not. Before the open or in its first minute none are due; later the plan's catch-up
  marks the slices it gives up on missing, with a warning (`_plan_release`). They are
  not the kernel's bars: the session's engine is built with none, and the plan's
  catch-up adds them slice by slice.
- **No base-interval probe and no price-slab ticks**: the base is 1 minute (1.1), and
  OpenAlgo's tick replaces any tick the reference tables leave to a price slab (a line
  says so, not a difference: the tables give no tick to differ from).
- **A daily or longer strategy** (`_plan_daily`): its warm-up history is read on a daily
  base, as a backtest of it reads it, and the broker's daily candles of the days before
  today with it (`recent`), the bars its pre-open decision is made on; its kernel session
  keeps the 1-minute base of every live session.

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
from .._risk_settings import rearm_warning
from .._settings import RunPlan
from .._warm_up_history import RunHistory, empty_base
from .._warm_up_load import load_history
from .._warm_up_source import HistorySource
from ..calendar import Calendar
from ..data import load_bars
from ..errors import ConfigError, NoDataError
from ..reference import Instrument
from ._plan_daily import history_plan, recent_bars
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
    (the bars layout, possibly none) and the warnings, in order; `recent`, for a daily or
    longer strategy, the broker's daily candles of the days before today (`_plan_daily`),
    else None."""

    plan: RunPlan
    instruments: tuple[Instrument, ...]
    market: MarketSettings
    config: dict[str, Any]
    history: RunHistory
    today: pd.DataFrame
    read_ns: int
    warnings: tuple[str, ...]
    recent: pd.DataFrame | None = None


def with_facts(
    instruments: Sequence[Instrument], facts: Sequence[InstrumentFacts]
) -> tuple[tuple[Instrument, ...], tuple[str, ...]]:
    """(the instruments with the preflight's lot size, tick size and freeze quantity in
    place of the reference tables', and in their provenance, the other entries kept in
    order; the warnings, in order): a warning for each value that differed, one for a
    tick the tables leave to a price slab, and one for a freeze quantity the tables keep
    since OpenAlgo has none (module docstring). Raises ConfigError for an instrument the
    facts do not cover (the preflight found it, so the session passes its report's
    facts) and for facts an instrument cannot hold."""
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
        if values["freeze_qty"] is None:
            del values["freeze_qty"]
            if instrument.freeze_qty is not None:
                notes.append(
                    f"{instrument.symbol}: OpenAlgo has no freeze quantity configured for "
                    f"{instrument.underlying or instrument.symbol}, so the reference tables' "
                    f"{instrument.freeze_qty} is used for the session"
                )
        notes += _differences(instrument, values)
        provenance = _provenance(instrument.provenance, values)
        replaced.append(
            dataclasses.replace(instrument, **values, generic=False, provenance=provenance)
        )
    return tuple(replaced), tuple(notes)


def _differences(instrument: Instrument, values: dict[str, Any]) -> list[str]:
    """The warnings for OpenAlgo's `values` of `instrument`'s fields (`with_facts`)."""
    notes = []
    for field, value in values.items():
        ours = getattr(instrument, field)
        if field == "tick_size" and ours is None:
            notes.append(
                f"{instrument.symbol}: OpenAlgo's tick_size {value} is used for the session "
                "(the reference tables give the tick by price slab)"
            )
        elif ours != value:
            notes.append(
                f"{instrument.symbol}: OpenAlgo's {field} {value} replaces {ours} of the "
                "reference tables for the session"
            )
    return notes


def _provenance(
    provenance: tuple[tuple[str, str, str], ...], replaced: dict[str, Any]
) -> tuple[tuple[str, str, str], ...]:
    """`provenance` with the entries of the `replaced` fields OpenAlgo's, in place, and
    those it lacks after the rest."""
    entry = {field: (field, FACTS_SOURCE, FACTS_STATUS) for field in replaced}
    kept = tuple(entry.get(row[0], row) for row in provenance)
    listed = {row[0] for row in provenance}
    return kept + tuple(entry[field] for field in replaced if field not in listed)


def read_today(
    spec: StrategySpec, client: Any, calendar: Calendar, read_ns: int
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """(today's closed 1-minute bars of the strategy's symbol, the bars layout, empty
    when the source has none, which the data layer raises as NoDataError: the catch-up
    then marks the slices it gives up on missing, with a warning (module docstring);
    the load's warnings), read through `client` (the session's data client) as a
    backtest of today loads its own bars (`_base_interval.choose_base` with its base
    given), the candles that had not closed at `read_ns` (UTC nanoseconds) dropped.
    Raises the data layer's other errors."""
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


def _live_warnings(plan: RunPlan) -> tuple[str, ...]:
    """The warnings of `plan` that hold for a live session: all of them but the backtest's
    note of `positional_stops="rearm_daily"`, which speaks of a backtest; the session says
    what the setting leaves unprotected in its own words at every start and close
    (`_session_rearm`)."""
    return tuple(note for note in plan.warnings if note != rearm_warning(plan.product))


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
    # A daily or longer strategy's history is daily, as a backtest of it reads it.
    read_plan = history_plan(plan)
    daily = read_plan is not plan
    loaded = load_history(read_plan, source, [spec.needs])
    own = {spec.symbol: empty_base() if daily else today}
    history = loaded.for_run(spec.needs, read_plan, calendar, own)
    recent = None
    if daily:
        recent = recent_bars(spec.symbol, spec.exchange, plan, client, calendar, present_ns)
    config = plan.native_config(deliver_events=spec.strategy.wants_events()) | market.config
    return StrategySetup(
        plan=plan,
        instruments=instruments,
        market=market,
        config=config,
        history=history,
        today=today,
        read_ns=present_ns,
        warnings=(*_live_warnings(plan), *read_notes, *market.warnings, *history.warnings, *notes),
        recent=recent,
    )


__all__ = [
    "FACTS_SOURCE",
    "FACT_FIELDS",
    "StrategySetup",
    "read_today",
    "strategy_setup",
    "with_facts",
]
