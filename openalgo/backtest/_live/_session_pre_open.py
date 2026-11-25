"""The pre-open decision of a live session's daily and longer strategies (design
20-positional-live section 8, decision 11.1, review M8; the lead's ruling N5b-1).

Such a strategy decides once a day, before the open, on the latest whole bar of its
interval of the broker's candles (`_plan_daily`), as its backtest on those bars decides at
that bar's close; its orders go at the open, as the backtest fills them at the next
session's first open.

- **Before the banner** (`warn_missed_bars`): the bars of its interval that closed after the
  last bar it decided (the journal's, `Journal.decided_bar`) and before the latest whole one
  closed while no session ran: they feed its indicators only, and a warning names them.
- **When trading begins** (`decide_before_open`, 16-session-design 3.1 step 15, once its
  entries are unlocked, so nothing is sent before the countdown): the kernel is given its
  bars closing after the last decided one (`Engine.pre_open_decision`, which decides the
  latest whole bar and feeds the rest to the indicators). A bar decided is noted in the
  journal at once (`put_decided_bar`), before any of its orders goes, so no later start of
  the day or of a later day decides it again; then `on_bar` runs as at any decision
  (`Trader.decide_before_open`), and the kernel's state is saved, with the orders waiting
  for the open in it. The kernel sends those orders as the session reaches its open; a
  decision made once the open has passed sends them at once, flagged `late_decision` on
  their fills (the result's trades, the fill's alert). A start that finds the latest bar
  decided decides nothing.

Times are UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

from typing import Any

from .._calendar_convert import format_ist
from .._engine import call
from ._plan_catch_up import INSTRUMENT
from ._plan_daily import decides_before_open, decision_bars
from ._session_saved import save

#: The event of each strategy's pre-open decision.
PRE_OPEN_EVENT = "pre_open_decision"


def daily_traders(session: Any) -> list[Any]:
    """The strategies of `session` that decide before the open, in order."""
    return [t for t in session.traders if decides_before_open(t.part.spec.interval)]


def warn_missed_bars(session: Any) -> None:
    """Name in a warning, for each daily or longer strategy, the bars of its interval that
    closed after its last decided bar and before the latest whole one (module docstring);
    none at a strategy's first start, whose earlier bars are its history. Raises
    JournalError when the journal cannot be read."""
    for trader in daily_traders(session):
        last = session.parts.journal.decided_bar(trader.name)
        if last is None:
            continue
        opens = trader.part.slices[0][0]
        closes = [bar[2] for bar in _bars(trader, last) if bar[2] <= opens]
        if len(closes) < 2:
            continue
        missed = ", ".join(_day(close) for close in closes[:-1])
        session.warn(
            f"{trader.name}: its {trader.part.spec.interval} bars closing on {missed} closed "
            "while no session ran: they feed its indicators only, and it decides the bar "
            f"closing on {_day(closes[-1])}"
        )


def decide_before_open(session: Any) -> None:
    """Each daily or longer strategy's pre-open decision (module docstring). Raises
    JournalError when the journal cannot be read or written, StrategyFailed for an
    exception of the strategy's code, and what the kernel raises."""
    journal = session.parts.journal
    for trader in daily_traders(session):
        last = journal.decided_bar(trader.name)
        bars = _bars(trader, last)
        decided = call(trader.part.engine.pre_open_decision, bars, last, session.now())
        if decided is None:
            session.event(PRE_OPEN_EVENT, strategy=trader.name, decided=False)
            continue
        bar_close = int(decided["bar_close"])
        journal.put_decided_bar(trader.name, bar_close)
        late = bool(decided["late"])
        session.event(
            PRE_OPEN_EVENT, strategy=trader.name, decided=True, bar_close_ns=bar_close, late=late
        )
        trader.decide_before_open(decided["pause"])
        save(session, trader)


def _bars(trader: Any, last_decided_ns: int | None) -> list[tuple]:
    """`trader`'s bars of its interval closing after `last_decided_ns`, as the kernel takes
    them."""
    store = trader.part.context.store(trader.symbol)
    return decision_bars(store, last_decided_ns, INSTRUMENT)


def _day(close_ns: int) -> str:
    """The IST date of a bar's close, "2026-11-13"."""
    return format_ist(close_ns)[:10]


__all__ = ["PRE_OPEN_EVENT", "daily_traders", "decide_before_open", "warn_missed_bars"]
