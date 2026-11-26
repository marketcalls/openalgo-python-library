"""The pre-open decision of a live session's daily and longer strategies (design
20-positional-live section 8, decision 11.1, review M8; the lead's ruling N5b-1).

Such a strategy decides once a day, before the open, on the latest whole bar of its
interval of the broker's candles (`_plan_daily`), as its backtest on those bars decides at
that bar's close; its orders go at the open, as the backtest fills them at the next
session's first open.

- **The bar due** (`_pre_open_due`): only the bar that closed at the exchange's last session
  before today may be decided, as a backtest decides it for today's open. One whose daily
  candle the broker has not served, or a weekly or longer bar due at an earlier open, is
  not decided, and a warning says why.
- **Before the banner** (`warn_missed_bars`): that warning, or the bars of its interval
  that closed after the last bar it decided (the journal's, `Journal.decided_bar`) and
  before the bar due, which closed while no session ran: they feed its indicators only,
  and a warning names them.
- **When trading begins** (`decide_before_open`, 16-session-design 3.1 step 15, once its
  entries are unlocked, so nothing is sent before the countdown): the kernel is given its
  bars closing after the last decided one (`Engine.pre_open_decision`, which decides the
  latest whole bar and feeds the rest to the indicators), told a later last decided bar
  when none is due. A bar decided is noted in the journal at once (`put_decided_bar`),
  before any of its orders goes, so no later start of the day or of a later day decides it
  again; then `on_bar` runs as at any decision (`Trader.decide_before_open`), and the
  kernel's state is saved, with the orders waiting for the open in it. The kernel sends
  those orders as the session reaches its open; a decision made once the open has passed
  sends them at once, flagged `late_decision` on their fills (the result's trades, the
  fill's alert). A start that finds the latest bar decided decides nothing. A strategy's
  exception in its `on_bar` is raised once every strategy has decided, as `StrategyFailed`,
  which stops the session safely as in the loop.
- **A stop before the open** (Ctrl+C, or a strategy's exception: the safe stop cancels the
  entries) ends the decision's orders that wait for the open before they reach the broker,
  and the bar stays decided in the journal, so no later start of the day decides it again
  (a decided bar is noted before its orders go, so a crash never decides one twice). The
  stop says so in a line naming the bar and the orders (`cancelled_before_open`), kept in
  the result's warnings and the daily summary; the day's next start says the bar was
  decided by an earlier start (`_pre_open_due.DECIDED_TODAY`).

Times are UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

from typing import Any

from .._calendar_convert import format_ist
from .._engine import call
from .._orders import VALIDITIES
from ._plan_catch_up import INSTRUMENT
from ._plan_daily import daily_candles, decides_before_open, decision_bars
from ._pre_open_due import DueBar, StrategyBars, due_bar
from ._session_saved import save
from ._session_trader import StrategyFailed

#: The event of each strategy's pre-open decision.
PRE_OPEN_EVENT = "pre_open_decision"
#: The line of a pre-open decision's orders a stop before the open cancels (module
#: docstring): they never reached the broker, and the journal holds the bar as decided.
CANCELLED_BEFORE_OPEN = (
    "{name}: the stop before the open cancels the orders of its decision on its {interval} "
    "bar closing on {day} ({orders}): they never went to the broker, and that bar is not "
    "decided again today"
)


def daily_traders(session: Any) -> list[Any]:
    """The strategies of `session` that decide before the open, in order."""
    return [t for t in session.traders if decides_before_open(t.part.spec.interval)]


def warn_missed_bars(session: Any) -> None:
    """Name in a warning, for each daily or longer strategy, the bar due it does not decide,
    or else the bars of its interval that closed after its last decided bar and before the
    latest whole one (module docstring); none of those at a strategy's first start, whose
    earlier bars are its history. Raises JournalError when the journal cannot be read."""
    for trader in daily_traders(session):
        last = session.parts.journal.decided_bar(trader.name)
        due = _due(session, trader, last)
        if due.note is not None:
            session.warn(due.note)
            continue
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
    JournalError when the journal cannot be read or written, StrategyFailed for the first
    exception of a strategy's code once every strategy has decided, and what the kernel
    raises."""
    failure: StrategyFailed | None = None
    for trader in daily_traders(session):
        try:
            _decide(session, trader)
        except StrategyFailed as failed:
            failure = failure or failed
    if failure is not None:
        raise failure


def _decide(session: Any, trader: Any) -> None:
    """`trader`'s pre-open decision (module docstring). Raises as `decide_before_open`."""
    journal = session.parts.journal
    last = journal.decided_bar(trader.name)
    told = _due(session, trader, last).last_decided_ns
    bars = _bars(trader, last)
    decided = call(trader.part.engine.pre_open_decision, bars, told, session.now())
    if decided is None:
        session.event(PRE_OPEN_EVENT, strategy=trader.name, decided=False)
        return
    bar_close = int(decided["bar_close"])
    journal.put_decided_bar(trader.name, bar_close)
    late = bool(decided["late"])
    session.event(
        PRE_OPEN_EVENT, strategy=trader.name, decided=True, bar_close_ns=bar_close, late=late
    )
    trader.decide_before_open(decided["pause"])
    save(session, trader)


def cancelled_before_open(session: Any, trader: Any) -> str | None:
    """The line naming the orders of `trader`'s pre-open decision that a stop now cancels
    before the open (module docstring): its entries decided before the open that wait for
    it, never sent (DAY orders: a GTT order rests at the broker already). None for a
    strategy that decides in the session, after the open, or with none waiting. Called
    before the stop cancels them."""
    opens = trader.part.slices[0][0]
    if not decides_before_open(trader.part.spec.interval) or session.now() >= opens:
        return None
    orders = [call(trader.engine.order, kernel_order) for kernel_order in trader.entry_orders()]
    waiting = [o for o in orders if o["validity"] == VALIDITIES["day"] and o["decided_at"] < opens]
    if not waiting:
        return None
    named = ", ".join(f"{o['side']} {o['quantity_units']:g} {o['order_type']}" for o in waiting)
    return CANCELLED_BEFORE_OPEN.format(
        name=trader.name,
        interval=trader.part.spec.interval,
        day=_day(max(o["decided_at"] for o in waiting)),
        orders=named,
    )


def _due(session: Any, trader: Any, last_decided_ns: int | None) -> DueBar:
    """What `trader`'s start decides (`_pre_open_due`), its journal's last decided bar
    closing at `last_decided_ns`."""
    part = trader.part
    bars = StrategyBars(
        name=trader.name,
        interval=part.spec.interval,
        exchange=trader.exchange,
        closes=part.context.store(trader.symbol).column("ts_close"),
        daily_closes=daily_candles(part.setup, trader.symbol)["ts_close"],
        open_ns=part.slices[0][0],
    )
    return due_bar(bars, session.parts.spec.day, last_decided_ns, part.calendar)


def _bars(trader: Any, last_decided_ns: int | None) -> list[tuple]:
    """`trader`'s bars of its interval closing after `last_decided_ns`, as the kernel takes
    them."""
    store = trader.part.context.store(trader.symbol)
    return decision_bars(store, last_decided_ns, INSTRUMENT)


def _day(close_ns: int) -> str:
    """The IST date of a bar's close, "2026-11-13"."""
    return format_ist(close_ns)[:10]


__all__ = [
    "CANCELLED_BEFORE_OPEN",
    "PRE_OPEN_EVENT",
    "cancelled_before_open",
    "daily_traders",
    "decide_before_open",
    "warn_missed_bars",
]
