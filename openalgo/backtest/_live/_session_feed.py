"""What the session loop does with the market's messages (16-session-design 2.5 A.1, A.10
to A.12; units 8 and 10), split from `_session_news`, which takes the order worker's.

- `FeedTick`: adapted, its receive time stamped now, given to the risk watch (`watch`:
  the feed's health, the clock skew), handed to its exchange's bar builder, kept as the
  instrument's last price, and given as a price to the kernel session of the strategy
  that trades it and to the stops supervisor (tick mode, R5; the price rule, 4.6).
- `Quote`: a REST quote while the feed is silent, to the risk watch, which drops one no
  longer asked for and gives the others to the supervisor as a price (`RiskWatch.quote`):
  never to the bar builder (a quote is not a trade).
- `Candles`: to the strategy of the instrument (`Trader.candles`).
- `FeedEvent`: to the feed link; a recovery resumes every bar builder over the outage and
  asks the worker to poll the orders the stream may have missed.
- `DataFailed`: to the risk watch (`RiskWatch.data_failed`).
- `OtherFill`: another client's fill the order stream reported, on an instrument a
  strategy trades, asks the reconciler for a comparison soon (`Reconciler.other_fill`), so
  a sale by hand is seen within seconds (7.1 "As built (unit 14"); on any other
  instrument it changes nothing.

Times are UTC nanoseconds, prices rupees. Used by the loop only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

from ._types import LOGGER_NAME
from ._worker_queue import StreamGap
from .inbox import Candles, DataFailed, FeedEvent, FeedTick, OtherFill, Quote
from .ticks import tick_from_feed

logger = logging.getLogger(LOGGER_NAME)


def take_tick(session: Any, message: FeedTick) -> None:
    """A tick of the feed (module docstring)."""
    tick = tick_from_feed(message.message, recv_ns=session.now(), tally=session.tally)
    if tick is None:
        return
    # Before the bar builder: a tick that ends a dead spell resumes the builders first.
    session.risk.tick(tick, message.arrived_ns)
    builder = session.builders.get(tick.exchange)
    if builder is not None:
        builder.on_tick(tick)
    session.last_tick[(tick.symbol, tick.exchange)] = tick
    trader = session.trader_of(tick.symbol, tick.exchange)
    if trader is not None and tick.ltp > 0:
        session.stops.price(trader, tick.ltp, tick.ts_recv_ns)


def take_quote(session: Any, message: Quote) -> None:
    """A REST quote of an instrument (module docstring)."""
    session.risk.quote(message)


def take_feed_event(session: Any, message: FeedEvent) -> None:
    """A hook of the feed (module docstring)."""
    recovery = None if session.link is None else session.link.note(message)
    if recovery is None:
        return
    if recovery.outage is not None:
        for builder in session.builders.values():
            builder.resume(recovery.outage.start_ns, session.now())
        session.event("outage", start_ns=recovery.outage.start_ns, end_ns=session.now())
    if recovery.stream_gap_since_ns is not None:
        session.hand_over([StreamGap(recovery.stream_gap_since_ns)])


def take_candles(session: Any, message: Candles) -> None:
    """Broker candles the data worker read (module docstring)."""
    trader = session.trader_of(message.symbol, message.exchange)
    if trader is None:
        logger.error("candles of %s:%s, which no strategy trades", message.exchange, message.symbol)
        return
    for note in trader.candles.take(message.rows, message.read_ns):
        session.warn(note)


def take_data_failed(session: Any, message: DataFailed) -> None:
    """A read of the data worker that failed (module docstring)."""
    session.risk.data_failed(message)


def take_other_fill(session: Any, message: OtherFill) -> None:
    """Another client's fill on the order stream (module docstring)."""
    trader = session.trader_of(message.symbol, message.exchange)
    if trader is None:
        return
    session.event("other_fill", strategy=trader.name, arrived_ns=message.arrived_ns)
    session.reconcile.other_fill(message.arrived_ns)


#: The market's messages, each with its handler, for the loop's dispatch (`_session_news`).
MARKET: Mapping[type, Callable[[Any, Any], None]] = MappingProxyType(
    {
        FeedTick: take_tick,
        Quote: take_quote,
        FeedEvent: take_feed_event,
        Candles: take_candles,
        DataFailed: take_data_failed,
        OtherFill: take_other_fill,
    }
)


__all__ = [
    "MARKET",
    "take_candles",
    "take_data_failed",
    "take_feed_event",
    "take_other_fill",
    "take_quote",
    "take_tick",
]
