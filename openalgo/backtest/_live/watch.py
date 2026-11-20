"""The risk watch of a live session (16-session-design 6.2 to 6.5, 2.5 B; unit 11): the
feed's health, the clock skew from the feed and the account's daily loss limit, run by
the loop.

- **Feed dead** (6.5): no tick on any instrument for `DEFAULT_DEAD_AFTER_S` while a
  session of the traded exchanges is open (`bars.FeedHealth`). One FEED_DEAD alert,
  every strategy's entries locked ("feed_dead"), no candle check read meanwhile (the
  data worker drops them), and REST quotes of each instrument with an open position
  every `QUOTE_FALLBACK_EVERY_S`. A quote is a price for the kernel (targets, trailing,
  the daily loss), never a bar; quotes that fail too are an ERROR alert: the broker's
  login may have expired.
- **Back** (6.5): the first tick after a dead spell resumes every bar builder from the
  receive time of the last tick before it to now, so the spell's volume lands on no
  bar, whether or not the connection dropped. The lock lifts at the close of the first
  whole slice after that tick, unless the feed died again by then.
- **Quiet** (6.5): an instrument with an open position that has not ticked for
  `QUIET_AFTER_S` while others tick gets REST quotes too; nothing is locked.
- **Missing slices** (2.5 B): a slice released with no bar of the instrument while the
  feed was blind to it (dead, or its connection down) is marked missing in the
  strategy's series, so no decision is made on a bar holding it until its candle is
  read (2.5 C).
- **Clock skew** (6.4, `_clock_skew`): the banner's warning for a broker whose feed
  cannot be measured (and the result's, as every session warning); an ERROR alert, and
  a result warning once, when a window's estimate is beyond `SKEW_ALERT_S`. No lock.
- **The account's daily rules** (6.3; 18-multi-strategy 2.5a): in each loop step, after
  its ticks and bars, the sum of the strategies' `daily_loss_measure()` at or past
  `account_max_daily_loss` fires the kill switch (`_kill_switch`) of every strategy with
  the reason "account_loss", and at or past `account_daily_profit_target` with
  "account_profit", each once: entries locked for the day, each position exited, an
  ERROR alert (`_account_watch`).

Every lock and unlock is written to the event stream. Times are UTC nanoseconds, amounts
rupees. Used by the loop only.

**Not made yet.** 6.5's banner warning for a session whose only symbol is thinly traded:
telling a thin symbol needs a measure of its trading and a threshold that the design does
not set, so only a session of options only is warned.
"""

from __future__ import annotations

import bisect
import logging
from typing import Any

from .._layout import NANOS_PER_SECOND
from ._account_watch import AccountRules
from ._clock_skew import ClockSkew, skew_words, unmeasured_words
from ._kill_switch import KillSwitch
from ._session_trader import Trader
from ._types import LOGGER_NAME, AlertKind, Tick
from .bars import DEFAULT_DEAD_AFTER_S, FeedHealth
from .data_worker import QUOTE_FALLBACK_EVERY_S, QuoteRead
from .inbox import DataFailed, PositionRows, Quote

logger = logging.getLogger(LOGGER_NAME)

#: The entry lock of a dead feed (16-session-design 6.2).
FEED_DEAD_LOCK = "feed_dead"
#: Seconds without a tick after which an instrument with an open position is quiet and
#: gets REST quotes while the others tick: the feed's own dead time (6.5).
QUIET_AFTER_S = DEFAULT_DEAD_AFTER_S

_Span = tuple[int, int | None]


class RiskWatch:
    """The risk watch of `session`, a `_session_loop.Session` (module docstring). `kill`
    is its kill switch, `skew` its clock skew estimate."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self.skew = ClockSkew(session.parts.broker)
        self.kill = KillSwitch(session)
        self.account = AccountRules(session, self.kill)
        self._health: FeedHealth | None = None
        self._watching_ns = 0
        self._last_tick_ns: int | None = None
        # The dead spell under way (since the last tick before it), and those that ended.
        self._dead_since_ns: int | None = None
        self._spells: list[_Span] = []
        self._unlock_at_ns: int | None = None
        self._quotes: tuple[tuple[str, str], ...] = ()
        self._skew_warned = self._untimed_warned = False

    # ----- the start (16-session-design 3.1 steps 13 and 14) -----------------------------

    def warn_at_start(self) -> None:
        """Note the start banner's warnings of the watch: a broker whose feed cannot be
        measured for clock skew (6.4), and a session of options only, whose quiet market
        cannot be told from a dead feed (6.5)."""
        session = self._session
        note = self.skew.unmeasurable_note()
        if note is not None:
            session.warn(note)
        # Only an option carries an option type (`reference.instruments`).
        instruments = [part.setup.instruments[0] for part in session.parts.plan.strategies]
        if all(instrument.option_type is not None for instrument in instruments):
            session.warn(
                "every instrument of the session is an option: a quiet market cannot be told "
                "from a dead feed, so a long quiet spell locks entries and asks for REST quotes"
            )

    def start(self, watching_from_ns: int) -> None:
        """Start watching the feed at `watching_from_ns`, once the session's bar builders
        exist: the feed's health over the traded exchanges' sessions, the first skew
        window."""
        session = self._session
        bounds = [builder.session for builder in session.builders.values()]
        span = (min(opened for opened, _ in bounds), max(closed for _, closed in bounds))
        self._health = FeedHealth(session.now, session=span)
        self._watching_ns = watching_from_ns
        self.skew.start(watching_from_ns)

    # ----- what the loop takes in (2.5 A) -------------------------------------------------

    def tick(self, tick: Tick, arrived_ns: int) -> None:
        """A tick the loop took, the feed callback having received it at `arrived_ns`;
        called before the bar builder takes it, so a tick that ends a dead spell resumes
        the builders first."""
        if self._health is None:
            return
        if self._dead_since_ns is not None:
            self._back(self._dead_since_ns, tick.ts_recv_ns)
        self._health.on_tick(tick.symbol, tick.exchange, tick.ts_recv_ns)
        self._last_tick_ns = max(tick.ts_recv_ns, self._last_tick_ns or tick.ts_recv_ns)
        self.skew.sample(tick, arrived_ns)

    def quote(self, message: Quote) -> None:
        """A REST quote the watch asked for: a price for the kernel of every strategy that
        trades the instrument, in call order, and for the stops supervisor's price rule
        (A.11; 18-multi-strategy 4.3), through `StopsSupervisor.price`, the one way a
        price reaches the kernel. A quote no longer asked for (the feed came back
        meanwhile) is dropped: the ticks are newer."""
        session = self._session
        wanted = (message.symbol, message.exchange) in self._quotes
        if not wanted or message.ltp <= 0:
            return
        for trader in session.groups.of(message.symbol, message.exchange):
            if not trader.finished:
                session.stops.price(trader, float(message.ltp), int(message.read_ns))

    def data_failed(self, message: DataFailed) -> None:
        """A read of the data worker failed: logged; a quote still asked for is an ERROR
        alert too (6.5), which the alerts send at most once a minute."""
        request = message.request
        logger.warning("a data read failed (%r): %s", request, message.error)
        if isinstance(request, QuoteRead) and (request.symbol, request.exchange) in self._quotes:
            self._session.alert(
                AlertKind.ERROR,
                f"REST quotes of {request.symbol} {request.exchange} fail too while its ticks "
                f"are silent ({message.error}): the broker's login may have expired. Log in "
                "to the broker in OpenAlgo",
            )

    def positions(self, message: PositionRows) -> None:
        """A position-book read: for the kill switch."""
        self.kill.positions(message)

    # ----- the loop's step B: slices released --------------------------------------------

    def released(self, trader: Trader, count: int) -> None:
        """`trader` released its last `count` slices live: each with no bar while the feed
        was blind to it is marked missing (module docstring)."""
        part = trader.part
        series = part.series
        for opened, closed in part.slices[part.released - count : part.released]:
            if series.bar(opened) is not None or series.missing_slices(opened, closed):
                continue
            if self._blind(opened, closed):
                series.mark_missing(opened)
                self._session.event("missing", strategy=trader.name, slice_open_ns=opened)

    def _blind(self, opened_ns: int, closed_ns: int) -> bool:
        """True when a dead spell or an outage of the feed's connection overlaps the
        slice [opened_ns, closed_ns)."""
        spans = list(self._spells)
        if self._dead_since_ns is not None:
            spans.append((self._dead_since_ns, None))
        link = self._session.link
        if link is not None:
            spans += [(outage.start_ns, outage.end_ns) for outage in link.outages]
            if link.down_since_ns is not None:
                spans.append((link.down_since_ns, None))
        return any(start < closed_ns and (end is None or end >= opened_ns) for start, end in spans)

    # ----- the loop's step D ---------------------------------------------------------------

    def duties(self, now_ns: int) -> None:
        """Step D: the skew window, the feed dying, the lock lifting, the quotes asked
        for, the account's daily rules and the kill switch."""
        if self._health is None:
            return
        self._skew_due(now_ns)
        if self._dead_since_ns is None and self._health.feed_dead():
            self._went_dead()
        if self._unlock_at_ns is not None and now_ns >= self._unlock_at_ns:
            self._unlock()
        self._ask_quotes()
        if self._session.trading:
            self.account.duties()
        self.kill.duties(now_ns)

    def _went_dead(self) -> None:
        session = self._session
        since = self._watching_ns if self._last_tick_ns is None else self._last_tick_ns
        self._dead_since_ns, self._unlock_at_ns = since, None
        session.event("feed_dead", since_ns=since)
        for trader in session.traders:
            if trader.lock(FEED_DEAD_LOCK):
                session.event("lock", strategy=trader.name, reason=FEED_DEAD_LOCK)
        if session.data is not None:
            session.data.feed_dead()
        held = list(
            dict.fromkeys(f"{t.symbol} {t.exchange}" for t in session.traders if t.open_units())
        )
        quotes = (
            f"REST quotes of {', '.join(held)} every {QUOTE_FALLBACK_EVERY_S:g} s keep the "
            "targets and the daily loss watched"
            if held
            else "no position is open"
        )
        session.alert(
            AlertKind.FEED_DEAD,
            f"no tick on any instrument for {DEFAULT_DEAD_AFTER_S:g} s: entries are locked "
            f"until the feed has ticked for a whole minute again; {quotes}",
        )

    def _back(self, since: int, now_ns: int) -> None:
        """The dead spell since `since` ends with a tick received at `now_ns`."""
        session = self._session
        self._dead_since_ns = None
        for builder in session.builders.values():
            builder.resume(since, now_ns)
        self._spells.append((since, now_ns))
        if session.data is not None:
            session.data.feed_alive()
        self._unlock_at_ns = self._first_whole_close(now_ns)
        session.event("feed_back", since_ns=since, unlock_at_ns=self._unlock_at_ns)
        logger.info("the feed ticks again after %.1f s", (now_ns - since) / NANOS_PER_SECOND)

    def _first_whole_close(self, now_ns: int) -> int | None:
        """The close of the first slice opening after `now_ns`, the earliest across the
        exchanges; None when no slice opens after it. A slice opening at `now_ns` itself
        is one `resume` keeps for seeding only, with no bar, so it is not whole."""
        closes = []
        for builder in self._session.builders.values():
            slices = builder.slices
            index = bisect.bisect_right([opened for opened, _ in slices], now_ns)
            if index < len(slices):
                closes.append(slices[index][1])
        return min(closes, default=None)

    def _unlock(self) -> None:
        session = self._session
        self._unlock_at_ns = None
        for trader in session.traders:
            if trader.unlock(FEED_DEAD_LOCK):
                session.event("unlock", strategy=trader.name, reason=FEED_DEAD_LOCK)

    def _ask_quotes(self) -> None:
        """Ask the data worker for the quotes due: every instrument with an open position
        while the feed is dead, else those of them that are quiet."""
        session, health = self._session, self._health
        held = [
            (t.symbol, t.exchange)
            for t in session.traders
            if not t.finished and t.open_units() != 0
        ]
        if self._dead_since_ns is None and health is not None:
            held = [key for key in held if health.quiet(*key, QUIET_AFTER_S)]
        wanted = tuple(dict.fromkeys(held))
        if wanted == self._quotes or session.data is None:
            return
        self._quotes = wanted
        session.data.quote(wanted)
        session.event("quotes", instruments=[f"{symbol}:{exchange}" for symbol, exchange in wanted])

    def _skew_due(self, now_ns: int) -> None:
        session = self._session
        estimate = self.skew.close_due(now_ns)
        if self.skew.untimed and not self._untimed_warned:
            self._untimed_warned = True
            session.warn(unmeasured_words(self.skew.broker, untimed=True))
        if estimate is None:
            return
        session.event("clock_skew", seconds=estimate.seconds, samples=estimate.samples)
        if estimate.beyond_limit:
            text = skew_words(estimate)
            session.alert(AlertKind.ERROR, text)
            if not self._skew_warned:
                self._skew_warned = True
                session.warn(text, printed=True)


__all__ = ["FEED_DEAD_LOCK", "QUIET_AFTER_S", "RiskWatch"]
