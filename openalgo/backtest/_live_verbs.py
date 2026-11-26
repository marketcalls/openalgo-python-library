"""The live verbs: `bt.live`, which trades strategies through OpenAlgo for today, and
`bt.record`, which records the feed's ticks. Re-exported unchanged by `run`, where
`bt.backtest` and the other verbs live.

Sandbox or live is chosen inside OpenAlgo, never here. Both import the live layer when
they run, not before, so importing this module loads none of it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .reference._store import run_wording
from .session_result import SessionResult
from .strategy import Strategy
from .units import Distance

if TYPE_CHECKING:
    from ._live._recorder_summary import RecordSummary


def live(
    strategy: type[Strategy] | Sequence[type[Strategy]],
    client: Any,
    *,
    capital: float | Mapping[str, float],
    max_daily_loss: float | Distance | Mapping[str, float | Distance],
    account_max_daily_loss: float | None = None,
    alerts_to: str | None = None,
    account: str | None = None,
    params: Mapping[str, Any] | None = None,
    costs: Any = "standard",
    risk_basis: str = "gross",
    after_risk_exit: str | Mapping[str, str] = "lock_session",
    entry_from: str | Mapping[str, str | None] | None = None,
    entry_until: str | Mapping[str, str | None] | None = None,
    square_off: str | Mapping[str, str | None] | None = None,
    max_positions: int | Mapping[str, int | None] | None = None,
    drawdown_halt: float | Distance | Mapping[str, float | Distance | None] | None = None,
    cooldown: int | Mapping[str, int | None] | None = None,
    round_levels: bool = True,
    strict_config: bool = True,
    live_dir: str | os.PathLike[str] | None = None,
    on_stop: str | None = None,
    max_trades_per_day: int | Mapping[str, int | None] | None = None,
    daily_profit_target: float | Distance | Mapping[str, float | Distance | None] | None = None,
    max_order_value: float | Distance | Mapping[str, float | Distance | None] | None = None,
    account_max_trades_per_day: int | None = None,
    account_daily_profit_target: float | Distance | None = None,
    max_orders_per_second: int = 8,
    positional_stops: str | Mapping[str, str] = "gtt",
    **not_live: Any,
) -> SessionResult:
    """Trade strategies through OpenAlgo for today, in the mode set inside OpenAlgo, and
    return the session's `SessionResult` when it ends.

    Sandbox or live is OpenAlgo's analyzer switch, never an argument: the session reads
    it at the start, shows it first in its start banner and in every alert, and adopts
    it; if the switch changes during the session, nothing more is sent. The same strategy
    class runs here as in `bt.backtest`, deciding in `on_bar` on 1-minute bars built from
    the feed's ticks, in the strategy's product: a stop given with `stop=` is held by the
    broker, as an order for MIS and as a GTT for a CNC or NRML position, which the session
    carries to the next day; targets are held by the session on ticks. A daily or longer
    strategy (CNC or NRML) decides once, when the session starts, on the broker's candle
    of its interval that closed last, and its orders go at the open (at once, flagged
    "late_decision", when the session starts after the open); the journal keeps the bar it
    decided, so no start decides it twice, and bars of days no session ran feed its
    series only.

    strategy: a `bt.Strategy` subclass, or a list of them; two may trade one symbol, one
        side at a time and with market orders only.
    client: an `openalgo.api` client. The session makes clients of its own from its
        settings and leaves it, and its callbacks, alone.
    capital: rupees allocated to each strategy (a risk limit): one number for every
        strategy, or a dict by strategy class name. Mandatory.
    max_daily_loss: each strategy's daily loss limit in rupees (or `bt.rupees(...)`), on
        `risk_basis`, as in `bt.backtest`; one value or a dict by class name. Mandatory: a
        class attribute of the same name must agree with it.
    account_max_daily_loss: rupees for all strategies together; at it every strategy is
        flattened and its entries locked for the day.
    alerts_to: the OpenAlgo username linked to its Telegram bot; None sends alerts to the
        log and the console only (refused in LIVE mode, a warning in SANDBOX mode).
    account: the name the journal, logs and lock are kept under; the broker's name by
        default (give one per account when one broker has two).
    params, costs, risk_basis, round_levels, strict_config: as in `bt.backtest`, with the
        same defaults. `costs` models the charges of the broker's fills, since its trade
        book carries none.
    after_risk_exit, entry_from, entry_until, square_off, max_positions, drawdown_halt,
        cooldown: each strategy's, as in `bt.backtest` given a list, with the same
        defaults: one value for every strategy, or a dict naming each by class name; a
        class attribute of the same name wins. None for a strategy in a dict leaves it
        without `max_positions`, `drawdown_halt` or `cooldown`.
    live_dir: the absolute folder of the accounts' journals, logs and locks (default: the
        OPENALGO_LIVE_DIR environment variable, else ~/.openalgo/live).
    on_stop: what Ctrl+C does with open positions: None (the default) exits the MIS
        positions and keeps the CNC and NRML ones with their GTT stops at the broker (an
        MIS-only session exits every position); "flatten" exits every position, CNC and
        NRML too, then stops; "keep" leaves every position with its stop at the broker.
        The start banner says which applies.
    max_trades_per_day, daily_profit_target, max_order_value: each strategy's daily
        limits, as in `bt.backtest` (one value or a dict naming each by class name; a class
        attribute wins), each off when None: None for a strategy in a dict leaves it
        without that limit. The daily profit target is held on ticks, as the daily loss
        limit is: its exits go at once and entries stay locked for the day.
    account_max_trades_per_day, account_daily_profit_target: the account's, for all
        strategies together: at the trade cap every strategy's entries are refused; at the
        profit target the kill switch exits every strategy and locks its entries for the
        day, as at `account_max_daily_loss`. A restart of the day keeps the counts and the
        locks.
    max_orders_per_second: the most order calls (place, modify and cancel together) the
        session makes in any second, 8 by default (OpenAlgo's limit of 10 less 2 for your
        own orders on the same server; give more for a server set higher). A call beyond it
        waits, stops and exits first; an entry or add that waits 5 s is not sent.
    positional_stops: how a CNC or NRML position's stop is held, as in `bt.backtest` (one
        value or a dict by class name). "gtt" (the default): a SINGLE GTT at the broker,
        kept overnight; a broker without GTT orders refuses the start in LIVE mode.
        "rearm_daily": a day stop order, as an MIS position's, placed at each session's
        open (or the position exited at market, flagged "stop_crossed", when the open is
        already beyond the stop) and ended by the close: nothing protects the position at
        the broker from the close until the next start, which the start banner, every
        close's WARN alert, the daily summary and the result's warnings say.

    The session ends at the end of the day (the MIS legs flat after their square-off, and
    the CNC and NRML legs carried with their stop GTTs confirmed after the close), on Ctrl+C
    (a safe stop: a second Ctrl+C stops within seconds, naming each position left open),
    or, once the analyzer switch changed, on Ctrl+C or a few minutes after the exchange's
    close. A `bt.backtest` argument with no meaning live (`data`, `start`, `end`, `now`,
    `source`, `base`, `fill`, `slippage`, `margin`, `quantity_mode`, `instrument`,
    `product`, `roll`, `benchmark`, `calendar`, `pool`, `account_capital`) is refused,
    saying why.

    Raises ConfigError for an unusable argument (one that chooses the mode among them);
    NotImplementedError, before any call to OpenAlgo, for what live trading does not run
    yet (condition strategies, signal and array strategies, several symbols or a
    universe, futures series, a daily or longer interval in MIS, two strategies with one
    name, two CNC or NRML strategies on one symbol and product, bars from broker
    candles), saying what to do instead;
    PreflightError, listing every check that failed (another session holding the
    account or the server among them); LiveError when OpenAlgo does not report its mode,
    the call is not on the main thread (Ctrl+C reaches only the main thread), the feed
    does not connect, or a restart cannot resume today's positions; JournalError when the
    journal cannot be used; and the strategy's own exception, after the session has
    stopped safely.
    """
    arguments = {
        name: value
        for name, value in locals().items()
        if name not in ("strategy", "client", "not_live")
    }
    from ._live import session as live_session
    from ._live._plan_args import live_arguments
    from ._live._result_build import session_result

    # Checked here first, so that no argument of the session's own reaches it from a call.
    # A run reads the built-in tables only: its reference words give the run's advice.
    with run_wording():
        live_arguments(strategy, **arguments, **not_live)
        outcome = live_session.run_session(strategy, client, **arguments, **not_live)
    return session_result(outcome)


def record(
    client: Any,
    instruments: Sequence[Mapping[str, str]],
    *,
    mode: str = "quote",
    cache_dir: str | os.PathLike[str] | None = None,
    until: Any = None,
) -> RecordSummary:
    """Record the ticks of `instruments` from OpenAlgo's WebSocket feed into the local
    cache, kept in the order they were received (`bt.read_ticks` reads a day of them
    back), and return the recording's `RecordSummary`.

    client: an `openalgo.api` client; the recording makes a feed client of its own from
        its settings, so a session or a callback on it is left alone. It can run in a
        process of its own beside a `bt.live` session.
    instruments: [{"exchange": "NSE", "symbol": "SBIN"}, ...].
    mode: the feed mode, "ltp", "quote" (the default) or "depth".
    cache_dir: an absolute folder (the package's cache folder by default): recorded
        ticks cannot be fetched again.
    until: when to stop: an IST time of today as "HH:MM", a datetime (naive is IST), or
        None for the latest close of the instruments' exchanges today. Ctrl+C stops it
        too; one recording covers one day.

    Raises ConfigError for an unusable argument (an `until` not after the start, or
    past the end of its IST day, or None once the day's sessions closed, among them),
    CalendarError when `until` is None on a day none of the instruments' exchanges
    trades, LiveError when the feed does not connect, another recording holds the cache,
    or ticks could not be written (the raw files are kept, and the next recording of the
    day finalizes them), and OSError when the cache cannot be written at the start. The
    return hint names `RecordSummary`, which loads with the live layer
    (`openalgo.backtest.session_result.RecordSummary`): it is for static type checkers.
    """
    from ._live._record_verb import record_day

    return record_day(client, instruments, mode=mode, cache_dir=cache_dir, until=until)


__all__ = ["live", "record"]
