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
    costs: Any = "zerodha",
    risk_basis: str = "gross",
    after_risk_exit: str = "lock_session",
    entry_from: str | None = None,
    entry_until: str | None = None,
    square_off: str | None = None,
    max_positions: int | None = None,
    drawdown_halt: float | Distance | None = None,
    cooldown: int | None = None,
    round_levels: bool = True,
    strict_config: bool = True,
    live_dir: str | os.PathLike[str] | None = None,
    on_stop: str = "flatten",
    **not_live: Any,
) -> SessionResult:
    """Trade strategies through OpenAlgo for today, in the mode set inside OpenAlgo, and
    return the session's `SessionResult` when it ends.

    Sandbox or live is OpenAlgo's analyzer switch, never an argument: the session reads
    it at the start, shows it first in its start banner and in every alert, and adopts
    it; if the switch changes during the session, nothing more is sent. The same strategy
    class runs here as in `bt.backtest`, deciding in `on_bar` on 1-minute bars built from
    the feed's ticks, with MIS orders; a stop given with `stop=` is held by the broker.

    strategy: a `bt.Strategy` subclass, or a list of them; each trades its own symbol.
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
    params, costs, risk_basis, after_risk_exit, entry_from, entry_until, square_off,
        max_positions, drawdown_halt, cooldown, round_levels, strict_config: as in
        `bt.backtest`, with the same defaults. `costs` models the charges of the broker's
        fills, since its trade book carries none.
    live_dir: the absolute folder of the accounts' journals, logs and locks (default: the
        OPENALGO_LIVE_DIR environment variable, else ~/.openalgo/live).
    on_stop: what Ctrl+C does with open positions: "flatten" (exit them, then stop) or
        "keep" (leave them with their stops at the broker).

    The session ends at the end of the day (flat after the MIS square-off), on Ctrl+C
    (a safe stop: a second Ctrl+C stops within seconds, naming each position left open),
    or, once the analyzer switch changed, on Ctrl+C or a few minutes after the exchange's
    close. A `bt.backtest` argument with no meaning live (`data`, `start`, `end`, `now`,
    `source`, `base`, `fill`, `slippage`, `margin`, `quantity_mode`, `instrument`,
    `product`, `roll`, `benchmark`, `calendar`, `pool`, `account_capital`) is refused,
    saying why.

    Raises ConfigError for an unusable argument (one that chooses the mode among them);
    NotImplementedError, before any call to OpenAlgo, for what live trading does not run
    yet (condition strategies, signal and array strategies, several symbols or a
    universe, futures series, daily or longer intervals, CNC and NRML, two strategies
    with one name, bars from broker candles), saying what to do instead;
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
