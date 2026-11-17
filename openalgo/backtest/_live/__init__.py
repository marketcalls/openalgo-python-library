"""The live layer of `bt.live` (Phase 2, PRD 15-phase2-plan): the adapter between the Rust
kernel and an OpenAlgo server, in the mode set inside OpenAlgo (D21).

Private: `import openalgo.backtest` does not import it, and `bt.live` imports it when it
runs. The package is named `_live`, not `live`: importing a subpackage binds it as an
attribute of its parent, which would replace the `bt.live` function.

| Module | Responsibility |
|---|---|
| `_types` | The values passed between the modules, and the interfaces the order path uses |
| `journal` | The SQLite journal: intents before the HTTP call, bindings, fills, stops, state |
| `guard` | The mode guard: the mode at start, on every response and update, polled |
| `bars` | Bars built from ticks and stamped at their close, the broker-candle check, feed health |
| `orders` | The order path: journal first, send once, resolve from the stream or by polling |
| `preflight` | The checks before a session, the account lock, the countdown |
| `alerts` | Alerts through OpenAlgo's Telegram, the daily summary, the session logs |
| `ticks` | The tick adapter: one feed message as a `Tick`, the last trade time read by one rule |
| `recorder` | The tick recorder behind `bt.record`: record, finalize, and the verified reads |
| `clients` | The session's five clients derived from the user's, each with its own timeout |
| `_shared_alerts` | One `AlertsLike` shared by every thread of a session |
| `inbox` | The messages workers and the feed pass to the loop, and the inbox that carries them |
| `feedlink` | The feed link: connect, subscribe, queue ticks and order updates, outages |
| `series` | A strategy's live series: base bars that grow by the slice, stores, views, indicators |
| `data_worker` | Quotes and broker candles read on their own thread, under the server's limits |
| `plan` | The session plan: `bt.live`'s arguments checked, each strategy set up, the catch-up |
| `venue_map` | The kernel's commands as order requests, and each answer as its kernel call |
| `worker` | The order worker: the one thread that sends the session's orders and reads its books |
| `session` | A live session from start to stop: the start, the loop, the safe stop, the close |
| `stops` | The stops supervisor: broker-held stops, exits that fail, protection, held words |
| `watch` | The risk watch: a dead or quiet feed, clock skew, the account loss and the kill switch |
| `reconcile` | The reconciler: the broker's books against the engine, the safe corrections |

A restart (`_restart_books`, `_session_restore`, `_session_saved`) resumes the day's session
from journal L's saved state, checked against the journal and the broker's books.

Times are UTC nanoseconds, prices rupees and quantities units, never lots. No module
prints: they log to the `openalgo.backtest.live` logger, and a session's own output goes
through its alerts and summary.
"""
