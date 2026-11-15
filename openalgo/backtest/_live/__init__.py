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

Times are UTC nanoseconds, prices rupees and quantities units, never lots. No module
prints: they log to the `openalgo.backtest.live` logger, and a session's own output goes
through its alerts and summary.
"""
