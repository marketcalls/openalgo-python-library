"""The live layer of `bt.live` (Phase 2, PRD 15-phase2-plan): the adapter between the Rust
kernel and an OpenAlgo server, in the mode set inside OpenAlgo (D21).

Private: `import openalgo.backtest` does not import it, and `bt.live` and `bt.record`
import it when they run. The package is named `_live`, not `live`: importing a
subpackage binds it as an attribute of its parent, which would replace the `bt.live`
function.

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
| `worker` | The order worker: the one thread that sends orders at the order rate, reads books |
| `session` | A live session: the start and the restart, the loop, the safe stop, the close |
| `stops` | The stops supervisor: broker-held stops, exits that fail, protection, held words |
| `watch` | The risk watch: a dead or quiet feed, clock skew, the account's limits, kill switch |
| `reconcile` | The reconciler: the books compared leg by leg, safe corrections, the exit guard |
| `_result_build` | `bt.live`'s `SessionResult`: each strategy's `Result`, tables, warnings |
| `_record_verb` | What `bt.record` runs: the recorder on a client of its own, until a time |

Each row's module keeps its helpers in private modules named after it (`_reconcile_*`,
`_session_*`, `_feedlink_*` and so on), each under the package's 400-line rule; the order
rate is `_order_throttle` (for `worker`) and the account's daily rules `_account_watch` (for
`watch`). Five serve several rows: `_groups`, the strategies of a session that trade one
instrument and product and the entry gates set on them, `_group_rules`, the rules of such
a group against the broker's net position, and `_margin_reject`, a closing order rejected
on such an instrument (18-multi-strategy 4), `_locked_entries`, the strategies whose
entries are locked, which the loop writes and the order worker reads before an entry or
add, and `_broker_units`, the one reader of a leg's units at the broker by its product
(20-positional-live 7.1), which `watch`, `stops`, `reconcile` and the restart read. The
audit record of 18-multi-strategy 12 is `_version_id` and `_session_versions` (each
strategy's version on every order, its manifest in the journal, `_journal_versions`; the
compiled code of a class whose source cannot be read, `_version_code`; the one rule for
the values both hash, `_version_values`),
`_preflight_expiry` and `_preflight_calendar` (the master contract's expiry and OpenAlgo's
holidays against the tables), and `_fill_comparison` (each fill against a backtest of the
same bars, priced by `_session_slippage`); `_exit_reasons` names why an order closes a
position. The order worker's GTT parts (20-positional-live 3) are `_gtt_send` (each GTT
request sent journal first and confirmed, `_gtt_send_confirm`, by the GTT book,
`_gtt_confirm`, and by the order book, `_gtt_checks`) and `_gtt_follow` (the GTTs the
broker holds followed, what it takes in `_gtt_follow_intake`, each one's record
`_gtt_followed`, a fired GTT's child found, `_gtt_child`, each finding posted,
`_gtt_outcomes`), on `_gtt_calls` and `_gtt_reads`; a GTT's leg's units are
judged against its reference by `_gtt_units` (the session's own fills on a leg, settled or
not, are the order path's `_own_fills`); their values
are `_gtt_types` and the book's rows `_gtt_book`; `_gtt_flight` holds a leg in flight for
the reconciler while its GTT's fate is open. The loop's GTT parts (20-positional-live 2,
3; unit N3b) are `_venue_gtts` (each strategy's GTT map: the kernel's GTT commands as GTT
requests, each answer and finding as its kernel call), `_session_gtts` (what the session
does around each), `_venue_children` and `_order_children` (a fired GTT's child followed
as an order, in the venue map and on the order path), `_stops_gtt` (a CNC or NRML
leg's stop GTT as the stops supervisor sees it), `_reconcile_guard_gtt` (the exit guard's
rules for a leg's GTT calls) and `_session_restore_gtts` (the GTTs a same-day restart
takes back); the end of the day is
`_session_day_end`, the legs carried past the close and their day-close records
`_session_carry`, and the worker's GTT requests `_worker_gtts`. A CNC or NRML leg under
`positional_stops="rearm_daily"` (20-positional-live 6; unit N5b) is `_session_rearm`'s (its
warnings at every start and close), a future the exchange settles on the session's day
`_session_expiry`'s, and a daily or longer strategy's bars and pre-open decision
(20-positional-live 8) `_plan_daily`'s and `_session_pre_open`'s. OpenAlgo's product names
are `.._products`'.

Times are UTC nanoseconds, prices rupees and quantities units, never lots. No module
prints: they log to the `openalgo.backtest.live` logger, and a session's own output goes
through its alerts and summary.
"""
