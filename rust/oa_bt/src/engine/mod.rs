//! The event kernel: one deterministic, single-threaded state machine that runs every
//! backtest (and, from Phase 2, live trading) with the same rules.
//!
//! # Data flow
//!
//! ```text
//!  Python (openalgo.backtest)                     Rust (oa_bt)
//!  -------------------------                      ------------
//!  bt.backtest(Strategy, ...)
//!    resolve settings (defaults, Assumptions)
//!    load bars (data layer), resolve instruments
//!    (reference tables), build the calendar   --> RunSetup { RunConfig, Calendar,
//!                                                   [InstrumentSpec], [BarColumns] }
//!                                                 Engine::new: validate, build the
//!                                                 grid (Timeline), align bars and build
//!                                                 strategy bars (MarketData)
//!    driver loop:                                 Engine::advance: run slices until a
//!      pause = engine.advance()               <--   decision point -> Pause
//!      callbacks(pause.events)
//!      on_bar(pause.bars) -> buy/sell/exit    --> Engine::submit / cancel (step 7)
//!      ... until Pause::Finished
//!    engine.finish()                          <-- RunOutput: fills, levels (position
//!    -> Result (DataFrames)                         history), equity, trades, orders,
//!                                                   events, stale marks, final positions
//! ```
//!
//! Python declares and decides; Rust executes. Event strategies (`on_bar` in Python)
//! are driven by the binding through the stepping API, one pause per strategy-bar
//! close. Condition and array strategies run a Rust driver (`signals`) over the same
//! API with no Python call per bar. Both paths reach the same `Engine` and the same
//! steps, so they cannot drift apart.
//!
//! # The stepping API
//!
//! | Call | When | What |
//! |---|---|---|
//! | `Engine::new(RunSetup)` | once | validate and prepare; the engine stands before slice 0 |
//! | `Engine::advance()` | until `Pause::Finished` | run steps to the next decision point |
//! | `Engine::submit(OrderRequest)` | while paused | submission checks, `Submitted` or `Rejected` |
//! | `Engine::cancel(OrderId)` | while paused | cancel a working order |
//! | `Engine::modify(OrderId, OrderRequest)` | while paused | a full replacement of a working order (`modify`) |
//! | `position`, `leg`, `order`, `working_orders`, `gtts`, `has_pending_entry`, `account`, `last_close`, `decision_instant`, `trading_state` | any time | queries |
//! | `Engine::finish()` | after `Pause::Finished` | the `RunOutput` |
//! | `pause::run(engine, driver)` | Rust drivers | the loop above, for a `Driver` |
//!
//! A pause is a decision point: `Pause::BarClose` in step 7 of a slice where a strategy
//! bar closes by the clock (or where events wait and the strategy has callbacks), and
//! `Pause::Timers` in step 2 for `self.at` and `self.every` (not built yet; orders it
//! receives fill at that same open). Each pause carries the events since the previous
//! one, so callbacks see everything of steps 2 to 6 before `on_bar` runs. Events of
//! step 8 and of submissions are delivered with the next pause.
//!
//! In live trading (Phase 2) the same machine runs with a live clock: the adapter feeds
//! bars or ticks before calling `advance`, and tick mode runs steps 3 to 6 for one
//! instrument per tick (03-architecture 4.2, last paragraph). The order and risk rules
//! do not change.
//!
//! # The eight steps of a slice (03-architecture 4.2)
//!
//! The grid is the calendar's base-interval lattice: every slot is a slice, with or
//! without bars (SPEC-07). Each step runs across all instruments in registration order
//! before the next step starts.
//!
//! | Step | Code | Built |
//! |---|---|---|
//! | 1 session open, reference events | `steps::open_slice`; intraday times resolved lazily by `session::session_times`; GTTs older than 360 days re-created (`gtt::renew_old_gtts`, GTT-2) | times and GTTs yes; rolls and lot-size changes no |
//! | 2 fills at the open | `steps::step_open_fills` -> `fills::fill_queued` (eligible when the open is at or after `decided_at`, SPEC-35; a missing bar keeps the order queued, SPEC-06; entry window and margin checked at the fill bar), the engine's exits after unfilled GTT children (`gtt_fire::gtt_exits_at_open`) before them, then `session::square_off_at_open` (the square-off timer, SPEC-05) | MARKET orders, converted stop-limit exits, square-off; timers and rolls no |
//! | 3 resting orders, leg levels | `steps::step_levels` -> `legs::test_leg_levels` -> `rms::LegState::test_bar` (MIS), `gtt_fire::test_gtts` (CNC and NRML levels and GTT orders), `resting::test_resting_orders` (LIMIT, SL, SL-M), exits through `fills::execute_fill` | yes |
//! | 4 ratchets | `steps::step_ratchets` -> `legs::ratchet_legs` -> `rms::LegState::ratchet`, then `gtt::sync_gtts` (GTTs follow the levels, GTT-3) | yes |
//! | 5 mark | `steps::step_mark`: close, or last close and stale | yes |
//! | 6 aggregate rules | `aggregate::step_aggregate_rules`: daily loss limit, drawdown halt; their exits at the next open (`risk_exits_at_open`, step 2) or the last close; the session boundary in step 1 (`risk_session_open`) | yes (combined rules: Phase 3) |
//! | 7 callbacks and `on_bar` | `steps::callbacks_pause` -> `Pause::BarClose`; `submit::submit` | yes |
//! | 8 session close | `steps::close_slice`: expire DAY orders, close open MIS positions at the last close (reason `eod`, flagged), settle the day's contract notes (`accounting::settle_contract_notes`); settlement and futures MTM no | partly |
//!
//! Every fill takes one path, `fills::execute_fill`: slippage (`market::slippage`, the
//! run's or the dated default table's), position and tranches (`portfolio`), the ledger,
//! the charges (`charges`, posted to the ledger; contract notes settle in step 8), the
//! leg state (`rms::leg`), then the records: the fill, `Filled` (when it has an order),
//! the position event, the trade. The glue to the market rules is in `accounting`, the
//! sizers' in `sizing`.
//!
//! # Determinism
//!
//! One thread; no clock, no randomness, no hash-ordered container. Instruments are
//! processed in registration order, orders in id order (ids are dense and in
//! submission order), legs in the order their positions opened. Prices are on the tick
//! grid (integer price units, `exec::price`) and ledger postings in paise
//! (`portfolio::ledger`), so the same inputs give bit-identical results.
//!
//! # Modules
//!
//! | Module | Responsibility |
//! |---|---|
//! | `config` | `RunConfig`, `RunSetup`, `BarColumns`, interval and mode enums; what is refused as not built |
//! | `clock` | `Timeline`: the grid, slice facts, strategy-bar closes by the clock |
//! | `market_data` | Base bars aligned to the grid; strategy bars via `data::resample` |
//! | `kernel` | `Engine`, `advance`, queries, `finish` |
//! | `steps` | Steps 1 to 8 |
//! | `fills` | Step 2 fills of queued orders, `execute_fill` |
//! | `legs` | Leg rules in steps 3 and 4, the add rule of SPEC-14 |
//! | `reverse` | `reverse=True`: close, then open the rest |
//! | `resting` | Step 3: resting LIMIT, SL and SL-M DAY orders, the stop-limit exit conversion |
//! | `modify` | `modify`: a full replacement of a working order |
//! | `order_fill` | Where and how an order fills; the price of any fill (slippage, limits) |
//! | `gtt` | GTTs holding CNC and NRML levels: placement, cancel-then-exit, step-4 sync, renewal |
//! | `gtt_plan` | Which GTTs a positional leg needs (GTT-11) and when one is modified (GTT-3) |
//! | `gtt_fire` | Step 3 for GTTs, the child fill, the engine's exit after an unfilled child |
//! | `gtt_orders` | Strategy GTT orders (`validity="gtt"`), the per-segment order shape |
//! | `accounting` | Slippage, margin and charges of a fill; contract notes at the session close |
//! | `sizing` | What a sizer sees, cost estimates in its budget, the engine default |
//! | `session` | Intraday times per session, the entry window, the square-off timer |
//! | `submit` | `submit`, `cancel`, the submission checks, DAY validity and market closed |
//! | `aggregate` | Step 6 (daily loss limit, drawdown halt), the step-1 session boundary, the exits they queue |
//! | `entry_rules` | Entry gates at submission (trading state, cooldown, maximum positions), the time exit, the fill hook |
//! | `strategy_risk` | `StrategyRisk`: trading state and the bookkeeping of the strategy and session rules |
//! | `pause` | `Pause`, `Driver`, `run` |
//! | `signals` | Condition and array strategies (not built yet) |
//! | `event` | `Event`, `EventStamp`, `Step` |
//! | `output` | `RunOutput` and its tables |
//! | `error` | `KernelError` |
//!
//! The areas below the engine: `exec` (orders, fills, checks, GTT), `portfolio`
//! (positions, ledger), `rms` (rules, leg state, trading state, vocabulary), `sizing`,
//! `market::{calendar, costs, margin, slippage}`, `data::resample`, `model`.
//!
//! # Phase 1 scope of this skeleton
//!
//! Built: one exchange, any number of instruments on it, intraday or daily strategy
//! intervals on an intraday or daily base, MARKET, LIMIT, SL and SL-M DAY orders
//! (entry, add, reduce, exit by fraction, `reverse=True`, modification), GTT orders,
//! the quantity, tick, freeze, product, entry-window, market-closed and reverse checks,
//! leg rules (stop, targets, every trail form, breakeven, `after_first_target`)
//! engine-held on MIS and held as GTTs on CNC and NRML, the square-off, charges on
//! every fill with contract-note rounding, the dated default slippage and margin
//! tables (cash, MIS with leverage on request, long options, NSE index futures), the
//! sizers and the engine default size, the position history, the equity series,
//! trades and events; the daily loss limit, the drawdown halt, the trading state, the
//! cooldown, maximum positions and the time exit. Everything else returns
//! `KernelError::NotBuilt` naming the feature.

mod accounting;
mod aggregate;
pub mod clock;
pub mod config;
mod entry_rules;
pub mod error;
pub mod event;
mod fills;
mod gtt;
mod gtt_fire;
mod gtt_orders;
mod gtt_plan;
pub mod kernel;
mod legs;
pub mod market_data;
mod modify;
mod order_fill;
pub mod output;
pub mod pause;
mod resting;
mod reverse;
mod session;
pub mod signals;
mod sizing;
mod steps;
mod strategy_risk;
pub mod submit;

pub use clock::{SliceInfo, Timeline};
pub use config::{
    BarColumns, CostConfig, FillTiming, IntradayOverrides, QuantityMode, RunConfig, RunSetup,
    StrategyInterval,
};
pub use error::KernelError;
pub use event::{Event, EventStamp, OrderEventKind, Step};
pub use kernel::{AccountView, Engine};
pub use market_data::StrategyBar;
pub use output::{
    EquityColumns, FillRecord, FinalPosition, GrossPnl, LevelRow, RunOutput, StaleMark, TradeRecord,
};
pub use pause::{run, Driver, Pause, TimerId};
pub use signals::{run_signals, SignalArrays};
pub use submit::Submission;
