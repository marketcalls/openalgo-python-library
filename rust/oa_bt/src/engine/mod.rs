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
//! | `Engine::modify(OrderId, OrderRequest)` | while paused | not built yet |
//! | `position`, `leg`, `order`, `working_orders`, `has_pending_entry`, `account`, `last_close`, `decision_instant` | any time | queries |
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
//! | 1 session open, reference events | `steps::open_slice`; intraday times resolved lazily by `session::session_times` | times yes; rolls and lot-size changes no |
//! | 2 fills at the open | `steps::step_open_fills` -> `fills::fill_queued` (eligible when the open is at or after `decided_at`, SPEC-35; a missing bar keeps the order queued, SPEC-06; entry window and margin checked at the fill bar), then `session::square_off_at_open` (the square-off timer, SPEC-05) | MARKET orders, square-off; timers and rolls no |
//! | 3 resting orders, leg levels | `steps::step_levels` -> `legs::test_leg_levels` -> `rms::LegState::test_bar`, exits through `fills::execute_fill` | leg levels yes (MIS); resting orders no |
//! | 4 ratchets | `steps::step_ratchets` -> `legs::ratchet_legs` -> `rms::LegState::ratchet` | yes |
//! | 5 mark | `steps::step_mark`: close, or last close and stale | yes |
//! | 6 aggregate rules | refused at `RunConfig::validate` until built | no |
//! | 7 callbacks and `on_bar` | `steps::callbacks_pause` -> `Pause::BarClose`; `submit::submit` | yes |
//! | 8 session close | `steps::close_slice`: expire DAY orders, close open MIS positions at the last close (reason `eod`, flagged); settlement and futures MTM no | partly |
//!
//! Every fill takes one path, `fills::execute_fill`: slippage (`market::slippage`),
//! position and tranches (`portfolio`), the ledger, the leg state (`rms::leg`), then the
//! records: the fill, `Filled` (when it has an order), the position event, the trade.
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
//! | `session` | Intraday times per session, the entry window, the square-off timer |
//! | `submit` | `submit`, `cancel`, `modify`, the submission checks, DAY validity and market closed |
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
//! intervals on an intraday or daily base, MARKET DAY orders (entry, add, reduce,
//! exit by fraction, `reverse=True`), the quantity, freeze, product, entry-window,
//! market-closed and reverse checks, engine-held leg rules on MIS (stop, targets,
//! every trail form, breakeven, `after_first_target`), the square-off, margin for cash
//! equity, zero slippage and the fixed slippage forms, the position history, the
//! equity series, trades and events. Everything else returns `KernelError::NotBuilt`
//! naming the feature.

pub mod clock;
pub mod config;
pub mod error;
pub mod event;
mod fills;
pub mod kernel;
mod legs;
pub mod market_data;
pub mod output;
pub mod pause;
mod reverse;
mod session;
pub mod signals;
mod steps;
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
