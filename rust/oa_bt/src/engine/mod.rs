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
//! | `Engine::submit_target(TargetRequest)` | while paused | the order for the difference to a target position (`target`) |
//! | `Engine::submit_rebalance(TargetRequest)` | while paused | the same order as part of a rebalance (`rebalance`) |
//! | `position`, `leg`, `order`, `working_orders`, `gtts`, `has_pending_entry`, `account`, `last_close`, `decision_instant`, `trading_state`, `strategy_bars`, `timer_counts` | any time | queries |
//! | `Engine::add_timer(TimerSpec)` | before the first `advance` | register `self.at` or `self.every` |
//! | `Engine::finish()` | after `Pause::Finished` | the `RunOutput` |
//! | `pause::run(engine, driver)` | Rust drivers | the loop above, for a `Driver` |
//!
//! A pause is a decision point: `Pause::BarClose` in step 7 of a slice where a strategy
//! bar closes by the clock (or where events wait and the strategy has callbacks), and
//! `Pause::Timers` in step 2 for `self.at` and `self.every` (`timers`; orders it
//! receives fill at that same open). A step 7 pause carries the events since the
//! previous one, so callbacks see everything of steps 2 to 6 before `on_bar` runs; a
//! timer pause carries none (SPEC-31). Events of step 8 and of submissions are
//! delivered with the next step 7 pause.
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
//! | 1 session open, reference events | `steps::open_slice`; intraday times resolved lazily by `session::session_times`; GTTs older than 360 days re-created (`gtt::renew_old_gtts`, GTT-2); futures switches are planned before the run (`futures`) | times, GTTs and rolls yes; lot-size change events no |
//! | 2 fills at the open | `steps::step_open_fills`: per instrument the risk exits of step 6 (`risk_exits_at_open`) and the engine's exits after unfilled GTT children (`gtt_fire::gtt_exits_at_open`), then `fills::fill_queued` (eligible when the open is at or after `decided_at`, SPEC-35; a missing bar keeps the order queued, SPEC-06; entry window and margin checked at the fill bar), the time exit, then `session::square_off_at_open` (the square-off timer, SPEC-05); after every instrument, `rebalance::fill_rebalance_buys` (the buys of a rebalance, after its sales), then `roll::roll_due_futures` (the futures auto-roll, near leg first, SPEC-19, GTT-9). In a slice where a timer is due, `timers::fill_due_orders` runs the engine exits and queued orders first, `timers::timers_pause` returns `Pause::Timers`, and `step_open_fills` then fills what the timers placed | MARKET orders, converted stop-limit exits, timers, square-off, rolls |
//! | 3 resting orders, leg levels | `steps::step_levels` -> `legs::test_leg_levels` -> `rms::LegState::test_bar` (MIS), `gtt_fire::test_gtts` (CNC and NRML levels and GTT orders), `resting::test_resting_orders` (LIMIT, SL, SL-M), exits through `fills::execute_fill` | yes |
//! | 4 ratchets | `steps::step_ratchets` -> `legs::ratchet_legs` -> `rms::LegState::ratchet`, then `gtt::sync_gtts` (GTTs follow the levels, GTT-3) | yes |
//! | 5 mark | `steps::step_mark`: close, or last close and stale | yes |
//! | 6 aggregate rules | `aggregate::step_aggregate_rules`: daily loss limit, drawdown halt; their exits at the next open (`risk_exits_at_open`, step 2) or the last close; the session boundary in step 1 (`risk_session_open`) | yes (combined rules: Phase 3) |
//! | 7 callbacks and `on_bar` | `steps::callbacks_pause` -> `Pause::BarClose`; `submit::submit` | yes |
//! | 8 session close | `steps::close_slice`: expire DAY orders, close open MIS positions at the last close (reason `eod`, flagged), settle expiring futures and post the futures mark to market (`settlement::settle_futures`), settle the day's contract notes (`accounting::settle_contract_notes`), credit the day's withheld sale proceeds (`sale_credit::credit_withheld_sales`) | yes (options expiry: Phase 3) |
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
//! | `run_end` | Where the run ends: the end of its range, its last bar or the present |
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
//! | `sale_credit` | Delivery shares, paid in full: the free cash that counts them at cost, and the part of their sale credited only on the next trading day (`market::sale_credit`) |
//! | `futures` | Futures series: the series, the switches and their spreads, the roll session, the settlement records |
//! | `futures_plan` | Registering a series and planning its switches from the bars; settlement prices |
//! | `roll` | Step 2: the futures auto-roll, the level shift, the GTTs at a roll |
//! | `settlement` | Step 8: futures expiry settlement and the daily mark to market |
//! | `sizing` | What a sizer sees, cost estimates in its budget, the engine default |
//! | `target` | `submit_target`: the order that takes a position to a target (`target()`) |
//! | `rebalance` | `submit_rebalance`: target orders of a rebalance; its buys fill after the slice's sales |
//! | `initial_risk` | What a trade's opening fill would lose at its entry stop (R-multiples) |
//! | `session` | Intraday times per session, the entry window, the square-off timer |
//! | `submit` | `submit`, `cancel`, the submission checks, DAY validity and market closed |
//! | `aggregate` | Step 6 (daily loss limit, drawdown halt), the step-1 session boundary, the exits they queue |
//! | `entry_rules` | Entry gates at submission (trading state, cooldown, maximum positions), the time exit, the fill hook |
//! | `scale_in` | The scale-in limits at submission: the facts of an add for `rms::scale_in_gate` |
//! | `strategy_risk` | `StrategyRisk`: trading state and the bookkeeping of the strategy and session rules |
//! | `pause` | `Pause`, `Driver`, `run` |
//! | `timers` | `self.at` and `self.every`: registration, when they are due, the timer pause |
//! | `history` | The strategy bars of the whole run, for indicators computed ahead |
//! | `signals` | Condition and array strategies: the Rust driver of 14-strategy-api 4.1 over the stepping API |
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
//! engine-held on MIS and held as GTTs on CNC and NRML, the square-off, timers
//! (`self.at`, `self.every`), charges on every fill with contract-note rounding, the
//! dated default slippage and margin tables (cash, MIS with leverage on request, long
//! options, futures at the default of D23 or the run's own `margin=`), delivery cash
//! (holdings at their cost, the dated credit for selling them), a CNC buy and
//! sell of one scrip on one day charged as intraday, the sizers (with a strategy's own
//! sizing function, `reinvest` and the size caps), the engine default size, target
//! positions, the initial risk of each trade, the position
//! history, the equity series, trades and events, the whole-run strategy bars for
//! indicators; the daily loss limit, the drawdown halt, the trading state, the
//! cooldown, maximum positions, the scale-in limits and the time exit; condition and
//! array strategies on one instrument (`signals`); futures series with the auto-roll
//! or settlement at expiry, and the daily futures mark to market (`futures`, `roll`,
//! `settlement`). Everything else returns `KernelError::NotBuilt` naming the feature.

mod accounting;
mod aggregate;
pub mod clock;
pub mod config;
mod entry_rules;
pub mod error;
pub mod event;
mod fills;
mod futures;
mod futures_plan;
mod gtt;
mod gtt_fire;
mod gtt_orders;
mod gtt_plan;
mod history;
mod initial_risk;
pub mod kernel;
mod legs;
pub mod market_data;
mod modify;
mod order_fill;
pub mod output;
pub mod pause;
mod rebalance;
mod resting;
mod reverse;
mod roll;
pub mod run_end;
mod sale_credit;
mod scale_in;
mod session;
mod settlement;
pub mod signals;
mod sizing;
mod steps;
mod strategy_risk;
pub mod submit;
mod target;
pub mod timers;

pub use clock::{SliceInfo, Timeline};
pub use config::{
    BarColumns, CostConfig, FillTiming, IntradayOverrides, QuantityMode, RunConfig, RunSetup,
    StrategyInterval,
};
pub use error::KernelError;
pub use event::{Event, EventStamp, OrderEventKind, Step};
pub use futures::{
    roll_session_day, session_after, FuturesOutput, FuturesSeries, RollRule, SeriesContract,
    SettlementKind, SettlementRecord, Switch,
};
pub use kernel::{AccountView, Engine};
pub use market_data::StrategyBar;
pub use output::{
    EquityColumns, FillRecord, FinalPosition, GrossPnl, LevelRow, RunOutput, StaleMark, TradeRecord,
};
pub use pause::{run, Driver, Pause, TimerId};
pub use run_end::{RunEnd, RunEndCause};
pub use signals::{run_signals, AtrInput, AtrSlot, EntrySignal, SignalArrays, SkippedEntry};
pub use submit::Submission;
pub use target::TargetRequest;
pub use timers::{TimerCount, TimerSpec};
