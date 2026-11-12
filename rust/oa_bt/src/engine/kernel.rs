//! `Engine`: the stepping state machine and its public API.
//!
//! The engine owns the run: configuration, calendar, instruments, bars, order book,
//! portfolio and leg states. `advance` runs the slice sequence to the next decision
//! point and returns a `Pause`; while paused, the driver calls `submit`, `cancel` and
//! the queries; `finish` returns the result once the grid is exhausted. The steps
//! themselves are in `steps`, order submission in `submit`, fills in `fills`.

use crate::charges::ChargeBook;
use crate::exec::{InstrumentId, InstrumentSpec, Order, OrderBook, OrderId};
use crate::market::calendar::Calendar;
use crate::model::{Exchange, Product, TsNanos};
use crate::portfolio::{Portfolio, Position, PositionKey};
use crate::rms::LegState;

use super::accounting::charge_book;
use super::clock::{SliceInfo, Timeline};
use super::config::{RunConfig, RunSetup};
use super::error::KernelError;
use super::event::{Event, EventStamp, Step};
use super::futures::FuturesDesk;
use super::gtt::GttDesk;
use super::market_data::MarketData;
use super::output::{FinalPosition, GrossPnl, Recorder, RunOutput};
use super::pause::Pause;
use super::run_end::{end_of_run, RunEnd};
use super::sale_credit::SessionBuys;
use super::session::SessionTimes;
use super::strategy_risk::StrategyRisk;
use super::timers::TimerBook;

/// Where the state machine stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Phase {
    /// The slice at this index runs next, from step 1.
    Ready(usize),
    /// Paused in step 7 of this slice; step 8 runs on the next `advance`.
    Paused(SliceInfo),
    /// Paused for timers in step 2 of this slice; the rest of the slice runs on the
    /// next `advance`.
    Timers(SliceInfo),
    /// The grid is exhausted.
    Finished,
}

/// The account as a strategy sees it, rupees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AccountView {
    /// Capital plus realized minus costs plus unrealized.
    pub equity_rupees: f64,
    /// What the account can trade with (`Portfolio::free_cash_rupees`): equity not
    /// blocked as margin, with delivery holdings at their cost and sale proceeds not yet
    /// credited left out.
    pub free_cash_rupees: f64,
    /// Gross P&L realized since the start.
    pub realized_rupees: f64,
    /// Open P&L at the last marks.
    pub unrealized_rupees: f64,
    /// Charges paid since the start.
    pub costs_rupees: f64,
    /// Margin blocked by open positions.
    pub margin_rupees: f64,
}

/// The kernel of one run.
#[derive(Clone, Debug)]
pub struct Engine {
    pub(super) config: RunConfig,
    pub(super) calendar: Calendar,
    pub(super) instruments: Vec<InstrumentSpec>,
    pub(super) timeline: Timeline,
    pub(super) market: MarketData,
    pub(super) book: OrderBook,
    pub(super) portfolio: Portfolio,
    /// Leg states of open positions, in the order the positions opened.
    pub(super) legs: Vec<(PositionKey, LegState)>,
    /// Last close seen per instrument, rupees.
    pub(super) last_close: Vec<Option<f64>>,
    pub(super) phase: Phase,
    /// Events not yet delivered with a pause.
    pub(super) undelivered: Vec<Event>,
    pub(super) record: Recorder,
    /// Intraday times of the most recent session asked for.
    pub(super) session: Option<SessionTimes>,
    /// Trading state and the bookkeeping of the strategy and session rules.
    pub(super) risk: StrategyRisk,
    /// Charges, contract notes and the market-table rows used.
    pub(super) charges: ChargeBook,
    /// GTT orders and the engine's exits after unfilled GTT children.
    pub(super) gtts: GttDesk,
    /// Timers registered with `add_timer`.
    pub(super) timers: TimerBook,
    /// Futures series, their switches and the settlement postings.
    pub(super) futures: FuturesDesk,
    /// Where the run ends (`run_end`).
    pub(super) run_end: RunEnd,
    /// The session's CNC buys, for the credit of a sale (`sale_credit`).
    pub(super) session_buys: SessionBuys,
}

/// The one exchange of a Phase 1 run.
fn single_exchange(instruments: &[InstrumentSpec]) -> Result<Exchange, KernelError> {
    let Some(first) = instruments.first() else {
        return Err(KernelError::InvalidConfig {
            field: "symbol",
            reason: "a run needs at least one instrument".to_string(),
        });
    };
    if instruments
        .iter()
        .any(|spec| spec.exchange != first.exchange)
    {
        return Err(KernelError::NotBuilt {
            feature: "a run across several exchanges",
            instead: "run the instruments of each exchange in a backtest of their own",
        });
    }
    Ok(first.exchange)
}

impl Engine {
    /// Build a run: validate the configuration, the instruments and the bars, build the
    /// grid and the strategy bars. The engine starts before the first slice.
    ///
    /// Errors: `InvalidConfig`, `InvalidData`, `Exec` (an unusable instrument),
    /// `Calendar`, `Resample`, `Portfolio` (capital), and `NotBuilt` for settings the
    /// kernel cannot honour yet.
    pub fn new(setup: RunSetup) -> Result<Engine, KernelError> {
        let RunSetup {
            config,
            calendar,
            instruments,
            bars,
        } = setup;
        config.validate()?;
        let exchange = single_exchange(&instruments)?;
        for spec in &instruments {
            spec.validate()?;
        }
        let mut timeline = Timeline::build(&calendar, exchange, &config)?;
        let market = MarketData::build(bars, &instruments, &timeline, &calendar, &config)?;
        let (slices, run_end) = end_of_run(&timeline, &market, instruments.len(), config.present)?;
        timeline.end_after(slices);
        let portfolio = Portfolio::new(config.capital_rupees)?;
        config
            .market
            .check_count(instruments.len())
            .map_err(|reason| KernelError::InvalidConfig {
                field: "market",
                reason,
            })?;
        Ok(Engine {
            charges: charge_book(&config.costs),
            gtts: GttDesk::default(),
            last_close: vec![None; instruments.len()],
            risk: StrategyRisk::new(&config.risk, config.capital_rupees, instruments.len()),
            config,
            calendar,
            instruments,
            timeline,
            market,
            book: OrderBook::default(),
            portfolio,
            legs: Vec::new(),
            phase: Phase::Ready(0),
            undelivered: Vec::new(),
            record: Recorder::default(),
            session: None,
            timers: TimerBook::default(),
            futures: FuturesDesk::default(),
            run_end,
            session_buys: SessionBuys::default(),
        })
    }

    /// Run the slice sequence to the next decision point.
    ///
    /// From a pause it first completes the paused slice (from a timer pause, the rest
    /// of step 2 onwards; from step 7, step 8). It returns `Pause::Timers` in step 2 of
    /// a slice where timers are due, `Pause::BarClose` at step 7 of a slice where a
    /// strategy bar closes (or events are waiting and `deliver_events` is set), and
    /// `Pause::Finished` once the grid is exhausted.
    ///
    /// Errors: `RunFinished` after `Pause::Finished`; any error of a step.
    pub fn advance(&mut self) -> Result<Pause, KernelError> {
        loop {
            let slice = match self.phase {
                Phase::Finished => return Err(KernelError::RunFinished),
                Phase::Paused(slice) => {
                    self.close_slice(&slice)?;
                    self.phase = Phase::Ready(slice.index + 1);
                    continue;
                }
                Phase::Timers(slice) => {
                    self.finish_open_slice(&slice)?;
                    slice
                }
                Phase::Ready(index) => {
                    let Some(slice) = self.timeline.slice(index) else {
                        self.phase = Phase::Finished;
                        let events = std::mem::take(&mut self.undelivered);
                        return Ok(Pause::Finished { events });
                    };
                    if let Some(pause) = self.open_slice(&slice)? {
                        self.phase = Phase::Timers(slice);
                        return Ok(pause);
                    }
                    slice
                }
            };
            if let Some(pause) = self.callbacks_pause(&slice) {
                self.phase = Phase::Paused(slice);
                return Ok(pause);
            }
            self.close_slice(&slice)?;
            self.phase = Phase::Ready(slice.index + 1);
        }
    }

    /// The decision instant of the current pause (UTC nanoseconds), `None` when the
    /// engine is not paused.
    pub fn decision_instant(&self) -> Option<TsNanos> {
        self.decision_point().map(|(_, decided_at, _)| decided_at)
    }

    /// The paused slice, the decision instant and the step of the current pause: the
    /// slice's close in step 7, its open at a timer pause in step 2.
    pub(super) fn decision_point(&self) -> Option<(SliceInfo, TsNanos, Step)> {
        match self.phase {
            Phase::Paused(slice) => Some((slice, slice.close, Step::Callbacks)),
            Phase::Timers(slice) => Some((slice, slice.open, Step::OpenFills)),
            _ => None,
        }
    }

    /// The instruments, in registration order.
    pub fn instruments(&self) -> &[InstrumentSpec] {
        &self.instruments
    }

    /// True once `advance` has returned `Pause::Finished`.
    pub fn is_finished(&self) -> bool {
        self.phase == Phase::Finished
    }

    /// The position of an instrument and product, if it was ever opened.
    pub fn position(&self, instrument: InstrumentId, product: Product) -> Option<&Position> {
        self.portfolio.position(PositionKey {
            instrument,
            product,
        })
    }

    /// The leg state (levels in force) of an open position.
    pub fn leg(&self, instrument: InstrumentId, product: Product) -> Option<&LegState> {
        let key = PositionKey {
            instrument,
            product,
        };
        self.legs
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, leg)| leg)
    }

    /// An order by id.
    pub fn order(&self, id: OrderId) -> Option<&Order> {
        self.book.get(id)
    }

    /// The orders that can still fill, in id order.
    pub fn working_orders(&self) -> Vec<OrderId> {
        self.book.working().map(|order| order.id).collect()
    }

    /// True when an order that would open or add is still working on the instrument
    /// (a queued entry counts as not flat for a condition strategy, SPEC-29).
    pub fn has_pending_entry(&self, instrument: InstrumentId) -> bool {
        self.book
            .working()
            .any(|order| order.request.instrument == instrument && !order.request.reduce_only)
    }

    /// The last close seen of an instrument, rupees.
    pub fn last_close(&self, instrument: InstrumentId) -> Option<f64> {
        self.last_close.get(instrument.index()).copied().flatten()
    }

    /// The account at the last marks.
    pub fn account(&self) -> AccountView {
        let account = &self.portfolio.account;
        AccountView {
            equity_rupees: self.portfolio.equity_rupees(),
            free_cash_rupees: self.portfolio.free_cash_rupees(),
            realized_rupees: account.realized_rupees,
            unrealized_rupees: self.portfolio.unrealized_rupees(),
            costs_rupees: account.costs_rupees,
            margin_rupees: account.margin_blocked_rupees,
        }
    }

    /// The result of the run.
    ///
    /// Errors: `RunNotFinished` before `advance` returned `Pause::Finished`.
    pub fn finish(self) -> Result<RunOutput, KernelError> {
        if self.phase != Phase::Finished {
            return Err(KernelError::RunNotFinished);
        }
        let final_positions = self
            .portfolio
            .positions()
            .iter()
            .map(|position| FinalPosition {
                instrument: position.key.instrument,
                product: position.key.product,
                quantity_units: position.quantity_units,
                average_price_rupees: (!position.is_flat())
                    .then_some(position.average_price_rupees),
                mark_rupees: position.mark_rupees,
                unrealized_rupees: position.unrealized_rupees(),
                stale: position.stale,
            })
            .collect();
        let realized = self.portfolio.account.realized_rupees;
        let unrealized = self.portfolio.unrealized_rupees();
        Ok(RunOutput {
            capital_rupees: self.config.capital_rupees,
            instruments: self.instruments,
            fills: self.record.fills,
            levels: self.record.levels,
            equity: self.record.equity,
            trades: self.record.trades.into_closed(),
            orders: self.book.orders().to_vec(),
            events: self.record.events,
            stale_marks: self.record.stale_marks,
            charges: self.charges.into_output(),
            futures: self.futures.into_output(),
            final_positions,
            gross_pnl: GrossPnl {
                realized_rupees: realized,
                unrealized_rupees: unrealized,
                total_rupees: realized + unrealized,
            },
            run_end: self.run_end,
        })
    }

    /// Record an event and queue it for delivery with the next pause.
    pub(super) fn emit(&mut self, event: Event) {
        self.record.events.push(event);
        self.undelivered.push(event);
    }

    /// The stamp of a step of a slice.
    pub(super) fn stamp(slice: &SliceInfo, step: Step) -> EventStamp {
        EventStamp {
            slice: u32::try_from(slice.index).unwrap_or(u32::MAX),
            slice_ts: slice.open,
            step,
        }
    }

    /// The instrument spec, or an error for an id the run does not know.
    pub(super) fn spec(&self, instrument: InstrumentId) -> Result<&InstrumentSpec, KernelError> {
        self.instruments
            .get(instrument.index())
            .ok_or_else(|| KernelError::InvalidConfig {
                field: "symbol",
                reason: format!("no instrument with index {}", instrument.0),
            })
    }

    /// Signed units held for a key (0 when never opened).
    pub(super) fn held_units(&self, key: PositionKey) -> f64 {
        self.portfolio
            .position(key)
            .map_or(0.0, |position| position.quantity_units)
    }
}
