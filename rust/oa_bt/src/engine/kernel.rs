//! `Engine`: the stepping state machine and its public API.
//!
//! The engine owns the run: configuration, calendar, instruments, bars, order book,
//! portfolio and leg states. `advance` runs the slice sequence to the next decision
//! point and returns a `Pause`; while paused, the driver calls `submit`, `cancel` and
//! the queries; `finish` returns the result once the grid is exhausted. The steps
//! themselves are in `steps`, order submission in `submit`, fills in `fills`.

use std::collections::BTreeMap;

use crate::charges::ChargeBook;
use crate::exec::{InstrumentId, InstrumentSpec, OrderBook, OrderId};
use crate::market::calendar::Calendar;
use crate::model::TsNanos;
use crate::portfolio::{Portfolio, PositionKey};
use crate::rms::LegState;

use super::accounting::charge_book;
use super::clock::{single_exchange, SliceInfo, Timeline};
use super::config::{RunConfig, RunSetup};
use super::error::KernelError;
use super::event::{Event, EventStamp, Step};
use super::exit_scale::ExitScales;
use super::futures::FuturesDesk;
use super::gtt::GttDesk;
use super::held_changes::HeldChange;
use super::live_clock::range_end;
use super::market_data::MarketData;
use super::output::{settle_trade_charges, FinalPosition, GrossPnl, Recorder, RunOutput};
use super::pause::Pause;
use super::run_end::{end_of_run, Ending, RunEnd};
use super::sale_credit::SessionBuys;
use super::session::SessionTimes;
use super::strategy_risk::StrategyRisk;
use super::timers::TimerBook;
use super::venue::VenueDesk;

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
    /// A live session paused for the timers due at the open of this slice, which has not
    /// run (`venue_clock`); the next `advance` makes the engine's exits due at that open.
    Ahead(SliceInfo),
    /// A live session's close of this slice, waiting for the venue to end its orders
    /// (`venue_clock`); the rest of step 8 runs once they have.
    Closing(SliceInfo),
    /// The grid is exhausted.
    Finished,
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
    /// The slices `advance` may run, as the index of the first one it may not: `None`
    /// runs them all (a backtest), `Some` holds before that slice (`hold`, `release`).
    pub(super) released: Option<usize>,
    /// Who fills the orders, and the commands for an external venue (`venue`).
    pub(super) venue: VenueDesk,
    /// Orders held for a delay, by id: the instant each may fill from (`delay`).
    pub(super) held: BTreeMap<OrderId, TsNanos>,
    /// The strategy's cancellations and modifications held for a delay, in the order
    /// they were decided (`held_changes`).
    pub(super) held_changes: Vec<HeldChange>,
    /// The positions a fill model opened in this slice, whose ratchets see only the fill
    /// bar's close (`fill_model`).
    pub(super) model_opened: Vec<PositionKey>,
    /// The factors of the trades' leg rules (`exit_scale`).
    pub(super) exit_scales: ExitScales,
}

impl Engine {
    /// Build a run: validate the configuration, the instruments and the bars, build the
    /// grid and the strategy bars. The engine starts before the first slice.
    ///
    /// Errors: `InvalidConfig`, `InvalidData`, `Exec` (an unusable instrument),
    /// `Calendar`, `Resample`, `Portfolio` (capital), and `NotBuilt` for settings the
    /// kernel cannot honour yet.
    pub fn new(setup: RunSetup) -> Result<Engine, KernelError> {
        Engine::assemble(setup, Ending::AtLastBar)
    }

    /// Build a run that ends as `ending` says (`new` for a backtest, `new_session` for
    /// a live session); the errors of `new`.
    pub(super) fn assemble(setup: RunSetup, ending: Ending) -> Result<Engine, KernelError> {
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
        let run_end = match ending {
            Ending::AtLastBar => {
                let (slices, run_end) =
                    end_of_run(&timeline, &market, instruments.len(), config.present)?;
                timeline.end_after(slices);
                run_end
            }
            Ending::AtRangeEnd => range_end(&timeline)?,
        };
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
            released: None,
            venue: VenueDesk::default(),
            held: BTreeMap::new(),
            held_changes: Vec::new(),
            model_opened: Vec::new(),
            exit_scales: ExitScales::default(),
        })
    }

    /// Run the slice sequence to the next decision point.
    ///
    /// From a pause it first completes the paused slice (from a timer pause, the rest
    /// of step 2 onwards; from step 7, step 8). It returns `Pause::Timers` in step 2 of
    /// a slice where timers are due, `Pause::BarClose` at step 7 of a slice where a
    /// strategy bar closes (or events are waiting and `deliver_events` is set), and
    /// `Pause::Finished` once the grid is exhausted, and `Pause::Waiting` before a slice
    /// not released to an engine that holds (`hold`).
    ///
    /// A live session also returns `Pause::Timers` at the close of the slice before the
    /// open the timers are due at, and `Pause::Waiting` at a session close until the
    /// venue has ended its orders (`venue_clock`).
    ///
    /// Errors: `RunFinished` after `Pause::Finished`; any error of a step.
    pub fn advance(&mut self) -> Result<Pause, KernelError> {
        loop {
            let slice = match self.phase {
                Phase::Finished => return Err(KernelError::RunFinished),
                Phase::Paused(slice) => match self.end_slice(&slice)? {
                    Some(pause) => return Ok(pause),
                    None => continue,
                },
                Phase::Closing(slice) => match self.wait_for_close(&slice)? {
                    Some(pause) => return Ok(pause),
                    None => continue,
                },
                Phase::Ahead(next) => {
                    self.resume_ahead(&next)?;
                    continue;
                }
                Phase::Timers(slice) => {
                    self.finish_open_slice(&slice)?;
                    slice
                }
                Phase::Ready(index) => {
                    let held = self.released.is_some_and(|released| index >= released);
                    if held && index < self.timeline.len() {
                        return Ok(Pause::Waiting { next_slice: index });
                    }
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
            if let Some(pause) = self.end_slice(&slice)? {
                return Ok(pause);
            }
        }
    }

    /// The paused slice, the decision instant and the step of the current pause: the
    /// slice's close in step 7, its open at a timer pause in step 2 (the open of the next
    /// slice for timers a live session fires ahead of it).
    pub(super) fn decision_point(&self) -> Option<(SliceInfo, TsNanos, Step)> {
        match self.phase {
            Phase::Paused(slice) => Some((slice, slice.close, Step::Callbacks)),
            Phase::Timers(slice) | Phase::Ahead(slice) => {
                Some((slice, slice.open, Step::OpenFills))
            }
            _ => None,
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
        let charges = self.charges.into_output();
        let mut fills = self.record.fills;
        let mut trades = self.record.trades.into_closed();
        settle_trade_charges(&mut fills, &mut trades, &charges);
        Ok(RunOutput {
            capital_rupees: self.config.capital_rupees,
            instruments: self.instruments,
            fills,
            levels: self.record.levels,
            equity: self.record.equity,
            trades,
            orders: self.book.orders().to_vec(),
            events: self.record.events,
            stale_marks: self.record.stale_marks,
            charges,
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
