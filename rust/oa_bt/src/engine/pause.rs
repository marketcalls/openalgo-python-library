//! Decision points: where `Engine::advance` stops and hands control to the driver.
//!
//! A driver is whatever decides: the Python binding running an event strategy's
//! `on_bar` and callbacks, or a Rust driver running a condition or array strategy with
//! no Python call per bar. While the engine is paused the driver may submit, cancel and
//! query; the next `advance` resumes the slice where it stopped.

use crate::model::TsNanos;

use super::error::KernelError;
use super::event::{Event, EventStamp};
use super::kernel::Engine;
use super::market_data::StrategyBar;

/// Identifier of a timer registered with `self.at(...)` or `self.every(...)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TimerId(pub u32);

/// Why the engine stopped.
#[derive(Clone, Debug, PartialEq)]
pub enum Pause {
    /// Step 2 of a slice: timers due at or before this open fired. Orders submitted now
    /// fill at this same open (03-architecture 4.2 step 2). Not produced until timers
    /// are built.
    Timers {
        stamp: EventStamp,
        /// The decision instant: this slice's open.
        decided_at: TsNanos,
        timers: Vec<TimerId>,
        /// Events since the previous pause.
        events: Vec<Event>,
    },
    /// Step 7 of a slice: deliver the events of steps 2 to 6 (and anything since the
    /// previous pause) to the callbacks, then run `on_bar` for the strategy bars that
    /// close here. Orders submitted now are queued for the next open at or after
    /// `decided_at`. `bars` is empty when the pause is only for events, and a strategy
    /// bar with no base bar present is absent from it (SPEC-07).
    BarClose {
        stamp: EventStamp,
        /// The decision instant: the close of this slice's base bar.
        decided_at: TsNanos,
        /// True when a strategy bar closes at this slice by the clock.
        strategy_bar_closed: bool,
        bars: Vec<StrategyBar>,
        events: Vec<Event>,
    },
    /// The grid is exhausted. Positions still open are marked at their last close and
    /// reported as open; `Engine::finish` returns the result.
    Finished { events: Vec<Event> },
}

impl Pause {
    /// The events delivered with this pause.
    pub fn events(&self) -> &[Event] {
        match self {
            Pause::Timers { events, .. }
            | Pause::BarClose { events, .. }
            | Pause::Finished { events } => events,
        }
    }
}

/// A driver written in Rust: called at every pause.
pub trait Driver {
    /// Decide at a pause: submit, cancel or query through `engine`.
    ///
    /// Errors: whatever the engine returns for the calls the driver makes.
    fn on_pause(&mut self, engine: &mut Engine, pause: &Pause) -> Result<(), KernelError>;
}

/// Run a Rust driver from the current position of the engine to the end of the grid.
///
/// Errors: the first error of `advance` or of the driver.
pub fn run(engine: &mut Engine, driver: &mut impl Driver) -> Result<(), KernelError> {
    loop {
        let pause = engine.advance()?;
        driver.on_pause(engine, &pause)?;
        if matches!(pause, Pause::Finished { .. }) {
            return Ok(());
        }
    }
}
