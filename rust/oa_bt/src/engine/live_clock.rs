//! The live clock (Phase 2, PRD 15-phase2-plan section 2): a session's grid is known in
//! advance from the calendar, but a slice can run only once it has closed and its bars
//! are in.
//!
//! - `Engine::new_session` builds a session before its bars: the grid of its range from
//!   the calendar, the bars already closed (none at the open), and an engine that holds.
//!   It ends at the end of its range, not at its last bar as a backtest does.
//! - `Engine::add_bar` hands over an instrument's base bar once its slice has closed; the
//!   instrument's strategy bars are rebuilt from its base bars by the backtest's
//!   resampler, so a strategy bar is the same aggregate in both.
//! - `Engine::release` lets `advance` run the slices before an index; `Engine::hold`
//!   makes any engine wait before the slices not released (`Pause::Waiting`).
//!
//! The adapter adds a slice's bars, then releases it. A slice with no bar for an
//! instrument (no trade in that minute) is released without one, as a backtest runs a
//! missing bar (SPEC-07). So a session given its bars slice by slice is the backtest of
//! those bars.

use crate::exec::InstrumentId;
use crate::market::calendar::format_ist;

use super::clock::Timeline;
use super::config::RunSetup;
use super::error::KernelError;
use super::kernel::{Engine, Phase};
use super::market_data::BaseBar;
use super::run_end::{Ending, RunEnd};

/// Where a session ends: the close of its range's last slice.
///
/// Errors: `InvalidConfig` naming `start` when the range holds no slice.
pub(super) fn range_end(timeline: &Timeline) -> Result<RunEnd, KernelError> {
    let Some(&close) = timeline.slice_closes().last() else {
        return Err(KernelError::InvalidConfig {
            field: "start",
            reason: "the session's range holds no slice of the calendar: pick a trading day"
                .to_string(),
        });
    };
    Ok(RunEnd {
        close,
        range_close: close,
        cause: None,
    })
}

impl Engine {
    /// Build a live session: the grid of `setup`'s range from the calendar, the bars of
    /// the slices already closed (empty at the open), an engine that holds before its
    /// first slice. It ends at the end of its range.
    ///
    /// Errors: those of `Engine::new`, and `InvalidConfig` naming `start` when the range
    /// holds no slice.
    pub fn new_session(setup: RunSetup) -> Result<Engine, KernelError> {
        let mut engine = Engine::assemble(setup, Ending::AtRangeEnd)?;
        engine.hold();
        Ok(engine)
    }

    /// Hold before every slice not released yet: from now on `advance` runs only the
    /// slices released with `release`, and returns `Pause::Waiting` before the first
    /// one that is not. A live session holds from the start (`new_session`); a backtest
    /// never holds. A paused slice still completes on the next `advance`.
    pub fn hold(&mut self) {
        let next = self.next_slice();
        self.released = Some(self.released.map_or(next, |released| released.max(next)));
    }

    /// Release the slices before index `slices` of the grid to an engine that holds
    /// (`hold`). Releasing fewer than already released changes nothing; an engine that
    /// does not hold runs every slice anyway.
    pub fn release(&mut self, slices: usize) {
        if let Some(released) = self.released.as_mut() {
            *released = (*released).max(slices);
        }
    }

    /// Hand over `instrument`'s base bar, once its slice has closed, to an engine that
    /// holds. The bar must open on a slice of the grid that has not run yet, after the
    /// instrument's last bar, and be a valid OHLC bar.
    ///
    /// Errors: `InvalidConfig` naming `bars` for an engine that does not hold (a
    /// backtest is given its bars when it is built) and naming `instrument` for an
    /// unknown instrument; `InvalidData` for a bar off the grid, for a slice that has
    /// run, out of order or not a valid OHLC bar; `Resample` when its strategy bars
    /// cannot be built.
    pub fn add_bar(&mut self, instrument: InstrumentId, bar: BaseBar) -> Result<(), KernelError> {
        if self.released.is_none() {
            return Err(KernelError::InvalidConfig {
                field: "bars",
                reason: "bars are added only to a live session (Engine::new_session); a \
                         backtest is given its bars when it is built"
                    .to_string(),
            });
        }
        let next = self.next_slice();
        let spec = self.spec(instrument)?.clone();
        let slot = match self.timeline.slice_opens().binary_search(&bar.ts_open) {
            Ok(slot) if slot >= next => slot,
            found => {
                let why = match found {
                    Ok(_) => "is for a slice that has already run",
                    Err(_) => "does not open on a slice of the session's grid",
                };
                return Err(KernelError::InvalidData {
                    symbol: spec.symbol,
                    reason: format!("the bar opening at {} {why}", format_ist(bar.ts_open)),
                });
            }
        };
        self.market.push_bar(&spec, instrument, slot, bar)?;
        let closes = self.timeline.slice_closes();
        self.market
            .rebuild_strategy_bars(&spec, instrument, closes, &self.calendar, &self.config)
    }

    /// Index of the next slice to run: the paused or timer slice completes first.
    fn next_slice(&self) -> usize {
        match self.phase {
            Phase::Ready(index) => index,
            Phase::Paused(slice) | Phase::Timers(slice) => slice.index + 1,
            Phase::Finished => self.timeline.len(),
        }
    }
}
