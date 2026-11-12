//! Intraday session rules: the entry window and the square-off of MIS positions
//! (03-architecture 4.2, decisions SPEC-04, SPEC-05 and SPEC-32).
//!
//! The times of a session are the calendar's dated defaults
//! (`market::calendar::default_intraday_times`) unless the run overrides them; they are
//! resolved on first use for each session date. The square-off is a timer: open MIS
//! positions exit at the open of the first base bar at or after `square_off`, reason
//! `scheduler`; one still open at the session close is closed in step 8 at its last
//! close (reason `eod`, flagged).

use crate::exec::checks::EntryWindow;
use crate::exec::InstrumentId;
use crate::market::calendar::default_intraday_times;
use crate::model::{ts_from_ist, Day, Product, TsNanos};
use crate::portfolio::PositionKey;
use crate::rms::ExitReason;

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::Step;
use super::fills::FillRequest;
use super::kernel::Engine;

/// The intraday times of one session, UTC nanoseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SessionTimes {
    /// The IST session date the times belong to.
    pub(super) day: Day,
    /// The MIS entry window.
    pub(super) entry: EntryWindow,
    /// The MIS square-off instant.
    pub(super) square_off: TsNanos,
}

impl Engine {
    /// The square-off timer (SPEC-05): open intraday positions of the instrument exit
    /// at the first base-bar open at or after `square_off`, reason `scheduler`.
    pub(super) fn square_off_at_open(
        &mut self,
        slice: &SliceInfo,
        instrument: InstrumentId,
        open: f64,
    ) -> Result<(), KernelError> {
        let key = PositionKey {
            instrument,
            product: Product::Mis,
        };
        let held = self.held_units(key);
        if held == 0.0 || slice.open < self.session_times(slice.day)?.square_off {
            return Ok(());
        }
        let request = FillRequest::engine_exit(key, held, open, ExitReason::Scheduler);
        self.execute_fill(Self::stamp(slice, Step::OpenFills), request)?;
        Ok(())
    }

    /// The intraday times of the session on `day` (defaults from the calendar, then the
    /// run's overrides).
    ///
    /// Errors: `InvalidConfig` when the exchange has no session that day, or no default
    /// time and no override.
    pub(super) fn session_times(&mut self, day: Day) -> Result<SessionTimes, KernelError> {
        if let Some(times) = self.session.filter(|times| times.day == day) {
            return Ok(times);
        }
        let exchange = self.timeline.exchange();
        let missing = |field: &'static str, what: &str| KernelError::InvalidConfig {
            field,
            reason: format!("{} on day {day}: {what}", exchange.code()),
        };
        let session = self
            .calendar
            .session(exchange, day)
            .ok_or_else(|| missing("calendar", "no session"))?;
        let defaults = default_intraday_times(&self.calendar, exchange, day);
        let overrides = self.config.intraday;
        let at = |minute: u32| ts_from_ist(day, minute);
        let advice = "no default intraday time: pass entry_until and square_off";
        let times = SessionTimes {
            day,
            entry: EntryWindow {
                from: overrides.entry_from_minute.map_or(session.open, at),
                until: overrides
                    .entry_until_minute
                    .map(at)
                    .or(defaults.map(|d| d.entry_until))
                    .ok_or_else(|| missing("entry_until", advice))?,
            },
            square_off: overrides
                .square_off_minute
                .map(at)
                .or(defaults.map(|d| d.square_off))
                .ok_or_else(|| missing("square_off", advice))?,
        };
        self.session = Some(times);
        Ok(times)
    }
}
