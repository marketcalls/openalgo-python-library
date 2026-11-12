//! Timers: `self.at("HH:MM", fn)` and `self.every(period, fn)` (14-strategy-api 2.3).
//!
//! In a slice where a timer is due, step 2 runs in three parts (03-architecture 4.2):
//! the risk exits and strategy orders due at this open fill (`fill_due_orders`); then
//! the timers fire, which is `Pause::Timers` (`timers_pause`); then
//! `steps::step_open_fills` fills the orders the timers placed, at this same open, and
//! runs the time exits and the square-off. A slice with no timer due runs step 2 in one
//! pass, exactly as without timers. A timer pause carries no events: callbacks are
//! delivered in step 7 of the slice (SPEC-31).
//!
//! When a timer is due:
//!
//! - `At(minute)`: at that IST time in every session. It fires at the first slice of the
//!   session that opens at or after the time (a time before the session open fires at
//!   the first slice). A session with no slice opening at or after it is counted as
//!   missed, never carried into the next session.
//! - `Every(unit)`: at the close of the first session of each week, month, quarter or
//!   year (the calendar's first session of the period, not the first of the run). The
//!   first open at or after that close is the next session's first slice, so the timer
//!   fires there and its orders fill at that open: the same fill as an order decided at
//!   that close (11-portfolio-rebalancing 3). Daily and longer strategies only.
//!
//! Timers fire in registration order. They are registered before the first `advance`.

use crate::data::resample::CalendarUnit;
use crate::exec::InstrumentId;
use crate::model::{civil_from_days, days_from_civil, ts_from_ist, weekday, Day};

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::Step;
use super::kernel::{Engine, Phase};
use super::pause::{Pause, TimerId};

/// Minutes in a day: an `At` time is below this.
const MINUTES_PER_DAY: u32 = 24 * 60;
const MONTHS_PER_QUARTER: u32 = 3;

/// When a timer fires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimerSpec {
    /// Each session, at this IST time (minutes since midnight).
    At { minute_of_day: u32 },
    /// At the close of the first session of each period of this unit.
    Every(CalendarUnit),
}

/// How often a timer fired, and how many sessions an `At` timer missed (no slice of
/// the session opened at or after its time).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimerCount {
    pub timer: TimerId,
    pub fired: u32,
    pub missed: u32,
}

#[derive(Clone, Debug, PartialEq)]
struct TimerState {
    spec: TimerSpec,
    /// The session date the timer last fired in.
    fired_day: Option<Day>,
    fired: u32,
    missed: u32,
}

/// The timers of a run and the sessions seen so far.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct TimerBook {
    timers: Vec<TimerState>,
    /// Session dates of the slices seen, most recent last: (two back, previous, current).
    sessions: (Option<Day>, Option<Day>, Option<Day>),
}

/// The first calendar date of the period of `unit` that holds `day`.
fn period_start(unit: CalendarUnit, day: Day) -> Day {
    let (year, month, _) = civil_from_days(day);
    match unit {
        CalendarUnit::Week => day - weekday(day) as Day,
        CalendarUnit::Month => days_from_civil(year, month, 1),
        CalendarUnit::Quarter => {
            let first = (month - 1) / MONTHS_PER_QUARTER * MONTHS_PER_QUARTER + 1;
            days_from_civil(year, first, 1)
        }
        CalendarUnit::Year => days_from_civil(year, 1, 1),
    }
}

impl Engine {
    /// Register a timer before the run starts; returns its id (ids are dense, in
    /// registration order).
    ///
    /// Errors: `InvalidConfig` (field `timer`) once the run has started, for an `At`
    /// time outside the day, and for `Every` on an intraday strategy interval.
    pub fn add_timer(&mut self, spec: TimerSpec) -> Result<TimerId, KernelError> {
        let refuse = |reason: &str| KernelError::InvalidConfig {
            field: "timer",
            reason: reason.to_string(),
        };
        if self.phase != Phase::Ready(0) {
            return Err(refuse(
                "timers are registered in init(), before the run starts",
            ));
        }
        match spec {
            TimerSpec::At { minute_of_day } if minute_of_day >= MINUTES_PER_DAY => {
                return Err(refuse("a timer time is an IST time from 00:00 to 23:59"));
            }
            TimerSpec::Every(_) if self.config.strategy_interval.is_intraday() => {
                return Err(refuse(
                    "self.every is for daily or longer strategies; use self.at on an \
                     intraday interval",
                ));
            }
            _ => {}
        }
        let id = TimerId(u32::try_from(self.timers.timers.len()).unwrap_or(u32::MAX));
        self.timers.timers.push(TimerState {
            spec,
            fired_day: None,
            fired: 0,
            missed: 0,
        });
        Ok(id)
    }

    /// How often each timer fired and missed, in registration order.
    pub fn timer_counts(&self) -> Vec<TimerCount> {
        self.timers
            .timers
            .iter()
            .enumerate()
            .map(|(index, state)| TimerCount {
                timer: TimerId(u32::try_from(index).unwrap_or(u32::MAX)),
                fired: state.fired,
                missed: state.missed,
            })
            .collect()
    }

    /// Step 2 before a timer pause: the risk exits and the strategy orders due at this
    /// open fill (SPEC-35), as in `steps::step_open_fills`; an instrument with no bar
    /// keeps its orders queued (SPEC-06).
    pub(super) fn fill_due_orders(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        for index in 0..self.instruments.len() {
            let instrument = InstrumentId(index as u32);
            let Some(bar) = self.market.bar(instrument, slice.index) else {
                continue;
            };
            self.risk_exits_at_open(slice, instrument, bar.open)?;
            self.gtt_exits_at_open(slice, instrument, bar.open)?;
            for id in self.book.queued_for(instrument) {
                self.fill_queued(slice, id, bar.open)?;
            }
        }
        Ok(())
    }

    /// The timers due at or before this open, as a pause; `None` when none is due.
    /// Also counts the sessions an `At` timer missed.
    pub(super) fn timers_pause(&mut self, slice: &SliceInfo) -> Option<Pause> {
        if self.timers.timers.is_empty() {
            return None;
        }
        let new_session = self.timers.sessions.2 != Some(slice.day);
        if new_session {
            let (_, previous, current) = self.timers.sessions;
            self.timers.sessions = (previous, current, Some(slice.day));
        }
        let mut due = Vec::new();
        for index in 0..self.timers.timers.len() {
            let fires = match self.timers.timers[index].spec {
                TimerSpec::At { minute_of_day } => {
                    let state = &self.timers.timers[index];
                    state.fired_day != Some(slice.day)
                        && slice.open >= ts_from_ist(slice.day, minute_of_day)
                }
                TimerSpec::Every(unit) => new_session && self.previous_opened_period(unit),
            };
            let state = &mut self.timers.timers[index];
            if fires {
                state.fired_day = Some(slice.day);
                state.fired += 1;
                due.push(TimerId(u32::try_from(index).unwrap_or(u32::MAX)));
            }
            let at = matches!(state.spec, TimerSpec::At { .. });
            if at && slice.closes_session && state.fired_day != Some(slice.day) {
                state.missed += 1;
            }
        }
        (!due.is_empty()).then(|| Pause::Timers {
            stamp: Self::stamp(slice, Step::OpenFills),
            decided_at: slice.open,
            timers: due,
        })
    }

    /// True when the previous session of the run was the calendar's first session of
    /// its period of `unit`.
    fn previous_opened_period(&self, unit: CalendarUnit) -> bool {
        let (two_back, previous, _) = self.timers.sessions;
        let Some(previous) = previous else {
            return false;
        };
        let start = period_start(unit, previous);
        match two_back {
            // The grid holds every session of the calendar, so the session before
            // `previous` in the run is the calendar's.
            Some(before) => before < start,
            None => {
                let exchange = self.timeline.exchange();
                (start..previous).all(|day| self.calendar.session(exchange, day).is_none())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_period_starts_on_its_first_calendar_date() {
        // 2025-07-10 is a Thursday in the third quarter.
        let day = days_from_civil(2025, 7, 10);
        let rows = [
            (CalendarUnit::Week, days_from_civil(2025, 7, 7)),
            (CalendarUnit::Month, days_from_civil(2025, 7, 1)),
            (CalendarUnit::Quarter, days_from_civil(2025, 7, 1)),
            (CalendarUnit::Year, days_from_civil(2025, 1, 1)),
        ];
        for (unit, start) in rows {
            assert_eq!(period_start(unit, day), start, "{unit:?}");
        }
    }

    #[test]
    fn the_first_quarter_starts_in_january() {
        let day = days_from_civil(2025, 3, 31);
        assert_eq!(
            period_start(CalendarUnit::Quarter, day),
            days_from_civil(2025, 1, 1)
        );
    }
}
