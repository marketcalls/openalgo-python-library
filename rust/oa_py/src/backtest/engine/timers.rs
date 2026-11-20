//! Timers across the boundary (`oa_bt::engine::timers`): `self.at` and `self.every`
//! registered before the first `advance` (the parent module's `add_timer_at` and
//! `add_timer_every`), and how often each fired and was missed (`timer_counts`).

use oa_bt::data::resample::CalendarUnit;
use oa_bt::engine::{Engine, TimerCount, TimerSpec};
use pyo3::prelude::*;

use super::errors::{engine_error, NativeEngineInputError};

/// `add_timer_at`: a timer at an IST time each session (minutes since midnight); its id.
pub(super) fn add_at(engine: &mut Engine, minute_of_day: u32) -> PyResult<u32> {
    let spec = TimerSpec::At { minute_of_day };
    Ok(engine.add_timer(spec).map_err(engine_error)?.0)
}

/// `add_timer_every`: a timer at the close of the first session of each period ("W",
/// "M", "Q" or "Y"); its id. An unknown period raises `NativeEngineInputError`.
pub(super) fn add_every(engine: &mut Engine, unit: &str) -> PyResult<u32> {
    let unit = CalendarUnit::from_code(unit).ok_or_else(|| {
        NativeEngineInputError::new_err(format!("unknown period {unit:?}: use W, M, Q or Y"))
    })?;
    let spec = TimerSpec::Every(unit);
    Ok(engine.add_timer(spec).map_err(engine_error)?.0)
}

/// `timer_counts`: (timer id, sessions fired, sessions missed) of every timer, in id
/// order.
pub(super) fn counts(engine: &Engine) -> Vec<(u32, u32, u32)> {
    let row = |count: &TimerCount| (count.timer.0, count.fired, count.missed);
    engine.timer_counts().iter().map(row).collect()
}
