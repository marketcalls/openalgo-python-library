//! A daily or longer strategy's pre-open decision across the boundary
//! (`oa_bt::engine::pre_open`; design 20-positional-live section 8, unit N4 item N4-3).
//!
//! - `pre_open_decision(bars, last_decided, now)`: in a live session of a daily or longer
//!   strategy, once, after its catch-up (and any restore), before the session runs live.
//!   `bars` are the broker's candles of the strategy's interval, each a tuple (instrument
//!   index, ts_open, ts_close, open, high, low, close, volume): instants in UTC nanoseconds
//!   (the open of the interval's first session and the close of its last, from the
//!   calendar), prices in rupees, volume in units. `last_decided` is the close (UTC
//!   nanoseconds) of the last bar the strategy decided (the journal's), or None; `now` the
//!   present. The bars whose interval has completed by the session's open feed the
//!   instruments' last close where the session has none; the latest of them is decided
//!   unless its close is at or before `last_decided`. Returns None when no bar was
//!   decided, else a dict: "pause", the pause tuple of `advance` (kind "bar_close",
//!   decided at the bar's close, with that bar and the events waiting) at which the
//!   strategy decides and submits as at any bar close, the next `advance` going on with
//!   the session; "bar_close", the decided bar's close, for the journal; "late", True when
//!   the session's open had passed. Its orders wait for the session's open (a tick at or
//!   after it, or the first slice) and are sent then; a late decision's are sent at once
//!   and flagged "late_decision" (the place command's "flags", the fills' flags). An
//!   intraday strategy, a second call, a call once the session has run a slice live, and a
//!   bar that is not of the strategy's interval raise `NativeEngineInputError`.
//! - `order_flags(order_id)`: the names of the flags a working order's fills carry
//!   ("late_decision", "gtt_child_unfilled", "stop_crossed"); empty for any other order.

use oa_bt::engine::{Engine, PreOpen, StrategyBar};
use oa_bt::exec::{InstrumentId, OrderId};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::errors::engine_error;
use super::tables::pause;

/// A bar of the strategy's interval as it crosses (module docs).
pub(super) type DecisionBar = (u32, i64, i64, f64, f64, f64, f64, f64);

fn strategy_bar(
    (instrument, ts_open, ts_close, open, high, low, close, volume): DecisionBar,
) -> StrategyBar {
    StrategyBar {
        instrument: InstrumentId(instrument),
        ts_open,
        ts_close,
        open,
        high,
        low,
        close,
        volume,
    }
}

/// `pre_open_decision` (module docs).
///
/// Errors: those of `Engine::pre_open_decision`, by kind.
pub(super) fn decision<'py>(
    py: Python<'py>,
    engine: &mut Engine,
    bars: Vec<DecisionBar>,
    (last_decided, now): (Option<i64>, i64),
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let bars: Vec<StrategyBar> = bars.into_iter().map(strategy_bar).collect();
    let outcome = engine
        .pre_open_decision(&bars, last_decided, now)
        .map_err(engine_error)?;
    let PreOpen::Decided {
        pause: decided,
        bar_close,
        late,
    } = outcome
    else {
        return Ok(None);
    };
    let dict = PyDict::new_bound(py);
    dict.set_item("pause", pause(py, &decided)?)?;
    dict.set_item("bar_close", bar_close)?;
    dict.set_item("late", late)?;
    Ok(Some(dict))
}

/// `order_flags` (module docs).
pub(super) fn order_flags(engine: &Engine, order_id: u64) -> Vec<&'static str> {
    engine.order_flags(OrderId(order_id)).names()
}
