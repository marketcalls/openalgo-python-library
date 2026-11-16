//! Bindings for `oa_bt::engine`: the stepping kernel as `Engine`.
//!
//! The Python driver (`openalgo/backtest/_driver.py`) builds an `Engine` from a config
//! dict, instrument dicts and bar arrays, then loops: `advance()` returns a pause tuple,
//! the driver runs the strategy's callbacks and `on_bar`, places orders with
//! `submit(request)`, and repeats until the pause kind is "finished"; `finish()`
//! returns the result tables. Conversions live in `setup` (inputs), `request` (orders)
//! `tables` (pauses, events, positions, orders) and `results` (the result tables). A
//! live session, its commands and its venue's results are in `venue`, its controls in
//! `session`, its legs (ticks, the broker-held stops, the price rule's calls) in `legs`,
//! its saved state and the broker's own fills in `live_state`.
//!
//! Errors: a part that is not built yet raises Python's `NotImplementedError`; bad
//! configuration, data or requests raise `NativeEngineInputError`; anything else
//! (a call at the wrong time, a kernel defect) raises `NativeEngineError`. The Python
//! package converts the last two to its public error types.

// Two lints fire inside PyO3 0.22 macro expansions on newer toolchains, not in this
// code: `create_exception!` checks `cfg(feature = "gil-refs")`, which this crate does
// not declare, and `#[pymethods]` converts the error of every `PyResult` return with
// `.into()`.
#![allow(unexpected_cfgs, clippy::useless_conversion)]

mod access;
mod charges;
mod columns;
mod errors;
mod futures;
mod legs;
mod live_state;
mod market;
mod request;
mod results;
mod session;
mod setup;
mod signals;
mod sizing;
mod tables;
mod venue;

use oa_bt::data::resample::CalendarUnit;
use oa_bt::engine::{Engine, ExitRequest, KernelError, Pause, Protect, TimerSpec};
use oa_bt::exec::{InstrumentId, OrderId};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::calendar::NativeCalendar;
use errors::{engine_error, venue_result_error, NativeEngineInputError};
use request::{submission_tuple, SubmissionTuple};
use setup::{parse_product, run_setup, BarArrays};
use venue::StampTuple;

/// The backtest kernel of one run (see `oa_bt::engine`).
#[pyclass(name = "Engine", module = "openalgo._oaindicators.backtest")]
pub struct NativeEngine {
    /// `None` once `finish` has taken the run.
    inner: Option<Engine>,
}

#[pymethods]
impl NativeEngine {
    /// Build a run from a calendar, the config dict, one dict per instrument and one
    /// tuple of six arrays per instrument (ts_open int64, open, high, low, close,
    /// volume float64), in registration order.
    #[new]
    fn new(
        calendar: PyRef<'_, NativeCalendar>,
        config: &Bound<'_, PyDict>,
        instruments: Vec<Bound<'_, PyDict>>,
        bars: Vec<BarArrays<'_>>,
    ) -> PyResult<Self> {
        let setup = run_setup(&calendar, config, &instruments, &bars)?;
        NativeEngine::built(Engine::new(setup))
    }

    /// Build a live session from the inputs of a run, with the bars of the slices
    /// already closed (see `venue`).
    #[staticmethod]
    fn session(
        calendar: PyRef<'_, NativeCalendar>,
        config: &Bound<'_, PyDict>,
        instruments: Vec<Bound<'_, PyDict>>,
        bars: Vec<BarArrays<'_>>,
    ) -> PyResult<Self> {
        let setup = run_setup(&calendar, config, &instruments, &bars)?;
        NativeEngine::built(Engine::new_session(setup))
    }

    /// Hand a live session an instrument's base bar once its slice has closed.
    fn add_bar(&mut self, instrument: u32, bar: venue::BarTuple) -> PyResult<()> {
        let engine = self.engine_mut()?;
        let added = engine.add_bar(InstrumentId(instrument), venue::base_bar(bar));
        added.map_err(engine_error)
    }

    /// Let a live session run the slices before index `slices` of its grid.
    fn release(&mut self, slices: usize) -> PyResult<()> {
        self.engine_mut()?.release(slices);
        Ok(())
    }

    /// Who fills the orders: "simulated" (a backtest) or "external" (a live session).
    fn venue(&self) -> PyResult<&'static str> {
        Ok(self.engine()?.venue().code())
    }

    /// A live session's commands since the last call, in order, as dicts (`venue`).
    fn take_commands<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        venue::commands(py, &self.engine_mut()?.take_commands())
    }

    /// Record the venue's fill of `units` newly filled of an order at `price` rupees at
    /// `at` (UTC nanoseconds); returns the stamp (slice, slice_ts, step). A fill beyond
    /// the position of an order that may not reverse it raises `VenueRefusalError` of
    /// kind "would_reverse" after recording the units that closed the position.
    fn apply_fill(
        &mut self,
        py: Python<'_>,
        order_id: u64,
        units: f64,
        price: f64,
        at: i64,
    ) -> PyResult<StampTuple> {
        let engine = self.engine_mut()?;
        venue::recorded(py, engine.apply_fill(OrderId(order_id), units, price, at))
    }

    /// Record that the venue rejected an order with a reject code; returns the stamp.
    fn apply_reject(
        &mut self,
        py: Python<'_>,
        order_id: u64,
        reason: &str,
    ) -> PyResult<StampTuple> {
        let reason = venue::reject_code(reason)?;
        let engine = self.engine_mut()?;
        venue::recorded(py, engine.apply_reject(OrderId(order_id), reason))
    }

    /// Record that the venue cancelled an order; returns the stamp.
    fn apply_cancel(&mut self, py: Python<'_>, order_id: u64) -> PyResult<StampTuple> {
        venue::recorded(py, self.engine_mut()?.apply_cancel(OrderId(order_id)))
    }

    /// Record that the venue refused to cancel an order: it works on.
    fn apply_cancel_refused(&mut self, py: Python<'_>, order_id: u64) -> PyResult<()> {
        let engine = self.engine_mut()?;
        venue::answered(py, engine.apply_cancel_refused(OrderId(order_id)))
    }

    /// Record that the venue confirmed an order's modification; returns the stamp.
    fn apply_modify(&mut self, py: Python<'_>, order_id: u64) -> PyResult<StampTuple> {
        venue::recorded(py, self.engine_mut()?.apply_modify(OrderId(order_id)))
    }

    /// Record that the venue refused an order's modification: its old terms stand.
    fn apply_modify_refused(&mut self, py: Python<'_>, order_id: u64) -> PyResult<()> {
        let engine = self.engine_mut()?;
        venue::answered(py, engine.apply_modify_refused(OrderId(order_id)))
    }

    /// Record the broker's own fill of a live session's position, with no order: `units`
    /// on `side` at `price` rupees at `at` (UTC nanoseconds) for the exit `reason`;
    /// returns the stamp (`live_state`).
    #[allow(clippy::too_many_arguments)]
    fn apply_external_fill(
        &mut self,
        py: Python<'_>,
        instrument: u32,
        product: &str,
        side: &str,
        units: f64,
        price: f64,
        at: i64,
        reason: &str,
    ) -> PyResult<StampTuple> {
        let fill = live_state::ExternalFill {
            instrument,
            product,
            side,
            units,
            price,
            at,
            reason,
        };
        live_state::apply_external_fill(py, self.engine_mut()?, fill)
    }

    /// A live session's state as plain data that survives a JSON round trip
    /// (`live_state`).
    fn live_state(&self, py: Python<'_>) -> PyResult<PyObject> {
        live_state::live_state(py, self.engine()?)
    }

    /// Restore a saved state into a session built for a restart; `{saved id: new id}` of
    /// the restored working orders (`live_state`).
    fn restore_live_state<'py>(
        &mut self,
        py: Python<'py>,
        state: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        live_state::restore_live_state(py, self.engine_mut()?, state)
    }

    /// Exit a live session's position for a reason of the session's; what it did, one of
    /// `EXIT_REQUESTS` (`session`); at market, or as LIMIT orders at `limit_rupees`
    /// (`legs`).
    #[pyo3(signature = (instrument, product, reason, limit_rupees=None))]
    fn request_exit(
        &mut self,
        instrument: u32,
        product: &str,
        reason: &str,
        limit_rupees: Option<f64>,
    ) -> PyResult<&'static str> {
        let (product, exit) = (parse_product(product)?, session::engine_exit(reason)?);
        let (engine, instrument) = (self.engine_mut()?, InstrumentId(instrument));
        let asked = match limit_rupees {
            None => engine.request_exit(instrument, product, exit),
            Some(limit) => engine.request_limit_exit(instrument, product, exit, limit),
        };
        asked.map(ExitRequest::code).map_err(engine_error)
    }

    /// Hand a live session its last price of an instrument, rupees at `at` (UTC
    /// nanoseconds): tick mode (`legs`).
    fn push_tick(&mut self, py: Python<'_>, instrument: u32, price: f64, at: i64) -> PyResult<()> {
        let engine = self.engine_mut()?;
        venue::answered(py, engine.push_tick(InstrumentId(instrument), price, at))
    }

    /// Place a leg's stop again; what it did, one of `PROTECT_OUTCOMES` (`legs`).
    fn protect(&mut self, instrument: u32, product: &str) -> PyResult<&'static str> {
        let (product, engine) = (parse_product(product)?, self.engine_mut()?);
        let done = engine.protect(InstrumentId(instrument), product);
        done.map(Protect::code).map_err(engine_error)
    }

    /// Reprice a leg's stop or an engine exit to a LIMIT at `limit_rupees`; True when the
    /// modification was sent now, False when it waits for an earlier one's answer (`legs`).
    fn reprice_exit(&mut self, py: Python<'_>, order_id: u64, limit_rupees: f64) -> PyResult<bool> {
        let engine = self.engine_mut()?;
        let repriced = engine.reprice_exit(OrderId(order_id), limit_rupees);
        repriced.map_err(|error| venue_result_error(py, error))
    }

    /// Lock a live session's entries for a reason; True when they were not locked for it.
    fn lock_entries(&mut self, reason: &str) -> PyResult<bool> {
        let engine = self.engine_mut()?;
        engine.lock_entries(reason).map_err(engine_error)
    }

    /// Lift a live session's entry lock of a reason; True when they were locked for it.
    fn unlock_entries(&mut self, reason: &str) -> PyResult<bool> {
        let engine = self.engine_mut()?;
        engine.unlock_entries(reason).map_err(engine_error)
    }

    /// The reasons a live session's entries are locked for, sorted.
    fn entry_locks(&self) -> PyResult<Vec<String>> {
        let locks = self.engine()?.entry_locks();
        Ok(locks.iter().map(|reason| reason.to_string()).collect())
    }

    /// Cap the equity a live session's engine default sizes from, rupees (`session`).
    fn cap_sizing_equity(&mut self, rupees: f64) -> PyResult<()> {
        let engine = self.engine_mut()?;
        engine.cap_sizing_equity(rupees).map_err(engine_error)
    }

    /// The strategy's daily loss measure, rupees on its risk basis (negative is a loss).
    fn daily_loss_measure(&self) -> PyResult<f64> {
        Ok(self.engine()?.daily_loss_measure())
    }

    /// End a live session where it stands (`session`); the "finished" pause tuple.
    fn stop_session<'py>(&mut self, py: Python<'py>) -> PyResult<tables::PauseTuple<'py>> {
        let events = self.engine_mut()?.stop_session().map_err(engine_error)?;
        tables::pause(py, &Pause::Finished { events })
    }

    /// Run to the next decision point; the pause as a tuple (kind, decided_at, bars,
    /// events, timers), see `tables::pause`.
    fn advance<'py>(&mut self, py: Python<'py>) -> PyResult<tables::PauseTuple<'py>> {
        let pause = self.engine_mut()?.advance().map_err(engine_error)?;
        tables::pause(py, &pause)
    }

    /// Place an order at the current decision point. Returns (outcome, order id, code):
    /// ("submitted", id, None), ("rejected", id, reject code) or ("skipped", None, skip
    /// reason).
    fn submit(&mut self, request: &Bound<'_, PyDict>) -> PyResult<SubmissionTuple> {
        let request = request::order_request(request)?;
        let outcome = self.engine_mut()?.submit(request).map_err(engine_error)?;
        Ok(submission_tuple(outcome))
    }

    /// Trade to a target position at the current decision point (`target()`, a target
    /// dict as `sizing` reads it). Returns None when the position already is the
    /// target, else the order's submission as `submit` returns it (("skipped", None,
    /// "rounded_to_zero") for a value or percent target that rounds to no units).
    fn target(&mut self, request: &Bound<'_, PyDict>) -> PyResult<Option<SubmissionTuple>> {
        let request = sizing::target_request(request)?;
        let engine = self.engine_mut()?;
        let outcome = engine.submit_target(request).map_err(engine_error)?;
        Ok(outcome.map(submission_tuple))
    }

    /// Trade to a target position as part of a rebalance (`rebalance()`, a target dict
    /// as `target` takes): the same order, marked and tagged `rebalance`, whose buy fills
    /// after the slice's sales. Returns what `target` returns.
    fn rebalance_target(
        &mut self,
        request: &Bound<'_, PyDict>,
    ) -> PyResult<Option<SubmissionTuple>> {
        let request = sizing::target_request(request)?;
        let engine = self.engine_mut()?;
        let outcome = engine.submit_rebalance(request).map_err(engine_error)?;
        Ok(outcome.map(submission_tuple))
    }

    /// What a sizer sees for an order request at the decision, as a dict (see
    /// `sizing::inputs`): what a strategy's sizing function receives.
    fn sizing_inputs<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyDict>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let request = request::order_request(request)?;
        let ctx = self.engine()?.sizing_inputs(&request);
        sizing::inputs(py, &ctx.map_err(engine_error)?, request.side)
    }

    /// Replace a working order's request in full at the current decision point (a dict
    /// as `submit` takes). Returns what `submit` returns; in a live session the order
    /// keeps its old terms until the venue confirms the new ones (`apply_modify`).
    fn modify(&mut self, order_id: u64, request: &Bound<'_, PyDict>) -> PyResult<SubmissionTuple> {
        let request = request::order_request(request)?;
        let engine = self.engine_mut()?;
        let outcome = engine.modify(OrderId(order_id), request);
        Ok(submission_tuple(outcome.map_err(engine_error)?))
    }

    /// Cancel a working order; returns its state afterwards (in a live session the order
    /// works on until the venue confirms the cancel with `apply_cancel`, and the call is
    /// taken while the session waits too, `session`).
    fn cancel(&mut self, order_id: u64) -> PyResult<&'static str> {
        let status = self.engine_mut()?.cancel(OrderId(order_id));
        Ok(status.map_err(engine_error)?.code())
    }

    /// The position of an instrument and product as a dict, or None if never opened.
    fn position<'py>(
        &self,
        py: Python<'py>,
        instrument: u32,
        product: &str,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let product = parse_product(product)?;
        self.engine()?
            .position(InstrumentId(instrument), product)
            .map(|p| tables::position(py, p))
            .transpose()
    }

    /// An order as a dict, or None for an unknown id.
    fn order<'py>(&self, py: Python<'py>, order_id: u64) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.engine()?
            .order(OrderId(order_id))
            .map(|o| tables::order(py, o))
            .transpose()
    }

    /// Ids of the orders that can still fill.
    fn working_orders(&self) -> PyResult<Vec<u64>> {
        let working = self.engine()?.working_orders();
        Ok(working.iter().map(|id| id.0).collect())
    }

    /// True while an order that would open or add works on the instrument.
    fn has_pending_entry(&self, instrument: u32) -> PyResult<bool> {
        Ok(self.engine()?.has_pending_entry(InstrumentId(instrument)))
    }

    /// Run a condition strategy to the end of the grid in Rust, with no Python call per
    /// bar, from its spec dict (see `signals`). Returns the entries a sizer skipped as
    /// (decision instant, reason) pairs. Then `finish()` returns the result.
    fn run_signals(&mut self, spec: &Bound<'_, PyDict>) -> PyResult<Vec<(i64, &'static str)>> {
        let signals = signals::signal_arrays(spec)?;
        let skipped =
            oa_bt::engine::run_signals(self.engine_mut()?, &signals).map_err(engine_error)?;
        Ok(skipped
            .iter()
            .map(|entry| (entry.decided_at, entry.reason.code()))
            .collect())
    }

    /// The last close seen of an instrument, rupees.
    fn last_close(&self, instrument: u32) -> PyResult<Option<f64>> {
        Ok(self.engine()?.last_close(InstrumentId(instrument)))
    }

    /// The account as (equity, free cash, realized, unrealized, costs, margin), rupees.
    fn account(&self) -> PyResult<tables::AccountTuple> {
        Ok(tables::account(&self.engine()?.account()))
    }

    /// Register a timer at an IST time each session (minutes since midnight); returns
    /// its id. Only before the first `advance`.
    fn add_timer_at(&mut self, minute_of_day: u32) -> PyResult<u32> {
        let spec = TimerSpec::At { minute_of_day };
        Ok(self.engine_mut()?.add_timer(spec).map_err(engine_error)?.0)
    }

    /// Register a timer at the close of the first session of each period ("W", "M", "Q"
    /// or "Y"); returns its id. Only before the first `advance`.
    fn add_timer_every(&mut self, unit: &str) -> PyResult<u32> {
        let unit = CalendarUnit::from_code(unit).ok_or_else(|| {
            NativeEngineInputError::new_err(format!("unknown period {unit:?}: use W, M, Q or Y"))
        })?;
        let spec = TimerSpec::Every(unit);
        Ok(self.engine_mut()?.add_timer(spec).map_err(engine_error)?.0)
    }

    /// (timer id, sessions fired, sessions missed) of every timer, in id order.
    fn timer_counts(&self) -> PyResult<Vec<(u32, u32, u32)>> {
        Ok(self
            .engine()?
            .timer_counts()
            .iter()
            .map(|count| (count.timer.0, count.fired, count.missed))
            .collect())
    }

    /// Every strategy bar of an instrument over the run, as a dict of arrays (ts_open,
    /// ts_close int64; open, high, low, close, volume float64): the bars the pauses
    /// deliver, in the same order.
    fn strategy_bars<'py>(&self, py: Python<'py>, instrument: u32) -> PyResult<Bound<'py, PyDict>> {
        let bars = self.engine()?.strategy_bars(InstrumentId(instrument));
        tables::strategy_bars(py, &bars)
    }

    /// Register a futures series (see `futures`); returns its index. Only before the
    /// first `advance`.
    fn add_futures_series(&mut self, series: &Bound<'_, PyDict>) -> PyResult<u32> {
        let series = futures::series(series)?;
        let engine = self.engine_mut()?;
        engine.add_futures_series(series).map_err(engine_error)
    }

    /// The switches of the run's futures series, in slice order, as dicts.
    fn futures_switches<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        futures::switches(py, self.engine()?.futures_switches())
    }

    /// The result tables, once the run has finished. The engine cannot be used after.
    fn finish<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if !self.engine()?.is_finished() {
            return Err(engine_error(KernelError::RunNotFinished));
        }
        let output = self.take()?.finish().map_err(engine_error)?;
        results::output(py, &output)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    errors::register(m)?;
    m.add_class::<NativeEngine>()?;
    // The engine default's cash buffer in venue mode, a fraction, for the Assumptions.
    m.add("VENUE_CASH_BUFFER", oa_bt::sizing::VENUE_CASH_BUFFER)?;
    venue::register(m)?;
    session::register(m)?;
    legs::register(m)?;
    futures::register(m)?;
    market::register(m)
}
