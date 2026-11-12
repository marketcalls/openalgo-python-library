//! Bindings for `oa_bt::engine`: the stepping kernel as `Engine`.
//!
//! The Python driver (`openalgo/backtest/_driver.py`) builds an `Engine` from a config
//! dict, instrument dicts and bar arrays, then loops: `advance()` returns a pause dict,
//! the driver runs the strategy's callbacks and `on_bar`, places orders with
//! `submit(request)`, and repeats until the pause kind is "finished"; `finish()`
//! returns the result tables. Conversions live in `setup` (inputs), `request` (orders)
//! `tables` (pauses, events, positions, orders) and `results` (the result tables).
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

mod charges;
mod futures;
mod market;
mod request;
mod results;
mod setup;
mod signals;
mod sizing;
mod tables;

use oa_bt::data::resample::CalendarUnit;
use oa_bt::engine::{Engine, KernelError, RunSetup, Submission, TimerSpec};
use oa_bt::exec::{InstrumentId, OrderId};
use oa_bt::model::Product;
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyNotImplementedError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::calendar::NativeCalendar;
use setup::BarArrays;

create_exception!(
    backtest,
    NativeEngineError,
    PyException,
    "The engine cannot go on: a call at the wrong time, or a kernel defect."
);
create_exception!(
    backtest,
    NativeEngineInputError,
    NativeEngineError,
    "The configuration, the data or an order request cannot be used."
);

fn engine_error(error: KernelError) -> PyErr {
    let message = error.to_string();
    if error.not_built_feature().is_some() {
        PyNotImplementedError::new_err(message)
    } else if error.is_input_error() {
        NativeEngineInputError::new_err(message)
    } else {
        NativeEngineError::new_err(message)
    }
}

/// A submission as (outcome, order id, code): ("submitted", id, None), ("rejected", id,
/// reject code) or ("skipped", None, skip reason).
type SubmissionTuple = (&'static str, Option<u64>, Option<&'static str>);

fn submission_tuple(outcome: Submission) -> SubmissionTuple {
    match outcome {
        Submission::Submitted(id) => ("submitted", Some(id.0), None),
        Submission::Rejected(id, code) => ("rejected", Some(id.0), Some(code.code())),
        Submission::Skipped(reason) => ("skipped", None, Some(reason.code())),
    }
}

fn parse_product(code: &str) -> PyResult<Product> {
    Product::from_code(code)
        .ok_or_else(|| NativeEngineInputError::new_err(format!("unknown product {code:?}")))
}

/// The backtest kernel of one run (see `oa_bt::engine`).
#[pyclass(name = "Engine", module = "openalgo._oaindicators.backtest")]
pub struct NativeEngine {
    /// `None` once `finish` has taken the run.
    inner: Option<Engine>,
}

impl NativeEngine {
    fn engine(&self) -> PyResult<&Engine> {
        self.inner
            .as_ref()
            .ok_or_else(|| NativeEngineError::new_err("this run has already finished"))
    }

    fn engine_mut(&mut self) -> PyResult<&mut Engine> {
        self.inner
            .as_mut()
            .ok_or_else(|| NativeEngineError::new_err("this run has already finished"))
    }
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
        let setup = RunSetup {
            config: setup::run_config(config)?,
            calendar: calendar.inner.clone(),
            instruments: instruments
                .iter()
                .map(setup::instrument)
                .collect::<PyResult<_>>()?,
            bars: bars.iter().map(BarArrays::columns).collect(),
        };
        let engine = Engine::new(setup).map_err(engine_error)?;
        Ok(NativeEngine {
            inner: Some(engine),
        })
    }

    /// Run to the next decision point; the pause as a dict.
    fn advance<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
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
    /// target, else the order's submission as `submit` returns it.
    fn target(&mut self, request: &Bound<'_, PyDict>) -> PyResult<Option<SubmissionTuple>> {
        let request = sizing::target_request(request)?;
        let outcome = self
            .engine_mut()?
            .submit_target(request)
            .map_err(engine_error)?;
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
        let ctx = self
            .engine()?
            .sizing_inputs(&request)
            .map_err(engine_error)?;
        sizing::inputs(py, &ctx, request.side)
    }

    /// Cancel a working order; returns its state afterwards.
    fn cancel(&mut self, order_id: u64) -> PyResult<&'static str> {
        let status = self
            .engine_mut()?
            .cancel(OrderId(order_id))
            .map_err(engine_error)?;
        Ok(status.code())
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
        Ok(self
            .engine()?
            .working_orders()
            .iter()
            .map(|id| id.0)
            .collect())
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
    fn account(&self) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
        let a = self.engine()?.account();
        Ok((
            a.equity_rupees,
            a.free_cash_rupees,
            a.realized_rupees,
            a.unrealized_rupees,
            a.costs_rupees,
            a.margin_rupees,
        ))
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
        self.engine_mut()?
            .add_futures_series(series)
            .map_err(engine_error)
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
        let engine = self
            .inner
            .take()
            .ok_or_else(|| NativeEngineError::new_err("this run has already finished"))?;
        let output = engine.finish().map_err(engine_error)?;
        results::output(py, &output)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add(
        "NativeEngineError",
        py.get_type_bound::<NativeEngineError>(),
    )?;
    m.add(
        "NativeEngineInputError",
        py.get_type_bound::<NativeEngineInputError>(),
    )?;
    m.add_class::<NativeEngine>()?;
    futures::register(m)?;
    market::register(m)
}
