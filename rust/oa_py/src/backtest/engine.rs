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
//! its saved state and the broker's own fills in `live_state`, a daily or longer
//! strategy's decision before the open in `pre_open`. The methods' bodies are
//! helpers in those files and in `orders` (the order calls of a decision point),
//! `queries`, `timers` and `signals`, so this file holds the one `#[pymethods]` block.
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
mod gtts;
mod legs;
mod live_state;
mod market;
mod orders;
mod pre_open;
mod queries;
mod request;
mod results;
mod session;
mod setup;
mod signals;
mod sizing;
mod tables;
mod timers;
mod venue;

use oa_bt::engine::{Engine, Pause};
use oa_bt::exec::{GttId, InstrumentId, OrderId};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::calendar::NativeCalendar;
use errors::{engine_error, venue_result_error};
use request::SubmissionTuple;
use setup::BarArrays;
use venue::StampTuple;

// What the fund's binding (`super::fund`) reads of this one.
pub(super) use errors::{error_of_kind, NativeEngineError, NativeEngineInputError};
pub(super) use setup::{field, optional, unknown};
pub(super) use signals::signal_arrays;
pub(super) use tables::{pause as pause_tuple, PauseTuple};

/// A dict argument or result.
type Dict<'py> = Bound<'py, PyDict>;

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
        config: &Dict<'_>,
        instruments: Vec<Dict<'_>>,
        bars: Vec<BarArrays<'_>>,
    ) -> PyResult<Self> {
        NativeEngine::from_inputs(&calendar, config, &instruments, &bars, Engine::new)
    }

    /// Build a live session from the inputs of a run, with the bars of the slices
    /// already closed (see `venue`).
    #[staticmethod]
    fn session(
        calendar: PyRef<'_, NativeCalendar>,
        config: &Dict<'_>,
        instruments: Vec<Dict<'_>>,
        bars: Vec<BarArrays<'_>>,
    ) -> PyResult<Self> {
        NativeEngine::from_inputs(&calendar, config, &instruments, &bars, Engine::new_session)
    }

    /// Hand a live session an instrument's base bar once its slice has closed.
    fn add_bar(&mut self, instrument: u32, bar: venue::BarTuple) -> PyResult<()> {
        venue::add_bar(self.engine_mut()?, instrument, bar)
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

    /// A live session's commands since the last call, in order, as dicts (`venue`). The
    /// commands of an order that has ended since they were made (rejected, cancelled or
    /// filled whole) are dropped.
    fn take_commands<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let engine = self.engine_mut()?;
        let commands = engine.take_commands();
        venue::commands(py, &commands, |order| engine.order_flags(order))
    }

    /// Record the venue's fill of `units` newly filled of an order at `price` rupees at
    /// `at` (UTC nanoseconds), with the names of the flags the venue reports with it
    /// (`flags`, of `VENUE_FILL_FLAGS`; `venue`); returns the stamp (slice, slice_ts,
    /// step). A fill beyond the position of an order that may not reverse it raises
    /// `VenueRefusalError` of kind "would_reverse" after recording the units that closed
    /// the position.
    #[pyo3(signature = (order_id, units, price, at, flags=Vec::new()))]
    fn apply_fill(
        &mut self,
        py: Python<'_>,
        order_id: u64,
        units: f64,
        price: f64,
        at: i64,
        flags: Vec<String>,
    ) -> PyResult<StampTuple> {
        let fill = (order_id, units, price, at);
        venue::fill(py, self.engine_mut()?, fill, &flags)
    }

    /// Record that the venue rejected an order with a reject code; returns the stamp.
    fn apply_reject(
        &mut self,
        py: Python<'_>,
        order_id: u64,
        reason: &str,
    ) -> PyResult<StampTuple> {
        venue::reject(py, self.engine_mut()?, order_id, reason)
    }

    /// Record that the venue rejected a closing order in a way that says the broker may not
    /// hold its units: its position is held until the session asks (`venue`); returns the
    /// stamp.
    fn apply_margin_reject(&mut self, py: Python<'_>, order_id: u64) -> PyResult<StampTuple> {
        venue::recorded(
            py,
            self.engine_mut()?.apply_margin_reject(OrderId(order_id)),
        )
    }

    /// Record that the venue cancelled an order; returns the stamp.
    fn apply_cancel(&mut self, py: Python<'_>, order_id: u64) -> PyResult<StampTuple> {
        venue::recorded(py, self.engine_mut()?.apply_cancel(OrderId(order_id)))
    }

    /// Record that the venue refused to cancel an order: it works on.
    fn apply_cancel_refused(&mut self, py: Python<'_>, order_id: u64) -> PyResult<()> {
        venue::answered(
            py,
            self.engine_mut()?.apply_cancel_refused(OrderId(order_id)),
        )
    }

    /// Record that the venue confirmed an order's modification; returns the stamp.
    fn apply_modify(&mut self, py: Python<'_>, order_id: u64) -> PyResult<StampTuple> {
        venue::recorded(py, self.engine_mut()?.apply_modify(OrderId(order_id)))
    }

    /// Record that the venue refused an order's modification: its old terms stand.
    fn apply_modify_refused(&mut self, py: Python<'_>, order_id: u64) -> PyResult<()> {
        venue::answered(
            py,
            self.engine_mut()?.apply_modify_refused(OrderId(order_id)),
        )
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
        let fill = (instrument, product, side, units, price, at, reason);
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
    ) -> PyResult<Dict<'py>> {
        live_state::restore_live_state(py, self.engine_mut()?, state)
    }

    /// A live session's state carried to the next trading day, once its session has closed
    /// (`live_state`).
    fn carried_state(&self, py: Python<'_>) -> PyResult<PyObject> {
        live_state::carried_state(py, self.engine()?)
    }

    /// Restore a carried state into the session of a later trading day; `{"gtts": {saved
    /// id: new id}, "orders": {saved id: new id}}` (`live_state`).
    fn restore_carried_state<'py>(
        &mut self,
        py: Python<'py>,
        state: &Bound<'py, PyAny>,
    ) -> PyResult<Dict<'py>> {
        live_state::restore_carried_state(py, self.engine_mut()?, state)
    }

    /// Restore the last saved state of a session of an earlier day that stopped without
    /// its close, into a session built for that day, standing where the save stood;
    /// `today` is the day number of the session that carries it on, `unsent` the GTTs
    /// whose waiting request never reached the venue; `{saved id: new id}` of the restored
    /// working orders (`live_state`).
    #[pyo3(signature = (state, today, unsent))]
    fn restore_stopped_state<'py>(
        &mut self,
        py: Python<'py>,
        state: &Bound<'py, PyAny>,
        today: i32,
        unsent: Vec<u64>,
    ) -> PyResult<Dict<'py>> {
        let unsent: Vec<GttId> = unsent.into_iter().map(GttId).collect();
        live_state::restore_stopped_state(py, self.engine_mut()?, state, today, &unsent)
    }

    /// Record that the venue confirmed a GTT's place; the stamp (`gtts`).
    fn apply_gtt_placed(&mut self, py: Python<'_>, gtt_id: u64) -> PyResult<StampTuple> {
        venue::recorded(py, self.engine_mut()?.apply_gtt_placed(GttId(gtt_id)))
    }

    /// Record that the venue refused a GTT's place; the stamp (`gtts`).
    fn apply_gtt_refused(&mut self, py: Python<'_>, gtt_id: u64) -> PyResult<StampTuple> {
        venue::recorded(py, self.engine_mut()?.apply_gtt_refused(GttId(gtt_id)))
    }

    /// Record that the venue confirmed a GTT's modification; the stamp (`gtts`).
    fn apply_gtt_modified(&mut self, py: Python<'_>, gtt_id: u64) -> PyResult<StampTuple> {
        venue::recorded(py, self.engine_mut()?.apply_gtt_modified(GttId(gtt_id)))
    }

    /// Record that the venue refused a GTT's modification (`gtts`).
    fn apply_gtt_modify_refused(&mut self, py: Python<'_>, gtt_id: u64) -> PyResult<()> {
        let refused = self.engine_mut()?.apply_gtt_modify_refused(GttId(gtt_id));
        venue::answered(py, refused)
    }

    /// Record that the venue cancelled a GTT, on the session's request or not; the stamp
    /// (`gtts`).
    fn apply_gtt_cancelled(&mut self, py: Python<'_>, gtt_id: u64) -> PyResult<StampTuple> {
        venue::recorded(py, self.engine_mut()?.apply_gtt_cancelled(GttId(gtt_id)))
    }

    /// Record that the venue refused a GTT's cancellation (`gtts`).
    fn apply_gtt_cancel_refused(&mut self, py: Python<'_>, gtt_id: u64) -> PyResult<()> {
        let refused = self.engine_mut()?.apply_gtt_cancel_refused(GttId(gtt_id));
        venue::answered(py, refused)
    }

    /// Record that a GTT ended at the venue without the session asking, `how` one of
    /// `GTT_ENDS`; the stamp (`gtts`).
    fn apply_gtt_ended(&mut self, py: Python<'_>, gtt_id: u64, how: &str) -> PyResult<StampTuple> {
        let how = gtts::gtt_end(how)?;
        venue::recorded(py, self.engine_mut()?.apply_gtt_ended(GttId(gtt_id), how))
    }

    /// Record that the venue fired a GTT at `at` (UTC nanoseconds); the kernel order id of
    /// its child (`gtts`).
    fn apply_gtt_triggered(&mut self, py: Python<'_>, gtt_id: u64, at: i64) -> PyResult<u64> {
        gtts::triggered(py, self.engine_mut()?, gtt_id, at)
    }

    /// A GTT of the run as a dict, None for an unknown id (`gtts`).
    fn gtt<'py>(&self, py: Python<'py>, gtt_id: u64) -> PyResult<Option<Dict<'py>>> {
        gtts::gtt(py, self.engine()?, gtt_id)
    }

    /// The GTTs of an instrument in a product, in id order, as dicts (`gtts`).
    fn gtts<'py>(
        &self,
        py: Python<'py>,
        instrument: u32,
        product: &str,
    ) -> PyResult<Bound<'py, PyList>> {
        gtts::gtts(py, self.engine()?, (instrument, product))
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
        let position = (instrument, product);
        session::request_exit(self.engine_mut()?, position, reason, limit_rupees)
    }

    /// Hand a live session its last price of an instrument, rupees at `at` (UTC
    /// nanoseconds): tick mode (`legs`).
    fn push_tick(&mut self, py: Python<'_>, instrument: u32, price: f64, at: i64) -> PyResult<()> {
        let ticked = self
            .engine_mut()?
            .push_tick(InstrumentId(instrument), price, at);
        venue::answered(py, ticked)
    }

    /// Place a leg's stop again; what it did, one of `PROTECT_OUTCOMES` (`legs`).
    fn protect(&mut self, instrument: u32, product: &str) -> PyResult<&'static str> {
        session::protect(self.engine_mut()?, instrument, product)
    }

    /// Reprice a leg's stop or an engine exit to a LIMIT at `limit_rupees`; True when the
    /// modification was sent now, False when it waits for an earlier one's answer (`legs`).
    fn reprice_exit(&mut self, py: Python<'_>, order_id: u64, limit_rupees: f64) -> PyResult<bool> {
        let repriced = self
            .engine_mut()?
            .reprice_exit(OrderId(order_id), limit_rupees);
        repriced.map_err(|error| venue_result_error(py, error))
    }

    /// Lock a live session's entries for a reason; True when they were not locked for it.
    fn lock_entries(&mut self, reason: &str) -> PyResult<bool> {
        self.engine_mut()?
            .lock_entries(reason)
            .map_err(engine_error)
    }

    /// Lift a live session's entry lock of a reason; True when they were locked for it.
    fn unlock_entries(&mut self, reason: &str) -> PyResult<bool> {
        self.engine_mut()?
            .unlock_entries(reason)
            .map_err(engine_error)
    }

    /// The reasons a live session's entries are locked for, sorted.
    fn entry_locks(&self) -> PyResult<Vec<String>> {
        Ok(queries::entry_locks(self.engine()?))
    }

    /// Cap the equity a live session's engine default sizes from, rupees (`session`).
    fn cap_sizing_equity(&mut self, rupees: f64) -> PyResult<()> {
        self.engine_mut()?
            .cap_sizing_equity(rupees)
            .map_err(engine_error)
    }

    /// Set the side the other strategies of a live session hold or enter on an instrument,
    /// "long", "short" or None (`session`).
    #[pyo3(signature = (instrument, product, side))]
    fn set_group_side(
        &mut self,
        instrument: u32,
        product: &str,
        side: Option<&str>,
    ) -> PyResult<()> {
        session::set_group_side(self.engine_mut()?, (instrument, product), side)
    }

    /// Mark an instrument as traded by several strategies of a live session, or not
    /// (`session`).
    fn set_shared(&mut self, instrument: u32, product: &str, shared: bool) -> PyResult<()> {
        session::set_shared(self.engine_mut()?, (instrument, product), shared)
    }

    /// The strategy's daily loss measure, rupees on its risk basis (negative is a loss).
    fn daily_loss_measure(&self) -> PyResult<f64> {
        Ok(self.engine()?.daily_loss_measure())
    }

    /// The entries the strategy has counted on its session day (`session`).
    fn trades_today(&self) -> PyResult<u32> {
        Ok(self.engine()?.trades_today())
    }

    /// The account's trade cap (None for none) and the other strategies' entries of the
    /// day, for a live session's next decisions (`session`).
    #[pyo3(signature = (limit, others))]
    fn set_account_trades(&mut self, limit: Option<u32>, others: u32) -> PyResult<()> {
        session::set_account_trades(self.engine_mut()?, limit, others)
    }

    /// Decide before the open of a daily or longer strategy's live session on the latest
    /// completed bar of `bars` (`pre_open`); None when none was decided.
    #[pyo3(signature = (bars, last_decided, now))]
    fn pre_open_decision<'py>(
        &mut self,
        py: Python<'py>,
        bars: Vec<pre_open::DecisionBar>,
        last_decided: Option<i64>,
        now: i64,
    ) -> PyResult<Option<Dict<'py>>> {
        pre_open::decision(py, self.engine_mut()?, bars, (last_decided, now))
    }

    /// The names of the flags a working order's fills carry (`pre_open`).
    fn order_flags(&self, order_id: u64) -> PyResult<Vec<&'static str>> {
        Ok(pre_open::order_flags(self.engine()?, order_id))
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
    fn submit(&mut self, request: &Dict<'_>) -> PyResult<SubmissionTuple> {
        orders::submit(self.engine_mut()?, request)
    }

    /// Trade to a target position at the current decision point (`target()`, a target
    /// dict as `sizing` reads it). Returns None when the position already is the
    /// target, else the order's submission as `submit` returns it (("skipped", None,
    /// "rounded_to_zero") for a value or percent target that rounds to no units).
    fn target(&mut self, request: &Dict<'_>) -> PyResult<Option<SubmissionTuple>> {
        orders::target(self.engine_mut()?, request)
    }

    /// Trade to a target position as part of a rebalance (`rebalance()`, a target dict
    /// as `target` takes): the same order, marked and tagged `rebalance`, whose buy fills
    /// after the slice's sales. Returns what `target` returns.
    fn rebalance_target(&mut self, request: &Dict<'_>) -> PyResult<Option<SubmissionTuple>> {
        orders::rebalance_target(self.engine_mut()?, request)
    }

    /// What a sizer sees for an order request at the decision, as a dict (see
    /// `sizing::inputs`): what a strategy's sizing function receives.
    fn sizing_inputs<'py>(&self, py: Python<'py>, request: &Dict<'_>) -> PyResult<Dict<'py>> {
        orders::sizing_inputs(py, self.engine()?, request)
    }

    /// Replace a working order's request in full at the current decision point (a dict
    /// as `submit` takes). Returns what `submit` returns; in a live session the order
    /// keeps its old terms until the venue confirms the new ones (`apply_modify`).
    fn modify(&mut self, order_id: u64, request: &Dict<'_>) -> PyResult<SubmissionTuple> {
        orders::modify(self.engine_mut()?, order_id, request)
    }

    /// Cancel a working order; returns its state afterwards (in a live session the order
    /// works on until the venue confirms the cancel with `apply_cancel`, and the call is
    /// taken while the session waits too, `session`).
    fn cancel(&mut self, order_id: u64) -> PyResult<&'static str> {
        orders::cancel(self.engine_mut()?, order_id)
    }

    /// The position of an instrument and product as a dict, or None if never opened.
    fn position<'py>(
        &self,
        py: Python<'py>,
        instrument: u32,
        product: &str,
    ) -> PyResult<Option<Dict<'py>>> {
        queries::position(py, self.engine()?, instrument, product)
    }

    /// An order as a dict, or None for an unknown id.
    fn order<'py>(&self, py: Python<'py>, order_id: u64) -> PyResult<Option<Dict<'py>>> {
        queries::order(py, self.engine()?, order_id)
    }

    /// The exit reason an order's fills record (a code of the trades' `exit_reason`):
    /// "leg_target" for the exit a leg's target makes on a tick, "scheduler" for the
    /// square-off, a risk exit's own, the one a strategy gave its exit; None when it names
    /// none. Raises `NativeEngineError` for an unknown id.
    fn order_exit_reason(&self, order_id: u64) -> PyResult<Option<&'static str>> {
        session::order_exit_reason(self.engine()?, OrderId(order_id))
    }

    /// Ids of the orders that can still fill.
    fn working_orders(&self) -> PyResult<Vec<u64>> {
        Ok(queries::working_orders(self.engine()?))
    }

    /// True while an order that would open or add works on the instrument.
    fn has_pending_entry(&self, instrument: u32) -> PyResult<bool> {
        Ok(self.engine()?.has_pending_entry(InstrumentId(instrument)))
    }

    /// Run a condition strategy to the end of the grid in Rust, with no Python call per
    /// bar, from its spec dict (see `signals`). Returns the entries a sizer skipped as
    /// (decision instant, reason) pairs. Then `finish()` returns the result.
    fn run_signals(&mut self, spec: &Dict<'_>) -> PyResult<Vec<signals::SkippedTuple>> {
        signals::run(self.engine_mut()?, spec)
    }

    /// The last close seen of an instrument, rupees.
    fn last_close(&self, instrument: u32) -> PyResult<Option<f64>> {
        Ok(self.engine()?.last_close(InstrumentId(instrument)))
    }

    /// Rupees: the price a backtest of this run fills a MARKET order of `side` at, at the
    /// open of the instrument's base bar opening at `bar_open` (UTC nanoseconds) at `open`
    /// rupees, by the run's slippage rule; records nothing (`market`).
    fn market_fill_price(
        &self,
        instrument: u32,
        side: &str,
        bar_open: i64,
        open: f64,
    ) -> PyResult<f64> {
        market::market_fill_price(self.engine()?, instrument, side, bar_open, open)
    }

    /// The account as (equity, free cash, realized, unrealized, costs, margin), rupees.
    fn account(&self) -> PyResult<tables::AccountTuple> {
        Ok(tables::account(&self.engine()?.account()))
    }

    /// Register a timer at an IST time each session (minutes since midnight); returns
    /// its id. Only before the first `advance`.
    fn add_timer_at(&mut self, minute_of_day: u32) -> PyResult<u32> {
        timers::add_at(self.engine_mut()?, minute_of_day)
    }

    /// Register a timer at the close of the first session of each period ("W", "M", "Q"
    /// or "Y"); returns its id. Only before the first `advance`.
    fn add_timer_every(&mut self, unit: &str) -> PyResult<u32> {
        timers::add_every(self.engine_mut()?, unit)
    }

    /// (timer id, sessions fired, sessions missed) of every timer, in id order.
    fn timer_counts(&self) -> PyResult<Vec<(u32, u32, u32)>> {
        Ok(timers::counts(self.engine()?))
    }

    /// Every strategy bar of an instrument over the run, as a dict of arrays (ts_open,
    /// ts_close int64; open, high, low, close, volume float64): the bars the pauses
    /// deliver, in the same order.
    fn strategy_bars<'py>(&self, py: Python<'py>, instrument: u32) -> PyResult<Dict<'py>> {
        queries::strategy_bars(py, self.engine()?, instrument)
    }

    /// Register a futures series (see `futures`); returns its index. Only before the
    /// first `advance`.
    fn add_futures_series(&mut self, series: &Dict<'_>) -> PyResult<u32> {
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
    fn finish<'py>(&mut self, py: Python<'py>) -> PyResult<Dict<'py>> {
        let output = self.take_finished()?.finish().map_err(engine_error)?;
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
    gtts::register(m)?;
    futures::register(m)?;
    market::register(m)
}
