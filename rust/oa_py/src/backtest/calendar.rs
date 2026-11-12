//! Bindings for `oa_bt::market::calendar`.
//!
//! Conventions at the boundary:
//! - Exchanges are OpenAlgo exchange strings ("NSE", "NFO", "MCX", ...).
//! - Dates are IST calendar dates as days since 1970-01-01 (Python:
//!   `date.toordinal() - 719163`), from `MIN_DAY` to `MAX_DAY`; any other day number is
//!   a `NativeCalendarError`, never an overflow.
//! - Instants are int64 UTC nanoseconds; session times are seconds since IST midnight.
//! - Session kind is 0 regular, 1 special. Coverage is 0 template, 1 unverified,
//!   2 sourced.
//!
//! A calendar can be frozen: every method that changes it then raises. Freezing is
//! held here, on the kernel object, so no handle to it can bypass it; `copy` returns
//! an unfrozen copy.
//!
//! Errors are raised as `NativeCalendarError` (calendar arguments) and
//! `NativeSeriesError` (a series that fails validation, or an unusable timestamp); the
//! Python package converts them to its public error types. Type conversions live in
//! `convert` and the check report in `report`.

// Two lints fire inside PyO3 0.22 macro expansions on newer toolchains, not in this
// code: `create_exception!` checks `cfg(feature = "gil-refs")`, which this crate does
// not declare, and `#[pymethods]` and `#[pyfunction]` convert the error of every
// `PyResult` return with `.into()`.
#![allow(unexpected_cfgs, clippy::useless_conversion)]

mod convert;
mod report;

use numpy::PyReadonlyArray1;
use oa_bt::market::calendar as cal;
use oa_bt::model::{Exchange, MAX_DAY, MIN_DAY};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use convert::{
    calendar_owners, exchange_codes, grid_columns, intraday_times_tuple, parse_coverage,
    parse_exchanges, rule_tuple, session_columns, session_tuple, sessions_per_year_tuple,
    GridColumns, IntradayTimesTuple, NativeCalendarError, RuleTuple, SessionColumns, SessionTuple,
    SessionsPerYearTuple,
};
use report::check_report;

pub(super) use convert::{calendar_error, parse_exchange, series_error, NativeSeriesError, PyDay};

/// Session calendar for every OpenAlgo exchange (see `oa_bt::market::calendar`).
#[pyclass(name = "Calendar", module = "openalgo._oaindicators.backtest")]
pub struct NativeCalendar {
    pub(super) inner: cal::Calendar,
    frozen: bool,
}

impl NativeCalendar {
    /// The calendar for a change, or an error when it is frozen.
    fn editable(&mut self) -> PyResult<&mut cal::Calendar> {
        if self.frozen {
            return Err(NativeCalendarError::new_err(
                "this calendar is read-only (it is shared): call .copy(), or create a new \
                 Calendar, to get one you can change",
            ));
        }
        Ok(&mut self.inner)
    }
}

fn exchange_and_day(exchange: &str, day: PyDay) -> PyResult<(Exchange, oa_bt::model::Day)> {
    Ok((parse_exchange(exchange)?, day.0))
}

#[pymethods]
impl NativeCalendar {
    /// A calendar with the built-in regular hours and no holiday data.
    #[new]
    fn new() -> PyResult<Self> {
        let inner = cal::Calendar::new().map_err(calendar_error)?;
        Ok(NativeCalendar {
            inner,
            frozen: false,
        })
    }

    /// An unfrozen copy.
    fn copy(&self) -> Self {
        NativeCalendar {
            inner: self.inner.clone(),
            frozen: false,
        }
    }

    /// Make this calendar read-only. It cannot be unfrozen.
    fn freeze(&mut self) {
        self.frozen = true;
    }

    #[getter]
    fn frozen(&self) -> bool {
        self.frozen
    }

    // ----- regular hours, dated rules and coverage -------------------------

    /// Set regular hours from `effective_from` onward, or for every date when it is
    /// None.
    #[pyo3(signature = (exchange, effective_from, open_sec, close_sec))]
    fn set_regular_hours(
        &mut self,
        exchange: &str,
        effective_from: Option<PyDay>,
        open_sec: u32,
        close_sec: u32,
    ) -> PyResult<()> {
        let exchange = parse_exchange(exchange)?;
        let from = effective_from.map(|day| day.0);
        self.editable()?
            .set_regular_hours(exchange, from, open_sec, close_sec)
            .map_err(calendar_error)
    }

    fn set_closed(&mut self, exchange: &str, day: PyDay, coverage: u8) -> PyResult<()> {
        let (exchange, day) = exchange_and_day(exchange, day)?;
        let coverage = parse_coverage(coverage)?;
        self.editable()?
            .set_closed(exchange, day, coverage)
            .map_err(calendar_error)
    }

    /// Give an exchange a session on a date. A `close_sec` of None closes at the
    /// regular close of the date, whatever the regular hours are when asked.
    #[pyo3(signature = (exchange, day, open_sec, close_sec, hours_unverified, coverage))]
    fn set_session(
        &mut self,
        exchange: &str,
        day: PyDay,
        open_sec: u32,
        close_sec: Option<u32>,
        hours_unverified: bool,
        coverage: u8,
    ) -> PyResult<()> {
        let (exchange, day) = exchange_and_day(exchange, day)?;
        let coverage = parse_coverage(coverage)?;
        let close = close_sec.map_or(cal::RuleClose::Regular, cal::RuleClose::At);
        self.editable()?
            .set_session(exchange, day, open_sec, close, hours_unverified, coverage)
            .map_err(calendar_error)
    }

    /// Dated rules of an exchange from `from_day` to `to_day` (inclusive), in date order.
    fn rules(&self, exchange: &str, from_day: PyDay, to_day: PyDay) -> PyResult<Vec<RuleTuple>> {
        let rules = self
            .inner
            .rules_in(parse_exchange(exchange)?, from_day.0, to_day.0);
        Ok(rules
            .into_iter()
            .map(|(day, rule)| rule_tuple(day, rule))
            .collect())
    }

    fn set_year_coverage(&mut self, exchange: &str, year: i32, coverage: u8) -> PyResult<()> {
        let exchange = parse_exchange(exchange)?;
        let coverage = parse_coverage(coverage)?;
        self.editable()?.set_year_coverage(exchange, year, coverage);
        Ok(())
    }

    fn year_coverage(&self, exchange: &str, year: i32) -> PyResult<u8> {
        Ok(self.inner.year_coverage(parse_exchange(exchange)?, year) as u8)
    }

    fn coverage(&self, exchange: &str, day: PyDay) -> PyResult<u8> {
        let (exchange, day) = exchange_and_day(exchange, day)?;
        Ok(self.inner.coverage(exchange, day) as u8)
    }

    // ----- sessions --------------------------------------------------------

    /// The session on a date, or None when the exchange is closed.
    fn session(&self, exchange: &str, day: PyDay) -> PyResult<Option<SessionTuple>> {
        let (exchange, day) = exchange_and_day(exchange, day)?;
        Ok(self.inner.session(exchange, day).map(session_tuple))
    }

    /// Sessions from `from_day` to `to_day` (inclusive) as columns.
    fn sessions(
        &self,
        py: Python<'_>,
        exchange: &str,
        from_day: PyDay,
        to_day: PyDay,
    ) -> PyResult<SessionColumns> {
        let exchange = parse_exchange(exchange)?;
        let sessions = self
            .inner
            .sessions(exchange, from_day.0, to_day.0)
            .map_err(calendar_error)?;
        Ok(session_columns(py, &sessions))
    }

    /// The annualisation factor of the union calendar of the exchanges.
    fn sessions_per_year(
        &self,
        exchanges: Vec<String>,
        from_day: PyDay,
        to_day: PyDay,
    ) -> PyResult<SessionsPerYearTuple> {
        let exchanges = parse_exchanges(&exchanges)?;
        let result = self
            .inner
            .sessions_per_year(&exchanges, from_day.0, to_day.0)
            .map_err(calendar_error)?;
        Ok(sessions_per_year_tuple(result))
    }

    /// The union-calendar grid as columns. Mask bit `i` and the i-th bar-close array
    /// stand for the i-th distinct exchange given. `base_nanos` of None is the daily
    /// base.
    #[pyo3(signature = (exchanges, from_day, to_day, base_nanos))]
    fn union_grid(
        &self,
        py: Python<'_>,
        exchanges: Vec<String>,
        from_day: PyDay,
        to_day: PyDay,
        base_nanos: Option<i64>,
    ) -> PyResult<GridColumns> {
        let exchanges = parse_exchanges(&exchanges)?;
        let base = base_nanos.map_or(cal::BaseInterval::Daily, cal::BaseInterval::Intraday);
        let grid = cal::union_grid(&self.inner, &exchanges, from_day.0, to_day.0, base)
            .map_err(calendar_error)?;
        Ok(grid_columns(py, grid))
    }

    /// Default intraday times on a date, or None when the exchange is closed.
    fn intraday_times(&self, exchange: &str, day: PyDay) -> PyResult<Option<IntradayTimesTuple>> {
        let (exchange, day) = exchange_and_day(exchange, day)?;
        let times = cal::default_intraday_times(&self.inner, exchange, day);
        Ok(times.map(intraday_times_tuple))
    }

    // ----- series validation -----------------------------------------------

    /// Count how a sorted intraday series sits against the calendar (see
    /// `report::check_report` for the keys). Data problems are counted, never raised.
    fn check_intraday<'py>(
        &self,
        py: Python<'py>,
        exchange: &str,
        ts_open: PyReadonlyArray1<'py, i64>,
        interval_nanos: i64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let exchange = parse_exchange(exchange)?;
        let check = cal::check_intraday(&self.inner, exchange, ts_open.as_slice()?, interval_nanos)
            .map_err(series_error)?;
        check_report(py, check)
    }

    /// Check a sorted intraday series and raise `NativeSeriesError` on a 5 hour 30
    /// minute offset, bars off the lattice and, unless allowed, bars on closed days.
    fn validate_intraday<'py>(
        &self,
        py: Python<'py>,
        exchange: &str,
        ts_open: PyReadonlyArray1<'py, i64>,
        interval_nanos: i64,
        allow_closed_days: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let exchange = parse_exchange(exchange)?;
        let options = cal::ValidateOptions { allow_closed_days };
        let ts_open = ts_open.as_slice()?;
        let check = cal::validate_intraday(&self.inner, exchange, ts_open, interval_nanos, options)
            .map_err(series_error)?;
        check_report(py, check)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let (all, own_calendar) = exchange_codes();
    m.add("EXCHANGE_CODES", all)?;
    m.add("CALENDAR_EXCHANGE_CODES", own_calendar)?;
    m.add("CALENDAR_OWNERS", calendar_owners())?;
    m.add("MIN_DAY", MIN_DAY)?;
    m.add("MAX_DAY", MAX_DAY)?;
    m.add("NO_BAR", cal::NO_BAR)?;
    m.add_class::<NativeCalendar>()?;
    m.add(
        "NativeCalendarError",
        py.get_type_bound::<NativeCalendarError>(),
    )?;
    m.add(
        "NativeSeriesError",
        py.get_type_bound::<NativeSeriesError>(),
    )?;
    Ok(())
}
