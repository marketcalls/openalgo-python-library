//! Conversions between the calendar's Rust types and what crosses to Python.

use numpy::{IntoPyArray, PyArray1};
use oa_bt::market::calendar as cal;
use oa_bt::model::{is_supported_day, Day, Exchange};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(
    backtest,
    NativeCalendarError,
    PyException,
    "A calendar argument is invalid (unknown exchange, bad session times, bad range)."
);
create_exception!(
    backtest,
    NativeSeriesError,
    PyException,
    "A series does not fit the session calendar, or holds an unusable timestamp."
);

pub(in crate::backtest) fn calendar_error(error: cal::CalendarError) -> PyErr {
    NativeCalendarError::new_err(error.to_string())
}

pub(in crate::backtest) fn series_error(error: cal::SeriesError) -> PyErr {
    NativeSeriesError::new_err(error.to_string())
}

/// An IST date from Python as days since 1970-01-01, checked against the supported
/// range (`oa_bt::model::MIN_DAY` to `MAX_DAY`) so that no instant derived from it can
/// overflow. Anything else is a `NativeCalendarError`.
#[derive(Clone, Copy, Debug)]
pub(in crate::backtest) struct PyDay(pub(in crate::backtest) Day);

impl<'py> FromPyObject<'py> for PyDay {
    fn extract_bound(value: &Bound<'py, PyAny>) -> PyResult<Self> {
        let number: i64 = value.extract()?;
        let day = number.clamp(i64::from(Day::MIN), i64::from(Day::MAX)) as Day;
        if i64::from(day) == number && is_supported_day(day) {
            Ok(PyDay(day))
        } else {
            Err(calendar_error(cal::CalendarError::UnsupportedDay(day)))
        }
    }
}

/// Parse an OpenAlgo exchange string.
pub(in crate::backtest) fn parse_exchange(code: &str) -> PyResult<Exchange> {
    Exchange::from_code(code)
        .ok_or_else(|| calendar_error(cal::CalendarError::UnknownExchange(code.to_string())))
}

pub(super) fn parse_exchanges(codes: &[String]) -> PyResult<Vec<Exchange>> {
    codes.iter().map(|code| parse_exchange(code)).collect()
}

/// Parse a coverage code: 0 template, 1 unverified, 2 sourced.
pub(super) fn parse_coverage(code: u8) -> PyResult<cal::Coverage> {
    cal::Coverage::from_u8(code).ok_or_else(|| {
        NativeCalendarError::new_err(format!(
            "invalid coverage level {code}: use 0 (template), 1 (unverified) or 2 (sourced)"
        ))
    })
}

/// A session as (day, open, close, kind, coverage, hours_unverified).
pub(super) type SessionTuple = (Day, i64, i64, u8, u8, bool);

pub(super) fn session_tuple(session: cal::Session) -> SessionTuple {
    (
        session.day,
        session.open,
        session.close,
        session.kind as u8,
        session.coverage as u8,
        session.hours_unverified,
    )
}

/// Sessions as columns, in the field order of `SessionTuple`.
pub(super) type SessionColumns = (
    Py<PyArray1<i32>>,
    Py<PyArray1<i64>>,
    Py<PyArray1<i64>>,
    Py<PyArray1<u8>>,
    Py<PyArray1<u8>>,
    Py<PyArray1<bool>>,
);

pub(super) fn session_columns(py: Python<'_>, sessions: &[cal::Session]) -> SessionColumns {
    fn column<T: numpy::Element>(py: Python<'_>, values: Vec<T>) -> Py<PyArray1<T>> {
        values.into_pyarray_bound(py).unbind()
    }
    (
        column(py, sessions.iter().map(|s| s.day).collect()),
        column(py, sessions.iter().map(|s| s.open).collect()),
        column(py, sessions.iter().map(|s| s.close).collect()),
        column(py, sessions.iter().map(|s| s.kind as u8).collect()),
        column(py, sessions.iter().map(|s| s.coverage as u8).collect()),
        column(py, sessions.iter().map(|s| s.hours_unverified).collect()),
    )
}

/// A dated rule as (day, window, hours_unverified, coverage). The window is None when
/// the exchange is closed, else (open_sec, close_sec) with a close of None meaning the
/// regular close of the date.
pub(super) type RuleTuple = (Day, Option<(u32, Option<u32>)>, bool, u8);

pub(super) fn rule_tuple(day: Day, rule: cal::DayRule) -> RuleTuple {
    let window = rule.window.map(|(open_sec, close)| {
        let close_sec = match close {
            cal::RuleClose::At(sec) => Some(sec),
            cal::RuleClose::Regular => None,
        };
        (open_sec, close_sec)
    });
    (day, window, rule.hours_unverified, rule.coverage as u8)
}

/// The union grid as columns: ts, day, active, session_open, session_close and one
/// bar-close array per exchange, in the order of `exchanges`.
pub(super) type GridColumns = (
    Py<PyArray1<i64>>,
    Py<PyArray1<i32>>,
    Py<PyArray1<u16>>,
    Py<PyArray1<u16>>,
    Py<PyArray1<u16>>,
    Vec<Py<PyArray1<i64>>>,
);

pub(super) fn grid_columns(py: Python<'_>, grid: cal::UnionGrid) -> GridColumns {
    let closes = grid
        .bar_close
        .into_iter()
        .map(|c| c.into_pyarray_bound(py).unbind())
        .collect();
    (
        grid.ts.into_pyarray_bound(py).unbind(),
        grid.day.into_pyarray_bound(py).unbind(),
        grid.active.into_pyarray_bound(py).unbind(),
        grid.session_open.into_pyarray_bound(py).unbind(),
        grid.session_close.into_pyarray_bound(py).unbind(),
        closes,
    )
}

/// The annualisation factor as (value, sessions, calendar_days, coverage,
/// template_only), template_only being (exchange code, year) pairs.
pub(super) type SessionsPerYearTuple = (f64, u32, i64, u8, Vec<(&'static str, i32)>);

pub(super) fn sessions_per_year_tuple(result: cal::SessionsPerYear) -> SessionsPerYearTuple {
    let template_only = result
        .template_only
        .iter()
        .map(|&(exchange, year)| (exchange.code(), year))
        .collect();
    (
        result.value,
        result.sessions,
        result.calendar_days,
        result.coverage as u8,
        template_only,
    )
}

/// Default intraday times as (entry_until, square_off, session_open, session_close,
/// relative_to_close, unverified), instants in UTC nanoseconds.
pub(super) type IntradayTimesTuple = (i64, i64, i64, i64, bool, bool);

pub(super) fn intraday_times_tuple(times: cal::IntradayTimes) -> IntradayTimesTuple {
    (
        times.entry_until,
        times.square_off,
        times.session_open,
        times.session_close,
        times.relative_to_close,
        times.unverified,
    )
}

/// Every OpenAlgo exchange code with the code of the exchange whose calendar it
/// follows (itself, or NSE and BSE for the index exchanges).
pub(super) fn calendar_owners() -> std::collections::HashMap<&'static str, &'static str> {
    Exchange::ALL
        .iter()
        .map(|&exchange| (exchange.code(), cal::calendar_exchange(exchange).code()))
        .collect()
}

/// The OpenAlgo exchange codes, and those of them that have their own calendar (the
/// index exchanges follow NSE and BSE).
pub(super) fn exchange_codes() -> (Vec<&'static str>, Vec<&'static str>) {
    let all: Vec<&'static str> = Exchange::ALL.iter().map(|e| e.code()).collect();
    let own_calendar: Vec<&'static str> = Exchange::ALL
        .iter()
        .filter(|&&exchange| cal::calendar_exchange(exchange) == exchange)
        .map(|e| e.code())
        .collect();
    (all, own_calendar)
}
