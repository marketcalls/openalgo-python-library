//! Bindings for `oa_bt::metrics`.
//!
//! Conventions at the boundary:
//! - Timestamps are contiguous int64 arrays of UTC nanoseconds; money, returns and
//!   rates are contiguous float64 arrays (rupees and fractions).
//! - Session dates come back as an int32 array of IST days since 1970-01-01.
//! - A drawdown episode is a tuple (peak or None, trough, recovery or None, depth,
//!   duration): point positions (a peak of None is the opening wealth, before the first
//!   point), the depth as a fraction and the duration in nanoseconds.
//! - An opening wealth (rupees, the capital before the first point) is an optional
//!   last argument; None takes the first point as the start (`oa_bt::metrics::curve`).
//! - Several results come back as a dict whose keys are the kernel's field names.
//!
//! Errors are raised as `NativeMetricsError`; the Python package converts them to its
//! public error type.

// Two lints fire inside PyO3 0.22 macro expansions on newer toolchains, not in this
// code: `create_exception!` checks `cfg(feature = "gil-refs")`, which this crate does
// not declare, and `#[pyfunction]` converts the error of every `PyResult` return with
// `.into()`.
#![allow(unexpected_cfgs, clippy::useless_conversion)]

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use oa_bt::metrics as mt;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyDict;

create_exception!(
    backtest,
    NativeMetricsError,
    PyException,
    "The input of a metric is unusable (malformed series, mismatched trade P&L, bad rate)."
);

fn metrics_error(error: mt::MetricsError) -> PyErr {
    NativeMetricsError::new_err(error.to_string())
}

type Times<'py> = PyReadonlyArray1<'py, i64>;
type Values<'py> = PyReadonlyArray1<'py, f64>;
type EpisodeTuple = (Option<usize>, usize, Option<usize>, f64, i64);
/// A combined equity series: (int64 UTC nanoseconds, float64 rupees).
type CombinedColumns<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>);

fn episode(value: Option<mt::DrawdownEpisode>) -> Option<EpisodeTuple> {
    value.map(|e| (e.peak, e.trough, e.recovery, e.depth, e.duration))
}

/// Wealth index, cumulative P&L, booked flows, ruin point, session returns and
/// drawdowns of an equity series with its cash flows and optional opening wealth (see
/// `oa_bt::metrics::curve`).
#[pyfunction]
#[pyo3(signature = (ts, equity, flow_ts, flow_amount, opening_wealth=None))]
fn metrics_curve<'py>(
    py: Python<'py>,
    ts: Times<'py>,
    equity: Values<'py>,
    flow_ts: Times<'py>,
    flow_amount: Values<'py>,
    opening_wealth: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let ts = ts.as_slice()?;
    let curve = mt::EquityCurve {
        ts,
        equity: equity.as_slice()?,
    };
    let flows = mt::CashFlows {
        ts: flow_ts.as_slice()?,
        amount: flow_amount.as_slice()?,
    };
    let wealth = mt::wealth(&curve, &flows, opening_wealth).map_err(metrics_error)?;
    let sessions = mt::session_returns(ts, &wealth);
    let drawdowns = mt::drawdowns(ts, &wealth.index, &wealth.pnl);
    let out = PyDict::new_bound(py);
    out.set_item("ruin", wealth.ruin)?;
    out.set_item("wealth", wealth.index.into_pyarray_bound(py))?;
    out.set_item("pnl", wealth.pnl.into_pyarray_bound(py))?;
    out.set_item("flow", wealth.flow.into_pyarray_bound(py))?;
    out.set_item("session_days", sessions.days.into_pyarray_bound(py))?;
    out.set_item("session_returns", sessions.returns.into_pyarray_bound(py))?;
    out.set_item("drawdown", drawdowns.series.into_pyarray_bound(py))?;
    out.set_item("max_drawdown", episode(drawdowns.max))?;
    out.set_item("longest_drawdown", episode(drawdowns.longest))?;
    out.set_item("max_drawdown_amount", drawdowns.max_amount)?;
    Ok(out)
}

/// Volatility, Sharpe and Sortino of session returns (see `oa_bt::metrics::ratios`).
#[pyfunction]
fn metrics_returns<'py>(
    py: Python<'py>,
    returns: Values<'py>,
    periods_per_year: f64,
    risk_free_annual: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let stats = mt::return_stats(returns.as_slice()?, periods_per_year, risk_free_annual)
        .map_err(metrics_error)?;
    let out = PyDict::new_bound(py);
    out.set_item("risk_free_per_period", stats.risk_free_per_period)?;
    out.set_item("mean_excess", stats.mean_excess)?;
    out.set_item("volatility", stats.volatility)?;
    out.set_item("sharpe", stats.sharpe)?;
    out.set_item("sortino", stats.sortino)?;
    Ok(out)
}

/// CAGR of a growth factor over `elapsed` nanoseconds.
#[pyfunction]
fn metrics_cagr(growth: f64, elapsed: i64) -> f64 {
    mt::cagr(growth, elapsed)
}

/// Calmar ratio of a CAGR and a maximum drawdown (a fraction, zero or negative).
#[pyfunction]
fn metrics_calmar(cagr: f64, max_drawdown: f64) -> f64 {
    mt::calmar(cagr, max_drawdown)
}

/// Statistics of closed trades (see `oa_bt::metrics::trades`).
#[pyfunction]
fn metrics_trades<'py>(
    py: Python<'py>,
    entry_ts: Times<'py>,
    exit_ts: Times<'py>,
    gross_pnl: Values<'py>,
    net_pnl: Values<'py>,
    costs: Values<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let columns = mt::TradeColumns {
        entry_ts: entry_ts.as_slice()?,
        exit_ts: exit_ts.as_slice()?,
        gross_pnl: gross_pnl.as_slice()?,
        net_pnl: net_pnl.as_slice()?,
        costs: costs.as_slice()?,
    };
    let s = mt::trade_stats(&columns).map_err(metrics_error)?;
    let out = PyDict::new_bound(py);
    let counts = [
        ("trades", s.trades),
        ("wins", s.wins),
        ("losses", s.losses),
        ("max_consecutive_wins", s.max_consecutive_wins),
        ("max_consecutive_losses", s.max_consecutive_losses),
    ];
    for (key, value) in counts {
        out.set_item(key, value)?;
    }
    let figures = [
        ("win_rate", s.win_rate),
        ("average_win", s.average_win),
        ("average_loss", s.average_loss),
        ("profit_factor", s.profit_factor),
        ("expectancy", s.expectancy),
        ("payoff_ratio", s.payoff_ratio),
        ("largest_win", s.largest_win),
        ("largest_loss", s.largest_loss),
        ("average_holding", s.average_holding),
        ("gross_pnl", s.gross_pnl),
        ("net_pnl", s.net_pnl),
        ("total_costs", s.total_costs),
        ("cost_share_of_gross", s.cost_share_of_gross),
    ];
    for (key, value) in figures {
        out.set_item(key, value)?;
    }
    Ok(out)
}

/// Share of the equity series' bars that overlap a held position. An open trade
/// passes the int64 maximum as its exit.
#[pyfunction]
fn metrics_exposure(ts: Times<'_>, entry_ts: Times<'_>, exit_ts: Times<'_>) -> PyResult<f64> {
    mt::exposure(ts.as_slice()?, entry_ts.as_slice()?, exit_ts.as_slice()?).map_err(metrics_error)
}

/// Beta, alpha and correlation against a benchmark (see `oa_bt::metrics::benchmark`).
#[pyfunction]
fn metrics_benchmark<'py>(
    py: Python<'py>,
    strategy: Values<'py>,
    benchmark: Values<'py>,
    periods_per_year: f64,
    risk_free_annual: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let stats = mt::benchmark_stats(
        strategy.as_slice()?,
        benchmark.as_slice()?,
        periods_per_year,
        risk_free_annual,
    )
    .map_err(metrics_error)?;
    let out = PyDict::new_bound(py);
    out.set_item("alpha", stats.alpha)?;
    out.set_item("beta", stats.beta)?;
    out.set_item("correlation", stats.correlation)?;
    out.set_item("sessions", stats.sessions)?;
    Ok(out)
}

/// Combined equity of several (timestamps, equity) series: their sum on the union of
/// the timestamps, with each series at its opening wealth (one per series, rupees)
/// before its first point when those are given (see `oa_bt::metrics::combine`).
#[pyfunction]
#[pyo3(signature = (curves, opening_wealth=None))]
fn metrics_combine<'py>(
    py: Python<'py>,
    curves: Vec<(Times<'py>, Values<'py>)>,
    opening_wealth: Option<Vec<f64>>,
) -> PyResult<CombinedColumns<'py>> {
    let mut borrowed = Vec::with_capacity(curves.len());
    for (ts, equity) in &curves {
        borrowed.push(mt::EquityCurve {
            ts: ts.as_slice()?,
            equity: equity.as_slice()?,
        });
    }
    let (ts, equity) =
        mt::combine_equity(&borrowed, opening_wealth.as_deref()).map_err(metrics_error)?;
    Ok((ts.into_pyarray_bound(py), equity.into_pyarray_bound(py)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add(
        "NativeMetricsError",
        py.get_type_bound::<NativeMetricsError>(),
    )?;
    m.add("METRICS_DAYS_PER_YEAR", mt::DAYS_PER_YEAR)?;
    m.add("METRICS_MIN_CAGR_DAYS", mt::MIN_CAGR_DAYS)?;
    m.add("METRICS_PNL_TOLERANCE_RUPEES", mt::PNL_TOLERANCE_RUPEES)?;
    m.add_function(wrap_pyfunction!(metrics_curve, m)?)?;
    m.add_function(wrap_pyfunction!(metrics_returns, m)?)?;
    m.add_function(wrap_pyfunction!(metrics_cagr, m)?)?;
    m.add_function(wrap_pyfunction!(metrics_calmar, m)?)?;
    m.add_function(wrap_pyfunction!(metrics_trades, m)?)?;
    m.add_function(wrap_pyfunction!(metrics_exposure, m)?)?;
    m.add_function(wrap_pyfunction!(metrics_benchmark, m)?)?;
    m.add_function(wrap_pyfunction!(metrics_combine, m)?)?;
    Ok(())
}
