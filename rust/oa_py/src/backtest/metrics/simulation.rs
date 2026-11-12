//! Bindings for Monte Carlo resampling and percentiles of `oa_bt::metrics`:
//! `metrics_bootstrap_sessions`, `metrics_bootstrap_trades` and `metrics_percentiles`.
//!
//! A study's settings cross as one tuple `(simulations, seed, periods_per_year,
//! risk_free_annual, elapsed_ns)` (see `oa_bt::metrics::MonteCarloSpec`). The result is a
//! dict of float64 arrays, one value per simulation, plus the band: an int64 array of
//! draw counts and a float64 array of shape (points, 5), one column per band percentile.

use numpy::{IntoPyArray, PyArray2};
use oa_bt::metrics as mt;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::{metrics_error, Values};

/// (simulations, seed, periods per year, annual risk-free rate, elapsed nanoseconds).
type SpecTuple = (usize, u64, f64, f64, i64);

fn spec(value: SpecTuple) -> mt::MonteCarloSpec {
    let (simulations, seed, periods_per_year, risk_free_annual, elapsed) = value;
    mt::MonteCarloSpec {
        simulations,
        seed,
        periods_per_year,
        risk_free_annual,
        elapsed,
    }
}

fn runs<'py>(py: Python<'py>, runs: mt::MonteCarloRuns) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new_bound(py);
    out.set_item("final_return", runs.final_return.into_pyarray_bound(py))?;
    out.set_item("cagr", runs.cagr.into_pyarray_bound(py))?;
    out.set_item("max_drawdown", runs.max_drawdown.into_pyarray_bound(py))?;
    out.set_item("sharpe", runs.sharpe.into_pyarray_bound(py))?;
    let steps: Vec<i64> = runs.band_steps.iter().map(|&s| s as i64).collect();
    out.set_item("band_steps", steps.into_pyarray_bound(py))?;
    let rows: Vec<Vec<f64>> = runs.band.iter().map(|row| row.to_vec()).collect();
    out.set_item("band", PyArray2::from_vec2_bound(py, &rows)?)?;
    out.set_item("band_probabilities", mt::BAND_PROBABILITIES.to_vec())?;
    Ok(out)
}

/// Monte Carlo from session returns (fractions; see `oa_bt::metrics::montecarlo`).
#[pyfunction]
fn metrics_bootstrap_sessions<'py>(
    py: Python<'py>,
    returns: Values<'py>,
    settings: SpecTuple,
) -> PyResult<Bound<'py, PyDict>> {
    let result = mt::bootstrap_sessions(returns.as_slice()?, &spec(settings));
    runs(py, result.map_err(metrics_error)?)
}

/// Monte Carlo from trades' net P&L in rupees, booked on `capital` rupees.
#[pyfunction]
fn metrics_bootstrap_trades<'py>(
    py: Python<'py>,
    net_pnl: Values<'py>,
    capital: f64,
    settings: SpecTuple,
) -> PyResult<Bound<'py, PyDict>> {
    let result = mt::bootstrap_trades(net_pnl.as_slice()?, capital, &spec(settings));
    runs(py, result.map_err(metrics_error)?)
}

/// Quantiles of `values` (NaN left out) at each probability, by linear interpolation
/// between order statistics (see `oa_bt::metrics::moments`).
#[pyfunction]
fn metrics_percentiles<'py>(
    py: Python<'py>,
    values: Values<'py>,
    probabilities: Values<'py>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let result = mt::percentiles(values.as_slice()?, probabilities.as_slice()?);
    Ok(result.map_err(metrics_error)?.into_pyarray_bound(py))
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("METRICS_MAX_BAND_POINTS", mt::MAX_BAND_POINTS)?;
    m.add_function(wrap_pyfunction!(metrics_bootstrap_sessions, m)?)?;
    m.add_function(wrap_pyfunction!(metrics_bootstrap_trades, m)?)?;
    m.add_function(wrap_pyfunction!(metrics_percentiles, m)?)?;
    Ok(())
}
