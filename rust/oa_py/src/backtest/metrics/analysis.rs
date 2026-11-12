//! Bindings for the distribution, trade-ratio, rolling and path statistics of
//! `oa_bt::metrics`: `metrics_sessions`, `metrics_gain_to_pain`, `metrics_trade_ratios`,
//! `metrics_rolling` and `metrics_path`. Conventions are those of the parent module.

use numpy::IntoPyArray;
use oa_bt::metrics as mt;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::{metrics_error, Times, Values};

fn set_figures(out: &Bound<'_, PyDict>, figures: &[(&str, f64)]) -> PyResult<()> {
    for (key, value) in figures {
        out.set_item(*key, *value)?;
    }
    Ok(())
}

fn session_counts(out: &Bound<'_, PyDict>, s: &mt::SessionStats) -> PyResult<()> {
    set_figures(
        out,
        &[
            ("expected_return", s.expected_return),
            ("average_return", s.average_return),
            ("best", s.best),
            ("worst", s.worst),
            ("win_rate", s.win_rate),
            ("average_winning", s.average_winning),
            ("average_losing", s.average_losing),
        ],
    )?;
    out.set_item("winning", s.winning)?;
    out.set_item("losing", s.losing)?;
    out.set_item("max_consecutive_wins", s.max_consecutive_wins)?;
    out.set_item("max_consecutive_losses", s.max_consecutive_losses)?;
    out.set_item("loss_streaks", s.loss_streaks.clone())
}

fn adjusted(out: &Bound<'_, PyDict>, a: &mt::AdjustedRatios) -> PyResult<()> {
    set_figures(
        out,
        &[
            ("autocorrelation", a.autocorrelation),
            ("autocorrelation_penalty", a.autocorrelation_penalty),
            ("smart_sharpe", a.smart_sharpe),
            ("smart_sortino", a.smart_sortino),
            ("adjusted_sortino", a.adjusted_sortino),
            ("probabilistic_sharpe", a.probabilistic_sharpe),
            ("probabilistic_sortino", a.probabilistic_sortino),
            ("probabilistic_adjusted_sortino", a.probabilistic_adjusted_sortino),
            ("skew", a.skew),
            ("kurtosis", a.kurtosis),
            ("risk_return_ratio", a.risk_return_ratio),
        ],
    )
}

fn tails(out: &Bound<'_, PyDict>, t: &mt::TailStats) -> PyResult<()> {
    set_figures(
        out,
        &[
            ("value_at_risk", t.value_at_risk),
            ("expected_shortfall", t.expected_shortfall),
            ("tail_ratio", t.tail_ratio),
            ("outlier_win_ratio", t.outlier_win_ratio),
            ("outlier_loss_ratio", t.outlier_loss_ratio),
            ("omega", t.omega),
        ],
    )
}

/// Every statistic of session returns in date order: counting statistics, corrected
/// ratios, tails (Omega at the per-session risk-free rate) and the ulcer and serenity
/// indexes, which take the run's `cagr` (see `oa_bt::metrics::sessions`, `adjusted`,
/// `tails`, `ulcer`). Losing streaks come back as a list of lengths.
#[pyfunction]
fn metrics_sessions<'py>(
    py: Python<'py>,
    returns: Values<'py>,
    periods_per_year: f64,
    risk_free_annual: f64,
    cagr: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let returns = returns.as_slice()?;
    let stats = mt::return_stats(returns, periods_per_year, risk_free_annual);
    let stats = stats.map_err(metrics_error)?;
    let counts = mt::session_stats(returns).map_err(metrics_error)?;
    let ratios = mt::adjusted_ratios(returns, periods_per_year, risk_free_annual);
    let tail = mt::tail_stats(returns, stats.risk_free_per_period).map_err(metrics_error)?;
    let closes = mt::close_drawdowns(returns);
    let ulcer = mt::ulcer_index(&closes);
    let cdar = mt::conditional_drawdown(&closes);
    let out = PyDict::new_bound(py);
    session_counts(&out, &counts)?;
    adjusted(&out, &ratios.map_err(metrics_error)?)?;
    tails(&out, &tail)?;
    set_figures(
        &out,
        &[
            ("ulcer_index", ulcer),
            ("conditional_drawdown", cdar),
            (
                "ulcer_performance_index",
                mt::ulcer_performance_index(cagr, risk_free_annual, ulcer),
            ),
            (
                "serenity_index",
                mt::serenity_index(cagr, risk_free_annual, ulcer, cdar, stats.volatility),
            ),
        ],
    )?;
    Ok(out)
}

/// Gain to pain of monthly returns (fractions).
#[pyfunction]
fn metrics_gain_to_pain(monthly_returns: Values<'_>) -> PyResult<f64> {
    mt::gain_to_pain(monthly_returns.as_slice()?).map_err(metrics_error)
}

/// Win/loss ratio, Kelly fraction, risk of ruin from `capital` rupees, CPC index and
/// losing streaks of closed trades (see `oa_bt::metrics::trade_ratios`).
#[pyfunction]
fn metrics_trade_ratios<'py>(
    py: Python<'py>,
    entry_ts: Times<'py>,
    exit_ts: Times<'py>,
    gross_pnl: Values<'py>,
    net_pnl: Values<'py>,
    costs: Values<'py>,
    capital: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let columns = mt::TradeColumns {
        entry_ts: entry_ts.as_slice()?,
        exit_ts: exit_ts.as_slice()?,
        gross_pnl: gross_pnl.as_slice()?,
        net_pnl: net_pnl.as_slice()?,
        costs: costs.as_slice()?,
    };
    let r = mt::trade_ratios(&columns, capital).map_err(metrics_error)?;
    let out = PyDict::new_bound(py);
    set_figures(
        &out,
        &[
            ("win_loss_ratio", r.win_loss_ratio),
            ("kelly_criterion", r.kelly_criterion),
            ("risk_of_ruin", r.risk_of_ruin),
            ("cpc_index", r.cpc_index),
        ],
    )?;
    out.set_item("loss_streaks", r.loss_streaks)?;
    Ok(out)
}

/// Rolling volatility, Sharpe, Sortino, win rate and (with a benchmark) beta over
/// windows of `window` sessions, as float64 arrays (see `oa_bt::metrics::rolling`).
#[pyfunction]
#[pyo3(signature = (returns, benchmark, window, periods_per_year, risk_free_annual))]
fn metrics_rolling<'py>(
    py: Python<'py>,
    returns: Values<'py>,
    benchmark: Option<Values<'py>>,
    window: usize,
    periods_per_year: f64,
    risk_free_annual: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let bench = benchmark.as_ref().map(|b| b.as_slice()).transpose()?;
    let r = mt::rolling_stats(
        returns.as_slice()?,
        bench,
        window,
        periods_per_year,
        risk_free_annual,
    )
    .map_err(metrics_error)?;
    let out = PyDict::new_bound(py);
    out.set_item("volatility", r.volatility.into_pyarray_bound(py))?;
    out.set_item("sharpe", r.sharpe.into_pyarray_bound(py))?;
    out.set_item("sortino", r.sortino.into_pyarray_bound(py))?;
    out.set_item("win_rate", r.win_rate.into_pyarray_bound(py))?;
    if bench.is_some() {
        out.set_item("beta", r.beta.into_pyarray_bound(py))?;
    }
    Ok(out)
}

/// Total return, CAGR over `elapsed` nanoseconds, volatility, Sharpe, Sortino and the
/// deepest close drawdown of a path of session returns (see `oa_bt::metrics::ulcer`).
#[pyfunction]
fn metrics_path<'py>(
    py: Python<'py>,
    returns: Values<'py>,
    periods_per_year: f64,
    risk_free_annual: f64,
    elapsed: i64,
) -> PyResult<Bound<'py, PyDict>> {
    let s = mt::path_stats(
        returns.as_slice()?,
        periods_per_year,
        risk_free_annual,
        elapsed,
    )
    .map_err(metrics_error)?;
    let out = PyDict::new_bound(py);
    set_figures(
        &out,
        &[
            ("total_return", s.total_return),
            ("cagr", s.cagr),
            ("volatility", s.volatility),
            ("sharpe", s.sharpe),
            ("sortino", s.sortino),
            ("max_drawdown", s.max_drawdown),
        ],
    )?;
    Ok(out)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("METRICS_TAIL_PROBABILITY", mt::TAIL_PROBABILITY)?;
    m.add("METRICS_OUTLIER_PROBABILITY", mt::OUTLIER_PROBABILITY)?;
    m.add_function(wrap_pyfunction!(metrics_sessions, m)?)?;
    m.add_function(wrap_pyfunction!(metrics_gain_to_pain, m)?)?;
    m.add_function(wrap_pyfunction!(metrics_trade_ratios, m)?)?;
    m.add_function(wrap_pyfunction!(metrics_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(metrics_path, m)?)?;
    Ok(())
}
