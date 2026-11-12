//! Monte Carlo by bootstrap: many alternative runs drawn from the run's own session
//! returns or trades, and the spread of their outcomes.
//!
//! Method (the bootstrap of Efron 1979, "Bootstrap methods: another look at the
//! jackknife", Annals of Statistics 7(1)): each simulation draws, with replacement and
//! with equal probability, as many items as the run had, in random order.
//!
//! - From session returns: the drawn returns are the path's session returns.
//! - From trades: the drawn net P&L in rupees are booked one after another on the
//!   capital at fixed rupee size (trades are sized in units, not as a share of equity);
//!   the path's return at each trade is its P&L over the equity before it. A path whose
//!   equity reaches zero or below is ruined there.
//!
//! Each path is measured as `ulcer::path_stats` measures one: total return, CAGR over
//! the run's elapsed time, maximum drawdown on the path's closes, and Sharpe annualised
//! by the given periods per year (sessions per year, or for trades the run's trades per
//! year). The band is the cumulative return of the paths after each number of draws, at
//! the 5th, 25th, 50th, 75th and 95th percentiles (`moments` quantiles), at no more than
//! `MAX_BAND_POINTS` evenly spaced numbers of draws from 0 to all of them.
//!
//! The generator is `rng::SplitMix64`, seeded by the caller: the same inputs and seed
//! give the same simulations, bit for bit.

use super::curve::check_finite;
use super::error::MetricsError;
use super::moments::percentiles;
use super::rng::SplitMix64;
use super::ulcer::path_stats;
use crate::model::TsNanos;

/// Probabilities of the band's percentiles.
pub const BAND_PROBABILITIES: [f64; 5] = [0.05, 0.25, 0.5, 0.75, 0.95];

/// Most points of the band: enough for a smooth chart, few enough for any run length.
pub const MAX_BAND_POINTS: usize = 201;

/// How many paths to draw and how to measure them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MonteCarloSpec {
    /// Number of simulated paths, at least 1.
    pub simulations: usize,
    /// Seed of the generator.
    pub seed: u64,
    /// Draws per year, which annualises the Sharpe ratio of a path.
    pub periods_per_year: f64,
    /// Annual risk-free rate, a fraction.
    pub risk_free_annual: f64,
    /// Elapsed time of the run, which annualises CAGR, in nanoseconds.
    pub elapsed: TsNanos,
}

/// The outcome of every simulated path, in simulation order, and the band.
#[derive(Clone, Debug, PartialEq)]
pub struct MonteCarloRuns {
    /// Fractions.
    pub final_return: Vec<f64>,
    /// Fractions per year.
    pub cagr: Vec<f64>,
    /// Fractions from 0 down to -1.
    pub max_drawdown: Vec<f64>,
    pub sharpe: Vec<f64>,
    /// The number of draws at each band point, from 0 to the run's count.
    pub band_steps: Vec<usize>,
    /// Cumulative return (a fraction) at each band point, one value per probability of
    /// `BAND_PROBABILITIES`.
    pub band: Vec<[f64; BAND_PROBABILITIES.len()]>,
}

fn band_steps(draws: usize) -> Vec<usize> {
    if draws < MAX_BAND_POINTS {
        return (0..=draws).collect();
    }
    let last = (MAX_BAND_POINTS - 1) as f64;
    (0..MAX_BAND_POINTS)
        .map(|j| (j as f64 * draws as f64 / last).round() as usize)
        .collect()
}

fn check_spec(spec: &MonteCarloSpec, items: usize) -> Result<(), MetricsError> {
    if spec.simulations == 0 {
        return Err(MetricsError::NoSimulations);
    }
    if items == 0 {
        return Err(MetricsError::NothingToResample);
    }
    Ok(())
}

/// Draw `spec.simulations` paths of `draws` items with `fill`, which writes one path's
/// returns into the buffer it is given, and measure them.
fn simulate(
    spec: &MonteCarloSpec,
    draws: usize,
    mut fill: impl FnMut(&mut SplitMix64, &mut Vec<f64>),
) -> Result<MonteCarloRuns, MetricsError> {
    let steps = band_steps(draws);
    let mut rng = SplitMix64::new(spec.seed);
    let mut at_step: Vec<Vec<f64>> = vec![Vec::with_capacity(spec.simulations); steps.len()];
    let mut runs = MonteCarloRuns {
        final_return: Vec::with_capacity(spec.simulations),
        cagr: Vec::with_capacity(spec.simulations),
        max_drawdown: Vec::with_capacity(spec.simulations),
        sharpe: Vec::with_capacity(spec.simulations),
        band_steps: steps.clone(),
        band: Vec::with_capacity(steps.len()),
    };
    let mut path = Vec::with_capacity(draws);
    for _ in 0..spec.simulations {
        path.clear();
        fill(&mut rng, &mut path);
        let stats = path_stats(
            &path,
            spec.periods_per_year,
            spec.risk_free_annual,
            spec.elapsed,
        )?;
        runs.final_return.push(stats.total_return);
        runs.cagr.push(stats.cagr);
        runs.max_drawdown.push(stats.max_drawdown);
        runs.sharpe.push(stats.sharpe);
        let mut wealth = 1.0_f64;
        let mut drawn = 0;
        for (column, &step) in at_step.iter_mut().zip(&steps) {
            while drawn < step {
                // Past the path's end (a ruined trade path) the wealth stays where it is.
                let r = path.get(drawn).copied().unwrap_or(0.0);
                wealth = (wealth * (1.0 + r)).max(0.0);
                drawn += 1;
            }
            column.push(wealth - 1.0);
        }
    }
    for column in &at_step {
        let mut row = [f64::NAN; BAND_PROBABILITIES.len()];
        for (slot, value) in row
            .iter_mut()
            .zip(percentiles(column, &BAND_PROBABILITIES)?)
        {
            *slot = value;
        }
        runs.band.push(row);
    }
    Ok(runs)
}

/// Monte Carlo from session returns (fractions; module docs).
///
/// Errors: `NoSimulations`; `NothingToResample` for no return; `NotFinite` for a return
/// that is NaN or infinite; those of `ratios::return_stats`.
pub fn bootstrap_sessions(
    returns: &[f64],
    spec: &MonteCarloSpec,
) -> Result<MonteCarloRuns, MetricsError> {
    check_spec(spec, returns.len())?;
    check_finite(returns, "session return")?;
    simulate(spec, returns.len(), |rng, path| {
        for _ in 0..returns.len() {
            path.push(
                returns
                    .get(rng.below(returns.len()))
                    .copied()
                    .unwrap_or(0.0),
            );
        }
    })
}

/// Monte Carlo from trades' net P&L in rupees on `capital` rupees (module docs).
///
/// Errors: `NoSimulations`; `NothingToResample` for no trade; `NotFinite` for a P&L that
/// is NaN or infinite; `InvalidOpeningWealth` for a capital that is not a positive
/// number; those of `ratios::return_stats`.
pub fn bootstrap_trades(
    net_pnl: &[f64],
    capital: f64,
    spec: &MonteCarloSpec,
) -> Result<MonteCarloRuns, MetricsError> {
    check_spec(spec, net_pnl.len())?;
    check_finite(net_pnl, "trade net P&L")?;
    if !(capital.is_finite() && capital > 0.0) {
        return Err(MetricsError::InvalidOpeningWealth { value: capital });
    }
    simulate(spec, net_pnl.len(), |rng, path| {
        let mut equity = capital;
        for _ in 0..net_pnl.len() {
            let pnl = net_pnl
                .get(rng.below(net_pnl.len()))
                .copied()
                .unwrap_or(0.0);
            path.push(pnl / equity);
            equity += pnl;
            if equity <= 0.0 {
                break;
            }
        }
    })
}
