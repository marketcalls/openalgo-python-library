//! Bootstrap Monte Carlo and its seeded generator.

use super::{assert_close, days};
use crate::metrics::{
    bootstrap_sessions, bootstrap_trades, percentiles, MetricsError, MonteCarloSpec, SplitMix64,
};

fn spec(simulations: usize, seed: u64) -> MonteCarloSpec {
    MonteCarloSpec {
        simulations,
        seed,
        periods_per_year: 249.0,
        risk_free_annual: 0.0,
        elapsed: days(400),
    }
}

/// Share of `values` equal to `target` (to the last rounding unit).
fn share(values: &[f64], target: f64) -> f64 {
    let hits = values
        .iter()
        .filter(|v| (*v - target).abs() < 1e-12)
        .count();
    hits as f64 / values.len() as f64
}

#[test]
fn splitmix64_reproduces_its_published_sequence() {
    // The first outputs of SplitMix64 seeded with 1234567, as published with the
    // reference implementation.
    let mut rng = SplitMix64::new(1_234_567);
    let expected = [
        6_457_827_717_110_365_317_u64,
        3_203_168_211_198_807_973,
        9_817_491_932_198_370_423,
        4_593_380_528_125_082_431,
        16_408_922_859_458_223_821,
    ];
    for value in expected {
        assert_eq!(rng.next_u64(), value);
    }
}

#[test]
fn integers_below_a_bound_are_uniform() {
    let mut rng = SplitMix64::new(7);
    let mut counts = [0usize; 3];
    for _ in 0..30_000 {
        counts[rng.below(3)] += 1;
    }
    // Each count is 10,000 give or take a few standard deviations of 82.
    assert!(
        counts.iter().all(|&c| (9_600..=10_400).contains(&c)),
        "{counts:?}"
    );
    assert_eq!(SplitMix64::new(1).below(1), 0);
}

#[test]
fn the_same_seed_gives_the_same_simulations() {
    let returns = [0.01, -0.02, 0.015, 0.005, -0.01];
    let first = bootstrap_sessions(&returns, &spec(200, 42)).unwrap();
    let again = bootstrap_sessions(&returns, &spec(200, 42)).unwrap();
    let other = bootstrap_sessions(&returns, &spec(200, 43)).unwrap();
    assert_eq!(first, again);
    assert_ne!(first.final_return, other.final_return);
}

#[test]
fn two_equally_likely_returns_give_a_binomial_distribution_of_outcomes() {
    // Two sessions drawn from +10 and -10 percent: 1.21 with probability 1/4, 0.99 with
    // 1/2 and 0.81 with 1/4.
    let runs = bootstrap_sessions(&[0.1, -0.1], &spec(4_000, 9)).unwrap();
    let finals = &runs.final_return;
    for (outcome, probability) in [(0.21, 0.25), (-0.01, 0.5), (-0.19, 0.25)] {
        let observed = share(finals, outcome);
        assert!(
            (observed - probability).abs() < 0.03,
            "{outcome}: {observed}"
        );
    }
    let p = percentiles(finals, &[0.05, 0.5, 0.95]).unwrap();
    assert_close(p[0], -0.19);
    assert_close(p[1], -0.01);
    assert_close(p[2], 0.21);
    // Drawdowns: none after +10 first then +10, -10 percent after one fall, -19 after two.
    let deepest = percentiles(&runs.max_drawdown, &[0.05, 0.5, 0.95]).unwrap();
    assert_close(deepest[0], -0.19);
    assert_close(deepest[1], -0.1);
    assert_close(deepest[2], 0.0);
}

#[test]
fn the_band_starts_at_zero_and_ends_at_the_percentiles_of_the_final_return() {
    let runs = bootstrap_sessions(&[0.1, -0.1], &spec(4_000, 9)).unwrap();
    assert_eq!(runs.band_steps, vec![0, 1, 2]);
    assert_eq!(runs.band[0], [0.0; 5]);
    let end = percentiles(&runs.final_return, &[0.05, 0.25, 0.5, 0.75, 0.95]).unwrap();
    for (actual, expected) in runs.band[2].iter().zip(end) {
        assert_close(*actual, expected);
    }
}

#[test]
fn a_long_run_is_banded_at_a_bounded_number_of_points() {
    let returns = vec![0.001; 1_000];
    let runs = bootstrap_sessions(&returns, &spec(3, 1)).unwrap();
    assert_eq!(runs.band_steps.len(), 201);
    assert_eq!((runs.band_steps[0], runs.band_steps[200]), (0, 1_000));
    assert_eq!(runs.band_steps[100], 500);
}

#[test]
fn trades_are_booked_on_the_capital_at_fixed_rupee_size() {
    // Two trades drawn from +100 and -100 on 1,000: 1,200, 1,000 or 800 at the end.
    let runs = bootstrap_trades(&[100.0, -100.0], 1_000.0, &spec(4_000, 5)).unwrap();
    for (outcome, probability) in [(0.2, 0.25), (0.0, 0.5), (-0.2, 0.25)] {
        let observed = share(&runs.final_return, outcome);
        assert!(
            (observed - probability).abs() < 0.03,
            "{outcome}: {observed}"
        );
    }
}

#[test]
fn a_trade_path_that_loses_the_capital_is_ruined() {
    // A run of two -600 trades: each path draws two, and -600 twice from 1,000 leaves
    // -200, so every path is ruined at its second trade.
    let runs = bootstrap_trades(&[-600.0, -600.0], 1_000.0, &spec(3, 5)).unwrap();
    assert!(runs.final_return.iter().all(|&r| r == -1.0));
    assert!(runs.max_drawdown.iter().all(|&d| d == -1.0));
    assert_eq!(runs.band[1], [-0.6; 5]);
}

#[test]
fn nothing_to_draw_or_no_simulation_is_refused() {
    assert_eq!(
        bootstrap_sessions(&[], &spec(10, 1)),
        Err(MetricsError::NothingToResample)
    );
    assert_eq!(
        bootstrap_sessions(&[0.01], &spec(0, 1)),
        Err(MetricsError::NoSimulations)
    );
    assert_eq!(
        bootstrap_trades(&[10.0], 0.0, &spec(10, 1)),
        Err(MetricsError::InvalidOpeningWealth { value: 0.0 })
    );
}
