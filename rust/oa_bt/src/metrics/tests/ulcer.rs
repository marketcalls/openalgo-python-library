//! Close drawdowns, the ulcer and serenity indexes, and path statistics.

use super::{assert_all_close, assert_close, days};
use crate::metrics::{
    close_drawdowns, conditional_drawdown, path_stats, serenity_index, ulcer_index,
    ulcer_performance_index,
};

#[test]
fn close_drawdowns_start_from_the_opening() {
    // Wealth 1.1, 0.99, 0.99, 1.188 against peaks 1.1, 1.1, 1.1, 1.188.
    assert_all_close(
        &close_drawdowns(&[0.1, -0.1, 0.0, 0.2]),
        &[0.0, -0.1, -0.1, 0.0],
    );
    // A first session below the opening is already a drawdown.
    assert_all_close(
        &close_drawdowns(&[-0.05, 0.02]),
        &[-0.05, 0.95 * 1.02 - 1.0],
    );
}

#[test]
fn the_ulcer_index_is_the_root_mean_square_drawdown() {
    // sqrt((0 + 0.01 + 0.01 + 0) / 4).
    let drawdowns = [0.0, -0.1, -0.1, 0.0];
    assert_close(ulcer_index(&drawdowns), 0.005_f64.sqrt());
    assert_eq!(ulcer_index(&[0.0, 0.0]), 0.0);
    assert!(ulcer_index(&[]).is_nan());
}

#[test]
fn the_conditional_drawdown_is_the_mean_of_the_worst_five_percent() {
    // Sorted -0.1, -0.1, 0, 0: the 5 percent quantile is -0.1, met twice.
    assert_close(conditional_drawdown(&[0.0, -0.1, -0.1, 0.0]), -0.1);
    // Twenty-one drawdowns: the quantile at position 1 is -0.2; -0.3 and -0.2 are below.
    let mut many = vec![0.0; 19];
    many.extend([-0.3, -0.2]);
    assert_close(conditional_drawdown(&many), -0.25);
}

#[test]
fn the_ulcer_performance_index_is_excess_cagr_over_the_ulcer_index() {
    assert_close(ulcer_performance_index(0.12, 0.02, 0.05), 2.0);
    assert_eq!(ulcer_performance_index(0.1, 0.0, 0.0), f64::INFINITY);
    assert!(ulcer_performance_index(-0.1, 0.0, 0.0).is_nan());
    assert!(ulcer_performance_index(f64::NAN, 0.0, 0.05).is_nan());
}

#[test]
fn the_serenity_index_divides_by_ulcer_index_times_pitfall() {
    // Pitfall 0.1 / 0.2 = 0.5; 0.1 / (0.05 x 0.5) = 4.
    assert_close(serenity_index(0.12, 0.02, 0.05, -0.1, 0.2), 4.0);
    assert_eq!(serenity_index(0.12, 0.02, 0.0, 0.0, 0.2), f64::INFINITY);
    assert!(serenity_index(0.12, 0.02, 0.05, -0.1, f64::NAN).is_nan());
}

#[test]
fn path_statistics_measure_one_path() {
    let year = (365.25 * 86_400e9) as i64;
    let stats = path_stats(&[0.1, -0.1, 0.0, 0.2], 4.0, 0.0, year).unwrap();
    // 1.1 x 0.9 x 1 x 1.2 = 1.188 over one year.
    assert_close(stats.total_return, 0.188);
    assert_close(stats.cagr, 0.188);
    assert_close(stats.max_drawdown, -0.1);
}

#[test]
fn a_path_that_loses_everything_ends_there() {
    let stats = path_stats(&[0.1, -1.0, 0.5], 4.0, 0.0, days(400)).unwrap();
    assert_eq!(
        (stats.total_return, stats.cagr, stats.max_drawdown),
        (-1.0, -1.0, -1.0)
    );
    // Volatility of the two returns lived, 0.1 and -1, times sqrt(4).
    assert_close(stats.volatility, (1.1_f64 * 1.1 / 2.0).sqrt() * 2.0);
}

#[test]
fn an_empty_path_has_no_return_and_no_drawdown() {
    let stats = path_stats(&[], 4.0, 0.0, days(400)).unwrap();
    assert_eq!(
        (stats.total_return, stats.max_drawdown, stats.cagr),
        (0.0, 0.0, 0.0)
    );
    assert!(stats.sharpe.is_nan());
}
