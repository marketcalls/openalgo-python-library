//! Drawdown series, the deepest and the longest episode, and the rupee drawdown.

use super::{assert_all_close, assert_close, days};
use crate::metrics::{drawdowns, DrawdownEpisode};

fn daily(count: i64) -> Vec<i64> {
    (0..count).map(days).collect()
}

#[test]
fn the_series_is_the_fall_from_the_running_peak() {
    let ts = daily(4);
    let result = drawdowns(&ts, &[1.0, 1.1, 0.99, 1.21], &[0.0; 4]);
    // 0.99 / 1.1 - 1 = -0.1.
    assert_all_close(&result.series, &[0.0, 0.0, -0.1, 0.0]);
    let max = result.max.unwrap();
    assert_eq!((max.peak, max.trough, max.recovery), (1, 2, Some(3)));
    assert_close(max.depth, -0.1);
    assert_eq!(max.duration, days(2));
}

#[test]
fn the_deepest_and_the_longest_episode_are_told_apart() {
    let ts = daily(7);
    let index = [1.0, 0.8, 1.0, 0.95, 0.96, 0.97, 0.99];
    let result = drawdowns(&ts, &index, &[0.0; 7]);
    let deepest = DrawdownEpisode {
        peak: 0,
        trough: 1,
        recovery: Some(2),
        depth: 0.8 - 1.0,
        duration: days(2),
    };
    assert_eq!(result.max, Some(deepest));
    // Still under water at the last point: it runs from day 2 to day 6.
    let longest = result.longest.unwrap();
    assert_eq!(
        (longest.peak, longest.trough, longest.recovery),
        (2, 3, None)
    );
    assert_eq!(longest.duration, days(4));
    assert_close(longest.depth, 0.95 - 1.0);
}

#[test]
fn a_curve_that_never_falls_has_no_episode() {
    let ts = daily(3);
    let result = drawdowns(&ts, &[1.0, 1.0, 1.2], &[0.0, 0.0, 20.0]);
    assert_eq!((result.max, result.longest), (None, None));
    assert_all_close(&result.series, &[0.0; 3]);
    assert_eq!(result.max_amount, 0.0);
}

#[test]
fn equal_depths_keep_the_first_episode() {
    let ts = daily(5);
    let result = drawdowns(&ts, &[1.0, 0.9, 1.0, 0.9, 1.0], &[0.0; 5]);
    assert_eq!(result.max.map(|e| e.peak), Some(0));
    assert_eq!(result.longest.map(|e| e.peak), Some(0));
}

#[test]
fn ruin_is_a_drawdown_of_minus_one() {
    let ts = daily(4);
    let result = drawdowns(&ts, &[1.0, 0.5, 0.0, 0.0], &[0.0; 4]);
    assert_close(result.max.unwrap().depth, -1.0);
    assert_eq!(result.max.unwrap().recovery, None);
}

#[test]
fn the_rupee_drawdown_is_the_largest_fall_of_cumulative_pnl() {
    let ts = daily(5);
    let pnl = [0.0, 10.0, -5.0, 30.0, 20.0];
    // 10 down to -5 is 15; 30 down to 20 is 10.
    assert_eq!(drawdowns(&ts, &[1.0; 5], &pnl).max_amount, 15.0);
}

#[test]
fn an_empty_series_has_no_drawdown() {
    let result = drawdowns(&[], &[], &[]);
    assert!(result.series.is_empty());
    assert_eq!((result.max, result.max_amount), (None, 0.0));
}
