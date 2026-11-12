//! Counting statistics of session returns and losing streaks.

use super::assert_close;
use crate::metrics::{geometric_mean, loss_streaks, session_stats, MetricsError};

#[test]
fn session_figures_follow_their_definitions() {
    let s = session_stats(&[0.1, -0.05, 0.0, -0.02, 0.03]).unwrap();
    // 1.1 x 0.95 x 1 x 0.98 x 1.03 = 1.054823, compounded over five sessions.
    assert_close(s.expected_return, 1.054_823_f64.powf(0.2) - 1.0);
    assert_close(s.average_return, 0.06 / 5.0);
    assert_eq!((s.best, s.worst), (0.1, -0.05));
    // The flat session neither won nor lost.
    assert_eq!((s.sessions, s.winning, s.losing), (5, 2, 2));
    assert_close(s.win_rate, 0.5);
    assert_close(s.average_winning, 0.065);
    assert_close(s.average_losing, -0.035);
    assert_eq!((s.max_consecutive_wins, s.max_consecutive_losses), (1, 1));
}

#[test]
fn losing_streaks_are_listed_in_date_order() {
    let returns = [-0.01, -0.02, 0.01, -0.01, -0.01, -0.01, 0.0, -0.03];
    assert_eq!(loss_streaks(&returns), vec![2, 3, 1]);
    let s = session_stats(&returns).unwrap();
    assert_eq!(s.max_consecutive_losses, 3);
    assert_eq!(s.loss_streaks, vec![2, 3, 1]);
}

#[test]
fn a_session_that_loses_everything_makes_the_expected_return_minus_one() {
    assert_eq!(geometric_mean(&[0.1, -1.0, 0.5]), -1.0);
}

#[test]
fn all_winning_and_all_losing_sessions() {
    let wins = session_stats(&[0.01, 0.02]).unwrap();
    assert_eq!(wins.win_rate, 1.0);
    assert!(wins.average_losing.is_nan() && wins.loss_streaks.is_empty());
    let losses = session_stats(&[-0.01, -0.02]).unwrap();
    assert_eq!(losses.win_rate, 0.0);
    assert!(losses.average_winning.is_nan());
    assert_eq!(losses.loss_streaks, vec![2]);
}

#[test]
fn flat_and_empty_series_have_no_win_rate() {
    let flat = session_stats(&[0.0, 0.0, 0.0]).unwrap();
    assert!(flat.win_rate.is_nan());
    assert_eq!((flat.expected_return, flat.average_return), (0.0, 0.0));
    let empty = session_stats(&[]).unwrap();
    assert!(empty.expected_return.is_nan() && empty.best.is_nan() && empty.win_rate.is_nan());
    assert_eq!(empty.sessions, 0);
}

#[test]
fn a_single_session_is_its_own_average_and_extremes() {
    let s = session_stats(&[0.02]).unwrap();
    assert_close(s.expected_return, 0.02);
    assert_eq!((s.average_return, s.best, s.worst), (0.02, 0.02, 0.02));
}

#[test]
fn an_undefined_return_is_refused() {
    assert!(matches!(
        session_stats(&[0.01, f64::NAN]),
        Err(MetricsError::NotFinite { index: 1, .. })
    ));
}
