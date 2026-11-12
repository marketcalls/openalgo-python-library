//! The opening wealth: the capital before the first point, which makes every point a
//! mark of the run (its P&L, its session and a fall below the capital all count).

use super::{assert_all_close, assert_close, days, ist, june_close};
use crate::metrics::{
    combine_equity, drawdowns, session_returns, wealth, CashFlows, EquityCurve, MetricsError,
};
use crate::model::days_from_civil;

/// The closes of Monday 2 to Wednesday 4 June 2025.
fn three_closes() -> [i64; 3] {
    [june_close(2), june_close(3), june_close(4)]
}

#[test]
fn the_index_and_pnl_are_measured_from_the_opening_wealth() {
    let ts = three_closes();
    let curve = EquityCurve {
        ts: &ts,
        equity: &[102.0, 99.96, 104.958],
    };
    let w = wealth(&curve, &CashFlows::NONE, Some(100.0)).unwrap();
    assert_all_close(&w.index, &[1.02, 0.9996, 1.04958]);
    assert_all_close(&w.pnl, &[2.0, -0.04, 4.958]);
    assert_eq!((w.ruin, w.opening_wealth), (None, Some(100.0)));
}

#[test]
fn the_first_session_of_a_daily_series_returns_from_the_opening_wealth() {
    let ts = three_closes();
    let curve = EquityCurve {
        ts: &ts,
        equity: &[102.0, 99.96, 104.958],
    };
    let sessions = session_returns(&ts, &wealth(&curve, &CashFlows::NONE, Some(100.0)).unwrap());
    let june = |day| days_from_civil(2025, 6, day);
    assert_eq!(sessions.days, vec![june(2), june(3), june(4)]);
    // 102 / 100 - 1, 99.96 / 102 - 1 and 104.958 / 99.96 - 1.
    assert_all_close(&sessions.returns, &[0.02, -0.02, 0.05]);
}

#[test]
fn an_intraday_first_point_is_in_the_first_session() {
    let ts = [ist(2025, 6, 2, 9, 16), june_close(2)];
    let curve = EquityCurve {
        ts: &ts,
        equity: &[100.5, 101.0],
    };
    let sessions = session_returns(&ts, &wealth(&curve, &CashFlows::NONE, Some(100.0)).unwrap());
    assert_eq!(sessions.days, vec![days_from_civil(2025, 6, 2)]);
    // 101 / 100 - 1: the 0.5 of the first point counts.
    assert_all_close(&sessions.returns, &[0.01]);
}

#[test]
fn one_point_after_the_opening_wealth_is_enough() {
    let ts = [june_close(2)];
    let curve = EquityCurve {
        ts: &ts,
        equity: &[101.0],
    };
    let w = wealth(&curve, &CashFlows::NONE, Some(100.0)).unwrap();
    assert_all_close(&session_returns(&ts, &w).returns, &[0.01]);
    let empty = EquityCurve {
        ts: &[],
        equity: &[],
    };
    let refused = MetricsError::TooFewPoints {
        needed: 1,
        found: 0,
    };
    assert_eq!(wealth(&empty, &CashFlows::NONE, Some(100.0)), Err(refused));
}

#[test]
fn equity_at_zero_at_the_first_point_is_ruin() {
    let ts = [june_close(2), june_close(3)];
    let curve = EquityCurve {
        ts: &ts,
        equity: &[0.0, 10.0],
    };
    let w = wealth(&curve, &CashFlows::NONE, Some(100.0)).unwrap();
    assert_eq!(w.ruin, Some(0));
    assert_all_close(&w.index, &[0.0, 0.0]);
    assert_all_close(&session_returns(&ts, &w).returns, &[-1.0]);
}

#[test]
fn a_deposit_after_the_opening_is_not_a_return() {
    let ts = [june_close(2), june_close(3)];
    let curve = EquityCurve {
        ts: &ts,
        equity: &[110.0, 160.0],
    };
    let flows = CashFlows {
        ts: &[june_close(3)],
        amount: &[50.0],
    };
    let w = wealth(&curve, &flows, Some(100.0)).unwrap();
    // 110 / 100 - 1 = 0.10, then (160 - 50) / 110 - 1 = 0.
    assert_all_close(&w.index, &[1.1, 1.1]);
    assert_all_close(&w.pnl, &[10.0, 10.0]);
}

#[test]
fn an_opening_wealth_that_is_not_a_positive_number_is_refused() {
    let ts = three_closes();
    let curve = EquityCurve {
        ts: &ts,
        equity: &[102.0, 99.96, 104.958],
    };
    for value in [0.0, -100.0, f64::NAN, f64::INFINITY] {
        let result = wealth(&curve, &CashFlows::NONE, Some(value));
        assert!(
            matches!(result, Err(MetricsError::InvalidOpeningWealth { .. })),
            "{value}: {result:?}"
        );
    }
}

#[test]
fn a_first_point_below_the_opening_is_a_drawdown_from_the_opening() {
    let ts = [days(0), days(1), days(2)];
    let result = drawdowns(&ts, &[0.95, 0.98, 1.05], &[-5.0, -2.0, 5.0]);
    assert_all_close(&result.series, &[-0.05, -0.02, 0.0]);
    let max = result.max.unwrap();
    assert_eq!((max.peak, max.trough, max.recovery), (None, 0, Some(2)));
    assert_close(max.depth, -0.05);
    // The opening has no time of its own: the episode is timed from the first point.
    assert_eq!(max.duration, days(2));
    assert_eq!(result.max_amount, 5.0);
}

#[test]
fn with_capitals_a_strategy_holds_its_capital_before_its_first_point() {
    let a = EquityCurve {
        ts: &[2, 3],
        equity: &[104.0, 110.0],
    };
    let b = EquityCurve {
        ts: &[1, 3, 4],
        equity: &[50.0, 40.0, 45.0],
    };
    let (ts, total) = combine_equity(&[a, b], Some(&[100.0, 50.0])).unwrap();
    assert_eq!(ts, vec![1, 2, 3, 4]);
    // A: its capital 100 until its first mark, then 104, 110, 110. B: 50, 50, 40, 45.
    assert_eq!(total, vec![150.0, 154.0, 150.0, 155.0]);
}

#[test]
fn capitals_need_one_positive_value_per_strategy() {
    let a = EquityCurve {
        ts: &[1],
        equity: &[100.0],
    };
    let missing = MetricsError::LengthMismatch {
        what: "opening wealth",
        expected: 1,
        found: 0,
    };
    assert_eq!(combine_equity(&[a], Some(&[])), Err(missing));
    let zero = MetricsError::InvalidOpeningWealth { value: 0.0 };
    assert_eq!(combine_equity(&[a], Some(&[0.0])), Err(zero));
}
