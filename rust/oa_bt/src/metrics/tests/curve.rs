//! The wealth index, cash flows, ruin and session returns.

use super::{assert_all_close, assert_close, ist, june_close};
use crate::metrics::{session_returns, wealth, CashFlows, EquityCurve, MetricsError};
use crate::model::days_from_civil;

/// Monday 2 June 2025 09:15 IST (the start), then the closes of 2, 3 and 4 June.
fn four_points() -> [i64; 4] {
    [
        ist(2025, 6, 2, 9, 15),
        june_close(2),
        june_close(3),
        june_close(4),
    ]
}

#[test]
fn the_index_is_equity_over_the_start_without_cash_flows() {
    let ts = four_points();
    let curve = EquityCurve {
        ts: &ts,
        equity: &[100.0, 110.0, 99.0, 121.0],
    };
    let w = wealth(&curve, &CashFlows::NONE).unwrap();
    assert_all_close(&w.index, &[1.0, 1.1, 0.99, 1.21]);
    assert_all_close(&w.pnl, &[0.0, 10.0, -1.0, 21.0]);
    assert_eq!(w.ruin, None);
}

#[test]
fn one_return_per_ist_date_from_the_last_point_of_the_date() {
    let ts = [
        ist(2025, 6, 2, 9, 15),
        ist(2025, 6, 2, 11, 0),
        june_close(2),
        june_close(3),
    ];
    let curve = EquityCurve {
        ts: &ts,
        equity: &[100.0, 104.0, 110.0, 99.0],
    };
    let w = wealth(&curve, &CashFlows::NONE).unwrap();
    let sessions = session_returns(&ts, &w);
    let (monday, tuesday) = (days_from_civil(2025, 6, 2), days_from_civil(2025, 6, 3));
    assert_eq!(sessions.days, vec![monday, tuesday]);
    // 110 / 100 - 1 and 99 / 110 - 1.
    assert_all_close(&sessions.returns, &[0.1, -0.1]);
}

#[test]
fn the_session_date_is_the_ist_date_not_the_utc_date() {
    // 04:00 IST on Tuesday is 22:30 UTC on Monday.
    let ts = [ist(2025, 6, 2, 9, 0), june_close(2), ist(2025, 6, 3, 4, 0)];
    let curve = EquityCurve {
        ts: &ts,
        equity: &[100.0, 101.0, 102.0],
    };
    let sessions = session_returns(&ts, &wealth(&curve, &CashFlows::NONE).unwrap());
    let tuesday = days_from_civil(2025, 6, 3);
    assert_eq!(sessions.days.last(), Some(&tuesday));
    assert_eq!(sessions.days.len(), 2);
}

#[test]
fn a_deposit_is_not_a_return() {
    let ts = four_points();
    let curve = EquityCurve {
        ts: &ts,
        equity: &[100.0, 110.0, 160.0, 168.0],
    };
    let flows = CashFlows {
        ts: &[june_close(3)],
        amount: &[50.0],
    };
    let w = wealth(&curve, &flows).unwrap();
    // Returns: 0.10, (160 - 50) / 110 - 1 = 0, 168 / 160 - 1 = 0.05.
    assert_all_close(&w.index, &[1.0, 1.1, 1.1, 1.1 * 1.05]);
    assert_all_close(&w.pnl, &[0.0, 10.0, 10.0, 18.0]);
    assert_all_close(&w.flow, &[0.0, 0.0, 50.0, 0.0]);
}

#[test]
fn a_flow_between_two_points_is_booked_at_the_next_point() {
    let ts = four_points();
    let curve = EquityCurve {
        ts: &ts,
        equity: &[100.0, 110.0, 80.0, 88.0],
    };
    let flows = CashFlows {
        ts: &[ist(2025, 6, 3, 10, 0)],
        amount: &[-33.0],
    };
    let w = wealth(&curve, &flows).unwrap();
    assert_all_close(&w.flow, &[0.0, 0.0, -33.0, 0.0]);
    // (80 + 33) / 110 - 1 is the Tuesday return.
    assert_close(w.index[2] / w.index[1] - 1.0, 113.0 / 110.0 - 1.0);
}

#[test]
fn an_equity_back_at_an_earlier_value_gives_exactly_the_earlier_index() {
    let ts = four_points();
    let curve = EquityCurve {
        ts: &ts,
        equity: &[100.0, 97.3, 101.7, 97.3],
    };
    let w = wealth(&curve, &CashFlows::NONE).unwrap();
    assert_eq!(w.index[3].to_bits(), w.index[1].to_bits());
}

#[test]
fn equity_at_zero_ruins_the_account_and_ends_the_returns() {
    let ts = four_points();
    let curve = EquityCurve {
        ts: &ts,
        equity: &[100.0, 50.0, 0.0, 20.0],
    };
    let w = wealth(&curve, &CashFlows::NONE).unwrap();
    assert_eq!(w.ruin, Some(2));
    assert_all_close(&w.index, &[1.0, 0.5, 0.0, 0.0]);
    let sessions = session_returns(&ts, &w);
    assert_all_close(&sessions.returns, &[-0.5, -1.0]);
}

#[test]
fn negative_equity_is_ruin_with_a_return_of_minus_one() {
    let ts = four_points();
    let curve = EquityCurve {
        ts: &ts,
        equity: &[100.0, -20.0, -25.0, 10.0],
    };
    let w = wealth(&curve, &CashFlows::NONE).unwrap();
    assert_eq!(w.ruin, Some(1));
    assert_all_close(&session_returns(&ts, &w).returns, &[-1.0]);
    // Cumulative P&L keeps the true rupee figures after ruin.
    assert_all_close(&w.pnl, &[0.0, -120.0, -125.0, -90.0]);
}

#[test]
fn a_flat_curve_has_zero_returns() {
    let ts = four_points();
    let curve = EquityCurve {
        ts: &ts,
        equity: &[100.0; 4],
    };
    let w = wealth(&curve, &CashFlows::NONE).unwrap();
    assert_all_close(&session_returns(&ts, &w).returns, &[0.0, 0.0, 0.0]);
}

#[test]
fn malformed_series_are_refused_with_the_row() {
    let ts = four_points();
    let cases: [(&[i64], &[f64], MetricsError); 5] = [
        (
            &ts[..1],
            &[100.0],
            MetricsError::TooFewPoints {
                needed: 2,
                found: 1,
            },
        ),
        (
            &ts,
            &[100.0, 1.0],
            MetricsError::LengthMismatch {
                what: "equity",
                expected: 4,
                found: 2,
            },
        ),
        (
            &[ts[0], ts[2], ts[1]],
            &[100.0, 1.0, 2.0],
            MetricsError::NotIncreasing {
                what: "equity timestamps",
                index: 2,
            },
        ),
        (
            &ts[..2],
            &[100.0, f64::NAN],
            MetricsError::NotFinite {
                what: "equity",
                index: 1,
            },
        ),
        (
            &ts[..2],
            &[0.0, 10.0],
            MetricsError::NonPositiveStart { equity: 0.0 },
        ),
    ];
    for (ts, equity, expected) in cases {
        let curve = EquityCurve { ts, equity };
        assert_eq!(wealth(&curve, &CashFlows::NONE), Err(expected));
    }
}

#[test]
fn a_flow_at_the_start_or_after_the_end_is_refused() {
    let ts = four_points();
    let curve = EquityCurve {
        ts: &ts,
        equity: &[100.0, 110.0, 99.0, 121.0],
    };
    for at in [ts[0], ts[3] + 1] {
        let flows = CashFlows {
            ts: &[at],
            amount: &[10.0],
        };
        assert_eq!(
            wealth(&curve, &flows),
            Err(MetricsError::FlowOutsideCurve { index: 0 })
        );
    }
}

#[test]
fn a_withdrawal_that_empties_the_account_is_refused() {
    let ts = four_points();
    let curve = EquityCurve {
        ts: &ts,
        equity: &[100.0, 110.0, 0.0, 0.0],
    };
    let flows = CashFlows {
        ts: &[ts[2]],
        amount: &[-110.0],
    };
    assert_eq!(
        wealth(&curve, &flows),
        Err(MetricsError::NoCapitalAfterFlow { index: 2 })
    );
}
