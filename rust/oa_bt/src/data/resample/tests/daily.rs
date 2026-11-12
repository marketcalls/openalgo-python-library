//! Intraday bars to one bar per session date.

use super::Columns;
use crate::data::resample::{resample_daily, ClosedDayPolicy, ResampleError};
use crate::market::calendar::testing::{calendar, d, opens, MINUTE};
use crate::model::{Exchange, NANOS_PER_DAY};

#[test]
fn a_daily_bar_spans_its_session_whatever_bars_exist() {
    let columns = Columns::ramp(opens(d(2025, 6, 2), (10, 0), (11, 0), MINUTE));
    let bars = columns.bars();
    let out = resample_daily(
        &calendar(),
        Exchange::Nse,
        &bars,
        MINUTE,
        ClosedDayPolicy::Error,
    )
    .unwrap()
    .bars;
    assert_eq!(out.ts_open, vec![1_748_835_900_000_000_000]); // 2025-06-02 09:15 IST
    assert_eq!(out.ts_close, vec![1_748_858_400_000_000_000]); // 2025-06-02 15:30 IST
    assert_eq!(
        (out.open[0], out.close[0], out.volume[0]),
        (0.0, 59.0, 60.0)
    );
}

#[test]
fn each_session_date_gets_one_bar() {
    let mut ts = opens(d(2025, 6, 2), (9, 15), (9, 20), MINUTE);
    ts.extend(opens(d(2025, 6, 3), (9, 15), (9, 20), MINUTE));
    let columns = Columns::ramp(ts);
    let bars = columns.bars();
    let out = resample_daily(
        &calendar(),
        Exchange::Nse,
        &bars,
        MINUTE,
        ClosedDayPolicy::Error,
    )
    .unwrap()
    .bars;
    assert_eq!(out.ts_open.len(), 2);
    assert_eq!(out.ts_open[1], 1_748_922_300_000_000_000); // 2025-06-03 09:15 IST
}

#[test]
fn a_day_with_only_post_close_bars_has_no_daily_bar() {
    let columns = Columns::ramp(opens(d(2025, 6, 2), (15, 30), (15, 40), MINUTE));
    let bars = columns.bars();
    let result = resample_daily(
        &calendar(),
        Exchange::Nse,
        &bars,
        MINUTE,
        ClosedDayPolicy::Error,
    )
    .unwrap();
    assert!(result.bars.ts_open.is_empty());
    assert_eq!(result.dropped.post_close_bars, 10);
}

#[test]
fn bars_on_a_closed_day_are_an_error_by_default() {
    let columns = Columns::ramp(opens(d(2025, 6, 8), (9, 15), (9, 20), MINUTE));
    let bars = columns.bars();
    let result = resample_daily(
        &calendar(),
        Exchange::Nse,
        &bars,
        MINUTE,
        ClosedDayPolicy::Error,
    );
    assert!(
        matches!(result, Err(ResampleError::ClosedDay { .. })),
        "{result:?}"
    );
}

#[test]
fn the_base_must_be_intraday() {
    let columns = Columns::ramp(Vec::new());
    let bars = columns.bars();
    for base_nanos in [0, NANOS_PER_DAY] {
        let result = resample_daily(
            &calendar(),
            Exchange::Nse,
            &bars,
            base_nanos,
            ClosedDayPolicy::Error,
        );
        let expected = ResampleError::InvalidInterval {
            base_nanos,
            target_nanos: NANOS_PER_DAY,
        };
        assert_eq!(result, Err(expected));
    }
}
