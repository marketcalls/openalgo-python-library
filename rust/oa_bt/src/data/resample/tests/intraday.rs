//! Intraday to coarser intraday bars on the bar lattice.

use super::Columns;
use crate::data::resample::{resample_intraday, ClosedDayPolicy, ResampleError, Resampled};
use crate::market::calendar::testing::{calendar, d, ist, opens, MINUTE};
use crate::market::calendar::{hm, Calendar, Coverage, RuleClose};
use crate::model::Exchange;

fn run(cal: &Calendar, columns: &Columns, target_min: i64) -> Result<Resampled, ResampleError> {
    let bars = columns.bars();
    resample_intraday(
        cal,
        Exchange::Nse,
        &bars,
        MINUTE,
        target_min * MINUTE,
        ClosedDayPolicy::Error,
    )
}

#[test]
fn one_minute_bars_make_five_minute_bars() {
    let columns = Columns::ramp(opens(d(2025, 6, 2), (9, 15), (9, 25), MINUTE));
    let out = run(&calendar(), &columns, 5).unwrap().bars;
    let first_open = 1_748_835_900_000_000_000; // 2025-06-02 09:15 IST
    let second_open = 1_748_836_200_000_000_000; // 2025-06-02 09:20 IST
    let second_close = 1_748_836_500_000_000_000; // 2025-06-02 09:25 IST
    assert_eq!(out.ts_open, vec![first_open, second_open]);
    assert_eq!(out.ts_close, vec![second_open, second_close]);
    // Prices are the row numbers 0 to 9.
    assert_eq!((out.open, out.high), (vec![0.0, 5.0], vec![4.0, 9.0]));
    assert_eq!((out.low, out.close), (vec![0.0, 5.0], vec![4.0, 9.0]));
    assert_eq!((out.volume, out.oi), (vec![5.0, 5.0], vec![40.0, 90.0]));
}

#[test]
fn a_missing_bar_leaves_the_others_in_their_buckets() {
    let mut ts = opens(d(2025, 6, 2), (9, 15), (9, 25), MINUTE);
    ts.remove(2); // 09:17
    let out = run(&calendar(), &Columns::ramp(ts), 5).unwrap().bars;
    assert_eq!(out.volume, vec![4.0, 5.0]);
    assert_eq!(out.open, vec![0.0, 4.0]);
}

#[test]
fn equal_intervals_only_filter_and_close_the_bars() {
    let day = d(2025, 6, 2);
    let columns = Columns::ramp(vec![ist(day, 9, 14), ist(day, 9, 15)]);
    let result = run(&calendar(), &columns, 1).unwrap();
    assert_eq!(result.bars.ts_close, vec![1_748_835_960_000_000_000]); // 2025-06-02 09:16 IST
    assert_eq!(result.dropped.pre_open_bars, 1);
}

#[test]
fn a_special_session_bucket_starts_on_the_lattice_before_its_open() {
    let mut cal = calendar();
    let day = d(2026, 11, 8);
    cal.set_session(
        Exchange::Nse,
        day,
        hm(14, 20),
        RuleClose::At(hm(15, 30)),
        false,
        Coverage::Sourced,
    )
    .unwrap();
    let columns = Columns::ramp(opens(day, (14, 20), (15, 30), MINUTE));
    let out = run(&cal, &columns, 15).unwrap().bars;
    assert_eq!(out.ts_open.len(), 5);
    assert_eq!(out.ts_open[0], 1_794_127_500_000_000_000); // 2026-11-08 14:15 IST
    assert_eq!(out.ts_close[0], 1_794_128_400_000_000_000); // 2026-11-08 14:30 IST
    assert_eq!(out.volume[0], 10.0); // 14:20 to 14:29
}

#[test]
fn bars_outside_the_session_are_dropped_and_counted() {
    let day = d(2025, 6, 2);
    let mut ts = opens(day, (9, 10), (9, 20), MINUTE);
    ts.extend(opens(day, (15, 25), (15, 35), MINUTE));
    let result = run(&calendar(), &Columns::ramp(ts), 5).unwrap();
    assert_eq!(
        (result.dropped.pre_open_bars, result.dropped.post_close_bars),
        (5, 5)
    );
    // 09:15 to 09:19 and 15:25 to 15:29.
    assert_eq!(result.bars.volume, vec![5.0, 5.0]);
}

#[test]
fn bars_on_a_closed_day_are_an_error_or_dropped() {
    let columns = Columns::ramp(opens(d(2025, 6, 7), (9, 15), (9, 20), MINUTE));
    let cal = calendar();
    let error = run(&cal, &columns, 5).unwrap_err();
    assert!(matches!(
        error,
        ResampleError::ClosedDay {
            coverage: Coverage::Template,
            ..
        }
    ));
    let bars = columns.bars();
    let dropped = resample_intraday(
        &cal,
        Exchange::Nse,
        &bars,
        MINUTE,
        5 * MINUTE,
        ClosedDayPolicy::Drop,
    )
    .unwrap();
    assert_eq!(
        (dropped.bars.ts_open.len(), dropped.dropped.closed_day_bars),
        (0, 5)
    );
}

#[test]
fn unknown_values_are_skipped_column_by_column() {
    let mut columns = Columns::ramp(opens(d(2025, 6, 2), (9, 15), (9, 20), MINUTE));
    columns.open[0] = f64::NAN;
    columns.oi[4] = f64::NAN;
    columns.volume[1] = f64::NAN;
    let out = run(&calendar(), &columns, 5).unwrap().bars;
    assert_eq!((out.open[0], out.oi[0], out.volume[0]), (1.0, 30.0, 4.0));
}

#[test]
fn a_bar_off_the_lattice_is_bucketed_by_its_open() {
    let day = d(2025, 6, 2);
    let columns = Columns::ramp(vec![ist(day, 9, 17), ist(day, 9, 32)]);
    let bars = columns.bars();
    let out = resample_intraday(
        &calendar(),
        Exchange::Nse,
        &bars,
        5 * MINUTE,
        15 * MINUTE,
        ClosedDayPolicy::Error,
    )
    .unwrap()
    .bars;
    let first = 1_748_835_900_000_000_000; // 2025-06-02 09:15 IST
    let second = 1_748_836_800_000_000_000; // 2025-06-02 09:30 IST
    assert_eq!(out.ts_open, vec![first, second]);
}

#[test]
fn unusable_intervals_are_errors() {
    let columns = Columns::ramp(Vec::new());
    let bars = columns.bars();
    let cases = [
        (MINUTE, 0),
        (5 * MINUTE, MINUTE),
        (2 * MINUTE, 5 * MINUTE),
        (MINUTE, 1_440 * MINUTE),
        (0, MINUTE),
    ];
    for (base_nanos, target_nanos) in cases {
        let result = resample_intraday(
            &calendar(),
            Exchange::Nse,
            &bars,
            base_nanos,
            target_nanos,
            ClosedDayPolicy::Error,
        );
        assert_eq!(
            result,
            Err(ResampleError::InvalidInterval {
                base_nanos,
                target_nanos
            })
        );
    }
}
