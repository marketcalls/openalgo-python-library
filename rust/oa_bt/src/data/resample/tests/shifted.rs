//! Bars whose boundaries end earlier (`shifted`), worked from ramps whose prices are the
//! row numbers.

use super::Columns;
use crate::data::resample::{
    resample_daily_ending_earlier, resample_intraday, resample_intraday_ending_earlier,
    ClosedDayPolicy, ResampleError,
};
use crate::market::calendar::testing::{calendar, d, ist, opens, MINUTE};
use crate::model::Exchange;

const POLICY: ClosedDayPolicy = ClosedDayPolicy::Error;

#[test]
fn five_minute_bars_ending_two_minutes_earlier() {
    // 09:15 to 09:25 (rows 0 to 9): buckets from 09:13 (rows 0 to 2, three minutes, the
    // first starting before the open), 09:18 (rows 3 to 7) and 09:23 (rows 8 and 9).
    let day = d(2025, 6, 2);
    let columns = Columns::ramp(opens(day, (9, 15), (9, 25), MINUTE));
    let out = resample_intraday_ending_earlier(
        &calendar(),
        Exchange::Nse,
        &columns.bars(),
        (MINUTE, 5 * MINUTE),
        2 * MINUTE,
        POLICY,
    )
    .unwrap()
    .bars;
    let at = |h, m| ist(day, h, m);
    assert_eq!(out.ts_open, vec![at(9, 13), at(9, 18), at(9, 23)]);
    assert_eq!(out.ts_close, vec![at(9, 18), at(9, 23), at(9, 28)]);
    assert_eq!(out.open, vec![0.0, 3.0, 8.0]);
    assert_eq!(out.close, vec![2.0, 7.0, 9.0]);
    assert_eq!(out.volume, vec![3.0, 5.0, 2.0]);
}

#[test]
fn a_shift_of_zero_is_the_ordinary_lattice() {
    let columns = Columns::ramp(opens(d(2025, 6, 2), (9, 15), (10, 15), MINUTE));
    let bars = columns.bars();
    let ordinary = resample_intraday(
        &calendar(),
        Exchange::Nse,
        &bars,
        MINUTE,
        15 * MINUTE,
        POLICY,
    );
    let shifted = resample_intraday_ending_earlier(
        &calendar(),
        Exchange::Nse,
        &bars,
        (MINUTE, 15 * MINUTE),
        0,
        POLICY,
    );
    assert_eq!(shifted, ordinary);
}

#[test]
fn the_last_bucket_of_a_session_is_cut_at_its_close() {
    // 15:25 to 15:30 with 5-minute bars ending one minute earlier: 15:24 to 15:29 holds
    // rows 0 to 3 and the bucket from 15:29 holds row 4, cut at the close, 15:30.
    let day = d(2025, 6, 2);
    let columns = Columns::ramp(opens(day, (15, 25), (15, 30), MINUTE));
    let out = resample_intraday_ending_earlier(
        &calendar(),
        Exchange::Nse,
        &columns.bars(),
        (MINUTE, 5 * MINUTE),
        MINUTE,
        POLICY,
    )
    .unwrap()
    .bars;
    assert_eq!(out.ts_open, vec![ist(day, 15, 24), ist(day, 15, 29)]);
    assert_eq!(out.ts_close, vec![ist(day, 15, 29), ist(day, 15, 30)]);
}

#[test]
fn a_shift_must_be_whole_base_bars_shorter_than_the_target() {
    let columns = Columns::ramp(opens(d(2025, 6, 2), (9, 15), (9, 25), MINUTE));
    let bars = columns.bars();
    for shift in [5 * MINUTE, MINUTE / 2, -MINUTE] {
        let result = resample_intraday_ending_earlier(
            &calendar(),
            Exchange::Nse,
            &bars,
            (MINUTE, 5 * MINUTE),
            shift,
            POLICY,
        );
        assert!(
            matches!(result, Err(ResampleError::InvalidInterval { .. })),
            "{shift}"
        );
    }
}

#[test]
fn days_ending_fifteen_minutes_earlier_take_the_last_quarter_hour_into_the_next() {
    // 2025-06-02 (Monday) 15:10 to 15:30 (rows 0 to 19) and 2025-06-03 09:15 to 09:20
    // (rows 20 to 24): the Monday's bar ends at 15:15 (rows 0 to 4, from the Friday's
    // 15:15); the Tuesday's runs from Monday 15:15 (rows 5 to 24).
    let monday = d(2025, 6, 2);
    let mut ts = opens(monday, (15, 10), (15, 30), MINUTE);
    ts.extend(opens(d(2025, 6, 3), (9, 15), (9, 20), MINUTE));
    let columns = Columns::ramp(ts);
    let out = resample_daily_ending_earlier(
        &calendar(),
        Exchange::Nse,
        &columns.bars(),
        MINUTE,
        15 * MINUTE,
        POLICY,
    )
    .unwrap()
    .bars;
    let friday = d(2025, 5, 30);
    assert_eq!(out.ts_open, vec![ist(friday, 15, 15), ist(monday, 15, 15)]);
    assert_eq!(
        out.ts_close,
        vec![ist(monday, 15, 15), ist(d(2025, 6, 3), 15, 15)]
    );
    assert_eq!((out.open, out.close), (vec![0.0, 5.0], vec![4.0, 24.0]));
    assert_eq!(out.volume, vec![5.0, 20.0]);
}

#[test]
fn a_day_cannot_end_earlier_by_nothing_or_a_day() {
    let columns = Columns::ramp(opens(d(2025, 6, 2), (9, 15), (9, 25), MINUTE));
    let bars = columns.bars();
    for shift in [0, 24 * 60 * MINUTE] {
        let result =
            resample_daily_ending_earlier(&calendar(), Exchange::Nse, &bars, MINUTE, shift, POLICY);
        assert!(
            matches!(result, Err(ResampleError::InvalidInterval { .. })),
            "{shift}"
        );
    }
}
