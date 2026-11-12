//! Input checks shared by every resampler.

use super::Columns;
use crate::data::resample::ResampleError;
use crate::market::calendar::testing::{d, ist};

#[test]
fn a_column_of_another_length_is_an_error() {
    let day = d(2025, 6, 2);
    let mut columns = Columns::ramp(vec![ist(day, 9, 15), ist(day, 9, 16)]);
    columns.volume.pop();
    let error = columns.bars().validate().unwrap_err();
    assert_eq!(
        error,
        ResampleError::LengthMismatch {
            column: "volume",
            len: 1,
            expected: 2
        }
    );
}

#[test]
fn unusable_timestamps_are_errors_by_row() {
    let day = d(2025, 6, 2);
    let (a, b) = (ist(day, 9, 15), ist(day, 9, 16));
    let cases = [
        (
            vec![a, i64::MIN],
            ResampleError::UnsupportedTimestamp {
                index: 1,
                ts: i64::MIN,
            },
        ),
        (vec![b, a], ResampleError::NotSorted { index: 1 }),
        (vec![a, a], ResampleError::Duplicate { index: 1, ts: a }),
    ];
    for (ts_open, expected) in cases {
        assert_eq!(Columns::ramp(ts_open).bars().validate(), Err(expected));
    }
}

#[test]
fn an_empty_series_is_valid() {
    assert_eq!(Columns::ramp(Vec::new()).bars().validate(), Ok(()));
}
