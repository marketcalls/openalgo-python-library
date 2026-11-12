//! Bar columns in and out of the resamplers, and the aggregate of one bucket.

use crate::market::calendar::is_supported_instant;
use crate::model::TsNanos;

use super::error::ResampleError;

/// Input bars as borrowed columns. `ts_open` holds bar opens in UTC nanoseconds,
/// sorted ascending with no duplicates. NaN in a value column means "unknown".
#[derive(Clone, Copy, Debug)]
pub struct BarsIn<'a> {
    pub ts_open: &'a [TsNanos],
    pub open: &'a [f64],
    pub high: &'a [f64],
    pub low: &'a [f64],
    pub close: &'a [f64],
    pub volume: &'a [f64],
    pub oi: &'a [f64],
}

impl BarsIn<'_> {
    /// Check that the columns have one length and that `ts_open` holds supported
    /// instants in strictly increasing order.
    ///
    /// Errors: `LengthMismatch`; `UnsupportedTimestamp` for a missing value (the int64
    /// minimum) or an instant outside the supported dates; `NotSorted`; `Duplicate`.
    pub fn validate(&self) -> Result<(), ResampleError> {
        let expected = self.ts_open.len();
        let columns = [
            ("open", self.open.len()),
            ("high", self.high.len()),
            ("low", self.low.len()),
            ("close", self.close.len()),
            ("volume", self.volume.len()),
            ("oi", self.oi.len()),
        ];
        for (column, len) in columns {
            if len != expected {
                return Err(ResampleError::LengthMismatch {
                    column,
                    len,
                    expected,
                });
            }
        }
        if let Some(index) = self
            .ts_open
            .iter()
            .position(|&ts| !is_supported_instant(ts))
        {
            return Err(ResampleError::UnsupportedTimestamp {
                index,
                ts: self.ts_open[index],
            });
        }
        for (index, pair) in self.ts_open.windows(2).enumerate() {
            if pair[1] < pair[0] {
                return Err(ResampleError::NotSorted { index: index + 1 });
            }
            if pair[1] == pair[0] {
                return Err(ResampleError::Duplicate {
                    index: index + 1,
                    ts: pair[1],
                });
            }
        }
        Ok(())
    }
}

/// Resampled bars as owned columns, all of one length. `ts_open` and `ts_close` are
/// UTC nanoseconds; a bar is complete, and visible to a strategy, at `ts_close`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BarsOut {
    pub ts_open: Vec<TsNanos>,
    pub ts_close: Vec<TsNanos>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
    pub oi: Vec<f64>,
}

impl BarsOut {
    pub(super) fn push(&mut self, ts_open: TsNanos, ts_close: TsNanos, bucket: &Accumulator) {
        self.ts_open.push(ts_open);
        self.ts_close.push(ts_close);
        self.open.push(bucket.open);
        self.high.push(bucket.high);
        self.low.push(bucket.low);
        self.close.push(bucket.close);
        self.volume.push(bucket.volume);
        self.oi.push(bucket.oi);
    }
}

/// Running aggregate of one bucket: first open, highest high, lowest low, last close,
/// summed volume and last known open interest. Every field starts unknown (NaN).
#[derive(Clone, Copy, Debug)]
pub(super) struct Accumulator {
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    oi: f64,
}

impl Accumulator {
    pub(super) fn new() -> Accumulator {
        Accumulator {
            open: f64::NAN,
            high: f64::NAN,
            low: f64::NAN,
            close: f64::NAN,
            volume: f64::NAN,
            oi: f64::NAN,
        }
    }

    /// Add row `row` of `bars`, skipping unknown (NaN) values column by column. The
    /// caller has validated `bars`, so `row` is inside every column.
    pub(super) fn push(&mut self, bars: &BarsIn<'_>, row: usize) {
        let (open, high, low) = (bars.open[row], bars.high[row], bars.low[row]);
        let (close, volume, oi) = (bars.close[row], bars.volume[row], bars.oi[row]);
        if self.open.is_nan() {
            self.open = open;
        }
        if !high.is_nan() && (self.high.is_nan() || high > self.high) {
            self.high = high;
        }
        if !low.is_nan() && (self.low.is_nan() || low < self.low) {
            self.low = low;
        }
        if !close.is_nan() {
            self.close = close;
        }
        if !volume.is_nan() {
            self.volume = if self.volume.is_nan() {
                volume
            } else {
                self.volume + volume
            };
        }
        if !oi.is_nan() {
            self.oi = oi;
        }
    }

    /// Whether the bucket holds at least one known price.
    pub(super) fn has_price(&self) -> bool {
        [self.open, self.high, self.low, self.close]
            .iter()
            .any(|price| !price.is_nan())
    }
}
