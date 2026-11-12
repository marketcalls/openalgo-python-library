//! Tests of the resamplers, one file per resampler.
//!
//! Expected bars are worked out by hand from the input in each test; expected instants
//! are literal UTC nanoseconds with the IST time in a comment.

mod bars;
mod daily;
mod intraday;
mod period;

use crate::model::TsNanos;

use super::BarsIn;

/// Owned bar columns for a test, with a `BarsIn` view.
struct Columns {
    ts_open: Vec<TsNanos>,
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    volume: Vec<f64>,
    oi: Vec<f64>,
}

impl Columns {
    /// Bars whose prices all equal the row number, with a volume of 1 and an open
    /// interest of ten times the row number.
    fn ramp(ts_open: Vec<TsNanos>) -> Columns {
        let price: Vec<f64> = (0..ts_open.len()).map(|row| row as f64).collect();
        let volume = vec![1.0; ts_open.len()];
        let oi = price.iter().map(|value| value * 10.0).collect();
        Columns {
            ts_open,
            open: price.clone(),
            high: price.clone(),
            low: price.clone(),
            close: price,
            volume,
            oi,
        }
    }

    fn bars(&self) -> BarsIn<'_> {
        BarsIn {
            ts_open: &self.ts_open,
            open: &self.open,
            high: &self.high,
            low: &self.low,
            close: &self.close,
            volume: &self.volume,
            oi: &self.oi,
        }
    }
}
