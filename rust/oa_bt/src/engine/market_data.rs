//! Market data of a run: each instrument's base bars aligned to the grid, and its
//! strategy bars.
//!
//! A base bar must open on a slice of the grid; a slice where an instrument has no bar
//! is a missing bar, which the steps handle (SPEC-06, SPEC-07). Bars outside the run's
//! slices are not used. Strategy bars are built by the resampler of `data::resample`
//! from the base bars inside the run, so a strategy bar is the aggregate of the base
//! bars present and a bucket with none is absent (SPEC-07).

use crate::data::resample::{resample_daily, resample_intraday, BarsIn, BarsOut, ClosedDayPolicy};
use crate::exec::{InstrumentId, InstrumentSpec};
use crate::market::calendar::{BaseInterval, Calendar};
use crate::model::TsNanos;
use crate::rms::BarPrices;

use super::clock::Timeline;
use super::config::{BarColumns, RunConfig, StrategyInterval};
use super::error::KernelError;

/// A strategy bar handed to `on_bar`: prices in rupees, volume in units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrategyBar {
    pub instrument: InstrumentId,
    /// Open of the bar on the strategy lattice, UTC nanoseconds.
    pub ts_open: TsNanos,
    /// Close of the bar, UTC nanoseconds: the decision instant.
    pub ts_close: TsNanos,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// One instrument's bars.
#[derive(Clone, Debug, PartialEq)]
struct InstrumentData {
    base: BarColumns,
    /// Row of `base` for every slice, `None` where the bar is missing.
    row_at_slice: Vec<Option<usize>>,
    strategy: BarsOut,
}

/// The bars of every instrument, in registration order.
#[derive(Clone, Debug, PartialEq)]
pub struct MarketData {
    instruments: Vec<InstrumentData>,
}

fn bad(spec: &InstrumentSpec, reason: String) -> KernelError {
    KernelError::InvalidData {
        symbol: spec.symbol.clone(),
        reason,
    }
}

/// Check one instrument's columns: one length, increasing opens, usable OHLC.
fn validate(spec: &InstrumentSpec, bars: &BarColumns) -> Result<(), KernelError> {
    let n = bars.ts_open.len();
    let lengths = [
        bars.open.len(),
        bars.high.len(),
        bars.low.len(),
        bars.close.len(),
    ];
    if lengths.iter().any(|&len| len != n) || bars.volume.len() != n {
        return Err(bad(spec, "the bar columns differ in length".to_string()));
    }
    if bars.ts_open.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(bad(
            spec,
            "bar opens are not strictly increasing".to_string(),
        ));
    }
    for row in 0..n {
        let (o, h, l, c) = (
            bars.open[row],
            bars.high[row],
            bars.low[row],
            bars.close[row],
        );
        let finite = [o, h, l, c].iter().all(|v| v.is_finite() && *v > 0.0);
        if !finite || l > o.min(c) || h < o.max(c) {
            return Err(bad(
                spec,
                format!("bar at {} is not a valid OHLC bar", bars.ts_open[row]),
            ));
        }
    }
    Ok(())
}

/// The row of every slice; a bar inside the run that is off the grid is an error.
fn align(
    spec: &InstrumentSpec,
    bars: &BarColumns,
    timeline: &Timeline,
) -> Result<Vec<Option<usize>>, KernelError> {
    let opens = timeline.slice_opens();
    let (first, last) = match (opens.first(), opens.last()) {
        (Some(&first), Some(&last)) => (first, last),
        _ => return Ok(Vec::new()),
    };
    let mut rows = vec![None; opens.len()];
    for (row, &ts) in bars.ts_open.iter().enumerate() {
        if ts < first || ts > last {
            continue;
        }
        let slot = opens.binary_search(&ts).map_err(|_| {
            bad(
                spec,
                format!("bar at {ts} does not open on the base-interval grid"),
            )
        })?;
        if let Some(cell) = rows.get_mut(slot) {
            *cell = Some(row);
        }
    }
    Ok(rows)
}

/// The aligned bars as resampler output: (open, close by the grid, values).
fn aligned_out(bars: &BarColumns, rows: &[Option<usize>], closes: &[TsNanos]) -> BarsOut {
    let mut out = BarsOut::default();
    for (slot, row) in rows.iter().enumerate() {
        let (Some(row), Some(&close)) = (row, closes.get(slot)) else {
            continue;
        };
        out.ts_open.push(bars.ts_open[*row]);
        out.ts_close.push(close);
        out.open.push(bars.open[*row]);
        out.high.push(bars.high[*row]);
        out.low.push(bars.low[*row]);
        out.close.push(bars.close[*row]);
        out.volume.push(bars.volume[*row]);
        out.oi.push(f64::NAN);
    }
    out
}

/// Strategy bars from the aligned base bars.
fn strategy_bars(
    calendar: &Calendar,
    spec: &InstrumentSpec,
    aligned: BarsOut,
    config: &RunConfig,
) -> Result<BarsOut, KernelError> {
    let input = BarsIn {
        ts_open: &aligned.ts_open,
        open: &aligned.open,
        high: &aligned.high,
        low: &aligned.low,
        close: &aligned.close,
        volume: &aligned.volume,
        oi: &aligned.oi,
    };
    let policy = ClosedDayPolicy::Error;
    let resampled = match (config.base, config.strategy_interval) {
        (BaseInterval::Intraday(base), StrategyInterval::Intraday(strategy))
            if strategy != base =>
        {
            resample_intraday(calendar, spec.exchange, &input, base, strategy, policy)?
        }
        (BaseInterval::Intraday(base), StrategyInterval::Daily) => {
            resample_daily(calendar, spec.exchange, &input, base, policy)?
        }
        (_, StrategyInterval::Period(_)) => {
            return Err(KernelError::NotBuilt {
                feature: "weekly, monthly, quarterly and yearly strategy intervals",
            })
        }
        _ => return Ok(aligned),
    };
    Ok(resampled.bars)
}

impl MarketData {
    /// Validate, align and aggregate the bars of every instrument.
    ///
    /// Errors: `InvalidData` for unusable columns or a bar off the grid; `InvalidConfig`
    /// when the bars and the instruments differ in number; `Resample` when the strategy
    /// bars cannot be built.
    pub fn build(
        bars: Vec<BarColumns>,
        instruments: &[InstrumentSpec],
        timeline: &Timeline,
        calendar: &Calendar,
        config: &RunConfig,
    ) -> Result<MarketData, KernelError> {
        if bars.len() != instruments.len() {
            return Err(KernelError::InvalidConfig {
                field: "data",
                reason: format!(
                    "{} bar sets for {} instruments",
                    bars.len(),
                    instruments.len()
                ),
            });
        }
        let mut data = Vec::with_capacity(bars.len());
        for (spec, base) in instruments.iter().zip(bars) {
            validate(spec, &base)?;
            let row_at_slice = align(spec, &base, timeline)?;
            let aligned = aligned_out(&base, &row_at_slice, timeline.slice_closes());
            let strategy = strategy_bars(calendar, spec, aligned, config)?;
            data.push(InstrumentData {
                base,
                row_at_slice,
                strategy,
            });
        }
        Ok(MarketData { instruments: data })
    }

    /// The base bar of an instrument at a slice, or `None` when it is missing.
    pub fn bar(&self, instrument: InstrumentId, slice: usize) -> Option<BarPrices> {
        let data = self.instruments.get(instrument.index())?;
        let row = (*data.row_at_slice.get(slice)?)?;
        Some(BarPrices {
            open: *data.base.open.get(row)?,
            high: *data.base.high.get(row)?,
            low: *data.base.low.get(row)?,
            close: *data.base.close.get(row)?,
        })
    }

    /// The strategy bar of an instrument that opens at `ts_open`, or `None` when no base
    /// bar of it was present.
    pub fn strategy_bar(&self, instrument: InstrumentId, ts_open: TsNanos) -> Option<StrategyBar> {
        let bars = &self.instruments.get(instrument.index())?.strategy;
        let row = bars.ts_open.binary_search(&ts_open).ok()?;
        Some(StrategyBar {
            instrument,
            ts_open,
            ts_close: *bars.ts_close.get(row)?,
            open: *bars.open.get(row)?,
            high: *bars.high.get(row)?,
            low: *bars.low.get(row)?,
            close: *bars.close.get(row)?,
            volume: *bars.volume.get(row)?,
        })
    }
}
