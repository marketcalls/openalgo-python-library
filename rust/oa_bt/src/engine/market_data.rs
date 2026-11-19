//! Market data of a run: each instrument's base bars aligned to the grid, and its
//! strategy bars.
//!
//! A base bar must open on a slice of the grid; a slice where an instrument has no bar
//! is a missing bar, which the steps handle (SPEC-06, SPEC-07). Bars outside the run's
//! slices are not used. Strategy bars are built by the resampler of `data::resample`
//! from the base bars inside the run, so a strategy bar is the aggregate of the base
//! bars present and a bucket with none is absent (SPEC-07).

use crate::data::resample::{
    resample_daily, resample_daily_ending_earlier, resample_intraday,
    resample_intraday_ending_earlier, BarsIn, BarsOut, ClosedDayPolicy,
};
use crate::exec::{InstrumentId, InstrumentSpec};
use crate::market::calendar::{format_ist, BaseInterval, Calendar};
use crate::model::{counted, TsNanos};
use crate::rms::BarPrices;

use super::clock::Timeline;
use super::config::{
    BarColumns, RunConfig, StrategyInterval, PERIOD_INTERVALS, PERIOD_INTERVALS_INSTEAD,
};
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

/// One base bar handed to a live session as its slice closes (`Engine::add_bar`):
/// the open instant in UTC nanoseconds, prices in rupees, volume in units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BaseBar {
    pub ts_open: TsNanos,
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
        let prices = [
            bars.open[row],
            bars.high[row],
            bars.low[row],
            bars.close[row],
        ];
        valid_prices(spec, bars.ts_open[row], prices)?;
    }
    Ok(())
}

/// Check one bar's open, high, low and close: positive, finite, the high and low around
/// the open and close.
fn valid_prices(
    spec: &InstrumentSpec,
    ts_open: TsNanos,
    ohlc: [f64; 4],
) -> Result<(), KernelError> {
    let [o, h, l, c] = ohlc;
    let finite = ohlc.iter().all(|v| v.is_finite() && *v > 0.0);
    if !finite || l > o.min(c) || h < o.max(c) {
        return Err(bad(
            spec,
            format!(
                "the bar opening at {} is not a valid OHLC bar",
                format_ist(ts_open)
            ),
        ));
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
                format!(
                    "the bar opening at {} does not open on the base-interval grid",
                    format_ist(ts)
                ),
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
    let shift = i64::from(config.perturbation.bar_shift);
    let resampled = match (config.base, config.strategy_interval) {
        (BaseInterval::Intraday(base), StrategyInterval::Intraday(strategy))
            if strategy != base && shift > 0 =>
        {
            let (intervals, exchange) = ((base, strategy), spec.exchange);
            resample_intraday_ending_earlier(
                calendar,
                exchange,
                &input,
                intervals,
                shift * base,
                policy,
            )?
        }
        (BaseInterval::Intraday(base), StrategyInterval::Intraday(strategy))
            if strategy != base =>
        {
            resample_intraday(calendar, spec.exchange, &input, base, strategy, policy)?
        }
        (BaseInterval::Intraday(base), StrategyInterval::Daily) if shift > 0 => {
            resample_daily_ending_earlier(
                calendar,
                spec.exchange,
                &input,
                base,
                shift * base,
                policy,
            )?
        }
        (BaseInterval::Intraday(base), StrategyInterval::Daily) => {
            resample_daily(calendar, spec.exchange, &input, base, policy)?
        }
        (_, StrategyInterval::Period(_)) => {
            return Err(KernelError::NotBuilt {
                feature: PERIOD_INTERVALS,
                instead: PERIOD_INTERVALS_INSTEAD,
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
                    "{} for {}",
                    counted(bars.len(), "bar set", "bar sets"),
                    counted(instruments.len(), "instrument", "instruments")
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

    /// Add a base bar of `instrument` at slice `slot` of the grid (`live_clock`); the
    /// caller has checked that the slot has not run, and rebuilds the strategy bars
    /// (`rebuild_strategy_bars`).
    ///
    /// Errors: `InvalidData` for a bar that is not a valid OHLC bar, has no finite
    /// volume, or opens no later than the instrument's last bar.
    pub(super) fn push_bar(
        &mut self,
        spec: &InstrumentSpec,
        instrument: InstrumentId,
        slot: usize,
        bar: BaseBar,
    ) -> Result<(), KernelError> {
        valid_prices(spec, bar.ts_open, [bar.open, bar.high, bar.low, bar.close])?;
        if !bar.volume.is_finite() || bar.volume < 0.0 {
            return Err(bad(
                spec,
                format!(
                    "the bar opening at {} has no usable volume",
                    format_ist(bar.ts_open)
                ),
            ));
        }
        let Some(data) = self.instruments.get_mut(instrument.index()) else {
            return Err(bad(spec, "the instrument has no bars".to_string()));
        };
        if data
            .base
            .ts_open
            .last()
            .is_some_and(|&last| bar.ts_open <= last)
        {
            return Err(bad(
                spec,
                format!(
                    "the bar opening at {} is not after the instrument's last bar",
                    format_ist(bar.ts_open)
                ),
            ));
        }
        let base = &mut data.base;
        base.ts_open.push(bar.ts_open);
        base.open.push(bar.open);
        base.high.push(bar.high);
        base.low.push(bar.low);
        base.close.push(bar.close);
        base.volume.push(bar.volume);
        if let Some(cell) = data.row_at_slice.get_mut(slot) {
            *cell = Some(base.ts_open.len() - 1);
        }
        Ok(())
    }

    /// Rebuild `instrument`'s strategy bars from its base bars, as `build` does;
    /// `closes` are the slices' closes.
    ///
    /// Errors: `Resample` when the strategy bars cannot be built.
    pub(super) fn rebuild_strategy_bars(
        &mut self,
        spec: &InstrumentSpec,
        instrument: InstrumentId,
        closes: &[TsNanos],
        calendar: &Calendar,
        config: &RunConfig,
    ) -> Result<(), KernelError> {
        let Some(data) = self.instruments.get_mut(instrument.index()) else {
            return Ok(());
        };
        let aligned = aligned_out(&data.base, &data.row_at_slice, closes);
        data.strategy = strategy_bars(calendar, spec, aligned, config)?;
        Ok(())
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
