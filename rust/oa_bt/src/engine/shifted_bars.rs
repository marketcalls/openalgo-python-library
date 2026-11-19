//! The decision instants of a run whose strategy bars end earlier (design 17, section
//! 2.7; `bar_shift=`): the shifted-bars check asks whether a strategy depends on where
//! its bars start.
//!
//! The clock's strategy bars are the buckets of the resampler that ends each bar
//! `bar_shift` base bars earlier (`data::resample::shifted`), taken over the run's grid
//! with the same bucket rules, so the bars the market data builds close exactly where the
//! clock pauses: on an intraday interval the session's lattice moved earlier, on a daily
//! one each day ending before its session's close, its bars after that instant opening
//! the next session's bar. A day's end must leave part of its session: a shift as long as
//! a session is refused.

use crate::data::resample::shifted::{day_bucket, shifted_start};
use crate::market::calendar::{format_day, BaseInterval, Calendar, LatticeCursor};
use crate::model::{Exchange, TsNanos};

use super::config::{RunConfig, StrategyInterval};
use super::error::KernelError;

/// The (open, close) of every strategy bar over the slices that open at `slice_opens`,
/// in order, each ending `config.perturbation.bar_shift` base bars earlier (module
/// docs).
///
/// Errors: `InvalidConfig` when a day's end falls at or before its session's open.
pub(super) fn shifted_strategy_bars(
    calendar: &Calendar,
    exchange: Exchange,
    slice_opens: &[TsNanos],
    config: &RunConfig,
) -> Result<Vec<(TsNanos, TsNanos)>, KernelError> {
    let BaseInterval::Intraday(base) = config.base else {
        return Ok(Vec::new());
    };
    let shift = i64::from(config.perturbation.bar_shift) * base;
    let mut cursor = LatticeCursor::new(calendar, exchange);
    let mut bars: Vec<(TsNanos, TsNanos)> = Vec::new();
    for &ts in slice_opens {
        let Some(lattice) = cursor.at(ts) else {
            continue;
        };
        let Some(session) = lattice.session else {
            continue;
        };
        let bucket = match config.strategy_interval {
            StrategyInterval::Intraday(target) => {
                let start = shifted_start(&lattice, ts, target, shift);
                Some((start, (start + target).min(session.close)))
            }
            _ if session.close - shift <= session.open => {
                return Err(KernelError::InvalidConfig {
                    field: "bar_shift",
                    reason: format!(
                        "the day of {} would end at or before its session's open: give a \
                         shift shorter than the session",
                        format_day(session.day)
                    ),
                });
            }
            _ => day_bucket(calendar, exchange, &session, ts, shift),
        };
        if let Some(bucket) = bucket {
            if bars.last() != Some(&bucket) {
                bars.push(bucket);
            }
        }
    }
    Ok(bars)
}
