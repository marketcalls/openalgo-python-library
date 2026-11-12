//! The run's clock: the grid of base-interval slices and the strategy-bar closes.
//!
//! Every slot of the calendar's base grid is a slice, whether or not any instrument
//! has a bar in it (SPEC-07), up to the run's end: the slots after the last bar of
//! every instrument, or after the present, are not run (`run_end`). A strategy bar
//! closes at a slice when the slice's base bar closes at a strategy-interval lattice
//! point or at the session close: decided by the clock, never by whether the last base
//! bar exists.
//!
//! Phase 1 runs the instruments of one exchange. Runs across exchanges with different
//! hours (Phase 3) keep this type and add one strategy-close list per exchange; their
//! decisions and fills are then ordered by instant (SPEC-35), never by slice position.

use crate::market::calendar::{union_grid, BaseInterval, Calendar, UnionGrid};
use crate::model::{Day, Exchange, TsNanos};

use super::config::{RunConfig, StrategyInterval, PERIOD_INTERVALS, PERIOD_INTERVALS_INSTEAD};
use super::error::KernelError;

/// What the steps need to know about one slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SliceInfo {
    /// Index of the slice in the run's grid.
    pub index: usize,
    /// Open of the slice's base bar, UTC nanoseconds.
    pub open: TsNanos,
    /// Close of the slice's base bar, UTC nanoseconds: the decision instant of step 7.
    pub close: TsNanos,
    /// IST session date.
    pub day: Day,
    /// The session opens with this slice.
    pub opens_session: bool,
    /// The session closes with this slice.
    pub closes_session: bool,
}

/// The grid and the strategy-bar closes of a run.
#[derive(Clone, Debug, PartialEq)]
pub struct Timeline {
    exchange: Exchange,
    grid: UnionGrid,
    /// Strategy bars by the clock as (open, close), in time order.
    strategy_bars: Vec<(TsNanos, TsNanos)>,
}

fn grid_for(
    calendar: &Calendar,
    exchange: Exchange,
    config: &RunConfig,
    base: BaseInterval,
) -> Result<UnionGrid, KernelError> {
    Ok(union_grid(
        calendar,
        &[exchange],
        config.first_day,
        config.last_day,
        base,
    )?)
}

/// The grid with only the slices inside the optional instant bounds.
fn bounded(grid: UnionGrid, first: Option<TsNanos>, last: Option<TsNanos>) -> UnionGrid {
    let keep = |ts: TsNanos| first.map_or(true, |f| ts >= f) && last.map_or(true, |l| ts <= l);
    let kept: Vec<usize> = (0..grid.ts.len()).filter(|&i| keep(grid.ts[i])).collect();
    let pick = |column: &[u16]| kept.iter().map(|&i| column[i]).collect();
    UnionGrid {
        exchanges: grid.exchanges.clone(),
        ts: kept.iter().map(|&i| grid.ts[i]).collect(),
        day: kept.iter().map(|&i| grid.day[i]).collect(),
        active: pick(&grid.active),
        session_open: pick(&grid.session_open),
        session_close: pick(&grid.session_close),
        bar_close: grid
            .bar_close
            .iter()
            .map(|closes| kept.iter().map(|&i| closes[i]).collect())
            .collect(),
    }
}

impl Timeline {
    /// The clock of a run on `exchange`.
    ///
    /// Errors: `Calendar` for an unusable range or base; `InvalidConfig` when the range
    /// holds no slice; `NotBuilt` for period strategy intervals.
    pub fn build(
        calendar: &Calendar,
        exchange: Exchange,
        config: &RunConfig,
    ) -> Result<Timeline, KernelError> {
        let base = grid_for(calendar, exchange, config, config.base)?;
        let grid = bounded(base, config.first_slice, config.last_slice);
        if grid.ts.is_empty() {
            return Err(KernelError::InvalidConfig {
                field: "start",
                reason: format!("{} has no session in the run's range", exchange.code()),
            });
        }
        let strategy_base = match config.strategy_interval {
            StrategyInterval::Intraday(nanos) => BaseInterval::Intraday(nanos),
            StrategyInterval::Daily => BaseInterval::Daily,
            StrategyInterval::Period(_) => {
                return Err(KernelError::NotBuilt {
                    feature: PERIOD_INTERVALS,
                    instead: PERIOD_INTERVALS_INSTEAD,
                })
            }
        };
        let clock = grid_for(calendar, exchange, config, strategy_base)?;
        let closes = clock.bar_close.first().cloned().unwrap_or_default();
        let strategy_bars = clock.ts.iter().copied().zip(closes).collect();
        Ok(Timeline {
            exchange,
            grid,
            strategy_bars,
        })
    }

    /// Keep the first `len` slices: the run ends with the last of them (`run_end`).
    pub(super) fn end_after(&mut self, len: usize) {
        let grid = &mut self.grid;
        grid.ts.truncate(len);
        grid.day.truncate(len);
        grid.active.truncate(len);
        grid.session_open.truncate(len);
        grid.session_close.truncate(len);
        for closes in &mut grid.bar_close {
            closes.truncate(len);
        }
    }

    /// The exchange whose calendar the run follows.
    pub fn exchange(&self) -> Exchange {
        self.exchange
    }

    /// Number of slices.
    pub fn len(&self) -> usize {
        self.grid.ts.len()
    }

    /// True when the run has no slice (never, once built).
    pub fn is_empty(&self) -> bool {
        self.grid.ts.is_empty()
    }

    /// The slice at `index`, or `None` past the end.
    pub fn slice(&self, index: usize) -> Option<SliceInfo> {
        let bit = 1u16;
        Some(SliceInfo {
            index,
            open: *self.grid.ts.get(index)?,
            close: *self.grid.bar_close.first()?.get(index)?,
            day: *self.grid.day.get(index)?,
            opens_session: self.grid.session_open.get(index)? & bit != 0,
            closes_session: self.grid.session_close.get(index)? & bit != 0,
        })
    }

    /// Open instants of every slice, in order.
    pub fn slice_opens(&self) -> &[TsNanos] {
        &self.grid.ts
    }

    /// Close instants of every slice, in order.
    pub fn slice_closes(&self) -> &[TsNanos] {
        self.grid.bar_close.first().map_or(&[], Vec::as_slice)
    }

    /// The strategy bar (open, close) that closes with this slice, if one does.
    pub fn strategy_bar_closing(&self, slice: &SliceInfo) -> Option<(TsNanos, TsNanos)> {
        self.strategy_bars
            .binary_search_by_key(&slice.close, |&(_, close)| close)
            .ok()
            .and_then(|found| self.strategy_bars.get(found).copied())
    }
}
