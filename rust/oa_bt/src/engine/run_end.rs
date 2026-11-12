//! Where a run ends (03-architecture 4.2, "End of run").
//!
//! A run ends at the earliest of three instants: the end of its range (`last_day`, and
//! `last_slice` when given), the close of the last base bar any instrument has, and the
//! present (`RunConfig::present`). The slices after it are not run, so a backtest never
//! runs past the moment it is made, nor past its data: the equity series, the trades and
//! the metrics stop at the run's end, and a position still open there is reported open,
//! marked at its last close. It is not squared off in step 8 of a session close the run
//! never reached, at a price from hours earlier.
//!
//! Slots without a bar inside the run stay slices (SPEC-07); only the slots after the
//! last bar of every instrument are dropped. The kernel reads no clock: the present is a
//! value the caller resolved (the Python layer's `now=`).

use crate::exec::InstrumentId;
use crate::model::TsNanos;

use super::clock::Timeline;
use super::error::KernelError;
use super::kernel::Engine;
use super::market_data::MarketData;

/// Why a run ends before the end of its range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunEndCause {
    /// No instrument has a bar after the run's last slice.
    LastBar,
    /// The slices after the run's last slice had not closed at the present.
    Present,
}

impl RunEndCause {
    /// The code the bindings report: "last_bar" or "present".
    pub fn code(self) -> &'static str {
        match self {
            RunEndCause::LastBar => "last_bar",
            RunEndCause::Present => "present",
        }
    }
}

/// Where a run ends, UTC nanoseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunEnd {
    /// Close of the run's last slice: the last equity point.
    pub close: TsNanos,
    /// Close of the last slice of the run's range (`last_day`, `last_slice`).
    pub range_close: TsNanos,
    /// Why the run ends before `range_close`; `None` when it ends there.
    pub cause: Option<RunEndCause>,
}

/// The number of slices of `timeline` the run keeps, and where it ends (module docs).
///
/// Errors: `InvalidConfig` naming `now` when no slice of the range has closed at the
/// present, and naming `data` when no instrument has a bar in the slices that remain.
pub(super) fn end_of_run(
    timeline: &Timeline,
    market: &MarketData,
    instruments: usize,
    present: Option<TsNanos>,
) -> Result<(usize, RunEnd), KernelError> {
    let closes = timeline.slice_closes();
    let closed = present.map_or(closes.len(), |now| closes.partition_point(|&c| c <= now));
    let range_close = match (closed, closes.last()) {
        (1.., Some(&last)) => last,
        _ => {
            return Err(KernelError::InvalidConfig {
                field: "now",
                reason: "no slice of the run's range has closed at the present: pass an \
                         earlier start, or a later now"
                    .to_string(),
            })
        }
    };
    let has_bar = |slice: usize| {
        (0..instruments).any(|i| market.bar(InstrumentId(i as u32), slice).is_some())
    };
    let Some(last_bar) = (0..closed).rev().find(|&slice| has_bar(slice)) else {
        return Err(KernelError::InvalidConfig {
            field: "data",
            reason: "no instrument has a bar in the run's range that closed at the present: \
                     check start, end, now and the data"
                .to_string(),
        });
    };
    let close = closes[last_bar];
    let cause = if last_bar + 1 < closed {
        Some(RunEndCause::LastBar)
    } else if closed < closes.len() {
        Some(RunEndCause::Present)
    } else {
        None
    };
    let end = RunEnd {
        close,
        range_close,
        cause,
    };
    Ok((last_bar + 1, end))
}

impl Engine {
    /// Where the run ends (module docs); known from `Engine::new` on.
    pub fn run_end(&self) -> RunEnd {
        self.run_end
    }
}
