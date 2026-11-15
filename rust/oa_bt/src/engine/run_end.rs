//! Where a run ends (03-architecture 4.2, "End of run").
//!
//! A run ends at the earliest of three instants: the end of its range (`last_day`, and
//! `last_slice` when given), the close of the last base bar any instrument has, and the
//! present (`RunConfig::present`). The slices after it are not run, so a backtest never
//! runs past the moment it is made, nor past its data: the equity series, the trades and
//! the metrics stop at the run's end, and a position still open there is reported open,
//! marked at its last close. It is not squared off in step 8 of a session close the run
//! never reached, at a price from hours earlier. The fills of that last session did
//! happen, though, so its contract notes are settled at the run's end with the same
//! rounding step 8 applies: the charges do not depend on where the run ends after its
//! last fill.
//!
//! Slots without a bar inside the run stay slices (SPEC-07); only the slots after the
//! last bar of every instrument are dropped. The kernel reads no clock: the present is a
//! value the caller resolved (the Python layer's `now=`).
//!
//! The data end before their last session closes only when that session stops earlier
//! than the data usually do. Some sessions always end with empty slots: from 2026-08-03
//! continuous trading in a stock with derivatives ends at 15:15 IST, the closing auction
//! makes no candle, and the calendar keeps 15:30 for every stock. So the empty slots
//! after the last bar stay slices, and the run ends at the end of its range (or at the
//! present), when they all lie in the last bar's session and that bar closes no earlier
//! before its session close than the last bar of the latest earlier session with a bar
//! did. A run with one session, or whose last session is cut short by the range, keeps
//! the plain rule.

use crate::exec::InstrumentId;
use crate::model::TsNanos;

use super::clock::{SliceInfo, Timeline};
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
    let last = if usual_empty_tail(timeline, &has_bar, last_bar, closed) {
        closed - 1
    } else {
        last_bar
    };
    let cause = if last + 1 < closed {
        Some(RunEndCause::LastBar)
    } else if closed < closes.len() {
        Some(RunEndCause::Present)
    } else {
        None
    };
    let end = RunEnd {
        close: closes[last],
        range_close,
        cause,
    };
    Ok((last + 1, end))
}

/// Whether the closed slots after `last_bar` (up to slice `closed`, exclusive) are the
/// empty end every session of the data has (module docs): they all lie in the last
/// bar's session, and that bar closes no earlier before its session close than the
/// last bar of the latest earlier session with a bar.
fn usual_empty_tail(
    timeline: &Timeline,
    has_bar: &impl Fn(usize) -> bool,
    last_bar: usize,
    closed: usize,
) -> bool {
    let day = |slice: usize| timeline.slice(slice).map(|info| info.day);
    if last_bar + 1 >= closed || day(closed - 1) != day(last_bar) {
        return false;
    }
    let session_start = (0..last_bar)
        .rev()
        .find(|&slice| day(slice) != day(last_bar))
        .map_or(0, |slice| slice + 1);
    let Some(earlier_bar) = (0..session_start).rev().find(|&slice| has_bar(slice)) else {
        return false;
    };
    match (
        time_left_in_session(timeline, last_bar),
        time_left_in_session(timeline, earlier_bar),
    ) {
        (Some(last_gap), Some(earlier_gap)) => last_gap <= earlier_gap,
        _ => false,
    }
}

/// Nanoseconds from the close of `slice` to the close of its session, or `None` when
/// the timeline does not reach that session's close (a range that ends inside it).
fn time_left_in_session(timeline: &Timeline, slice: usize) -> Option<i64> {
    let info = timeline.slice(slice)?;
    let session_end = (slice..timeline.len())
        .map_while(|index| timeline.slice(index).filter(|next| next.day == info.day))
        .last()?;
    session_end
        .closes_session
        .then_some(session_end.close - info.close)
}

impl Engine {
    /// Where the run ends (module docs); known from `Engine::new` on.
    pub fn run_end(&self) -> RunEnd {
        self.run_end
    }

    /// At the run's last slice, when it does not close its session: settle that
    /// session's contract notes as step 8 would (module docs). Nothing else of step 8
    /// runs, so positions and orders stay as they are.
    pub(super) fn settle_at_run_end(&mut self, slice: &SliceInfo) {
        if !slice.closes_session && slice.index + 1 == self.timeline.len() {
            self.settle_contract_notes(slice);
        }
    }
}
