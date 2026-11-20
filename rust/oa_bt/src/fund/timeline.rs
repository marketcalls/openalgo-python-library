//! The fund's merged time line (18-multi-strategy 2.3, 2.6): where each engine stands on
//! its own grid (`Lane`), the next instant over all of them, and the engines that take a
//! phase at it.
//!
//! An engine's next instant is the open of its next slice while it stands before it, and
//! that slice's close once its open phase has run. Of the next instants of every engine
//! still running, the earliest comes next; at an instant that is both a close and an open,
//! the close (the round) comes first. Engines are listed in call order.

use crate::engine::kernel::Phase;
use crate::engine::{Engine, SliceInfo};
use crate::model::{Day, TsNanos};

use super::FundError;

/// Where an engine stands on its grid, as the fund has moved it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Cursor {
    /// Before this slice: its open phase runs at its open.
    Before(SliceInfo),
    /// This slice's open phase has run: its round runs at its close.
    Open(SliceInfo),
    /// Its grid is exhausted: its run finishes before the next instant.
    Done,
    /// Its run has finished: it takes no more phases.
    Ended,
}

/// One engine on the time line.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Lane {
    pub(super) cursor: Cursor,
    /// The engine's events already read for the refused orders (`rounds`).
    pub(super) events_read: usize,
    /// What tells the lane's engine from another: its capital (rupees), its number of
    /// slices and its first slice's open (`check_engines`).
    identity: (f64, usize, TsNanos),
}

impl Lane {
    /// True when `engine` is the lane's own (its identity, `check_engines`).
    pub(super) fn holds(&self, engine: &Engine) -> bool {
        self.identity == identity(engine)
    }
}

/// The next instant of the time line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Instant {
    /// A round: the slices closing at `at`, the first of them (in call order) on IST
    /// session day `day`. Slices closing at one instant are of one session day: no Indian
    /// session runs past midnight.
    Close { at: TsNanos, day: Day },
    /// The open phases of the slices opening at this instant.
    Open(TsNanos),
}

fn identity(engine: &Engine) -> (f64, usize, TsNanos) {
    let grid = engine.timeline();
    let first = grid.slice_opens().first().copied().unwrap_or(TsNanos::MIN);
    (engine.capital_rupees(), grid.len(), first)
}

/// The cursor of an engine standing before slice `index`: `Done` past its grid.
pub(super) fn before(engine: &Engine, index: usize) -> Cursor {
    engine
        .timeline()
        .slice(index)
        .map_or(Cursor::Done, Cursor::Before)
}

/// One lane per engine, each before its first slice.
///
/// Errors: `InvalidConfig` (field `engines`) for no engine, a live session and an engine
/// that has advanced.
pub(super) fn lanes(engines: &[&Engine]) -> Result<Vec<Lane>, FundError> {
    let refuse = |reason: String| FundError::InvalidConfig {
        field: "engines",
        reason,
    };
    if engines.is_empty() {
        return Err(refuse("a portfolio run needs at least one strategy".into()));
    }
    let mut lanes = Vec::with_capacity(engines.len());
    for (strategy, engine) in engines.iter().enumerate() {
        if engine.is_live() {
            return Err(refuse(format!(
                "strategy {strategy} is a live session, whose account is the broker's: a \
                 fund runs backtests"
            )));
        }
        if engine.phase() != Phase::Ready(0) {
            return Err(refuse(format!(
                "strategy {strategy}'s engine has advanced already: give engines before \
                 their first advance"
            )));
        }
        lanes.push(Lane {
            cursor: before(engine, 0),
            events_read: 0,
            identity: identity(engine),
        });
    }
    Ok(lanes)
}

/// Check that `engines` are the lanes' own, in call order.
///
/// Errors: `InvalidConfig` (field `engines`) naming the first that is not.
pub(super) fn check_engines(lanes: &[Lane], engines: &[&mut Engine]) -> Result<(), FundError> {
    let refuse = |reason: String| FundError::InvalidConfig {
        field: "engines",
        reason,
    };
    if engines.len() != lanes.len() {
        return Err(refuse(format!(
            "the fund runs {} strategies and was given {} engines: give its engines, in \
             call order",
            lanes.len(),
            engines.len()
        )));
    }
    let wrong = lanes
        .iter()
        .zip(engines)
        .position(|(lane, engine)| !lane.holds(engine));
    match wrong {
        Some(strategy) => Err(refuse(format!(
            "engine {strategy} is not the fund's engine of strategy {strategy}: give its \
             engines, in call order"
        ))),
        None => Ok(()),
    }
}

/// The next instant over every lane still running (module docs); `None` once every
/// engine has finished.
pub(super) fn next_instant(lanes: &[Lane]) -> Option<Instant> {
    // The earliest close with the session day of its first slice in call order, and the
    // earliest open; a later lane replaces them only when strictly earlier.
    let mut close: Option<(TsNanos, Day)> = None;
    let mut open: Option<TsNanos> = None;
    for lane in lanes {
        match lane.cursor {
            Cursor::Open(slice) if close.map_or(true, |(at, _)| slice.close < at) => {
                close = Some((slice.close, slice.day));
            }
            Cursor::Before(slice) if open.map_or(true, |at| slice.open < at) => {
                open = Some(slice.open);
            }
            _ => {}
        }
    }
    match (close, open) {
        (Some((close, _)), Some(open)) if open < close => Some(Instant::Open(open)),
        (Some((at, day)), _) => Some(Instant::Close { at, day }),
        (None, Some(open)) => Some(Instant::Open(open)),
        (None, None) => None,
    }
}

/// The lanes whose slice closes at `at` (a round's participants), in call order, into
/// `turns`.
pub(super) fn closing_at(lanes: &[Lane], at: TsNanos, turns: &mut Vec<usize>) {
    turns.clear();
    let closes = |lane: &Lane| matches!(lane.cursor, Cursor::Open(slice) if slice.close == at);
    turns.extend((0..lanes.len()).filter(|&i| closes(&lanes[i])));
}

/// The lanes whose slice opens at `at`, in call order, into `turns`.
pub(super) fn opening_at(lanes: &[Lane], at: TsNanos, turns: &mut Vec<usize>) {
    turns.clear();
    let opens = |lane: &Lane| matches!(lane.cursor, Cursor::Before(slice) if slice.open == at);
    turns.extend((0..lanes.len()).filter(|&i| opens(&lanes[i])));
}

/// Keep in `turns` (a round's participants, in call order) the lanes whose grid ended at
/// the round.
pub(super) fn keep_done(lanes: &[Lane], turns: &mut Vec<usize>) {
    turns.retain(|&i| lanes[i].cursor == Cursor::Done);
}
