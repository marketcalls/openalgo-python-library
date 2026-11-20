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
use crate::exec::InstrumentSpec;
use crate::model::{Day, Exchange, TsNanos};

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
    identity: Identity,
}

/// What tells the lane's engine from another (`check_engines`). Two engines alike in all
/// of it (one capital and one grid on the same instruments) cannot be told apart.
#[derive(Clone, Debug, PartialEq)]
struct Identity {
    /// The strategy's capital, rupees.
    capital_rupees: f64,
    /// The grid's number of slices.
    slices: usize,
    /// The first slice's open, UTC nanoseconds (`first_open`).
    first_open: TsNanos,
    /// Each instrument's symbol and exchange, in registration order.
    instruments: Vec<(String, Exchange)>,
}

impl Lane {
    /// True when `engine` is the lane's own (its identity, `check_engines`). It compares
    /// in place, allocating nothing: the fund checks every engine at every `advance`.
    pub(super) fn holds(&self, engine: &Engine) -> bool {
        let id = &self.identity;
        let specs = engine.instruments();
        let same = |((symbol, exchange), spec): (&(String, Exchange), &InstrumentSpec)| {
            *symbol == spec.symbol && *exchange == spec.exchange
        };
        id.capital_rupees == engine.capital_rupees()
            && id.slices == engine.timeline().len()
            && id.first_open == first_open(engine)
            && id.instruments.len() == specs.len()
            && id.instruments.iter().zip(specs).all(same)
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

/// The open of the engine's first slice, UTC nanoseconds; `TsNanos::MIN` for an empty
/// grid.
fn first_open(engine: &Engine) -> TsNanos {
    let opens = engine.timeline().slice_opens();
    opens.first().copied().unwrap_or(TsNanos::MIN)
}

fn identity(engine: &Engine) -> Identity {
    let instruments = engine.instruments().iter();
    Identity {
        capital_rupees: engine.capital_rupees(),
        slices: engine.timeline().len(),
        first_open: first_open(engine),
        instruments: instruments
            .map(|spec| (spec.symbol.clone(), spec.exchange))
            .collect(),
    }
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
    let refuse = |strategy: Option<usize>, reason: &str| FundError::InvalidConfig {
        field: "engines",
        strategy,
        reason: reason.to_string(),
    };
    if engines.is_empty() {
        return Err(refuse(None, "a portfolio run needs at least one strategy"));
    }
    let mut lanes = Vec::with_capacity(engines.len());
    for (strategy, engine) in engines.iter().enumerate() {
        if engine.is_live() {
            return Err(refuse(
                Some(strategy),
                "is a live session, whose account is the broker's: a fund runs backtests",
            ));
        }
        if engine.phase() != Phase::Ready(0) {
            return Err(refuse(
                Some(strategy),
                "has an engine that has advanced already: give engines before their first \
                 advance",
            ));
        }
        lanes.push(Lane {
            cursor: before(engine, 0),
            events_read: 0,
            identity: identity(engine),
        });
    }
    Ok(lanes)
}

/// Check that `engines` are the lanes' own, in call order, as far as their identities
/// tell them apart (`Lane::holds`).
///
/// Errors: `InvalidConfig` (field `engines`) for a number of engines other than the
/// lanes', and naming the first strategy whose engine is not its own.
pub(super) fn check_engines(lanes: &[Lane], engines: &[&mut Engine]) -> Result<(), FundError> {
    let refuse = |strategy: Option<usize>, reason: String| FundError::InvalidConfig {
        field: "engines",
        strategy,
        reason,
    };
    if engines.len() != lanes.len() {
        return Err(refuse(
            None,
            format!(
                "the fund runs {} strategies and was given {} engines: give its engines, in \
                 call order",
                lanes.len(),
                engines.len()
            ),
        ));
    }
    let wrong = lanes
        .iter()
        .zip(engines)
        .position(|(lane, engine)| !lane.holds(engine));
    match wrong {
        Some(strategy) => Err(refuse(
            Some(strategy),
            "was given an engine that is not its own: give the fund's engines, in call order"
                .to_string(),
        )),
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
