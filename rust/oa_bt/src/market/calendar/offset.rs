//! Detection of a 5 hour 30 minute timestamp offset in an intraday series.
//!
//! India Standard Time is UTC plus 5:30, so a source that mixes the two up yields bars
//! that are all 5 hours 30 minutes late or early. Such a series still looks like bars,
//! and a single late bar or a single day proves nothing, so the check has three
//! outcomes and a fixed decision rule.
//!
//! A bar is misplaced when it is not a session bar on the lattice: before the open,
//! after the close, on a closed day or off the lattice. A day is covered when its
//! session bars span at least half of its session, from the first to the last.
//!
//! 1. Count the misplaced bars as given, and again after undoing each shift (all
//!    timestamps moved 5:30 earlier, and 5:30 later).
//! 2. Aligned: neither shift leaves fewer misplaced bars than the series as given.
//! 3. Otherwise take the shift that leaves the fewest. The series is shifted when
//!    undoing it removes at least nine in ten of the misplaced bars and leaves at
//!    least two covered days.
//! 4. Anything else is inconclusive: a shift fits better, but the evidence is one day,
//!    or thin trading, or only part of the misplaced bars. A thin evening series, a
//!    few bars after the close and one complete shifted day all end here, with the
//!    counts, so the caller can warn and a person can decide.

use crate::model::{counted, Exchange, TsNanos, IST_OFFSET_NANOS};

use super::lattice::{LatticeCursor, Placement};
use super::store::Calendar;

/// Undoing the shift must remove at least this many in ten of the misplaced bars.
const DECISIVE_TENTHS: u64 = 9;
/// Days that must be covered after undoing the shift before it is called a shift.
const MIN_COVERED_DAYS: u32 = 2;

/// Which way the timestamps of a series are off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShiftDirection {
    /// Every timestamp is 5 hours 30 minutes later than the true instant.
    Late5h30,
    /// Every timestamp is 5 hours 30 minutes earlier than the true instant.
    Early5h30,
}

impl ShiftDirection {
    /// A stable name for reports: "late_5h30" or "early_5h30".
    pub fn code(self) -> &'static str {
        match self {
            ShiftDirection::Late5h30 => "late_5h30",
            ShiftDirection::Early5h30 => "early_5h30",
        }
    }

    /// Nanoseconds to subtract from every timestamp to undo the shift.
    fn undo_nanos(self) -> i64 {
        match self {
            ShiftDirection::Late5h30 => IST_OFFSET_NANOS,
            ShiftDirection::Early5h30 => -IST_OFFSET_NANOS,
        }
    }
}

/// The evidence behind a shifted or inconclusive outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShiftEvidence {
    /// The shift that fits the series best.
    pub direction: ShiftDirection,
    /// Misplaced bars with the timestamps as given.
    pub misplaced: u64,
    /// Misplaced bars after undoing the shift.
    pub misplaced_if_undone: u64,
    /// Days whose session bars span at least half their session after undoing it.
    pub covered_days_if_undone: u32,
}

/// The outcome of the offset check.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OffsetCheck {
    /// No shift places the bars better than they are placed as given.
    #[default]
    Aligned,
    /// The timestamps are 5 hours 30 minutes off. This is an error.
    Shifted(ShiftEvidence),
    /// A shift would place the bars better, but not decisively. This is a warning.
    Inconclusive(ShiftEvidence),
}

impl OffsetCheck {
    /// "aligned", "shifted" or "inconclusive".
    pub fn code(&self) -> &'static str {
        match self {
            OffsetCheck::Aligned => "aligned",
            OffsetCheck::Shifted(_) => "shifted",
            OffsetCheck::Inconclusive(_) => "inconclusive",
        }
    }

    /// The evidence, when the outcome is not aligned.
    pub fn evidence(&self) -> Option<&ShiftEvidence> {
        match self {
            OffsetCheck::Aligned => None,
            OffsetCheck::Shifted(evidence) | OffsetCheck::Inconclusive(evidence) => Some(evidence),
        }
    }

    /// Why the outcome is what it is, in words, or an empty string when aligned.
    pub fn reason(&self) -> String {
        let Some(evidence) = self.evidence() else {
            return String::new();
        };
        let ShiftEvidence {
            direction,
            misplaced,
            misplaced_if_undone,
            covered_days_if_undone,
        } = *evidence;
        let (way, undo) = match direction {
            ShiftDirection::Late5h30 => ("late", "earlier"),
            ShiftDirection::Early5h30 => ("early", "later"),
        };
        let counts = format!(
            "{} outside the session or off its lattice as given, and \
             {misplaced_if_undone} if every timestamp is moved 5 hours 30 minutes {undo}",
            counted(misplaced, "bar is", "bars are")
        );
        let days = counted(covered_days_if_undone, "day", "days");
        match self {
            OffsetCheck::Shifted(_) => format!(
                "the timestamps are 5 hours 30 minutes {way}: {counts}, which then covers \
                 {days}"
            ),
            _ => format!(
                "the offset check is inconclusive: {counts}, but that covers only \
                 {days} by at least half a session and the rule \
                 needs {MIN_COVERED_DAYS} with nine in ten misplaced bars explained. The \
                 timestamps may be 5 hours 30 minutes {way}, or the series may be thin, or \
                 the calendar may lack a session"
            ),
        }
    }
}

/// How a series fits the calendar after its timestamps are moved.
struct Fit {
    misplaced: u64,
    covered_days: u32,
}

/// The session bars of one session met so far: the first and the last.
struct Span {
    session_open: TsNanos,
    session_nanos: i64,
    first: TsNanos,
    last: TsNanos,
}

impl Span {
    /// Whether the bars span at least half of the session.
    fn covers_half(&self, interval_nanos: i64) -> bool {
        (self.last - self.first + interval_nanos) * 2 >= self.session_nanos
    }
}

/// The fit of a sorted series after subtracting `undo_nanos` from every timestamp.
fn fit(
    calendar: &Calendar,
    exchange: Exchange,
    ts_open: &[TsNanos],
    interval_nanos: i64,
    undo_nanos: i64,
) -> Fit {
    let mut cursor = LatticeCursor::new(calendar, exchange);
    let mut fit = Fit {
        misplaced: 0,
        covered_days: 0,
    };
    let mut span: Option<Span> = None;
    for &raw in ts_open {
        let ts = raw.saturating_sub(undo_nanos);
        let session = cursor
            .at(ts)
            .filter(|lattice| lattice.place(ts, interval_nanos) == Placement::InSession)
            .and_then(|lattice| lattice.session);
        let Some(session) = session else {
            fit.misplaced += 1;
            continue;
        };
        match span.as_mut() {
            Some(current) if current.session_open == session.open => current.last = ts,
            _ => {
                let finished = span.replace(Span {
                    session_open: session.open,
                    session_nanos: session.duration_nanos(),
                    first: ts,
                    last: ts,
                });
                fit.covered_days +=
                    u32::from(finished.is_some_and(|s| s.covers_half(interval_nanos)));
            }
        }
    }
    fit.covered_days += u32::from(span.is_some_and(|s| s.covers_half(interval_nanos)));
    fit
}

/// Run the offset check on a sorted series of supported instants.
pub(super) fn check_offset(
    calendar: &Calendar,
    exchange: Exchange,
    ts_open: &[TsNanos],
    interval_nanos: i64,
) -> OffsetCheck {
    let as_given = fit(calendar, exchange, ts_open, interval_nanos, 0);
    let mut best: Option<ShiftEvidence> = None;
    for direction in [ShiftDirection::Late5h30, ShiftDirection::Early5h30] {
        let undone = fit(
            calendar,
            exchange,
            ts_open,
            interval_nanos,
            direction.undo_nanos(),
        );
        let fits_better = undone.misplaced < as_given.misplaced
            && best.map_or(true, |found| undone.misplaced < found.misplaced_if_undone);
        if fits_better {
            best = Some(ShiftEvidence {
                direction,
                misplaced: as_given.misplaced,
                misplaced_if_undone: undone.misplaced,
                covered_days_if_undone: undone.covered_days,
            });
        }
    }
    let Some(evidence) = best else {
        return OffsetCheck::Aligned;
    };
    let explained = evidence.misplaced - evidence.misplaced_if_undone;
    let decisive = explained.saturating_mul(10)
        >= evidence.misplaced.saturating_mul(DECISIVE_TENTHS)
        && evidence.covered_days_if_undone >= MIN_COVERED_DAYS;
    if decisive {
        OffsetCheck::Shifted(evidence)
    } else {
        OffsetCheck::Inconclusive(evidence)
    }
}
