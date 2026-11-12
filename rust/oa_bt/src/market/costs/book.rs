//! A set of schedule versions, the lookup by trade date, and how far the rates a
//! lookup returns can be trusted.

use std::borrow::Cow;

use crate::model::{is_supported_day, Day};

use super::builtin::builtin_schedules;
use super::error::CostError;
use super::schedule::CostSchedule;

/// How far the rates used for a trade can be trusted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Trust {
    /// The rates were verified for the trade date.
    Verified = 0,
    /// The rates were never verified.
    Unverified = 1,
    /// The trade date is after the date the rates were verified through.
    PastVerification = 2,
    /// The trade date is before the first version, whose rates were used because
    /// backfill is on.
    Backfilled = 3,
}

impl Trust {
    /// Every trust level, in id order.
    pub const ALL: [Trust; 4] = [
        Trust::Verified,
        Trust::Unverified,
        Trust::PastVerification,
        Trust::Backfilled,
    ];

    /// A short code for results: verified, unverified, past_verification or
    /// backfilled.
    pub fn code(self) -> &'static str {
        match self {
            Trust::Verified => "verified",
            Trust::Unverified => "unverified",
            Trust::PastVerification => "past_verification",
            Trust::Backfilled => "backfilled",
        }
    }
}

/// Which schedule version priced a trade and how far its rates can be trusted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RateStatus {
    /// Position of the version in `ScheduleBook::versions`.
    pub schedule: usize,
    /// How far the rates can be trusted.
    pub trust: Trust,
}

impl RateStatus {
    pub(super) fn new(
        schedule: usize,
        verified_through: Option<Day>,
        day: Day,
        backfilled: bool,
    ) -> RateStatus {
        let trust = match verified_through {
            _ if backfilled => Trust::Backfilled,
            None => Trust::Unverified,
            Some(through) if day > through => Trust::PastVerification,
            Some(_) => Trust::Verified,
        };
        RateStatus { schedule, trust }
    }
}

/// The schedule versions a cost model looks up by trade date.
#[derive(Clone, Debug, PartialEq)]
pub struct ScheduleBook {
    /// Sorted by `valid_from`; ranges never overlap.
    versions: Cow<'static, [CostSchedule]>,
    backfill: bool,
}

impl ScheduleBook {
    /// The versions compiled into the library. Errors: `InvalidScheduleData` or
    /// `InvalidSchedule` if the embedded data file is malformed (a unit test guards
    /// against shipping one).
    pub fn builtin() -> Result<ScheduleBook, CostError> {
        Ok(ScheduleBook {
            versions: Cow::Borrowed(builtin_schedules()?),
            backfill: false,
        })
    }

    /// A book made only of the given versions.
    ///
    /// Versions are sorted by start date. A version with no end date that is
    /// followed by a later version ends the day before that version starts (the
    /// usual meaning of an effective date).
    ///
    /// Errors: `EmptyScheduleBook` for no versions, `InvalidSchedule` when a version
    /// fails `CostSchedule::validate` or two versions share an id, and
    /// `ScheduleOverlap` when two versions cover the same date.
    pub fn from_versions(versions: Vec<CostSchedule>) -> Result<ScheduleBook, CostError> {
        Ok(ScheduleBook {
            versions: Cow::Owned(ordered(versions)?),
            backfill: false,
        })
    }

    /// The built-in versions plus user-supplied ones, merged by the rules of
    /// `from_versions` (same errors, plus those of `builtin`). Use it to add an
    /// earlier period, or a new version that takes over from the current one on its
    /// start date.
    pub fn builtin_with(extra: Vec<CostSchedule>) -> Result<ScheduleBook, CostError> {
        let mut versions = builtin_schedules()?.to_vec();
        versions.extend(extra);
        ScheduleBook::from_versions(versions)
    }

    /// Opt in (or out) of charging trades dated before the first version at the
    /// first version's rates. Off by default: those trades are an error, because
    /// the first version's rates are not the rates that applied then.
    pub fn with_backfill(mut self, backfill: bool) -> ScheduleBook {
        self.backfill = backfill;
        self
    }

    /// Whether trades before the first version are charged at its rates.
    pub fn backfill(&self) -> bool {
        self.backfill
    }

    /// The versions, oldest first.
    pub fn versions(&self) -> &[CostSchedule] {
        &self.versions
    }

    /// Position (in `versions`) of the version that prices the IST trade date `day`,
    /// and whether it does so only because backfill is on.
    ///
    /// Errors: `UnsupportedDay` for a date outside the supported range;
    /// `NoScheduleForDay` when `day` is before the first version and backfill is
    /// off; `ScheduleGap` when it is after the last version ended or between two
    /// versions.
    #[inline]
    pub fn lookup(&self, day: Day) -> Result<(usize, bool), CostError> {
        if !is_supported_day(day) {
            return Err(CostError::UnsupportedDay { day });
        }
        let versions: &[CostSchedule] = &self.versions;
        // Newest first: backtests spend most of their time in recent versions.
        for (index, version) in versions.iter().enumerate().rev() {
            if day >= version.valid_from {
                return match version.valid_to {
                    Some(end) if day > end => Err(CostError::ScheduleGap { day }),
                    _ => Ok((index, false)),
                };
            }
        }
        match versions.first() {
            Some(_) if self.backfill => Ok((0, true)),
            Some(first) => Err(CostError::NoScheduleForDay {
                day,
                first_valid_from: first.valid_from,
            }),
            None => Err(CostError::EmptyScheduleBook),
        }
    }
}

/// Validate, sort and close a set of versions: the invariant of a `ScheduleBook`.
pub(super) fn ordered(mut versions: Vec<CostSchedule>) -> Result<Vec<CostSchedule>, CostError> {
    if versions.is_empty() {
        return Err(CostError::EmptyScheduleBook);
    }
    for version in &versions {
        version.validate()?;
    }
    versions.sort_by_key(|version| version.valid_from);
    reject_duplicate_ids(&versions)?;
    close_open_ranges(&mut versions)?;
    Ok(versions)
}

fn reject_duplicate_ids(versions: &[CostSchedule]) -> Result<(), CostError> {
    for (position, version) in versions.iter().enumerate() {
        if versions[..position]
            .iter()
            .any(|earlier| earlier.id == version.id)
        {
            return Err(CostError::InvalidSchedule(format!(
                "two schedule versions share the id {:?}",
                version.id
            )));
        }
    }
    Ok(())
}

/// End every open-ended version the day before its successor starts, and reject
/// versions (sorted by start date) that still overlap.
fn close_open_ranges(versions: &mut [CostSchedule]) -> Result<(), CostError> {
    for next in 1..versions.len() {
        let next_from = versions[next].valid_from;
        let previous = &versions[next - 1];
        let overlaps = previous.valid_from == next_from
            || matches!(previous.valid_to, Some(end) if end >= next_from);
        if overlaps {
            return Err(CostError::ScheduleOverlap {
                first: previous.id.clone(),
                second: versions[next].id.clone(),
            });
        }
        if previous.valid_to.is_none() {
            versions[next - 1].valid_to = Some(next_from - 1);
        }
    }
    Ok(())
}
