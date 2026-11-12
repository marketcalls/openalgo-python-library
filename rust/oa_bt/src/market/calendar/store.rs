//! The calendar container: regular hours, dated rules and coverage marks.
//!
//! This module only stores and edits calendar data. Reading sessions out of it is in
//! `query`. Every method that takes a date answers with a typed error or `None` for a
//! date outside `MIN_DAY..=MAX_DAY`; none of them can panic.

use std::collections::BTreeMap;

use crate::model::{civil_from_days, is_supported_day, Day, Exchange, TsNanos};
use crate::model::{IST_OFFSET_NANOS, NANOS_PER_DAY, NANOS_PER_SEC};

use super::error::CalendarError;
use super::session::{calendar_exchange, Coverage, DayRule, RuleClose};
use super::tables::{builtin, row_on, HoursRow, IntradayRow, PerExchange, RegularHours};
use super::SECS_PER_DAY;

/// Session calendar for every OpenAlgo exchange.
///
/// `Calendar::new` holds the built-in regular hours and no holiday data, so every
/// date is `Coverage::Template` until data is loaded through the setters.
#[derive(Clone, Debug)]
pub struct Calendar {
    /// Regular hours per exchange, by date. Starts as the built-in table.
    hours: PerExchange<HoursRow>,
    /// Default intraday times per exchange, by date: the built-in table.
    pub(super) intraday: &'static PerExchange<IntradayRow>,
    pub(super) rules: BTreeMap<(Exchange, Day), DayRule>,
    year_coverage: BTreeMap<(Exchange, i32), Coverage>,
}

/// The calendar year of an IST date.
pub(super) fn year_of(day: Day) -> i32 {
    civil_from_days(day).0
}

/// The UTC instant of a supported IST date and a time in seconds since IST midnight.
pub(super) fn ts_at(day: Day, sec_of_day: u32) -> TsNanos {
    i64::from(day) * NANOS_PER_DAY + i64::from(sec_of_day) * NANOS_PER_SEC - IST_OFFSET_NANOS
}

pub(super) fn check_day(day: Day) -> Result<(), CalendarError> {
    if is_supported_day(day) {
        Ok(())
    } else {
        Err(CalendarError::UnsupportedDay(day))
    }
}

impl Calendar {
    /// A calendar with the built-in regular hours and no holiday data.
    ///
    /// Errors: `Table` when an embedded table is malformed.
    pub fn new() -> Result<Calendar, CalendarError> {
        let tables = builtin()?;
        Ok(Calendar {
            hours: tables.hours.clone(),
            intraday: &tables.intraday,
            rules: BTreeMap::new(),
            year_coverage: BTreeMap::new(),
        })
    }

    // ----- regular hours ---------------------------------------------------

    /// Set the regular hours of an exchange, in seconds since IST midnight.
    ///
    /// With `effective_from` the hours apply from that IST date onward; with `None`
    /// they replace every row of the exchange. Hours set here count as verified.
    ///
    /// Errors: `UnsupportedDay`; `InvalidSession` unless `open_sec < close_sec <=
    /// 86_400`, or when a dated rule of the exchange that closes at the regular close
    /// would then open at or after it.
    pub fn set_regular_hours(
        &mut self,
        exchange: Exchange,
        effective_from: Option<Day>,
        open_sec: u32,
        close_sec: u32,
    ) -> Result<(), CalendarError> {
        let owner = calendar_exchange(exchange);
        if open_sec >= close_sec || close_sec > SECS_PER_DAY {
            return Err(CalendarError::InvalidSession {
                exchange,
                day: None,
                open_sec,
                close_sec,
            });
        }
        let first_day = effective_from.unwrap_or(Day::MIN);
        if let Some(day) = effective_from {
            check_day(day)?;
        }
        let mut rows: Vec<HoursRow> = self.hours[owner as usize]
            .iter()
            .filter(|row| row.valid_from < first_day)
            .map(|row| HoursRow {
                valid_to: row.valid_to.min(first_day.saturating_sub(1)),
                ..*row
            })
            .collect();
        rows.push(HoursRow {
            valid_from: first_day,
            valid_to: Day::MAX,
            open_sec,
            close_sec,
            close_us_dst_sec: None,
            verified: true,
        });
        // A rule that closes at the regular close must still open before it.
        for (&(_, day), rule) in self.rules.range((owner, first_day)..=(owner, Day::MAX)) {
            if let Some((rule_open_sec, RuleClose::Regular)) = rule.window {
                if rule_open_sec >= close_sec {
                    let day = Some(day);
                    let open_sec = rule_open_sec;
                    return Err(CalendarError::InvalidSession {
                        exchange,
                        day,
                        open_sec,
                        close_sec,
                    });
                }
            }
        }
        self.hours[owner as usize] = rows;
        Ok(())
    }

    /// The regular hours of an exchange in force on a date, ignoring weekends,
    /// holidays and special sessions. `None` for a date outside the supported range.
    pub fn regular_hours(&self, exchange: Exchange, day: Day) -> Option<RegularHours> {
        if !is_supported_day(day) {
            return None;
        }
        let rows = &self.hours[calendar_exchange(exchange) as usize];
        row_on(rows, day).map(|row| row.resolve(day))
    }

    /// The origin of the bar lattice of an exchange on a date: IST midnight plus the
    /// regular open in force that date, as a UTC instant. Bars of interval N open at
    /// `anchor + k * N` on every date, special sessions included; this is the rule of
    /// the OpenAlgo server's aggregation. `None` for an unsupported date.
    pub fn lattice_anchor(&self, exchange: Exchange, day: Day) -> Option<TsNanos> {
        self.regular_hours(exchange, day)
            .map(|hours| ts_at(day, hours.open_sec))
    }

    // ----- dated rules -----------------------------------------------------

    /// Mark an exchange closed on a date (a trading holiday). A coverage of
    /// `Template` is stored as `Unverified`: an explicit rule is never a template.
    ///
    /// Errors: `UnsupportedDay`.
    pub fn set_closed(
        &mut self,
        exchange: Exchange,
        day: Day,
        coverage: Coverage,
    ) -> Result<(), CalendarError> {
        check_day(day)?;
        let rule = DayRule {
            window: None,
            hours_unverified: false,
            coverage: coverage.max(Coverage::Unverified),
        };
        self.rules.insert((calendar_exchange(exchange), day), rule);
        Ok(())
    }

    /// Give an exchange a session on a date: a special session (Muhurat, a weekend
    /// session, an evening session on a holiday) or an override of the regular times.
    /// `open_sec` is in seconds since IST midnight; `hours_unverified` marks times the
    /// source could not confirm.
    ///
    /// Errors: `UnsupportedDay`; `InvalidSession` unless the open is before the close
    /// and the close is at most 24:00.
    pub fn set_session(
        &mut self,
        exchange: Exchange,
        day: Day,
        open_sec: u32,
        close: RuleClose,
        hours_unverified: bool,
        coverage: Coverage,
    ) -> Result<(), CalendarError> {
        check_day(day)?;
        let close_sec = match close {
            RuleClose::At(sec) => sec,
            RuleClose::Regular => self.regular_hours(exchange, day).map_or(0, |h| h.close_sec),
        };
        if open_sec >= close_sec || close_sec > SECS_PER_DAY {
            let day = Some(day);
            return Err(CalendarError::InvalidSession {
                exchange,
                day,
                open_sec,
                close_sec,
            });
        }
        let rule = DayRule {
            window: Some((open_sec, close)),
            hours_unverified,
            coverage: coverage.max(Coverage::Unverified),
        };
        self.rules.insert((calendar_exchange(exchange), day), rule);
        Ok(())
    }

    /// Every dated rule of an exchange from `from` to `to` (inclusive), in date order.
    pub fn rules_in(&self, exchange: Exchange, from: Day, to: Day) -> Vec<(Day, DayRule)> {
        if from > to {
            return Vec::new();
        }
        let key = calendar_exchange(exchange);
        self.rules
            .range((key, from)..=(key, to))
            .map(|(&(_, day), &rule)| (day, rule))
            .collect()
    }

    // ----- coverage --------------------------------------------------------

    /// Record that the holiday list of an exchange for a year has been loaded.
    /// `Coverage::Template` removes the mark.
    pub fn set_year_coverage(&mut self, exchange: Exchange, year: i32, coverage: Coverage) {
        let key = (calendar_exchange(exchange), year);
        if coverage == Coverage::Template {
            self.year_coverage.remove(&key);
        } else {
            self.year_coverage.insert(key, coverage);
        }
    }

    /// Coverage of an exchange for a whole year.
    pub fn year_coverage(&self, exchange: Exchange, year: i32) -> Coverage {
        let key = (calendar_exchange(exchange), year);
        self.year_coverage
            .get(&key)
            .copied()
            .unwrap_or(Coverage::Template)
    }

    /// Coverage of one date: the dated rule's own level when there is one, otherwise
    /// the level of the year.
    pub fn coverage(&self, exchange: Exchange, day: Day) -> Coverage {
        match self.rules.get(&(calendar_exchange(exchange), day)) {
            Some(rule) => rule.coverage,
            None => self.year_coverage(exchange, year_of(day)),
        }
    }
}
