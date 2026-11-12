//! Reading sessions out of a calendar: one date, a range, and the annualisation factor.

use crate::model::{weekday, Day, Exchange, TsNanos, MAX_DAY, MIN_DAY};

use super::error::CalendarError;
use super::session::{calendar_exchange, Coverage, RuleClose, Session, SessionKind};
use super::store::{check_day, ts_at, year_of, Calendar};

/// First weekday number (Monday = 0) of the weekend.
const SATURDAY: u32 = 5;
/// Average length of a calendar year in days, for annualisation.
const DAYS_PER_YEAR: f64 = 365.25;
/// Shortest window `sessions_per_year` accepts, in calendar days: four whole weeks,
/// so that every weekday and both weekend days are counted four times. A shorter
/// window gives a figure that depends on where it starts (Monday to Friday alone
/// would annualise to 365 sessions).
pub const MIN_ANNUALISATION_DAYS: i64 = 28;

/// The annualisation factor of a date range, with what it rests on.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionsPerYear {
    /// Sessions per year: the session count scaled to 365.25 calendar days.
    pub value: f64,
    /// Dates in the range on which at least one of the exchanges has a session.
    pub sessions: u32,
    /// Calendar days in the range, both ends included.
    pub calendar_days: i64,
    /// The weakest year coverage among the exchanges and years of the range.
    pub coverage: Coverage,
    /// The exchange-years of the range that have no holiday list. For each of them
    /// the count includes that year's holidays, so `value` is overstated.
    pub template_only: Vec<(Exchange, i32)>,
}

pub(super) fn check_range(from: Day, to: Day) -> Result<(), CalendarError> {
    check_day(from)?;
    check_day(to)?;
    if from > to {
        Err(CalendarError::InvalidRange { from, to })
    } else {
        Ok(())
    }
}

impl Calendar {
    /// The session of an exchange on an IST date, or `None` when it is closed or the
    /// date is outside the supported range.
    ///
    /// A dated rule decides first, then the weekend, then the exchange's regular hours.
    pub fn session(&self, exchange: Exchange, day: Day) -> Option<Session> {
        let regular = self.regular_hours(exchange, day)?;
        let is_weekday = weekday(day) < SATURDAY;
        let Some(rule) = self.rules.get(&(calendar_exchange(exchange), day)) else {
            return is_weekday.then(|| Session {
                day,
                open: ts_at(day, regular.open_sec),
                close: ts_at(day, regular.close_sec),
                kind: SessionKind::Regular,
                coverage: self.year_coverage(exchange, year_of(day)),
                hours_unverified: regular.unverified,
            });
        };
        let (open_sec, close) = rule.window?;
        let (close_sec, close_unverified) = match close {
            RuleClose::At(sec) => (sec, false),
            RuleClose::Regular => (regular.close_sec, regular.unverified),
        };
        if close_sec <= open_sec {
            // Only reachable when regular hours were replaced under a rule that
            // closes at the regular close; `set_regular_hours` refuses that.
            return None;
        }
        let on_regular_hours =
            is_weekday && (open_sec, close_sec) == (regular.open_sec, regular.close_sec);
        Some(Session {
            day,
            open: ts_at(day, open_sec),
            close: ts_at(day, close_sec),
            kind: if on_regular_hours {
                SessionKind::Regular
            } else {
                SessionKind::Special
            },
            coverage: rule.coverage,
            hours_unverified: rule.hours_unverified || close_unverified,
        })
    }

    /// Whether the exchange has a session on the IST date.
    pub fn is_open(&self, exchange: Exchange, day: Day) -> bool {
        self.session(exchange, day).is_some()
    }

    /// Every session of an exchange from `from` to `to`, both inclusive.
    ///
    /// Errors: `UnsupportedDay`; `InvalidRange` when `from` is after `to`.
    pub fn sessions(
        &self,
        exchange: Exchange,
        from: Day,
        to: Day,
    ) -> Result<Vec<Session>, CalendarError> {
        check_range(from, to)?;
        Ok((from..=to)
            .filter_map(|day| self.session(exchange, day))
            .collect())
    }

    /// Average sessions per year from `from` to `to` (both inclusive) on the union
    /// calendar of `exchanges`: the number of dates on which any of them trades,
    /// scaled to 365.25 calendar days. This is the annualisation factor for
    /// volatility-based ratios on daily returns. The result names the exchange-years
    /// without a holiday list, for which the figure is overstated.
    ///
    /// Errors: `NoExchanges`; `UnsupportedDay`; `InvalidRange`; `WindowTooShort` for
    /// fewer than `MIN_ANNUALISATION_DAYS` calendar days.
    pub fn sessions_per_year(
        &self,
        exchanges: &[Exchange],
        from: Day,
        to: Day,
    ) -> Result<SessionsPerYear, CalendarError> {
        if exchanges.is_empty() {
            return Err(CalendarError::NoExchanges);
        }
        check_range(from, to)?;
        let calendar_days = i64::from(to) - i64::from(from) + 1;
        if calendar_days < MIN_ANNUALISATION_DAYS {
            let minimum = MIN_ANNUALISATION_DAYS;
            return Err(CalendarError::WindowTooShort {
                days: calendar_days,
                minimum,
            });
        }
        let trades = |day: &Day| {
            exchanges
                .iter()
                .any(|&exchange| self.is_open(exchange, *day))
        };
        let sessions = (from..=to).filter(trades).count() as u32;

        let mut coverage = Coverage::Sourced;
        let mut template_only = Vec::new();
        for &exchange in exchanges {
            for year in year_of(from)..=year_of(to) {
                let level = self.year_coverage(exchange, year);
                coverage = coverage.min(level);
                if level == Coverage::Template {
                    template_only.push((exchange, year));
                }
            }
        }
        let value = f64::from(sessions) * DAYS_PER_YEAR / calendar_days as f64;
        Ok(SessionsPerYear {
            value,
            sessions,
            calendar_days,
            coverage,
            template_only,
        })
    }

    /// Open of the first session and close of the last session of an exchange from
    /// `from` to `to`, or `None` when it has no session there. Weekly and longer bars
    /// take their bounds from here.
    pub fn period_bounds(
        &self,
        exchange: Exchange,
        from: Day,
        to: Day,
    ) -> Option<(TsNanos, TsNanos)> {
        let (from, to) = (from.max(MIN_DAY), to.min(MAX_DAY));
        if from > to {
            return None;
        }
        let first = (from..=to).find_map(|day| self.session(exchange, day))?;
        let last = (from..=to)
            .rev()
            .find_map(|day| self.session(exchange, day))?;
        Some((first.open, last.close))
    }
}
