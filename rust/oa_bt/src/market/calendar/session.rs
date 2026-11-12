//! The value types the calendar hands out: sessions, coverage levels and dated rules.

use crate::model::{Day, Exchange, TsNanos};

/// The exchange whose calendar `exchange` follows: `NSE_INDEX` follows NSE and
/// `BSE_INDEX` follows BSE, every other exchange has its own.
pub fn calendar_exchange(exchange: Exchange) -> Exchange {
    match exchange {
        Exchange::NseIndex => Exchange::Nse,
        Exchange::BseIndex => Exchange::Bse,
        other => other,
    }
}

/// Whether a session runs on the exchange's regular hours.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SessionKind {
    /// Regular hours on a weekday.
    Regular = 0,
    /// Any other session: Muhurat, a weekend session, an evening-only session on a
    /// holiday, or a day whose times were overridden.
    Special = 1,
}

/// How well a date is backed by holiday data. The names match the Python protocol
/// (`openalgo.backtest._protocols.COVERAGE_LEVELS`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Coverage {
    /// No holiday list was loaded for this exchange and year: weekday rule only.
    Template = 0,
    /// Data was loaded, from a secondary source or by assumption from another exchange.
    Unverified = 1,
    /// Data read from a cited exchange or regulator document, or supplied by the user
    /// as such.
    Sourced = 2,
}

impl Coverage {
    /// The level for a numeric code (0 template, 1 unverified, 2 sourced).
    pub fn from_u8(value: u8) -> Option<Coverage> {
        match value {
            0 => Some(Coverage::Template),
            1 => Some(Coverage::Unverified),
            2 => Some(Coverage::Sourced),
            _ => None,
        }
    }
}

/// One trading session of one exchange: the half-open interval `[open, close)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Session {
    /// IST session date.
    pub day: Day,
    /// Session open, UTC nanoseconds.
    pub open: TsNanos,
    /// Session close, UTC nanoseconds. The session does not include this instant.
    pub close: TsNanos,
    /// Regular hours, or a special session.
    pub kind: SessionKind,
    /// How well this date is backed by holiday data.
    pub coverage: Coverage,
    /// True when the session times are not verified: they come from an unverified
    /// row of the regular-hours table, or from a dated rule whose source could not
    /// confirm them.
    pub hours_unverified: bool,
}

impl Session {
    /// Length of the session in nanoseconds.
    pub fn duration_nanos(&self) -> i64 {
        self.close - self.open
    }

    /// Whether an instant falls inside the session.
    pub fn contains(&self, ts: TsNanos) -> bool {
        ts >= self.open && ts < self.close
    }
}

/// When the session of a dated rule closes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuleClose {
    /// At this time, in seconds since IST midnight.
    At(u32),
    /// At the exchange's regular close for that date, whatever the regular hours are
    /// when the session is asked for (an evening session on a holiday).
    Regular,
}

/// A dated rule for one exchange and date. It replaces the weekday rule and the
/// regular hours for that date.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DayRule {
    /// `None` when the exchange is closed, otherwise the open in seconds since IST
    /// midnight and the close.
    pub window: Option<(u32, RuleClose)>,
    /// True when the source could not confirm the times.
    pub hours_unverified: bool,
    /// How well the rule is backed: never `Coverage::Template`.
    pub coverage: Coverage,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_exchanges_follow_their_exchange() {
        assert_eq!(calendar_exchange(Exchange::NseIndex), Exchange::Nse);
        assert_eq!(calendar_exchange(Exchange::BseIndex), Exchange::Bse);
        assert_eq!(calendar_exchange(Exchange::Mcx), Exchange::Mcx);
    }

    #[test]
    fn coverage_levels_order_from_template_to_sourced() {
        assert!(Coverage::Template < Coverage::Unverified);
        assert!(Coverage::Unverified < Coverage::Sourced);
    }

    #[test]
    fn coverage_codes_round_trip() {
        for level in [Coverage::Template, Coverage::Unverified, Coverage::Sourced] {
            assert_eq!(Coverage::from_u8(level as u8), Some(level));
        }
        assert_eq!(Coverage::from_u8(3), None);
    }

    #[test]
    fn a_session_contains_its_open_but_not_its_close() {
        let session = Session {
            day: 0,
            open: 100,
            close: 200,
            kind: SessionKind::Regular,
            coverage: Coverage::Template,
            hours_unverified: false,
        };
        assert!(session.contains(100));
        assert!(session.contains(199));
        assert!(!session.contains(200));
        assert!(!session.contains(99));
        assert_eq!(session.duration_nanos(), 100);
    }
}
