//! Shared domain types and time helpers.
//!
//! Every other module depends on this file, so it stays small and stable. Types that
//! belong to one area (cost segments, session templates, bar layouts) live in that
//! area's module, not here.

/// UTC nanoseconds since the Unix epoch.
pub type TsNanos = i64;

/// An IST calendar date as days since 1970-01-01 (1970-01-01 is day 0).
pub type Day = i32;

pub const NANOS_PER_SEC: i64 = 1_000_000_000;
pub const NANOS_PER_MIN: i64 = 60 * NANOS_PER_SEC;
pub const NANOS_PER_HOUR: i64 = 60 * NANOS_PER_MIN;
pub const NANOS_PER_DAY: i64 = 24 * NANOS_PER_HOUR;

/// India Standard Time is UTC+05:30 all year.
pub const IST_OFFSET_NANOS: i64 = 5 * NANOS_PER_HOUR + 30 * NANOS_PER_MIN;

/// OpenAlgo exchange codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Exchange {
    Nse = 0,
    Bse = 1,
    Nfo = 2,
    Bfo = 3,
    Mcx = 4,
    Cds = 5,
    Bcd = 6,
    NseIndex = 7,
    BseIndex = 8,
}

impl Exchange {
    pub const ALL: [Exchange; 9] = [
        Exchange::Nse,
        Exchange::Bse,
        Exchange::Nfo,
        Exchange::Bfo,
        Exchange::Mcx,
        Exchange::Cds,
        Exchange::Bcd,
        Exchange::NseIndex,
        Exchange::BseIndex,
    ];

    /// The OpenAlgo exchange string, for example "NSE" or "NSE_INDEX".
    pub fn code(self) -> &'static str {
        match self {
            Exchange::Nse => "NSE",
            Exchange::Bse => "BSE",
            Exchange::Nfo => "NFO",
            Exchange::Bfo => "BFO",
            Exchange::Mcx => "MCX",
            Exchange::Cds => "CDS",
            Exchange::Bcd => "BCD",
            Exchange::NseIndex => "NSE_INDEX",
            Exchange::BseIndex => "BSE_INDEX",
        }
    }

    /// Parse an OpenAlgo exchange string (exact, upper case).
    pub fn from_code(code: &str) -> Option<Exchange> {
        Exchange::ALL.iter().copied().find(|e| e.code() == code)
    }

    pub fn from_u8(value: u8) -> Option<Exchange> {
        Exchange::ALL.iter().copied().find(|e| *e as u8 == value)
    }
}

/// OpenAlgo product types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Product {
    /// Equity delivery.
    Cnc = 0,
    /// Intraday.
    Mis = 1,
    /// F&O carry-forward.
    Nrml = 2,
}

impl Product {
    pub fn code(self) -> &'static str {
        match self {
            Product::Cnc => "CNC",
            Product::Mis => "MIS",
            Product::Nrml => "NRML",
        }
    }

    pub fn from_code(code: &str) -> Option<Product> {
        match code {
            "CNC" => Some(Product::Cnc),
            "MIS" => Some(Product::Mis),
            "NRML" => Some(Product::Nrml),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Side {
    Buy = 0,
    Sell = 1,
}

impl Side {
    pub fn code(self) -> &'static str {
        match self {
            Side::Buy => "BUY",
            Side::Sell => "SELL",
        }
    }

    pub fn from_code(code: &str) -> Option<Side> {
        match code {
            "BUY" => Some(Side::Buy),
            "SELL" => Some(Side::Sell),
            _ => None,
        }
    }

    /// +1.0 for a buy, -1.0 for a sell.
    pub fn sign(self) -> f64 {
        match self {
            Side::Buy => 1.0,
            Side::Sell => -1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum InstrumentKind {
    Equity = 0,
    Index = 1,
    Future = 2,
    Option = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum OptionType {
    /// CE
    Call = 0,
    /// PE
    Put = 1,
}

// ---------------------------------------------------------------------------
// Dates and IST time
// ---------------------------------------------------------------------------

/// Days since 1970-01-01 for a proleptic Gregorian date.
pub fn days_from_civil(year: i32, month: u32, day: u32) -> Day {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u32;
    let mp = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe as i32 - 719_468
}

/// (year, month, day) for a day number.
pub fn civil_from_days(day: Day) -> (i32, u32, u32) {
    let z = day + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe as i32 + era * 400 + if m <= 2 { 1 } else { 0 };
    (y, m, d)
}

/// A date written as the integer YYYYMMDD, for example 20260401.
pub fn day_from_ymd(ymd: i32) -> Day {
    days_from_civil(ymd / 10_000, ((ymd / 100) % 100) as u32, (ymd % 100) as u32)
}

/// The integer YYYYMMDD for a day number.
pub fn ymd_from_day(day: Day) -> i32 {
    let (y, m, d) = civil_from_days(day);
    y * 10_000 + (m as i32) * 100 + d as i32
}

/// Day of week, Monday = 0 through Sunday = 6.
pub fn weekday(day: Day) -> u32 {
    // 1970-01-01 was a Thursday.
    (day as i64 + 3).rem_euclid(7) as u32
}

/// The IST calendar date of a UTC instant.
pub fn ist_day(ts: TsNanos) -> Day {
    (ts + IST_OFFSET_NANOS).div_euclid(NANOS_PER_DAY) as Day
}

/// Nanoseconds since IST midnight for a UTC instant.
pub fn ist_nanos_of_day(ts: TsNanos) -> i64 {
    (ts + IST_OFFSET_NANOS).rem_euclid(NANOS_PER_DAY)
}

/// Minutes since IST midnight for a UTC instant (09:15 IST is 555).
pub fn ist_minute_of_day(ts: TsNanos) -> u32 {
    (ist_nanos_of_day(ts) / NANOS_PER_MIN) as u32
}

/// The UTC instant of an IST date and minute of day.
pub fn ts_from_ist(day: Day, minute_of_day: u32) -> TsNanos {
    day as i64 * NANOS_PER_DAY + minute_of_day as i64 * NANOS_PER_MIN - IST_OFFSET_NANOS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_round_trip() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(2000, 3, 1), 11_017);
        assert_eq!(days_from_civil(2026, 11, 12), 20_769);
        for day in -1_000..40_000 {
            let (y, m, d) = civil_from_days(day);
            assert_eq!(days_from_civil(y, m, d), day);
        }
        assert_eq!(civil_from_days(19_782), (2024, 2, 29));
    }

    #[test]
    fn ymd_round_trip() {
        assert_eq!(day_from_ymd(20260401), days_from_civil(2026, 4, 1));
        assert_eq!(ymd_from_day(days_from_civil(2019, 1, 1)), 20190101);
    }

    #[test]
    fn weekdays() {
        assert_eq!(weekday(0), 3); // Thursday
        assert_eq!(weekday(days_from_civil(2026, 11, 12)), 3); // Thursday
        assert_eq!(weekday(days_from_civil(2026, 11, 9)), 0); // Monday
        assert_eq!(weekday(days_from_civil(1969, 12, 28)), 6); // Sunday
    }

    #[test]
    fn ist_conversions() {
        // 2026-11-12 09:15 IST is 03:45 UTC.
        let day = days_from_civil(2026, 11, 12);
        let ts = ts_from_ist(day, 9 * 60 + 15);
        assert_eq!(
            ts,
            day as i64 * NANOS_PER_DAY + 3 * NANOS_PER_HOUR + 45 * NANOS_PER_MIN
        );
        assert_eq!(ist_day(ts), day);
        assert_eq!(ist_minute_of_day(ts), 555);
        // 20:00 UTC is 01:30 IST on the next date.
        let late = day as i64 * NANOS_PER_DAY + 20 * NANOS_PER_HOUR;
        assert_eq!(ist_day(late), day + 1);
        assert_eq!(ist_minute_of_day(late), 90);
        // Before the epoch.
        assert_eq!(ist_day(-1), 0);
        assert_eq!(ist_day(-IST_OFFSET_NANOS - 1), -1);
    }

    #[test]
    fn codes() {
        for e in Exchange::ALL {
            assert_eq!(Exchange::from_code(e.code()), Some(e));
            assert_eq!(Exchange::from_u8(e as u8), Some(e));
        }
        assert_eq!(Exchange::from_code("nse"), None);
        assert_eq!(Product::from_code("MIS"), Some(Product::Mis));
        assert_eq!(Side::from_code("SELL"), Some(Side::Sell));
        assert_eq!(Side::Buy.sign(), 1.0);
    }
}
