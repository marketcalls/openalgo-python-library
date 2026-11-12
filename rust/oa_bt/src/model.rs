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

    pub fn from_u8(value: u8) -> Option<Side> {
        match value {
            0 => Some(Side::Buy),
            1 => Some(Side::Sell),
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

impl InstrumentKind {
    /// The OpenAlgo instrument type string: "EQ", "INDEX", "FUT" or "OPT".
    pub fn code(self) -> &'static str {
        match self {
            InstrumentKind::Equity => "EQ",
            InstrumentKind::Index => "INDEX",
            InstrumentKind::Future => "FUT",
            InstrumentKind::Option => "OPT",
        }
    }

    /// Parse an instrument type string. "CE" and "PE" are accepted for options.
    pub fn from_code(code: &str) -> Option<InstrumentKind> {
        match code {
            "EQ" => Some(InstrumentKind::Equity),
            "INDEX" => Some(InstrumentKind::Index),
            "FUT" => Some(InstrumentKind::Future),
            "OPT" | "CE" | "PE" => Some(InstrumentKind::Option),
            _ => None,
        }
    }
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

/// First and last IST dates whose every instant fits in `TsNanos` (1677 to 2262).
/// Code that accepts dates from outside the kernel validates against this range.
pub const MIN_DAY: Day = -106_750;
pub const MAX_DAY: Day = 106_750;

/// True when every instant of `day` can be written as `TsNanos`.
pub fn is_supported_day(day: Day) -> bool {
    (MIN_DAY..=MAX_DAY).contains(&day)
}

// None of the functions below can panic. They are exact for supported days and
// saturate for inputs beyond them, so a bad date from outside becomes a typed error at
// the boundary that validates it, never a panic inside the kernel.

/// Days since 1970-01-01 for a proleptic Gregorian date.
pub fn days_from_civil(year: i32, month: u32, day: u32) -> Day {
    let month = i64::from(month);
    let y = i64::from(year) - i64::from(month <= 2);
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * mp + 2) / 5 + i64::from(day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    saturate_to_i32(era * 146_097 + doe - 719_468)
}

/// (year, month, day) for a day number.
pub fn civil_from_days(day: Day) -> (i32, u32, u32) {
    let z = i64::from(day) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    (saturate_to_i32(y), m as u32, d as u32)
}

/// A date written as the integer YYYYMMDD, for example 20260401.
pub fn day_from_ymd(ymd: i32) -> Day {
    days_from_civil(ymd / 10_000, ((ymd / 100) % 100) as u32, (ymd % 100) as u32)
}

/// The integer YYYYMMDD for a day number.
pub fn ymd_from_day(day: Day) -> i32 {
    let (y, m, d) = civil_from_days(day);
    saturate_to_i32(i64::from(y) * 10_000 + i64::from(m) * 100 + i64::from(d))
}

/// Day of week, Monday = 0 through Sunday = 6.
pub fn weekday(day: Day) -> u32 {
    // 1970-01-01 was a Thursday.
    (i64::from(day) + 3).rem_euclid(7) as u32
}

/// The IST calendar date of a UTC instant.
pub fn ist_day(ts: TsNanos) -> Day {
    saturate_to_i32(
        ts.saturating_add(IST_OFFSET_NANOS)
            .div_euclid(NANOS_PER_DAY),
    )
}

/// Nanoseconds since IST midnight for a UTC instant.
pub fn ist_nanos_of_day(ts: TsNanos) -> i64 {
    ts.saturating_add(IST_OFFSET_NANOS)
        .rem_euclid(NANOS_PER_DAY)
}

/// Minutes since IST midnight for a UTC instant (09:15 IST is 555).
pub fn ist_minute_of_day(ts: TsNanos) -> u32 {
    (ist_nanos_of_day(ts) / NANOS_PER_MIN) as u32
}

/// The UTC instant of an IST date and minute of day.
pub fn ts_from_ist(day: Day, minute_of_day: u32) -> TsNanos {
    i64::from(day)
        .saturating_mul(NANOS_PER_DAY)
        .saturating_add(i64::from(minute_of_day).saturating_mul(NANOS_PER_MIN))
        .saturating_sub(IST_OFFSET_NANOS)
}

fn saturate_to_i32(value: i64) -> i32 {
    value.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
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
    fn extreme_inputs_saturate_instead_of_panicking() {
        for day in [
            i32::MIN,
            MIN_DAY - 1,
            MIN_DAY,
            0,
            MAX_DAY,
            MAX_DAY + 1,
            i32::MAX,
        ] {
            let (y, m, d) = civil_from_days(day);
            assert!(
                (1..=12).contains(&m) && (1..=31).contains(&d),
                "{day}: {y}-{m}-{d}"
            );
            let _ = (ymd_from_day(day), weekday(day), ts_from_ist(day, u32::MAX));
        }
        for ts in [i64::MIN, -1, 0, i64::MAX] {
            let _ = (ist_day(ts), ist_minute_of_day(ts));
        }
        let _ = (
            days_from_civil(i32::MAX, u32::MAX, u32::MAX),
            day_from_ymd(i32::MIN),
        );
        assert!(is_supported_day(MIN_DAY) && is_supported_day(MAX_DAY));
        assert!(!is_supported_day(MAX_DAY + 1) && !is_supported_day(MIN_DAY - 1));
        assert_eq!(ist_day(ts_from_ist(MAX_DAY, 1_439)), MAX_DAY);
        assert_eq!(ist_day(ts_from_ist(MIN_DAY, 0)), MIN_DAY);
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
        assert_eq!(Side::from_u8(1), Some(Side::Sell));
        assert_eq!(Side::from_u8(2), None);
        assert_eq!(
            InstrumentKind::from_code("PE"),
            Some(InstrumentKind::Option)
        );
        assert_eq!(
            InstrumentKind::from_code("FUT").map(InstrumentKind::code),
            Some("FUT")
        );
        assert_eq!(InstrumentKind::from_code("eq"), None);
        assert_eq!(Side::Buy.sign(), 1.0);
    }
}
