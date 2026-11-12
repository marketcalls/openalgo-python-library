//! The error type of the cost engine.
//!
//! Nothing in this module falls back silently: an unknown segment, preset, scrip
//! group, commodity or trade date is an error whose message says what to do. The
//! variants the per-fill path can return carry no heap data, so an error on that
//! path does not allocate either.

use std::fmt;

use crate::model::{civil_from_days, Day, Exchange, InstrumentKind, Product, MAX_DAY, MIN_DAY};

use super::resolve::bse_group_names;
use super::segment::Segment;

/// The kind of name that failed to resolve in `CostError::UnknownName`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NameKind {
    /// A cost segment code such as "NSE_EQ_DELIVERY".
    Segment,
    /// An OpenAlgo exchange code such as "NSE".
    Exchange,
    /// An OpenAlgo product code: CNC, MIS or NRML.
    Product,
    /// An instrument type: EQ, FUT, OPT, CE or PE.
    Instrument,
    /// A BSE scrip group such as "A" or "XT".
    BseGroup,
    /// An MCX commodity such as "CRUDEOIL".
    Commodity,
    /// A brokerage preset name.
    Preset,
    /// A rounding mode name.
    Rounding,
}

impl NameKind {
    fn label(self) -> &'static str {
        match self {
            NameKind::Segment => "cost segment",
            NameKind::Exchange => "exchange",
            NameKind::Product => "product",
            NameKind::Instrument => "instrument type",
            NameKind::BseGroup => "BSE scrip group",
            NameKind::Commodity => "MCX commodity",
            NameKind::Preset => "brokerage preset",
            NameKind::Rounding => "rounding mode",
        }
    }

    /// What the caller should use instead.
    fn advice(self) -> &'static str {
        match self {
            NameKind::Segment => {
                "use a code from the segment table (Segment::ALL), for example \
                 NSE_EQ_DELIVERY, NFO_OPT or MCX_FUT"
            }
            NameKind::Exchange => "use NSE, BSE, NFO, BFO or MCX (upper case)",
            NameKind::Product => "use CNC, MIS or NRML",
            NameKind::Instrument => "use EQ, FUT or OPT (CE and PE are accepted for options)",
            NameKind::BseGroup => {
                "the groups with a known charge are listed in the BSE group table"
            }
            NameKind::Commodity => {
                "it is in neither the agri nor the non-agri list, so pass the segment \
                 explicitly: MCX_FUT for a non-agri commodity, MCX_FUT_AGRI for an agri one"
            }
            NameKind::Preset => {
                "built-in presets are zerodha and zero_brokerage; build others from a flat \
                 amount, a percentage with a cap, or custom rules"
            }
            NameKind::Rounding => "use contract_note or calculator",
        }
    }
}

/// Everything the cost engine can refuse to do. Each message says what was wrong and
/// what to do about it.
#[derive(Clone, Debug, PartialEq)]
pub enum CostError {
    /// A name that is not in its table: see `NameKind`.
    UnknownName {
        /// What kind of name it was.
        kind: NameKind,
        /// The name as given.
        name: String,
    },
    /// A real market segment that this engine does not cost yet (currency
    /// derivatives and MCX options).
    UnsupportedSegment(String),
    /// A numeric segment id outside the segment table.
    UnknownSegmentId(u8),
    /// A numeric side that is neither 0 (buy) nor 1 (sell).
    UnknownSide(u8),
    /// A valid exchange, instrument kind and product that do not go together, or
    /// that need a detail (the underlying) that was not given.
    InvalidCombination {
        /// The exchange asked for.
        exchange: Exchange,
        /// The instrument kind asked for.
        kind: InstrumentKind,
        /// The product asked for.
        product: Product,
        /// What would be valid instead.
        hint: &'static str,
    },
    /// A BSE equity trade with no scrip group: the exchange charge depends on it.
    MissingBseGroup,
    /// An index exchange: indices are not tradable, so they have no costs.
    NotTradable(Exchange),
    /// Exercise STT was requested for a segment that is not an options segment.
    NotAnOption(Segment),
    /// A brokerage rule or fee that is negative, not finite or implausible.
    InvalidBrokerage(String),
    /// A price, quantity or amount that is negative, infinite or NaN.
    InvalidInput {
        /// Which argument was bad.
        field: &'static str,
        /// The value given.
        value: f64,
    },
    /// A trade date outside the range the engine supports.
    UnsupportedDay {
        /// The day number given (days since 1970-01-01).
        day: Day,
    },
    /// Batch inputs of different lengths.
    LengthMismatch {
        /// Which input has the wrong length.
        name: &'static str,
        /// The length of the first input.
        expected: usize,
        /// The length found.
        got: usize,
    },
    /// The trade date is earlier than the first schedule version.
    NoScheduleForDay {
        /// The IST trade date asked for.
        day: Day,
        /// The first date any version covers.
        first_valid_from: Day,
    },
    /// The trade date falls after the last version ended or between two versions.
    ScheduleGap {
        /// The IST trade date asked for.
        day: Day,
    },
    /// The schedule in force on the trade date has no row for the segment.
    SegmentNotInSchedule {
        /// The segment with no rate row.
        segment: Segment,
        /// The IST trade date asked for.
        day: Day,
    },
    /// A schedule version that fails validation.
    InvalidSchedule(String),
    /// The schedule data file cannot be read.
    InvalidScheduleData {
        /// Line number in the file, starting at 1.
        line: usize,
        /// What is wrong with it.
        reason: String,
    },
    /// Two schedule versions cover the same date.
    ScheduleOverlap {
        /// Id of the earlier version.
        first: String,
        /// Id of the later version.
        second: String,
    },
    /// A schedule book with no versions.
    EmptyScheduleBook,
}

impl CostError {
    /// Shorthand for `CostError::UnknownName`.
    pub(super) fn unknown(kind: NameKind, name: &str) -> CostError {
        CostError::UnknownName {
            kind,
            name: name.to_string(),
        }
    }
}

/// An ISO date string for a day number, used only in messages.
pub(super) fn iso_date(day: Day) -> String {
    let (year, month, dom) = civil_from_days(day);
    format!("{year:04}-{month:02}-{dom:02}")
}

type Out<'a, 'b> = &'a mut fmt::Formatter<'b>;

fn unknown_name(f: Out, kind: NameKind, name: &str) -> fmt::Result {
    write!(f, "unknown {} {name:?}: {}", kind.label(), kind.advice())?;
    if kind == NameKind::BseGroup {
        write!(f, " ({})", bse_group_names())?;
    }
    Ok(())
}

fn unsupported_segment(f: Out, what: &str) -> fmt::Result {
    write!(
        f,
        "{what} is not costed yet: currency derivatives (CDS, BCD) and MCX options are out of \
         scope for this release, so leave these instruments out or cost them yourself"
    )
}

fn invalid_combination(
    f: Out,
    exchange: Exchange,
    kind: InstrumentKind,
    product: Product,
    hint: &str,
) -> fmt::Result {
    write!(
        f,
        "no cost segment for exchange {}, instrument {}, product {}: {hint}",
        exchange.code(),
        kind.code(),
        product.code()
    )
}

fn missing_bse_group(f: Out) -> fmt::Result {
    write!(
        f,
        "a BSE equity trade needs its scrip group, because the exchange charge runs from Rs 275 \
         to Rs 1,00,000 per crore depending on it: pass the group from the instrument master \
         (one of {})",
        bse_group_names()
    )
}

fn unsupported_day(f: Out, day: Day) -> fmt::Result {
    write!(
        f,
        "trade date with day number {day} is outside the supported range {} to {}",
        iso_date(MIN_DAY),
        iso_date(MAX_DAY)
    )
}

fn no_schedule_for_day(f: Out, day: Day, first: Day) -> fmt::Result {
    write!(
        f,
        "no cost schedule covers {}: the first schedule version starts on {}; supply a schedule \
         for the earlier period, or opt in to backfill to charge earlier trades at the first \
         version's rates",
        iso_date(day),
        iso_date(first)
    )
}

fn schedule_gap(f: Out, day: Day) -> fmt::Result {
    write!(
        f,
        "no cost schedule covers {}: the date falls after the last schedule version ended or \
         between two versions; supply a schedule version for it",
        iso_date(day)
    )
}

fn segment_not_in_schedule(f: Out, segment: Segment, day: Day) -> fmt::Result {
    write!(
        f,
        "the cost schedule in force on {} has no rates for segment {}: add a row for it to that \
         schedule version",
        iso_date(day),
        segment.code()
    )
}

impl fmt::Display for CostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use CostError::*;
        match self {
            UnknownName { kind, name } => unknown_name(f, *kind, name),
            UnsupportedSegment(what) => unsupported_segment(f, what),
            UnknownSegmentId(id) => write!(f, "unknown cost segment id {id}: use Segment::ALL"),
            UnknownSide(side) => write!(f, "unknown side id {side}: use 0 for BUY and 1 for SELL"),
            InvalidCombination {
                exchange,
                kind,
                product,
                hint,
            } => invalid_combination(f, *exchange, *kind, *product, hint),
            MissingBseGroup => missing_bse_group(f),
            NotTradable(exchange) => write!(
                f,
                "{} is an index exchange with no trading costs: cost the future or option on \
                 NFO or BFO instead",
                exchange.code()
            ),
            NotAnOption(segment) => write!(
                f,
                "exercise STT is for options only, got segment {}: use an options segment",
                segment.code()
            ),
            InvalidBrokerage(why) => write!(f, "invalid brokerage: {why}"),
            InvalidInput { field, value } => {
                write!(
                    f,
                    "{field} must be a finite number that is zero or more, got {value}"
                )
            }
            UnsupportedDay { day } => unsupported_day(f, *day),
            LengthMismatch {
                name,
                expected,
                got,
            } => write!(
                f,
                "batch input {name} has {got} elements, expected {expected}: all inputs must \
                 have the same length"
            ),
            NoScheduleForDay {
                day,
                first_valid_from,
            } => no_schedule_for_day(f, *day, *first_valid_from),
            ScheduleGap { day } => schedule_gap(f, *day),
            SegmentNotInSchedule { segment, day } => segment_not_in_schedule(f, *segment, *day),
            InvalidSchedule(why) => write!(f, "invalid cost schedule: {why}"),
            InvalidScheduleData { line, reason } => {
                write!(f, "cost schedule data, line {line}: {reason}")
            }
            ScheduleOverlap { first, second } => write!(
                f,
                "cost schedule versions {first:?} and {second:?} cover the same dates: give \
                 each version its own date range"
            ),
            EmptyScheduleBook => f.write_str("a cost model needs at least one schedule version"),
        }
    }
}

impl std::error::Error for CostError {}
