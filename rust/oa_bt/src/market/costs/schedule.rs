//! Statutory and exchange rates as data, versioned by effective date.
//!
//! A `CostSchedule` is a plain struct: a date range, where the figures came from, and
//! one row of rates per segment, each row with the date its figures were verified
//! through. The built-in versions (parsed from `schedules.csv`) and user-supplied
//! versions are the same type and go through the same validation. `book` holds a set
//! of versions and finds the one in force on a trade date.
//!
//! Units: every field ending in `_rate` is a fraction, so 0.001 is 0.1 percent.

use crate::model::{is_supported_day, Day};

use super::error::{iso_date, CostError};
use super::segment::{Segment, SEGMENT_COUNT};

/// Plausibility bounds on schedule rates, as fractions. They are not rates anyone
/// charges: they exist to catch a rate given in percent units by mistake (0.1 meant
/// as 0.1 percent would be a 10 percent tax). The exchange bound is loose because
/// the BSE SS and ST groups really pay 1 percent.
const MAX_TAX_RATE: f64 = 0.005;
const MAX_EXCHANGE_TXN_RATE: f64 = 0.02;
const MAX_IPFT_RATE: f64 = 0.00005;
const MAX_SEBI_RATE: f64 = 0.00005;
const MAX_STAMP_RATE: f64 = 0.001;
const MAX_GST_RATE: f64 = 0.5;

/// What the calculator computes a segment's STT or CTT on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TaxBase {
    /// Each side's own value at that side's rate.
    Sides,
    /// The rounded turnover of the round trip (equity delivery, taxed at one rate on
    /// both sides).
    Turnover,
}

impl TaxBase {
    /// The code used in the data file: "sides" or "turnover".
    pub fn code(self) -> &'static str {
        match self {
            TaxBase::Sides => "sides",
            TaxBase::Turnover => "turnover",
        }
    }

    /// Parse a data-file code.
    pub fn from_code(code: &str) -> Option<TaxBase> {
        match code {
            "sides" => Some(TaxBase::Sides),
            "turnover" => Some(TaxBase::Turnover),
            _ => None,
        }
    }
}

/// What the calculator rounds a segment's STT or CTT to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TaxUnit {
    /// To the paisa and then to the rupee (STT).
    Rupee,
    /// To the paisa only (CTT on commodity futures).
    Paise,
}

impl TaxUnit {
    /// The code used in the data file: "rupee" or "paise".
    pub fn code(self) -> &'static str {
        match self {
            TaxUnit::Rupee => "rupee",
            TaxUnit::Paise => "paise",
        }
    }

    /// Parse a data-file code.
    pub fn from_code(code: &str) -> Option<TaxUnit> {
        match code {
            "rupee" => Some(TaxUnit::Rupee),
            "paise" => Some(TaxUnit::Paise),
            _ => None,
        }
    }
}

/// The rates of one segment. Every rate is a fraction of traded value in rupees
/// (price times quantity in units; premium value for options), so 0.001 means
/// 0.1 percent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SegmentRates {
    /// STT (or CTT on MCX) on the buy side.
    pub stt_buy_rate: f64,
    /// STT (or CTT on MCX) on the sell side.
    pub stt_sell_rate: f64,
    /// Exchange transaction charge, without the IPFT levy.
    pub exchange_txn_rate: f64,
    /// Investor Protection Fund Trust levy, charged with the transaction charge and
    /// reported inside it. Zero where the exchange has none.
    pub ipft_rate: f64,
    /// SEBI turnover fee.
    pub sebi_rate: f64,
    /// Stamp duty on the buy side.
    pub stamp_buy_rate: f64,
    /// STT on an option that is exercised, as a fraction of intrinsic value, paid
    /// by the buyer. Zero for segments that are not options.
    pub stt_exercise_rate: f64,
    /// Calculator mode only: what the tax is computed on.
    pub calculator_tax_base: TaxBase,
    /// Calculator mode only: what the tax is rounded to.
    pub calculator_tax_unit: TaxUnit,
}

impl SegmentRates {
    /// Exchange transaction charge including IPFT: the rate a rate card quotes.
    #[inline]
    pub fn exchange_total_rate(&self) -> f64 {
        self.exchange_txn_rate + self.ipft_rate
    }

    /// Each rate with its name and plausibility bound.
    fn bounded(&self) -> [(&'static str, f64, f64); 7] {
        [
            ("stt_buy_rate", self.stt_buy_rate, MAX_TAX_RATE),
            ("stt_sell_rate", self.stt_sell_rate, MAX_TAX_RATE),
            (
                "exchange_txn_rate",
                self.exchange_txn_rate,
                MAX_EXCHANGE_TXN_RATE,
            ),
            ("ipft_rate", self.ipft_rate, MAX_IPFT_RATE),
            ("sebi_rate", self.sebi_rate, MAX_SEBI_RATE),
            ("stamp_buy_rate", self.stamp_buy_rate, MAX_STAMP_RATE),
            ("stt_exercise_rate", self.stt_exercise_rate, MAX_TAX_RATE),
        ]
    }
}

/// One segment's row in a schedule version: its rates and how far they are verified.
#[derive(Clone, Debug, PartialEq)]
pub struct SegmentRow {
    /// The rates.
    pub rates: SegmentRates,
    /// The last trade date (IST) the rates were checked against their source for,
    /// or None when they were never verified. A later trade is still charged at
    /// these rates, flagged as unverified.
    pub verified_through: Option<Day>,
    /// Caveats specific to this row, such as conflicting sources. May be empty.
    pub notes: String,
}

/// One version of the rate card.
#[derive(Clone, Debug, PartialEq)]
pub struct CostSchedule {
    /// A short unique name, recorded in run manifests.
    pub id: String,
    /// First trade date (IST) the version applies to.
    pub valid_from: Day,
    /// Last trade date (IST) the version applies to, inclusive, or None while it is
    /// still in force.
    pub valid_to: Option<Day>,
    /// GST rate (a fraction) on brokerage plus exchange transaction charge, plus the
    /// SEBI fee when `gst_on_sebi_fee`, and on the DP fee.
    pub gst_rate: f64,
    /// True when GST falls on the SEBI fee too: from 2022-07-18 (SEBI circular
    /// SEBI/HO/GSD/TAD/CIR/P/2022/0097); before, the SEBI fee was exempt.
    pub gst_on_sebi_fee: bool,
    /// The last trade date the GST rate was verified for, or None.
    pub gst_verified_through: Option<Day>,
    /// Where the figures came from.
    pub source: String,
    /// Caveats: what was assumed and what was not checked.
    pub notes: String,
    /// One row per segment, indexed by `Segment::index`. None means the version
    /// has no rates for that segment, and costing a fill there is an error.
    pub rows: [Option<SegmentRow>; SEGMENT_COUNT],
}

fn check_rate(what: &str, name: &str, rate: f64, bound: f64) -> Result<(), String> {
    if (0.0..=bound).contains(&rate) {
        return Ok(());
    }
    Err(format!(
        "{what}: {name} is {rate}, outside 0 to {bound}; rates are fractions: 0.001 is 0.1 \
         percent"
    ))
}

/// What GST falls on under one version, and at what rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct GstRule {
    rate: f64,
    on_sebi_fee: bool,
}

impl GstRule {
    /// GST on a fill's brokerage, exchange charge and SEBI fee (rupees), in rupees; the
    /// SEBI fee counts only when the version puts it in the base.
    #[inline]
    pub(super) fn on(self, brokerage: f64, exchange_txn: f64, sebi_fee: f64) -> f64 {
        let base = if self.on_sebi_fee {
            brokerage + exchange_txn + sebi_fee
        } else {
            brokerage + exchange_txn
        };
        self.rate * base
    }
}

impl CostSchedule {
    /// The row of a segment, if the version has one.
    #[inline]
    pub fn row(&self, segment: Segment) -> Option<&SegmentRow> {
        self.rows[segment.index()].as_ref()
    }

    /// The version's GST: its rate and whether the SEBI fee is in its base.
    #[inline]
    pub(super) fn gst_rule(&self) -> GstRule {
        GstRule {
            rate: self.gst_rate,
            on_sebi_fee: self.gst_on_sebi_fee,
        }
    }

    /// Check the version on its own.
    ///
    /// Errors: `InvalidSchedule` when the id is empty, a date is outside the
    /// supported range, the range ends before it starts, there is no rate row, a
    /// rate is outside the plausible range for its kind (which catches a rate given
    /// in percent units), or a row taxed on turnover has different buy and sell
    /// rates.
    pub fn validate(&self) -> Result<(), CostError> {
        self.check()
            .map_err(|why| CostError::InvalidSchedule(format!("version {:?}: {why}", self.id)))
    }

    fn check(&self) -> Result<(), String> {
        if self.id.trim().is_empty() {
            return Err("a schedule version needs an id".to_string());
        }
        self.check_dates()?;
        check_rate("the schedule", "gst_rate", self.gst_rate, MAX_GST_RATE)?;
        if self.rows.iter().all(Option::is_none) {
            return Err("it has no segment rates".to_string());
        }
        for segment in Segment::ALL {
            let Some(row) = self.row(segment) else {
                continue;
            };
            let what = format!("segment {}", segment.code());
            for (name, rate, bound) in row.rates.bounded() {
                check_rate(&what, name, rate, bound)?;
            }
            let rates = &row.rates;
            let one_rate = rates.stt_buy_rate == rates.stt_sell_rate;
            if rates.calculator_tax_base == TaxBase::Turnover && !one_rate {
                return Err(format!(
                    "{what}: tax on turnover needs equal buy and sell rates"
                ));
            }
        }
        Ok(())
    }

    fn check_dates(&self) -> Result<(), String> {
        let verified = self
            .rows
            .iter()
            .flatten()
            .filter_map(|row| row.verified_through);
        let dates = [
            Some(self.valid_from),
            self.valid_to,
            self.gst_verified_through,
        ];
        if dates
            .into_iter()
            .flatten()
            .chain(verified)
            .any(|day| !is_supported_day(day))
        {
            return Err("a date is outside the supported range".to_string());
        }
        match self.valid_to {
            Some(end) if end < self.valid_from => Err(format!(
                "it starts on {} and cannot end before that",
                iso_date(self.valid_from)
            )),
            _ => Ok(()),
        }
    }
}
