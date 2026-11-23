//! Brokerage presets.
//!
//! Brokerage is the only part of a trade's cost the broker decides. Statutory and
//! exchange charges come from the dated schedule and are the same under every
//! preset.
//!
//! Units: a `rate` is a fraction of order value (0.0003 is 0.03 percent). A function
//! or rule named `percent` takes percent units (0.03 is 0.03 percent). Amounts and
//! caps are rupees.

use super::error::{CostError, NameKind};
use super::segment::BrokerageClass;

/// Zerodha intraday equity and futures brokerage rate: 0.03 percent of order value.
/// Source: reference note section 1 (Zerodha charges page).
const ZERODHA_RATE: f64 = 0.0003;
/// Zerodha's cap on percentage brokerage, and its flat options brokerage: Rs 20 per
/// executed order. Source: reference note section 1.
const ZERODHA_RUPEES_PER_ORDER: f64 = 20.0;
/// Depository participant fee on a delivery sell, in rupees per scrip per day before
/// GST: Rs 3.5 CDSL plus Rs 9.5 Zerodha. Source: reference note section 4.
const DP_DEPOSITORY_FEE: f64 = 3.5;
const DP_BROKER_FEE: f64 = 9.5;
const DP_FEE_ZERODHA: f64 = DP_DEPOSITORY_FEE + DP_BROKER_FEE;

// The "standard" preset: what most brokers charge, set by the maintainer on 2026-11-23
// ("Equity Delivery: Rs 20 per executed order (or up to 2.5% of order value if lower).
// Equity Intraday: Rs 20 per executed order or 0.1% of the trade value, whichever is
// lower"), with futures and options as the lead ruled for unit C of 2.4.1.
/// Most charged per executed order under the standard preset, and its flat options
/// brokerage, in rupees.
const STANDARD_RUPEES_PER_ORDER: f64 = 20.0;
/// Standard equity delivery rate: 2.5 percent of order value, capped per order.
const STANDARD_DELIVERY_RATE: f64 = 0.025;
/// Standard equity intraday rate: 0.1 percent of order value, capped per order.
const STANDARD_INTRADAY_RATE: f64 = 0.001;
/// Standard futures rate (equity, currency and commodity): 0.03 percent of order value,
/// capped per order.
const STANDARD_FUTURES_RATE: f64 = 0.0003;

/// Plausibility bounds on a brokerage rate, as fractions of order value. They are
/// not rates anyone charges: they exist to catch a rate given in the wrong unit. A
/// rate above 5 percent was probably a percentage passed as a fraction, and a
/// non-zero rate below 0.001 percent was probably a fraction passed as a percentage.
const MAX_BROKERAGE_RATE: f64 = 0.05;
const MIN_BROKERAGE_RATE: f64 = 0.00001;

const PERCENT: f64 = 100.0;

/// A standard-preset rate (a fraction of order value) capped at Rs 20 per executed order.
const fn standard_capped(rate: f64) -> BrokerageRule {
    BrokerageRule::Rate {
        rate,
        cap: STANDARD_RUPEES_PER_ORDER,
    }
}

/// Brokerage on one executed order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BrokerageRule {
    /// No brokerage.
    Zero,
    /// A flat amount in rupees per executed order.
    Flat(f64),
    /// A fraction of the order value, capped per executed order.
    Rate {
        /// Fraction of order value: 0.0003 is 0.03 percent.
        rate: f64,
        /// Most charged per executed order, in rupees. `f64::INFINITY` means no
        /// cap.
        cap: f64,
    },
}

impl BrokerageRule {
    /// A rule from percent units: `percent(0.03, Some(20.0))` is 0.03 percent of the
    /// order value, at most Rs 20 per executed order. `None` means no cap.
    pub fn percent(percent: f64, cap: Option<f64>) -> BrokerageRule {
        BrokerageRule::Rate {
            rate: percent / PERCENT,
            cap: cap.unwrap_or(f64::INFINITY),
        }
    }

    /// Unrounded brokerage in rupees for one executed order worth `value` rupees. An
    /// order with no value (zero quantity) is not an executed order and costs
    /// nothing.
    #[inline]
    pub fn charge(&self, value: f64) -> f64 {
        match *self {
            BrokerageRule::Zero => 0.0,
            BrokerageRule::Flat(amount) => {
                if value > 0.0 {
                    amount
                } else {
                    0.0
                }
            }
            BrokerageRule::Rate { rate, cap } => (value * rate).min(cap),
        }
    }

    /// The most one executed order pays under the rule, in rupees: nothing, the flat
    /// amount, or the cap (`f64::INFINITY` for a rate with no cap).
    pub(super) fn most_per_order(&self) -> f64 {
        match *self {
            BrokerageRule::Zero => 0.0,
            BrokerageRule::Flat(amount) => amount,
            BrokerageRule::Rate { cap, .. } => cap,
        }
    }

    /// The same rule, charging at most `rupees` per executed order.
    pub(super) fn at_most(&self, rupees: f64) -> BrokerageRule {
        match *self {
            BrokerageRule::Zero => BrokerageRule::Zero,
            BrokerageRule::Flat(amount) => BrokerageRule::Flat(amount.min(rupees)),
            BrokerageRule::Rate { rate, cap } => BrokerageRule::Rate {
                rate,
                cap: cap.min(rupees),
            },
        }
    }

    fn validate(&self) -> Result<(), CostError> {
        match *self {
            BrokerageRule::Zero => Ok(()),
            BrokerageRule::Flat(amount) => check_rupees("the flat amount", amount),
            BrokerageRule::Rate { rate, cap } => {
                check_rate(rate)?;
                // An infinite cap means no cap; NaN and negatives are rejected.
                if cap >= 0.0 {
                    Ok(())
                } else {
                    Err(CostError::InvalidBrokerage(format!(
                        "the cap must be zero or more rupees, got {cap}"
                    )))
                }
            }
        }
    }
}

fn check_rate(rate: f64) -> Result<(), CostError> {
    let plausible = rate == 0.0 || (MIN_BROKERAGE_RATE..=MAX_BROKERAGE_RATE).contains(&rate);
    if plausible {
        return Ok(());
    }
    Err(CostError::InvalidBrokerage(format!(
        "a brokerage rate of {rate} ({} percent of order value) is not plausible: rates are \
         fractions (0.0003 is 0.03 percent) and percent() takes percent units (0.03 is 0.03 \
         percent); the accepted range is {} to {} percent, or zero",
        rate * PERCENT,
        MIN_BROKERAGE_RATE * PERCENT,
        MAX_BROKERAGE_RATE * PERCENT
    )))
}

fn check_rupees(what: &str, rupees: f64) -> Result<(), CostError> {
    if (0.0..f64::INFINITY).contains(&rupees) {
        Ok(())
    } else {
        Err(CostError::InvalidBrokerage(format!(
            "{what} must be a finite number of rupees that is zero or more, got {rupees}"
        )))
    }
}

/// Where a preset's DP fee comes from, so a result can say whose fee it charged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DpFeeSource {
    /// The Zerodha preset's own published fee.
    ZerodhaPreset,
    /// No fee was given for this preset, so Zerodha's stands in for it.
    ZerodhaStandIn,
    /// Supplied by the caller.
    Supplied,
}

impl DpFeeSource {
    /// A short code for results: zerodha_preset, zerodha_stand_in or supplied.
    pub fn code(self) -> &'static str {
        match self {
            DpFeeSource::ZerodhaPreset => "zerodha_preset",
            DpFeeSource::ZerodhaStandIn => "zerodha_stand_in",
            DpFeeSource::Supplied => "supplied",
        }
    }
}

/// A brokerage preset: one rule per brokerage class (equity delivery, equity
/// intraday, futures, options) plus the DP fee.
///
/// The fields are private so that a preset can only be built through a constructor
/// that validates it; read them with `rule`, `dp_fee` and `dp_fee_source`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Brokerage {
    delivery: BrokerageRule,
    intraday: BrokerageRule,
    futures: BrokerageRule,
    options: BrokerageRule,
    dp_fee: f64,
    dp_fee_source: DpFeeSource,
}

impl Brokerage {
    /// Standard (the default), what most brokers charge: equity delivery 2.5 percent or
    /// Rs 20 per executed order, whichever is lower; equity intraday 0.1 percent or Rs 20;
    /// futures (equity, currency and commodity) 0.03 percent or Rs 20; options (equity,
    /// currency and commodity) Rs 20 per executed order. No DP fee is published for it,
    /// so Zerodha's Rs 13 before GST stands in (`DpFeeSource::ZerodhaStandIn`), as for
    /// `custom`.
    pub const fn standard() -> Brokerage {
        Brokerage {
            delivery: standard_capped(STANDARD_DELIVERY_RATE),
            intraday: standard_capped(STANDARD_INTRADAY_RATE),
            futures: standard_capped(STANDARD_FUTURES_RATE),
            options: BrokerageRule::Flat(STANDARD_RUPEES_PER_ORDER),
            dp_fee: DP_FEE_ZERODHA,
            dp_fee_source: DpFeeSource::ZerodhaStandIn,
        }
    }

    /// Zerodha: delivery free; intraday and futures 0.03 percent or Rs 20 per executed
    /// order, whichever is lower; options Rs 20 per executed order; DP fee Rs 13 before
    /// GST (Rs 3.5 CDSL plus Rs 9.5 Zerodha). Its brokerage calculator is what
    /// `Rounding::Calculator` reproduces.
    pub const fn zerodha() -> Brokerage {
        let capped = BrokerageRule::Rate {
            rate: ZERODHA_RATE,
            cap: ZERODHA_RUPEES_PER_ORDER,
        };
        Brokerage {
            delivery: BrokerageRule::Zero,
            intraday: capped,
            futures: capped,
            options: BrokerageRule::Flat(ZERODHA_RUPEES_PER_ORDER),
            dp_fee: DP_FEE_ZERODHA,
            dp_fee_source: DpFeeSource::ZerodhaPreset,
        }
    }

    /// No brokerage in any segment. Statutory charges still apply, and so does a DP
    /// fee: see `custom` for where it comes from.
    pub const fn zero_brokerage() -> Brokerage {
        Brokerage::uniform(BrokerageRule::Zero)
    }

    /// The same flat amount in rupees per executed order in every segment, delivery
    /// included. The DP fee is as for `custom`. Errors: `InvalidBrokerage` for a
    /// negative, infinite or NaN amount.
    pub fn flat(amount: f64) -> Result<Brokerage, CostError> {
        Brokerage::uniform(BrokerageRule::Flat(amount)).validated()
    }

    /// A percentage of order value in every segment, in percent units: `percent(0.03,
    /// Some(20.0))` is 0.03 percent, at most Rs 20 per executed order (`None` for no
    /// cap). The DP fee is as for `custom`.
    ///
    /// Errors: `InvalidBrokerage` for a percentage outside 0.001 to 5 (zero is
    /// allowed) or a negative or NaN cap.
    pub fn percent(percent: f64, cap: Option<f64>) -> Result<Brokerage, CostError> {
        Brokerage::uniform(BrokerageRule::percent(percent, cap)).validated()
    }

    /// User-defined rules per brokerage class.
    ///
    /// The DP fee: no fee is given here, so Zerodha's Rs 13 before GST stands in
    /// for it, and results say so (`DpFeeSource::ZerodhaStandIn`). Every broker
    /// passes on a depository fee, so zero would understate delivery costs; set your
    /// broker's figure with `with_dp_fee`.
    ///
    /// Errors: `InvalidBrokerage` when a rule has a negative, non-finite or
    /// implausible amount or rate.
    pub fn custom(
        delivery: BrokerageRule,
        intraday: BrokerageRule,
        futures: BrokerageRule,
        options: BrokerageRule,
    ) -> Result<Brokerage, CostError> {
        let dp_fee_source = DpFeeSource::ZerodhaStandIn;
        Brokerage {
            delivery,
            intraday,
            futures,
            options,
            dp_fee: DP_FEE_ZERODHA,
            dp_fee_source,
        }
        .validated()
    }

    /// The same preset with the caller's DP fee, in rupees per scrip per sell day
    /// before GST. Errors: `InvalidBrokerage` for a negative, infinite or NaN fee.
    pub fn with_dp_fee(mut self, dp_fee: f64) -> Result<Brokerage, CostError> {
        self.dp_fee = dp_fee;
        self.dp_fee_source = DpFeeSource::Supplied;
        self.validated()
    }

    /// A built-in preset by name: "standard", "zerodha" or "zero_brokerage". Errors:
    /// `UnknownName` for any other name; there is no default preset to fall back on.
    pub fn from_name(name: &str) -> Result<Brokerage, CostError> {
        match name {
            "standard" => Ok(Brokerage::standard()),
            "zerodha" => Ok(Brokerage::zerodha()),
            "zero_brokerage" => Ok(Brokerage::zero_brokerage()),
            other => Err(CostError::unknown(NameKind::Preset, other)),
        }
    }

    /// Depository participant fee in rupees per scrip per sell day, before GST (see
    /// `CostModel::dp_charge`).
    #[inline]
    pub fn dp_fee(&self) -> f64 {
        self.dp_fee
    }

    /// Where the DP fee comes from.
    pub fn dp_fee_source(&self) -> DpFeeSource {
        self.dp_fee_source
    }

    /// The rule for a brokerage class.
    #[inline]
    pub fn rule(&self, class: BrokerageClass) -> &BrokerageRule {
        match class {
            BrokerageClass::Delivery => &self.delivery,
            BrokerageClass::Intraday => &self.intraday,
            BrokerageClass::Futures => &self.futures,
            BrokerageClass::Options => &self.options,
        }
    }

    /// One rule for every class, with Zerodha's DP fee standing in.
    const fn uniform(rule: BrokerageRule) -> Brokerage {
        Brokerage {
            delivery: rule,
            intraday: rule,
            futures: rule,
            options: rule,
            dp_fee: DP_FEE_ZERODHA,
            dp_fee_source: DpFeeSource::ZerodhaStandIn,
        }
    }

    fn validated(self) -> Result<Brokerage, CostError> {
        self.delivery.validate()?;
        self.intraday.validate()?;
        self.futures.validate()?;
        self.options.validate()?;
        check_rupees("the DP fee", self.dp_fee)?;
        Ok(self)
    }
}

impl Default for Brokerage {
    /// The standard preset.
    fn default() -> Brokerage {
        Brokerage::standard()
    }
}
