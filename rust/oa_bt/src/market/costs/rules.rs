//! The arithmetic: what a fill owes under a row of rates.
//!
//! Two rules, one per rounding mode. `accrue_fill` is the unrounded accrual the
//! kernel uses on every fill. `calculator_cost` rounds the way the Zerodha brokerage
//! calculator does and exists for parity checks. Both are pure and do not allocate.

use crate::model::Side;

use super::breakdown::CostBreakdown;
use super::brokerage::BrokerageRule;
use super::rounding::{calculator_paise, calculator_rupee};
use super::schedule::{GstRule, SegmentRates, TaxBase, TaxUnit};

/// Unrounded cost of one fill worth `value` rupees (price times quantity in units;
/// premium value for an option).
///
/// STT or CTT falls on the side the rates say, stamp duty on the buy side, exchange
/// charge and SEBI fee on both sides, and GST (`gst`) on brokerage plus exchange charge,
/// plus the SEBI fee when the version taxes it, never on STT or stamp duty.
#[inline]
pub(super) fn accrue_fill(
    rates: &SegmentRates,
    gst: GstRule,
    rule: &BrokerageRule,
    side: Side,
    value: f64,
) -> CostBreakdown {
    let brokerage = rule.charge(value);
    let (stt, stamp_duty) = match side {
        Side::Buy => (value * rates.stt_buy_rate, value * rates.stamp_buy_rate),
        Side::Sell => (value * rates.stt_sell_rate, 0.0),
    };
    let exchange_txn = value * rates.exchange_total_rate();
    let sebi_fee = value * rates.sebi_rate;
    let gst = gst.on(brokerage, exchange_txn, sebi_fee);
    CostBreakdown {
        brokerage,
        stt,
        exchange_txn,
        sebi_fee,
        stamp_duty,
        gst,
    }
}

/// Brokerage on one leg the way the calculator computes it: a percentage above the
/// cap is the cap, anything else is rounded to the paisa.
#[inline]
fn calculator_brokerage(rule: &BrokerageRule, value: f64) -> f64 {
    match *rule {
        BrokerageRule::Rate { rate, cap } => {
            let uncapped = value * rate;
            if uncapped > cap {
                cap
            } else {
                calculator_paise(uncapped)
            }
        }
        BrokerageRule::Zero | BrokerageRule::Flat(_) => rule.charge(value),
    }
}

/// Cost of a buy at `buy_price` and a sell at `sell_price` of `quantity` units,
/// rounded the way the Zerodha brokerage calculator rounds. A single fill is the
/// same computation with the other price at zero.
///
/// The rules, read from the calculator (https://zerodha.com/static/js/brokerage.js,
/// 2026-11-12). The order of the floating-point operations follows the calculator
/// too, because a different order can move a result across a rounding boundary.
/// - Turnover is (buy price + sell price) x quantity, rounded to the paisa.
/// - Brokerage is computed per leg and the sum rounded to the paisa.
/// - The tax (STT or CTT) is on each side's value, or on the turnover, as the
///   row's `calculator_tax_base` says; it is rounded to the paisa and then, when
///   `calculator_tax_unit` is the rupee, to the rupee.
/// - The exchange transaction charge and the IPFT levy are each computed on the
///   turnover and rounded to the paisa, then added.
/// - The SEBI fee is computed on the turnover and rounded to the paisa.
/// - GST is the rate times (brokerage + exchange charge + SEBI fee) as rounded
///   above, rounded to the paisa. For a version that leaves the SEBI fee out of the
///   GST base (trades before 2022-07-18) it is left out here too.
/// - Stamp duty is on the buy value, rounded to the paisa and then to the rupee.
/// - The total is the sum of the lines rounded to the paisa
///   (`CostModel::reported_total`).
///
/// Exactness. For equity and equity derivatives the result is the calculator's, to
/// the paisa. For MCX futures it can differ by one paisa: the calculator takes lots
/// and a contract multiplier and forms its products as price x lots x multiplier
/// (and CTT as rate x price x lots x multiplier), while this engine takes units, so
/// a product can differ in its last binary digit and round the other way.
///
/// One more difference from the calculator: a leg with no value is not an executed
/// order and pays no flat brokerage (the calculator charges Rs 20 on an options leg
/// with a price but zero quantity).
pub(super) fn calculator_cost(
    rates: &SegmentRates,
    gst: GstRule,
    rule: &BrokerageRule,
    buy_price: f64,
    sell_price: f64,
    quantity: f64,
) -> CostBreakdown {
    let buy_value = buy_price * quantity;
    let sell_value = sell_price * quantity;
    let turnover = calculator_paise((buy_price + sell_price) * quantity);

    let brokerage = calculator_paise(
        calculator_brokerage(rule, buy_value) + calculator_brokerage(rule, sell_value),
    );

    let tax = match rates.calculator_tax_base {
        TaxBase::Turnover => turnover * rates.stt_sell_rate,
        TaxBase::Sides => buy_value * rates.stt_buy_rate + sell_value * rates.stt_sell_rate,
    };
    let stt = match rates.calculator_tax_unit {
        TaxUnit::Paise => calculator_paise(tax),
        TaxUnit::Rupee => calculator_rupee(calculator_paise(tax)),
    };

    let exchange_txn = calculator_paise(
        calculator_paise(turnover * rates.exchange_txn_rate)
            + calculator_paise(turnover * rates.ipft_rate),
    );
    let sebi_fee = calculator_paise(turnover * rates.sebi_rate);
    let gst = calculator_paise(gst.on(brokerage, exchange_txn, sebi_fee));
    let stamp_duty = calculator_rupee(calculator_paise(buy_value * rates.stamp_buy_rate));

    CostBreakdown {
        brokerage,
        stt,
        exchange_txn,
        sebi_fee,
        stamp_duty,
        gst,
    }
}
