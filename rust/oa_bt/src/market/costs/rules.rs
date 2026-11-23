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
use super::segment::Segment;

/// What the calculator does with a segment's IPFT levy (calculator rounding only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CalculatorIpft {
    /// Added to the exchange line, and so to the GST base and the total.
    Charged,
    /// Added to the exchange line only: the GST base and the total leave it out.
    ShownOnly,
}

/// The calculator's IPFT rule of a segment.
///
/// For NSE currency futures the calculator shows the IPFT levy in its exchange line but
/// leaves it out of GST and the total; every other segment charges what its exchange
/// line shows. Source: the calculator script (read 2026-11-23, `cal_curr_fut` sums the
/// transaction charge into its GST base and total before it adds the IPFT to the line)
/// and the maintainer's screenshot of the same day: USDINR futures, one lot, buy 88.3300
/// and sell 88.3525, turnover 1,76,682.50, exchange line 0.71 (0.6184 + 0.0883), GST
/// 7.34 (18 percent of 40 + 0.62 + 0.18) and total 48.14. Its currency options
/// (`cal_curr_opt`) charge the IPFT in all three: exchange 58.48, GST 17.76, total
/// 116.42 for the same prices as premiums.
pub(super) fn calculator_ipft(segment: Segment) -> CalculatorIpft {
    match segment {
        Segment::CdsFut => CalculatorIpft::ShownOnly,
        _ => CalculatorIpft::Charged,
    }
}

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
        shown_not_charged: 0.0,
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
///   GST base (trades before 2022-07-18) it is left out here too. Where `ipft` is
///   `ShownOnly` (NSE currency futures) the exchange charge in the base is the
///   transaction charge without the IPFT, and the IPFT is `shown_not_charged`.
/// - Stamp duty is on the buy value, rounded to the paisa and then to the rupee.
/// - The total is the sum of the lines rounded to the paisa, less what is shown but
///   not charged (`CostModel::reported_total`).
///
/// Exactness. For equity, equity derivatives and NSE currency derivatives the result
/// is the calculator's, to the paisa. For MCX futures and options it can differ by one
/// paisa: the calculator takes lots and a contract multiplier and forms its products
/// as price x lots x multiplier (and CTT as rate x price x lots x multiplier), while
/// this engine takes units, so a product can differ in its last binary digit and round
/// the other way.
///
/// Two more differences from the calculator: a leg with no value is not an executed
/// order and pays no flat brokerage (the calculator charges Rs 20 on an options leg
/// with a price but zero quantity); and the options rule of a preset is applied to
/// currency and commodity options as it is, while the calculator charges them 0.03
/// percent of (premium + strike) x units capped at Rs 20, which is below Rs 20 only on
/// an order whose strike value is under Rs 66,667.
pub(super) fn calculator_cost(
    rates: &SegmentRates,
    gst: GstRule,
    rule: &BrokerageRule,
    ipft: CalculatorIpft,
    (buy_price, sell_price, quantity): (f64, f64, f64),
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

    let transaction = calculator_paise(turnover * rates.exchange_txn_rate);
    let exchange_txn = calculator_paise(transaction + calculator_paise(turnover * rates.ipft_rate));
    let (charged_exchange, shown_not_charged) = match ipft {
        CalculatorIpft::Charged => (exchange_txn, 0.0),
        CalculatorIpft::ShownOnly => (transaction, exchange_txn - transaction),
    };
    let sebi_fee = calculator_paise(turnover * rates.sebi_rate);
    let gst = calculator_paise(gst.on(brokerage, charged_exchange, sebi_fee));
    let stamp_duty = calculator_rupee(calculator_paise(buy_value * rates.stamp_buy_rate));

    CostBreakdown {
        brokerage,
        stt,
        exchange_txn,
        sebi_fee,
        stamp_duty,
        gst,
        shown_not_charged,
    }
}
