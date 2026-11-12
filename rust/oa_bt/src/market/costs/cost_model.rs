//! The cost model: dated schedules, a brokerage preset and a rounding mode, and the
//! functions that price a fill, a round trip, the DP charge and exercise STT.

use crate::model::{Day, Side};

use super::book::{RateStatus, ScheduleBook};
use super::breakdown::{CostBreakdown, DpCharge};
use super::brokerage::Brokerage;
use super::error::CostError;
use super::rounding::{calculator_paise, calculator_rupee, round_sum, Rounding, Unit};
use super::rules::{accrue_fill, calculator_cost};
use super::schedule::{CostSchedule, SegmentRow};
use super::segment::Segment;

/// Reject a negative, infinite or NaN amount. NaN is in no range, so it is rejected
/// with the negatives.
#[inline]
pub(super) fn check_amount(field: &'static str, value: f64) -> Result<(), CostError> {
    if (0.0..f64::INFINITY).contains(&value) {
        Ok(())
    } else {
        Err(CostError::InvalidInput { field, value })
    }
}

/// The schedule version and segment row that price a trade, with their status.
pub(super) struct Priced<'a> {
    pub(super) schedule: &'a CostSchedule,
    pub(super) row: &'a SegmentRow,
    pub(super) status: RateStatus,
}

/// A complete cost model: schedule versions, a brokerage preset and a rounding mode.
#[derive(Clone, Debug, PartialEq)]
pub struct CostModel {
    schedules: ScheduleBook,
    brokerage: Brokerage,
    rounding: Rounding,
}

impl CostModel {
    /// A model from its three parts. Both `ScheduleBook` and `Brokerage` are
    /// validated when they are built, so this cannot fail.
    pub fn new(schedules: ScheduleBook, brokerage: Brokerage, rounding: Rounding) -> CostModel {
        CostModel {
            schedules,
            brokerage,
            rounding,
        }
    }

    /// The default model: the built-in Indian schedules, the Zerodha preset and
    /// contract-note rounding. Errors: as `ScheduleBook::builtin`.
    pub fn builtin() -> Result<CostModel, CostError> {
        Ok(CostModel::new(
            ScheduleBook::builtin()?,
            Brokerage::zerodha(),
            Rounding::ContractNote,
        ))
    }

    /// The schedule versions the model looks up by trade date.
    pub fn schedules(&self) -> &ScheduleBook {
        &self.schedules
    }

    /// The brokerage preset.
    pub fn brokerage(&self) -> &Brokerage {
        &self.brokerage
    }

    /// The rounding mode.
    pub fn rounding(&self) -> Rounding {
        self.rounding
    }

    /// The total of a breakdown as this model's rounding mode reports it, in rupees.
    ///
    /// In calculator mode the components are whole paise and the calculator's last
    /// step rounds their sum to the paisa. In contract-note mode it is the plain sum
    /// of the unrounded accruals.
    pub fn reported_total(&self, cost: &CostBreakdown) -> f64 {
        match self.rounding {
            Rounding::ContractNote => cost.total(),
            Rounding::Calculator => round_sum(cost.total(), Unit::Paisa),
        }
    }

    /// Find what prices a trade in `segment` on the IST trade date `day`.
    #[inline]
    pub(super) fn priced(&self, segment: Segment, day: Day) -> Result<Priced<'_>, CostError> {
        let (index, backfilled) = self.schedules.lookup(day)?;
        let schedule = &self.schedules.versions()[index];
        let row = schedule
            .row(segment)
            .ok_or(CostError::SegmentNotInSchedule { segment, day })?;
        let status = RateStatus::new(index, row.verified_through, day, backfilled);
        Ok(Priced {
            schedule,
            row,
            status,
        })
    }

    /// Which schedule version prices a trade in `segment` on the IST trade date
    /// `day`, and whether its rates are verified for that date.
    ///
    /// Errors: `UnsupportedDay`, `NoScheduleForDay` or `ScheduleGap` when no version
    /// prices `day`, and `SegmentNotInSchedule` when the version has no row for
    /// `segment`.
    pub fn status(&self, segment: Segment, day: Day) -> Result<RateStatus, CostError> {
        self.priced(segment, day).map(|priced| priced.status)
    }

    /// Which schedule version supplies GST on the IST date `day` (for the DP charge)
    /// and whether its GST rate is verified for that date. Errors: as `status`,
    /// without the segment.
    pub fn gst_status(&self, day: Day) -> Result<RateStatus, CostError> {
        let (index, backfilled) = self.schedules.lookup(day)?;
        let verified_through = self.schedules.versions()[index].gst_verified_through;
        Ok(RateStatus::new(index, verified_through, day, backfilled))
    }

    /// Price one fill under rates that were already looked up. The caller has
    /// checked the price, the quantity and their product.
    #[inline]
    pub(super) fn cost_under(
        &self,
        priced: &Priced<'_>,
        segment: Segment,
        side: Side,
        price: f64,
        quantity: f64,
    ) -> CostBreakdown {
        let rule = self.brokerage.rule(segment.brokerage_class());
        let rates = &priced.row.rates;
        let gst_rate = priced.schedule.gst_rate;
        match (self.rounding, side) {
            (Rounding::ContractNote, _) => {
                accrue_fill(rates, gst_rate, rule, side, price * quantity)
            }
            (Rounding::Calculator, Side::Buy) => {
                calculator_cost(rates, gst_rate, rule, price, 0.0, quantity)
            }
            (Rounding::Calculator, Side::Sell) => {
                calculator_cost(rates, gst_rate, rule, 0.0, price, quantity)
            }
        }
    }

    /// The cost in rupees of one fill, at the rates in force on its IST trade date
    /// `day`. This is the function the kernel calls per fill: pure, no allocation.
    ///
    /// `price` is in rupees per price unit and `quantity` is the number of price
    /// units, never lots, so that `price * quantity` is the traded value in rupees.
    /// For an equity that is shares; for an option the price is the premium; for an
    /// MCX future it is lots times the contract multiplier (GOLD is quoted per 10
    /// grams with a 1 kg lot, so one lot is 100 units: passing 1 would understate
    /// every charge a hundred times). When the traded value is known, call
    /// `fill_cost_value` instead.
    ///
    /// One call is one executed order: brokerage with a per-order cap or a flat
    /// per-order amount is charged once per call. An order filled in several parts
    /// must be costed once on its total, not once per part. A zero quantity costs
    /// nothing.
    ///
    /// In contract-note mode the components are unrounded accruals (feed them to
    /// `ContractNoteAggregator`). In calculator mode they are rounded the way the
    /// Zerodha calculator rounds a one-sided trade.
    ///
    /// Errors: `InvalidInput` when the price, the quantity or their product is
    /// negative, infinite or NaN, and the errors of `status`.
    #[inline]
    pub fn fill_cost(
        &self,
        segment: Segment,
        side: Side,
        price: f64,
        quantity: f64,
        day: Day,
    ) -> Result<CostBreakdown, CostError> {
        check_amount("price", price)?;
        check_amount("quantity", quantity)?;
        check_amount("traded value (price x quantity)", price * quantity)?;
        let priced = self.priced(segment, day)?;
        Ok(self.cost_under(&priced, segment, side, price, quantity))
    }

    /// The cost in rupees of one fill worth `value` rupees (the premium value for an
    /// option), at the rates in force on its IST trade date `day`.
    ///
    /// The same as `fill_cost` with the units already multiplied out, for callers
    /// that hold the traded value and should not have to reason about lot
    /// multipliers. One call is one executed order. Errors: as `fill_cost`.
    #[inline]
    pub fn fill_cost_value(
        &self,
        segment: Segment,
        side: Side,
        value: f64,
        day: Day,
    ) -> Result<CostBreakdown, CostError> {
        check_amount("traded value", value)?;
        let priced = self.priced(segment, day)?;
        Ok(self.cost_under(&priced, segment, side, value, 1.0))
    }

    /// The cost in rupees of buying `quantity` units on `buy_day` and selling them
    /// on `sell_day` (IST trade dates), each leg one executed order.
    ///
    /// When both legs are on the same day they are costed together: in calculator
    /// mode that is exactly the Zerodha calculator's round trip (rounded together,
    /// not leg by leg), and in contract-note mode the sum of the two unrounded
    /// fills. On different days each leg is costed with `fill_cost` at the rates of
    /// its own day and the two are added.
    ///
    /// Errors: as `fill_cost`, for either leg.
    pub fn round_trip_cost(
        &self,
        segment: Segment,
        buy_price: f64,
        sell_price: f64,
        quantity: f64,
        buy_day: Day,
        sell_day: Day,
    ) -> Result<CostBreakdown, CostError> {
        if buy_day == sell_day && self.rounding == Rounding::Calculator {
            return self.calculator_round_trip(segment, buy_price, sell_price, quantity, buy_day);
        }
        let mut cost = self.fill_cost(segment, Side::Buy, buy_price, quantity, buy_day)?;
        cost.add(&self.fill_cost(segment, Side::Sell, sell_price, quantity, sell_day)?);
        Ok(cost)
    }

    fn calculator_round_trip(
        &self,
        segment: Segment,
        buy_price: f64,
        sell_price: f64,
        quantity: f64,
        day: Day,
    ) -> Result<CostBreakdown, CostError> {
        check_amount("buy price", buy_price)?;
        check_amount("sell price", sell_price)?;
        check_amount("quantity", quantity)?;
        check_amount("turnover", (buy_price + sell_price) * quantity)?;
        let priced = self.priced(segment, day)?;
        let rule = self.brokerage.rule(segment.brokerage_class());
        let gst_rate = priced.schedule.gst_rate;
        Ok(calculator_cost(
            &priced.row.rates,
            gst_rate,
            rule,
            buy_price,
            sell_price,
            quantity,
        ))
    }

    /// The depository participant charge for selling one scrip from holdings on the
    /// IST date `day`.
    ///
    /// It is charged once per scrip per day on which that scrip is sold from the
    /// demat account, whatever the quantity or number of orders, and it is not part
    /// of the contract note. A CNC buy and sell of the same scrip on the same day
    /// is an intraday trade and has no DP charge, so the caller decides when the
    /// charge applies; this function only prices it. The fee comes from the
    /// brokerage preset (the result says whose fee it is) and GST from the schedule
    /// in force on `day` (see `gst_status`). GST is rounded to the paisa in both
    /// rounding modes, because the charge is posted to the ledger as it stands.
    ///
    /// Errors: `UnsupportedDay`, `NoScheduleForDay` or `ScheduleGap` when no version
    /// prices `day`.
    pub fn dp_charge(&self, day: Day) -> Result<DpCharge, CostError> {
        let (index, _) = self.schedules.lookup(day)?;
        let fee = self.brokerage.dp_fee();
        let gst = round_sum(fee * self.schedules.versions()[index].gst_rate, Unit::Paisa);
        Ok(DpCharge {
            fee,
            gst,
            fee_source: self.brokerage.dp_fee_source(),
        })
    }

    /// STT in rupees on an option that expires in the money and is exercised, paid
    /// by the buyer, at the rate in force on the IST date `day` (the expiry date).
    ///
    /// `intrinsic` is the intrinsic value per unit in rupees at the settlement
    /// price (zero when out of the money) and `quantity` is in units. The result is
    /// unrounded in contract-note mode (add it to that day's note with
    /// `ContractNoteAggregator::add_stt`) and rounded to the rupee in calculator
    /// mode.
    ///
    /// Errors: `NotAnOption` when `segment` is not an options segment,
    /// `InvalidInput` when an amount or their product is negative, infinite or NaN,
    /// and the errors of `status`.
    pub fn exercise_stt(
        &self,
        segment: Segment,
        intrinsic: f64,
        quantity: f64,
        day: Day,
    ) -> Result<f64, CostError> {
        if !segment.is_option() {
            return Err(CostError::NotAnOption(segment));
        }
        check_amount("intrinsic value", intrinsic)?;
        check_amount("quantity", quantity)?;
        check_amount(
            "exercised value (intrinsic x quantity)",
            intrinsic * quantity,
        )?;
        let priced = self.priced(segment, day)?;
        let accrued = intrinsic * quantity * priced.row.rates.stt_exercise_rate;
        Ok(match self.rounding {
            Rounding::ContractNote => accrued,
            Rounding::Calculator => calculator_rupee(calculator_paise(accrued)),
        })
    }
}
