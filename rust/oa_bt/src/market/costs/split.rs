//! One executed order charged at two segments' rates: a CNC order of cash equity that is
//! partly bought and sold within the day, so part intraday and part delivery
//! (`charges::same_day` decides the split).

use crate::model::{Day, Side};

use super::breakdown::CostBreakdown;
use super::cost_model::CostModel;
use super::error::CostError;
use super::segment::Segment;

impl CostModel {
    /// The cost in rupees of one executed order whose `intraday` units are charged at the
    /// intraday segment's rates and whose `delivery` units at the delivery segment's, each
    /// given as (segment, quantity in price units), at the rates in force on its IST trade
    /// date `day`.
    ///
    /// Brokerage is charged once for the order, as a broker charges an executed order: the
    /// two parts' brokerage added, but never more than the larger of the two rules' most
    /// per order (a flat amount, or a rate's cap). The intraday part is charged first and
    /// the delivery part at most what is left, and GST falls on what each part pays. Every
    /// other charge is the parts' own. A part with no units costs nothing, so an order
    /// wholly on one side costs what `fill_cost` gives.
    ///
    /// Errors: as `fill_cost`, for either part.
    pub fn split_fill_cost(
        &self,
        side: Side,
        price: f64,
        intraday: (Segment, f64),
        delivery: (Segment, f64),
        day: Day,
    ) -> Result<CostBreakdown, CostError> {
        let rule_of = |segment: Segment| self.brokerage().rule(segment.brokerage_class());
        let (intraday_rule, delivery_rule) = (rule_of(intraday.0), rule_of(delivery.0));
        let order_most = intraday_rule
            .most_per_order()
            .max(delivery_rule.most_per_order());
        let mut cost = CostBreakdown::ZERO;
        if intraday.1 > 0.0 {
            let (segment, quantity) = intraday;
            let part =
                self.fill_cost_by_rule(intraday_rule, segment, side, price, quantity, day)?;
            cost.add(&part);
        }
        if delivery.1 > 0.0 {
            let (segment, quantity) = delivery;
            // What the intraday part paid is never above `order_most`; the floor only
            // guards a cap that is not a whole number of paise in calculator rounding.
            let left = delivery_rule.at_most((order_most - cost.brokerage).max(0.0));
            let part = self.fill_cost_by_rule(&left, segment, side, price, quantity, day)?;
            cost.add(&part);
        }
        Ok(cost)
    }
}
