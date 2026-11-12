//! Delivery (CNC) fills of cash equity bought and sold on one day (04-execution 2): a
//! CNC buy and sell of one scrip on the same day is charged as intraday, with no DP
//! charge.
//!
//! Brokers net a scrip's delivery buys and sells of a trade day: the quantity bought and
//! sold within the day is an intraday trade, the rest is delivery. A fill is charged when
//! it happens, before the day is over, so this book matches every CNC fill of a scrip
//! against the day's earlier fills on the other side, first in first out:
//!
//! - the matched units of the new fill are charged at the intraday rates and the rest at
//!   the delivery rates;
//! - each earlier fill that the new one matched is charged again with its new split
//!   (`Repriced`), and the difference is posted with the new fill;
//! - the DP charge is paid once per scrip and day when units are sold from holdings,
//!   and refunded when later buys of the day turn every such sale into an intraday one.
//!
//! A fill split between the two rates is costed as two parts, so a per-order brokerage
//! that is not zero on delivery would be charged on each part.

use crate::exec::InstrumentId;
use crate::market::costs::{CostBreakdown, CostError, CostModel, Segment};
use crate::model::{Day, Side};

/// The two segments of one scrip's CNC fills.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Segments {
    pub(super) delivery: Segment,
    pub(super) intraday: Segment,
}

/// One CNC fill of the day and its split, units.
#[derive(Clone, Copy, Debug, PartialEq)]
struct DayFill {
    record: usize,
    instrument: InstrumentId,
    day: Day,
    side: Side,
    price_rupees: f64,
    intraday_units: f64,
    delivery_units: f64,
}

/// An earlier fill whose split a new fill changed: its charge record and new split.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Repriced {
    pub(super) record: usize,
    pub(super) side: Side,
    pub(super) price_rupees: f64,
    pub(super) intraday_units: f64,
    pub(super) delivery_units: f64,
}

/// What a new fill matched.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Matched {
    /// Units of the new fill that are intraday.
    pub(super) units: f64,
    /// The earlier fills to charge again.
    pub(super) repriced: Vec<Repriced>,
}

/// The CNC fills of the current trade day and the DP charges paid on it.
#[derive(Clone, Debug, Default)]
pub(super) struct SameDayBook {
    fills: Vec<DayFill>,
    /// (scrip, day, charge record that carries the DP charge).
    dp_paid: Vec<(InstrumentId, Day, usize)>,
}

impl SameDayBook {
    /// Match a new fill (`record` is its future charge record) against the day's
    /// unmatched fills of the scrip on the other side, first in first out, and keep it.
    /// Fills arrive in time order, so fills of earlier days are dropped.
    pub(super) fn match_fill(
        &mut self,
        record: usize,
        instrument: InstrumentId,
        day: Day,
        side: Side,
        price_rupees: f64,
        units: f64,
    ) -> Matched {
        self.fills.retain(|fill| fill.day == day);
        self.dp_paid.retain(|(_, paid_on, _)| *paid_on == day);
        let mut left = units;
        let mut repriced = Vec::new();
        for fill in self.fills.iter_mut() {
            if fill.instrument != instrument || fill.side == side || fill.delivery_units <= 0.0 {
                continue;
            }
            if left <= 0.0 {
                break;
            }
            let units = fill.delivery_units.min(left);
            fill.delivery_units -= units;
            fill.intraday_units += units;
            left -= units;
            repriced.push(Repriced {
                record: fill.record,
                side: fill.side,
                price_rupees: fill.price_rupees,
                intraday_units: fill.intraday_units,
                delivery_units: fill.delivery_units,
            });
        }
        self.fills.push(DayFill {
            record,
            instrument,
            day,
            side,
            price_rupees,
            intraday_units: units - left,
            delivery_units: left,
        });
        Matched {
            units: units - left,
            repriced,
        }
    }

    /// The charge record that carries the scrip's DP charge of `day`, if one was paid.
    pub(super) fn dp_record(&self, instrument: InstrumentId, day: Day) -> Option<usize> {
        self.dp_paid
            .iter()
            .find(|(i, d, _)| *i == instrument && *d == day)
            .map(|(_, _, record)| *record)
    }

    /// The scrip's DP charge of `day` is carried by `record`.
    pub(super) fn note_dp(&mut self, instrument: InstrumentId, day: Day, record: usize) {
        self.dp_paid.push((instrument, day, record));
    }

    /// The scrip's DP charge of `day` was refunded.
    pub(super) fn forget_dp(&mut self, instrument: InstrumentId, day: Day) {
        self.dp_paid
            .retain(|(i, d, _)| !(*i == instrument && *d == day));
    }

    /// Units of the scrip still sold from holdings (delivery) on `day`.
    pub(super) fn delivery_sold(&self, instrument: InstrumentId, day: Day) -> f64 {
        self.fills
            .iter()
            .filter(|f| f.instrument == instrument && f.day == day && f.side == Side::Sell)
            .map(|f| f.delivery_units)
            .sum()
    }
}

/// The cost of a CNC fill with `intraday_units` at the intraday rates and
/// `delivery_units` at the delivery rates.
///
/// Errors: those of `CostModel::fill_cost`.
pub(super) fn split_cost(
    model: &CostModel,
    segments: Segments,
    side: Side,
    price_rupees: f64,
    (intraday_units, delivery_units): (f64, f64),
    day: Day,
) -> Result<CostBreakdown, CostError> {
    let mut cost = CostBreakdown::ZERO;
    if intraday_units > 0.0 {
        cost.add(&model.fill_cost(segments.intraday, side, price_rupees, intraday_units, day)?);
    }
    if delivery_units > 0.0 {
        cost.add(&model.fill_cost(segments.delivery, side, price_rupees, delivery_units, day)?);
    }
    Ok(cost)
}

/// `new` less `old`, component by component (amounts may be negative).
pub(super) fn difference(new: &CostBreakdown, old: &CostBreakdown) -> CostBreakdown {
    let (new, old) = (new.components(), old.components());
    CostBreakdown::from_components(std::array::from_fn(|k| new[k] - old[k]))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCRIP: InstrumentId = InstrumentId(0);
    const DAY: Day = 20_000;

    #[test]
    fn a_sell_after_a_same_day_buy_is_matched_and_the_buy_repriced() {
        let mut book = SameDayBook::default();
        let buy = book.match_fill(0, SCRIP, DAY, Side::Buy, 1_000.0, 400.0);
        assert_eq!(buy.units, 0.0);
        let sell = book.match_fill(1, SCRIP, DAY, Side::Sell, 1_100.0, 400.0);
        assert_eq!(sell.units, 400.0);
        assert_eq!(
            sell.repriced,
            vec![Repriced {
                record: 0,
                side: Side::Buy,
                price_rupees: 1_000.0,
                intraday_units: 400.0,
                delivery_units: 0.0,
            }]
        );
        assert_eq!(book.delivery_sold(SCRIP, DAY), 0.0);
    }

    #[test]
    fn a_sell_larger_than_the_days_buys_sells_the_rest_from_holdings() {
        let mut book = SameDayBook::default();
        book.match_fill(0, SCRIP, DAY, Side::Buy, 1_000.0, 100.0);
        let sell = book.match_fill(1, SCRIP, DAY, Side::Sell, 1_100.0, 300.0);
        assert_eq!(sell.units, 100.0);
        assert_eq!(book.delivery_sold(SCRIP, DAY), 200.0);
    }

    #[test]
    fn fills_of_an_earlier_day_are_not_matched() {
        let mut book = SameDayBook::default();
        book.match_fill(0, SCRIP, DAY, Side::Buy, 1_000.0, 100.0);
        let sell = book.match_fill(1, SCRIP, DAY + 1, Side::Sell, 1_100.0, 100.0);
        assert_eq!(sell, Matched::default());
    }
}
