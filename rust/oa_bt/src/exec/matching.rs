//! Resting prices against one base bar: where a LIMIT fills, where a stop or GTT trigger
//! is reached, and what a limit placed at a trigger does in the rest of the bar
//! (03-architecture 4.2 step 3, 04-execution 4.1, 4.2 and 4.4).
//!
//! The rule of every test is the one of the leg levels: the open comes first, so a price
//! already crossed at the open is taken at the open; otherwise a price the bar's range
//! reaches is taken at that price. A touch is enough. Comparisons are on whole price
//! units (`price::whole_units`), so binary noise never decides a touch.
//!
//! A bar's prices carry no order in time beyond "the open comes first". A limit placed
//! when a trigger fires is therefore filled at the price that fired it when it is
//! marketable there (decision GTT-5), and otherwise rests through the rest of the bar
//! and fills at its limit when the bar trades back to it (decision GTT-7); the rest of
//! the bar is taken to be its whole range after the open.
//!
//! These functions only say where a price is reached. Slippage, the position and the
//! order's state are the engine's (`fill::FillPricing`, `engine::resting`,
//! `engine::gtt_fire`).

use crate::model::Side;
use crate::rms::leg::BarPrices;

use super::fill::{FillAt, FillPricing};
use super::price::whole_units;

/// From which side the price reaches a trigger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TriggerDirection {
    /// The trigger fires when the price rises to it or above: a buy stop, a long's
    /// target, a short's stop.
    Rising,
    /// The trigger fires when the price falls to it or below: a sell stop, a long's
    /// stop, a short's target.
    Falling,
}

impl TriggerDirection {
    /// The direction of a stop order's trigger: a buy stop waits for a rise, a sell stop
    /// for a fall.
    pub fn of_stop(side: Side) -> TriggerDirection {
        match side {
            Side::Buy => TriggerDirection::Rising,
            Side::Sell => TriggerDirection::Falling,
        }
    }

    /// The direction of a trigger placed while the price is `price_rupees`: one at or
    /// above the price waits for a rise, one below it for a fall.
    pub fn toward(trigger_rupees: f64, price_rupees: f64) -> TriggerDirection {
        if whole_units(trigger_rupees) >= whole_units(price_rupees) {
            TriggerDirection::Rising
        } else {
            TriggerDirection::Falling
        }
    }

    /// True when `price_rupees` is at or beyond `trigger_rupees` in this direction.
    pub fn reached(self, price_rupees: f64, trigger_rupees: f64) -> bool {
        match self {
            TriggerDirection::Rising => whole_units(price_rupees) >= whole_units(trigger_rupees),
            TriggerDirection::Falling => whole_units(price_rupees) <= whole_units(trigger_rupees),
        }
    }
}

/// Where in a bar a price was taken, and at what price before slippage (rupees).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BarPoint {
    /// The open, or a level inside the range.
    pub at: FillAt,
    /// The price, rupees.
    pub price_rupees: f64,
}

impl BarPoint {
    fn open(bar: &BarPrices) -> BarPoint {
        BarPoint {
            at: FillAt::Open,
            price_rupees: bar.open,
        }
    }

    fn level(price_rupees: f64) -> BarPoint {
        BarPoint {
            at: FillAt::Level,
            price_rupees,
        }
    }
}

/// What a limit placed when a trigger fired did in the rest of its bar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AfterTrigger {
    /// Marketable when it was placed: it fills at the price that fired it (GTT-5).
    AtFire(BarPoint),
    /// The bar had opened beyond the limit and later traded back to it: it fills at
    /// the limit (GTT-7).
    AtLimit(BarPoint),
    /// It did not fill through the bar; the best price the bar offered after the
    /// trigger, rupees (the high for a sell, the low for a buy).
    Unfilled { best_after_trigger_rupees: f64 },
}

impl AfterTrigger {
    /// Where the limit filled and how that fill is priced, `None` when it did not fill.
    /// Marketable when placed, it takes slippage like a MARKET order but never past
    /// its limit (decision GTT-5 reading A); filled at its limit after the bar came back
    /// to it, it is a resting limit and takes none (decision GTT-7 reading A).
    pub fn fill(self, limit_rupees: f64) -> Option<(BarPoint, FillPricing)> {
        match self {
            AfterTrigger::AtFire(point) => {
                Some((point, FillPricing::MarketableLimit { limit_rupees }))
            }
            AfterTrigger::AtLimit(point) => Some((point, FillPricing::Resting)),
            AfterTrigger::Unfilled { .. } => None,
        }
    }

    /// The best price the bar offered after the trigger when the limit did not fill,
    /// rupees; `None` when it filled.
    pub fn best_after_trigger(self) -> Option<f64> {
        match self {
            AfterTrigger::Unfilled {
                best_after_trigger_rupees,
            } => Some(best_after_trigger_rupees),
            _ => None,
        }
    }
}

/// True when a limit order on `side` at `limit_rupees` accepts `price_rupees`: at or
/// below the limit for a buy, at or above it for a sell.
pub fn accepts(side: Side, limit_rupees: f64, price_rupees: f64) -> bool {
    match side {
        Side::Buy => whole_units(price_rupees) <= whole_units(limit_rupees),
        Side::Sell => whole_units(price_rupees) >= whole_units(limit_rupees),
    }
}

/// The best price a bar offers an order on `side`: its low for a buy, its high for a
/// sell.
fn best_for(side: Side, bar: &BarPrices) -> f64 {
    match side {
        Side::Buy => bar.low,
        Side::Sell => bar.high,
    }
}

/// Where a trigger is reached in a bar: at the open when the bar opens at or beyond it,
/// at the trigger when the range reaches it, `None` when the bar does not reach it.
pub fn trigger_point(
    direction: TriggerDirection,
    trigger_rupees: f64,
    bar: &BarPrices,
) -> Option<BarPoint> {
    if direction.reached(bar.open, trigger_rupees) {
        return Some(BarPoint::open(bar));
    }
    let extreme = match direction {
        TriggerDirection::Rising => bar.high,
        TriggerDirection::Falling => bar.low,
    };
    direction
        .reached(extreme, trigger_rupees)
        .then(|| BarPoint::level(trigger_rupees))
}

/// Where a LIMIT that rests through the whole bar fills: at the open when the open is at
/// the limit or better, at the limit when the range reaches it, `None` otherwise.
pub fn limit_point(side: Side, limit_rupees: f64, bar: &BarPrices) -> Option<BarPoint> {
    if accepts(side, limit_rupees, bar.open) {
        return Some(BarPoint::open(bar));
    }
    accepts(side, limit_rupees, best_for(side, bar)).then(|| BarPoint::level(limit_rupees))
}

/// A limit on `side` at `limit_rupees` placed when a trigger fired at `fired` in `bar`
/// (a GTT child, a triggered SL order).
///
/// Marketable at the price that fired it, it fills there (decision GTT-5): at the
/// trigger inside the range, or at the open when the bar opened beyond the trigger but
/// not beyond the limit. Otherwise the bar opened beyond the limit, and the limit rests
/// through the rest of the bar like any LIMIT: it fills at the limit when the bar
/// trades back to it, a touch being enough (decision GTT-7), and is unfilled otherwise.
pub fn limit_after_trigger(
    side: Side,
    limit_rupees: f64,
    fired: BarPoint,
    bar: &BarPrices,
) -> AfterTrigger {
    if accepts(side, limit_rupees, fired.price_rupees) {
        return AfterTrigger::AtFire(fired);
    }
    let best = best_for(side, bar);
    if accepts(side, limit_rupees, best) {
        AfterTrigger::AtLimit(BarPoint::level(limit_rupees))
    } else {
        AfterTrigger::Unfilled {
            best_after_trigger_rupees: best,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bar of (open, high, low, close).
    fn bar(open: f64, high: f64, low: f64, close: f64) -> BarPrices {
        BarPrices {
            open,
            high,
            low,
            close,
        }
    }

    fn at_open(price: f64) -> Option<BarPoint> {
        Some(BarPoint {
            at: FillAt::Open,
            price_rupees: price,
        })
    }

    fn at_level(price: f64) -> Option<BarPoint> {
        Some(BarPoint {
            at: FillAt::Level,
            price_rupees: price,
        })
    }

    // Every LIMIT case runs on one bar: open 100.00, high 102.00, low 98.00.
    const OPEN: f64 = 100.0;

    fn the_bar() -> BarPrices {
        bar(OPEN, 102.0, 98.0, 101.0)
    }

    #[test]
    fn a_limit_buy_at_or_above_the_open_fills_at_the_open() {
        assert_eq!(limit_point(Side::Buy, 100.0, &the_bar()), at_open(100.0));
        assert_eq!(limit_point(Side::Buy, 101.0, &the_bar()), at_open(100.0));
    }

    #[test]
    fn a_limit_buy_below_the_open_inside_the_range_fills_at_its_limit() {
        assert_eq!(limit_point(Side::Buy, 99.0, &the_bar()), at_level(99.0));
        assert_eq!(limit_point(Side::Buy, 98.0, &the_bar()), at_level(98.0));
    }

    #[test]
    fn a_limit_buy_below_the_range_does_not_fill() {
        assert_eq!(limit_point(Side::Buy, 97.95, &the_bar()), None);
    }

    #[test]
    fn a_limit_sell_at_or_below_the_open_fills_at_the_open() {
        assert_eq!(limit_point(Side::Sell, 100.0, &the_bar()), at_open(100.0));
        assert_eq!(limit_point(Side::Sell, 99.0, &the_bar()), at_open(100.0));
    }

    #[test]
    fn a_limit_sell_above_the_open_inside_the_range_fills_at_its_limit() {
        assert_eq!(limit_point(Side::Sell, 101.5, &the_bar()), at_level(101.5));
        assert_eq!(limit_point(Side::Sell, 102.0, &the_bar()), at_level(102.0));
    }

    #[test]
    fn a_limit_sell_above_the_range_does_not_fill() {
        assert_eq!(limit_point(Side::Sell, 102.05, &the_bar()), None);
    }

    #[test]
    fn a_touch_counts_whatever_the_binary_noise() {
        // 0.1 + 0.2 is 0.30000000000000004 in binary; on price units it is 0.30.
        let low_bar = bar(0.4, 0.4, 0.1 + 0.2, 0.35);
        assert_eq!(limit_point(Side::Buy, 0.3, &low_bar), at_level(0.3));
    }

    #[test]
    fn a_buy_stop_is_reached_at_the_open_on_a_gap_up_and_at_its_trigger_inside_the_range() {
        let rising = TriggerDirection::of_stop(Side::Buy);
        assert_eq!(rising, TriggerDirection::Rising);
        assert_eq!(trigger_point(rising, 99.0, &the_bar()), at_open(100.0));
        assert_eq!(trigger_point(rising, 101.0, &the_bar()), at_level(101.0));
        assert_eq!(trigger_point(rising, 102.5, &the_bar()), None);
    }

    #[test]
    fn a_sell_stop_is_reached_at_the_open_on_a_gap_down_and_at_its_trigger_inside_the_range() {
        let falling = TriggerDirection::of_stop(Side::Sell);
        assert_eq!(falling, TriggerDirection::Falling);
        assert_eq!(trigger_point(falling, 100.5, &the_bar()), at_open(100.0));
        assert_eq!(trigger_point(falling, 98.5, &the_bar()), at_level(98.5));
        assert_eq!(trigger_point(falling, 97.5, &the_bar()), None);
    }

    #[test]
    fn a_trigger_waits_on_the_side_of_the_price_it_was_placed_at() {
        assert_eq!(
            TriggerDirection::toward(105.0, 100.0),
            TriggerDirection::Rising
        );
        assert_eq!(
            TriggerDirection::toward(95.0, 100.0),
            TriggerDirection::Falling
        );
    }

    #[test]
    fn a_child_marketable_at_its_trigger_fills_at_the_trigger() {
        // Trace 06a, 2025-07-09: stop 1460.00 reached inside the range of (1485.00,
        // 1492.00, 1455.00); the sell child's limit 1445.40 accepts 1460.00 (GTT-5).
        let day = bar(1485.0, 1492.0, 1455.0, 1470.0);
        let fired = BarPoint {
            at: FillAt::Level,
            price_rupees: 1460.0,
        };
        let outcome = limit_after_trigger(Side::Sell, 1445.4, fired, &day);
        assert_eq!(outcome, AfterTrigger::AtFire(fired));
        let pricing = FillPricing::MarketableLimit {
            limit_rupees: 1445.4,
        };
        assert_eq!(outcome.fill(1445.4), Some((fired, pricing)));
    }

    #[test]
    fn a_child_whose_bar_opened_beyond_the_limit_fills_at_the_limit_when_the_bar_comes_back() {
        // Trace 06a, 2025-07-11: open 1410.00 is beyond the stop 1425.00 and the limit
        // 1410.75; the high 1418.00 trades back through the limit (GTT-7).
        let day = bar(1410.0, 1418.0, 1402.0, 1415.0);
        let fired = BarPoint {
            at: FillAt::Open,
            price_rupees: 1410.0,
        };
        let outcome = limit_after_trigger(Side::Sell, 1410.75, fired, &day);
        let expected = BarPoint {
            at: FillAt::Level,
            price_rupees: 1410.75,
        };
        assert_eq!(outcome, AfterTrigger::AtLimit(expected));
        assert_eq!(
            outcome.fill(1410.75),
            Some((expected, FillPricing::Resting))
        );
    }

    #[test]
    fn a_child_whose_bar_never_comes_back_to_the_limit_is_unfilled() {
        // Trace 06d, 2025-07-09: open 1440.00 beyond the limit 1445.40, high 1442.00.
        let day = bar(1440.0, 1442.0, 1420.0, 1425.0);
        let fired = BarPoint {
            at: FillAt::Open,
            price_rupees: 1440.0,
        };
        let outcome = limit_after_trigger(Side::Sell, 1445.4, fired, &day);
        let expected = AfterTrigger::Unfilled {
            best_after_trigger_rupees: 1442.0,
        };
        assert_eq!(outcome, expected);
        assert_eq!(outcome.fill(1445.4), None);
        assert_eq!(outcome.best_after_trigger(), Some(1442.0));
    }

    #[test]
    fn a_child_opened_beyond_the_trigger_but_not_the_limit_fills_at_the_open() {
        // A buy child (a short's stop) with trigger 101.00 and limit 102.05: the bar
        // opens at 101.50, beyond the trigger and inside the limit.
        let day = bar(101.5, 103.0, 101.0, 102.0);
        let fired = BarPoint {
            at: FillAt::Open,
            price_rupees: 101.5,
        };
        let outcome = limit_after_trigger(Side::Buy, 102.05, fired, &day);
        assert_eq!(outcome, AfterTrigger::AtFire(fired));
    }
}
