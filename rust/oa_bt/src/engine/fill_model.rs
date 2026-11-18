//! Fill models (design 17, section 2.15; `fill="next_close"` and the others of
//! `FillTiming::Model`): the strategy's orders that would fill at the next base bar's
//! open fill at another price of that bar, or at its average with the bar after it, so
//! the liquidity check can ask whether a strategy survives worse fill prices.
//!
//! | Model | Reference price of the fill bar |
//! |---|---|
//! | `next_close` | its close |
//! | `next_mid` | (high + low) / 2 |
//! | `next_ohlc4` | (open + high + low + close) / 4 |
//! | `..._2` | that price of the fill bar and of the instrument's next base bar, averaged: on an intraday base the next bar of the same session, on a daily base the next session's; the fill bar alone when there is none |
//!
//! The reference is a price of the whole bar, so the queued orders fill in step 3, after
//! the legs' levels, the GTTs and the resting orders were tested on that bar, as an
//! order filled inside the bar (`FillAt::Level`): a stop hit inside the bar acts before
//! them, and a position they open is first tested against its levels in the next slice.
//! The fill takes the run's slippage and is rounded to the tick away from the trader, as
//! every market fill is. The engine's own exits at an open (the risk exits, the time
//! exit, the square-off, the exit after an unfilled GTT child), the futures roll and the
//! fills at a level keep their prices: a fill model prices what the strategy decides.
//! The rebalance buys fill after the rebalance's sales, as at the open.
//!
//! The two-bar models read the bar after the fill bar: a pessimistic price model for a
//! backtest, never a live rule.

use crate::exec::{BarPoint, FillAt, FillPricing, InstrumentId};
use crate::market::calendar::BaseInterval;
use crate::rms::BarPrices;

use super::clock::SliceInfo;
use super::config::{BarPrice, FillModel, FillTiming};
use super::error::KernelError;
use super::event::Step;
use super::kernel::Engine;
use super::order_fill::OrderFill;

/// The price `price` of `bar`, rupees (module docs).
fn bar_price(bar: &BarPrices, price: BarPrice) -> f64 {
    match price {
        BarPrice::Close => bar.close,
        BarPrice::Mid => (bar.high + bar.low) / 2.0,
        BarPrice::Ohlc4 => (bar.open + bar.high + bar.low + bar.close) / 4.0,
    }
}

impl OrderFill {
    /// A queued order filled at a model's reference price of its bar, in step 3.
    pub(super) fn at_model(reference_rupees: f64) -> OrderFill {
        OrderFill {
            step: Step::Levels,
            point: BarPoint {
                at: FillAt::Level,
                price_rupees: reference_rupees,
            },
            pricing: FillPricing::Market,
            gtt: None,
        }
    }
}

impl Engine {
    /// The run's fill model, `None` for a run that fills at the next open.
    pub(super) fn fill_model(&self) -> Option<FillModel> {
        match self.config.fill {
            FillTiming::Model(model) => Some(model),
            _ => None,
        }
    }

    /// Step 3, after the resting orders: every queued order of each instrument with a bar
    /// in the slice at the model's reference price, then the rebalance buys (module
    /// docs). Nothing in a run that fills at the next open, or in a live session.
    ///
    /// Errors: those of `fill_order`.
    pub(super) fn fill_queued_at_model(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        if self.fill_model().is_none() || self.is_external() {
            return Ok(());
        }
        for index in 0..self.instruments.len() {
            let instrument = InstrumentId(index as u32);
            let Some(how) = self.queued_fill(instrument, slice) else {
                continue;
            };
            for id in self.book.queued_for(instrument) {
                self.fill_queued_how(slice, id, how)?;
            }
        }
        self.fill_rebalance_buys(slice)
    }

    /// How a queued order of `instrument` fills in `slice`: at the open, or at the run's
    /// model price (module docs); `None` when the instrument has no bar in the slice.
    pub(super) fn queued_fill(
        &self,
        instrument: InstrumentId,
        slice: &SliceInfo,
    ) -> Option<OrderFill> {
        let bar = self.market.bar(instrument, slice.index)?;
        let Some(model) = self.fill_model() else {
            return Some(OrderFill::at_open(bar.open));
        };
        let here = bar_price(&bar, model.price);
        let reference = match self.bar_after(instrument, slice).filter(|_| model.two_bars) {
            Some(next) => (here + bar_price(&next, model.price)) / 2.0,
            None => here,
        };
        Some(OrderFill::at_model(reference))
    }

    /// The instrument's next base bar after `slice`'s: on an intraday base within the
    /// same session, on a daily base the next one it has (module docs).
    fn bar_after(&self, instrument: InstrumentId, slice: &SliceInfo) -> Option<BarPrices> {
        let daily = self.config.base == BaseInterval::Daily;
        (slice.index + 1..self.timeline.len())
            .map_while(|index| {
                let next = self.timeline.slice(index)?;
                (daily || next.day == slice.day).then_some(index)
            })
            .find_map(|index| self.market.bar(instrument, index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BAR: BarPrices = BarPrices {
        open: 100.0,
        high: 104.0,
        low: 98.0,
        close: 102.0,
    };

    #[test]
    fn each_model_reads_its_price_of_the_bar() {
        // Close 102; mid (104 + 98) / 2 = 101; average (100 + 104 + 98 + 102) / 4 = 101.
        assert_eq!(bar_price(&BAR, BarPrice::Close), 102.0);
        assert_eq!(bar_price(&BAR, BarPrice::Mid), 101.0);
        assert_eq!(bar_price(&BAR, BarPrice::Ohlc4), 101.0);
    }

    #[test]
    fn a_model_fill_is_inside_the_bar_in_step_3_at_market() {
        let how = OrderFill::at_model(101.0);
        assert_eq!(how.step, Step::Levels);
        assert_eq!(how.point.at, FillAt::Level);
        assert_eq!(how.point.price_rupees, 101.0);
        assert_eq!(how.pricing, FillPricing::Market);
    }
}
