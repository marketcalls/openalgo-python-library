//! What a trade showed while it was open: its maximum adverse and favourable excursions
//! (MAE and MFE), and what slippage cost its fills. The trade book (`output::TradeBook`)
//! keeps an `Excursion` for each open trade and writes it on the trade's record when the
//! trade closes; a live session saves it with the open trade (`venue_state`). Backtest
//! and live share this one definition.
//!
//! # The running P&L
//!
//! At a price p, a trade's running P&L is its P&L realized so far (its partial exits at
//! average cost, each rounded to the paisa as the ledger posts it) plus the position
//! still open valued at p: realized + units x (p - average price) x multiplier, with
//! signed units, so long and short alike, before charges. The MAE is the lowest running
//! P&L the trade showed and the MFE the highest, rupees to the paisa. The running P&L at
//! the entry fill is 0, so the MAE is at most 0 and the MFE at least 0, and the trade's
//! gross P&L lies between them.
//!
//! # Which prices count
//!
//! Only a price known to have traded while the trade was open, with the position known
//! at that moment, so no excursion is claimed that the trade may not have seen:
//!
//! | Where | Prices, with the position |
//! |---|---|
//! | a slice the trade holds from its open with no fill of the trade inside its bar | its high and low, held (step 5) |
//! | a slice with a fill of the trade inside its bar | its close, after the slice's fills (step 5) |
//! | a fill at the open (`FillAt::Open`) | the open, before the fill when the trade was open |
//! | the first fill of the trade inside a slice's bar (`FillAt::Level`) | the open, before the fill when the trade was open |
//! | every fill (entries, adds, partial and full exits) | its price, before the fill when the trade was open, and after it |
//! | a tick (`push_tick`, a live session) | its price, held, unless a working order on the trade's closing side could have filled by then |
//!
//! So a trade entered at the next open sees its whole opening slice, and one entered by
//! a stop or a limit inside a slice only its fill price and the slice's close; an exit at
//! an open sees that open and its fill, one inside a slice (a stop, a target, a trail, a
//! timed exit) the slice's open and its fill; an entry and an exit inside one slice see
//! their two fill prices only. A fill at the close (the square-off, an expiry) comes
//! after step 5 and adds its price. The bars are the engine's base bars (1-minute bars on
//! a 1-minute base, finer than the strategy's interval).
//!
//! A live session hands over a slice's bar after the fills recorded in it, so its fills
//! see no open, and its ticks carry the prices in between. A venue reports a fill after
//! the ticks that made it: a tick through a broker-held stop comes while the session
//! still holds the position the stop has closed. So a tick does not count while an order
//! on the trade's closing side may have filled at or before its price (`could_have_closed`:
//! a MARKET exit sent, a limit the price reached, a stop whose trigger it reached or that
//! has triggered); the fill, once reported, counts at its own price.
//!
//! # Slippage
//!
//! A fill the kernel prices records what slippage cost it: the fill price against the
//! price before slippage (the open, a level, a last close), times units and the
//! multiplier, rupees to the paisa, positive when the fill was worse (dearer for a buy,
//! cheaper for a sell); 0 for a resting limit, which takes none. A fill the venue priced
//! (every fill of a live session) has none modelled: `None`, NaN in the results, never 0.
//! Slippage is already in the fill price and so in the gross and net P&L: it is shown,
//! never deducted again.

use crate::exec::{FillAt, InstrumentId, Order, OrderStatus, OrderType};
use crate::model::Side;
use crate::portfolio::{round_to_paise, Position, PositionKey};
use crate::rms::BarPrices;

use super::clock::SliceInfo;
use super::fills::FillRequest;
use super::kernel::Engine;

/// A position at a moment: signed units and the average price in rupees (0 when flat).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Holding {
    pub(super) units: f64,
    pub(super) average_rupees: f64,
}

impl Holding {
    fn of(position: &Position) -> Holding {
        Holding {
            units: position.quantity_units,
            average_rupees: position.average_price_rupees,
        }
    }
}

/// What a fill showed besides itself: the position before and after it, the
/// instrument's multiplier, the open of its slice when the slice's bar is known, and
/// what slippage cost it (`None` when the venue priced it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct FillView {
    pub(super) before: Holding,
    pub(super) after: Holding,
    pub(super) multiplier: f64,
    pub(super) slice_open_rupees: Option<f64>,
    pub(super) slippage_rupees: Option<f64>,
}

/// The running P&L of a trade at `price` (module docs), rupees.
fn running(realized_rupees: f64, held: Holding, multiplier: f64, price_rupees: f64) -> f64 {
    realized_rupees + held.units * (price_rupees - held.average_rupees) * multiplier
}

/// `rupees` to the paisa, never -0.0: a tiny loss that rounds to nothing reads 0.
pub(super) fn paise(rupees: f64) -> f64 {
    // IEEE 754: -0.0 + 0.0 is +0.0.
    round_to_paise(rupees) + 0.0
}

/// What slippage cost a fill of `units` on `side` priced at `price_rupees` from
/// `reference_rupees`, rupees to the paisa: positive when the fill was worse.
pub(super) fn slippage_cost(
    side: Side,
    reference_rupees: f64,
    price_rupees: f64,
    units: f64,
    multiplier: f64,
) -> f64 {
    let worse_by = match side {
        Side::Buy => price_rupees - reference_rupees,
        Side::Sell => reference_rupees - price_rupees,
    };
    paise(worse_by * units * multiplier)
}

/// The excursion of an open trade so far (module docs): the lowest and highest running
/// P&L seen, unrounded rupees, and the last slice with a fill of the trade inside its
/// bar, whose high and low do not count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Excursion {
    pub(super) worst_rupees: f64,
    pub(super) best_rupees: f64,
    pub(super) inside_fill_slice: Option<u32>,
}

impl Default for Excursion {
    /// A trade at its entry: a running P&L of 0.
    fn default() -> Excursion {
        Excursion {
            worst_rupees: 0.0,
            best_rupees: 0.0,
            inside_fill_slice: None,
        }
    }
}

impl Excursion {
    fn see(&mut self, running_rupees: f64) {
        if running_rupees < self.worst_rupees {
            self.worst_rupees = running_rupees;
        }
        if running_rupees > self.best_rupees {
            self.best_rupees = running_rupees;
        }
    }

    /// A fill of the trade in slice `slice`, priced at `price_rupees` at `at` of the bar;
    /// `opened` when it opened the trade; `realized` the trade's realized P&L before and
    /// after it, rupees.
    pub(super) fn see_fill(
        &mut self,
        (slice, at): (u32, FillAt),
        price_rupees: f64,
        view: &FillView,
        opened: bool,
        realized: (f64, f64),
    ) {
        let (before, after) = realized;
        let open_counts = match at {
            FillAt::Open => true,
            FillAt::Level => self.inside_fill_slice != Some(slice),
            FillAt::Close => false,
        };
        if !opened {
            if let (true, Some(open)) = (open_counts, view.slice_open_rupees) {
                self.see(running(before, view.before, view.multiplier, open));
            }
            self.see(running(before, view.before, view.multiplier, price_rupees));
        }
        self.see(running(after, view.after, view.multiplier, price_rupees));
        if at == FillAt::Level {
            self.inside_fill_slice = Some(slice);
        }
    }

    /// Step 5 of slice `slice`, whose bar is `bar`, with the trade open and `held`: its
    /// high and low, or only its close when a fill of the trade was inside the bar.
    pub(super) fn see_bar(
        &mut self,
        slice: u32,
        bar: &BarPrices,
        (held, multiplier): (Holding, f64),
        realized_rupees: f64,
    ) {
        if self.inside_fill_slice == Some(slice) {
            self.see(running(realized_rupees, held, multiplier, bar.close));
            return;
        }
        self.see(running(realized_rupees, held, multiplier, bar.high));
        self.see(running(realized_rupees, held, multiplier, bar.low));
    }

    /// A price that traded while the trade was open and `held` (a tick).
    pub(super) fn see_price(
        &mut self,
        price_rupees: f64,
        (held, multiplier): (Holding, f64),
        realized_rupees: f64,
    ) {
        self.see(running(realized_rupees, held, multiplier, price_rupees));
    }

    /// The MAE and the MFE, rupees to the paisa.
    pub(super) fn rounded(&self) -> (f64, f64) {
        (paise(self.worst_rupees), paise(self.best_rupees))
    }
}

/// A trade's slippage with one more fill's: the sum, or `None` once a fill has none.
pub(super) fn add_slippage(total: Option<f64>, fill: Option<f64>) -> Option<f64> {
    Some(total? + fill?)
}

/// Whether the working `order` may have closed part of a position that is long (`long`)
/// or short by the time the market traded at `price_rupees` (module docs): an order on
/// the closing side that is MARKET, a limit the price reached, or a stop whose trigger the
/// price reached or that has triggered already (a stop-limit is counted as soon as its
/// trigger is reached, since the price may have passed its limit on the way).
fn could_have_closed(order: &Order, long: bool, price_rupees: f64) -> bool {
    let closing_side = if long { Side::Sell } else { Side::Buy };
    if order.request.side != closing_side {
        return false;
    }
    let triggered = matches!(
        order.status,
        OrderStatus::Triggered | OrderStatus::PartiallyFilled
    );
    let reached_limit = |limit: f64| {
        if long {
            price_rupees >= limit
        } else {
            price_rupees <= limit
        }
    };
    let reached_trigger = |trigger: f64| {
        if long {
            price_rupees <= trigger
        } else {
            price_rupees >= trigger
        }
    };
    match order.request.order_type {
        OrderType::Market => true,
        OrderType::Limit { limit_rupees } => reached_limit(limit_rupees),
        OrderType::StopMarket { trigger_rupees } | OrderType::StopLimit { trigger_rupees, .. } => {
            triggered || reached_trigger(trigger_rupees)
        }
    }
}

impl Engine {
    /// The position of `key` as it stands (flat when it was never opened).
    pub(super) fn holding(&self, key: PositionKey) -> Holding {
        self.portfolio
            .position(key)
            .map_or_else(Holding::default, Holding::of)
    }

    /// What a fill of `request` at `price_rupees` showed, once it is applied to the
    /// position, which stood at `before` (`FillView`).
    pub(super) fn fill_view(
        &self,
        (slice, request): (usize, &FillRequest),
        before: Holding,
        price_rupees: f64,
        multiplier: f64,
    ) -> FillView {
        let slippage_rupees = (!self.is_external()).then(|| {
            let reference = request.reference_rupees;
            slippage_cost(
                request.side,
                reference,
                price_rupees,
                request.units,
                multiplier,
            )
        });
        FillView {
            before,
            after: self.holding(request.key),
            multiplier,
            slice_open_rupees: self
                .market
                .bar(request.key.instrument, slice)
                .map(|bar| bar.open),
            slippage_rupees,
        }
    }

    /// Step 5: every open trade sees its slice's bar (`Excursion::see_bar`); a trade on
    /// an instrument with no bar in the slice sees nothing.
    pub(super) fn see_slice_bars(&mut self, slice: &SliceInfo) {
        let index = u32::try_from(slice.index).unwrap_or(u32::MAX);
        for trade in &mut self.record.trades.open {
            let Some(bar) = self.market.bar(trade.key.instrument, slice.index) else {
                continue;
            };
            let Some(position) = self.portfolio.position(trade.key) else {
                continue;
            };
            let held = (Holding::of(position), position.multiplier);
            trade.excursion.see_bar(index, &bar, held, trade.gross);
        }
    }

    /// A tick of `instrument` at `price_rupees`: every open trade of the instrument sees
    /// it (`Excursion::see_price`), but one with a working order on its closing side that
    /// may have filled by then (`could_have_closed`).
    pub(super) fn see_tick(&mut self, instrument: InstrumentId, price_rupees: f64) {
        for trade in &mut self.record.trades.open {
            if trade.key.instrument != instrument {
                continue;
            }
            let Some(position) = self.portfolio.position(trade.key) else {
                continue;
            };
            let (key, long) = (trade.key, position.quantity_units > 0.0);
            let closing = self.book.working().any(|order| {
                let request = &order.request;
                request.instrument == key.instrument
                    && request.product == key.product
                    && could_have_closed(order, long, price_rupees)
            });
            if closing {
                continue;
            }
            let held = (Holding::of(position), position.multiplier);
            trade.excursion.see_price(price_rupees, held, trade.gross);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LONG_100_AT_500: Holding = Holding {
        units: 100.0,
        average_rupees: 500.0,
    };
    const FLAT: Holding = Holding {
        units: 0.0,
        average_rupees: 0.0,
    };

    fn view(before: Holding, after: Holding, open: Option<f64>) -> FillView {
        FillView {
            before,
            after,
            multiplier: 1.0,
            slice_open_rupees: open,
            slippage_rupees: Some(0.0),
        }
    }

    fn bar(open: f64, high: f64, low: f64, close: f64) -> BarPrices {
        BarPrices {
            open,
            high,
            low,
            close,
        }
    }

    #[test]
    fn slippage_is_positive_when_the_fill_is_worse_than_its_reference() {
        // (side, reference, price, units, multiplier, rupees): a buy dearer by two ticks
        // of 0.05 on 100 units costs 10.00; a sell cheaper by 0.10 on 75 units of a
        // futures contract with multiplier 1 costs 7.50; a resting fill costs 0; a sell
        // better than its reference (a marketable limit) is negative.
        let table = [
            (Side::Buy, 500.0, 500.1, 100.0, 1.0, 10.0),
            (Side::Sell, 24_000.0, 23_999.9, 75.0, 1.0, 7.5),
            (Side::Buy, 99.5, 99.5, 10.0, 1.0, 0.0),
            (Side::Sell, 100.0, 100.05, 10.0, 1.0, -0.5),
            (Side::Buy, 72_000.0, 72_001.0, 1.0, 100.0, 100.0),
        ];
        for (side, reference, price, units, multiplier, rupees) in table {
            assert_eq!(
                slippage_cost(side, reference, price, units, multiplier),
                rupees,
                "{side:?} {reference} -> {price}"
            );
        }
    }

    #[test]
    fn a_resting_fill_has_no_negative_zero_slippage() {
        let cost = slippage_cost(Side::Sell, 100.0, 100.0, 7.0, 1.0);
        assert!(cost == 0.0 && cost.is_sign_positive());
    }

    #[test]
    fn a_trade_with_a_fill_without_slippage_has_none() {
        assert_eq!(add_slippage(Some(1.5), Some(2.25)), Some(3.75));
        assert_eq!(add_slippage(Some(1.5), None), None);
        assert_eq!(add_slippage(None, Some(2.0)), None);
    }

    #[test]
    fn a_slice_held_from_its_open_counts_its_high_and_low() {
        // Long 100 at 500.00: a bar 501, 506, 497, 503 shows -300.00 at the low and
        // +600.00 at the high.
        let mut seen = Excursion::default();
        seen.see_bar(
            4,
            &bar(501.0, 506.0, 497.0, 503.0),
            (LONG_100_AT_500, 1.0),
            0.0,
        );
        assert_eq!(seen.rounded(), (-300.0, 600.0));
    }

    #[test]
    fn an_entry_inside_a_slice_counts_its_fill_and_the_close_not_the_range() {
        // A buy stop fills inside slice 4 at 500.00 (the bar's open is 498.00, its low
        // 495.00, its high 507.00 and its close 503.00): before the fill the trade did
        // not exist, so neither the open nor the low counts; the close does, +300.00.
        let mut seen = Excursion::default();
        let entry = view(FLAT, LONG_100_AT_500, Some(498.0));
        seen.see_fill((4, FillAt::Level), 500.0, &entry, true, (0.0, 0.0));
        seen.see_bar(
            4,
            &bar(498.0, 507.0, 495.0, 503.0),
            (LONG_100_AT_500, 1.0),
            0.0,
        );
        assert_eq!(seen.rounded(), (0.0, 300.0));
    }

    #[test]
    fn an_exit_inside_a_slice_counts_the_open_and_its_fill() {
        // Long 100 at 500.00 into slice 9, which opens at 502.00 and stops out inside at
        // 495.00: +200.00 at the open, -500.00 at the stop.
        let mut seen = Excursion::default();
        let exit = view(LONG_100_AT_500, FLAT, Some(502.0));
        seen.see_fill((9, FillAt::Level), 495.0, &exit, false, (0.0, -500.0));
        assert_eq!(seen.rounded(), (-500.0, 200.0));
    }

    #[test]
    fn a_second_fill_inside_the_same_slice_does_not_count_the_open() {
        // An entry inside slice 3 at 500.00, the bar opening at 490.00, then a sell
        // inside the same slice at 504.00: only the two fills count, 0 and +400.00.
        let mut seen = Excursion::default();
        let entry = view(FLAT, LONG_100_AT_500, Some(490.0));
        seen.see_fill((3, FillAt::Level), 500.0, &entry, true, (0.0, 0.0));
        let exit = view(LONG_100_AT_500, FLAT, Some(490.0));
        seen.see_fill((3, FillAt::Level), 504.0, &exit, false, (0.0, 400.0));
        assert_eq!(seen.rounded(), (0.0, 400.0));
    }

    #[test]
    fn a_short_gains_as_the_price_falls() {
        // Short 50 at 200.00 through a bar 199, 204, 190, 195: -200.00 at the high,
        // +500.00 at the low.
        let short = Holding {
            units: -50.0,
            average_rupees: 200.0,
        };
        let mut seen = Excursion::default();
        seen.see_bar(1, &bar(199.0, 204.0, 190.0, 195.0), (short, 1.0), 0.0);
        assert_eq!(seen.rounded(), (-200.0, 500.0));
    }

    #[test]
    fn a_partial_exit_carries_its_realized_pnl() {
        // 100 long at 500.00, 40 sold at 510.00 (+400.00 realized), 60 left: a later low
        // of 490.00 shows 400 + 60 x (490 - 500) = -200.00.
        let mut seen = Excursion::default();
        let left = Holding {
            units: 60.0,
            average_rupees: 500.0,
        };
        seen.see_price(490.0, (left, 1.0), 400.0);
        assert_eq!(seen.rounded(), (-200.0, 0.0));
    }

    fn working(side: Side, order_type: OrderType, status: OrderStatus) -> Order {
        use crate::exec::{OrderId, OrderRequest, QuantitySpec};
        Order {
            id: OrderId(2),
            request: OrderRequest {
                order_type,
                ..OrderRequest::market(InstrumentId(0), side, QuantitySpec::Units(100.0))
            },
            status,
            decided_at: 0,
            good_until: 0,
            quantity_units: 100.0,
            filled_units: 0.0,
            average_fill_rupees: 0.0,
            reject_code: None,
        }
    }

    #[test]
    fn a_tick_is_not_known_with_the_position_once_a_closing_order_may_have_filled() {
        // A long's sell stop at 497.10, sell target limit at 503.00, a sell MARKET exit,
        // and a buy (not on the closing side); a short's buy stop at 503.00.
        let stop = OrderType::StopMarket {
            trigger_rupees: 497.1,
        };
        let stop_limit = OrderType::StopLimit {
            trigger_rupees: 497.1,
            limit_rupees: 496.0,
        };
        let target = OrderType::Limit {
            limit_rupees: 503.0,
        };
        let short_stop = OrderType::StopMarket {
            trigger_rupees: 503.0,
        };
        let (open, fired) = (OrderStatus::Accepted, OrderStatus::Triggered);
        let table = [
            (
                "stop, tick above it",
                Side::Sell,
                stop,
                open,
                true,
                498.0,
                false,
            ),
            (
                "stop, tick at it",
                Side::Sell,
                stop,
                open,
                true,
                497.1,
                true,
            ),
            (
                "stop, tick through it",
                Side::Sell,
                stop,
                open,
                true,
                496.0,
                true,
            ),
            ("stop triggered", Side::Sell, stop, fired, true, 499.0, true),
            (
                "stop-limit at its trigger",
                Side::Sell,
                stop_limit,
                open,
                true,
                497.0,
                true,
            ),
            (
                "target not reached",
                Side::Sell,
                target,
                open,
                true,
                502.9,
                false,
            ),
            (
                "target reached",
                Side::Sell,
                target,
                open,
                true,
                503.0,
                true,
            ),
            (
                "market exit sent",
                Side::Sell,
                OrderType::Market,
                open,
                true,
                500.0,
                true,
            ),
            (
                "a buy while long",
                Side::Buy,
                OrderType::Market,
                open,
                true,
                500.0,
                false,
            ),
            (
                "short stop below it",
                Side::Buy,
                short_stop,
                open,
                false,
                502.0,
                false,
            ),
            (
                "short stop through it",
                Side::Buy,
                short_stop,
                open,
                false,
                503.5,
                true,
            ),
        ];
        for (what, side, order_type, status, long, price, closed) in table {
            let order = working(side, order_type, status);
            assert_eq!(could_have_closed(&order, long, price), closed, "{what}");
        }
    }

    #[test]
    fn the_multiplier_scales_the_running_pnl() {
        // One unit of a contract with multiplier 100 bought at 72,000.00: a tick at
        // 71,950.00 is -5,000.00.
        let held = Holding {
            units: 1.0,
            average_rupees: 72_000.0,
        };
        let mut seen = Excursion::default();
        seen.see_price(71_950.0, (held, 100.0), 0.0);
        assert_eq!(seen.rounded(), (-5_000.0, 0.0));
    }
}
