//! What holds a live session's trade back from a price for its MAE and MFE (`excursion`):
//! the closing orders the venue may have filled without having reported it yet.
//!
//! A venue reports a fill after the prices that made it: a tick through a broker-held
//! stop, or a bar whose low passed it, comes while the session still holds the position
//! the stop has closed, and a fill reported late (fills are recorded when they are
//! received) lands after the bar of the minute it was made in. So a live session's trade
//! is held back once a price reaches a working order on its closing side
//! (`could_have_closed`: a MARKET exit sent, a limit the price reached, a stop whose
//! trigger it reached or that has triggered): that price does not count, and no later
//! tick or bar does, until the venue reports every such order ended (filled whole,
//! cancelled, rejected or expired), whatever the prices do meanwhile, since a bounce may
//! come after the position was closed. The order's fill, once reported, counts at its own
//! price.
//!
//! An order whose modification has been sent and not answered (`VenueDesk::modifying`: a
//! trail or a breakeven moving a broker-held stop, a reprice) may already work on its new
//! terms at the venue, so it is judged by its working terms and by those of the
//! modification, and it is reached when either is. Reached by the new terms alone, it
//! holds the trade back until the venue answers (`Hold::pending_closes`): a confirmation
//! keeps the hold, since the new terms were working; a refusal releases it, since the
//! order kept its old terms throughout and no price judged reached them (the prices it
//! has not judged yet, such as the bar's, are judged by them in their turn). After the
//! answer only the terms then working count.
//!
//! A tick is judged against the orders' terms when it is taken. A bar is handed over
//! after its minute, when the kernel may have moved the broker's stop on the minute's
//! ticks (a trail, a breakeven) after its low or high traded, so a bar is judged only by
//! the prices the trade's ticks of the slice did not carry: its low when below every one
//! of them, its high when above every one, its low and high when it took none. A bar the
//! ticks carried counts unless the trade is held back already.
//!
//! # Why a hold lasts until the orders end
//!
//! Save by the refusal above, a hold is not released when the price comes back, nor by
//! a later price that suggests the trade was still open: a broker's tick feed does not
//! carry every traded price, so a price of a bar that the trade's ticks did not carry may
//! have come after the entry and reached the stop, and releasing the hold could claim an
//! excursion the trade did not see. The cost falls on the safe side: a live trade held
//! back on a bar keeps the excursion of its fills only (besides the prices it saw before
//! the hold) until the venue reports its closing orders ended. A trade entered inside a
//! bar shows it most: a low or high of that bar that traded before the entry, when the
//! trade had no stop, still holds it back, so its live MAE and MFE can read smaller than
//! the prices it went through.
//!
//! A backtest records its fills in steps 2 and 3, before the bar is seen in step 5, and
//! takes no ticks: it holds nothing back.

use crate::exec::{InstrumentId, Order, OrderBook, OrderId, OrderStatus, OrderType};
use crate::model::Side;
use crate::portfolio::PositionKey;
use crate::rms::BarPrices;

use super::clock::SliceInfo;
use super::excursion::Holding;
use super::kernel::Engine;
use super::venue::Replacement;

/// A live session's order book and the modifications of its orders sent and not
/// answered (`VenueDesk::modifying`), as a hold judges a price against them.
pub(super) type Orders<'a> = (&'a OrderBook, &'a [(OrderId, Replacement)]);

/// The lowest and highest ticks a trade took in one slice, rupees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct SliceTicks {
    pub(super) slice: u32,
    pub(super) low_rupees: f64,
    pub(super) high_rupees: f64,
}

/// What holds a live session's trade back (module docs): the orders on its closing side
/// that a price has reached by their working terms and that the venue has not reported
/// ended; those a price has reached only by the terms of their modification the venue
/// has not answered; and the ticks it took in the last slice it took one in. An order is
/// in one list at most.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Hold {
    pub(super) unconfirmed_closes: Vec<OrderId>,
    pub(super) pending_closes: Vec<OrderId>,
    pub(super) ticks: Option<SliceTicks>,
}

impl Hold {
    /// A tick at `price_rupees` taken in slice `slice`.
    fn take_tick(&mut self, slice: u32, price_rupees: f64) {
        self.ticks = Some(match self.ticks {
            Some(seen) if seen.slice == slice => SliceTicks {
                low_rupees: seen.low_rupees.min(price_rupees),
                high_rupees: seen.high_rupees.max(price_rupees),
                ..seen
            },
            _ => SliceTicks {
                slice,
                low_rupees: price_rupees,
                high_rupees: price_rupees,
            },
        });
    }

    /// The prices of `bar`, the bar of slice `slice`, that the trade's ticks of the slice
    /// did not carry (module docs).
    pub(super) fn beyond_ticks(&self, slice: u32, bar: &BarPrices) -> Vec<f64> {
        match self.ticks {
            Some(seen) if seen.slice == slice => [
                (bar.low < seen.low_rupees).then_some(bar.low),
                (bar.high > seen.high_rupees).then_some(bar.high),
            ]
            .into_iter()
            .flatten()
            .collect(),
            _ => vec![bar.low, bar.high],
        }
    }

    /// Whether the trade on `key`, `long` or short, is held back (module docs) once
    /// `prices` are judged against the working orders of the book and the terms of their
    /// modifications not answered (`orders`): the orders it waits on that have ended are
    /// dropped; a working order on its closing side that one of `prices` reaches by its
    /// own terms is added to the unconfirmed closes, one reached only by the terms of its
    /// modification to the pending closes.
    pub(super) fn held_back(
        &mut self,
        (book, modifying): Orders<'_>,
        (key, long): (PositionKey, bool),
        prices: &[f64],
    ) -> bool {
        let works = |id: &OrderId| book.get(*id).is_some_and(|order| order.status.is_working());
        self.unconfirmed_closes.retain(works);
        self.pending_closes.retain(works);
        for order in book.working() {
            let request = &order.request;
            if request.instrument != key.instrument || request.product != key.product {
                continue;
            }
            let reached = |terms: OrderType| {
                prices
                    .iter()
                    .any(|price| could_have_closed(order, terms, long, *price))
            };
            let pending = modifying.iter().find(|(id, _)| *id == order.id);
            if reached(request.order_type) {
                self.pending_closes.retain(|id| *id != order.id);
                add_once(&mut self.unconfirmed_closes, order.id);
            } else if pending.is_some_and(|(_, wanted)| reached(wanted.request.order_type))
                && !self.unconfirmed_closes.contains(&order.id)
            {
                add_once(&mut self.pending_closes, order.id);
            }
        }
        !(self.unconfirmed_closes.is_empty() && self.pending_closes.is_empty())
    }

    /// The venue answered the modification of `order`, `confirmed` or refused (module
    /// docs): a hold by its new terms alone stays when it was confirmed, and ends when it
    /// was refused.
    fn answered(&mut self, order: OrderId, confirmed: bool) {
        let Some(at) = self.pending_closes.iter().position(|id| *id == order) else {
            return;
        };
        self.pending_closes.remove(at);
        if confirmed {
            add_once(&mut self.unconfirmed_closes, order);
        }
    }
}

/// `id` added to `ids` unless they have it.
fn add_once(ids: &mut Vec<OrderId>, id: OrderId) {
    if !ids.contains(&id) {
        ids.push(id);
    }
}

/// Whether the working `order`, priced as `terms` (its own order type, or that of its
/// modification not answered), may have closed part of a position that is long (`long`)
/// or short by the time the market traded at `price_rupees` (module docs): an order on
/// the closing side that is MARKET, a limit the price reached, or a stop whose trigger the
/// price reached or that has triggered already (a stop-limit is counted as soon as its
/// trigger is reached, since the price may have passed its limit on the way).
fn could_have_closed(order: &Order, terms: OrderType, long: bool, price_rupees: f64) -> bool {
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
    match terms {
        OrderType::Market => true,
        OrderType::Limit { limit_rupees } => reached_limit(limit_rupees),
        OrderType::StopMarket { trigger_rupees } | OrderType::StopLimit { trigger_rupees, .. } => {
            triggered || reached_trigger(trigger_rupees)
        }
    }
}

impl Engine {
    /// A tick of `instrument` at `price_rupees`, taken in `slice`: every open trade of the
    /// instrument takes it, and sees it (`Excursion::see_price`) unless it is held back
    /// (`Hold::held_back`).
    pub(super) fn see_tick(&mut self, slice: &SliceInfo, instrument: InstrumentId, price: f64) {
        let index = u32::try_from(slice.index).unwrap_or(u32::MAX);
        for trade in &mut self.record.trades.open {
            if trade.key.instrument != instrument {
                continue;
            }
            let Some(position) = self.portfolio.position(trade.key) else {
                continue;
            };
            let hold = &mut trade.excursion.hold;
            hold.take_tick(index, price);
            let side = (trade.key, position.quantity_units > 0.0);
            let orders = (&self.book, self.venue.modifying.as_slice());
            if hold.held_back(orders, side, &[price]) {
                continue;
            }
            let held = (Holding::of(position), position.multiplier);
            trade.excursion.see_price(price, held, trade.gross);
        }
    }

    /// The venue answered the modification of `order`, `confirmed` or refused: every open
    /// trade held back by its new terms alone keeps the hold or is released from it
    /// (`Hold::answered`).
    pub(super) fn hold_on_modify_answer(&mut self, order: OrderId, confirmed: bool) {
        for trade in &mut self.record.trades.open {
            trade.excursion.hold.answered(order, confirmed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn working(side: Side, order_type: OrderType, status: OrderStatus) -> Order {
        use crate::exec::{OrderRequest, QuantitySpec};
        Order {
            id: OrderId(1),
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

    fn bar(open: f64, high: f64, low: f64, close: f64) -> BarPrices {
        BarPrices {
            open,
            high,
            low,
            close,
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
        let (sell, buy, market) = (Side::Sell, Side::Buy, OrderType::Market);
        let table = [
            ("stop, tick above it", sell, stop, open, true, 498.0, false),
            ("stop, tick at it", sell, stop, open, true, 497.1, true),
            ("stop, tick through it", sell, stop, open, true, 496.0, true),
            ("stop triggered", sell, stop, fired, true, 499.0, true),
            (
                "stop-limit at its trigger",
                sell,
                stop_limit,
                open,
                true,
                497.0,
                true,
            ),
            ("target not reached", sell, target, open, true, 502.9, false),
            ("target reached", sell, target, open, true, 503.0, true),
            ("market exit sent", sell, market, open, true, 500.0, true),
            ("a buy while long", buy, market, open, true, 500.0, false),
            (
                "short stop below it",
                buy,
                short_stop,
                open,
                false,
                502.0,
                false,
            ),
            (
                "short stop through it",
                buy,
                short_stop,
                open,
                false,
                503.5,
                true,
            ),
        ];
        for (what, side, order_type, status, long, price, closed) in table {
            let order = working(side, order_type, status);
            let terms = order.request.order_type;
            assert_eq!(
                could_have_closed(&order, terms, long, price),
                closed,
                "{what}"
            );
        }
    }

    #[test]
    fn a_bar_is_judged_by_the_prices_its_ticks_did_not_carry() {
        // The bar of slice 7 is 500, 506, 497, 503. Ticks of slice 7 from 498 to 506 leave
        // its low; ticks from 497 to 504 its high; ticks from 497 to 506 nothing; ticks of
        // slice 6, or none, its low and high.
        let of_slice_7 = bar(500.0, 506.0, 497.0, 503.0);
        let table = [
            (Some((7, 498.0, 506.0)), vec![497.0]),
            (Some((7, 497.0, 504.0)), vec![506.0]),
            (Some((7, 497.0, 506.0)), vec![]),
            (Some((6, 497.0, 506.0)), vec![497.0, 506.0]),
            (None, vec![497.0, 506.0]),
        ];
        for (ticks, judged) in table {
            let hold = Hold {
                unconfirmed_closes: Vec::new(),
                pending_closes: Vec::new(),
                ticks: ticks.map(|(slice, low_rupees, high_rupees)| SliceTicks {
                    slice,
                    low_rupees,
                    high_rupees,
                }),
            };
            assert_eq!(hold.beyond_ticks(7, &of_slice_7), judged, "{ticks:?}");
        }
    }

    #[test]
    fn ticks_of_one_slice_widen_its_range_and_a_new_slice_starts_another() {
        let mut hold = Hold::default();
        for (slice, price) in [(3, 500.0), (3, 498.5), (3, 501.0), (4, 499.0)] {
            hold.take_tick(slice, price);
        }
        let expected = SliceTicks {
            slice: 4,
            low_rupees: 499.0,
            high_rupees: 499.0,
        };
        assert_eq!(hold.ticks, Some(expected));
        let mut three = Hold::default();
        for price in [500.0, 498.5, 501.0] {
            three.take_tick(3, price);
        }
        assert_eq!(
            three.ticks.map(|t| (t.low_rupees, t.high_rupees)),
            Some((498.5, 501.0))
        );
    }

    #[test]
    fn a_price_that_reaches_a_closing_order_holds_the_trade_back() {
        // One working order (o1) on SBIN in MIS and the prices judged (a bar's low and
        // high): a long's sell stop at 497.10 and sell target at 503.00, a short's buy
        // stop at 503.00 and buy target at 497.00, a MARKET exit, and a buy while long.
        let stop_at = |trigger_rupees| OrderType::StopMarket { trigger_rupees };
        let limit_at = |limit_rupees| OrderType::Limit { limit_rupees };
        let (sell, buy, market) = (Side::Sell, Side::Buy, OrderType::Market);
        let table = [
            (
                "long, stop reached",
                sell,
                stop_at(497.1),
                true,
                [497.0, 501.0],
                true,
            ),
            (
                "long, stop not reached",
                sell,
                stop_at(497.1),
                true,
                [497.15, 501.0],
                false,
            ),
            (
                "long, target reached",
                sell,
                limit_at(503.0),
                true,
                [499.0, 503.0],
                true,
            ),
            (
                "long, target not reached",
                sell,
                limit_at(503.0),
                true,
                [499.0, 502.95],
                false,
            ),
            (
                "short, stop reached",
                buy,
                stop_at(503.0),
                false,
                [499.0, 503.5],
                true,
            ),
            (
                "short, target reached",
                buy,
                limit_at(497.0),
                false,
                [497.0, 501.0],
                true,
            ),
            (
                "long, MARKET exit",
                sell,
                market,
                true,
                [499.0, 501.0],
                true,
            ),
            ("long, a buy", buy, market, true, [499.0, 501.0], false),
        ];
        for (what, side, order_type, long, prices, held) in table {
            let order = working(side, order_type, OrderStatus::Accepted);
            let key = PositionKey {
                instrument: order.request.instrument,
                product: order.request.product,
            };
            let mut book = OrderBook::default();
            book.insert(order).expect("o1 is the first order");
            let mut hold = Hold::default();
            assert_eq!(
                hold.held_back((&book, &[]), (key, long), &prices),
                held,
                "{what}"
            );
        }
    }

    #[test]
    fn a_trade_held_back_stays_so_while_the_order_works_and_whatever_the_price() {
        // The long's stop at 497.10 was reached at 496.00 and still works: neither a price
        // back at 505.00 nor a bar its ticks carried (no price judged) releases the trade.
        let order = working(
            Side::Sell,
            OrderType::StopMarket {
                trigger_rupees: 497.1,
            },
            OrderStatus::Accepted,
        );
        let key = PositionKey {
            instrument: order.request.instrument,
            product: order.request.product,
        };
        let mut book = OrderBook::default();
        book.insert(order).expect("o1 is the first order");
        let mut hold = Hold::default();
        assert!(hold.held_back((&book, &[]), (key, true), &[496.0]));
        assert!(hold.held_back((&book, &[]), (key, true), &[505.0]));
        assert!(hold.held_back((&book, &[]), (key, true), &[]));
        assert_eq!(hold.unconfirmed_closes, vec![OrderId(1)]);
    }

    #[test]
    fn a_price_reaching_only_a_modification_not_answered_holds_the_trade_until_the_answer() {
        // The long's stop (o1) works at 497.10 and its move to 502.00 waits for the
        // venue's answer. A price of 501.50 reaches only the new trigger and holds the
        // trade back; after the answer a price of 505.00 reaches neither trigger. The
        // confirmation keeps the hold, the refusal ends it. A price of 497.00 reaches the
        // working trigger too, and the hold stays whatever the answer.
        use crate::exec::OrderRequest;
        let stop_at = |trigger_rupees| OrderType::StopMarket { trigger_rupees };
        let table = [
            (501.5, true, true),
            (501.5, false, false),
            (497.0, false, true),
        ];
        for (price, confirmed, held_after) in table {
            let order = working(Side::Sell, stop_at(497.1), OrderStatus::Accepted);
            let key = PositionKey {
                instrument: order.request.instrument,
                product: order.request.product,
            };
            let moved = OrderRequest {
                order_type: stop_at(502.0),
                ..order.request.clone()
            };
            let in_flight = [(
                OrderId(1),
                Replacement {
                    request: moved.clone(),
                    decided_at: 0,
                    good_until: 0,
                    units: 100.0,
                },
            )];
            let mut book = OrderBook::default();
            book.insert(order).expect("o1 is the first order");
            let mut hold = Hold::default();
            let before = hold.held_back((&book, &in_flight), (key, true), &[price]);
            assert!(before, "{price}");
            if confirmed {
                book.replace(OrderId(1), moved, 0, 0, 100.0)
                    .expect("o1 takes the new terms");
            }
            hold.answered(OrderId(1), confirmed);
            let after = hold.held_back((&book, &[]), (key, true), &[505.0]);
            assert_eq!(after, held_after, "{price}, confirmed: {confirmed}");
        }
    }
}
