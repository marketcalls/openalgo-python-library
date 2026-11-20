//! The day's trades of a strategy, for the trade caps of 18-multi-strategy 2.5a (rules 1,
//! 2 and 7): the strategy's `max_trades_per_day` and the account's
//! `account_max_trades_per_day`.
//!
//! A trade is an entry: an order that opens a position from flat, or a reversal's opening
//! part. Adds, exits, stops, targets, the square-off, risk exits and rolls never count. An
//! entry counts from its submission while it works, and stays counted once any part of it
//! fills; one that ends with no fill (cancelled, rejected, expired, refused by the broker)
//! stops counting. So entries decided together cannot pass a cap between them, as with
//! `max_positions`. The count is kept by the IST session day of the decision and starts
//! again at the next session day.
//!
//! `TradeCount` keeps the entries that filled as a number and the entries still waiting
//! for a fill by order id; whether a waiting one still works is the order book's, read
//! when the count is taken, so an entry that ends unfilled stops counting without being
//! told.

use crate::exec::OrderId;
use crate::model::Day;

/// The account's trade cap as one engine sees it: the cap, and the trades of the other
/// strategies of the account on the engine's session day.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccountTrades {
    /// `account_max_trades_per_day`, at least 1.
    pub limit: u32,
    /// The entries of the other strategies counted on the session day.
    pub others: u32,
}

/// The day's entries of one strategy (module docs).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TradeCount {
    /// The IST session day the count is of; `None` before the first entry.
    day: Option<Day>,
    /// Entries of `day` that have filled, in part or whole.
    filled: u32,
    /// Entries of `day` with no fill yet, by order id, in the order they were entered.
    waiting: Vec<OrderId>,
}

impl TradeCount {
    /// The count as a live session saved it: its day, the entries that filled and those
    /// waiting for a fill.
    pub fn resume(day: Option<Day>, filled: u32, waiting: Vec<OrderId>) -> TradeCount {
        TradeCount {
            day,
            filled,
            waiting,
        }
    }

    /// The day the count is of, the entries that filled and those waiting (to save it).
    pub fn parts(&self) -> (Option<Day>, u32, &[OrderId]) {
        (self.day, self.filled, &self.waiting)
    }

    /// Count `order`, an entry decided on session day `day` that passed its checks; a
    /// later day than the count's starts the count again.
    pub fn entered(&mut self, day: Day, order: OrderId) {
        if self.day != Some(day) {
            *self = TradeCount {
                day: Some(day),
                ..TradeCount::default()
            };
        }
        self.waiting.push(order);
    }

    /// `order` filled, in part or whole: an entry waiting for its fill stays counted for
    /// good. Any other order changes nothing.
    pub fn filled(&mut self, order: OrderId) {
        if let Some(at) = self.waiting.iter().position(|waiting| *waiting == order) {
            self.waiting.remove(at);
            self.filled += 1;
        }
    }

    /// The entries counted on session day `day`: those that filled and those waiting
    /// that still work (`works`, the order book's word); 0 for another day.
    pub fn on(&self, day: Day, works: impl Fn(OrderId) -> bool) -> u32 {
        if self.day != Some(day) {
            return 0;
        }
        let waiting = self.waiting.iter().filter(|order| works(**order)).count();
        self.filled + u32::try_from(waiting).unwrap_or(u32::MAX)
    }

    /// True when `order` is an entry counted while it waits for its fill.
    pub fn is_waiting(&self, order: OrderId) -> bool {
        self.waiting.contains(&order)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY: Day = 20_270;

    #[test]
    fn an_entry_counts_while_it_works_and_once_it_has_filled() {
        let mut count = TradeCount::default();
        count.entered(DAY, OrderId(1));
        count.entered(DAY, OrderId(2));
        assert_eq!(count.on(DAY, |_| true), 2);
        count.filled(OrderId(1));
        // Order 1 filled and no longer works; order 2 still works.
        assert_eq!(count.on(DAY, |order| order == OrderId(2)), 2);
    }

    #[test]
    fn an_entry_that_ends_with_no_fill_stops_counting() {
        let mut count = TradeCount::default();
        count.entered(DAY, OrderId(4));
        assert_eq!(count.on(DAY, |_| false), 0);
    }

    #[test]
    fn a_fill_of_another_order_changes_nothing() {
        let mut count = TradeCount::default();
        count.entered(DAY, OrderId(4));
        count.filled(OrderId(9));
        assert_eq!(count.on(DAY, |_| false), 0);
        assert!(count.is_waiting(OrderId(4)));
    }

    #[test]
    fn the_count_starts_again_on_a_later_session_day() {
        let mut count = TradeCount::default();
        count.entered(DAY, OrderId(1));
        count.filled(OrderId(1));
        assert_eq!(count.on(DAY + 1, |_| true), 0);
        count.entered(DAY + 1, OrderId(2));
        assert_eq!(count.on(DAY + 1, |_| true), 1);
        assert_eq!(count.on(DAY, |_| true), 0);
    }

    #[test]
    fn a_saved_count_resumes_as_it_was() {
        let mut count = TradeCount::default();
        count.entered(DAY, OrderId(1));
        count.filled(OrderId(1));
        count.entered(DAY, OrderId(3));
        let (day, filled, waiting) = count.parts();
        let resumed = TradeCount::resume(day, filled, waiting.to_vec());
        assert_eq!(resumed, count);
    }
}
