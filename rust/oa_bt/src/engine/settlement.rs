//! Step 8 for futures: settlement at expiry and the daily mark to market (04-execution
//! 2 and 7; 03-architecture 4.2 step 8).
//!
//! At the close of a contract's last trading day, after the DAY orders expired and the
//! square-off check:
//!
//! 1. The contract's working orders (GTT orders that would outlive it) are cancelled.
//! 2. Every position still open on it is settled at the final settlement price: the
//!    data's settlement price for that day, or else the last close, flagged
//!    `filled_at_last_close`. The settlement is an engine-held exit with reason
//!    `expiry`, priced exactly there (no slippage) and not charged (no trade happens);
//!    its GTTs are cancelled first. Expiry is therefore dated on the expiry session.
//!
//! Then every futures position is marked to market at the session's settlement price
//! (the data's, else the last close, flagged): the posting is the position's P&L at that
//! price, realized plus open, less what earlier postings settled, rounded to the paisa.
//! A position closed during the session gets its last posting; one that opened and
//! closed within the session posts its whole P&L. The postings are cash flows in the
//! `settlements` table; they do not move equity (which already marks every position at
//! its close), do not change the average price used for P&L and do not restart the daily
//! loss measure (05-risk-management 5).

use crate::exec::{FillAt, FillFlags, FillPricing, InstrumentId, OrderAction, OrderId};
use crate::model::{Day, InstrumentKind};
use crate::portfolio::{round_to_paise, PositionKey};
use crate::rms::ExitReason;

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::{EventStamp, OrderEventKind, Step};
use super::fills::FillRequest;
use super::futures::{FillOrigin, SettlementKind, SettlementRecord};
use super::kernel::Engine;

/// A position's state at the close, for its posting.
struct Marked {
    key: PositionKey,
    quantity_units: f64,
    average_rupees: f64,
    realized_rupees: f64,
    multiplier: f64,
}

impl Engine {
    /// Step 8: settle the contracts expiring today, then post the mark to market of
    /// every futures position (module docs).
    ///
    /// Errors: those of `execute_fill`.
    pub(super) fn settle_futures(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        let stamp = Self::stamp(slice, Step::SessionClose);
        let expiring: Vec<InstrumentId> = (0..self.instruments.len())
            .map(|index| InstrumentId(index as u32))
            .filter(|&instrument| self.expires_on(instrument, slice.day))
            .collect();
        for instrument in expiring {
            self.cancel_orders_at_expiry(stamp, instrument)?;
            self.settle_at_expiry(stamp, slice.day, instrument)?;
        }
        self.post_mark_to_market(slice);
        Ok(())
    }

    /// True when `instrument` is a future whose last trading day is `day`.
    fn expires_on(&self, instrument: InstrumentId, day: Day) -> bool {
        self.instruments
            .get(instrument.index())
            .is_some_and(|spec| {
                spec.kind == InstrumentKind::Future && spec.last_trading_day == Some(day)
            })
    }

    /// The settlement price of a day: the data's, else the last close (true: used).
    fn settlement_price_of(&self, instrument: InstrumentId, day: Day) -> Option<(f64, bool)> {
        match self.futures.settlement_price(instrument, day) {
            Some(price) => Some((price, false)),
            None => self.last_close(instrument).map(|close| (close, true)),
        }
    }

    fn cancel_orders_at_expiry(
        &mut self,
        stamp: EventStamp,
        instrument: InstrumentId,
    ) -> Result<(), KernelError> {
        let working: Vec<OrderId> = self
            .book
            .working()
            .filter(|order| order.request.instrument == instrument)
            .map(|order| order.id)
            .collect();
        for id in working {
            self.end_order(stamp, id, OrderAction::Cancel, OrderEventKind::Canceled)?;
        }
        Ok(())
    }

    fn settle_at_expiry(
        &mut self,
        stamp: EventStamp,
        day: Day,
        instrument: InstrumentId,
    ) -> Result<(), KernelError> {
        let Some((price, last_close)) = self.settlement_price_of(instrument, day) else {
            return Ok(());
        };
        let open: Vec<PositionKey> = self
            .portfolio
            .positions()
            .iter()
            .filter(|position| position.key.instrument == instrument && !position.is_flat())
            .map(|position| position.key)
            .collect();
        for key in open {
            let held = self.held_units(key);
            let exit =
                FillRequest::engine_exit(key, held, price, ExitReason::Expiry).at(FillAt::Close);
            let mut request = FillRequest {
                pricing: FillPricing::Resting,
                origin: FillOrigin::ExpirySettlement,
                ..exit
            };
            if last_close {
                request = request.flagged(FillFlags::FILLED_AT_LAST_CLOSE);
            }
            self.execute_fill(stamp, request)?;
        }
        Ok(())
    }

    /// One posting per futures position whose P&L moved or that is still open.
    fn post_mark_to_market(&mut self, slice: &SliceInfo) {
        let marked: Vec<Marked> = self
            .portfolio
            .positions()
            .iter()
            .filter(|position| {
                self.instruments
                    .get(position.key.instrument.index())
                    .is_some_and(|spec| spec.kind == InstrumentKind::Future)
            })
            .map(|position| Marked {
                key: position.key,
                quantity_units: position.quantity_units,
                average_rupees: position.average_price_rupees,
                realized_rupees: position.realized_rupees,
                multiplier: position.multiplier,
            })
            .collect();
        for position in marked {
            self.post_one(slice, &position);
        }
    }

    fn post_one(&mut self, slice: &SliceInfo, position: &Marked) {
        let instrument = position.key.instrument;
        let Some((price, last_close)) = self.settlement_price_of(instrument, slice.day) else {
            return;
        };
        let open = position.quantity_units != 0.0;
        let unrealized = if open {
            (price - position.average_rupees) * position.quantity_units * position.multiplier
        } else {
            0.0
        };
        let total = position.realized_rupees + unrealized;
        let settled = &mut self.futures.settled;
        let slot = match settled.iter().position(|(key, _)| *key == position.key) {
            Some(slot) => slot,
            None => {
                settled.push((position.key, 0.0));
                settled.len() - 1
            }
        };
        let amount = round_to_paise(total - settled[slot].1);
        if !open && amount == 0.0 {
            return;
        }
        settled[slot].1 = round_to_paise(settled[slot].1 + amount);
        let kind = if self.expires_on(instrument, slice.day) {
            SettlementKind::Expiry
        } else {
            SettlementKind::Daily
        };
        self.futures.records.push(SettlementRecord {
            day: slice.day,
            slice_ts: slice.open,
            instrument,
            product: position.key.product,
            quantity_units: position.quantity_units,
            price_rupees: price,
            last_close_used: open && last_close,
            amount_rupees: amount,
            kind,
        });
    }
}
