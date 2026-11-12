//! Step 2: the futures auto-roll (04-execution 7; decisions SPEC-19, SPEC-20 and GTT-9
//! reading A).
//!
//! At a switch that rolls positions (`futures::Switch`), after the strategy's orders,
//! the engine's exits and the square-off have filled at this open, every open position
//! on the near contract, in the order the positions opened:
//!
//! 1. A level already crossed at the near contract's open exits there instead: a GTT
//!    (CNC, NRML) whose trigger is crossed fires in step 2 and its child fills by the
//!    GTT rules; an engine-held MIS level fills at the open with its own reason. What is
//!    left of the position rolls; a position whose GTT child did not fill keeps waiting
//!    for the engine's exit on the near contract and does not roll.
//! 2. The near contract is closed at its open and the next opened at its open, near
//!    leg first, both plus slippage and charged, both with reason `roll` and no order
//!    (position events only). The near contract's GTTs are cancelled before its fill
//!    (cancel-then-exit).
//! 3. The position keeps its leg: stop, targets, trail, breakeven, the favourable
//!    extreme and the entry reference move by the spread (next raw open minus near raw
//!    open, before slippage), while the average price used for P&L is the next fill
//!    (SPEC-20). New GTTs with the shifted triggers are placed on the next contract and
//!    tested in step 3 of this slice.
//!
//! The roll keeps the units (quantity is in units, never lots). When the exchange
//! changed the lot size between the two contracts, the next contract's position is the
//! nearest whole number of its lots to those units, and at least one lot, and the leg
//! follows the new quantity (a smaller position shrinks it as a partial exit does; a
//! larger one is an add at the shifted reference). The next position blocks its margin
//! from the dated table; a roll is never refused for margin, since it continues a
//! position the run already holds. The near contract's working orders are cancelled at
//! the switch: entries trade the next contract from the roll session on.

use crate::exec::{OrderAction, OrderId};
use crate::model::Side;
use crate::portfolio::{FillKind, PositionKey};
use crate::rms::{BarPrices, ExitReason, LegState};

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::{EventStamp, OrderEventKind, Step};
use super::fills::FillRequest;
use super::futures::{FillOrigin, Switch};
use super::gtt::held_as_gtt;
use super::kernel::Engine;

/// Units of the next contract for a position of `held_units` units: the same units when
/// they are whole lots of it, else the nearest whole number of its lots and at least one
/// (lot size in units).
fn rolled_units(held_units: f64, next_lot_units: u32) -> f64 {
    let lot = f64::from(next_lot_units);
    let lots = (held_units / lot).round().max(1.0);
    lots * lot
}

impl Engine {
    /// Step 2, last: every switch of this slice that rolls positions (module docs).
    ///
    /// Errors: those of `execute_fill`, the margin table and the leg rules.
    pub(super) fn roll_due_futures(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        let due: Vec<Switch> = self
            .futures
            .switches
            .iter()
            .filter(|switch| switch.slice == slice.index && switch.rolls_positions)
            .copied()
            .collect();
        for switch in due {
            self.roll_switch(slice, &switch)?;
        }
        Ok(())
    }

    fn roll_switch(&mut self, slice: &SliceInfo, switch: &Switch) -> Result<(), KernelError> {
        let stamp = Self::stamp(slice, Step::OpenFills);
        let near_bar = self.market.bar(switch.from, slice.index);
        let next_bar = self.market.bar(switch.to, slice.index);
        let (Some(near_bar), Some(next_bar)) = (near_bar, next_bar) else {
            // A switch is placed only on a slice where both contracts have a bar.
            return Ok(());
        };
        let open: Vec<PositionKey> = self
            .portfolio
            .positions()
            .iter()
            .filter(|position| position.key.instrument == switch.from && !position.is_flat())
            .map(|position| position.key)
            .collect();
        for key in open {
            self.exit_levels_crossed_at_open(slice, key, &near_bar)?;
            let waiting = self.gtts.follow_ups.iter().any(|exit| exit.key == key);
            if self.held_units(key) == 0.0 || waiting {
                continue;
            }
            let opens = (near_bar.open, next_bar.open);
            self.roll_position(stamp, slice, key, switch, opens)?;
        }
        self.cancel_working_orders_of(stamp, switch)
    }

    /// A level of the position already crossed at the near contract's open exits it
    /// there (SPEC-19): its GTT fires in step 2 (GTT-9), or an MIS level fills at the
    /// open.
    fn exit_levels_crossed_at_open(
        &mut self,
        slice: &SliceInfo,
        key: PositionKey,
        bar: &BarPrices,
    ) -> Result<(), KernelError> {
        if held_as_gtt(key.product) {
            for id in self.gtts.book.protecting(key.instrument, key.product) {
                let crossed = self.gtts.book.get(id).is_some_and(|gtt| {
                    gtt.legs
                        .iter()
                        .any(|leg| leg.direction.reached(bar.open, leg.trigger_rupees))
                });
                if crossed && self.held_units(key) != 0.0 {
                    self.test_gtt(slice, id, bar, Step::OpenFills)?;
                }
            }
            return Ok(());
        }
        let at_open = BarPrices {
            open: bar.open,
            high: bar.open,
            low: bar.open,
            close: bar.open,
        };
        let hits = match self.legs.iter().find(|(k, _)| *k == key) {
            Some((_, leg)) => leg.test_bar(&at_open),
            None => return Ok(()),
        };
        for hit in hits {
            if self.held_units(key) == 0.0 {
                break;
            }
            self.exit_at_level(Self::stamp(slice, Step::OpenFills), key, hit)?;
        }
        Ok(())
    }

    /// Close the near contract and open the next, then give the next position the
    /// near position's leg, shifted by the spread, and its GTTs.
    fn roll_position(
        &mut self,
        stamp: EventStamp,
        slice: &SliceInfo,
        key: PositionKey,
        switch: &Switch,
        (near_open, next_open): (f64, f64),
    ) -> Result<(), KernelError> {
        let held = self.held_units(key);
        let leg = self
            .legs
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, leg)| leg.clone());
        let close = FillRequest {
            origin: FillOrigin::Roll,
            ..FillRequest::engine_exit(key, held, near_open, ExitReason::Signal)
        };
        self.execute_fill(stamp, close)?;
        let next_key = PositionKey {
            instrument: switch.to,
            product: key.product,
        };
        let next_spec = self.spec(switch.to)?.clone();
        let units = rolled_units(held.abs(), next_spec.lot_size_units);
        let side = if held > 0.0 { Side::Buy } else { Side::Sell };
        let margin = self.opening_margin(slice.day, next_key, side, units, next_open, None)?;
        let open = FillRequest {
            side,
            units,
            margin_rupees: margin,
            origin: FillOrigin::Roll,
            ..FillRequest::engine_exit(next_key, held, next_open, ExitReason::Signal)
        };
        let effect = self.execute_fill(stamp, open)?;
        let (Some(mut leg), FillKind::Entry) = (leg, effect.kind) else {
            return Ok(());
        };
        leg.context.tick_rupees = next_spec.tick_size_rupees;
        leg.context.multiplier = next_spec.multiplier;
        leg.context.lot_size_units = next_spec.lot_size_units;
        leg.shift(switch.spread_rupees)?;
        follow_quantity(&mut leg, units)?;
        if let Some((_, slot)) = self.legs.iter_mut().find(|(k, _)| *k == next_key) {
            *slot = leg;
        }
        self.record.touch(next_key);
        self.place_gtts_after_entry(stamp, next_key, &effect)
    }

    /// Cancel the near contract's working orders at the switch.
    fn cancel_working_orders_of(
        &mut self,
        stamp: EventStamp,
        switch: &Switch,
    ) -> Result<(), KernelError> {
        let working: Vec<OrderId> = self
            .book
            .working()
            .filter(|order| order.request.instrument == switch.from)
            .map(|order| order.id)
            .collect();
        for id in working {
            self.end_order(stamp, id, OrderAction::Cancel, OrderEventKind::Canceled)?;
        }
        Ok(())
    }
}

/// The rolled leg follows a quantity that changed with the lot size: a smaller one as a
/// partial exit, a larger one as an add at the (shifted) reference.
fn follow_quantity(leg: &mut LegState, units: f64) -> Result<(), KernelError> {
    let held = leg.held_units();
    if units < held {
        leg.on_reduce(units, None)?;
    } else if units > held {
        let reference = leg.reference_rupees;
        leg.on_add(reference, units - held)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::rolled_units;

    #[test]
    fn a_roll_keeps_the_units_when_they_are_whole_lots_of_the_next_contract() {
        assert_eq!(rolled_units(150.0, 75), 150.0);
        // Three lots of 25 are one lot of 75.
        assert_eq!(rolled_units(75.0, 75), 75.0);
    }

    #[test]
    fn a_new_lot_size_takes_the_nearest_whole_number_of_lots() {
        // 150 units are 2.3 lots of 65: two lots, 130 units.
        assert_eq!(rolled_units(150.0, 65), 130.0);
        // 100 units are 1.54 lots of 65: two lots, 130 units.
        assert_eq!(rolled_units(100.0, 65), 130.0);
    }

    #[test]
    fn a_position_smaller_than_the_new_lot_rolls_into_one_lot() {
        // 25 units are a third of a lot of 75: one lot, so the position goes on.
        assert_eq!(rolled_units(25.0, 75), 75.0);
    }
}
