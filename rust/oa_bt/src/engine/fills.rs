//! Fills: queued orders at the open (step 2), and the one path every fill takes into
//! the portfolio, the leg states and the records.

use crate::exec::{
    Fill, FillAt, FillFlags, FillReason, Order, OrderAction, OrderId, QuantitySpec, RejectCode,
};
use crate::market::margin::MarginRequest;
use crate::market::slippage::slipped_price;
use crate::model::{Product, Side};
use crate::portfolio::{FillEffect, FillKind, PositionChange, PositionKey};
use crate::rms::{ExitReason, LegContext, LegRules, LegState};
use crate::sizing::round_down_to_venue;

use super::clock::SliceInfo;
use super::config::QuantityMode;
use super::error::KernelError;
use super::event::{Event, EventStamp, OrderEventKind, Step};
use super::kernel::Engine;
use super::output::FillRecord;

/// One fill to execute.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct FillRequest {
    pub(super) key: PositionKey,
    pub(super) side: Side,
    pub(super) units: f64,
    /// The price before slippage: the open, a level or the last close.
    pub(super) reference_rupees: f64,
    pub(super) at: FillAt,
    pub(super) order: Option<OrderId>,
    /// Reason recorded if the fill reduces the position.
    pub(super) exit_reason: ExitReason,
    pub(super) flags: FillFlags,
    /// Margin the fill blocks, rupees (0 for an exit).
    pub(super) margin_rupees: f64,
    /// Rules for the leg an entry opens.
    pub(super) leg_rules: LegRules,
}

impl FillRequest {
    /// An engine-held exit of a whole position: no order, MARKET at the open.
    pub(super) fn engine_exit(
        key: PositionKey,
        held_units: f64,
        reference_rupees: f64,
        reason: ExitReason,
    ) -> FillRequest {
        let side = if held_units > 0.0 {
            Side::Sell
        } else {
            Side::Buy
        };
        FillRequest {
            key,
            side,
            units: held_units.abs(),
            reference_rupees,
            at: FillAt::Open,
            order: None,
            exit_reason: reason,
            flags: FillFlags::NONE,
            margin_rupees: 0.0,
            leg_rules: LegRules::default(),
        }
    }

    /// The same request priced at another point of the bar.
    pub(super) fn at(self, at: FillAt) -> FillRequest {
        FillRequest { at, ..self }
    }

    /// The same request with flags added.
    pub(super) fn flagged(self, flags: FillFlags) -> FillRequest {
        let flags = self.flags.with(flags);
        FillRequest { flags, ..self }
    }
}

/// What a queued order does at its fill bar.
enum AtFill {
    /// Fill this many units; `true` when it opens or adds.
    Fill(f64, bool),
    /// Refused by a check at the fill bar.
    Reject(RejectCode),
    /// Nothing left to reduce: cancelled.
    Cancel,
}

impl Engine {
    /// Fill a queued order at the open, when the open is at or after its decision
    /// instant (SPEC-35); otherwise it stays queued.
    pub(super) fn fill_queued(
        &mut self,
        slice: &SliceInfo,
        id: OrderId,
        open: f64,
    ) -> Result<(), KernelError> {
        let Some(order) = self.book.get(id).cloned() else {
            return Err(KernelError::UnknownOrder(id));
        };
        if slice.open < order.decided_at {
            return Ok(());
        }
        let stamp = Self::stamp(slice, Step::OpenFills);
        let request = &order.request;
        let key = PositionKey {
            instrument: request.instrument,
            product: request.product,
        };
        let (units, opening) = match self.at_fill(slice, &order, key)? {
            AtFill::Fill(units, opening) => (units, opening),
            AtFill::Reject(code) => {
                let kind = OrderEventKind::Rejected(code);
                return self.end_order(stamp, id, OrderAction::Reject, kind);
            }
            AtFill::Cancel => {
                let kind = OrderEventKind::Canceled;
                return self.end_order(stamp, id, OrderAction::Cancel, kind);
            }
        };
        let margin = if opening {
            self.margin_for(slice, key, request.side, units, open)?
        } else {
            0.0
        };
        if opening && margin > self.portfolio.free_cash_rupees() {
            let kind = OrderEventKind::Rejected(RejectCode::Margin);
            return self.end_order(stamp, id, OrderAction::Reject, kind);
        }
        let fill = FillRequest {
            key,
            side: request.side,
            units,
            reference_rupees: open,
            at: FillAt::Open,
            order: Some(id),
            exit_reason: request.exit_reason.unwrap_or(ExitReason::Signal),
            flags: FillFlags::NONE,
            margin_rupees: margin,
            leg_rules: request.leg_rules.clone(),
        };
        self.execute_fill(stamp, fill)?;
        Ok(())
    }

    /// Move an order to a terminal state without a fill, with its event.
    pub(super) fn end_order(
        &mut self,
        stamp: EventStamp,
        id: OrderId,
        action: OrderAction,
        kind: OrderEventKind,
    ) -> Result<(), KernelError> {
        self.book.apply(id, action)?;
        if let (Some(order), OrderEventKind::Rejected(code)) = (self.book.get_mut(id), kind) {
            order.reject_code = Some(code);
        }
        self.emit(Event::Order {
            stamp,
            order: id,
            kind,
        });
        Ok(())
    }

    /// Resolve a queued order against the position and the checks of the fill bar.
    fn at_fill(
        &mut self,
        slice: &SliceInfo,
        order: &Order,
        key: PositionKey,
    ) -> Result<AtFill, KernelError> {
        let request = &order.request;
        let held = self.held_units(key);
        let reduces = held != 0.0 && (held > 0.0) != (request.side == Side::Buy);
        if request.reduce_only {
            if !reduces {
                return Ok(AtFill::Cancel);
            }
            let spec = self.spec(key.instrument)?;
            let fractional =
                self.config.quantity_mode == QuantityMode::Fractional && !spec.is_derivative();
            let wanted = match request.quantity {
                QuantitySpec::PositionFraction(fraction) => {
                    round_down_to_venue(held.abs() * fraction, spec.lot_size_units, fractional)
                }
                _ => order.quantity_units,
            };
            let units = wanted.min(held.abs());
            return Ok(if units > 0.0 {
                AtFill::Fill(units, false)
            } else {
                AtFill::Cancel
            });
        }
        if reduces {
            if order.quantity_units > held.abs() {
                return Err(KernelError::NotBuilt {
                    feature: "an order larger than the opposite position (the excess rule and \
                              reverse=True)",
                });
            }
            return Ok(AtFill::Fill(order.quantity_units, false));
        }
        if request.product == Product::Mis {
            if let Err(code) = self.session_times(slice.day)?.entry.check(slice.open) {
                return Ok(AtFill::Reject(code));
            }
        }
        Ok(AtFill::Fill(order.quantity_units, true))
    }

    /// Margin an opening fill blocks, rupees.
    fn margin_for(
        &self,
        slice: &SliceInfo,
        key: PositionKey,
        side: Side,
        units: f64,
        price: f64,
    ) -> Result<f64, KernelError> {
        let spec = self.spec(key.instrument)?;
        let request = MarginRequest {
            exchange: spec.exchange,
            kind: spec.kind,
            product: key.product,
            side,
            quantity_units: units,
            price_rupees: price,
            multiplier: spec.multiplier,
            leverage: 1.0,
            day: slice.day,
        };
        Ok(self.config.margin.initial_margin(&request)?)
    }

    /// Execute one fill: price it, apply it to the position and the account, update the
    /// leg, and record the fill, its events and the trade.
    ///
    /// Errors: `Slippage`, `Portfolio`, `Rms` (a leg that cannot be opened).
    pub(super) fn execute_fill(
        &mut self,
        stamp: EventStamp,
        request: FillRequest,
    ) -> Result<FillEffect, KernelError> {
        let spec = self.spec(request.key.instrument)?.clone();
        let price = slipped_price(
            self.config.slippage,
            request.side,
            request.reference_rupees,
            spec.tick_size_rupees,
        )?;
        let position = self.portfolio.position_mut(request.key, spec.multiplier);
        let effect = position.apply(request.side, request.units, price, stamp.slice_ts)?;
        position.margin_rupees += request.margin_rupees;
        let average = position.average_price_rupees;
        self.portfolio.account.post_realized(effect.realized_rupees);
        self.portfolio.refresh_margin();
        let context = LegContext {
            side: request.side,
            tick_rupees: spec.tick_size_rupees,
            multiplier: spec.multiplier,
            round_levels: self.config.risk.round_levels,
        };
        self.update_leg(&request, &effect, context, average)?;
        let reason = match effect.kind {
            FillKind::Entry => FillReason::Entry,
            FillKind::Add => FillReason::Add,
            FillKind::Reduce => FillReason::Exit(request.exit_reason),
        };
        let fill = Fill {
            order: request.order,
            instrument: request.key.instrument,
            side: request.side,
            quantity_units: request.units,
            price_rupees: price,
            at: request.at,
            reason,
            target_index: None,
            flags: request.flags,
            costs_rupees: 0.0,
        };
        self.record_fill(stamp, request.key, fill, &effect)?;
        self.risk_after_fill(stamp, request.key, &fill, &effect);
        Ok(effect)
    }

    fn update_leg(
        &mut self,
        request: &FillRequest,
        effect: &FillEffect,
        context: LegContext,
        average: f64,
    ) -> Result<(), KernelError> {
        let key = request.key;
        let slot = self.legs.iter().position(|(k, _)| *k == key);
        match (effect.kind, effect.change, slot) {
            (FillKind::Entry, _, _) => {
                let rules = request.leg_rules.clone();
                let leg = LegState::open(rules, context, average, request.units)?;
                self.legs.retain(|(k, _)| *k != key);
                self.legs.push((key, leg));
            }
            (FillKind::Add, _, Some(slot)) => self.legs[slot].1.on_add(average, request.units)?,
            (FillKind::Reduce, PositionChange::Closed, Some(slot)) => {
                self.legs.remove(slot);
            }
            (FillKind::Reduce, _, Some(slot)) => {
                self.legs[slot].1.on_reduce(effect.quantity_after.abs())?;
            }
            _ => {}
        }
        Ok(())
    }

    fn record_fill(
        &mut self,
        stamp: EventStamp,
        key: PositionKey,
        fill: Fill,
        effect: &FillEffect,
    ) -> Result<(), KernelError> {
        if let Some(id) = fill.order {
            self.book.apply(id, OrderAction::Fill)?;
            if let Some(order) = self.book.get_mut(id) {
                order.filled_units = fill.quantity_units;
                order.average_fill_rupees = fill.price_rupees;
            }
            let kind = OrderEventKind::Filled {
                quantity_units: fill.quantity_units,
                price_rupees: fill.price_rupees,
            };
            self.emit(Event::Order {
                stamp,
                order: id,
                kind,
            });
        }
        self.emit(Event::Position {
            stamp,
            instrument: key.instrument,
            product: key.product,
            change: effect.change,
            quantity_units: effect.quantity_after,
        });
        let record = FillRecord {
            stamp,
            product: key.product,
            fill,
        };
        self.record.trades.record(&record, effect);
        self.record.fills.push(record);
        self.record.touch(key);
        Ok(())
    }
}
