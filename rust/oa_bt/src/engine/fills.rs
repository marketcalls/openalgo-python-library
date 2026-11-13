//! Fills: queued orders at the open (step 2), an order filled at a point of its bar
//! (`fill_order`, also step 3 for resting orders and GTT children), and the one path
//! every fill takes into the portfolio, the leg states and the records.

use crate::exec::{
    Fill, FillAt, FillFlags, FillPricing, FillReason, GttId, Order, OrderAction, OrderId,
    QuantitySpec, RejectCode,
};
use crate::model::{Product, Side};
use crate::portfolio::{FillEffect, FillKind, PositionChange, PositionKey};
use crate::rms::{ExitReason, LegContext, LegRules, LegState};
use crate::sizing::round_down_to_venue;

use super::clock::SliceInfo;
use super::config::QuantityMode;
use super::error::KernelError;
use super::event::{Event, EventStamp, OrderEventKind};
use super::futures::FillOrigin;
use super::kernel::Engine;
use super::order_fill::OrderFill;
use super::output::FillRecord;
use super::rebalance::waits_for_sales;

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
    /// The target a target exit fills (0 is the first).
    pub(super) target_index: Option<u16>,
    /// How the reference becomes the price (slippage, a limit).
    pub(super) pricing: FillPricing,
    /// The GTT whose child this fill is, or whose unfilled child it follows.
    pub(super) gtt: Option<GttId>,
    /// A trade, a futures roll or an expiry settlement (its reason and charges).
    pub(super) origin: FillOrigin,
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
            target_index: None,
            pricing: FillPricing::Market,
            gtt: None,
            origin: FillOrigin::Trade,
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
    /// `reverse=True` beyond the position: close these units, then open these.
    Reverse(f64, f64),
}

impl Engine {
    /// Fill a queued order at the open, when the open is at or after its decision
    /// instant (SPEC-35); otherwise it stays queued. A rebalance buy is left for
    /// `rebalance::fill_rebalance_buys`, after every instrument's queued orders.
    pub(super) fn fill_queued(
        &mut self,
        slice: &SliceInfo,
        id: OrderId,
        open: f64,
    ) -> Result<(), KernelError> {
        let Some(order) = self.book.get(id).cloned() else {
            return Err(KernelError::UnknownOrder(id));
        };
        if slice.open < order.decided_at || waits_for_sales(&order) {
            return Ok(());
        }
        self.fill_order(slice, &order, OrderFill::at_open(open))
    }

    /// Fill an order as `how` says, after the checks of the fill bar: the position
    /// (reduce, reverse), the entry window and margin.
    ///
    /// Errors: those of `execute_fill`, the session times, sizing and the margin model.
    pub(super) fn fill_order(
        &mut self,
        slice: &SliceInfo,
        order: &Order,
        how: OrderFill,
    ) -> Result<(), KernelError> {
        let (id, price) = (order.id, how.point.price_rupees);
        let stamp = Self::stamp(slice, how.step);
        let request = &order.request;
        let key = PositionKey {
            instrument: request.instrument,
            product: request.product,
        };
        let (units, opening) = match self.at_fill(slice, order, key)? {
            AtFill::Fill(units, opening) => (units, opening),
            AtFill::Reject(code) => {
                let kind = OrderEventKind::Rejected(code);
                return self.end_order(stamp, id, OrderAction::Reject, kind);
            }
            AtFill::Cancel => {
                let kind = OrderEventKind::Canceled;
                return self.end_order(stamp, id, OrderAction::Cancel, kind);
            }
            AtFill::Reverse(close, open_units) => {
                return self.fill_reversal(slice, order, (close, open_units), how);
            }
        };
        let Some(units) = self.size_at_fill(slice, order, units, opening, price)? else {
            let kind = OrderEventKind::Rejected(RejectCode::Margin);
            return self.end_order(stamp, id, OrderAction::Reject, kind);
        };
        let margin = if opening {
            let leverage = request.leverage;
            self.opening_margin(slice.day, key, request.side, units, price, leverage)?
        } else {
            0.0
        };
        if opening && margin > self.free_cash_rupees() {
            let kind = OrderEventKind::Rejected(RejectCode::Margin);
            return self.end_order(stamp, id, OrderAction::Reject, kind);
        }
        let fill = FillRequest {
            key,
            side: request.side,
            units,
            reference_rupees: price,
            at: how.point.at,
            order: Some(id),
            exit_reason: request.exit_reason.unwrap_or(ExitReason::Signal),
            flags: FillFlags::NONE,
            margin_rupees: margin,
            leg_rules: request.leg_rules.clone(),
            target_index: None,
            pricing: how.pricing,
            gtt: how.gtt,
            origin: FillOrigin::Trade,
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
        self.cancel_order_gtt(stamp, id)?;
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
                return Ok(if request.reverse {
                    AtFill::Reverse(held.abs(), order.quantity_units - held.abs())
                } else {
                    AtFill::Reject(RejectCode::ReverseNotAllowed)
                });
            }
            return Ok(AtFill::Fill(order.quantity_units, false));
        }
        if request.product == Product::Mis {
            if let Err(code) = self.session_times(slice.day)?.entry.check(slice.open) {
                return Ok(AtFill::Reject(code));
            }
        }
        // Delivery cannot be shorted (04-execution 2): a resting sale whose long was
        // closed after it was placed must not open a short.
        if request.product == Product::Cnc && request.side == Side::Sell {
            return Ok(AtFill::Reject(RejectCode::ProductRule));
        }
        Ok(AtFill::Fill(order.quantity_units, true))
    }

    /// Execute one fill: price it, apply it to the position and the account, charge it,
    /// update the leg, and record the fill, its events and the trade.
    ///
    /// Errors: `Slippage`, `Portfolio`, `Rms` (a leg that cannot be opened),
    /// `InvalidConfig` or `NotBuilt` from the charges.
    pub(super) fn execute_fill(
        &mut self,
        stamp: EventStamp,
        request: FillRequest,
    ) -> Result<FillEffect, KernelError> {
        self.cancel_gtts_before_exit(stamp, &request)?;
        let spec = self.spec(request.key.instrument)?.clone();
        let day = self.trade_day(stamp);
        let price = self.fill_price(&request, spec.tick_size_rupees, day)?;
        let position = self.portfolio.position_mut(request.key, spec.multiplier);
        let effect = position.apply(request.side, request.units, price, stamp.slice_ts)?;
        position.margin_rupees += request.margin_rupees;
        let average = position.average_price_rupees;
        self.portfolio.account.post_realized(effect.realized_rupees);
        self.portfolio.refresh_margin();
        self.credit_delivery_sale(&request, price, spec.multiplier, day, effect.kind)?;
        let (side, units) = (request.side, request.units);
        let costs = match request.origin {
            FillOrigin::ExpirySettlement => self.uncharged_fill(request.key, effect),
            _ => self.charge_fill(request.key, side, units, price, day, effect)?,
        };
        let context = LegContext {
            side: request.side,
            tick_rupees: spec.tick_size_rupees,
            multiplier: spec.multiplier,
            round_levels: self.config.risk.round_levels,
            lot_size_units: spec.lot_size_units,
            fractional: self.config.quantity_mode == QuantityMode::Fractional
                && !spec.is_derivative(),
        };
        self.update_leg(&request, &effect, context, average)?;
        let reason = match effect.kind {
            _ if request.origin == FillOrigin::Roll => FillReason::Roll,
            FillKind::Entry => FillReason::Entry,
            FillKind::Add => FillReason::Add,
            FillKind::Reduce => FillReason::Exit(request.exit_reason),
        };
        let fill = Fill {
            order: request.order,
            gtt: request.gtt,
            instrument: request.key.instrument,
            side: request.side,
            quantity_units: request.units,
            price_rupees: price,
            at: request.at,
            reason,
            target_index: request.target_index,
            flags: request.flags,
            costs_rupees: costs,
        };
        self.record_fill(stamp, request.key, fill, &effect)?;
        self.risk_after_fill(stamp, request.key, &fill, &effect);
        self.place_gtts_after_entry(stamp, request.key, &effect)?;
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
            (FillKind::Add, _, Some(slot)) => {
                let leg = &mut self.legs[slot].1;
                leg.accepts_add_rules(&request.leg_rules)?;
                leg.on_add(average, request.units)?;
            }
            (FillKind::Reduce, PositionChange::Closed, Some(slot)) => {
                self.legs.remove(slot);
            }
            (FillKind::Reduce, _, Some(slot)) => {
                let remaining = effect.quantity_after.abs();
                self.legs[slot]
                    .1
                    .on_reduce(remaining, request.target_index)?;
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
            // A reversal fills its order in two parts: the close, then the open.
            let part = self.book.get(id).is_some_and(|order| {
                order.filled_units + fill.quantity_units < order.quantity_units
            });
            let action = if part {
                OrderAction::FillPart
            } else {
                OrderAction::Fill
            };
            self.book.apply(id, action)?;
            if let Some(order) = self.book.get_mut(id) {
                let value = order.filled_units * order.average_fill_rupees
                    + fill.quantity_units * fill.price_rupees;
                order.filled_units += fill.quantity_units;
                order.average_fill_rupees = value / order.filled_units;
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
        if let Some(gtt) = fill.gtt {
            self.gtt_child_filled(stamp, gtt, fill.price_rupees)?;
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
        if effect.kind == FillKind::Entry {
            let risk = self.initial_risk(key, &fill);
            self.record.trades.set_initial_risk(key, risk);
        }
        self.record.fills.push(record);
        self.record.touch(key);
        Ok(())
    }
}
