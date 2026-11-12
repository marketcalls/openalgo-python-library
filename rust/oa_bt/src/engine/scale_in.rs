//! The scale-in limits at submission (03-architecture 4.4 item 7, after the cooldown and
//! the maximum positions): the facts of an add, read from the position, the orders
//! still waiting to add to it and its leg, for `rms::scale_in_gate`.
//!
//! An order adds when it is on the side of the position it trades, or when that
//! position is flat and an order on the same side is already waiting to open it (so
//! two orders decided together cannot pass a limit between them). An order that only
//! reduces, or reverses an opposite position, does not add. A modification does not
//! count the order it replaces as waiting. A rebalance order (`OrderRequest::rebalance`)
//! is an intended add: averaging down does not refuse it; the other limits apply.
//!
//! Every fact is taken at the decision price, the instrument's last close (SPEC-15):
//! the add and the adds still waiting count as filled at that price. A quantity that
//! is sized at the fill (the engine default of fractional cash equity) is estimated as
//! the decision would size it.

use crate::exec::{Order, OrderId, OrderRequest, QuantitySpec, RejectCode};
use crate::model::{Side, TsNanos};
use crate::portfolio::{Position, PositionKey};
use crate::rms::{scale_in_gate, AddFacts, LegContext, LegState};
use crate::sizing::{SizeOutcome, SizerSpec};

use super::clock::SliceInfo;
use super::config::QuantityMode;
use super::error::KernelError;
use super::kernel::Engine;

impl Engine {
    /// The scale-in limits for `request`, whose quantity resolved to `units` (0 when it
    /// is sized at the fill), decided in `slice`. `replacing` is the order a
    /// modification replaces.
    ///
    /// Errors: those of sizing (an engine default estimated at the decision) and of the
    /// leg rules (the stop after the add). A broken limit is not an error:
    /// `Ok(Err(RejectCode::ScaleInLimit))`.
    pub(super) fn scale_in_limits(
        &self,
        request: &OrderRequest,
        units: f64,
        slice: &SliceInfo,
        replacing: Option<OrderId>,
    ) -> Result<Result<(), RejectCode>, KernelError> {
        let mut limits = self.config.risk.scale;
        // A rebalance buys what has fallen on purpose: an intended add (`rebalance`).
        limits.allow_averaging_down |= request.rebalance;
        if request.reduce_only || limits.is_open() {
            return Ok(Ok(()));
        }
        let key = PositionKey {
            instrument: request.instrument,
            product: request.product,
        };
        let held = self.held_units(key);
        if held != 0.0 && (held > 0.0) != (request.side == Side::Buy) {
            return Ok(Ok(()));
        }
        let waiting: Vec<&Order> = self
            .book
            .working()
            .filter(|order| {
                Some(order.id) != replacing
                    && !order.request.reduce_only
                    && order.request.side == request.side
                    && order.request.instrument == key.instrument
                    && order.request.product == key.product
            })
            .collect();
        if held == 0.0 && waiting.is_empty() {
            return Ok(Ok(()));
        }
        let facts = self.add_facts(request, units, slice, key, &waiting)?;
        Ok(scale_in_gate(&limits, &facts))
    }

    /// The facts of an add to `key` (see the module documentation).
    fn add_facts(
        &self,
        request: &OrderRequest,
        units: f64,
        slice: &SliceInfo,
        key: PositionKey,
        waiting: &[&Order],
    ) -> Result<AddFacts, KernelError> {
        let limits = self.config.risk.scale;
        let spec = self.spec(key.instrument)?;
        let position = self.portfolio.position(key).filter(|p| !p.is_flat());
        let decision = self.last_close(key.instrument);
        // On a flat position the first order waiting opens it; the others are adds.
        let waiting_adds = waiting
            .len()
            .saturating_sub(usize::from(position.is_none()));
        let filled_adds = position.map_or(0, |p| p.adds);
        let adds = filled_adds.saturating_add(u32::try_from(waiting_adds).unwrap_or(u32::MAX));
        let mut adding_units = 0.0;
        if limits.max_position.is_some() || limits.max_risk.is_some() {
            adding_units = self.units_at_decision(request, units, slice)?;
            for order in waiting {
                adding_units +=
                    self.units_at_decision(&order.request, order.quantity_units, slice)?;
            }
        }
        let (bars_since_last_tranche, last_tranche_price_rupees) =
            match position.and_then(|p| p.tranches.last()) {
                Some(tranche) if waiting.is_empty() => (
                    self.bars_closed_since(tranche.opened_at, slice),
                    Some(tranche.price_rupees),
                ),
                _ => (0, decision),
            };
        let mut facts = AddFacts {
            side: request.side,
            held_units: position.map_or(0.0, |p| p.quantity_units.abs()),
            average_price_rupees: position.map(|p| p.average_price_rupees),
            adds,
            adding_units,
            decision_price_rupees: decision,
            bars_since_last_tranche,
            last_tranche_price_rupees,
            equity_rupees: self.portfolio.equity_rupees(),
            lot_size_units: spec.lot_size_units,
            tick_rupees: spec.tick_size_rupees,
            multiplier: spec.multiplier,
            open_risk_rupees: None,
        };
        if limits.max_risk.is_some() {
            facts.open_risk_rupees = self.open_risk_after(key, &facts, position, waiting)?;
        }
        Ok(facts)
    }

    /// Units an order adds, at the decision: as resolved, or for the engine default
    /// sized at the fill, the default sized at the decision price (0 when it would be
    /// skipped).
    fn units_at_decision(
        &self,
        request: &OrderRequest,
        units: f64,
        slice: &SliceInfo,
    ) -> Result<f64, KernelError> {
        if units > 0.0 {
            return Ok(units);
        }
        let QuantitySpec::Sizer(SizerSpec::EngineDefault) = request.quantity else {
            return Ok(0.0);
        };
        Ok(
            match self.size_at_decision(request, SizerSpec::EngineDefault, slice)? {
                SizeOutcome::Units(units) => units,
                SizeOutcome::Skipped(_) => 0.0,
            },
        )
    }

    /// Base bars closed after the slice that opened at `opened_at` (a tranche's fill
    /// slice), counted as the cooldown counts them.
    fn bars_closed_since(&self, opened_at: TsNanos, slice: &SliceInfo) -> usize {
        let opens = self.timeline.slice_opens();
        let filled_in = opens
            .partition_point(|&open| open <= opened_at)
            .saturating_sub(1);
        slice.index.saturating_sub(filled_in)
    }

    /// The open risk of the position after the add, rupees, from the average price to
    /// the stop the leg would hold after it; `None` when it would have no stop or there
    /// is no decision price.
    fn open_risk_after(
        &self,
        key: PositionKey,
        add: &AddFacts,
        position: Option<&Position>,
        waiting: &[&Order],
    ) -> Result<Option<f64>, KernelError> {
        let Some(decision) = add.decision_price_rupees else {
            return Ok(None);
        };
        let units_after = add.units_after();
        if units_after <= 0.0 {
            return Ok(None);
        }
        let held_cost = add.held_units * add.average_price_rupees.unwrap_or(0.0);
        let average_after = (held_cost + add.adding_units * decision) / units_after;
        let leg = match (position, self.leg(key.instrument, key.product)) {
            (Some(_), Some(leg)) => {
                let mut leg = leg.clone();
                if add.adding_units > 0.0 {
                    leg.on_add(average_after, add.adding_units)?;
                }
                leg
            }
            _ => {
                // A flat position takes the rules of the order waiting to open it.
                let Some(opening) = waiting.first() else {
                    return Ok(None);
                };
                let rules = opening.request.leg_rules.clone();
                let context = self.add_leg_context(key, add.side)?;
                LegState::open(rules, context, average_after, units_after)?
            }
        };
        Ok(leg.stop.map(|stop| {
            let loss_per_unit = (average_after - stop.level_rupees) * add.side.sign();
            loss_per_unit.max(0.0) * units_after * add.multiplier
        }))
    }

    /// The leg context of a position on `side`, as a fill would build it.
    fn add_leg_context(&self, key: PositionKey, side: Side) -> Result<LegContext, KernelError> {
        let spec = self.spec(key.instrument)?;
        Ok(LegContext {
            side,
            tick_rupees: spec.tick_size_rupees,
            multiplier: spec.multiplier,
            round_levels: self.config.risk.round_levels,
            lot_size_units: spec.lot_size_units,
            fractional: self.config.quantity_mode == QuantityMode::Fractional
                && !spec.is_derivative(),
        })
    }
}
