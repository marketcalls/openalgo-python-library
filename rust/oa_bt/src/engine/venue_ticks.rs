//! Tick mode in an external session (`venue`; 03-architecture 4.2, last paragraph;
//! 16-session-design 2.5 and 4.4, requirement R5): `Engine::push_tick` runs steps 3 to 6
//! for one instrument on the session's last price.
//!
//! For the instrument's legs, in the order they opened (`rms::LegState::on_tick`): the
//! favourable extreme, the trail and the breakeven move first, then the stop, then the
//! targets, all on that one price.
//!
//! - **Ratchets** (step 4) write `leg_trail_armed` and `leg_breakeven_armed` once each and
//!   a row of the position history; a stop they move is sent to the broker's stop under
//!   its throttle (`venue_legs`).
//! - **The stop** is the broker's: a price at or through it executes nothing here.
//! - **A target** (step 3) the price reaches writes `leg_target_hit` (its level and index)
//!   and makes the strategy's exit of the target's units: a reduce-only MARKET DAY order
//!   with the exit reason `leg_target`, taken by the position's exit owner (`venue_owner`)
//!   as a strategy exit: a full one cancels the stop first, a partial one waits for the
//!   stop to be modified down (R4). Its fill marks the target filled, and
//!   `after_first_target="breakeven"` then moves the stop. A target fires once per leg:
//!   an exit of it that ends without filling (rejected by the broker, cancelled) is not
//!   made again on a later tick, so a broker that keeps refusing it is not sent one exit
//!   per tick and the stop is not cut and restored each time; the session retries a
//!   failed exit on its own schedule (16-session-design 4.8).
//! - **Mark and aggregate rules** (steps 5 and 6): the instrument's open positions are
//!   marked at the price, and its open trades see it (their MAE and MFE, `excursion`),
//!   then the daily loss limit and the drawdown halt are tested on that mark
//!   (`aggregate`); their exits go through the exit owner.
//!
//! What a tick writes is recorded in the slice in progress (`venue_fills`), at the steps
//! above. While the market is closed (the session's closing slice) a tick only marks.

use crate::exec::{InstrumentId, Order, OrderId, OrderRequest, QuantitySpec};
use crate::model::{Product, TsNanos};
use crate::portfolio::PositionKey;
use crate::rms::{ExitReason, LevelHit, LevelKind};

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::{EventStamp, Step};
use super::kernel::Engine;
use super::legs::level_event;
use super::venue::SendAt;
use super::venue_owner::closing_side;
use super::venue_roles::OrderRole;

impl Engine {
    /// The session's last price of `instrument`, `price_rupees` at `at` (UTC nanoseconds):
    /// tick mode (module docs). Taken while the session waits, as its venue's results.
    ///
    /// Errors: `Venue(NotExternal)` in a backtest; `Venue(AtBarClose)` while paused at a
    /// bar close and `Venue(SessionOver)` after the session; `InvalidConfig` naming
    /// `symbol` for an instrument the session does not trade; `InvalidData` for a price
    /// that is not finite or not above 0; those of the leg rules and of the exits.
    pub fn push_tick(
        &mut self,
        instrument: InstrumentId,
        price_rupees: f64,
        at: TsNanos,
    ) -> Result<(), KernelError> {
        let slice = self.result_slice()?;
        let symbol = &self.spec(instrument)?.symbol;
        if !(price_rupees.is_finite() && price_rupees > 0.0) {
            return Err(KernelError::InvalidData {
                symbol: symbol.clone(),
                reason: format!("a tick at {price_rupees}: pass the last price in rupees, above 0"),
            });
        }
        self.venue.clock_ns = self.venue.clock_ns.max(at);
        let point = self.waiting_at(&slice);
        let market_open = self.day_validity(&slice, point.decided_at).is_ok();
        if market_open {
            self.tick_legs(&slice, instrument, price_rupees)?;
        }
        for position in self.portfolio.positions_mut() {
            if position.key.instrument == instrument && !position.is_flat() {
                position.mark_rupees = Some(price_rupees);
            }
        }
        self.see_tick(&slice, instrument, price_rupees);
        if market_open {
            self.aggregate_rules_at(&slice, point.decided_at)?;
        }
        self.progress_exits(point)
    }

    /// Steps 3 and 4 of a tick for `instrument`'s legs (module docs).
    ///
    /// Errors: those of the leg rules and of the target exits.
    fn tick_legs(
        &mut self,
        slice: &SliceInfo,
        instrument: InstrumentId,
        price_rupees: f64,
    ) -> Result<(), KernelError> {
        let at = self.waiting_at(slice);
        let keys: Vec<PositionKey> = self
            .leg_order()
            .into_iter()
            .filter(|key| key.instrument == instrument && key.product == Product::Mis)
            .collect();
        for key in keys {
            let outcome = match self.leg_mut(key) {
                Some(leg) => leg.on_tick(price_rupees)?,
                None => continue,
            };
            if outcome.ratchet.levels_moved {
                self.record.touch(key);
            }
            let stamp = Self::stamp(slice, Step::Ratchets);
            for event in outcome.ratchet.events {
                self.emit_leg_event(stamp, key, event);
            }
            for hit in outcome.hits {
                if let LevelKind::Target(index) = hit.kind {
                    self.exit_at_target(at, key, hit, index)?;
                }
            }
        }
        Ok(())
    }

    /// A target of `key`'s leg reached on a tick: its event, and the strategy's exit of its
    /// units, unless the target has fired already for the leg (module docs).
    ///
    /// Errors: those of the order book and of the exit owner.
    fn exit_at_target(
        &mut self,
        at: SendAt,
        key: PositionKey,
        hit: LevelHit,
        index: u16,
    ) -> Result<(), KernelError> {
        let fired = self.venue.fired_targets.contains(&(key, index));
        let held = self.held_units(key);
        let units = hit.quantity_units.min(held.abs());
        // The engine's exit, or exits already sent for the whole position, close what the
        // target would: a target exit then could only end `Canceled`.
        let covered = self.has_engine_exit(key) || held.abs() - self.left_to_exit(key) <= 0.0;
        let Ok(good_until) = self.day_validity(&at.slice, at.decided_at) else {
            return Ok(());
        };
        if fired || covered || units <= 0.0 {
            return Ok(());
        }
        let stamp = Self::stamp(&at.slice, Step::Levels);
        self.emit_leg_event(stamp, key, level_event(&hit));
        let request = OrderRequest {
            product: key.product,
            reduce_only: true,
            exit_reason: Some(ExitReason::LegTarget),
            ..OrderRequest::market(
                key.instrument,
                closing_side(held),
                QuantitySpec::Units(units),
            )
        };
        let id = self.insert_engine_order(at, request, units, good_until)?;
        self.record_role(id, OrderRole::Exit);
        self.venue.target_exits.push((id, index));
        self.venue.fired_targets.push((key, index));
        self.own_strategy_exit(at, id)
    }

    /// The target an engine-made target exit fills, by its order.
    pub(super) fn target_of(&self, id: OrderId) -> Option<u16> {
        let exits = &self.venue.target_exits;
        exits
            .iter()
            .find(|(known, _)| *known == id)
            .map(|(_, index)| *index)
    }

    /// After a fill of `placed` when it is a target exit: `after_first_target="breakeven"`
    /// moves the leg's stop (its event); the broker's stop follows (`venue_legs`).
    ///
    /// Errors: `InvalidRule` for a level that is not a positive price.
    pub(super) fn after_target_exit_fill(
        &mut self,
        stamp: EventStamp,
        placed: &Order,
    ) -> Result<(), KernelError> {
        if self.target_of(placed.id).is_none() {
            return Ok(());
        }
        let key = PositionKey {
            instrument: placed.request.instrument,
            product: placed.request.product,
        };
        let moved = match self.leg_mut(key) {
            Some(leg) => leg.after_target_fill()?,
            None => None,
        };
        if let Some(event) = moved {
            self.record.touch(key);
            self.emit_leg_event(stamp, key, event);
        }
        Ok(())
    }
}
