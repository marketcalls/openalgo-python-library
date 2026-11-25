//! A leg's broker-held stop in an external session (`venue`; decision D31,
//! 16-session-design 4.1, 4.3, 4.5 and 4.10, requirements R2, R3, R4 and R6).
//!
//! A backtest holds a leg's stop itself and fills it at its level (`legs`). A live session
//! hands an MIS leg's stop to the broker as an order and keeps that order on the leg (a
//! CNC or NRML leg's stop is a GTT instead, kept by `venue_gtt_sync`, unless the run's
//! `positional_stops="rearm_daily"` holds it as an MIS leg's, placed again at each session's
//! open: `rearm`):
//!
//! - **Placed** once the leg has a stop level: with the fill that opens the leg, for the
//!   units filled, or when a trail or a breakeven first sets one. It is a reduce-only DAY
//!   order on the closing side in the segment's shape (`exec::segment::venue_order_type`:
//!   SL-M for cash equity and futures; for options an SL with its limit
//!   `OPTIONS_STOP_LIMIT_BUFFER_PERCENT` beyond the trigger, since NFO and BFO take no
//!   SL-M), its trigger the leg's level on the tick grid, with the role `leg_stop` and the
//!   exit reason `leg_sl`.
//! - **Kept on the leg** (`venue_stop_terms`): its units follow the position less what the
//!   position's exits still have to fill, at once; its trigger follows a trail or a
//!   breakeven under a throttle (a move of 0.25 percent, at most once every 5 s).
//! - **Reached**, the kernel executes nothing: the broker fills the stop, and its fill
//!   closes the leg (`leg_sl_hit` before its first fill, reason `leg_sl`; a fill beyond the
//!   position is refused, `venue_fills`).
//! - **A partial exit** (a `reduce()`, a partial target) keeps the stop: it waits while the
//!   stop covers more than will stay open, the stop is modified down, and the exit is sent
//!   once the venue confirms it (`venue_owner`); a refusal leaves the exit waiting and the
//!   modification goes again after the throttle. A full exit cancels the stop first, as
//!   every order on the closing side.
//! - **Ended without a fill** while its leg is open and no exit of the position waits for
//!   it or holds it (R3): rejected by the broker, the leg is exited at market at once
//!   (`stop_rejected`); ended any other way without a cancellation the kernel sent
//!   (cancelled or expired by the broker of its own accord, or not placed: the live layer
//!   reports a stop it could not place with `apply_cancel`), it is re-placed once at once,
//!   and a re-placed stop that ends without a fill exits the leg at market (`stop_rejected`
//!   when rejected, `unprotected` otherwise). A stop that filled all its units while the
//!   leg holds more (an add not covered yet) exits the rest at market (`stop_unfilled`).
//!   Nothing is sent while the market is closed.
//! - **Rejected for margin** (`apply_margin_reject`, `venue_group`): the broker may not
//!   hold the units, so the leg is neither exited at market nor protected again, and the
//!   position is held: no stop of it is placed or given more units until the session's
//!   `protect` (which re-places the stop) or `request_exit`.
//! - **Cancelled for an exit** (or by the strategy), the stop is not placed again by the
//!   kernel: the session's `protect` re-places it (`venue_protect`).
//! - **Repriced** to a LIMIT by the session's price rule (`venue_protect`), it no longer
//!   follows the leg's level: it is exiting the position.

use crate::exec::price::round_to_tick;
use crate::exec::segment::venue_order_type;
use crate::exec::{OrderId, OrderRequest, OrderType, QuantitySpec};
use crate::model::TsNanos;
use crate::portfolio::PositionKey;
use crate::rms::{ExitReason, RiskEvent, RiskEventName, RiskScope};

use super::error::KernelError;
use super::event::{Event, EventStamp};
use super::kernel::Engine;
use super::rearm::StopAsk;
use super::venue::SendAt;
use super::venue_owner::closing_side;
use super::venue_roles::{EngineExit, OrderRole, RiskExitReason};

/// A leg's broker-held stop (module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct LegStop {
    pub(super) key: PositionKey,
    /// The last stop order placed for the leg; `None` until the leg has a stop level.
    pub(super) order: Option<OrderId>,
    /// When its last modification was sent, UTC nanoseconds.
    pub(super) modified_at: Option<TsNanos>,
    /// The venue refused its last modification.
    pub(super) refused: bool,
    /// It is the kernel's re-placement of a stop that ended without a fill (R3).
    pub(super) replaced: bool,
    /// The session repriced it to a LIMIT: it exits the position (`venue_protect`).
    pub(super) repriced: bool,
}

impl LegStop {
    fn new(key: PositionKey) -> LegStop {
        LegStop {
            key,
            order: None,
            modified_at: None,
            refused: false,
            replaced: false,
            repriced: false,
        }
    }
}

/// How a leg's stop ended at the venue (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StopEnd {
    /// The venue rejected it.
    Rejected,
    /// The venue cancelled it; `requested` when the kernel had sent the cancellation.
    Canceled { requested: bool },
    /// It filled all its units.
    Filled,
    /// The venue rejected it, and the session took the rejection as the broker not holding
    /// the units (`venue_group`): the leg is neither exited nor protected again.
    MarginRejected,
}

impl Engine {
    /// A leg opened at `key`: its stop is placed once it has a level (module docs), none
    /// of its targets has fired (`venue_ticks`), and a margin rejection of an earlier leg's
    /// closing order no longer holds the position (`venue_group`). A stop of an earlier
    /// leg of the position is no longer the leg's: a new leg opens from flat only once no
    /// such stop works (`venue_gates`), and a reversal cancels it first or leaves it on the
    /// side that no longer closes, where it is cancelled (`venue_cover`).
    pub(super) fn open_leg_stop(&mut self, key: PositionKey) {
        self.venue.leg_stops.retain(|stop| stop.key != key);
        if self.stop_is_gtt(key.product) {
            self.open_leg_gtt(key);
        } else {
            self.venue.leg_stops.push(LegStop::new(key));
        }
        self.venue.fired_targets.retain(|(known, _)| *known != key);
        self.venue.group.release(key);
    }

    /// The leg's stop record of `key`, if the session keeps one.
    pub(super) fn leg_stop_of(&self, key: PositionKey) -> Option<LegStop> {
        self.venue
            .leg_stops
            .iter()
            .find(|stop| stop.key == key)
            .copied()
    }

    pub(super) fn leg_stop_mut(&mut self, key: PositionKey) -> Option<&mut LegStop> {
        self.venue.leg_stops.iter_mut().find(|stop| stop.key == key)
    }

    /// Ask for a new stop of `key`'s leg, `replaced` for the kernel's re-placement (R3):
    /// it is placed by `sync_leg_stop` within the room it has (`venue_stop_terms`).
    pub(super) fn renew_leg_stop(&mut self, key: PositionKey, replaced: bool) {
        let renewed = LegStop {
            replaced,
            ..LegStop::new(key)
        };
        match self.leg_stop_mut(key) {
            Some(stop) => *stop = renewed,
            None => self.venue.leg_stops.push(renewed),
        }
    }

    /// The working stop order of `key`'s leg, repriced or not.
    pub(super) fn working_leg_stop(&self, key: PositionKey) -> Option<(OrderId, LegStop)> {
        let stop = self.leg_stop_of(key)?;
        let id = stop.order?;
        let works = self.book.get(id).is_some_and(|o| o.status.is_working());
        works.then_some((id, stop))
    }

    /// The working stop of `key`'s leg that rests at its trigger (not repriced).
    pub(super) fn resting_leg_stop(&self, key: PositionKey) -> Option<OrderId> {
        let (id, stop) = self.working_leg_stop(key)?;
        (!stop.repriced).then_some(id)
    }

    /// The working stop of `key`'s leg that the session repriced to exit it.
    pub(super) fn repriced_leg_stop(&self, key: PositionKey) -> Option<OrderId> {
        let (id, stop) = self.working_leg_stop(key)?;
        stop.repriced.then_some(id)
    }

    /// The leg's stop level of `key`, rupees on the tick grid; `None` without one.
    ///
    /// Errors: `Exec` for an unusable tick.
    pub(super) fn stop_trigger(&self, key: PositionKey) -> Result<Option<f64>, KernelError> {
        let Some(level) = self
            .leg(key.instrument, key.product)
            .and_then(|leg| leg.stop)
        else {
            return Ok(None);
        };
        let tick = self.spec(key.instrument)?.tick_size_rupees;
        Ok(Some(round_to_tick(level.level_rupees, tick)?))
    }

    /// The units `key`'s stop covers (module docs): the position less what its exits sent
    /// still have to fill and less a partial exit that waits for the stop; 0 or more.
    /// When the waiting exits together close what is open, they wait for the stop's
    /// cancellation, and the stop covers all of it until that goes through (a refused
    /// cancellation is retried, 16-session-design 4.5). A partial exit of a position a
    /// margin rejection holds waits for the hold, not for the stop: it takes nothing from
    /// the stop (`venue_group`).
    ///
    /// Errors: those of `partial_exit_units` and `waiting_exits_close_all`.
    pub(super) fn stop_cover_units(&self, key: PositionKey) -> Result<f64, KernelError> {
        let open = self.held_units(key).abs() - self.left_to_exit(key);
        if self.waiting_exits_close_all(key, open)? {
            return Ok(open.max(0.0));
        }
        let first_waiting = self
            .venue
            .exits
            .iter()
            .find(|exit| exit.key == key && exit.sent.is_empty());
        let waiting = match first_waiting {
            Some(exit) if !self.venue.group.holds(key) => {
                self.partial_exit_units(exit, open)?.unwrap_or(0.0)
            }
            _ => 0.0,
        };
        Ok((open - waiting).max(0.0))
    }

    /// The order type of a stop of a position held on `held` signed units at `trigger`:
    /// SL-M in the segment's shape.
    ///
    /// Errors: `Exec` for an unusable trigger or tick.
    pub(super) fn stop_type(
        &self,
        key: PositionKey,
        held: f64,
        trigger: f64,
    ) -> Result<OrderType, KernelError> {
        let spec = self.spec(key.instrument)?;
        let asked = OrderType::StopMarket {
            trigger_rupees: trigger,
        };
        let side = closing_side(held);
        Ok(venue_order_type(
            spec.kind,
            asked,
            side,
            spec.tick_size_rupees,
        )?)
    }

    /// Place a stop of `units` for `key`'s leg at its level, decided as `at` says; `replaced`
    /// for the kernel's re-placement (R3). Nothing is placed for a leg with no level, no
    /// units to cover, or a closed market. True when it was placed.
    ///
    /// Errors: those of the order book and of the stop's shape.
    pub(super) fn place_leg_stop(
        &mut self,
        at: SendAt,
        key: PositionKey,
        units: f64,
        replaced: bool,
    ) -> Result<bool, KernelError> {
        let held = self.held_units(key);
        let Some(trigger) = self.stop_trigger(key)? else {
            return Ok(false);
        };
        let Ok(good_until) = self.day_validity(&at.slice, at.decided_at) else {
            return Ok(false);
        };
        if held == 0.0 || units <= 0.0 {
            return Ok(false);
        }
        let request = OrderRequest {
            product: key.product,
            reduce_only: true,
            exit_reason: Some(ExitReason::LegSl),
            order_type: self.stop_type(key, held, trigger)?,
            ..OrderRequest::market(
                key.instrument,
                closing_side(held),
                QuantitySpec::Units(units),
            )
        };
        let id = self.insert_engine_order(at, request, units, good_until)?;
        self.place(id, OrderRole::LegStop);
        if self.leg_stop_of(key).is_none() {
            self.venue.leg_stops.push(LegStop::new(key));
        }
        if let Some(stop) = self.leg_stop_mut(key) {
            *stop = LegStop {
                order: Some(id),
                replaced,
                ..LegStop::new(key)
            };
        }
        Ok(true)
    }

    /// Before the first fill of order `id` when it is a leg's stop: `leg_sl_hit`, with its
    /// trigger as the level (the leg's level once it was repriced). A fired GTT's child
    /// wrote it when the GTT fired (`venue_gtt_child`).
    pub(super) fn before_stop_fill(&mut self, stamp: EventStamp, id: OrderId) {
        let Some(order) = self.book.get(id) else {
            return;
        };
        let child = self.gtt_child_of(id).is_some();
        if self.order_role(id) != Some(OrderRole::LegStop) || order.filled_units > 0.0 || child {
            return;
        }
        let instrument = order.request.instrument;
        let level = order.request.order_type.trigger_rupees().or_else(|| {
            let leg = self.leg(instrument, order.request.product)?;
            leg.stop.map(|stop| stop.level_rupees)
        });
        let event = RiskEvent {
            name: RiskEventName::LegSlHit,
            scope: RiskScope::Leg,
            level_rupees: level,
            target_index: None,
            observed_rupees: None,
        };
        self.emit(Event::Risk {
            stamp,
            instrument: Some(instrument),
            event,
        });
    }

    /// After the venue ended order `id` as `how`: when it is the stop at work of a leg still
    /// open that no exit closing the whole position waits for or holds, protect the leg
    /// (module docs), decided as `at` says. A partial exit that waits for the stop does not
    /// close the leg: the rest is protected again.
    ///
    /// Errors: those of `place_leg_stop` and of the exit owner.
    pub(super) fn after_stop_end(
        &mut self,
        at: SendAt,
        id: OrderId,
        how: StopEnd,
    ) -> Result<(), KernelError> {
        let found = self
            .venue
            .leg_stops
            .iter()
            .find(|stop| stop.order == Some(id))
            .copied();
        let Some(stop) = found else {
            return Ok(());
        };
        let key = stop.key;
        if self.held_units(key) == 0.0 || self.exit_closes_leg(key)? {
            return Ok(());
        }
        if self.rearm_stop_expired(&at, key, how) {
            return Ok(());
        }
        let reason = match (how, stop.replaced) {
            (StopEnd::Canceled { requested: true } | StopEnd::MarginRejected, _) => return Ok(()),
            (StopEnd::Canceled { requested: false }, false) => {
                self.renew_leg_stop(key, true);
                return self.sync_leg_stop(at, key, StopAsk::Kernel);
            }
            (StopEnd::Canceled { requested: false }, true) => RiskExitReason::Unprotected,
            (StopEnd::Rejected, _) => RiskExitReason::StopRejected,
            (StopEnd::Filled, _) => RiskExitReason::StopUnfilled,
        };
        self.request_engine_exit(at, key, EngineExit::Risk(reason), None)
    }

    /// True when an exit closes all of `key`'s position: the engine's exit holds it, the
    /// exits of it that wait ask for all that is open, or its exits sent leave nothing.
    ///
    /// Errors: those of `waiting_exits_close_all`.
    pub(super) fn exit_closes_leg(&self, key: PositionKey) -> Result<bool, KernelError> {
        if self.has_engine_exit(key) {
            return Ok(true);
        }
        let open = self.held_units(key).abs() - self.left_to_exit(key);
        if !self.has_waiting_exit(key) {
            return Ok(open <= 0.0);
        }
        self.waiting_exits_close_all(key, open)
    }

    /// Mark `key`'s working stop as repriced (`venue_protect`): it exits the position.
    pub(super) fn mark_stop_repriced(&mut self, id: OrderId) {
        let stop = self
            .venue
            .leg_stops
            .iter_mut()
            .find(|stop| stop.order == Some(id));
        if let Some(stop) = stop {
            stop.repriced = true;
        }
    }
}
