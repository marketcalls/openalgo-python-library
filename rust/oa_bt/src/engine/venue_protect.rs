//! The session's own calls for a leg (`venue`; 16-session-design 4.6, 4.8 and 4.10,
//! requirement R7, the kernel's third unit), taken while it waits or at a decision point,
//! as its other controls (`venue_session`).
//!
//! | Call | What |
//! |---|---|
//! | `protect(instrument, product)` | place the leg's stop again at its level for what the position's exits leave, when no stop of it works and no exit of the position waits or holds it; a resting order sent only to close the position is cancelled first for the room the stop takes (`venue_stop_terms`); says what it did (`Protect`) |
//! | `reprice_exit(order, limit_rupees)` | modify a leg's stop the price has reached and not filled (or an engine exit at a limit) to a LIMIT at `limit_rupees`, for the order's total units (a broker's modify takes the total), a partly filled one too; sent now, or with the venue's answer to a modification of it that waits |
//! | `request_limit_exit(instrument, product, exit, limit_rupees)` | `request_exit` (`venue_session`) whose orders are LIMIT orders at `limit_rupees`: the price rule's marketable fallback |
//!
//! A repriced stop no longer follows its leg (`venue_legs`): it exits the position, so the
//! strategy's exits of the position are sized on what it leaves and an engine exit cancels
//! it first (`venue_owner`). A price must be a positive price on the instrument's tick.

use crate::exec::price::is_on_tick;
use crate::exec::{ExecError, InstrumentId, OrderId, OrderRequest, OrderType, QuantitySpec};
use crate::model::Product;
use crate::portfolio::PositionKey;

use super::error::KernelError;
use super::kernel::Engine;
use super::venue::Replacement;
use super::venue_roles::{EngineExit, OrderRole};
use super::venue_session::ExitRequest;

/// What `Engine::protect` did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Protect {
    /// The leg's stop was placed.
    Placed,
    /// The leg's stop is placed once the resting orders sent only to close the position,
    /// cancelled for its room, have ended (`venue_stop_terms`).
    WaitsForCancels,
    /// A stop of the leg works already.
    StopWorks,
    /// An exit of the position waits or works: it closes what the stop would cover.
    Exiting,
    /// The leg has no stop level (no `stop=`, a trail or breakeven not armed yet).
    NoStopLevel,
    /// The position is flat.
    Flat,
    /// The market is closed: no stop can be placed.
    MarketClosed,
}

impl Protect {
    /// Every outcome's code, in the order of the variants.
    pub const CODES: [&'static str; 7] = [
        "placed",
        "waits_for_cancels",
        "stop_works",
        "exiting",
        "no_stop_level",
        "flat",
        "market_closed",
    ];

    /// The code the bindings use.
    pub fn code(self) -> &'static str {
        match self {
            Protect::Placed => "placed",
            Protect::WaitsForCancels => "waits_for_cancels",
            Protect::StopWorks => "stop_works",
            Protect::Exiting => "exiting",
            Protect::NoStopLevel => "no_stop_level",
            Protect::Flat => "flat",
            Protect::MarketClosed => "market_closed",
        }
    }
}

fn invalid(reason: String) -> KernelError {
    KernelError::Exec(ExecError::InvalidRequest { reason })
}

impl Engine {
    /// Place the stop of the leg of `instrument` in `product` again (module docs).
    ///
    /// Errors: `Venue(NotExternal)` in a backtest; `Venue(SessionOver)` after the session;
    /// `InvalidConfig` naming `symbol` for an instrument the session does not trade; those
    /// of the order book and of the stop's shape.
    pub fn protect(
        &mut self,
        instrument: InstrumentId,
        product: Product,
    ) -> Result<Protect, KernelError> {
        self.require_session()?;
        self.spec(instrument)?;
        let at = self.control_point()?;
        let key = PositionKey {
            instrument,
            product,
        };
        // The session asks: a margin rejection no longer holds the position, and the exits
        // the hold kept waiting move on first (`venue_group`).
        if self.venue.group.release(key) {
            self.progress_exits(at)?;
        }
        if self.held_units(key) == 0.0 {
            return Ok(Protect::Flat);
        }
        if self
            .leg(instrument, product)
            .and_then(|leg| leg.stop)
            .is_none()
        {
            return Ok(Protect::NoStopLevel);
        }
        let cover = self.stop_cover_units(key)?;
        if self.has_waiting_exit(key) || self.has_engine_exit(key) || cover <= 0.0 {
            return Ok(Protect::Exiting);
        }
        if self.working_leg_stop(key).is_some() {
            return Ok(Protect::StopWorks);
        }
        if self.day_validity(&at.slice, at.decided_at).is_err() {
            return Ok(Protect::MarketClosed);
        }
        self.renew_leg_stop(key, false);
        self.sync_leg_stop(at, key)?;
        Ok(match self.working_leg_stop(key) {
            Some(_) => Protect::Placed,
            None => Protect::WaitsForCancels,
        })
    }

    /// Modify order `order` to a LIMIT at `limit_rupees` (module docs). True when the
    /// modification was sent now, false when it waits for the venue's answer to the one
    /// sent before.
    ///
    /// Errors: `Venue(NotExternal)` in a backtest; `Venue(SessionOver)` after the session;
    /// `UnknownOrder`, `Venue(OrderEnded)` and `Venue(NotSent)` as for `apply_fill`;
    /// `Exec(InvalidRequest)` for an order that is not a leg's stop or an engine exit, and
    /// for a limit that is not a positive price on the instrument's tick.
    pub fn reprice_exit(&mut self, order: OrderId, limit_rupees: f64) -> Result<bool, KernelError> {
        self.require_session()?;
        self.control_point()?;
        let placed = self.live_order(order)?;
        let instrument = placed.request.instrument;
        let repriceable = matches!(
            self.order_role(order),
            Some(OrderRole::LegStop | OrderRole::RiskExit(_) | OrderRole::SquareOff)
        );
        if !repriceable {
            return Err(invalid(format!(
                "order {order} is neither a leg's stop nor the engine's exit: only those are \
                 repriced; cancel the order and place a new one instead"
            )));
        }
        self.check_limit(instrument, limit_rupees)?;
        if self.venue.cancelling.contains(&order) {
            return Err(invalid(format!(
                "order {order} is being cancelled: its cancellation waits for the venue's \
                 answer, and an exit follows it"
            )));
        }
        self.mark_stop_repriced(order);
        if self.venue.modifying.iter().any(|(id, _)| *id == order) {
            self.venue.reprices.retain(|(id, _)| *id != order);
            self.venue.reprices.push((order, limit_rupees));
            return Ok(false);
        }
        self.send_reprice(order, limit_rupees);
        Ok(true)
    }

    /// `request_exit` (`venue_session`) with the exit's orders LIMIT orders at
    /// `limit_rupees` (module docs).
    ///
    /// Errors: those of `request_exit`; `Exec(InvalidRequest)` for a limit that is not a
    /// positive price on the instrument's tick.
    pub fn request_limit_exit(
        &mut self,
        instrument: InstrumentId,
        product: Product,
        exit: EngineExit,
        limit_rupees: f64,
    ) -> Result<ExitRequest, KernelError> {
        self.require_session()?;
        self.check_limit(instrument, limit_rupees)?;
        self.request_exit_at(instrument, product, exit, Some(limit_rupees))
    }

    /// Send the reprice of `order` that waited for the venue's answer to its last
    /// modification, now that it has one (module docs).
    pub(super) fn send_waiting_reprice(&mut self, order: OrderId) {
        let slot = self.venue.reprices.iter().position(|(id, _)| *id == order);
        let works = self.book.get(order).is_some_and(|o| o.status.is_working());
        if let (Some(slot), true) = (slot, works) {
            let (_, limit_rupees) = self.venue.reprices.remove(slot);
            self.send_reprice(order, limit_rupees);
        }
    }

    /// Send `Modify` of `order` to a LIMIT at `limit_rupees` for its total units.
    fn send_reprice(&mut self, order: OrderId, limit_rupees: f64) {
        let Some(placed) = self.book.get(order) else {
            return;
        };
        let units = placed.quantity_units;
        let replacement = Replacement {
            request: OrderRequest {
                order_type: OrderType::Limit { limit_rupees },
                quantity: QuantitySpec::Units(units),
                ..placed.request.clone()
            },
            decided_at: placed.decided_at,
            good_until: placed.good_until,
            units,
        };
        self.send_modify(order, replacement);
    }

    /// Refuse a limit that is not a positive price on `instrument`'s tick.
    ///
    /// Errors: `InvalidConfig` naming `symbol` for an unknown instrument;
    /// `Exec(InvalidRequest)` for the limit.
    fn check_limit(&self, instrument: InstrumentId, limit_rupees: f64) -> Result<(), KernelError> {
        let tick = self.spec(instrument)?.tick_size_rupees;
        let usable = limit_rupees.is_finite() && limit_rupees > 0.0;
        if !usable || !is_on_tick(limit_rupees, tick)? {
            return Err(invalid(format!(
                "limit {limit_rupees}: pass a positive price in rupees on the tick of {tick}"
            )));
        }
        Ok(())
    }
}
