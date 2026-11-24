//! The GTTs of a live session (`venue`; design 20-positional-live sections 2, 3 and 13.1
//! PK1, unit N1): what the session follows of each GTT besides its state, the commands it
//! sends for them, and what the live layer may ask about them.
//!
//! A backtest places, modifies and cancels its GTTs itself (`gtt`, `gtt_orders`). A live
//! session sends `VenueCommand::PlaceGtt`, `ModifyGtt` and `CancelGtt` at those moments
//! instead, taken with `take_commands` in the order they were made, and the GTT moves on
//! only with the venue's answer (`venue_gtt_answers`; the states are `exec::gtt`'s):
//!
//! - **A place** puts the GTT in the book as `Requested`: it is not in force, and its leg
//!   is unprotected, until the venue confirms it (`apply_gtt_placed`).
//! - **A modification** is a full replacement (trigger, child limit, units). The GTT keeps
//!   its confirmed terms (`ModifyPending`) until the venue's answer; the terms sent wait
//!   here (`GttFollow::pending`). One at a time: what is wanted meanwhile is the leg's, and
//!   goes out once the answer has come (`venue_gtt_sync`).
//! - **A cancellation** waits as `CancelPending`. One wanted while the place or a
//!   modification waits is sent once that is answered; one the venue refused is sent
//!   again in a later slice, as the engine's own withdrawal of an order is.
//! - **A stop GTT no leg holds** (`Engine::loose_stop_gtts`): one whose leg opened again
//!   from flat, or that was carried to be cancelled (`venue_state::carry`), while the venue
//!   had not taken its cancellation. It is cancelled, again in every later slice until the
//!   venue takes it; meanwhile it keeps the position's entries out (`venue_gates`), a
//!   new stop GTT on its side waits for it (`venue_gtt_sync`), its units take their room
//!   on the position's closing side (`venue_cover`), and an exit of the whole position
//!   waits for its cancellation (`venue_owner`): the broker never holds two stop GTTs, or a
//!   stop GTT and an exit, that sell or buy one position.
//!
//! Every command carries the GTT's id, its instrument and product, the side of its child,
//! its trigger and child limit in rupees (rounded to the tick as `gtt_plan` rounds them),
//! its units, and whether the trigger lies below the price it was placed from: the live
//! layer sends a trigger in the field of its direction (design 3.6). The kernel never
//! fires a GTT in a session; the venue does (`venue_gtt_child`).

use crate::exec::{
    Gtt, GttAction, GttId, GttLeg, GttLegRole, GttPurpose, GttStatus, InstrumentId, OrderId,
    TriggerDirection,
};
use crate::model::{Product, Side, TsNanos};
use crate::portfolio::PositionKey;

use super::error::KernelError;
use super::kernel::Engine;
use super::venue::VenueCommand;
use super::venue_owner::closing_side;

/// A SINGLE GTT's terms as a live session sends them (module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GttTerms {
    /// The instrument, by registration index.
    pub instrument: InstrumentId,
    /// CNC or NRML.
    pub product: Product,
    /// The side of the child order the GTT places when it fires.
    pub side: Side,
    /// The child order's units, never lots.
    pub units: f64,
    /// The trigger, rupees on the tick grid.
    pub trigger_rupees: f64,
    /// The child order's limit, rupees on the tick grid.
    pub limit_rupees: f64,
    /// True when the trigger lies below the price it was placed from (it fires on a fall:
    /// a long's stop); false above it (a short's stop).
    pub trigger_below: bool,
}

impl GttTerms {
    /// The terms `gtt` rests on (or was asked for, while its place waits).
    pub fn of(gtt: &Gtt) -> GttTerms {
        GttTerms {
            instrument: gtt.instrument,
            product: gtt.product,
            side: gtt.side,
            units: gtt.quantity_units,
            trigger_rupees: gtt.leg.trigger_rupees,
            limit_rupees: gtt.leg.child_limit_rupees,
            trigger_below: gtt.leg.direction == TriggerDirection::Falling,
        }
    }

    /// These terms' trigger and child limit as the leg of a GTT for `role`.
    pub(super) fn leg(&self, role: GttLegRole) -> GttLeg {
        let direction = match self.trigger_below {
            true => TriggerDirection::Falling,
            false => TriggerDirection::Rising,
        };
        GttLeg {
            role,
            trigger_rupees: self.trigger_rupees,
            direction,
            child_limit_rupees: self.limit_rupees,
        }
    }
}

/// What a session follows of one GTT besides its state (module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct GttFollow {
    pub(super) gtt: GttId,
    /// The terms of the modification the venue has not answered.
    pub(super) pending: Option<GttTerms>,
    /// When its last modification was sent, by the session's clock, UTC nanoseconds.
    pub(super) modified_at: Option<TsNanos>,
    /// The venue refused its last modification, a units cut on the resting trigger: the
    /// next cut waits for the minute (`venue_gtt_sync`).
    pub(super) refused: bool,
    /// A cancellation is wanted: sent once the request that waits is answered.
    pub(super) cancel_wanted: bool,
    /// The slice in which the venue refused its last cancellation: it is sent again in a
    /// later one.
    pub(super) cancel_refused_slice: Option<u32>,
}

impl GttFollow {
    pub(super) fn new(gtt: GttId) -> GttFollow {
        GttFollow {
            gtt,
            pending: None,
            modified_at: None,
            refused: false,
            cancel_wanted: false,
            cancel_refused_slice: None,
        }
    }
}

/// A CNC or NRML leg's stop GTT (`venue_gtt_sync`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct GttLegStop {
    pub(super) key: PositionKey,
    /// The leg's last stop GTT; `None` until the leg has had one, or once a new one is
    /// asked for (`venue_gtt_protect`).
    pub(super) gtt: Option<GttId>,
    /// The leg's stop GTTs the broker rejected after their placement, this session.
    pub(super) rejections: u8,
}

/// The child order of a GTT the venue fired (`venue_gtt_child`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct GttChild {
    pub(super) gtt: GttId,
    /// The kernel order the session follows the child as: the engine's exit of a leg's
    /// stop GTT, or the strategy's own GTT order.
    pub(super) order: OrderId,
    /// When the venue fired the GTT, UTC nanoseconds.
    pub(super) triggered_at: TsNanos,
}

/// The GTT part of a live session's venue desk (module docs).
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct VenueGtts {
    /// One per GTT the session has sent, in id order.
    pub(super) follows: Vec<GttFollow>,
    /// The CNC and NRML legs' stop GTTs, in the order the legs opened.
    pub(super) legs: Vec<GttLegStop>,
    /// The children of the GTTs the venue fired, in the order they fired.
    pub(super) children: Vec<GttChild>,
}

/// A GTT of a live session as the live layer queries it.
#[derive(Clone, Debug, PartialEq)]
pub struct GttView {
    /// The GTT: its state, its purpose (a leg's stop, or the strategy order it rests as)
    /// and the terms it rests on (asked for, while its place waits).
    pub gtt: Gtt,
    /// Its terms (`gtt`'s, as a command carries them).
    pub terms: GttTerms,
    /// The terms of a modification the venue has not answered.
    pub pending: Option<GttTerms>,
    /// The kernel order of its child, once the venue fired it.
    pub child: Option<OrderId>,
}

impl Engine {
    /// The GTT `id` of a run, with what a live session follows of it (module docs).
    pub fn gtt_view(&self, id: GttId) -> Option<GttView> {
        let gtt = self.gtts.book.get(id)?.clone();
        let follow = self.gtt_follow(id);
        let child = self.venue.gtts.children.iter().find(|c| c.gtt == id);
        Some(GttView {
            terms: GttTerms::of(&gtt),
            pending: follow.and_then(|follow| follow.pending),
            child: child.map(|child| child.order),
            gtt,
        })
    }

    /// The GTTs of `instrument` in `product`, in id order (`gtt_view`).
    pub fn gtt_views(&self, instrument: InstrumentId, product: Product) -> Vec<GttView> {
        let ids: Vec<GttId> = self
            .gtts
            .book
            .gtts()
            .iter()
            .filter(|gtt| gtt.instrument == instrument && gtt.product == product)
            .map(|gtt| gtt.id)
            .collect();
        ids.into_iter().filter_map(|id| self.gtt_view(id)).collect()
    }

    /// What the session follows of GTT `id`.
    pub(super) fn gtt_follow(&self, id: GttId) -> Option<GttFollow> {
        self.venue
            .gtts
            .follows
            .iter()
            .find(|f| f.gtt == id)
            .copied()
    }

    pub(super) fn gtt_follow_mut(&mut self, id: GttId) -> &mut GttFollow {
        let follows = &mut self.venue.gtts.follows;
        let slot = match follows.iter().position(|f| f.gtt == id) {
            Some(slot) => slot,
            None => {
                follows.push(GttFollow::new(id));
                follows.len() - 1
            }
        };
        &mut follows[slot]
    }

    /// The stop GTT record of `key`'s leg.
    pub(super) fn leg_gtt_of(&self, key: PositionKey) -> Option<GttLegStop> {
        self.venue
            .gtts
            .legs
            .iter()
            .find(|leg| leg.key == key)
            .copied()
    }

    pub(super) fn leg_gtt_mut(&mut self, key: PositionKey) -> Option<&mut GttLegStop> {
        self.venue.gtts.legs.iter_mut().find(|leg| leg.key == key)
    }

    /// The stop GTT of `key`'s leg while it is at the venue or asked for there.
    pub(super) fn stop_gtt_at_venue(&self, key: PositionKey) -> Option<&Gtt> {
        let id = self.leg_gtt_of(key)?.gtt?;
        self.gtts.book.get(id).filter(|gtt| gtt.status.at_venue())
    }

    /// The stop GTTs at the venue (or asked for there) that no leg holds, in id order
    /// (module docs).
    pub(super) fn loose_stop_gtts(&self) -> Vec<GttId> {
        let legs = &self.venue.gtts.legs;
        self.gtts
            .book
            .gtts()
            .iter()
            .filter(|gtt| gtt.purpose == GttPurpose::Protect && gtt.status.at_venue())
            .filter(|gtt| !legs.iter().any(|leg| leg.gtt == Some(gtt.id)))
            .map(|gtt| gtt.id)
            .collect()
    }

    /// The stop GTTs no leg holds (`loose_stop_gtts`) that sell or buy `key`'s position on
    /// its closing side, in id order; none for a flat position. Until the venue takes
    /// their cancellation they cover its units at the broker as its leg's own would.
    pub(super) fn loose_closing_stop_gtts(&self, key: PositionKey) -> Vec<GttId> {
        let held = self.held_units(key);
        if held == 0.0 {
            return Vec::new();
        }
        let side = closing_side(held);
        self.loose_stop_gtts()
            .into_iter()
            .filter(|id| {
                self.gtts.book.get(*id).is_some_and(|gtt| {
                    gtt.instrument == key.instrument
                        && gtt.product == key.product
                        && gtt.side == side
                })
            })
            .collect()
    }

    /// The GTT strategy order `order` rests as, while the venue has not fired it.
    pub(super) fn resting_gtt_of(&self, order: OrderId) -> Option<GttId> {
        self.gtts
            .book
            .gtts()
            .iter()
            .find(|gtt| gtt.purpose == GttPurpose::Order(order) && gtt.status.at_venue())
            .map(|gtt| gtt.id)
    }

    /// The fired GTT whose child order is `order`.
    pub(super) fn gtt_child_of(&self, order: OrderId) -> Option<GttChild> {
        let children = &self.venue.gtts.children;
        children.iter().find(|child| child.order == order).copied()
    }

    /// True while a place, modification or cancellation of a GTT waits for the venue's
    /// answer.
    pub(super) fn gtt_request_waits(&self) -> bool {
        self.gtts.book.gtts().iter().any(|gtt| {
            matches!(
                gtt.status,
                GttStatus::Requested | GttStatus::ModifyPending | GttStatus::CancelPending
            )
        })
    }

    /// Put `gtt` in the book as asked for (`Requested`) and send its place; `order` is
    /// the strategy order it rests as.
    ///
    /// Errors: `Exec` when its id is out of sequence (a kernel defect).
    pub(super) fn send_place_gtt(
        &mut self,
        gtt: Gtt,
        order: Option<OrderId>,
    ) -> Result<GttId, KernelError> {
        let terms = GttTerms::of(&gtt);
        let id = self.gtts.book.insert(Gtt {
            status: GttStatus::Requested,
            ..gtt
        })?;
        self.venue.gtts.follows.push(GttFollow::new(id));
        let command = VenueCommand::PlaceGtt {
            gtt: id,
            order,
            terms,
        };
        self.venue.commands.push(command);
        Ok(id)
    }

    /// Send a modification of the resting GTT `id` to `terms` (module docs).
    ///
    /// Errors: `Exec` when the GTT does not rest (a kernel defect).
    pub(super) fn send_modify_gtt(
        &mut self,
        id: GttId,
        terms: GttTerms,
    ) -> Result<(), KernelError> {
        self.gtts.book.apply(id, GttAction::AskModify)?;
        let now = self.venue.clock_ns;
        let follow = self.gtt_follow_mut(id);
        follow.pending = Some(terms);
        follow.modified_at = Some(now);
        follow.refused = false;
        self.venue
            .commands
            .push(VenueCommand::ModifyGtt { gtt: id, terms });
        Ok(())
    }

    /// Ask for the cancellation of GTT `id` in slice `slice` (module docs): sent now when
    /// it rests with nothing waiting, and not in the slice the venue refused the last one;
    /// otherwise wanted, and sent once the request that waits is answered.
    ///
    /// Errors: `Exec` from the book (a kernel defect).
    pub(super) fn ask_cancel_gtt(&mut self, slice: u32, id: GttId) -> Result<(), KernelError> {
        let Some(status) = self.gtts.book.get(id).map(|gtt| gtt.status) else {
            return Ok(());
        };
        if !status.at_venue() || status == GttStatus::CancelPending {
            return Ok(());
        }
        let follow = self.gtt_follow_mut(id);
        follow.cancel_wanted = true;
        let refused_now = follow.cancel_refused_slice == Some(slice);
        if status != GttStatus::Active || refused_now {
            return Ok(());
        }
        follow.cancel_wanted = false;
        self.gtts.book.apply(id, GttAction::AskCancel)?;
        if let Some(gtt) = self.gtts.book.get(id) {
            let terms = GttTerms::of(gtt);
            self.venue
                .commands
                .push(VenueCommand::CancelGtt { gtt: id, terms });
        }
        Ok(())
    }

    /// A GTT whose request was just answered, resting again: the cancellation wanted
    /// meanwhile goes now (module docs).
    ///
    /// Errors: those of `ask_cancel_gtt`.
    pub(super) fn send_wanted_cancel(&mut self, slice: u32, id: GttId) -> Result<(), KernelError> {
        if self
            .gtt_follow(id)
            .is_some_and(|follow| follow.cancel_wanted)
        {
            self.ask_cancel_gtt(slice, id)?;
        }
        Ok(())
    }
}
