//! What an external session asks of its venue (`venue`, "Commands out"): one
//! `VenueCommand` per order placed, modified or cancelled, and per GTT placed, modified or
//! cancelled (`venue_gtts`), taken by the adapter with `Engine::take_commands` in the
//! order they were made.

use crate::exec::{GttId, InstrumentId, OrderId, OrderType};
use crate::model::{Product, Side};

use super::venue_gtts::GttTerms;
use super::venue_roles::OrderRole;

/// What an external session asks of its venue. Quantities are units, prices rupees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VenueCommand {
    /// Place a DAY order.
    Place {
        /// The kernel order the venue's results name.
        order: OrderId,
        /// The instrument, by registration index.
        instrument: InstrumentId,
        /// CNC, MIS or NRML.
        product: Product,
        /// Buy or sell.
        side: Side,
        /// Units to trade, never lots.
        units: f64,
        /// MARKET, LIMIT, SL or SL-M, with its limit and trigger prices.
        order_type: OrderType,
        /// What the order is for (`venue_roles`).
        role: OrderRole,
    },
    /// Replace a working order's units, type, limit and trigger (a full replacement).
    Modify {
        /// The kernel order to replace.
        order: OrderId,
        /// The replacement's units.
        units: f64,
        /// The replacement's type, with its limit and trigger prices.
        order_type: OrderType,
    },
    /// Cancel a working order.
    Cancel {
        /// The kernel order to cancel.
        order: OrderId,
    },
    /// Place a SINGLE GTT (design 20-positional-live 3.6): a CNC or NRML leg's stop, or a
    /// strategy's GTT order.
    PlaceGtt {
        /// The kernel GTT the venue's answers name.
        gtt: GttId,
        /// The strategy's GTT order it rests as; `None` for a leg's stop.
        order: Option<OrderId>,
        /// Its trigger, child limit and units.
        terms: GttTerms,
    },
    /// Replace a GTT's trigger, child limit and units (a full replacement).
    ModifyGtt {
        /// The kernel GTT to replace.
        gtt: GttId,
        /// The replacement's terms.
        terms: GttTerms,
    },
    /// Cancel a GTT.
    CancelGtt {
        /// The kernel GTT to cancel.
        gtt: GttId,
        /// Its terms as they rest.
        terms: GttTerms,
    },
}

impl VenueCommand {
    /// The kernel order the command is for; `None` for a GTT's command.
    pub fn order(&self) -> Option<OrderId> {
        match *self {
            VenueCommand::Place { order, .. }
            | VenueCommand::Modify { order, .. }
            | VenueCommand::Cancel { order } => Some(order),
            _ => None,
        }
    }

    /// The kernel GTT the command is for; `None` for an order's command.
    pub fn gtt(&self) -> Option<GttId> {
        match *self {
            VenueCommand::PlaceGtt { gtt, .. }
            | VenueCommand::ModifyGtt { gtt, .. }
            | VenueCommand::CancelGtt { gtt, .. } => Some(gtt),
            _ => None,
        }
    }
}
