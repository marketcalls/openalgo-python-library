//! What a saved state must hold across its parts (`venue_state`), checked once it has been
//! read whole and before any of it is applied. The readers check each field's shape; these
//! check that the parts agree as the session that wrote them kept them, so a state edited
//! by hand into one no session could have been in is refused naming where it breaks
//! (`LiveStateError::Malformed`), not restored into a session that would act on it.
//!
//! | Part | Holds |
//! |---|---|
//! | `positions` | an open position's tranches hold its units; a flat one has none |
//! | `legs` | each leg is of an open position, on its side, holding its units, and every open position has its leg (each entry opens one: `fills::update_leg`) |
//! | `orders` | `sent` is false for a strategy exit that waits for its cancels and for a pre-open decision's order waiting for the open (`desk.orders_at_open`), and only for them (units and filled units: `orders::read_order`) |
//! | `desk.leg_stops` | a leg's working stop is a `leg_stop` order of the leg's position |

use crate::exec::OrderId;
use crate::model::Side;
use crate::portfolio::{Position, PositionKey};
use crate::rms::LegState;

use super::super::venue::{ExitBy, PositionExit};
use super::super::venue_legs::LegStop;
use super::super::venue_roles::OrderRole;
use super::error::LiveStateError;
use super::orders::{SavedOrder, ENDED_ORDER};
use super::value::Read;

/// How far the units of a position's tranches, or of its leg, may stray from the units it
/// holds, relative to them: the tranches shrink in proportion on every reduction
/// (`portfolio::Position`), which leaves float rounding many orders of magnitude below
/// this, while a state no session wrote strays by whole units.
const UNITS_RELATIVE_TOLERANCE: f64 = 1.0e-9;

/// True when `units` are the `held` units, within the rounding above.
fn same_units(units: f64, held: f64) -> bool {
    (units - held).abs() <= UNITS_RELATIVE_TOLERANCE * held.abs().max(1.0)
}

/// The refusal for item `index` of the list at `path` (map keys from the root of `read`).
fn refuse(read: &Read<'_>, path: &[&str], index: usize, reason: String) -> LiveStateError {
    let mut at = read.clone();
    for key in path {
        match at.field(key) {
            Ok(next) => at = next,
            Err(error) => return error,
        }
    }
    match at.items() {
        Ok(items) => items.get(index).unwrap_or(&at).malformed(reason),
        Err(error) => error,
    }
}

/// The parts of a saved state these checks read, as read (`venue_state::Saved`).
pub(super) struct Parts<'a> {
    pub(super) positions: &'a [Position],
    pub(super) legs: &'a [(PositionKey, LegState)],
    pub(super) orders: &'a [SavedOrder],
    pub(super) exits: &'a [PositionExit],
    pub(super) leg_stops: &'a [LegStop],
    /// A pre-open decision's orders waiting for the session's open (`pre_open`).
    pub(super) at_open: &'a [OrderId],
}

/// Check that the parts of the state at `read` agree (module docs).
///
/// Errors: `Malformed` naming the first item that does not.
pub(super) fn check_parts(read: &Read<'_>, parts: &Parts<'_>) -> Result<(), LiveStateError> {
    check_positions(read, parts)?;
    check_legs(read, parts)?;
    check_sent(read, parts)?;
    check_leg_stops(read, parts)
}

fn check_positions(read: &Read<'_>, parts: &Parts<'_>) -> Result<(), LiveStateError> {
    for (index, position) in parts.positions.iter().enumerate() {
        let held = position.quantity_units.abs();
        let in_tranches: f64 = position.tranches.iter().map(|t| t.open_units).sum();
        let open = !position.is_flat();
        if !same_units(in_tranches, held) || (!open && !position.tranches.is_empty()) {
            let reason = format!("holds {held} units, and its tranches {in_tranches}");
            return Err(refuse(read, &["positions"], index, reason));
        }
        let has_leg = parts.legs.iter().any(|(key, _)| *key == position.key);
        if open && !has_leg {
            let reason = "is open and has no leg: every entry opens one".to_string();
            return Err(refuse(read, &["positions"], index, reason));
        }
    }
    Ok(())
}

fn check_legs(read: &Read<'_>, parts: &Parts<'_>) -> Result<(), LiveStateError> {
    for (index, (key, leg)) in parts.legs.iter().enumerate() {
        let position = parts
            .positions
            .iter()
            .find(|position| position.key == *key && !position.is_flat());
        let Some(position) = position else {
            let reason = "is the leg of no open position".to_string();
            return Err(refuse(read, &["legs"], index, reason));
        };
        let long = position.quantity_units > 0.0;
        let held = position.quantity_units.abs();
        if (leg.context.side == Side::Buy) != long || !same_units(leg.held_units(), held) {
            let reason = format!(
                "holds {} units {}, and its position {}",
                leg.held_units(),
                leg.context.side.code(),
                position.quantity_units
            );
            return Err(refuse(read, &["legs"], index, reason));
        }
    }
    Ok(())
}

/// True when `id` is a strategy exit that waits for its cancels: never sent.
fn waits_unsent(exits: &[PositionExit], id: OrderId) -> bool {
    exits
        .iter()
        .any(|exit| exit.by == ExitBy::Strategy(id) && exit.sent.is_empty())
}

fn check_sent(read: &Read<'_>, parts: &Parts<'_>) -> Result<(), LiveStateError> {
    for (index, saved) in parts.orders.iter().enumerate() {
        let id = saved.order.id;
        let unsent = waits_unsent(parts.exits, id) || parts.at_open.contains(&id);
        if saved.sent == unsent {
            let reason = format!(
                "is saved {}sent, which only a strategy exit waiting for its cancels and an \
                 order waiting for the session's open are not",
                if saved.sent { "" } else { "not " }
            );
            return Err(refuse(read, &["orders"], index, reason));
        }
    }
    Ok(())
}

fn check_leg_stops(read: &Read<'_>, parts: &Parts<'_>) -> Result<(), LiveStateError> {
    for (index, stop) in parts.leg_stops.iter().enumerate() {
        let Some(id) = stop.order.filter(|id| *id != ENDED_ORDER) else {
            continue;
        };
        let order = parts.orders.iter().find(|saved| saved.order.id == id);
        let is_its_stop = order.is_some_and(|saved| {
            let request = &saved.order.request;
            let key = PositionKey {
                instrument: request.instrument,
                product: request.product,
            };
            saved.role == Some(OrderRole::LegStop) && key == stop.key
        });
        if !is_its_stop {
            let reason = "names a working order that is not a leg stop of its position".to_string();
            return Err(refuse(read, &["desk", "leg_stops"], index, reason));
        }
    }
    Ok(())
}
