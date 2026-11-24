//! What a carried state's GTTs and GTT orders must hold (`carry`), checked once it has
//! been read whole and before any of it is applied, so a state no session could have
//! carried is refused naming where it breaks (`LiveStateError::Malformed`).
//!
//! | Part | Holds |
//! |---|---|
//! | `orders` | each is a strategy GTT order (`validity="gtt"`) resting as exactly one carried GTT |
//! | `gtts` | each rests at the venue; a stop GTT holds the stop of an open CNC or NRML leg, on its closing side, for at most the units held, one per position; an order's GTT names a carried GTT order |

use crate::exec::{Gtt, GttId, GttPurpose, Validity};
use crate::model::Side;
use crate::portfolio::{Position, PositionKey};
use crate::rms::LegState;

use super::super::gtt::held_as_gtt;
use super::error::LiveStateError;
use super::orders::SavedOrder;
use super::value::Read;

/// How far a stop GTT's units may stray above the position's, relative to them (the
/// tranches' float rounding, `checks`).
const UNITS_RELATIVE_TOLERANCE: f64 = 1.0e-9;

/// A carried GTT, read whole: its saved id and the GTT under it.
pub(super) struct CarriedGtt {
    pub(super) saved_id: GttId,
    pub(super) gtt: Gtt,
}

/// Check the carried GTTs (`gtts`, each with the value it was read from) against the
/// carried positions, legs and orders (module docs).
///
/// Errors: `Malformed` naming the first that does not hold.
pub(super) fn check_carried_gtts(
    gtts: &[(Gtt, Read<'_>)],
    positions: &[Position],
    legs: &[(PositionKey, LegState)],
    orders: &[SavedOrder],
) -> Result<Vec<CarriedGtt>, LiveStateError> {
    let mut stops: Vec<PositionKey> = Vec::new();
    let mut checked = Vec::new();
    for (gtt, read) in gtts {
        if !gtt.status.at_venue() {
            return Err(read.malformed("is not at the venue: a carried GTT rests there"));
        }
        match gtt.purpose {
            GttPurpose::Protect => {
                let key = PositionKey {
                    instrument: gtt.instrument,
                    product: gtt.product,
                };
                check_stop(gtt, read, positions, legs)?;
                if stops.contains(&key) {
                    return Err(read.malformed("is a second stop GTT of its position"));
                }
                stops.push(key);
            }
            GttPurpose::Order(id) => {
                if !orders.iter().any(|saved| saved.order.id == id) {
                    return Err(read.malformed("names no carried GTT order"));
                }
            }
        }
        checked.push(CarriedGtt {
            saved_id: gtt.id,
            gtt: gtt.clone(),
        });
    }
    for saved in orders {
        let rests_as = gtts
            .iter()
            .filter(|(gtt, _)| gtt.purpose == GttPurpose::Order(saved.order.id))
            .count();
        if saved.order.request.validity != Validity::Gtt || rests_as != 1 {
            return Err(LiveStateError::Malformed {
                path: "live_state.orders".to_string(),
                reason: format!(
                    "carries order {} that does not rest as one GTT: only GTT orders carry",
                    saved.order.id
                ),
            });
        }
    }
    Ok(checked)
}

/// A stop GTT holds an open CNC or NRML leg's stop on its closing side, for at most the
/// units held.
fn check_stop(
    gtt: &Gtt,
    read: &Read<'_>,
    positions: &[Position],
    legs: &[(PositionKey, LegState)],
) -> Result<(), LiveStateError> {
    let key = PositionKey {
        instrument: gtt.instrument,
        product: gtt.product,
    };
    let position = positions
        .iter()
        .find(|position| position.key == key && !position.is_flat());
    let has_leg = legs.iter().any(|(leg, _)| *leg == key);
    let Some(position) = position.filter(|_| has_leg && held_as_gtt(key.product)) else {
        return Err(read.malformed("is the stop GTT of no open CNC or NRML leg"));
    };
    let held = position.quantity_units;
    let closes = if held > 0.0 { Side::Sell } else { Side::Buy };
    let most = held.abs() * (1.0 + UNITS_RELATIVE_TOLERANCE);
    if gtt.side != closes || gtt.quantity_units > most {
        return Err(read.malformed(format!(
            "sells or buys {} {} units, and its position holds {held}",
            gtt.side.code(),
            gtt.quantity_units
        )));
    }
    Ok(())
}
