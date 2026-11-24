//! The GTTs of a live session's saved state (`venue_state`; design 20-positional-live
//! 4.1, unit N1 item N1-4): every GTT of the session's book with its id, state, terms and
//! purpose, and what the session follows of them (`venue_gtts`): the modification or
//! cancellation waiting for an answer, the legs' stop GTTs, and the children of the GTTs
//! the venue fired.
//!
//! A same-day restart restores the book whole, so every GTT keeps its id: the restored
//! session's book is empty before the restore, and the GTTs are put back from g1 in id
//! order. A GTT names the strategy order it rests as by the order's reference
//! (`orders::write_ref`), a child by its order's.

use crate::exec::{Gtt, GttBook, GttId, GttLegRole, GttPurpose, GttStatus, OrderBook};

use super::super::venue::VenueCommand;
use super::super::venue_gtts::{GttChild, GttFollow, GttLegStop, GttTerms, VenueGtts};
use super::error::LiveStateError;
use super::orders::{write_ref, Ids};
use super::requests::{key_fields, read_key, read_side};
use super::value::{list, map, Read, StateValue};

/// A GTT's terms: instrument, product, child side, units, trigger, child limit, and
/// whether the trigger lies below the price it was placed from.
pub(super) fn write_terms(terms: &GttTerms) -> StateValue {
    map([
        ("instrument", terms.instrument.0.into()),
        ("product", terms.product.code().into()),
        ("side", terms.side.code().into()),
        ("units", terms.units.into()),
        ("trigger_rupees", terms.trigger_rupees.into()),
        ("limit_rupees", terms.limit_rupees.into()),
        ("trigger_below", terms.trigger_below.into()),
    ])
}

/// The terms `write_terms` wrote.
///
/// Errors: `Malformed` for a field missing or unusable.
pub(super) fn read_terms(read: &Read<'_>, instruments: usize) -> Result<GttTerms, LiveStateError> {
    let key = read_key(read, instruments)?;
    Ok(GttTerms {
        instrument: key.instrument,
        product: key.product,
        side: read_side(read, "side")?,
        units: read.f64_of("units")?,
        trigger_rupees: read.f64_of("trigger_rupees")?,
        limit_rupees: read.f64_of("limit_rupees")?,
        trigger_below: read.bool_of("trigger_below")?,
    })
}

fn role_code(role: GttLegRole) -> &'static str {
    role.code()
}

/// A GTT's command the adapter has not taken (`orders::write_command`): "place_gtt",
/// "modify_gtt" or "cancel_gtt" with the GTT's id and its terms, and for a place the
/// strategy order it rests as, named in `orders`; `None` for an order's command.
pub(super) fn write_gtt_command(orders: &OrderBook, command: &VenueCommand) -> Option<StateValue> {
    let head = |kind: &str, gtt: GttId| [("command", kind.into()), ("gtt", gtt.0.into())];
    Some(match *command {
        VenueCommand::PlaceGtt { gtt, order, terms } => {
            let [kind, id] = head("place_gtt", gtt);
            let order = order.map(|id| write_ref(orders, id)).into();
            map([kind, id, ("order", order), ("terms", write_terms(&terms))])
        }
        VenueCommand::ModifyGtt { gtt, terms } => {
            let [kind, id] = head("modify_gtt", gtt);
            map([kind, id, ("terms", write_terms(&terms))])
        }
        VenueCommand::CancelGtt { gtt, terms } => {
            let [kind, id] = head("cancel_gtt", gtt);
            map([kind, id, ("terms", write_terms(&terms))])
        }
        _ => return None,
    })
}

/// The GTT's command `write_gtt_command` wrote, its order re-keyed by `ids`; `None` for
/// an order's command.
///
/// Errors: `Malformed` for a field missing or unusable.
pub(super) fn read_gtt_command(
    read: &Read<'_>,
    ids: &Ids,
    instruments: usize,
) -> Result<Option<VenueCommand>, LiveStateError> {
    let kind = read.field("command")?;
    let gtt = || read.whole_of("gtt").map(GttId);
    let terms = || read_terms(&read.field("terms")?, instruments);
    Ok(Some(match kind.text()? {
        "place_gtt" => VenueCommand::PlaceGtt {
            gtt: gtt()?,
            order: read
                .maybe("order")?
                .map(|order| ids.working(&order))
                .transpose()?,
            terms: terms()?,
        },
        "modify_gtt" => VenueCommand::ModifyGtt {
            gtt: gtt()?,
            terms: terms()?,
        },
        "cancel_gtt" => VenueCommand::CancelGtt {
            gtt: gtt()?,
            terms: terms()?,
        },
        _ => return Ok(None),
    }))
}

/// A GTT of `book`'s run, whose strategy order is named in `orders`.
pub(super) fn write_gtt(orders: &OrderBook, gtt: &Gtt) -> StateValue {
    let order = match gtt.purpose {
        GttPurpose::Protect => StateValue::Null,
        GttPurpose::Order(id) => write_ref(orders, id),
    };
    map([
        ("gtt", gtt.id.0.into()),
        ("status", gtt.status.code().into()),
        ("role", role_code(gtt.leg.role).into()),
        ("order", order),
        ("terms", write_terms(&GttTerms::of(gtt))),
        ("created_on", gtt.created_on.into()),
        ("eligible_from_ns", gtt.eligible_from.into()),
        ("last_modified_slice", gtt.last_modified_slice.into()),
    ])
}

/// A GTT `write_gtt` wrote, its order re-keyed by `ids`.
///
/// Errors: `Malformed` for a field missing or unusable, a role other than "stop" or
/// "order", a stop with an order or an order with none.
pub(super) fn read_gtt(
    read: &Read<'_>,
    ids: &Ids,
    instruments: usize,
) -> Result<Gtt, LiveStateError> {
    let roles = [GttLegRole::Stop, GttLegRole::Order];
    let role = read.field("role")?.code(&roles, role_code)?;
    let purpose = match (role, read.maybe("order")?) {
        (GttLegRole::Stop, None) => GttPurpose::Protect,
        (GttLegRole::Order, Some(order)) => GttPurpose::Order(ids.reference(&order)?),
        _ => return Err(read.malformed("a stop GTT names no order and an order's names one")),
    };
    let terms = read_terms(&read.field("terms")?, instruments)?;
    Ok(Gtt {
        id: GttId(read.whole_of("gtt")?),
        instrument: terms.instrument,
        product: terms.product,
        side: terms.side,
        quantity_units: terms.units,
        leg: terms.leg(role),
        purpose,
        status: read
            .field("status")?
            .code(&GttStatus::ALL, GttStatus::code)?,
        created_on: read.whole_of("created_on")?,
        eligible_from: read.whole_of("eligible_from_ns")?,
        last_modified_slice: read.opt_whole_of("last_modified_slice")?,
    })
}

/// The book of GTTs `write_gtt` wrote, every GTT under its saved id.
///
/// Errors: `Malformed` for a GTT that cannot be read or is not in id order from g1.
pub(super) fn read_gtt_book(
    read: &Read<'_>,
    ids: &Ids,
    instruments: usize,
) -> Result<GttBook, LiveStateError> {
    let mut book = GttBook::default();
    for item in read.items()? {
        let gtt = read_gtt(&item, ids, instruments)?;
        book.insert(gtt)
            .map_err(|_| item.malformed("is not the next GTT in id order from g1"))?;
    }
    Ok(book)
}

fn write_follow(follow: &GttFollow) -> StateValue {
    map([
        ("gtt", follow.gtt.0.into()),
        ("pending", follow.pending.as_ref().map(write_terms).into()),
        ("modified_at_ns", follow.modified_at.into()),
        ("refused", follow.refused.into()),
        ("cancel_wanted", follow.cancel_wanted.into()),
        ("cancel_refused_slice", follow.cancel_refused_slice.into()),
    ])
}

fn read_follow(read: &Read<'_>, instruments: usize) -> Result<GttFollow, LiveStateError> {
    Ok(GttFollow {
        gtt: GttId(read.whole_of("gtt")?),
        pending: read
            .maybe("pending")?
            .map(|terms| read_terms(&terms, instruments))
            .transpose()?,
        modified_at: read.opt_whole_of("modified_at_ns")?,
        refused: read.bool_of("refused")?,
        cancel_wanted: read.bool_of("cancel_wanted")?,
        cancel_refused_slice: read.opt_whole_of("cancel_refused_slice")?,
    })
}

/// What the session follows of its GTTs (module docs), its orders named in `orders`.
pub(super) fn write_venue_gtts(orders: &OrderBook, gtts: &VenueGtts) -> StateValue {
    map([
        ("follows", list(&gtts.follows, write_follow)),
        (
            "legs",
            list(&gtts.legs, |leg| {
                let [instrument, product] = key_fields(leg.key);
                map([
                    instrument,
                    product,
                    ("gtt", leg.gtt.map(|id| id.0).into()),
                    ("rejections", u32::from(leg.rejections).into()),
                ])
            }),
        ),
        (
            "children",
            list(&gtts.children, |child| {
                map([
                    ("gtt", child.gtt.0.into()),
                    ("order", write_ref(orders, child.order)),
                    ("triggered_at_ns", child.triggered_at.into()),
                ])
            }),
        ),
    ])
}

/// What `write_venue_gtts` wrote, its orders re-keyed by `ids`.
///
/// Errors: `Malformed` for a field missing or unusable.
pub(super) fn read_venue_gtts(
    read: &Read<'_>,
    ids: &Ids,
    instruments: usize,
) -> Result<VenueGtts, LiveStateError> {
    Ok(VenueGtts {
        follows: read
            .items_of("follows")?
            .iter()
            .map(|follow| read_follow(follow, instruments))
            .collect::<Result<_, _>>()?,
        legs: read
            .items_of("legs")?
            .iter()
            .map(|leg| {
                Ok(GttLegStop {
                    key: read_key(leg, instruments)?,
                    gtt: leg.opt_whole_of("gtt")?.map(GttId),
                    rejections: leg.whole_of("rejections")?,
                })
            })
            .collect::<Result<_, LiveStateError>>()?,
        children: read
            .items_of("children")?
            .iter()
            .map(|child| {
                Ok(GttChild {
                    gtt: GttId(child.whole_of("gtt")?),
                    order: ids.reference(&child.field("order")?)?,
                    triggered_at: child.whole_of("triggered_at_ns")?,
                })
            })
            .collect::<Result<_, LiveStateError>>()?,
    })
}
