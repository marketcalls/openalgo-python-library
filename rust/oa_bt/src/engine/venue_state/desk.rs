//! What a live session has asked its venue and not heard back, and what it keeps on its
//! positions, in its saved state (`venue_state`): the cancellations and modifications
//! that wait for an answer (a modification with its replacement terms), the exits of the
//! positions with the orders sent for them (and the flags of an engine exit's fills), the
//! legs' broker-held stops, the targets fired and the exits made for them, the reprices
//! waiting, what the session follows of its GTTs (`gtts`), and the session's controls
//! (entry locks, the sizing cap).
//!
//! Of the controls, a restore takes only the sizing cap, and only into a session that has
//! none (`Engine::restore_live_state`): the entry locks are written for the live layer to
//! read and are checked for their shape, nothing more.
//!
//! What names a working order names it by its saved id; what names an order that has
//! ended is saved only where it tells something (an exit's orders, a leg's last stop
//! order: `orders::write_ref`), and dropped where nothing looks it up any more, as the
//! session itself forgets it (`Engine::tidy_venue`).

use crate::exec::{FillFlags, OrderBook, OrderId};
use crate::portfolio::PositionKey;

use super::super::venue::{ExitBy, PositionExit, Replacement, VenueDesk};
use super::super::venue_gtts::VenueGtts;
use super::super::venue_legs::LegStop;
use super::super::venue_roles::{EngineExit, OrderRole};
use super::error::LiveStateError;
use super::gtts::{read_venue_gtts, write_venue_gtts};
use super::orders::{write_ref, Ids};
use super::requests::{key_fields, read_key, read_request, write_request};
use super::value::{list, map, Read, StateValue};

/// The parts of the venue desk a saved state restores (module docs).
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct SavedDesk {
    pub(super) cancelling: Vec<OrderId>,
    pub(super) cancels_sent: Vec<(OrderId, u32)>,
    pub(super) modifying: Vec<(OrderId, Replacement)>,
    pub(super) exits: Vec<PositionExit>,
    pub(super) leg_stops: Vec<LegStop>,
    pub(super) target_exits: Vec<(OrderId, u16)>,
    pub(super) fired_targets: Vec<(PositionKey, u16)>,
    pub(super) reprices: Vec<(OrderId, f64)>,
    /// The flags of the engine's exit orders whose fills are flagged.
    pub(super) exit_flags: Vec<(OrderId, FillFlags)>,
    /// What the session follows of its GTTs.
    pub(super) gtts: VenueGtts,
    /// The saved cap on the engine default's sizing equity, rupees.
    pub(super) sizing_cap_rupees: Option<f64>,
}

/// The names of `flags` (`FillFlags::names`).
fn write_flags(flags: FillFlags) -> StateValue {
    list(flags.names(), StateValue::from)
}

/// The flags `write_flags` wrote.
///
/// Errors: `Malformed` for a name that is no flag's.
fn read_flags(read: &Read<'_>) -> Result<FillFlags, LiveStateError> {
    let mut flags = FillFlags::NONE;
    for name in read.items()? {
        flags = flags.with(name.parsed(FillFlags::from_name)?);
    }
    Ok(flags)
}

fn engine_exit_code(exit: EngineExit) -> &'static str {
    match exit {
        EngineExit::SquareOff => OrderRole::SquareOff.code(),
        EngineExit::Risk(reason) => reason.code(),
    }
}

fn write_exit(book: &OrderBook, exit: &PositionExit) -> StateValue {
    let [instrument, product] = key_fields(exit.key);
    let (strategy, engine) = match exit.by {
        ExitBy::Strategy(id) => (write_ref(book, id), StateValue::Null),
        ExitBy::Engine(by) => (StateValue::Null, engine_exit_code(by).into()),
    };
    map([
        instrument,
        product,
        ("strategy_order", strategy),
        ("engine_exit", engine),
        ("sent", list(&exit.sent, |id| write_ref(book, *id))),
        ("limit_rupees", exit.limit_rupees.into()),
        ("flags", write_flags(exit.flags)),
    ])
}

fn read_exit(
    read: &Read<'_>,
    ids: &Ids,
    instruments: usize,
) -> Result<PositionExit, LiveStateError> {
    let by = match (read.maybe("strategy_order")?, read.maybe("engine_exit")?) {
        (Some(order), None) => ExitBy::Strategy(ids.reference(&order)?),
        (None, Some(code)) => ExitBy::Engine(code.parsed(EngineExit::from_code)?),
        _ => {
            return Err(
                read.malformed("names neither the strategy's order nor the engine's exit, or both")
            )
        }
    };
    Ok(PositionExit {
        key: read_key(read, instruments)?,
        by,
        sent: read
            .items_of("sent")?
            .iter()
            .map(|order| ids.reference(order))
            .collect::<Result<_, _>>()?,
        limit_rupees: read.opt_f64_of("limit_rupees")?,
        flags: read_flags(&read.field("flags")?)?,
    })
}

fn write_leg_stop(book: &OrderBook, stop: &LegStop) -> StateValue {
    let [instrument, product] = key_fields(stop.key);
    map([
        instrument,
        product,
        ("order", stop.order.map(|id| write_ref(book, id)).into()),
        ("modified_at_ns", stop.modified_at.into()),
        ("refused", stop.refused.into()),
        ("replaced", stop.replaced.into()),
        ("repriced", stop.repriced.into()),
    ])
}

fn read_leg_stop(
    read: &Read<'_>,
    ids: &Ids,
    instruments: usize,
) -> Result<LegStop, LiveStateError> {
    Ok(LegStop {
        key: read_key(read, instruments)?,
        order: read
            .maybe("order")?
            .map(|order| ids.reference(&order))
            .transpose()?,
        modified_at: read.opt_whole_of("modified_at_ns")?,
        refused: read.bool_of("refused")?,
        replaced: read.bool_of("replaced")?,
        repriced: read.bool_of("repriced")?,
    })
}

/// The modification of working order `id` that waits for the venue's answer.
fn write_replacement(id: OrderId, wanted: &Replacement) -> StateValue {
    map([
        ("order", id.0.into()),
        ("request", write_request(&wanted.request)),
        ("decided_at_ns", wanted.decided_at.into()),
        ("good_until_ns", wanted.good_until.into()),
        ("units", wanted.units.into()),
    ])
}

/// The modification `write_replacement` wrote, for the restored orders `ids`.
///
/// Errors: `Malformed` for a field missing or unusable.
fn read_replacement(
    read: &Read<'_>,
    ids: &Ids,
    instruments: usize,
) -> Result<(OrderId, Replacement), LiveStateError> {
    let replacement = Replacement {
        request: read_request(&read.field("request")?, instruments)?,
        decided_at: read.whole_of("decided_at_ns")?,
        good_until: read.whole_of("good_until_ns")?,
        units: read.f64_of("units")?,
    };
    Ok((ids.working(&read.field("order")?)?, replacement))
}

/// An engine exit order's fill flags as the desk wrote them, for the restored orders
/// `ids`.
///
/// Errors: `Malformed` for a field missing or unusable.
fn read_exit_flags(read: &Read<'_>, ids: &Ids) -> Result<(OrderId, FillFlags), LiveStateError> {
    read_order_and(read, ids, |read| read_flags(&read.field("flags")?))
}

/// A working order the desk names under "order", re-keyed by `ids`, with what `field`
/// reads beside it.
///
/// Errors: `Malformed` for a field missing or unusable.
fn read_order_and<T>(
    read: &Read<'_>,
    ids: &Ids,
    field: impl Fn(&Read<'_>) -> Result<T, LiveStateError>,
) -> Result<(OrderId, T), LiveStateError> {
    Ok((ids.working(&read.field("order")?)?, field(read)?))
}

/// A target of an open leg whose exit has been made, by position.
fn write_fired_target((key, target): &(PositionKey, u16)) -> StateValue {
    let [instrument, product] = key_fields(*key);
    map([instrument, product, ("target", (*target).into())])
}

/// The desk (module docs) of a session whose orders are in `book`.
pub(super) fn write_desk(book: &OrderBook, desk: &VenueDesk) -> StateValue {
    let works = |id: &OrderId| book.get(*id).is_some_and(|order| order.status.is_working());
    let order = |id: OrderId| StateValue::from(id.0);
    map([
        (
            "cancelling",
            list(desk.cancelling.iter().filter(|id| works(id)), |id| {
                order(*id)
            }),
        ),
        (
            "cancels_sent",
            list(
                desk.cancels_sent.iter().filter(|(id, _)| works(id)),
                |(id, slice)| map([("order", order(*id)), ("slice", (*slice).into())]),
            ),
        ),
        (
            "modifying",
            list(
                desk.modifying.iter().filter(|(id, _)| works(id)),
                |(id, wanted)| write_replacement(*id, wanted),
            ),
        ),
        ("exits", list(&desk.exits, |exit| write_exit(book, exit))),
        (
            "leg_stops",
            list(&desk.leg_stops, |stop| write_leg_stop(book, stop)),
        ),
        (
            "target_exits",
            list(
                desk.target_exits.iter().filter(|(id, _)| works(id)),
                |(id, target)| map([("order", order(*id)), ("target", (*target).into())]),
            ),
        ),
        (
            "fired_targets",
            list(&desk.fired_targets, write_fired_target),
        ),
        (
            "reprices",
            list(
                desk.reprices.iter().filter(|(id, _)| works(id)),
                |(id, limit)| map([("order", order(*id)), ("limit_rupees", (*limit).into())]),
            ),
        ),
        (
            "exit_flags",
            list(
                desk.exit_flags.iter().filter(|(id, _)| works(id)),
                |(id, flags)| map([("order", order(*id)), ("flags", write_flags(*flags))]),
            ),
        ),
        ("gtts", write_venue_gtts(book, &desk.gtts)),
        (
            "entry_locks",
            list(&desk.controls.entry_locks, |reason| reason.as_str().into()),
        ),
        ("sizing_cap_rupees", desk.controls.sizing_cap_rupees.into()),
    ])
}

/// The desk `write_desk` wrote, for the restored orders `ids`.
///
/// Errors: `Malformed` for a field missing or unusable, and for an order named that is
/// not a saved working order.
pub(super) fn read_desk(
    read: &Read<'_>,
    ids: &Ids,
    instruments: usize,
) -> Result<SavedDesk, LiveStateError> {
    let each = |key: &str| read.items_of(key);
    // Their shape only: the saved locks are the live layer's to re-apply (module docs).
    for reason in each("entry_locks")? {
        reason.text()?;
    }
    Ok(SavedDesk {
        cancelling: each("cancelling")?
            .iter()
            .map(|order| ids.working(order))
            .collect::<Result<_, _>>()?,
        cancels_sent: each("cancels_sent")?
            .iter()
            .map(|sent| Ok((ids.working(&sent.field("order")?)?, sent.whole_of("slice")?)))
            .collect::<Result<_, LiveStateError>>()?,
        modifying: each("modifying")?
            .iter()
            .map(|wanted| read_replacement(wanted, ids, instruments))
            .collect::<Result<_, _>>()?,
        exits: each("exits")?
            .iter()
            .map(|exit| read_exit(exit, ids, instruments))
            .collect::<Result<_, _>>()?,
        leg_stops: each("leg_stops")?
            .iter()
            .map(|stop| read_leg_stop(stop, ids, instruments))
            .collect::<Result<_, _>>()?,
        target_exits: each("target_exits")?
            .iter()
            .map(|exit| read_order_and(exit, ids, |read| read.whole_of("target")))
            .collect::<Result<_, _>>()?,
        fired_targets: each("fired_targets")?
            .iter()
            .map(|fired| Ok((read_key(fired, instruments)?, fired.whole_of("target")?)))
            .collect::<Result<_, LiveStateError>>()?,
        reprices: each("reprices")?
            .iter()
            .map(|wanted| read_order_and(wanted, ids, |read| read.f64_of("limit_rupees")))
            .collect::<Result<_, _>>()?,
        exit_flags: each("exit_flags")?
            .iter()
            .map(|flagged| read_exit_flags(flagged, ids))
            .collect::<Result<_, _>>()?,
        gtts: read_venue_gtts(&read.field("gtts")?, ids, instruments)?,
        sizing_cap_rupees: read.opt_f64_of("sizing_cap_rupees")?,
    })
}
