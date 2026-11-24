//! The orders of a live session's saved state (`venue_state`): every working order with
//! its role, request, status, units, filled units and fill price, its times, what its
//! fills have been charged so far (a partly filled order is costed on its total) and
//! whether it was sent (a strategy exit that waits for the cancellation of the orders on
//! its position's closing side was not: nothing of it is at the venue, and the restart's
//! order check has no intent to look for), and the commands the adapter has not taken yet.
//!
//! A restored order takes a new id: the session restarts with an empty book and the
//! saved working orders are numbered from o1 in the order of their saved ids (`Ids`). A
//! reference to an order is saved as its id while the order works and as "ended" once it
//! has ended; a restored "ended" reference points at `ENDED_ORDER`, an id no order has,
//! which every rule reads as an order that has ended (a leg's stop that ended without
//! being re-placed is told apart from a leg that has had none). A command for an order
//! that has ended is not saved: the order has nothing left to send.

use crate::charges::OrderCharges;
use crate::exec::{GttBook, GttId, Order, OrderBook, OrderId, OrderStatus};
use crate::market::costs::CostBreakdown;

use super::super::venue::{ChargePart, VenueCommand};
use super::super::venue_roles::{OrderRole, RiskExitReason};
use super::error::LiveStateError;
use super::gtts::{read_terms, write_terms};
use super::requests::{read_instrument, read_order_type, read_request, read_side, write_request};
use super::value::{list, map, Read, StateValue};
use crate::model::Product;

/// How a reference to an order that has ended is saved.
const ENDED: &str = "ended";

/// What a restored reference to an ended order points at: ids start at o1, so no order
/// has this one, and the rules read it as an order that has ended (module docs).
pub(super) const ENDED_ORDER: OrderId = OrderId(0);

/// A reference to order `id`: its id while it works in `book`, "ended" otherwise.
pub(super) fn write_ref(book: &OrderBook, id: OrderId) -> StateValue {
    if book.get(id).is_some_and(|order| order.status.is_working()) {
        id.0.into()
    } else {
        ENDED.into()
    }
}

/// The new ids of the saved working orders (module docs).
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Ids {
    /// (saved id, new id), in saved order.
    pub(super) pairs: Vec<(OrderId, OrderId)>,
}

impl Ids {
    /// The new id of the saved working order `read` names.
    ///
    /// Errors: `Malformed` for anything but the id of a saved working order.
    pub(super) fn working(&self, read: &Read<'_>) -> Result<OrderId, LiveStateError> {
        let saved = OrderId(read.whole()?);
        self.pairs
            .iter()
            .find(|(known, _)| *known == saved)
            .map(|(_, new)| *new)
            .ok_or_else(|| {
                read.malformed(format!("names {saved}, which is not a saved working order"))
            })
    }

    /// The order a reference names: a working order's new id, or `ENDED_ORDER`.
    ///
    /// Errors: those of `working`.
    pub(super) fn reference(&self, read: &Read<'_>) -> Result<OrderId, LiveStateError> {
        if read.is_text(ENDED) {
            return Ok(ENDED_ORDER);
        }
        self.working(read)
    }
}

fn write_cost(cost: &CostBreakdown) -> StateValue {
    map([
        ("brokerage", cost.brokerage.into()),
        ("stt", cost.stt.into()),
        ("exchange_txn", cost.exchange_txn.into()),
        ("sebi_fee", cost.sebi_fee.into()),
        ("stamp_duty", cost.stamp_duty.into()),
        ("gst", cost.gst.into()),
    ])
}

fn read_cost(read: &Read<'_>) -> Result<CostBreakdown, LiveStateError> {
    Ok(CostBreakdown {
        brokerage: read.f64_of("brokerage")?,
        stt: read.f64_of("stt")?,
        exchange_txn: read.f64_of("exchange_txn")?,
        sebi_fee: read.f64_of("sebi_fee")?,
        stamp_duty: read.f64_of("stamp_duty")?,
        gst: read.f64_of("gst")?,
        // Only calculator rounding of NSE currency futures, which live sessions refuse,
        // shows an amount it does not charge.
        ..CostBreakdown::ZERO
    })
}

/// A working order with its role, whether it was `sent` (module docs) and what its fills
/// were charged (`charged`, the session's charges of every order part).
pub(super) fn write_order(
    order: &Order,
    role: Option<OrderRole>,
    sent: bool,
    charged: &[(ChargePart, OrderCharges)],
) -> StateValue {
    let parts = charged.iter().filter(|((id, _), _)| *id == order.id);
    map([
        ("order", order.id.0.into()),
        ("role", role.map(OrderRole::code).into()),
        (
            "reason",
            role.and_then(OrderRole::reason)
                .map(RiskExitReason::code)
                .into(),
        ),
        ("sent", sent.into()),
        ("request", write_request(&order.request)),
        ("status", order.status.code().into()),
        ("decided_at_ns", order.decided_at.into()),
        ("good_until_ns", order.good_until.into()),
        ("quantity_units", order.quantity_units.into()),
        ("filled_units", order.filled_units.into()),
        ("average_fill_rupees", order.average_fill_rupees.into()),
        (
            "charged",
            list(parts, |((_, opening), so_far)| {
                map([
                    ("opening", (*opening).into()),
                    ("units", so_far.units.into()),
                    ("value_rupees", so_far.value_rupees.into()),
                    ("cost", write_cost(&so_far.cost)),
                    ("charged_rupees", so_far.charged_rupees.into()),
                ])
            }),
        ),
    ])
}

fn read_role(read: &Read<'_>) -> Result<Option<OrderRole>, LiveStateError> {
    let Some(role) = read.maybe("role")? else {
        return Ok(None);
    };
    if let Some(reason) = read.maybe("reason")? {
        let risk = OrderRole::RiskExit(reason.code(&RiskExitReason::ALL, RiskExitReason::code)?);
        if !role.is_text(risk.code()) {
            return Err(role.malformed("has a reason, which only a risk exit has"));
        }
        return Ok(Some(risk));
    }
    let plain = [
        OrderRole::Entry,
        OrderRole::Add,
        OrderRole::LegStop,
        OrderRole::Exit,
        OrderRole::SquareOff,
    ];
    role.code(&plain, OrderRole::code).map(Some)
}

/// A restored working order (`read_order`).
pub(super) struct SavedOrder {
    /// The order under its new id.
    pub(super) order: Order,
    /// What it is for (`venue_roles`); `None` for an order the session made no role for.
    pub(super) role: Option<OrderRole>,
    /// Whether it was sent (module docs), checked against the saved exits
    /// (`checks::check_sent`).
    pub(super) sent: bool,
    /// What each part of its fills was charged so far.
    pub(super) charged: Vec<(ChargePart, OrderCharges)>,
}

/// A working order as `write_order` wrote it, under its new id `id`.
///
/// Errors: `Malformed` for a field missing or unusable, a status that does not work, or
/// units that are not above 0 or filled units outside 0 to its units.
pub(super) fn read_order(
    read: &Read<'_>,
    id: OrderId,
    instruments: usize,
) -> Result<SavedOrder, LiveStateError> {
    let status = read.field("status")?;
    let status = status
        .code(&OrderStatus::ALL, OrderStatus::code)
        .and_then(|code| {
            if code.is_working() {
                Ok(code)
            } else {
                Err(status.malformed("is not the status of a working order"))
            }
        })?;
    let charged = read
        .items_of("charged")?
        .iter()
        .map(|part| {
            let so_far = OrderCharges {
                units: part.f64_of("units")?,
                value_rupees: part.f64_of("value_rupees")?,
                cost: read_cost(&part.field("cost")?)?,
                charged_rupees: part.f64_of("charged_rupees")?,
            };
            Ok(((id, part.bool_of("opening")?), so_far))
        })
        .collect::<Result<_, LiveStateError>>()?;
    let quantity_units = read.f64_of("quantity_units")?;
    let filled_units = read.f64_of("filled_units")?;
    if quantity_units <= 0.0 || !(0.0..=quantity_units).contains(&filled_units) {
        return Err(read.malformed(format!(
            "has {filled_units} units filled of {quantity_units}: an order has units above 0 \
             and filled units from 0 to them"
        )));
    }
    let order = Order {
        id,
        request: read_request(&read.field("request")?, instruments)?,
        status,
        decided_at: read.whole_of("decided_at_ns")?,
        good_until: read.whole_of("good_until_ns")?,
        quantity_units,
        filled_units,
        average_fill_rupees: read.f64_of("average_fill_rupees")?,
        reject_code: None,
    };
    Ok(SavedOrder {
        order,
        role: read_role(read)?,
        sent: read.bool_of("sent")?,
        charged,
    })
}

/// A command the adapter has not taken, `None` for one naming an order that has ended or
/// a GTT no longer at the venue (module docs).
pub(super) fn write_command(
    book: &OrderBook,
    gtts: &GttBook,
    command: &VenueCommand,
) -> Option<StateValue> {
    let alive = match (command.order(), command.gtt()) {
        (Some(order), _) => book.get(order).is_some_and(|o| o.status.is_working()),
        (None, Some(gtt)) => gtts.get(gtt).is_some_and(|g| g.status.at_venue()),
        (None, None) => false,
    };
    if !alive {
        return None;
    }
    let terms = |units: f64, order_type: crate::exec::OrderType| {
        [
            ("units", StateValue::from(units)),
            ("order_type", order_type.code().into()),
            ("limit_rupees", order_type.limit_rupees().into()),
            ("trigger_rupees", order_type.trigger_rupees().into()),
        ]
    };
    Some(match *command {
        VenueCommand::Place {
            order,
            instrument,
            product,
            side,
            units,
            order_type,
            role,
        } => {
            let [units, kind, limit, trigger] = terms(units, order_type);
            map([
                ("command", "place".into()),
                ("order", order.0.into()),
                ("instrument", instrument.0.into()),
                ("product", product.code().into()),
                ("side", side.code().into()),
                units,
                kind,
                limit,
                trigger,
                ("role", role.code().into()),
                ("reason", role.reason().map(RiskExitReason::code).into()),
            ])
        }
        VenueCommand::Modify {
            order,
            units,
            order_type,
        } => {
            let [units, kind, limit, trigger] = terms(units, order_type);
            map([
                ("command", "modify".into()),
                ("order", order.0.into()),
                units,
                kind,
                limit,
                trigger,
            ])
        }
        VenueCommand::Cancel { order } => {
            map([("command", "cancel".into()), ("order", order.0.into())])
        }
        VenueCommand::PlaceGtt { gtt, order, terms } => map([
            ("command", "place_gtt".into()),
            ("gtt", gtt.0.into()),
            ("order", order.map(|id| write_ref(book, id)).into()),
            ("terms", write_terms(&terms)),
        ]),
        VenueCommand::ModifyGtt { gtt, terms } => map([
            ("command", "modify_gtt".into()),
            ("gtt", gtt.0.into()),
            ("terms", write_terms(&terms)),
        ]),
        VenueCommand::CancelGtt { gtt, terms } => map([
            ("command", "cancel_gtt".into()),
            ("gtt", gtt.0.into()),
            ("terms", write_terms(&terms)),
        ]),
    })
}

/// The command `write_command` wrote, for the restored orders `ids`.
///
/// Errors: `Malformed` for a field missing or unusable.
pub(super) fn read_command(
    read: &Read<'_>,
    ids: &Ids,
    instruments: usize,
) -> Result<VenueCommand, LiveStateError> {
    let kind = read.field("command")?;
    let gtt = || read.whole_of("gtt").map(GttId);
    let terms = || read_terms(&read.field("terms")?, instruments);
    match kind.text()? {
        "place_gtt" => {
            return Ok(VenueCommand::PlaceGtt {
                gtt: gtt()?,
                order: read
                    .maybe("order")?
                    .map(|order| ids.working(&order))
                    .transpose()?,
                terms: terms()?,
            })
        }
        "modify_gtt" => {
            return Ok(VenueCommand::ModifyGtt {
                gtt: gtt()?,
                terms: terms()?,
            })
        }
        "cancel_gtt" => {
            return Ok(VenueCommand::CancelGtt {
                gtt: gtt()?,
                terms: terms()?,
            })
        }
        _ => {}
    }
    let order = ids.working(&read.field("order")?)?;
    Ok(match kind.text()? {
        "place" => VenueCommand::Place {
            order,
            instrument: read_instrument(&read.field("instrument")?, instruments)?,
            product: read.field("product")?.parsed(Product::from_code)?,
            side: read_side(read, "side")?,
            units: read.f64_of("units")?,
            order_type: read_order_type(read)?,
            role: read_role(read)?.ok_or_else(|| read.malformed("places an order with no role"))?,
        },
        "modify" => VenueCommand::Modify {
            order,
            units: read.f64_of("units")?,
            order_type: read_order_type(read)?,
        },
        "cancel" => VenueCommand::Cancel { order },
        other => return Err(kind.malformed(format!("{other:?} is not a command"))),
    })
}
