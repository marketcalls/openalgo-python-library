//! A live session's strategy as a member of an instrument group (`venue`;
//! 18-multi-strategy 4.1, 4.13 and 4.14, requirements KR10 to KR12): the gates the session
//! sets on a strategy whose instrument other strategies of the session trade too, and its
//! answer to a rejection of a closing order that says the broker may not hold the units.
//!
//! A broker nets every strategy of an account into one position per instrument and
//! product, so the session keeps the strategies of one instrument (its group) on one side,
//! and trading market orders only there, so that every comparison of their summed units
//! with the broker's can run. A group of one sets neither gate: its member trades as a
//! strategy alone does.
//!
//! | Call | When | What |
//! |---|---|---|
//! | `set_group_side(instrument, product, side)` | any time | the side (`GroupSide`) the other members hold or are entering, or `None`: while set, an order that opens or adds (`OrderRole::opens`: an entry, an add, a reversal) on the other side is rejected at submission and at modification, `group_other_side` |
//! | `set_shared(instrument, product, shared)` | any time | while shared, a strategy order that rests (LIMIT, SL, SL-M), an entry or an exit, is rejected at submission and at modification, `shared_resting` |
//! | `apply_margin_reject(order)` | as `apply_reject` | a closing order (a leg's stop, a strategy exit, an engine exit) whose rejection the session takes as the broker not holding its units (any rejection of a closing order on an instrument of a group of two or more, 18-multi-strategy 11.0a): it ends rejected, reject code `venue`, and its position is held (below) |
//!
//! Exits are never gated by the side: a reduce-only order and an order against the
//! position within the room its closing side has (`venue_roles`) only close. No gate
//! judges an order the engine makes (the leg's stop, the engine's exits, the price rule's
//! LIMIT orders: `venue_legs`, `venue_exit_orders`, `venue_protect`), since those take
//! no submission check. The gates come first among a session's checks (`venue_gates`):
//! they stand whatever the entry locks do, so the strategy reads what keeps the order out.
//!
//! After a margin rejection the broker may not hold, on that side, the units the kernel
//! believes the position has: every closing order of it would meet the same refusal, or,
//! with margin to spare, open the other side. So the rejection holds the position, and
//! while it is held the kernel sends nothing that closes it, or would cover more of it,
//! by itself:
//!
//! - a leg's stop so ended is neither exited at market nor placed again (requirement R3
//!   does not apply, `venue_legs`), and no stop of the position is placed or raised: one
//!   that still works may be cut down, cancelled or have its trigger moved, never given
//!   more units (`venue_stop_terms`; so the put-back of R4 after a partial exit ended
//!   unfilled waits too);
//! - an exit of the position that waits (one that waited for the rejected order, or one
//!   made while held: the strategy's, a target's, a risk rule's) stays unsent, and nothing
//!   is cancelled for it (`venue_owner`);
//! - no engine exit is made for it at the later opens (`venue_exits`: the risk lock's,
//!   the time exit, the square-off), whatever closing order was rejected.
//!
//! The hold ends when the session asks for an exit of the position (`request_exit`, which
//! also sends at once an engine exit the hold kept waiting) or `protect`s it (which moves
//! the waiting exits on and puts the stop back), or when a new leg opens there. What the
//! session sends itself (`reprice_exit`) and the venue's results are taken as always. The
//! hold is not part of the saved state (`venue_state`): a restart's comparison with the
//! broker decides afresh.

use crate::exec::{
    ExecError, InstrumentId, Order, OrderAction, OrderId, OrderRequest, OrderType, RejectCode,
};
use crate::model::{Product, Side};
use crate::portfolio::PositionKey;

use super::error::KernelError;
use super::event::{EventStamp, OrderEventKind};
use super::kernel::Engine;
use super::venue_legs::StopEnd;
use super::venue_roles::OrderRole;

/// The side a position is held on: the side the other members of an instrument group hold
/// or are entering (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupSide {
    /// Bought: a long position.
    Long,
    /// Sold: a short position.
    Short,
}

impl GroupSide {
    /// Every side's code, in the order of the variants.
    pub const CODES: [&'static str; 2] = ["long", "short"];

    /// The code the bindings use: "long" or "short", as a trade's direction reads.
    pub fn code(self) -> &'static str {
        match self {
            GroupSide::Long => "long",
            GroupSide::Short => "short",
        }
    }

    /// The side named by `code`; `None` for any other.
    pub fn from_code(code: &str) -> Option<GroupSide> {
        [GroupSide::Long, GroupSide::Short]
            .into_iter()
            .find(|side| side.code() == code)
    }

    /// The side an order on `side` opens or adds to.
    fn opened_by(side: Side) -> GroupSide {
        match side {
            Side::Buy => GroupSide::Long,
            Side::Sell => GroupSide::Short,
        }
    }
}

/// What the live layer has set on the members of instrument groups, and the positions a
/// margin rejection holds (module docs); positions in the order they were set.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct GroupDesk {
    /// The side the other members hold or are entering, per position.
    sides: Vec<(PositionKey, GroupSide)>,
    /// The positions on an instrument several strategies of the session trade.
    shared: Vec<PositionKey>,
    /// The positions a margin rejection of a closing order holds: the kernel sends nothing
    /// that closes them by itself.
    held: Vec<PositionKey>,
}

impl GroupDesk {
    /// True while a margin rejection holds `key` (module docs).
    pub(super) fn holds(&self, key: PositionKey) -> bool {
        self.held.contains(&key)
    }

    /// Hold `key` after a margin rejection of one of its closing orders.
    fn hold(&mut self, key: PositionKey) {
        if !self.holds(key) {
            self.held.push(key);
        }
    }

    /// End the hold of `key`: the session asked for an exit or for `protect`, or a new leg
    /// opened. True when `key` was held.
    pub(super) fn release(&mut self, key: PositionKey) -> bool {
        let was_held = self.holds(key);
        self.held.retain(|held| *held != key);
        was_held
    }

    fn side_of(&self, key: PositionKey) -> Option<GroupSide> {
        let found = self.sides.iter().find(|(known, _)| *known == key);
        found.map(|(_, side)| *side)
    }

    fn is_shared(&self, key: PositionKey) -> bool {
        self.shared.contains(&key)
    }
}

/// The reject code of the gate that refuses a strategy order of `role` on `side`, priced as
/// `order_type`, on a position whose group holds or enters `group_side` and which is
/// `shared` (module docs); `None` when it passes.
fn group_refusal(
    group_side: Option<GroupSide>,
    shared: bool,
    role: OrderRole,
    side: Side,
    order_type: OrderType,
) -> Option<RejectCode> {
    let other_side = group_side.is_some_and(|held| held != GroupSide::opened_by(side));
    if role.opens() && other_side {
        return Some(RejectCode::GroupOtherSide);
    }
    if shared && order_type != OrderType::Market {
        return Some(RejectCode::SharedResting);
    }
    None
}

fn not_closing(order: OrderId, role: Option<OrderRole>) -> KernelError {
    let role = role.map_or("an order the session never sent", OrderRole::code);
    KernelError::Exec(ExecError::InvalidRequest {
        reason: format!(
            "order {order} is {role}: only a closing order (a leg's stop, an exit, the \
             engine's exit) takes a margin rejection; pass the venue's rejection of an entry \
             or add to apply_reject"
        ),
    })
}

impl Engine {
    /// Set the side the other members of the group of `instrument` in `product` hold or
    /// are entering, or clear it with `None` (module docs).
    ///
    /// Errors: `Venue(NotExternal)` in a backtest; `InvalidConfig` naming `symbol` for an
    /// instrument the session does not trade.
    pub fn set_group_side(
        &mut self,
        instrument: InstrumentId,
        product: Product,
        side: Option<GroupSide>,
    ) -> Result<(), KernelError> {
        let key = self.group_key(instrument, product)?;
        let sides = &mut self.venue.group.sides;
        sides.retain(|(known, _)| *known != key);
        if let Some(side) = side {
            sides.push((key, side));
        }
        Ok(())
    }

    /// Mark `instrument` in `product` as traded by several strategies of the session, or
    /// by this one alone (module docs).
    ///
    /// Errors: those of `set_group_side`.
    pub fn set_shared(
        &mut self,
        instrument: InstrumentId,
        product: Product,
        shared: bool,
    ) -> Result<(), KernelError> {
        let key = self.group_key(instrument, product)?;
        let positions = &mut self.venue.group.shared;
        positions.retain(|known| *known != key);
        if shared {
            positions.push(key);
        }
        Ok(())
    }

    /// Record that the venue rejected the closing order `order` in a way that says the
    /// broker may not hold its units (module docs): it ends `Rejected` with the reject
    /// code `venue`, a partly filled order keeping its fills, and its position is held:
    /// the kernel sends nothing that closes it, or covers more of it, by itself until the
    /// session asks. Returns the stamp, as `apply_reject` does.
    ///
    /// Errors: those of `apply_reject` (`Venue(NotExternal)` in a backtest,
    /// `Venue(AtBarClose)` while paused at a bar close, `UnknownOrder`,
    /// `Venue(OrderEnded)`, `Venue(NotSent)`); `Exec(InvalidRequest)` for an order that
    /// opens or adds, which keeps working.
    pub fn apply_margin_reject(&mut self, order: OrderId) -> Result<EventStamp, KernelError> {
        self.result_slice()?;
        let request = &self.live_order(order)?.request;
        let key = PositionKey {
            instrument: request.instrument,
            product: request.product,
        };
        let role = self.order_role(order);
        if !role.is_some_and(|role| !role.opens()) {
            return Err(not_closing(order, role));
        }
        // Held before the order ends, so that what its end moves on (the exits that waited
        // for it, the stop's units) already finds the position held.
        self.venue.group.hold(key);
        let kind = OrderEventKind::Rejected(RejectCode::Venue);
        self.end_at_venue(order, OrderAction::Reject, kind, StopEnd::MarginRejected)
    }

    /// The units of `wanted` that `key`'s working leg stop `order` may be asked to cover:
    /// while a margin rejection holds the position, at most what it covers now (module
    /// docs).
    pub(super) fn stop_units_allowed(&self, key: PositionKey, order: &Order, wanted: f64) -> f64 {
        if self.venue.group.holds(key) {
            return wanted.min(order.quantity_units - order.filled_units);
        }
        wanted
    }

    /// The gates of a group on the strategy's `request` of `units` (module docs), at
    /// submission or at the modification of `replacing`. A backtest, and a position no gate
    /// is set on, pass.
    ///
    /// Errors: the reject code of the gate that refuses the order.
    pub(super) fn group_gates(
        &self,
        request: &OrderRequest,
        units: f64,
        replacing: Option<OrderId>,
    ) -> Result<(), RejectCode> {
        if !self.is_external() {
            return Ok(());
        }
        let key = PositionKey {
            instrument: request.instrument,
            product: request.product,
        };
        let group = &self.venue.group;
        let (side, shared) = (group.side_of(key), group.is_shared(key));
        if side.is_none() && !shared {
            return Ok(());
        }
        let role = self.submission_role(request, units, replacing);
        match group_refusal(side, shared, role, request.side, request.order_type) {
            Some(code) => Err(code),
            None => Ok(()),
        }
    }

    /// The position a group control names.
    ///
    /// Errors: those of `set_group_side`.
    fn group_key(
        &self,
        instrument: InstrumentId,
        product: Product,
    ) -> Result<PositionKey, KernelError> {
        self.require_session()?;
        self.spec(instrument)?;
        Ok(PositionKey {
            instrument,
            product,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::RiskExitReason;

    #[test]
    fn a_group_refuses_the_other_side_to_orders_that_open_and_resting_orders_when_shared() {
        let (long, short) = (Some(GroupSide::Long), Some(GroupSide::Short));
        let market = OrderType::Market;
        let resting = OrderType::Limit {
            limit_rupees: 495.0,
        };
        let other_side = Some(RejectCode::GroupOtherSide);
        let shared_resting = Some(RejectCode::SharedResting);
        let risk_exit = OrderRole::RiskExit(RiskExitReason::DailyLoss);
        // (group side, shared, role, side, order type, refusal)
        let table = [
            (long, false, OrderRole::Entry, Side::Buy, market, None),
            (
                long,
                false,
                OrderRole::Entry,
                Side::Sell,
                market,
                other_side,
            ),
            (
                short,
                false,
                OrderRole::Entry,
                Side::Buy,
                market,
                other_side,
            ),
            (short, false, OrderRole::Add, Side::Sell, market, None),
            (short, false, OrderRole::Add, Side::Buy, market, other_side),
            (long, false, OrderRole::Exit, Side::Sell, market, None),
            (short, false, OrderRole::Exit, Side::Buy, market, None),
            (None, false, OrderRole::Entry, Side::Sell, resting, None),
            (None, true, OrderRole::Entry, Side::Buy, market, None),
            (
                None,
                true,
                OrderRole::Entry,
                Side::Buy,
                resting,
                shared_resting,
            ),
            (
                None,
                true,
                OrderRole::Exit,
                Side::Sell,
                resting,
                shared_resting,
            ),
            (
                long,
                true,
                OrderRole::Entry,
                Side::Sell,
                resting,
                other_side,
            ),
            (long, true, OrderRole::Exit, Side::Sell, market, None),
            (long, true, risk_exit, Side::Sell, market, None),
        ];
        for (group_side, shared, role, side, order_type, refusal) in table {
            assert_eq!(
                group_refusal(group_side, shared, role, side, order_type),
                refusal,
                "{group_side:?} shared {shared}: {role:?} {side:?} {order_type:?}"
            );
        }
    }

    #[test]
    fn a_hold_is_released_for_its_position_only_and_says_whether_it_held() {
        let key = |index| PositionKey {
            instrument: InstrumentId(index),
            product: Product::Mis,
        };
        let mut desk = GroupDesk::default();
        desk.hold(key(0));
        desk.hold(key(1));
        desk.hold(key(0));
        assert_eq!(desk.held, [key(0), key(1)], "one hold per position");
        assert!(desk.release(key(0)));
        assert!(!desk.release(key(0)), "released already");
        assert!(!desk.holds(key(0)));
        assert!(desk.holds(key(1)));
    }
}
