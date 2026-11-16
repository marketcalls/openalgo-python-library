//! The venue of a run: who fills its orders (PRD 15-phase2-plan section 2).
//!
//! | Venue | Built by | Who fills |
//! |---|---|---|
//! | `Simulated` | `Engine::new`, every backtest | the kernel, against the bars (steps 2 and 3) |
//! | `External` | `Engine::new_session`, a live session | the broker, through the adapter |
//!
//! In an external session the kernel decides and checks exactly as in a backtest (every
//! submission check and reject code, the risk rules, the cost model) and records every
//! fill through the one fill path (`fills::execute_fill`), but matches nothing and ends no
//! order itself: steps 2 and 3 fill nothing, no order expires, and step 8 invents no
//! closing fill nor an expiry settlement (a position still open at the end is reported
//! open). The parts:
//!
//! - **Commands out** (`venue_orders`). An order that passes the checks is sent at once as
//!   `VenueCommand::Place`, except an exit (below). The engine's own withdrawals send
//!   `Cancel` at most once in a slice; `cancel` sends it at once, again after the venue
//!   refused it, but never twice while one waits for the venue's answer. The order works
//!   on until the venue's result, since a fill may come first. `modify` sends `Modify`
//!   and the order keeps its old terms until the venue confirms the new ones
//!   (`apply_modify`) or refuses them. The adapter takes the commands with
//!   `Engine::take_commands`, in the order they were made.
//! - **Exits** (`venue_owner`, `venue_exit_orders`). An exit is the strategy's `exit()` or
//!   `reduce()` (a reduce-only MARKET order), a leg's target (`venue_ticks`), or one of
//!   the engine's exits (`venue_exits`: the daily loss limit and drawdown halt, the
//!   square-off, the time exit; and the session's own). An exit first cancels every other
//!   working order of its position on the closing side and is sent once they have all
//!   ended, in orders of at most the freeze quantity; a partial exit cuts the leg's stop
//!   down to what stays open instead of cancelling it (`venue_legs`). The strategy's exits
//!   of a position add up as in a backtest, each sized on what the earlier ones leave; an
//!   engine exit takes over from them and closes the whole position.
//! - **Reduce-only orders never reverse** (`venue_cover`). The orders working on a
//!   position's closing side may not hold more units than it has: a reduce-only order
//!   beyond that is rejected (`exit_cover`), and any other order against the position
//!   beyond it is an entry (`venue_roles`). Reduce-only orders are cut down or cancelled
//!   when the position shrinks, and a fill of an order sent to close beyond the position
//!   is recorded only up to it (`VenueError::WouldReverse`).
//! - **Results in**: fills (`venue_fills`), rejections and cancellations (`venue_ends`),
//!   and the venue's answers to the cancellations and modifications it was sent
//!   (`venue_answers`); the session's last prices (`venue_ticks`).
//! - **The clock** (`venue_clock`). What a backtest does at the open of a slice (the
//!   engine's exits, the timers) happens when the session reaches that open, the close of
//!   the slice before it. At a session close the kernel sends `Cancel` for the DAY orders
//!   a backtest would expire, and the close completes once the venue has ended them all;
//!   the session cannot finish while an order works.
//! - **Quantities.** An order's units are resolved when it is sent: a fraction of the
//!   position then held, rounded down to the venue quantity. A live session trades whole
//!   shares and lots: `quantity_mode="fractional"` is refused.
//! - **What the fill path cannot record** is refused at submission (`venue_checks`), and a
//!   fill is checked whole before anything of it is recorded. The free cash a submission
//!   is checked against counts what the session's working entries would block.
//! - **Roles** (`venue_roles`). Every order sent carries what it is for: an entry, an add,
//!   the strategy's exit, the engine's risk exit with its reason, the square-off.
//! - **The session's controls** (`venue_session`): an exit the live layer asks for, a
//!   cancellation while waiting, the entry locks, the cap on the sizing equity, the daily
//!   loss measure, and the stop of the session where it stands. **The entry gates** a
//!   session adds at submission (`venue_gates`): the entry locks, nothing in flight on the
//!   instrument, a position at most the freeze quantity.
//! - **Leg rules** (`stop=`, `target=`, trailing and breakeven, decision D31). The broker
//!   holds a leg's stop as an order (`venue_legs`): placed when the leg has a stop level,
//!   its units and trigger kept on the leg's, a partial exit sent once it has been cut down
//!   to what stays open, and the leg protected again when it ends without a fill. The
//!   kernel holds the targets, the trail and the breakeven on the session's prices
//!   (`venue_ticks`, tick mode): a target exits through the position's exit owner, a trail
//!   or a breakeven moves the broker's stop. The session's own calls for a leg (`protect`,
//!   `reprice_exit`, an exit at a limit) are in `venue_protect`. Leg rules are MIS only in
//!   a session: CNC and NRML legs are held as GTTs.
//! - **Not built yet** (`KernelError::NotBuilt`, `venue_requests`): leg rules on CNC and NRML, GTT orders,
//!   futures series and rebalancing.
//!
//! The contract for the live layer: every fill of an order is delivered (`apply_fill`)
//! before its terminal result (`apply_cancel`, `apply_reject`); after it the order takes
//! no more fills. A result the kernel cannot record is a `VenueError` (`venue_error`);
//! it changes nothing, but for `WouldReverse`, which records the fill up to the position.

use crate::charges::OrderCharges;
use crate::exec::{InstrumentId, OrderId, OrderRequest, OrderType};
use crate::model::{Product, Side, TsNanos};
use crate::portfolio::PositionKey;

use super::clock::SliceInfo;
use super::config::{QuantityMode, RunConfig};
use super::error::KernelError;
use super::event::{EventStamp, Step};
use super::kernel::Engine;
use super::venue_legs::LegStop;
use super::venue_roles::{EngineExit, OrderRole};
use super::venue_session::SessionControls;

/// Who fills a run's orders.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Venue {
    /// The kernel matches every order against the bars (a backtest).
    #[default]
    Simulated,
    /// A broker fills the orders; the adapter feeds its results back (a live session).
    External,
}

impl Venue {
    /// The name the bindings report: "simulated" or "external".
    pub fn code(self) -> &'static str {
        match self {
            Venue::Simulated => "simulated",
            Venue::External => "external",
        }
    }
}

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
}

/// Who asked for an exit (`venue_owner`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum ExitBy {
    /// The strategy's exit order (`exit()`, `reduce()`).
    Strategy(OrderId),
    /// One of the engine's exits: the square-off or a risk exit (`venue_roles`).
    Engine(EngineExit),
}

/// One exit of a position, the strategy's or the engine's (`venue_owner`).
#[derive(Clone, Debug, PartialEq)]
pub(super) struct PositionExit {
    pub(super) key: PositionKey,
    pub(super) by: ExitBy,
    /// The orders sent for it, a strategy's own order first; empty while it waits for the
    /// orders it cancels.
    pub(super) sent: Vec<OrderId>,
    /// The limit, rupees, of an engine exit the session asked for at a price
    /// (`venue_protect`); `None` for a MARKET exit.
    pub(super) limit_rupees: Option<f64>,
}

/// A modification sent to the venue and not yet confirmed.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Replacement {
    pub(super) request: OrderRequest,
    pub(super) decided_at: TsNanos,
    pub(super) good_until: TsNanos,
    pub(super) units: f64,
}

/// Where an order the engine sends now is decided and stamped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct SendAt {
    pub(super) slice: SliceInfo,
    pub(super) step: Step,
    pub(super) decided_at: TsNanos,
}

impl SendAt {
    pub(super) fn stamp(&self) -> EventStamp {
        Engine::stamp(&self.slice, self.step)
    }
}

/// An order's fills that reduce its position, or those that open or add (a reversal has
/// both), for costing them once on their total (`charges::OrderCharges`).
pub(super) type ChargePart = (OrderId, bool);

/// The venue of a run, and what an external one has been asked and not yet answered.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct VenueDesk {
    pub(super) venue: Venue,
    /// Commands not yet taken by the adapter, in the order they were made.
    pub(super) commands: Vec<VenueCommand>,
    /// Working orders whose cancellation has been sent and not yet answered.
    pub(super) cancelling: Vec<OrderId>,
    /// The slice index in which each working order's last cancellation was sent: one the
    /// venue refused is sent again by the engine's own withdrawals in a later slice, not
    /// in that one (`withdraw_order`), and by `cancel` at once (`cancel_on_request`).
    pub(super) cancels_sent: Vec<(OrderId, u32)>,
    /// Working orders whose modification has been sent and not yet answered.
    pub(super) modifying: Vec<(OrderId, Replacement)>,
    /// The exits of the positions, in the order they were made: a position has the
    /// strategy's, one after another, or the engine's alone.
    pub(super) exits: Vec<PositionExit>,
    /// What each order's fills have been charged so far.
    pub(super) charged: Vec<(ChargePart, OrderCharges)>,
    /// The role of every order the session accepted, in id order (`venue_roles`).
    pub(super) roles: Vec<(OrderId, OrderRole)>,
    /// The session's entry locks and sizing cap (`venue_session`).
    pub(super) controls: SessionControls,
    /// The legs' broker-held stops, in the order the legs opened (`venue_legs`).
    pub(super) leg_stops: Vec<LegStop>,
    /// The target each of the engine's target exits fills, by order (`venue_ticks`).
    pub(super) target_exits: Vec<(OrderId, u16)>,
    /// The LIMIT, rupees, a reprice wants for an order whose last modification waits for
    /// the venue's answer, sent with that answer (`venue_protect`).
    pub(super) reprices: Vec<(OrderId, f64)>,
    /// The latest instant the session has been given, UTC nanoseconds: a tick's or a
    /// fill's time, or the open of a slice it reached. The throttle of a stop's
    /// modifications reads it (`venue_legs`).
    pub(super) clock_ns: TsNanos,
}

impl VenueDesk {
    /// The venue of a live session.
    pub(super) fn external() -> VenueDesk {
        VenueDesk {
            venue: Venue::External,
            ..VenueDesk::default()
        }
    }

    /// What the order part's fills have been charged so far (zero for its first fill).
    pub(super) fn charged_so_far(&mut self, part: ChargePart) -> &mut OrderCharges {
        let slot = match self.charged.iter().position(|(known, _)| *known == part) {
            Some(slot) => slot,
            None => {
                self.charged.push((part, OrderCharges::default()));
                self.charged.len() - 1
            }
        };
        &mut self.charged[slot].1
    }
}

/// Refuse a session setting the venue cannot honour: a live session trades whole
/// shares and lots.
///
/// Errors: `InvalidConfig` naming `quantity_mode` for fractional quantities.
pub(super) fn check_session_config(config: &RunConfig) -> Result<(), KernelError> {
    if config.quantity_mode == QuantityMode::Fractional {
        return Err(KernelError::InvalidConfig {
            field: "quantity_mode",
            reason: "a live session trades whole shares and lots: use quantity_mode=\"venue\""
                .to_string(),
        });
    }
    Ok(())
}

/// True for an exit: a reduce-only MARKET order (`exit()`, `reduce()`).
pub(super) fn is_exit(request: &OrderRequest) -> bool {
    request.reduce_only && request.order_type == OrderType::Market
}

impl Engine {
    /// Who fills this run's orders.
    pub fn venue(&self) -> Venue {
        self.venue.venue
    }

    /// The commands made since the last call, in the order they were made (none in a
    /// backtest).
    pub fn take_commands(&mut self) -> Vec<VenueCommand> {
        std::mem::take(&mut self.venue.commands)
    }

    /// True in a live session: a broker fills the orders.
    pub(super) fn is_external(&self) -> bool {
        self.venue.venue == Venue::External
    }

    /// True for a strategy's exit that waits for the orders it cancels: not sent yet.
    pub(super) fn is_unsent(&self, id: OrderId) -> bool {
        self.venue
            .exits
            .iter()
            .any(|exit| exit.by == ExitBy::Strategy(id) && exit.sent.is_empty())
    }

    /// True when the engine's exit holds `key` (`venue_owner`).
    pub(super) fn has_engine_exit(&self, key: PositionKey) -> bool {
        self.venue
            .exits
            .iter()
            .any(|exit| exit.key == key && matches!(exit.by, ExitBy::Engine(_)))
    }

    /// True when an exit of `key` waits for the orders it cancels.
    pub(super) fn has_waiting_exit(&self, key: PositionKey) -> bool {
        self.venue
            .exits
            .iter()
            .any(|exit| exit.key == key && exit.sent.is_empty())
    }

    /// Forget what the venue was asked about orders that have ended, and the exits whose
    /// orders have all ended.
    pub(super) fn tidy_venue(&mut self) {
        let book = &self.book;
        let works = |id: &OrderId| book.get(*id).is_some_and(|o| o.status.is_working());
        self.venue.cancelling.retain(works);
        self.venue.cancels_sent.retain(|(id, _)| works(id));
        self.venue.modifying.retain(|(id, _)| works(id));
        self.venue.target_exits.retain(|(id, _)| works(id));
        self.venue.reprices.retain(|(id, _)| works(id));
        self.venue
            .exits
            .retain(|exit| match (exit.by, exit.sent.is_empty()) {
                (ExitBy::Strategy(id), true) => works(&id),
                (ExitBy::Engine(_), true) => true,
                (_, false) => exit.sent.iter().any(works),
            });
    }
}
