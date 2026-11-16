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
//!   `VenueCommand::Place`, except an exit (below). `cancel`, and the engine's own
//!   withdrawals, send `Cancel`; the order works on until the venue's result, since a fill
//!   may come first. `modify` sends `Modify` and the order keeps its old terms until the
//!   venue confirms the new ones (`apply_modify`) or refuses them. The adapter takes the
//!   commands with `Engine::take_commands`, in the order they were made.
//! - **Exits** (`venue_owner`). An exit is the strategy's `exit()` or `reduce()` (a
//!   reduce-only MARKET order) or one of the engine's exits (`venue_exits`: the daily loss
//!   limit and drawdown halt, the square-off, the time exit). No working order counts as
//!   covering a position: an exit first cancels every working order of its position on
//!   the closing side and is sent once they have all ended, for the position then open.
//!   Meanwhile it is the position's one exit owner: no second exit order is made.
//! - **Results in** (`venue_fills`): fills, rejections, cancellations, and the venue's
//!   answers to a cancellation or modification it refused.
//! - **The clock** (`venue_clock`). What a backtest does at the open of a slice (the
//!   engine's exits, the timers) happens when the session reaches that open, the close of
//!   the slice before it. At a session close the kernel sends `Cancel` for the DAY orders
//!   a backtest would expire, and the close completes once the venue has ended them all;
//!   the session cannot finish while an order works.
//! - **Quantities.** An order's units are resolved when it is sent: a fraction of the
//!   position then held, rounded down to the venue quantity. A live session trades whole
//!   shares and lots: `quantity_mode="fractional"` is refused.
//! - **What the fill path cannot record** is refused at submission (`venue_checks`), and a
//!   fill is checked whole before anything of it is recorded.
//! - **Not built yet** (`KernelError::NotBuilt`): leg rules (`stop=`, `target=`, trailing
//!   and breakeven, held by the broker and on ticks in the next unit, decision D31), GTT
//!   orders, futures series and rebalancing.
//!
//! The contract for the live layer: every fill of an order is delivered (`apply_fill`)
//! before its terminal result (`apply_cancel`, `apply_reject`); after it the order takes
//! no more fills. A result the kernel cannot record is a `VenueError` (`venue_error`).

use crate::charges::OrderCharges;
use crate::exec::{ExecError, InstrumentId, Order, OrderId, OrderRequest, OrderType, Validity};
use crate::model::{Product, Side, TsNanos};
use crate::portfolio::PositionKey;
use crate::rms::ExitReason;

use super::clock::SliceInfo;
use super::config::{QuantityMode, RunConfig};
use super::error::KernelError;
use super::event::{EventStamp, Step};
use super::kernel::Engine;

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
    /// One of the engine's exits, with the backtest's reason.
    Engine(ExitReason),
}

/// The one exit of a position.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct ExitOwner {
    pub(super) key: PositionKey,
    pub(super) by: ExitBy,
    /// The orders sent for it; empty while it waits for the orders it cancels.
    pub(super) sent: Vec<OrderId>,
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
    /// Working orders whose modification has been sent and not yet answered.
    pub(super) modifying: Vec<(OrderId, Replacement)>,
    /// The exit owner of each position that has one, in the order they were made.
    pub(super) exits: Vec<ExitOwner>,
    /// What each order's fills have been charged so far.
    pub(super) charged: Vec<(ChargePart, OrderCharges)>,
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

fn not_built(feature: &'static str, instead: &'static str) -> KernelError {
    KernelError::NotBuilt { feature, instead }
}

fn invalid(reason: &str) -> KernelError {
    KernelError::Exec(ExecError::InvalidRequest {
        reason: reason.to_string(),
    })
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

    /// Refuse, in an external session, what the venue cannot carry yet (module docs).
    ///
    /// Errors: `NotBuilt` for leg rules, a GTT order and a rebalance order.
    pub(super) fn check_venue_request(&self, request: &OrderRequest) -> Result<(), KernelError> {
        if !self.is_external() {
            return Ok(());
        }
        if !request.leg_rules.is_empty() {
            return Err(not_built(
                "stop=, target=, trailing and breakeven in a live session",
                "exit from on_bar with self.exit() for now: broker-held stops and \
                 engine-held targets arrive with the next release",
            ));
        }
        if request.validity == Validity::Gtt {
            return Err(not_built(
                "a GTT order in a live session",
                "live trading is MIS only: place a DAY order",
            ));
        }
        if request.rebalance {
            return Err(not_built(
                "rebalance() in a live session",
                "trade each symbol with self.target(), the sales before the buys",
            ));
        }
        Ok(())
    }

    /// Refuse, in an external session, a modification the venue cannot carry: an order
    /// that has partly filled, one whose last modification is not answered yet, and an
    /// exit not sent yet.
    ///
    /// Errors: `NotBuilt` for a partly filled order; `Exec(InvalidRequest)` for the others.
    pub(super) fn check_venue_replacement(&self, order: &Order) -> Result<(), KernelError> {
        if !self.is_external() {
            return Ok(());
        }
        if order.filled_units > 0.0 {
            return Err(not_built(
                "modifying a partly filled order in a live session",
                "cancel the order and place a new one for the rest",
            ));
        }
        if self.venue.modifying.iter().any(|(id, _)| *id == order.id) {
            return Err(invalid(
                "the order's last modification waits for the venue's answer: modify it again \
                 once the venue has confirmed or refused it",
            ));
        }
        if self.is_unsent(order.id) {
            return Err(invalid(
                "the exit waits for the orders it cancels and has not been sent: cancel it and \
                 exit again instead",
            ));
        }
        Ok(())
    }

    /// Refuse, in an external session, a replacement left with no units (an exit with
    /// nothing left to reduce): the venue takes no order of none.
    ///
    /// Errors: `Exec(InvalidRequest)` saying to cancel the order instead.
    pub(super) fn check_replacement_units(&self, units: f64) -> Result<(), KernelError> {
        if self.is_external() && units <= 0.0 {
            return Err(invalid(
                "the replacement has nothing left to reduce: cancel the order instead",
            ));
        }
        Ok(())
    }

    /// True for a strategy's exit that waits for the orders it cancels: not sent yet.
    pub(super) fn is_unsent(&self, id: OrderId) -> bool {
        self.venue
            .exits
            .iter()
            .any(|owner| owner.by == ExitBy::Strategy(id) && owner.sent.is_empty())
    }

    /// The exit owner of `key`, if it has one.
    pub(super) fn exit_owner(&self, key: PositionKey) -> Option<&ExitOwner> {
        self.venue.exits.iter().find(|owner| owner.key == key)
    }

    /// Forget what the venue was asked about orders that have ended, and the exit owners
    /// whose orders have all ended.
    pub(super) fn tidy_venue(&mut self) {
        let book = &self.book;
        let works = |id: &OrderId| book.get(*id).is_some_and(|o| o.status.is_working());
        self.venue.cancelling.retain(works);
        self.venue.modifying.retain(|(id, _)| works(id));
        self.venue.exits.retain(|owner| {
            let own = match owner.by {
                ExitBy::Strategy(id) => works(&id),
                ExitBy::Engine(_) => true,
            };
            own && (owner.sent.is_empty() || owner.sent.iter().any(works))
        });
    }
}
