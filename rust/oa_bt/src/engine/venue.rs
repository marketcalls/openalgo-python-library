//! The venue of a run: who fills its orders (PRD 15-phase2-plan section 2).
//!
//! | Venue | Built by | Who fills |
//! |---|---|---|
//! | `Simulated` | `Engine::new`, every backtest | the kernel, against the bars (steps 2 and 3) |
//! | `External` | `Engine::new_session`, a live session | the broker, through the adapter |
//!
//! In an external session the kernel decides and checks exactly as in a backtest (every
//! submission check and reject code, the risk rules, the cost model) and records every
//! fill through the one fill path (`fills::execute_fill`), but matches nothing itself:
//! steps 2 and 3 fill no order, and step 8 invents no closing fill for an open MIS
//! position nor an expiry settlement (a position still open at the end is reported open).
//!
//! - **Commands out.** An order that passes the submission checks is handed over at once
//!   as `VenueCommand::Place`; `modify` replaces the order and sends `Modify`; `cancel`,
//!   and the engine's own withdrawals (the entries cancelled when step 6 locks them, the
//!   MIS entries still working once the entry window has closed), send `Cancel`. A
//!   cancelled order works on until the venue confirms it (`apply_cancel`), since a fill
//!   may come first. The adapter takes the commands with `Engine::take_commands`, in the
//!   order they were made.
//! - **Engine orders.** The exits a backtest fills without an order (SPEC-21) are orders
//!   here (`venue_exits`): reduce-only MARKET orders with the backtest's exit reason.
//! - **Results in.** `apply_fill`, `apply_reject` and `apply_cancel` (`venue_fills`).
//! - **Quantities.** The venue needs every order's units when it is sent, so they are
//!   resolved at submission: a fraction of the position held at the decision, rounded
//!   down to the venue quantity (`exit(fraction=)`), and an order that only reduces is
//!   held to the part of the position no working order on the other side already covers,
//!   so the working exits never exceed the position. An order left with no units ends
//!   `Canceled` at once and sends nothing, as a backtest cancels it at its fill bar. A
//!   live session trades whole shares and lots: `quantity_mode="fractional"` is refused.
//! - **Not built yet** (`KernelError::NotBuilt`): leg rules (`stop=`, `target=`, trailing
//!   and breakeven, held by the broker and on ticks in the next unit, decision D31), GTT
//!   orders, futures series and rebalancing.
//!
//! A result the venue sends that cannot be recorded is a `VenueError` (`venue_error`).

use crate::exec::{
    ExecError, InstrumentId, Order, OrderAction, OrderId, OrderRequest, OrderStatus, OrderType,
    QuantitySpec, Validity,
};
use crate::model::{Product, Side};
use crate::portfolio::PositionKey;
use crate::sizing::round_down_to_venue;

use super::config::{QuantityMode, RunConfig};
use super::error::KernelError;
use super::event::{EventStamp, OrderEventKind};
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
    /// Place a DAY order. `order_type` carries the limit and the trigger.
    Place {
        order: OrderId,
        instrument: InstrumentId,
        product: Product,
        side: Side,
        units: f64,
        order_type: OrderType,
    },
    /// Replace a working order's units, type, limit and trigger (a full replacement).
    Modify {
        order: OrderId,
        units: f64,
        order_type: OrderType,
    },
    /// Cancel a working order.
    Cancel { order: OrderId },
}

/// The venue of a run, and what an external one has been asked and not yet confirmed.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct VenueDesk {
    venue: Venue,
    /// Commands not yet taken by the adapter, in the order they were made.
    commands: Vec<VenueCommand>,
    /// Working orders whose cancellation has been sent and not yet confirmed.
    cancelling: Vec<OrderId>,
}

impl VenueDesk {
    /// The venue of a live session.
    pub(super) fn external() -> VenueDesk {
        VenueDesk {
            venue: Venue::External,
            ..VenueDesk::default()
        }
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
    /// that has partly filled.
    ///
    /// Errors: `NotBuilt` for a partly filled order.
    pub(super) fn check_venue_replacement(&self, order: &Order) -> Result<(), KernelError> {
        if self.is_external() && order.filled_units > 0.0 {
            return Err(not_built(
                "modifying a partly filled order in a live session",
                "cancel the order and place a new one for the rest",
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
            return Err(KernelError::Exec(ExecError::InvalidRequest {
                reason: "the replacement has nothing left to reduce: cancel the order instead"
                    .to_string(),
            }));
        }
        Ok(())
    }

    /// The units an order is sent with: in a backtest `units` as resolved; in an external
    /// session an order that only reduces gets the units it reduces now (module docs).
    /// `replacing` is the order a modification replaces, which covers nothing of the
    /// position once replaced.
    pub(super) fn venue_units(
        &self,
        request: &OrderRequest,
        units: f64,
        replacing: Option<OrderId>,
    ) -> f64 {
        if !self.is_external() || !request.reduce_only {
            return units;
        }
        let key = PositionKey {
            instrument: request.instrument,
            product: request.product,
        };
        let held = self.held_units(key);
        let reduces = held != 0.0 && (held > 0.0) != (request.side == Side::Buy);
        if !reduces {
            return 0.0;
        }
        let wanted = match request.quantity {
            QuantitySpec::PositionFraction(fraction) => {
                let lot = self
                    .instruments
                    .get(request.instrument.index())
                    .map_or(1, |spec| spec.lot_size_units);
                round_down_to_venue(held.abs() * fraction, lot, false)
            }
            _ => units,
        };
        let open = held.abs() - self.units_closing(key, held, replacing).min(held.abs());
        wanted.min(open)
    }

    /// Units of the working orders on `key` on the other side of a position of `held`
    /// signed units, `except` one: what they would close if they filled.
    pub(super) fn units_closing(
        &self,
        key: PositionKey,
        held: f64,
        except: Option<OrderId>,
    ) -> f64 {
        let closing_side = if held > 0.0 { Side::Sell } else { Side::Buy };
        self.book
            .working()
            .filter(|order| {
                Some(order.id) != except
                    && order.request.instrument == key.instrument
                    && order.request.product == key.product
                    && order.request.side == closing_side
            })
            .map(|order| order.quantity_units - order.filled_units)
            .sum()
    }

    /// In an external session, after an order has passed the submission checks: send it,
    /// or end it `Canceled` when it has no units left to reduce. A backtest sends nothing.
    ///
    /// Errors: those of `fills::end_order`.
    pub(super) fn send_order(&mut self, stamp: EventStamp, id: OrderId) -> Result<(), KernelError> {
        if !self.is_external() {
            return Ok(());
        }
        let order = self.book.get(id).ok_or(KernelError::UnknownOrder(id))?;
        if order.quantity_units <= 0.0 {
            let kind = OrderEventKind::Canceled;
            return self.end_order(stamp, id, OrderAction::Cancel, kind);
        }
        let command = VenueCommand::Place {
            order: id,
            instrument: order.request.instrument,
            product: order.request.product,
            side: order.request.side,
            units: order.quantity_units,
            order_type: order.request.order_type,
        };
        self.venue.commands.push(command);
        Ok(())
    }

    /// In an external session, after a modification: send the replacement.
    pub(super) fn send_modify(&mut self, id: OrderId) {
        if !self.is_external() {
            return;
        }
        if let Some(order) = self.book.get(id) {
            let command = VenueCommand::Modify {
                order: id,
                units: order.quantity_units,
                order_type: order.request.order_type,
            };
            self.venue.commands.push(command);
        }
    }

    /// Withdraw a working order: a backtest ends it `Canceled` now; an external session
    /// sends `Cancel` once and leaves the order working until the venue confirms it.
    /// Returns the order's state afterwards.
    ///
    /// Errors: those of `fills::end_order`.
    pub(super) fn withdraw_order(
        &mut self,
        stamp: EventStamp,
        id: OrderId,
    ) -> Result<OrderStatus, KernelError> {
        if !self.is_external() {
            self.end_order(stamp, id, OrderAction::Cancel, OrderEventKind::Canceled)?;
            return Ok(OrderStatus::Canceled);
        }
        let status = self
            .book
            .get(id)
            .ok_or(KernelError::UnknownOrder(id))?
            .status;
        let book = &self.book;
        let cancelling = &mut self.venue.cancelling;
        cancelling.retain(|asked| book.get(*asked).is_some_and(|o| o.status.is_working()));
        if status.is_working() && !cancelling.contains(&id) {
            cancelling.push(id);
            self.venue.commands.push(VenueCommand::Cancel { order: id });
        }
        Ok(status)
    }
}
