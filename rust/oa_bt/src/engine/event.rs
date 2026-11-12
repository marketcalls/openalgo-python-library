//! The event stream of a run (decision SPEC-21): order, position and risk events, each
//! stamped with the slice and the step of 03-architecture 4.2 that produced it.
//!
//! The stream is the smallest one that lets `on_fill`, `on_order`, `on_position` and
//! `on_risk_event` be tested:
//!
//! - An order that passes the submission checks emits `Submitted` in step 7 of its
//!   decision slice; one that fails them emits `Rejected` alone. A backtest emits no
//!   `Accepted`.
//! - A fill emits `Filled` and then the position event, in the fill's step.
//! - An engine-held exit (stop, target) emits its risk event and then the position
//!   event, and no order event; a square-off or a roll emits only the position event.
//! - Events are in order of occurrence: by slice, then step, then instrument in
//!   registration order, then order id.
//! - A change of the strategy's trading state (ACTIVE, REDUCING, HALTED) is an event of
//!   its own, written right after the risk event that caused it, or in step 1 when a
//!   session boundary ends REDUCING. Strategies receive it through `on_risk_event` as
//!   `trading_state_changed`.
//! - A GTT writes the events of its life (`exec::gtt_event`, decision GTT-8). Its child
//!   order writes no order event: a protective GTT's exit is engine-held, and a
//!   strategy GTT's fill is reported on the strategy's order.

use crate::exec::{GttEvent, InstrumentId, OrderId, RejectCode};
use crate::model::{Product, TsNanos};
use crate::portfolio::PositionChange;
use crate::rms::{RiskEvent, TradingState, TradingStateCause};

/// The eight steps of one time slice (03-architecture 4.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Step {
    /// 1. Session open and reference events.
    SessionOpen = 1,
    /// 2. Queued orders and timers fill at the open.
    OpenFills = 2,
    /// 3. Resting orders and leg levels against the bar.
    Levels = 3,
    /// 4. Trailing and breakeven ratchets.
    Ratchets = 4,
    /// 5. Mark to the close.
    Mark = 5,
    /// 6. Aggregate rules on the synchronised closes.
    AggregateRules = 6,
    /// 7. Callbacks and `on_bar`.
    Callbacks = 7,
    /// 8. Session close.
    SessionClose = 8,
}

impl Step {
    /// The step number, 1 to 8.
    pub fn number(self) -> u8 {
        self as u8
    }
}

/// When an event happened: the grid slice (by index and by its open instant) and the
/// step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EventStamp {
    /// Index of the slice in the run's grid.
    pub slice: u32,
    /// Open of the slice's base bar, UTC nanoseconds (the slice's name).
    pub slice_ts: TsNanos,
    /// The step.
    pub step: Step,
}

/// What happened to an order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OrderEventKind {
    /// Passed the submission checks.
    Submitted,
    /// Acknowledged by the venue (live only).
    Accepted,
    /// Refused by a check.
    Rejected(RejectCode),
    /// A stop order's trigger was reached.
    Triggered,
    /// Filled: quantity in units and price in rupees of this fill.
    Filled {
        quantity_units: f64,
        price_rupees: f64,
    },
    /// Cancelled by the strategy or the engine.
    Canceled,
    /// Its DAY validity ended unfilled.
    Expired,
    /// The strategy replaced its request (a full replacement, same id).
    Modified,
}

impl OrderEventKind {
    /// The event name.
    pub fn code(self) -> &'static str {
        match self {
            OrderEventKind::Submitted => "Submitted",
            OrderEventKind::Accepted => "Accepted",
            OrderEventKind::Rejected(_) => "Rejected",
            OrderEventKind::Triggered => "Triggered",
            OrderEventKind::Filled { .. } => "Filled",
            OrderEventKind::Canceled => "Canceled",
            OrderEventKind::Expired => "Expired",
            OrderEventKind::Modified => "Modified",
        }
    }
}

/// One event of the stream.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Event {
    /// An order event.
    Order {
        stamp: EventStamp,
        order: OrderId,
        kind: OrderEventKind,
    },
    /// A position opened, changed or closed; the quantity after it, signed units.
    Position {
        stamp: EventStamp,
        instrument: InstrumentId,
        product: Product,
        change: PositionChange,
        quantity_units: f64,
    },
    /// A risk rule fired or a trading state changed; the instrument for a leg rule.
    Risk {
        stamp: EventStamp,
        instrument: Option<InstrumentId>,
        event: RiskEvent,
    },
    /// The strategy's trading state changed, and why.
    TradingState {
        stamp: EventStamp,
        from: TradingState,
        to: TradingState,
        cause: TradingStateCause,
    },
    /// A GTT was placed, modified, fired, resolved or cancelled.
    Gtt { stamp: EventStamp, event: GttEvent },
}

impl Event {
    /// When it happened.
    pub fn stamp(&self) -> EventStamp {
        match self {
            Event::Order { stamp, .. }
            | Event::Position { stamp, .. }
            | Event::Risk { stamp, .. }
            | Event::TradingState { stamp, .. }
            | Event::Gtt { stamp, .. } => *stamp,
        }
    }
}
