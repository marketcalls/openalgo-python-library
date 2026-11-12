//! Orders and execution: the instrument facts the kernel trades with, order requests
//! and the order state machine, GTT orders, fills, the price grid, bar matching,
//! pre-trade checks and the order book.
//!
//! | Module | Responsibility |
//! |---|---|
//! | `instrument` | `InstrumentSpec`: exchange, kind, lot size, tick, multiplier, freeze quantity, cost segment |
//! | `price` | The tick grid: levels to the nearest tick (SPEC-03), slipped fills away from the trader (SPEC-27) |
//! | `order` | `OrderRequest` (what a strategy asks for) and `Order` (a book entry), DAY validity, `OrderId` |
//! | `state` | `OrderStatus`, `OrderAction` and the table of legal transitions |
//! | `segment` | The per-segment order table: the stop order a segment accepts (04-execution 4.1) |
//! | `matching` | Where a LIMIT fills and a trigger is reached in one base bar |
//! | `gtt` | GTT orders (SINGLE, OCO): types, transition table, the decided policy, child pricing |
//! | `gtt_book` | `GttBook`: every GTT of a run |
//! | `gtt_event` | The events of a GTT's life |
//! | `fill` | `Fill`: quantity, price, where in the bar, why (`FillReason`), flags; `FillPricing` |
//! | `checks` | Pre-trade checks and `RejectCode` |
//! | `book` | `OrderBook`: every order by id, the queue waiting for an open |
//! | `error` | `ExecError` |
//!
//! Matching lives in the engine's slice steps, which call into this module: MARKET
//! orders fill at the open in step 2; LIMIT, SL and SL-M orders and GTT triggers are
//! tested in step 3 (`engine::resting`, `engine::gtt_fire`).

pub mod book;
pub mod checks;
pub mod error;
pub mod fill;
pub mod gtt;
pub mod gtt_book;
pub mod gtt_event;
pub mod instrument;
pub mod matching;
pub mod order;
pub mod price;
pub mod segment;
pub mod state;

pub use book::OrderBook;
pub use checks::{EntryWindow, RejectCode};
pub use error::ExecError;
pub use fill::{Fill, FillAt, FillFlags, FillPricing, FillReason};
pub use gtt::{
    Gtt, GttAction, GttId, GttKind, GttLeg, GttLegRole, GttPolicy, GttPurpose, GttSlot, GttStatus,
};
pub use gtt_book::GttBook;
pub use gtt_event::{GttEvent, GttEventName};
pub use instrument::{InstrumentId, InstrumentSpec};
pub use matching::{AfterTrigger, BarPoint, TriggerDirection};
pub use order::{Order, OrderId, OrderRequest, OrderType, QuantitySpec, Validity, GTT_GOOD_UNTIL};
pub use state::{OrderAction, OrderStatus, ORDER_TRANSITIONS};
