//! What the driver may ask the engine at any time: the decision instant, the instruments,
//! positions, legs, orders and the account.

use crate::exec::{InstrumentId, InstrumentSpec, Order, OrderId};
use crate::model::{Product, TsNanos};
use crate::portfolio::{Position, PositionKey};
use crate::rms::LegState;

use super::kernel::{Engine, Phase};

/// The account as a strategy sees it, rupees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AccountView {
    /// Capital plus realized minus costs plus unrealized.
    pub equity_rupees: f64,
    /// What the account can trade with (`Portfolio::free_cash_rupees`): equity not
    /// blocked as margin, with delivery holdings of shares at their cost and sale
    /// proceeds not yet credited left out.
    pub free_cash_rupees: f64,
    /// Gross P&L realized since the start.
    pub realized_rupees: f64,
    /// Open P&L at the last marks.
    pub unrealized_rupees: f64,
    /// Charges paid since the start.
    pub costs_rupees: f64,
    /// Margin blocked by open positions.
    pub margin_rupees: f64,
}

impl Engine {
    /// The decision instant of the current pause (UTC nanoseconds), `None` when the
    /// engine is not paused.
    pub fn decision_instant(&self) -> Option<TsNanos> {
        self.decision_point().map(|(_, decided_at, _)| decided_at)
    }

    /// The instruments, in registration order.
    pub fn instruments(&self) -> &[InstrumentSpec] {
        &self.instruments
    }

    /// True once `advance` has returned `Pause::Finished`.
    pub fn is_finished(&self) -> bool {
        self.phase == Phase::Finished
    }

    /// The position of an instrument and product, if it was ever opened.
    pub fn position(&self, instrument: InstrumentId, product: Product) -> Option<&Position> {
        self.portfolio.position(PositionKey {
            instrument,
            product,
        })
    }

    /// The leg state (levels in force) of an open position.
    pub fn leg(&self, instrument: InstrumentId, product: Product) -> Option<&LegState> {
        let key = PositionKey {
            instrument,
            product,
        };
        self.legs
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, leg)| leg)
    }

    /// An order by id.
    pub fn order(&self, id: OrderId) -> Option<&Order> {
        self.book.get(id)
    }

    /// The orders that can still fill, in id order.
    pub fn working_orders(&self) -> Vec<OrderId> {
        self.book.working().map(|order| order.id).collect()
    }

    /// True when an order that would open or add is still working on the instrument
    /// (a queued entry counts as not flat for a condition strategy, SPEC-29).
    pub fn has_pending_entry(&self, instrument: InstrumentId) -> bool {
        self.book
            .working()
            .any(|order| order.request.instrument == instrument && !order.request.reduce_only)
    }

    /// The last close seen of an instrument, rupees.
    pub fn last_close(&self, instrument: InstrumentId) -> Option<f64> {
        self.last_close.get(instrument.index()).copied().flatten()
    }

    /// The account at the last marks.
    pub fn account(&self) -> AccountView {
        let account = &self.portfolio.account;
        AccountView {
            equity_rupees: self.portfolio.equity_rupees(),
            free_cash_rupees: self.free_cash_rupees(),
            realized_rupees: account.realized_rupees,
            unrealized_rupees: self.portfolio.unrealized_rupees(),
            costs_rupees: account.costs_rupees,
            margin_rupees: account.margin_blocked_rupees,
        }
    }
}
