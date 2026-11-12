//! Positions and accounts: net positions built from tranches, and the sub-account of
//! the strategy.
//!
//! | Module | Responsibility |
//! |---|---|
//! | `position` | `Position`, `Tranche`, `PositionKey`: fills applied at average cost |
//! | `ledger` | `SubAccount`: capital, realized P&L, costs, margin, paisa postings |
//!
//! `Portfolio` holds both for one strategy. Positions are kept in the order they were
//! first opened, which is deterministic because fills are.

pub mod ledger;
pub mod position;

pub use ledger::{round_to_paise, SubAccount};
pub use position::{
    FillEffect, FillKind, PortfolioError, Position, PositionChange, PositionKey, Tranche,
};

/// The positions and the account of one strategy.
#[derive(Clone, Debug, PartialEq)]
pub struct Portfolio {
    /// The strategy's account.
    pub account: SubAccount,
    positions: Vec<Position>,
}

impl Portfolio {
    /// A portfolio with no position and `capital_rupees` allocated.
    ///
    /// Errors: `InvalidCapital` unless the capital is a positive finite amount.
    pub fn new(capital_rupees: f64) -> Result<Portfolio, PortfolioError> {
        Ok(Portfolio {
            account: SubAccount::new(capital_rupees)?,
            positions: Vec::new(),
        })
    }

    /// The position for a key, if one was ever opened.
    pub fn position(&self, key: PositionKey) -> Option<&Position> {
        self.positions.iter().find(|position| position.key == key)
    }

    /// The position for a key, created flat when there is none.
    pub fn position_mut(&mut self, key: PositionKey, multiplier: f64) -> &mut Position {
        let index = match self.positions.iter().position(|p| p.key == key) {
            Some(index) => index,
            None => {
                self.positions.push(Position::flat(key, multiplier));
                self.positions.len() - 1
            }
        };
        &mut self.positions[index]
    }

    /// Every position ever opened, in the order they were first opened.
    pub fn positions(&self) -> &[Position] {
        &self.positions
    }

    /// Every position, for marking.
    pub fn positions_mut(&mut self) -> &mut [Position] {
        &mut self.positions
    }

    /// Unrealized P&L of every open position at its last mark, rupees.
    pub fn unrealized_rupees(&self) -> f64 {
        self.positions.iter().map(Position::unrealized_rupees).sum()
    }

    /// Recompute the margin blocked from the positions.
    pub fn refresh_margin(&mut self) {
        let blocked = self.positions.iter().map(|p| p.margin_rupees).sum();
        self.account.set_margin_blocked(blocked);
    }

    /// Equity at the last marks, rupees.
    pub fn equity_rupees(&self) -> f64 {
        self.account.equity_rupees(self.unrealized_rupees())
    }

    /// Free cash at the last marks, rupees.
    pub fn free_cash_rupees(&self) -> f64 {
        self.account.free_cash_rupees(self.unrealized_rupees())
    }
}
