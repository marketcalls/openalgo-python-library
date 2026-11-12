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

use crate::model::Product;

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

    /// Free cash at the last marks, rupees: equity less the margin blocked and the sale
    /// proceeds not yet credited, with delivery (CNC) holdings counted as a cash account
    /// counts them. Delivery is paid in full, so a holding takes its cost from the cash
    /// (not the margin quoted at its fill's reference price), and its open profit or
    /// loss is not cash until it is sold.
    pub fn free_cash_rupees(&self) -> f64 {
        let delivery_beyond_margin: f64 = self
            .positions
            .iter()
            .filter(|position| position.key.product == Product::Cnc && !position.is_flat())
            .map(|position| {
                let cost = position.average_price_rupees
                    * position.quantity_units.abs()
                    * position.multiplier;
                cost + position.unrealized_rupees() - position.margin_rupees
            })
            .sum();
        self.account.free_cash_rupees(self.unrealized_rupees()) - delivery_beyond_margin
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::instrument::InstrumentId;
    use crate::model::Side;

    /// 10,00,000 of capital and 900 units bought at 1,000.00 in `product`, blocking
    /// `margin` (the delivery table quotes the value at the fill's reference price),
    /// then marked at `mark`.
    fn holding(product: Product, margin: f64, mark: f64) -> Portfolio {
        let key = PositionKey {
            instrument: InstrumentId(0),
            product,
        };
        let mut portfolio = Portfolio::new(1_000_000.0).unwrap_or_else(|e| panic!("{e}"));
        let position = portfolio.position_mut(key, 1.0);
        assert!(position.apply(Side::Buy, 900.0, 1_000.0, 0).is_ok());
        position.margin_rupees = margin;
        position.mark_rupees = Some(mark);
        portfolio.refresh_margin();
        portfolio
    }

    #[test]
    fn a_delivery_holding_takes_its_cost_and_its_open_profit_or_loss_is_not_cash() {
        // 9,00,000 paid for the holding leave 1,00,000 at a mark of 1,100.00 or 900.00.
        for mark in [1_100.0, 900.0] {
            let portfolio = holding(Product::Cnc, 900_000.0, mark);
            assert_eq!(portfolio.free_cash_rupees(), 100_000.0, "mark {mark}");
        }
    }

    #[test]
    fn a_delivery_holding_takes_the_price_paid_not_the_margin_quoted_before_slippage() {
        // Margin quoted at an open of 999.50 (8,99,550); the 9,00,000 paid are what left.
        let portfolio = holding(Product::Cnc, 899_550.0, 1_000.0);
        assert_eq!(portfolio.free_cash_rupees(), 100_000.0);
    }

    #[test]
    fn an_intraday_position_counts_its_open_profit_against_its_margin() {
        // MIS at 1x blocks 9,00,000; a mark of 1,100.00 adds 90,000 of open profit.
        let portfolio = holding(Product::Mis, 900_000.0, 1_100.0);
        assert_eq!(portfolio.free_cash_rupees(), 190_000.0);
    }
}
