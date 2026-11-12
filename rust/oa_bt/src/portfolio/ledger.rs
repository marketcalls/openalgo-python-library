//! The sub-account of one strategy: its allocated capital and what has been posted to
//! it (03-architecture 4.3: each strategy has its own fixed allocation; the combined
//! result is the sum of the strategies).
//!
//! Every posting is rounded to the paisa (decision D6), so the ledger never carries
//! fractions of a paisa however many fills it sees. Equity is
//! `capital + realized - costs + unrealized`; free cash is equity less the margin the
//! open positions block and less the proceeds of today's sales of delivery holdings that
//! the broker credits only on the next trading day (`market::sale_credit`).

use super::position::PortfolioError;

/// Paise in one rupee.
const PAISE_PER_RUPEE: f64 = 100.0;

/// An amount in rupees rounded to the nearest paisa.
pub fn round_to_paise(rupees: f64) -> f64 {
    (rupees * PAISE_PER_RUPEE).round() / PAISE_PER_RUPEE
}

/// A strategy's account.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubAccount {
    /// Capital allocated at the start, rupees.
    pub capital_rupees: f64,
    /// Gross P&L realized so far, rupees, in paise.
    pub realized_rupees: f64,
    /// Charges paid so far, rupees, in paise.
    pub costs_rupees: f64,
    /// Margin the open positions block, rupees, in paise.
    pub margin_blocked_rupees: f64,
    /// Proceeds of the current session's sales of delivery holdings not yet credited for
    /// trading (credited at the session close), rupees, in paise.
    pub uncredited_sale_rupees: f64,
}

impl SubAccount {
    /// An account holding `capital_rupees`.
    ///
    /// Errors: `InvalidCapital` unless the capital is a positive finite amount.
    pub fn new(capital_rupees: f64) -> Result<SubAccount, PortfolioError> {
        if !capital_rupees.is_finite() || capital_rupees <= 0.0 {
            return Err(PortfolioError::InvalidCapital {
                capital: capital_rupees,
            });
        }
        Ok(SubAccount {
            capital_rupees,
            realized_rupees: 0.0,
            costs_rupees: 0.0,
            margin_blocked_rupees: 0.0,
            uncredited_sale_rupees: 0.0,
        })
    }

    /// Post realized P&L of a fill.
    pub fn post_realized(&mut self, rupees: f64) {
        self.realized_rupees = round_to_paise(self.realized_rupees + round_to_paise(rupees));
    }

    /// Post the charges of a fill.
    pub fn post_costs(&mut self, rupees: f64) {
        self.costs_rupees = round_to_paise(self.costs_rupees + round_to_paise(rupees));
    }

    /// Set the margin the open positions block.
    pub fn set_margin_blocked(&mut self, rupees: f64) {
        self.margin_blocked_rupees = round_to_paise(rupees);
    }

    /// Equity with the given unrealized P&L, rupees.
    pub fn equity_rupees(&self, unrealized_rupees: f64) -> f64 {
        self.capital_rupees + self.realized_rupees - self.costs_rupees + unrealized_rupees
    }

    /// Withhold part of a sale of delivery holdings until the next trading day.
    pub fn withhold_sale_proceeds(&mut self, rupees: f64) {
        self.uncredited_sale_rupees =
            round_to_paise(self.uncredited_sale_rupees + round_to_paise(rupees));
    }

    /// Credit every withheld sale proceed (at the session close).
    pub fn credit_sale_proceeds(&mut self) {
        self.uncredited_sale_rupees = 0.0;
    }

    /// Equity with the given unrealized P&L, less the margin blocked and the sale
    /// proceeds not yet credited, rupees.
    pub fn free_cash_rupees(&self, unrealized_rupees: f64) -> f64 {
        self.equity_rupees(unrealized_rupees)
            - self.margin_blocked_rupees
            - self.uncredited_sale_rupees
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn postings_are_rounded_to_the_paisa() {
        let account = SubAccount::new(100_000.0);
        assert!(account.is_ok());
        if let Ok(mut account) = account {
            account.post_realized(10.004);
            account.post_costs(1.236);
            assert_eq!(account.realized_rupees, 10.0);
            assert_eq!(account.costs_rupees, 1.24);
            assert!((account.equity_rupees(5.0) - 100_013.76).abs() < 1e-6);
        }
    }

    #[test]
    fn withheld_sale_proceeds_are_not_free_until_credited() {
        // 10,00,000 of capital with 1,80,000 of a sale withheld: 8,20,000 free, then all.
        let account = SubAccount::new(1_000_000.0);
        assert!(account.is_ok());
        if let Ok(mut account) = account {
            account.withhold_sale_proceeds(180_000.0);
            assert_eq!(account.free_cash_rupees(0.0), 820_000.0);
            account.credit_sale_proceeds();
            assert_eq!(account.free_cash_rupees(0.0), 1_000_000.0);
        }
    }

    #[test]
    fn capital_must_be_positive() {
        assert!(SubAccount::new(0.0).is_err());
        assert!(SubAccount::new(f64::INFINITY).is_err());
    }
}
