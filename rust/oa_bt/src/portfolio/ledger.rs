//! The sub-account of one strategy: its allocated capital and what has been posted to
//! it (03-architecture 4.3: each strategy has its own fixed allocation; the combined
//! result is the sum of the strategies).
//!
//! Every posting is rounded to the paisa (decision D6), so the ledger never carries
//! fractions of a paisa however many fills it sees. Equity is
//! `capital + realized - costs + unrealized`; free cash is equity less the margin the
//! open positions block.

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

    /// Equity not blocked as margin, rupees.
    pub fn free_cash_rupees(&self, unrealized_rupees: f64) -> f64 {
        self.equity_rupees(unrealized_rupees) - self.margin_blocked_rupees
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
    fn capital_must_be_positive() {
        assert!(SubAccount::new(0.0).is_err());
        assert!(SubAccount::new(f64::INFINITY).is_err());
    }
}
