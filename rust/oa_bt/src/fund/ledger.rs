//! `AccountBook`: the account's ledger of a portfolio run (18-multi-strategy 2.4).
//!
//! It holds the account's capital `C` and, per engine in call order, its capital `c_i` and
//! its last figures (`EngineFigures`: free cash `f_i`, equity `q_i`, margin `m_i`, daily
//! loss measure `d_i` and the session day it belongs to), each rounded to the paisa as
//! every ledger posting is (D6). Sums are taken in call order and rounded to the paisa:
//!
//! - the account's free cash `F = C - sum(c_i) + sum(f_i)`;
//! - the account's equity `Q = C - sum(c_i) + sum(q_i)`;
//! - the account's margin `sum(m_i)`;
//! - engine `i`'s room: `F - f_i`, the account's free cash without its own, which its
//!   link to the account holds in a shared pool (`engine::AccountLink`).
//!
//! An engine that has ended keeps its last figures here (2.6).

use crate::engine::EngineFigures;
use crate::model::Day;
use crate::portfolio::round_to_paise;

/// The account's ledger (module docs).
#[derive(Clone, Debug, PartialEq)]
pub(super) struct AccountBook {
    /// `C - sum(c_i)`, rupees: what the account holds beyond the strategies' capitals
    /// (negative for an over-allocated account).
    unallocated_rupees: f64,
    /// Each engine's last figures, in paise, in call order.
    figures: Vec<EngineFigures>,
}

/// `figures` rounded to the paisa.
fn in_paise(figures: EngineFigures) -> EngineFigures {
    EngineFigures {
        free_cash_rupees: round_to_paise(figures.free_cash_rupees),
        equity_rupees: round_to_paise(figures.equity_rupees),
        margin_rupees: round_to_paise(figures.margin_rupees),
        daily_loss_rupees: round_to_paise(figures.daily_loss_rupees),
        session_day: figures.session_day,
    }
}

impl AccountBook {
    /// The ledger of an account of `capital_rupees` over engines of `capitals` (rupees, in
    /// call order) standing at `figures`.
    pub(super) fn new(
        capital_rupees: f64,
        capitals: &[f64],
        figures: Vec<EngineFigures>,
    ) -> AccountBook {
        let allocated: f64 = capitals.iter().map(|&c| round_to_paise(c)).sum();
        AccountBook {
            unallocated_rupees: round_to_paise(round_to_paise(capital_rupees) - allocated),
            figures: figures.into_iter().map(in_paise).collect(),
        }
    }

    /// Take engine `strategy`'s figures after its turn.
    pub(super) fn take(&mut self, strategy: usize, figures: EngineFigures) {
        if let Some(slot) = self.figures.get_mut(strategy) {
            *slot = in_paise(figures);
        }
    }

    /// The account's free cash `F`, rupees.
    pub(super) fn free_cash_rupees(&self) -> f64 {
        self.sum(|figures| figures.free_cash_rupees)
    }

    /// The account's equity `Q`, rupees.
    pub(super) fn equity_rupees(&self) -> f64 {
        self.sum(|figures| figures.equity_rupees)
    }

    /// The margin the strategies' open positions block, rupees.
    pub(super) fn margin_rupees(&self) -> f64 {
        let total: f64 = self
            .figures
            .iter()
            .map(|figures| figures.margin_rupees)
            .sum();
        round_to_paise(total)
    }

    /// Engine `strategy`'s room, `F - f_i`, rupees.
    pub(super) fn room_rupees(&self, strategy: usize) -> f64 {
        let own = self
            .figures
            .get(strategy)
            .map_or(0.0, |f| f.free_cash_rupees);
        round_to_paise(self.free_cash_rupees() - own)
    }

    /// Each engine's daily loss measure with the session day it belongs to, in call
    /// order.
    pub(super) fn measures(&self) -> impl Iterator<Item = (f64, Option<Day>)> + '_ {
        let measure = |figures: &EngineFigures| (figures.daily_loss_rupees, figures.session_day);
        self.figures.iter().map(measure)
    }

    /// `C - sum(c_i)` plus the sum of one figure of every engine, in call order, rounded
    /// to the paisa.
    fn sum(&self, figure: impl Fn(&EngineFigures) -> f64) -> f64 {
        let total: f64 = self.figures.iter().map(figure).sum();
        round_to_paise(self.unallocated_rupees + total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn figures(free_cash: f64, equity: f64, margin: f64) -> EngineFigures {
        EngineFigures {
            free_cash_rupees: free_cash,
            equity_rupees: equity,
            margin_rupees: margin,
            daily_loss_rupees: 0.0,
            session_day: None,
        }
    }

    /// Three strategies of Rs 2,00,000, Rs 2,00,000 and Rs 4,00,000 on an account of Rs
    /// 7,00,000 (over-allocated by Rs 1,00,000).
    fn over_allocated() -> AccountBook {
        let capitals = [200_000.0, 200_000.0, 400_000.0];
        let start = capitals.iter().map(|&c| figures(c, c, 0.0)).collect();
        AccountBook::new(700_000.0, &capitals, start)
    }

    #[test]
    fn an_account_at_the_start_holds_its_capital_as_free_cash_and_equity() {
        let book = over_allocated();
        assert_eq!(book.free_cash_rupees(), 700_000.0);
        assert_eq!(book.equity_rupees(), 700_000.0);
        assert_eq!(book.margin_rupees(), 0.0);
    }

    #[test]
    fn a_room_is_the_account_free_cash_without_the_engine_own() {
        let mut book = over_allocated();
        // Strategy 0 blocks 1,50,000 of margin: its free cash is 50,000.
        book.take(0, figures(50_000.0, 200_000.0, 150_000.0));
        // F = 7,00,000 - 8,00,000 + (50,000 + 2,00,000 + 4,00,000) = 5,50,000.
        assert_eq!(book.free_cash_rupees(), 550_000.0);
        assert_eq!(book.room_rupees(0), 500_000.0);
        assert_eq!(book.room_rupees(1), 350_000.0);
        assert_eq!(book.room_rupees(2), 150_000.0);
        assert_eq!(book.margin_rupees(), 150_000.0);
    }

    #[test]
    fn an_open_loss_lowers_the_account_equity_and_free_cash() {
        let mut book = over_allocated();
        // Strategy 2 holds 3,00,000 of margin and is 30,000 down: equity 3,70,000, free
        // cash 70,000.
        book.take(2, figures(70_000.0, 370_000.0, 300_000.0));
        // Q = -1,00,000 + 2,00,000 + 2,00,000 + 3,70,000 = 6,70,000.
        assert_eq!(book.equity_rupees(), 670_000.0);
        // F = -1,00,000 + 2,00,000 + 2,00,000 + 70,000 = 3,70,000.
        assert_eq!(book.free_cash_rupees(), 370_000.0);
    }

    #[test]
    fn figures_are_taken_in_paise() {
        let mut book = over_allocated();
        book.take(1, figures(199_999.996, 200_000.004, 0.004));
        assert_eq!(book.free_cash_rupees(), 700_000.0);
        assert_eq!(book.equity_rupees(), 700_000.0);
        assert_eq!(book.margin_rupees(), 0.0);
    }
}
