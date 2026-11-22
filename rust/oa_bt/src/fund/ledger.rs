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
//!   link to the account holds in a shared pool (`engine::AccountLink`);
//! - the other engines' entries of a session day, for the account's trade cap: the sum
//!   of each other engine's day's trades whose session day it is (18-multi-strategy
//!   2.5a rule 7).
//!
//! An engine that has ended keeps its last figures here (2.6).
//!
//! The fund hands the book an engine's figures after every turn, four times a round, and
//! links an engine from `F` before three of them, while most turns change nothing (an
//! engine's figures move at a fill or a mark). So the book keeps `F` with the figures and
//! takes a turn's figures only when they differ, bit for bit, from the engine's last: the
//! rounding and the sum are then the same as when they were last taken, and skipping
//! them changes no figure.

use crate::engine::EngineFigures;
use crate::model::Day;
use crate::portfolio::round_to_paise;

/// The account's ledger (module docs).
#[derive(Clone, Debug, PartialEq)]
pub(super) struct AccountBook {
    /// `C - sum(c_i)`, rupees: what the account holds beyond the strategies' capitals
    /// (negative for an over-allocated account).
    unallocated_rupees: f64,
    /// Each engine's figures as it last gave them, in call order (module docs).
    given: Vec<EngineFigures>,
    /// Each engine's last figures, in paise, in call order.
    figures: Vec<EngineFigures>,
    /// The account's free cash `F`, rupees, as `figures` give it.
    free_cash_rupees: f64,
}

/// True when `a` and `b` are the same figures bit for bit (so a 0.0 and a -0.0 differ,
/// and a NaN equals itself): what makes taking `b` after `a` change nothing.
fn same_bits(a: &EngineFigures, b: &EngineFigures) -> bool {
    let bits = |f: &EngineFigures| {
        [
            f.free_cash_rupees.to_bits(),
            f.equity_rupees.to_bits(),
            f.margin_rupees.to_bits(),
            f.daily_loss_rupees.to_bits(),
        ]
    };
    bits(a) == bits(b) && a.session_day == b.session_day && a.trades == b.trades
}

/// `figures` rounded to the paisa.
fn in_paise(figures: EngineFigures) -> EngineFigures {
    EngineFigures {
        free_cash_rupees: round_to_paise(figures.free_cash_rupees),
        equity_rupees: round_to_paise(figures.equity_rupees),
        margin_rupees: round_to_paise(figures.margin_rupees),
        daily_loss_rupees: round_to_paise(figures.daily_loss_rupees),
        session_day: figures.session_day,
        trades: figures.trades,
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
        let mut book = AccountBook {
            unallocated_rupees: round_to_paise(round_to_paise(capital_rupees) - allocated),
            figures: figures.iter().copied().map(in_paise).collect(),
            given: figures,
            free_cash_rupees: 0.0,
        };
        book.free_cash_rupees = book.sum(|figures| figures.free_cash_rupees);
        book
    }

    /// Take engine `strategy`'s figures after its turn (module docs).
    pub(super) fn take(&mut self, strategy: usize, figures: EngineFigures) {
        let Some(given) = self.given.get_mut(strategy) else {
            return;
        };
        if same_bits(given, &figures) {
            return;
        }
        *given = figures;
        self.figures[strategy] = in_paise(figures);
        self.free_cash_rupees = self.sum(|figures| figures.free_cash_rupees);
    }

    /// The account's free cash `F`, rupees.
    pub(super) fn free_cash_rupees(&self) -> f64 {
        self.free_cash_rupees
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
        round_to_paise(self.free_cash_rupees - own)
    }

    /// The entries every engine but `strategy` has counted on session day `day` (module
    /// docs).
    pub(super) fn trades_of_others(&self, strategy: usize, day: Day) -> u32 {
        let counted = |(index, figures): (usize, &EngineFigures)| {
            (index != strategy && figures.session_day == Some(day)).then_some(figures.trades)
        };
        let total = self.figures.iter().enumerate().filter_map(counted);
        total.fold(0, u32::saturating_add)
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
            trades: 0,
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
    fn the_account_free_cash_follows_an_engine_figures_away_and_back() {
        let mut book = over_allocated();
        // Strategy 1 blocks 1,20,000 (free cash 80,000), takes the same figures again,
        // then closes at no profit: F goes 7,00,000 to 5,80,000 and back.
        let blocked = figures(80_000.0, 200_000.0, 120_000.0);
        book.take(1, blocked);
        book.take(1, blocked);
        assert_eq!(book.free_cash_rupees(), 580_000.0);
        assert_eq!(book.room_rupees(1), 500_000.0);
        book.take(1, figures(200_000.0, 200_000.0, 0.0));
        assert_eq!(book.free_cash_rupees(), 700_000.0);
        assert_eq!(book.room_rupees(1), 500_000.0);
        assert_eq!(book.room_rupees(0), 500_000.0);
    }

    #[test]
    fn the_others_trades_count_only_on_the_day_asked_for() {
        let mut book = over_allocated();
        let day = 20_270;
        let on = |day, trades| EngineFigures {
            session_day: Some(day),
            trades,
            ..figures(200_000.0, 200_000.0, 0.0)
        };
        book.take(0, on(day, 2));
        // Strategy 1's 4 trades are of the day before: they do not count today.
        book.take(1, on(day - 1, 4));
        book.take(2, on(day, 1));
        assert_eq!(book.trades_of_others(0, day), 1);
        assert_eq!(book.trades_of_others(1, day), 3);
        assert_eq!(book.trades_of_others(2, day), 2);
        assert_eq!(book.trades_of_others(2, day - 1), 4);
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
