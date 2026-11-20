//! What a portfolio run's fund (`crate::fund`, 18-multi-strategy 2.3 and 2.4) reads of one
//! engine between its turns: where the engine stands, its grid, the figures the account's
//! ledger takes, the events it has recorded and the orders an account breach cancelled.
//!
//! The fund runs the phases of each slice itself (`open_phase`, `resume_open_phase`,
//! `levels_phase`, `decision_phase`, `end_slice`), so these are crate-internal: a run of
//! one engine never reads them.

use crate::exec::OrderId;
use crate::model::Day;

use super::clock::Timeline;
use super::error::KernelError;
use super::event::{Event, EventStamp};
use super::kernel::{Engine, Phase};
use super::pause::Pause;

/// One engine's figures as the account's ledger takes them after each of its turns,
/// rupees as the engine holds them (the ledger rounds them to the paisa).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct EngineFigures {
    /// What the strategy can trade with (`Engine::account`).
    pub(crate) free_cash_rupees: f64,
    /// The strategy's equity at its last marks.
    pub(crate) equity_rupees: f64,
    /// The margin its open positions block.
    pub(crate) margin_rupees: f64,
    /// Its daily loss measure on the run's risk basis, negative for a loss
    /// (`Engine::daily_loss_measure`).
    pub(crate) daily_loss_rupees: f64,
    /// The IST session day the measure belongs to: the day of the slice the engine last
    /// opened; `None` before its first slice.
    pub(crate) session_day: Option<Day>,
}

impl Engine {
    /// Where the state machine stands.
    pub(crate) fn phase(&self) -> Phase {
        self.phase
    }

    /// The run's grid.
    pub(crate) fn timeline(&self) -> &Timeline {
        &self.timeline
    }

    /// The strategy's capital, rupees.
    pub(crate) fn capital_rupees(&self) -> f64 {
        self.config.capital_rupees
    }

    /// True for a live session, whose account is the broker's.
    pub(crate) fn is_live(&self) -> bool {
        self.is_external()
    }

    /// The figures the account's ledger takes (`EngineFigures`), as `account` and
    /// `daily_loss_measure` give them, the open P&L summed once: the fund takes them after
    /// every turn of every engine.
    pub(crate) fn figures(&self) -> EngineFigures {
        let unrealized = self.portfolio.unrealized_rupees();
        let account = &self.portfolio.account;
        let basis = self.config.risk.risk_basis;
        EngineFigures {
            free_cash_rupees: self.free_cash_rupees(),
            equity_rupees: account.equity_rupees(unrealized),
            margin_rupees: account.margin_blocked_rupees,
            daily_loss_rupees: self.risk.daily_loss.measure_rupees(
                basis,
                self.account_totals(),
                unrealized,
            ),
            session_day: self.risk.session_day,
        }
    }

    /// Every event recorded so far, in order.
    pub(crate) fn recorded_events(&self) -> &[Event] {
        &self.record.events
    }

    /// The working orders an account breach has cancelled since the last call, each with
    /// the stamp of its cancellation, in the order they were cancelled (`account_breach`).
    pub(crate) fn take_account_cancels(&mut self) -> Vec<(EventStamp, OrderId)> {
        std::mem::take(&mut self.account.cancels)
    }

    /// From before the slice past the end of the grid, finish the run: `Pause::Finished`
    /// with the events not yet delivered, as `advance` returns it.
    ///
    /// Errors: `RunFinished` once finished; `InvalidConfig` (field `fund`) when the engine
    /// stands anywhere else.
    pub(crate) fn finish_grid(&mut self) -> Result<Pause, KernelError> {
        match self.phase {
            Phase::Ready(index) if index >= self.timeline.len() => self.advance(),
            Phase::Finished => Err(KernelError::RunFinished),
            _ => Err(KernelError::InvalidConfig {
                field: "fund",
                reason: "the run still has slices to run".to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::engine::account_test_kit::{july_2025, market, minute_bars, pause_at, Setup};
    use crate::model::{Product, Side};

    #[test]
    fn the_figures_are_the_account_and_daily_loss_measure_of_the_engine() {
        // SBIN bought 100 MIS at 1,000.00 in minute 1, marked at 990.00 from minute 2:
        // equity 9,99,000, margin 1,00,000, free cash 8,99,000, a daily loss of -1,000.
        let day = july_2025(1);
        let bars = minute_bars(&[day], |_, k| if k < 2 { [1_000.0; 4] } else { [990.0; 4] });
        let mut engine = Setup::minutes(day, day).engine(bars);
        pause_at(&mut engine, 0);
        assert!(engine
            .submit(market(Side::Buy, 100.0, Product::Mis))
            .is_ok());
        pause_at(&mut engine, 3);
        let figures = engine.figures();
        let account = engine.account();
        assert_eq!(
            (
                figures.equity_rupees,
                figures.margin_rupees,
                figures.free_cash_rupees
            ),
            (999_000.0, 100_000.0, 899_000.0)
        );
        assert_eq!(
            (
                figures.equity_rupees,
                figures.margin_rupees,
                figures.free_cash_rupees
            ),
            (
                account.equity_rupees,
                account.margin_rupees,
                account.free_cash_rupees
            )
        );
        assert_eq!(figures.daily_loss_rupees, -1_000.0);
        assert_eq!(figures.daily_loss_rupees, engine.daily_loss_measure());
        assert_eq!(figures.session_day, Some(day));
    }
}
