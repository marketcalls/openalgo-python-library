//! The daily loss limit: what it measures and when it is breached (05-risk-management
//! 5, 14-strategy-api 3, decisions SPEC-16 and SPEC-18).
//!
//! A run is the strategy's positions from a fill that leaves it no longer flat until
//! every position is flat again. The measure is the realized P&L of the runs that ended
//! in this session plus the live run's MTM since its first fill (realized plus
//! unrealized). A futures roll does not end a run, so its fills never start or end
//! one. On the net basis every charge posted to those runs is subtracted, roll legs
//! included.
//!
//! This is OpenAlgo's rule, adapted (lead ruling): the live module counts the runs that
//! started since the session boundary, the engine counts the runs that ended in this
//! session. A run that started in an earlier session and ends in this one therefore
//! counts whole in this one, including what it realized earlier, and a run still open
//! counts from its first fill, not from the previous close.
//!
//! Runs follow one another and nothing is posted while the strategy is flat, so the
//! runs counted in a session are one unbroken stretch of the account's history. The
//! measure is the change in the account's realized P&L (and charges) since the first
//! fill of the first counted run, plus the open MTM. The tracker keeps account totals
//! at those moments rather than a list of runs, so a posting that is not a fill (the
//! daily futures settlement) is counted without being seen, and does not restart the
//! run's MTM.

use crate::portfolio::round_to_paise;

use super::rules::RiskBasis;

/// What the account has posted since the start of the run, rupees.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AccountTotals {
    /// Gross realized P&L.
    pub realized_rupees: f64,
    /// Charges.
    pub costs_rupees: f64,
}

/// The daily loss measure of one strategy (see the module documentation).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DailyLossMeasure {
    /// Account totals when the strategy last became flat (zero before the first run).
    flat_totals: AccountTotals,
    /// Account totals at the first fill of the live run, while a run is live.
    live_run_from: Option<AccountTotals>,
    /// Account totals at the first fill of the first run counted in this session: the
    /// first run that ended in it, or the live run. `None` while nothing is counted.
    counted_from: Option<AccountTotals>,
}

impl DailyLossMeasure {
    /// A new session began: runs that ended before it stop counting, and the live run
    /// counts from its first fill.
    pub fn start_session(&mut self) {
        self.counted_from = self.live_run_from;
    }

    /// Account for a fill. `totals` are the account's after it, `flat` says whether
    /// every position of the strategy is flat after it, and `roll` marks the fills of a
    /// futures roll, which neither start nor end a run.
    pub fn after_fill(&mut self, totals: AccountTotals, flat: bool, roll: bool) {
        if roll {
            return;
        }
        match (self.live_run_from, flat) {
            (None, false) => {
                self.live_run_from = Some(self.flat_totals);
                self.counted_from.get_or_insert(self.flat_totals);
            }
            (Some(_), true) => {
                self.live_run_from = None;
                self.flat_totals = totals;
            }
            _ => {}
        }
    }

    /// True while a run is live.
    pub fn run_is_live(&self) -> bool {
        self.live_run_from.is_some()
    }

    /// The measure in rupees, rounded to the paisa: realized P&L of the counted runs
    /// plus `unrealized_rupees` (the open MTM at the closes), less their charges on the
    /// net basis. Negative is a loss.
    pub fn measure_rupees(
        &self,
        basis: RiskBasis,
        totals: AccountTotals,
        unrealized_rupees: f64,
    ) -> f64 {
        let from = self.counted_from.unwrap_or(totals);
        let gross = totals.realized_rupees - from.realized_rupees + unrealized_rupees;
        let charges = match basis {
            RiskBasis::Gross => 0.0,
            RiskBasis::Net => totals.costs_rupees - from.costs_rupees,
        };
        round_to_paise(gross - charges)
    }
}

/// True when a measure has reached the limit: a loss of `limit_rupees` or more.
/// Both are compared in whole paise, so floating-point noise cannot decide a breach
/// that falls exactly on the limit.
pub fn is_breached(measure_rupees: f64, limit_rupees: f64) -> bool {
    round_to_paise(measure_rupees) <= -round_to_paise(limit_rupees)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn totals(realized_rupees: f64, costs_rupees: f64) -> AccountTotals {
        AccountTotals {
            realized_rupees,
            costs_rupees,
        }
    }

    fn gross(measure: &DailyLossMeasure, realized: f64, unrealized: f64) -> f64 {
        measure.measure_rupees(RiskBasis::Gross, totals(realized, 0.0), unrealized)
    }

    #[test]
    fn a_live_run_counts_its_open_mtm() {
        // Trace 8b, 2025-07-28: long 75 at 25010.00, close 24960.00.
        let mut measure = DailyLossMeasure::default();
        measure.start_session();
        measure.after_fill(totals(0.0, 0.0), false, false);
        assert_eq!(gross(&measure, 0.0, -3_750.0), -3_750.0);
    }

    #[test]
    fn a_roll_does_not_end_the_run_so_its_realized_loss_keeps_counting() {
        // Trace 8b: entry on 2025-07-28; on 2025-07-30 the roll realizes -6750.00 on the
        // near contract and the next contract is marked -9375.00 at the close.
        let mut measure = DailyLossMeasure::default();
        measure.start_session(); // 2025-07-28
        measure.after_fill(totals(0.0, 0.0), false, false);
        measure.start_session(); // 2025-07-29
        measure.start_session(); // 2025-07-30: near leg closed, next leg opened
        measure.after_fill(totals(-6_750.0, 0.0), true, true);
        measure.after_fill(totals(-6_750.0, 0.0), false, true);
        assert_eq!(gross(&measure, -6_750.0, -9_375.0), -16_125.0);
        assert!(measure.run_is_live());
    }

    #[test]
    fn a_run_that_ended_in_an_earlier_session_no_longer_counts() {
        let mut measure = DailyLossMeasure::default();
        measure.start_session();
        measure.after_fill(totals(0.0, 0.0), false, false);
        measure.after_fill(totals(-4_000.0, 0.0), true, false);
        assert_eq!(gross(&measure, -4_000.0, 0.0), -4_000.0);
        measure.start_session();
        assert_eq!(gross(&measure, -4_000.0, 0.0), 0.0);
    }

    #[test]
    fn a_run_that_started_earlier_and_ended_today_counts_whole_today() {
        // Day 1: entry, a partial exit realizes -1000.00. Day 2: the rest exits for a
        // further -2500.00. The run ended on day 2, so day 2 counts -3500.00 (the live
        // module, counting runs started since the boundary, would count nothing).
        let mut measure = DailyLossMeasure::default();
        measure.start_session();
        measure.after_fill(totals(0.0, 0.0), false, false);
        measure.after_fill(totals(-1_000.0, 0.0), false, false);
        measure.start_session();
        measure.after_fill(totals(-3_500.0, 0.0), true, false);
        assert_eq!(gross(&measure, -3_500.0, 0.0), -3_500.0);
    }

    #[test]
    fn a_live_run_counts_from_its_first_fill_not_from_the_previous_close() {
        // Bought on day 1 at 100.00 for 100 units; day 1 closes at 95.00, day 2 at 94.00.
        // Day 2 counts (94.00 - 100.00) x 100 = -600.00, not the day's own -100.00.
        let mut measure = DailyLossMeasure::default();
        measure.start_session();
        measure.after_fill(totals(0.0, 0.0), false, false);
        measure.start_session();
        assert_eq!(gross(&measure, 0.0, -600.0), -600.0);
    }

    #[test]
    fn runs_ended_and_live_in_one_session_add_up() {
        // Run 1 realizes +2000.00 and ends; run 2 opens and is marked -5000.00.
        let mut measure = DailyLossMeasure::default();
        measure.start_session();
        measure.after_fill(totals(0.0, 0.0), false, false);
        measure.after_fill(totals(2_000.0, 0.0), true, false);
        measure.after_fill(totals(2_000.0, 0.0), false, false);
        assert_eq!(gross(&measure, 2_000.0, -5_000.0), -3_000.0);
    }

    #[test]
    fn the_net_basis_subtracts_every_charge_of_the_counted_runs() {
        // Earlier sessions paid 40.00 of charges; today's run paid 25.00 and realized
        // -1000.00: gross -1000.00, net -1025.00.
        let mut measure = DailyLossMeasure::default();
        measure.start_session();
        measure.after_fill(totals(0.0, 20.0), false, false);
        measure.after_fill(totals(500.0, 40.0), true, false);
        measure.start_session();
        measure.after_fill(totals(500.0, 52.5), false, false);
        measure.after_fill(totals(-500.0, 65.0), true, false);
        let now = totals(-500.0, 65.0);
        assert_eq!(measure.measure_rupees(RiskBasis::Gross, now, 0.0), -1_000.0);
        assert_eq!(measure.measure_rupees(RiskBasis::Net, now, 0.0), -1_025.0);
    }

    #[test]
    fn a_strategy_with_no_run_this_session_measures_zero() {
        let measure = DailyLossMeasure::default();
        assert_eq!(gross(&measure, 0.0, 0.0), 0.0);
    }

    #[test]
    fn a_loss_exactly_at_the_limit_is_a_breach() {
        assert!(is_breached(-10_000.0, 10_000.0));
        assert!(is_breached(-16_125.0, 10_000.0));
        assert!(!is_breached(-9_999.99, 10_000.0));
        // (100.10 - 100.00) x 3 is 0.30 rupees, but just under it in binary floating point;
        // it still breaches a 0.30 limit.
        assert!(is_breached(-(100.10_f64 - 100.00) * 3.0, 0.30));
    }
}
