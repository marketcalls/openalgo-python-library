//! The drawdown halt (05-risk-management 4, lead ruling): when the strategy's equity
//! falls the given percent below its peak, every position exits (reason
//! `drawdown_halt`), the event `risk_halt` is written and the strategy is HALTED for
//! the rest of the run. The latch itself is the trading state (`TradingLock`); this
//! module only watches the equity.
//!
//! Equity is the strategy's capital plus realized P&L plus the open MTM at the closes,
//! less charges on the `net` risk basis (one basis for every strategy and session rule,
//! 14-strategy-api 3). The peak starts at the capital and only rises.

use crate::portfolio::round_to_paise;

/// A breach of the drawdown halt, in rupees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrawdownBreach {
    /// The equity at which the halt fires: the peak less the limit.
    pub floor_rupees: f64,
    /// The equity observed.
    pub equity_rupees: f64,
}

/// The peak equity and the halt's limit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrawdownWatch {
    /// The limit, percent of the peak (above 0 and below 100).
    limit_percent: f64,
    /// The highest equity observed, rupees.
    peak_rupees: f64,
}

impl DrawdownWatch {
    /// A watch with the peak at the starting equity (the strategy's capital).
    pub fn new(limit_percent: f64, starting_equity_rupees: f64) -> DrawdownWatch {
        DrawdownWatch {
            limit_percent,
            peak_rupees: starting_equity_rupees,
        }
    }

    /// The highest equity observed so far, rupees.
    pub fn peak_rupees(&self) -> f64 {
        self.peak_rupees
    }

    /// Observe the equity at a close. The peak rises to it; a breach is returned when
    /// it is at or below the floor, compared in whole paise.
    pub fn observe(&mut self, equity_rupees: f64) -> Option<DrawdownBreach> {
        self.peak_rupees = self.peak_rupees.max(equity_rupees);
        let floor = round_to_paise(self.peak_rupees * (1.0 - self.limit_percent / 100.0));
        (round_to_paise(equity_rupees) <= floor).then_some(DrawdownBreach {
            floor_rupees: floor,
            equity_rupees,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equity_exactly_at_the_floor_breaches() {
        // Capital 1000000.00, limit 5 percent: the floor is 950000.00.
        let mut watch = DrawdownWatch::new(5.0, 1_000_000.0);
        assert_eq!(watch.observe(950_000.01), None);
        let breach = watch.observe(950_000.0);
        assert_eq!(
            breach,
            Some(DrawdownBreach {
                floor_rupees: 950_000.0,
                equity_rupees: 950_000.0,
            })
        );
    }

    #[test]
    fn the_floor_follows_the_peak_up() {
        // The peak rises to 1200000.00, so 5 percent is 60000.00: 1140000.00 breaches,
        // although it is above the starting capital.
        let mut watch = DrawdownWatch::new(5.0, 1_000_000.0);
        assert_eq!(watch.observe(1_200_000.0), None);
        assert_eq!(watch.observe(1_140_000.01), None);
        let breach = watch.observe(1_140_000.0);
        assert_eq!(breach.map(|b| b.floor_rupees), Some(1_140_000.0));
        assert_eq!(watch.peak_rupees(), 1_200_000.0);
    }

    #[test]
    fn a_fall_does_not_lower_the_peak() {
        let mut watch = DrawdownWatch::new(10.0, 1_000_000.0);
        watch.observe(980_000.0);
        assert_eq!(watch.peak_rupees(), 1_000_000.0);
        assert_eq!(
            watch.observe(900_000.0).map(|b| b.floor_rupees),
            Some(900_000.0)
        );
    }
}
