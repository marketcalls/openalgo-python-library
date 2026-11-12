//! The facts about an instrument that the kernel trades with.
//!
//! The Python package resolves these from the reference tables (or the user's
//! `instrument=`) for the run and passes them in; the kernel never looks anything up.
//! Quantity is in units, never lots: one lot is `lot_size_units` units, and a
//! derivative order is a whole number of lots. P&L in rupees is
//! `(exit - entry) * quantity_units * multiplier`.

use crate::market::costs::Segment;
use crate::model::{Day, Exchange, InstrumentKind};

use super::error::ExecError;
use super::price::tick_units;

/// Position of an instrument in registration order. Every step of a slice processes
/// instruments in this order (03-architecture 4.2), so results never depend on
/// anything else.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InstrumentId(pub u32);

impl InstrumentId {
    /// The index into per-instrument vectors.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// One instrument as it stands for the run.
#[derive(Clone, Debug, PartialEq)]
pub struct InstrumentSpec {
    /// OpenAlgo symbol, for messages and results.
    pub symbol: String,
    /// OpenAlgo exchange; decides the session calendar and the intraday times.
    pub exchange: Exchange,
    /// Equity, index (not tradable), future or option.
    pub kind: InstrumentKind,
    /// Units in one lot (1 for cash equity).
    pub lot_size_units: u32,
    /// Tick size in rupees. One tick for the whole run: tick slabs by price are
    /// resolved by the Python package before the run.
    pub tick_size_rupees: f64,
    /// Rupee value of a one-rupee price move for one unit (1 for equity and NSE
    /// derivatives; MCX GOLD is 100).
    pub multiplier: f64,
    /// Largest quantity of one order in units, when the exchange sets one.
    pub freeze_quantity_units: Option<u32>,
    /// Cost segment for this instrument and the run's product, or `None` when the run
    /// is cost-free (`costs=None`).
    pub cost_segment: Option<Segment>,
    /// IST date the contract stops trading (derivatives), when it has one.
    pub last_trading_day: Option<Day>,
}

impl InstrumentSpec {
    /// Check the definition.
    ///
    /// Errors: `InvalidInstrument` for an empty symbol, a zero lot size or a
    /// multiplier that is not a positive finite number; `InvalidTick` for a tick that
    /// is not a positive whole number of price units.
    pub fn validate(&self) -> Result<(), ExecError> {
        let invalid = |reason: &str| ExecError::InvalidInstrument {
            symbol: self.symbol.clone(),
            reason: reason.to_string(),
        };
        if self.symbol.trim().is_empty() {
            return Err(invalid("the symbol is empty"));
        }
        if self.lot_size_units == 0 {
            return Err(invalid("the lot size must be at least 1 unit"));
        }
        if !self.multiplier.is_finite() || self.multiplier <= 0.0 {
            return Err(invalid("the multiplier must be a positive number"));
        }
        tick_units(self.tick_size_rupees)?;
        Ok(())
    }

    /// True for an instrument that can be traded (an index cannot).
    pub fn is_tradable(&self) -> bool {
        self.kind != InstrumentKind::Index
    }

    /// True for a future or an option, which always trade in whole lots.
    pub fn is_derivative(&self) -> bool {
        matches!(self.kind, InstrumentKind::Future | InstrumentKind::Option)
    }

    /// Rupee value of a position of `quantity_units` at `price_rupees`.
    pub fn notional_rupees(&self, price_rupees: f64, quantity_units: f64) -> f64 {
        price_rupees * quantity_units * self.multiplier
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn equity() -> InstrumentSpec {
        InstrumentSpec {
            symbol: "SBIN".to_string(),
            exchange: Exchange::Nse,
            kind: InstrumentKind::Equity,
            lot_size_units: 1,
            tick_size_rupees: 0.05,
            multiplier: 1.0,
            freeze_quantity_units: None,
            cost_segment: None,
            last_trading_day: None,
        }
    }

    #[test]
    fn a_cash_equity_definition_is_valid() {
        assert_eq!(equity().validate(), Ok(()));
        assert!(equity().is_tradable() && !equity().is_derivative());
    }

    #[test]
    fn unusable_definitions_are_refused() {
        let zero_lot = InstrumentSpec {
            lot_size_units: 0,
            ..equity()
        };
        let no_tick = InstrumentSpec {
            tick_size_rupees: 0.0,
            ..equity()
        };
        let bad_multiplier = InstrumentSpec {
            multiplier: f64::NAN,
            ..equity()
        };
        for spec in [zero_lot, no_tick, bad_multiplier] {
            assert!(spec.validate().is_err(), "{spec:?}");
        }
    }
}
