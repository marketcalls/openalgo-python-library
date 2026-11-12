//! Net positions built from tranches (12-position-sizing 1).
//!
//! One net position per instrument and product. Every entry fill is a tranche; the
//! average price is the quantity-weighted average of the open tranches. A reduction
//! realizes P&L at average cost and shrinks every open tranche in proportion, which
//! keeps the average of the remaining tranches equal to the average cost (FIFO
//! matching is a later option). A fill never crosses zero: a reversal is a close and
//! then an open, two calls with their own costs and events.

use std::fmt;

use crate::exec::instrument::InstrumentId;
use crate::model::{Product, Side, TsNanos};

/// Relative tolerance below which a quantity counts as zero (fractional quantities).
const FLAT_TOLERANCE: f64 = 1.0e-9;

/// What identifies a net position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PositionKey {
    pub instrument: InstrumentId,
    pub product: Product,
}

/// One entry fill still (partly) held.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tranche {
    /// Position in the position's tranche list since it opened (0 is the entry).
    pub number: u32,
    /// Units the entry fill added.
    pub opened_units: f64,
    /// Units of it still held (shrinks in proportion on every reduction).
    pub open_units: f64,
    /// Fill price, rupees.
    pub price_rupees: f64,
    /// Fill instant, UTC nanoseconds.
    pub opened_at: TsNanos,
}

/// How a fill changed a position (the position event it produces).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PositionChange {
    /// From flat to a position.
    Opened = 0,
    /// The quantity changed and the position is still open.
    Changed = 1,
    /// Back to flat.
    Closed = 2,
}

impl PositionChange {
    /// The event name.
    pub fn code(self) -> &'static str {
        match self {
            PositionChange::Opened => "PositionOpened",
            PositionChange::Changed => "PositionChanged",
            PositionChange::Closed => "PositionClosed",
        }
    }
}

/// Whether a fill opened, added to or reduced a position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillKind {
    /// From flat.
    Entry,
    /// Same side as the position.
    Add,
    /// Opposite side, at most the position's quantity.
    Reduce,
}

/// The effect of one fill on a position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FillEffect {
    /// Entry, add or reduction.
    pub kind: FillKind,
    /// The position event.
    pub change: PositionChange,
    /// P&L realized by the fill at average cost, rupees, before charges.
    pub realized_rupees: f64,
    /// Signed quantity after the fill, units.
    pub quantity_after: f64,
}

/// An inconsistency in the bookkeeping.
#[derive(Clone, Debug, PartialEq)]
pub enum PortfolioError {
    /// A fill with a non-positive quantity or price.
    InvalidFill { reason: String },
    /// A fill that would take the position through zero (a reversal is two fills).
    CrossesZero {
        position_units: f64,
        fill_units: f64,
    },
    /// Capital that is not a positive amount.
    InvalidCapital { capital: f64 },
}

impl fmt::Display for PortfolioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PortfolioError::InvalidFill { reason } => write!(f, "invalid fill: {reason}"),
            PortfolioError::CrossesZero {
                position_units,
                fill_units,
            } => write!(
                f,
                "kernel defect: a fill of {fill_units} units would cross a position of \
                 {position_units} units through zero"
            ),
            PortfolioError::InvalidCapital { capital } => write!(
                f,
                "capital {capital} is not usable: give a positive amount of rupees"
            ),
        }
    }
}

impl std::error::Error for PortfolioError {}

/// One net position.
#[derive(Clone, Debug, PartialEq)]
pub struct Position {
    /// Instrument and product.
    pub key: PositionKey,
    /// Rupees per one-rupee move per unit.
    pub multiplier: f64,
    /// Signed quantity in units: positive long, negative short, zero flat.
    pub quantity_units: f64,
    /// Average price of the open tranches, rupees (0 when flat).
    pub average_price_rupees: f64,
    /// Open tranches, oldest first.
    pub tranches: Vec<Tranche>,
    /// Adds since the position opened.
    pub adds: u32,
    /// P&L realized on this instrument and product over the run, rupees, gross.
    pub realized_rupees: f64,
    /// Margin the position blocks, rupees.
    pub margin_rupees: f64,
    /// Last mark price, rupees (the last close seen).
    pub mark_rupees: Option<f64>,
    /// True when the last mark is a last close carried into a slice with no bar.
    pub stale: bool,
}

impl Position {
    /// A flat position.
    pub fn flat(key: PositionKey, multiplier: f64) -> Position {
        Position {
            key,
            multiplier,
            quantity_units: 0.0,
            average_price_rupees: 0.0,
            tranches: Vec::new(),
            adds: 0,
            realized_rupees: 0.0,
            margin_rupees: 0.0,
            mark_rupees: None,
            stale: false,
        }
    }

    /// True when no quantity is held.
    pub fn is_flat(&self) -> bool {
        self.quantity_units.abs() <= FLAT_TOLERANCE
    }

    /// Unrealized P&L at the last mark, rupees (0 when flat or never marked).
    pub fn unrealized_rupees(&self) -> f64 {
        match self.mark_rupees {
            Some(mark) if !self.is_flat() => {
                (mark - self.average_price_rupees) * self.quantity_units * self.multiplier
            }
            _ => 0.0,
        }
    }

    /// Apply one fill.
    ///
    /// Errors: `InvalidFill` for a non-positive quantity or price; `CrossesZero` when
    /// an opposite fill is larger than the position.
    pub fn apply(
        &mut self,
        side: Side,
        units: f64,
        price_rupees: f64,
        at: TsNanos,
    ) -> Result<FillEffect, PortfolioError> {
        if !(units.is_finite() && units > 0.0 && price_rupees.is_finite() && price_rupees > 0.0) {
            return Err(PortfolioError::InvalidFill {
                reason: format!("{units} units at {price_rupees} rupees"),
            });
        }
        if self.is_flat() {
            return Ok(self.open(side, units, price_rupees, at));
        }
        let long = self.quantity_units > 0.0;
        if long == (side == Side::Buy) {
            return Ok(self.add(side, units, price_rupees, at));
        }
        self.reduce(side, units, price_rupees)
    }

    fn tranche(&self, units: f64, price_rupees: f64, at: TsNanos) -> Tranche {
        Tranche {
            number: self.adds + u32::from(!self.tranches.is_empty()),
            opened_units: units,
            open_units: units,
            price_rupees,
            opened_at: at,
        }
    }

    fn open(&mut self, side: Side, units: f64, price_rupees: f64, at: TsNanos) -> FillEffect {
        self.tranches = vec![self.tranche(units, price_rupees, at)];
        self.quantity_units = side.sign() * units;
        self.average_price_rupees = price_rupees;
        self.adds = 0;
        self.effect(FillKind::Entry, PositionChange::Opened, 0.0)
    }

    fn add(&mut self, side: Side, units: f64, price_rupees: f64, at: TsNanos) -> FillEffect {
        let held = self.quantity_units.abs();
        let cost = held * self.average_price_rupees + units * price_rupees;
        self.tranches.push(self.tranche(units, price_rupees, at));
        self.quantity_units += side.sign() * units;
        self.average_price_rupees = cost / (held + units);
        self.adds += 1;
        self.effect(FillKind::Add, PositionChange::Changed, 0.0)
    }

    fn reduce(
        &mut self,
        side: Side,
        units: f64,
        price_rupees: f64,
    ) -> Result<FillEffect, PortfolioError> {
        let held = self.quantity_units.abs();
        if units > held * (1.0 + FLAT_TOLERANCE) {
            return Err(PortfolioError::CrossesZero {
                position_units: self.quantity_units,
                fill_units: units,
            });
        }
        let direction = self.quantity_units.signum();
        let realized =
            (price_rupees - self.average_price_rupees) * units * direction * self.multiplier;
        let remaining_share = (held - units).max(0.0) / held;
        self.quantity_units += side.sign() * units;
        self.realized_rupees += realized;
        self.margin_rupees *= remaining_share;
        for tranche in &mut self.tranches {
            tranche.open_units *= remaining_share;
        }
        if self.is_flat() {
            self.quantity_units = 0.0;
            self.average_price_rupees = 0.0;
            self.tranches.clear();
            self.adds = 0;
            self.margin_rupees = 0.0;
            return Ok(self.effect(FillKind::Reduce, PositionChange::Closed, realized));
        }
        Ok(self.effect(FillKind::Reduce, PositionChange::Changed, realized))
    }

    fn effect(&self, kind: FillKind, change: PositionChange, realized: f64) -> FillEffect {
        FillEffect {
            kind,
            change,
            realized_rupees: realized,
            quantity_after: self.quantity_units,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn position() -> Position {
        let key = PositionKey {
            instrument: InstrumentId(0),
            product: Product::Mis,
        };
        Position::flat(key, 1.0)
    }

    #[test]
    fn a_reduction_realizes_at_average_cost_and_keeps_the_average() {
        // Trace 3: 100 at 1000.00, add 100 at 1004.00 (average 1002.00), sell 100 at
        // 1012.00, add 100 at 1012.00: the average is 1007.00.
        let mut held = position();
        let steps = [
            (Side::Buy, 1000.0, FillKind::Entry, 0.0),
            (Side::Buy, 1004.0, FillKind::Add, 0.0),
            (Side::Sell, 1012.0, FillKind::Reduce, 1000.0),
            (Side::Buy, 1012.0, FillKind::Add, 0.0),
        ];
        for (side, price, kind, realized) in steps {
            let effect = held.apply(side, 100.0, price, 0);
            assert_eq!(
                effect.map(|e| (e.kind, e.realized_rupees)),
                Ok((kind, realized))
            );
        }
        assert_eq!(held.average_price_rupees, 1007.0);
        assert_eq!(held.quantity_units, 200.0);
        let open: f64 = held.tranches.iter().map(|t| t.open_units).sum();
        assert_eq!(open, 200.0);
    }

    #[test]
    fn closing_returns_to_flat() {
        let mut held = position();
        let opened = held.apply(Side::Sell, 50.0, 420.0, 0).map(|e| e.change);
        assert_eq!(opened, Ok(PositionChange::Opened));
        let closed = held.apply(Side::Buy, 50.0, 422.7, 1);
        assert_eq!(closed.map(|e| e.change), Ok(PositionChange::Closed));
        assert!(held.is_flat());
        assert!((held.realized_rupees + 135.0).abs() < 1e-9);
    }

    #[test]
    fn a_fill_never_crosses_zero() {
        let mut held = position();
        assert!(held.apply(Side::Buy, 100.0, 500.0, 0).is_ok());
        let crossing = held.apply(Side::Sell, 150.0, 501.0, 1);
        assert!(matches!(crossing, Err(PortfolioError::CrossesZero { .. })));
        assert!(held.apply(Side::Buy, 0.0, 500.0, 2).is_err());
    }
}
