//! The controls shared by every sizer (12-position-sizing 5, FR-SIZE-07): what equity a
//! sizer sizes from, and the caps on the quantity it gives.
//!
//! `reinvest` (on by default) sizes from the current equity, so sizes compound; off, it
//! sizes from the capital allocated at the start. Everything in sizing that speaks of
//! equity reads that one figure (`SizingContext::sizing_equity`): the percent sizer on
//! its equity basis, fixed and ATR risk, the engine default, the percent and leverage
//! caps, and `target(percent=)`.
//!
//! The caps bound what a sizer asks for, and the tightest wins:
//!
//! | Cap | Bounds |
//! |---|---|
//! | `position_units`, `position_lots`, `position_rupees`, `position_percent` | the position on the order's side after the order: units, whole lots, its value at the sizing price, or that value as a percent of the sizing equity |
//! | `leverage` | that value as a multiple of the sizing equity |
//! | `risk_rupees`, `risk_percent` | the order's own loss at its stop (units times stop distance times multiplier), in rupees or percent of the sizing equity |
//!
//! Units already held on the order's side use up a position cap; an opposite position
//! leaves room for the part of the order that only closes it. A cap reduces the
//! quantity and the venue rule then rounds it down; when nothing is left the order is
//! skipped with reason `cap`. An order without a stop has no bounded loss, so under a
//! risk cap it is skipped. Caps bound sizers only: an explicit `qty=` or `lots=` and
//! `target()` trade the quantity asked for (the scale-in limits still judge adds).

use super::{positive, unit_value, SizingContext, SizingError};

/// Percent in a whole.
const PERCENT: f64 = 100.0;

/// The caps on what a sizer gives; `None` sets no cap. See the module documentation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SizeCaps {
    /// Largest position on the order's side after it, units.
    pub position_units: Option<f64>,
    /// Largest position on the order's side after it, whole lots.
    pub position_lots: Option<u32>,
    /// Largest value of that position at the sizing price, rupees.
    pub position_rupees: Option<f64>,
    /// Largest value of that position, percent of the sizing equity.
    pub position_percent: Option<f64>,
    /// Largest value of that position as a multiple of the sizing equity (1 is 1x).
    pub leverage: Option<f64>,
    /// Largest loss of the order at its stop, rupees.
    pub risk_rupees: Option<f64>,
    /// Largest loss of the order at its stop, percent of the sizing equity.
    pub risk_percent: Option<f64>,
}

/// The sizing controls of a run: the strategy's `reinvest` and `size_caps`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SizingControls {
    /// Size from the current equity (true, the default) or from the initial capital.
    pub reinvest: bool,
    /// Caps on what a sizer gives.
    pub caps: SizeCaps,
}

impl Default for SizingControls {
    /// Reinvesting, with no cap: the documented default of 12-position-sizing 5.
    fn default() -> SizingControls {
        SizingControls {
            reinvest: true,
            caps: SizeCaps::default(),
        }
    }
}

fn usable(value: Option<f64>) -> bool {
    value.map_or(true, |value| value.is_finite() && value > 0.0)
}

impl SizingControls {
    /// Check the controls before the run.
    ///
    /// Errors: `InvalidSizer` for a cap that is not a positive finite number, a cap of
    /// zero lots, or a risk cap of 100 percent of equity or more.
    pub fn validate(&self) -> Result<(), SizingError> {
        let caps = &self.caps;
        let amounts = [
            ("position units", caps.position_units),
            ("position rupees", caps.position_rupees),
            ("position percent", caps.position_percent),
            ("leverage", caps.leverage),
            ("risk rupees", caps.risk_rupees),
            ("risk percent", caps.risk_percent),
        ];
        let problem = if let Some((name, _)) = amounts.iter().find(|(_, cap)| !usable(*cap)) {
            Some(format!("the {name} cap must be a positive number"))
        } else if caps.position_lots == Some(0) {
            Some("the position lots cap must be at least 1".to_string())
        } else if caps.risk_percent.is_some_and(|percent| percent >= PERCENT) {
            Some("the risk percent cap must be below 100 percent of equity".to_string())
        } else {
            None
        };
        match problem {
            Some(reason) => Err(SizingError::InvalidSizer { reason }),
            None => Ok(()),
        }
    }
}

impl SizeCaps {
    /// Units the caps leave for an order in `ctx`, before the venue rule: infinite with
    /// no cap, never below zero.
    ///
    /// Errors: `InvalidSizer` when a cap on value needs the sizing price and it is not a
    /// positive number.
    pub(super) fn room_units(&self, ctx: &SizingContext) -> Result<f64, SizingError> {
        let equity = ctx.sizing_equity();
        let held = ctx.held_on_side_units;
        let mut room = f64::INFINITY;
        let mut bound = |units: f64| room = room.min(units);
        if let Some(units) = self.position_units {
            bound(units - held);
        }
        if let Some(lots) = self.position_lots {
            bound(f64::from(lots) * f64::from(ctx.lot_size_units.max(1)) - held);
        }
        let values = [
            self.position_rupees,
            self.position_percent
                .map(|percent| equity * percent / PERCENT),
            self.leverage.map(|multiple| equity * multiple),
        ];
        for value in values.into_iter().flatten() {
            bound(value / unit_value(ctx)? - held);
        }
        let losses = [
            self.risk_rupees,
            self.risk_percent.map(|percent| equity * percent / PERCENT),
        ];
        for loss in losses.into_iter().flatten() {
            match ctx.stop_distance_rupees {
                Some(distance) if distance > 0.0 => {
                    bound(loss / (distance * positive("the multiplier", ctx.multiplier)?));
                }
                _ => bound(0.0),
            }
        }
        Ok(room.max(0.0))
    }
}
