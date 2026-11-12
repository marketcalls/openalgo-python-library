//! The scale-in limits (12-position-sizing 2, decision SPEC-15; reject code
//! `scale_in_limit` of 05-risk-management 7): what a strategy may add to a position it
//! already holds.
//!
//! The limits are checked at submission, in step 7 of the decision slice, after the
//! cooldown and the maximum positions (03-architecture 4.4 item 7), for an order that
//! adds to a position on its own side. An entry from flat is governed by the entry
//! gates instead (`entry_gates`). Each limit sets nothing unless it is given, except
//! averaging down, which is refused unless `allow_averaging_down` is set (SPEC-15).
//!
//! Every limit is judged at the decision price, the last close the strategy saw when it
//! ordered (SPEC-15): the add, and every add still waiting for its fill, count as if
//! they filled at that price. So adds decided together cannot pass a limit between them.
//!
//! | Limit | An add is refused when |
//! |---|---|
//! | `max_adds` | the position already has that many adds, filled or waiting |
//! | `max_position` | the position after the add is above the cap: units, lots, its value at the decision price, or that value as a percent of equity |
//! | `min_spacing` | fewer base bars than given have closed after the bar in which the last tranche filled (counted as the cooldown counts them), or the decision price is nearer to the last tranche's price than the distance given |
//! | `max_risk` | the open risk after the add is above the cap, or the position would have no stop |
//! | `allow_averaging_down` (false) | the decision price is worse than the average price: below it for a long, above it for a short |
//!
//! The open risk of a position is what it loses if its stop is hit, from the average
//! price: (average price minus stop) times units times multiplier for a long, the
//! mirror for a short, never below zero. The stop after the add is the leg's own rule
//! (12-position-sizing 4): recomputed from the new average and never looser than the
//! stop in force.

use crate::exec::RejectCode;
use crate::model::Side;

use super::error::RmsError;
use super::rules::{Distance, ScaleLimits};

/// Relative tolerance on caps and distances, so a position exactly at a cap after
/// floating-point arithmetic is not refused.
const CAP_TOLERANCE: f64 = 1.0e-9;
/// Percent in a whole.
const PERCENT: f64 = 100.0;

/// The largest position `max_position` allows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PositionCap {
    /// Units (shares, or units of the underlying).
    Units(f64),
    /// Whole lots of the instrument.
    Lots(u32),
    /// Rupees of value at the decision price (units times price times multiplier).
    Rupees(f64),
    /// That value as a percent of the strategy's equity.
    PercentOfEquity(f64),
}

/// The spacing `min_spacing` asks between two tranches of a position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Spacing {
    /// Base bars closed after the bar in which the last tranche filled.
    Bars(u32),
    /// A price distance from the last tranche's fill price: points, a percent of that
    /// price, or ticks of the instrument.
    Price(Distance),
}

/// The largest open risk `max_risk` allows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RiskCap {
    /// Rupees.
    Rupees(f64),
    /// A percent of the strategy's equity.
    PercentOfEquity(f64),
}

/// What the scale-in gate knows about one add, in plain numbers (rupees, units).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AddFacts {
    /// `Buy` for a long position, `Sell` for a short one.
    pub side: Side,
    /// Units held now, absolute.
    pub held_units: f64,
    /// Average price of the units held; `None` while nothing has filled (the add joins
    /// an entry still waiting).
    pub average_price_rupees: Option<f64>,
    /// Adds the position has: filled, and waiting for their fill.
    pub adds: u32,
    /// Units of the adds waiting and of this add (read by `max_position` and
    /// `max_risk`).
    pub adding_units: f64,
    /// The decision price: the instrument's last close; `None` before its first close.
    pub decision_price_rupees: Option<f64>,
    /// Base bars closed after the bar in which the last tranche filled (0 when a
    /// tranche is still waiting).
    pub bars_since_last_tranche: usize,
    /// The last tranche's price: its fill price, or the decision price for one still
    /// waiting.
    pub last_tranche_price_rupees: Option<f64>,
    /// The strategy's equity, rupees.
    pub equity_rupees: f64,
    /// Lot size, units.
    pub lot_size_units: u32,
    /// Tick size, rupees.
    pub tick_rupees: f64,
    /// Rupees per one-rupee move per unit.
    pub multiplier: f64,
    /// Open risk of the position after the add, rupees; `None` when it would have no
    /// stop (read by `max_risk` only).
    pub open_risk_rupees: Option<f64>,
}

impl AddFacts {
    /// Units held after the add.
    pub fn units_after(&self) -> f64 {
        self.held_units + self.adding_units
    }
}

/// The scale-in gate: the first limit an add breaks refuses it.
///
/// Errors: `RejectCode::ScaleInLimit` when the add breaks a limit, or when a limit
/// that needs the decision price is set and the instrument has had no close yet.
pub fn scale_in_gate(limits: &ScaleLimits, add: &AddFacts) -> Result<(), RejectCode> {
    let refused = Err(RejectCode::ScaleInLimit);
    if limits.max_adds.is_some_and(|most| add.adds >= most) {
        return refused;
    }
    if let Some(cap) = limits.max_position {
        if !position_fits(cap, add)? {
            return refused;
        }
    }
    if let Some(spacing) = limits.min_spacing {
        if !spaced(spacing, add)? {
            return refused;
        }
    }
    if let Some(cap) = limits.max_risk {
        let most = match cap {
            RiskCap::Rupees(rupees) => rupees,
            RiskCap::PercentOfEquity(percent) => add.equity_rupees * percent / PERCENT,
        };
        match add.open_risk_rupees {
            Some(risk) if within(risk, most) => {}
            _ => return refused,
        }
    }
    if !limits.allow_averaging_down && averages_down(add)? {
        return refused;
    }
    Ok(())
}

/// The decision price, or a refusal when there is none yet.
fn price(add: &AddFacts) -> Result<f64, RejectCode> {
    add.decision_price_rupees.ok_or(RejectCode::ScaleInLimit)
}

/// True when `value` is at most `cap`, within the tolerance.
fn within(value: f64, cap: f64) -> bool {
    value <= cap * (1.0 + CAP_TOLERANCE)
}

fn position_fits(cap: PositionCap, add: &AddFacts) -> Result<bool, RejectCode> {
    let units = add.units_after();
    Ok(match cap {
        PositionCap::Units(most) => within(units, most),
        PositionCap::Lots(lots) => within(units, f64::from(lots) * f64::from(add.lot_size_units)),
        PositionCap::Rupees(most) => within(units * price(add)? * add.multiplier, most),
        PositionCap::PercentOfEquity(percent) => {
            let value = units * price(add)? * add.multiplier;
            within(value, add.equity_rupees * percent / PERCENT)
        }
    })
}

fn spaced(spacing: Spacing, add: &AddFacts) -> Result<bool, RejectCode> {
    let distance = match spacing {
        Spacing::Bars(bars) => return Ok(add.bars_since_last_tranche >= bars as usize),
        Spacing::Price(distance) => distance,
    };
    let last = add
        .last_tranche_price_rupees
        .ok_or(RejectCode::ScaleInLimit)?;
    let needed = match distance {
        Distance::Points(points) => points,
        Distance::Percent(percent) => last * percent / PERCENT,
        Distance::Ticks(ticks) => f64::from(ticks) * add.tick_rupees,
        // Refused by `ScaleLimits::validate` before the run.
        Distance::Rupees(_) | Distance::Atr { .. } => return Err(RejectCode::ScaleInLimit),
    };
    let moved = (price(add)? - last).abs();
    Ok(moved >= needed * (1.0 - CAP_TOLERANCE))
}

/// True when the decision price is worse than the average price of the units held.
fn averages_down(add: &AddFacts) -> Result<bool, RejectCode> {
    let Some(average) = add.average_price_rupees else {
        return Ok(false);
    };
    let decision = price(add)?;
    Ok(match add.side {
        Side::Buy => decision < average,
        Side::Sell => decision > average,
    })
}

impl ScaleLimits {
    /// True when no limit is set and averaging down is allowed: the gate can refuse
    /// nothing.
    pub fn is_open(&self) -> bool {
        self.max_adds.is_none()
            && self.max_position.is_none()
            && self.min_spacing.is_none()
            && self.max_risk.is_none()
            && self.allow_averaging_down
    }

    /// Check the limits before the run.
    ///
    /// Errors: `InvalidRule` for a cap or distance that is not a positive finite number,
    /// a spacing of zero bars, a spacing in rupees or ATR (give points, a percent or
    /// ticks), or a risk cap of 100 percent of equity or more.
    pub fn validate(&self) -> Result<(), RmsError> {
        let invalid = |rule: &'static str, reason: &str| RmsError::InvalidRule {
            rule,
            reason: reason.to_string(),
        };
        let positive = |value: f64| value.is_finite() && value > 0.0;
        let position_ok = match self.max_position {
            None => true,
            Some(PositionCap::Lots(lots)) => lots > 0,
            Some(
                PositionCap::Units(value)
                | PositionCap::Rupees(value)
                | PositionCap::PercentOfEquity(value),
            ) => positive(value),
        };
        if !position_ok {
            return Err(invalid(
                "max_position",
                "give a positive number of units, lots, rupees or percent of equity",
            ));
        }
        match self.min_spacing {
            Some(Spacing::Bars(0)) => {
                return Err(invalid("min_spacing", "give at least 1 bar, or None"));
            }
            Some(Spacing::Price(Distance::Rupees(_) | Distance::Atr { .. })) => {
                return Err(invalid(
                    "min_spacing",
                    "give bars, or a price distance in points, percent or ticks",
                ));
            }
            Some(Spacing::Price(distance)) => distance.validate("min_spacing")?,
            _ => {}
        }
        let risk_ok = match self.max_risk {
            None => true,
            Some(RiskCap::Rupees(rupees)) => positive(rupees),
            Some(RiskCap::PercentOfEquity(percent)) => positive(percent) && percent < PERCENT,
        };
        if risk_ok {
            Ok(())
        } else {
            Err(invalid(
                "max_risk",
                "give positive rupees, or a percent of equity above 0 and below 100",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A long of 100 units at an average of 1000.00, its last tranche filled 3 bars ago
    /// at 1000.00, adding 50 units at a decision price of 1010.00; equity 500000.00, lot
    /// 25, tick 0.05, open risk after the add 1500.00.
    fn long_add() -> AddFacts {
        AddFacts {
            side: Side::Buy,
            held_units: 100.0,
            average_price_rupees: Some(1000.0),
            adds: 1,
            adding_units: 50.0,
            decision_price_rupees: Some(1010.0),
            bars_since_last_tranche: 3,
            last_tranche_price_rupees: Some(1000.0),
            equity_rupees: 500_000.0,
            lot_size_units: 25,
            tick_rupees: 0.05,
            multiplier: 1.0,
            open_risk_rupees: Some(1_500.0),
        }
    }

    fn allowing(limits: ScaleLimits) -> ScaleLimits {
        ScaleLimits {
            allow_averaging_down: true,
            ..limits
        }
    }

    const REFUSED: Result<(), RejectCode> = Err(RejectCode::ScaleInLimit);

    #[test]
    fn no_limit_is_set_unless_given() {
        assert_eq!(scale_in_gate(&ScaleLimits::default(), &long_add()), Ok(()));
    }

    #[test]
    fn max_adds_counts_the_adds_already_made() {
        let limits = |most| ScaleLimits {
            max_adds: Some(most),
            ..ScaleLimits::default()
        };
        assert_eq!(scale_in_gate(&limits(2), &long_add()), Ok(()));
        assert_eq!(scale_in_gate(&limits(1), &long_add()), REFUSED);
        assert_eq!(scale_in_gate(&limits(0), &long_add()), REFUSED);
    }

    #[test]
    fn max_position_caps_the_position_after_the_add_in_each_unit() {
        // After the add: 150 units, 6 lots of 25, 150 x 1010.00 = 151500.00 of value,
        // which is 30.3 percent of 500000.00.
        let cases = [
            (PositionCap::Units(150.0), Ok(())),
            (PositionCap::Units(149.0), REFUSED),
            (PositionCap::Lots(6), Ok(())),
            (PositionCap::Lots(5), REFUSED),
            (PositionCap::Rupees(151_500.0), Ok(())),
            (PositionCap::Rupees(151_000.0), REFUSED),
            (PositionCap::PercentOfEquity(30.3), Ok(())),
            (PositionCap::PercentOfEquity(30.0), REFUSED),
        ];
        for (cap, expected) in cases {
            let limits = ScaleLimits {
                max_position: Some(cap),
                ..ScaleLimits::default()
            };
            assert_eq!(scale_in_gate(&limits, &long_add()), expected, "{cap:?}");
        }
    }

    #[test]
    fn min_spacing_in_bars_counts_the_bars_closed_since_the_last_tranche() {
        let limits = |bars| ScaleLimits {
            min_spacing: Some(Spacing::Bars(bars)),
            ..ScaleLimits::default()
        };
        assert_eq!(scale_in_gate(&limits(3), &long_add()), Ok(()));
        assert_eq!(scale_in_gate(&limits(4), &long_add()), REFUSED);
    }

    #[test]
    fn min_spacing_as_a_price_distance_is_measured_from_the_last_tranche() {
        // The decision price 1010.00 is 10.00 from the last tranche at 1000.00: one
        // percent of it, or 200 ticks of 0.05.
        let cases = [
            (Distance::Points(10.0), Ok(())),
            (Distance::Points(10.05), REFUSED),
            (Distance::Percent(1.0), Ok(())),
            (Distance::Percent(1.1), REFUSED),
            (Distance::Ticks(200), Ok(())),
            (Distance::Ticks(201), REFUSED),
        ];
        for (distance, expected) in cases {
            let limits = ScaleLimits {
                min_spacing: Some(Spacing::Price(distance)),
                ..ScaleLimits::default()
            };
            assert_eq!(
                scale_in_gate(&limits, &long_add()),
                expected,
                "{distance:?}"
            );
        }
    }

    #[test]
    fn max_risk_caps_the_open_risk_after_the_add_and_needs_a_stop() {
        let limits = |cap| ScaleLimits {
            max_risk: Some(cap),
            ..ScaleLimits::default()
        };
        assert_eq!(
            scale_in_gate(&limits(RiskCap::Rupees(1_500.0)), &long_add()),
            Ok(())
        );
        assert_eq!(
            scale_in_gate(&limits(RiskCap::Rupees(1_499.0)), &long_add()),
            REFUSED
        );
        // 0.3 percent of 500000.00 is 1500.00.
        let percent = limits(RiskCap::PercentOfEquity(0.3));
        assert_eq!(scale_in_gate(&percent, &long_add()), Ok(()));
        let no_stop = AddFacts {
            open_risk_rupees: None,
            ..long_add()
        };
        assert_eq!(
            scale_in_gate(&limits(RiskCap::Rupees(1e12)), &no_stop),
            REFUSED
        );
    }

    #[test]
    fn averaging_down_is_refused_unless_allowed_and_judged_at_the_decision_price() {
        let lower = AddFacts {
            decision_price_rupees: Some(999.95),
            ..long_add()
        };
        let equal = AddFacts {
            decision_price_rupees: Some(1000.0),
            ..long_add()
        };
        let default = ScaleLimits::default();
        assert_eq!(scale_in_gate(&default, &lower), REFUSED);
        assert_eq!(scale_in_gate(&default, &equal), Ok(()));
        assert_eq!(scale_in_gate(&allowing(default), &lower), Ok(()));
        // A short averages down when the decision price is above the average.
        let short = |decision| AddFacts {
            side: Side::Sell,
            decision_price_rupees: Some(decision),
            ..long_add()
        };
        assert_eq!(scale_in_gate(&default, &short(1000.05)), REFUSED);
        assert_eq!(scale_in_gate(&default, &short(995.0)), Ok(()));
    }

    #[test]
    fn an_add_to_an_entry_still_waiting_has_no_average_to_average_down_from() {
        let waiting = AddFacts {
            held_units: 0.0,
            average_price_rupees: None,
            decision_price_rupees: Some(900.0),
            ..long_add()
        };
        assert_eq!(scale_in_gate(&ScaleLimits::default(), &waiting), Ok(()));
    }

    #[test]
    fn a_limit_on_value_cannot_be_judged_before_the_first_close() {
        let unpriced = AddFacts {
            decision_price_rupees: None,
            average_price_rupees: None,
            ..long_add()
        };
        let value = ScaleLimits {
            max_position: Some(PositionCap::Rupees(1e12)),
            ..ScaleLimits::default()
        };
        let units = ScaleLimits {
            max_position: Some(PositionCap::Units(1e6)),
            ..ScaleLimits::default()
        };
        assert_eq!(scale_in_gate(&value, &unpriced), REFUSED);
        assert_eq!(scale_in_gate(&units, &unpriced), Ok(()));
    }

    #[test]
    fn unusable_limits_are_refused_before_the_run() {
        let cases = [
            ScaleLimits {
                max_position: Some(PositionCap::Units(0.0)),
                ..ScaleLimits::default()
            },
            ScaleLimits {
                max_position: Some(PositionCap::Lots(0)),
                ..ScaleLimits::default()
            },
            ScaleLimits {
                min_spacing: Some(Spacing::Bars(0)),
                ..ScaleLimits::default()
            },
            ScaleLimits {
                min_spacing: Some(Spacing::Price(Distance::Rupees(100.0))),
                ..ScaleLimits::default()
            },
            ScaleLimits {
                min_spacing: Some(Spacing::Price(Distance::Points(-1.0))),
                ..ScaleLimits::default()
            },
            ScaleLimits {
                max_risk: Some(RiskCap::PercentOfEquity(100.0)),
                ..ScaleLimits::default()
            },
            ScaleLimits {
                max_risk: Some(RiskCap::Rupees(f64::NAN)),
                ..ScaleLimits::default()
            },
        ];
        for limits in cases {
            assert!(limits.validate().is_err(), "{limits:?}");
        }
        assert_eq!(ScaleLimits::default().validate(), Ok(()));
    }
}
