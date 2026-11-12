//! Risk rule configuration: what a strategy asks for, before anything is evaluated.
//!
//! Units (05-risk-management 2, decisions SPEC-24 and SPEC-25): a distance is in points
//! (rupees of price), percent of the leg's fill average, ticks, rupees of leg MTM, or
//! an ATR multiple whose ATR value the strategy layer resolves at the decision (the
//! ATR at the close before the entry), so the kernel never computes an indicator. A
//! bare number in `bt.trail(x, y)` and `bt.breakeven(trigger, offset)` is points.
//!
//! Every level is computed from the leg's reference price, which is the average price
//! of the position (`reference="avg_price"`, 12-position-sizing 4), and rounded to the
//! nearest tick unless `round_levels` is off (SPEC-03).

use super::error::RmsError;

/// A distance from the reference price.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Distance {
    /// Rupees of price (`bt.points(x)` or a bare number).
    Points(f64),
    /// Percent of the leg's fill average (`bt.pct(x)`): a fixed number of points.
    Percent(f64),
    /// Whole ticks of the instrument (`bt.ticks(n)`).
    Ticks(u32),
    /// Rupees of the leg's MTM (`bt.rupees(x)`): points are this divided by quantity
    /// times multiplier.
    Rupees(f64),
    /// A multiple of the ATR value resolved at the decision (`bt.atr(mult, period)`).
    Atr { multiple: f64, atr_rupees: f64 },
}

impl Distance {
    /// The amount, for validation.
    fn amount(self) -> f64 {
        match self {
            Distance::Points(value) | Distance::Percent(value) | Distance::Rupees(value) => value,
            Distance::Ticks(count) => f64::from(count),
            Distance::Atr {
                multiple,
                atr_rupees,
            } => multiple * atr_rupees,
        }
    }

    /// Errors: `InvalidRule` unless the distance is finite and positive.
    pub fn validate(self, what: &'static str) -> Result<(), RmsError> {
        let amount = self.amount();
        if amount.is_finite() && amount > 0.0 {
            Ok(())
        } else {
            Err(RmsError::InvalidRule {
                rule: what,
                reason: format!("{self:?} must be a positive distance"),
            })
        }
    }
}

/// One of several targets: its distance and the fraction of the position it closes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TargetSpec {
    /// Distance from the reference price.
    pub distance: Distance,
    /// Fraction (0 to 1) of each tranche assigned to this target (SPEC-08, SPEC-09).
    pub fraction: f64,
}

/// A trailing stop in OpenAlgo's `{x, y}` form (05-risk-management 3).
///
/// `step` zero is continuous: the trail arms when the favourable excursion reaches
/// `trigger`, and the stop is the best price minus `trigger` (long). `step` above zero
/// is stepped: the stop rises by `step` for every `trigger` of favourable move and
/// needs a stop. `bt.trail_points(x)` is `{x, 0}`, `bt.trail_pct(p)` is `{p percent of
/// the fill average, 0}` and `bt.trail_atr` is `{ATR multiple, 0}` (SPEC-25).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrailSpec {
    /// The `x` of `{x, y}`.
    pub trigger: Distance,
    /// The `y` of `{x, y}`; `None` is the continuous form.
    pub step: Option<Distance>,
}

/// Move the stop to the reference price plus `offset` once the favourable excursion
/// reaches `trigger`. Moves only if that tightens the stop, and creates a stop when
/// none exists (SPEC-24).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BreakevenSpec {
    /// Favourable excursion that arms the move.
    pub trigger: Distance,
    /// Offset beyond the reference price (may cover the round-trip costs); `None`
    /// moves the stop to the reference price itself.
    pub offset: Option<Distance>,
}

/// What happens to the stop after the first target fills.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AfterFirstTarget {
    /// Nothing.
    #[default]
    Keep,
    /// Move the stop to breakeven.
    Breakeven,
}

/// The leg rules of one order: they attach to the position its fill opens, and an add
/// may repeat them but not change them (SPEC-14).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LegRules {
    /// Fixed stop.
    pub stop: Option<Distance>,
    /// Single target (exclusive with `targets`).
    pub target: Option<Distance>,
    /// Several targets with partial exits.
    pub targets: Vec<TargetSpec>,
    /// Trailing stop.
    pub trail: Option<TrailSpec>,
    /// Breakeven move.
    pub breakeven: Option<BreakevenSpec>,
    /// Stop move after the first target.
    pub after_first_target: AfterFirstTarget,
    /// Exit after this many base bars in the position (`max_hold`), reason `time_exit`.
    pub max_hold_bars: Option<u32>,
}

impl LegRules {
    /// True when no leg rule is set.
    pub fn is_empty(&self) -> bool {
        *self == LegRules::default()
    }

    /// Check the rules for contradictions before the order is accepted.
    ///
    /// Errors: `InvalidRule` for a distance that is not positive, a target together
    /// with targets, target fractions outside 0 to 1 or above 1 in total, and a stepped
    /// trail without a stop (an error unless `strict_config` is off, 05-risk 8).
    pub fn validate(&self, strict_config: bool) -> Result<(), RmsError> {
        let invalid = |reason: &str| RmsError::InvalidRule {
            rule: "leg rules",
            reason: reason.to_string(),
        };
        if self.target.is_some() && !self.targets.is_empty() {
            return Err(invalid("give either target= or targets=, not both"));
        }
        let mut total = 0.0;
        for target in &self.targets {
            target.distance.validate("target")?;
            if !(target.fraction > 0.0 && target.fraction <= 1.0) {
                return Err(invalid(
                    "each target fraction must be above 0 and at most 1",
                ));
            }
            total += target.fraction;
        }
        if total > 1.0 + f64::EPSILON {
            return Err(invalid("target fractions add up to more than 1"));
        }
        for (distance, what) in [(self.stop, "stop"), (self.target, "target")] {
            if let Some(distance) = distance {
                distance.validate(what)?;
            }
        }
        if let Some(trail) = self.trail {
            trail.trigger.validate("trail trigger")?;
            if trail.step.is_some() && self.stop.is_none() && strict_config {
                return Err(invalid("a stepped trail needs a stop"));
            }
        }
        Ok(())
    }
}

/// Which P&L the combined and session rules use, in every mode (14-strategy-api 3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RiskBasis {
    /// Before charges (the live module's basis).
    #[default]
    Gross,
    /// After the modelled charges.
    Net,
}

/// What follows a strategy-level exit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AfterRiskExit {
    /// Entries are locked until the next session.
    #[default]
    LockSession,
    /// Trading continues.
    AllowReentry,
}

/// Limits checked before every add (12-position-sizing 2). No limit unless set
/// (SPEC-15); averaging down is judged at the decision price.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScaleLimits {
    /// Additional entries a condition strategy may make while in a position.
    pub pyramiding: u32,
    /// Most adds to one position.
    pub max_adds: Option<u32>,
    /// Largest position, in units.
    pub max_position_units: Option<f64>,
    /// Fewest base bars between two adds.
    pub min_spacing_bars: Option<u32>,
    /// Largest open risk of the whole position after an add, in rupees.
    pub max_risk_rupees: Option<f64>,
    /// Allow adding while the decision price is worse than the average price.
    pub allow_averaging_down: bool,
}

/// The strategy- and session-level risk configuration of a run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrategyRiskConfig {
    /// Daily loss limit in rupees (`max_daily_loss`). Once the day's loss on the run's
    /// `risk_basis` (realized plus open MTM, SPEC-16) reaches it, the positions exit
    /// and the strategy is REDUCING until the next session, whatever `after_risk_exit`
    /// says; every new entry is then rejected `daily_loss_gate` at placement. The block
    /// has no switch, and before a breach there is no loss-budget check (lead ruling).
    pub max_daily_loss_rupees: Option<f64>,
    /// P&L basis of combined and session rules.
    pub risk_basis: RiskBasis,
    /// What follows a combined stop or lock-profit exit (not a daily-loss breach).
    pub after_risk_exit: AfterRiskExit,
    /// Most open positions.
    pub max_positions: Option<u32>,
    /// Halt at this drawdown from peak equity, percent. It latches: HALTED for the rest
    /// of a backtest, until an operator reset in live (lead ruling).
    pub drawdown_halt_percent: Option<f64>,
    /// Base bars after an exit before a new entry.
    pub cooldown_bars: Option<u32>,
    /// Round levels to the tick (engine default; `False` restores the live module).
    pub round_levels: bool,
    /// Configuration errors are raised rather than ignored (engine default).
    pub strict_config: bool,
    /// Scale-in limits.
    pub scale: ScaleLimits,
}

impl Default for StrategyRiskConfig {
    /// The engine defaults of 05-risk-management 8: levels rounded, strict.
    fn default() -> StrategyRiskConfig {
        StrategyRiskConfig {
            max_daily_loss_rupees: None,
            risk_basis: RiskBasis::Gross,
            after_risk_exit: AfterRiskExit::LockSession,
            max_positions: None,
            drawdown_halt_percent: None,
            cooldown_bars: None,
            round_levels: true,
            strict_config: true,
            scale: ScaleLimits::default(),
        }
    }
}

impl StrategyRiskConfig {
    /// The rules that act on a run or a session rather than on one leg (steps 6 and
    /// the pre-trade gates). Names of the ones that are set, for messages.
    pub fn aggregate_rules_set(&self) -> Vec<&'static str> {
        let mut set = Vec::new();
        if self.max_daily_loss_rupees.is_some() {
            set.push("max_daily_loss");
        }
        if self.max_positions.is_some() {
            set.push("max_positions");
        }
        if self.drawdown_halt_percent.is_some() {
            set.push("drawdown_halt");
        }
        if self.cooldown_bars.is_some() {
            set.push("cooldown");
        }
        set
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_rules_are_valid() {
        assert!(LegRules::default().is_empty());
        assert_eq!(LegRules::default().validate(true), Ok(()));
    }

    #[test]
    fn contradictory_rules_are_refused() {
        let both = LegRules {
            target: Some(Distance::Points(10.0)),
            targets: vec![TargetSpec {
                distance: Distance::Points(5.0),
                fraction: 0.5,
            }],
            ..LegRules::default()
        };
        let stepped_without_stop = LegRules {
            trail: Some(TrailSpec {
                trigger: Distance::Points(2.0),
                step: Some(Distance::Points(1.0)),
            }),
            ..LegRules::default()
        };
        assert!(both.validate(true).is_err());
        assert!(stepped_without_stop.validate(true).is_err());
        assert_eq!(stepped_without_stop.validate(false), Ok(()));
        assert!(Distance::Points(-1.0).validate("stop").is_err());
    }
}
