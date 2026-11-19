//! Leg rules scaled per trade (design 17, section 2.12, randomized exits): the
//! randomized-exits check keeps the strategy's decisions and varies the distances of its
//! engine-held exits, trade by trade.
//!
//! With `exit_scales` given (`Perturbation`), the leg a trade opens takes the next factor
//! (trade k factor k, in the order the legs open, from the first again past the last),
//! and every distance of its rules is multiplied by it: the stop, the target or each of
//! the targets, the trail's trigger and step, the breakeven's trigger and offset. A
//! distance in points, percent or rupees is multiplied as it stands, an ATR distance's
//! multiple is, and one in ticks becomes the same number of rupees of price (ticks x
//! tick size x factor), since a scaled tick count would round. Target fractions and
//! `after_first_target` stay. An add takes its trade's factor, so the rule that an add
//! repeats the position's rules (SPEC-14) compares like with like; a futures roll carries
//! the leg and its factor to the next contract and takes none of its own. The order's
//! own rules, and what they size at the decision, are not changed: only the levels of
//! the leg.

use crate::portfolio::{FillKind, PositionKey};
use crate::rms::{BreakevenSpec, Distance, LegRules, TargetSpec, TrailSpec};

use super::fills::FillRequest;
use super::futures::FillOrigin;
use super::kernel::Engine;

/// The factors of the legs open now, by position, and the trades opened so far.
#[derive(Clone, Debug, Default)]
pub(super) struct ExitScales {
    opened: usize,
    legs: Vec<(PositionKey, f64)>,
}

/// `distance` multiplied by `factor` (module docs); `tick_rupees` turns ticks to price.
fn scaled(distance: Distance, factor: f64, tick_rupees: f64) -> Distance {
    match distance {
        Distance::Points(value) => Distance::Points(value * factor),
        Distance::Percent(value) => Distance::Percent(value * factor),
        Distance::Rupees(value) => Distance::Rupees(value * factor),
        Distance::Ticks(count) => Distance::Points(f64::from(count) * tick_rupees * factor),
        Distance::Atr {
            multiple,
            atr_rupees,
        } => Distance::Atr {
            multiple: multiple * factor,
            atr_rupees,
        },
    }
}

/// `rules` with every distance multiplied by `factor` (module docs).
pub(super) fn scaled_rules(rules: &LegRules, factor: f64, tick_rupees: f64) -> LegRules {
    let scale = |distance: Distance| scaled(distance, factor, tick_rupees);
    LegRules {
        stop: rules.stop.map(scale),
        target: rules.target.map(scale),
        targets: rules
            .targets
            .iter()
            .map(|target| TargetSpec {
                distance: scale(target.distance),
                fraction: target.fraction,
            })
            .collect(),
        trail: rules.trail.map(|trail| TrailSpec {
            trigger: scale(trail.trigger),
            step: trail.step.map(scale),
        }),
        breakeven: rules.breakeven.map(|breakeven| BreakevenSpec {
            trigger: scale(breakeven.trigger),
            offset: breakeven.offset.map(scale),
        }),
        after_first_target: rules.after_first_target,
    }
}

impl Engine {
    /// The rules of the leg `request` opens or adds to (`kind`), scaled by its trade's
    /// factor (module docs); the request's own rules when the run scales none. Opening a
    /// trade takes the next factor.
    pub(super) fn leg_rules_of(&mut self, request: &FillRequest, kind: FillKind) -> LegRules {
        let factors = &self.config.perturbation.exit_scales;
        if factors.is_empty() || request.origin == FillOrigin::Roll {
            return request.leg_rules.clone();
        }
        let key = request.key;
        let factor = match kind {
            FillKind::Entry => {
                let factor = factors[self.exit_scales.opened % factors.len()];
                self.exit_scales.opened += 1;
                self.exit_scales.legs.retain(|(k, _)| *k != key);
                self.exit_scales.legs.push((key, factor));
                factor
            }
            _ => self.exit_scale_of(key),
        };
        let tick = self
            .instruments
            .get(key.instrument.index())
            .map_or(0.0, |spec| spec.tick_size_rupees);
        scaled_rules(&request.leg_rules, factor, tick)
    }

    /// The factor of the trade open on `key`, 1 when none was taken.
    fn exit_scale_of(&self, key: PositionKey) -> f64 {
        self.exit_scales
            .legs
            .iter()
            .find(|(k, _)| *k == key)
            .map_or(1.0, |(_, factor)| *factor)
    }

    /// A futures roll moved the leg of `from` to `to`: its factor goes with it.
    pub(super) fn carry_exit_scale(&mut self, from: PositionKey, to: PositionKey) {
        let factor = self.exit_scale_of(from);
        self.exit_scales.legs.retain(|(k, _)| *k != to);
        self.exit_scales.legs.push((to, factor));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rms::AfterFirstTarget;

    #[test]
    fn every_distance_is_multiplied_and_ticks_become_price() {
        let rules = LegRules {
            stop: Some(Distance::Percent(2.0)),
            targets: vec![TargetSpec {
                distance: Distance::Ticks(10),
                fraction: 0.5,
            }],
            trail: Some(TrailSpec {
                trigger: Distance::Points(4.0),
                step: Some(Distance::Rupees(100.0)),
            }),
            breakeven: Some(BreakevenSpec {
                trigger: Distance::Atr {
                    multiple: 2.0,
                    atr_rupees: 3.0,
                },
                offset: None,
            }),
            after_first_target: AfterFirstTarget::Breakeven,
            ..LegRules::default()
        };
        let scaled = scaled_rules(&rules, 1.25, 0.05);
        assert_eq!(scaled.stop, Some(Distance::Percent(2.5)));
        // 10 ticks of 0.05 is 0.50 of price; 1.25 times is 0.625.
        assert_eq!(scaled.targets[0].distance, Distance::Points(0.625));
        assert_eq!(scaled.targets[0].fraction, 0.5);
        let trail = scaled.trail.map(|t| (t.trigger, t.step));
        assert_eq!(
            trail,
            Some((Distance::Points(5.0), Some(Distance::Rupees(125.0))))
        );
        let breakeven = scaled.breakeven.map(|b| b.trigger);
        let multiple = Distance::Atr {
            multiple: 2.5,
            atr_rupees: 3.0,
        };
        assert_eq!(breakeven, Some(multiple));
        assert_eq!(scaled.after_first_target, AfterFirstTarget::Breakeven);
    }

    #[test]
    fn a_factor_of_one_keeps_every_rule_but_ticks() {
        let rules = LegRules {
            stop: Some(Distance::Points(3.0)),
            target: Some(Distance::Percent(4.0)),
            ..LegRules::default()
        };
        assert_eq!(scaled_rules(&rules, 1.0, 0.05), rules);
    }
}
