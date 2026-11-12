//! ATR distances and the ATR-risk sizer in the orders of a condition strategy
//! (`bt.atr`, `bt.trail_atr`, `bt.size.atr_risk` in `entry_when` and `short_when`).
//!
//! An `on_bar` order resolves each ATR when it is placed: the `openalgo.ta` ATR of the
//! strategy bars closed by then (SPEC-25). A condition's order template is built once,
//! so the Python layer computes every ATR the template names over the run's strategy
//! bars, one value per bar (the ATR is causal, so its value at a bar reads bars up to
//! that bar only), and the driver puts the deciding bar's value into a copy of the
//! template before it submits it. Both forms therefore place the same orders.

use crate::exec::{OrderRequest, QuantitySpec};
use crate::model::TsNanos;
use crate::rms::Distance;
use crate::sizing::SizerSpec;

use super::super::error::KernelError;
use super::invalid;

/// Where an ATR goes in an order template.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AtrSlot {
    /// `stop=`.
    Stop,
    /// `target=`.
    Target,
    /// The target at this position of `targets=`.
    Targets(usize),
    /// The trigger of `trail=` (`bt.trail_atr`).
    TrailTrigger,
    /// The step of `trail=`.
    TrailStep,
    /// The trigger of `breakeven=`.
    BreakevenTrigger,
    /// The offset of `breakeven=`.
    BreakevenOffset,
    /// The ATR of `size=bt.size.atr_risk(...)`.
    Sizer,
}

/// One ATR an order template names: where it goes, its period (named in messages) and
/// its value at every strategy bar in rupees, NaN where it has none (fewer than
/// `period` bars closed).
#[derive(Clone, Debug, PartialEq)]
pub struct AtrInput {
    /// Where the value goes.
    pub slot: AtrSlot,
    /// The ATR period, strategy bars.
    pub period: u32,
    /// The ATR at each strategy bar, rupees.
    pub rupees: Vec<f64>,
}

fn atr_of(distance: Option<&mut Distance>) -> Option<&mut f64> {
    match distance {
        Some(Distance::Atr { atr_rupees, .. }) => Some(atr_rupees),
        _ => None,
    }
}

/// The ATR value an order holds at `slot`, when that slot holds an ATR.
fn slot_mut(order: &mut OrderRequest, slot: AtrSlot) -> Option<&mut f64> {
    let rules = &mut order.leg_rules;
    match slot {
        AtrSlot::Stop => atr_of(rules.stop.as_mut()),
        AtrSlot::Target => atr_of(rules.target.as_mut()),
        AtrSlot::Targets(index) => atr_of(rules.targets.get_mut(index).map(|t| &mut t.distance)),
        AtrSlot::TrailTrigger => atr_of(rules.trail.as_mut().map(|trail| &mut trail.trigger)),
        AtrSlot::TrailStep => atr_of(rules.trail.as_mut().and_then(|trail| trail.step.as_mut())),
        AtrSlot::BreakevenTrigger => atr_of(rules.breakeven.as_mut().map(|b| &mut b.trigger)),
        AtrSlot::BreakevenOffset => {
            atr_of(rules.breakeven.as_mut().and_then(|b| b.offset.as_mut()))
        }
        AtrSlot::Sizer => match &mut order.quantity {
            QuantitySpec::Sizer(SizerSpec::AtrRisk { atr_rupees, .. }) => Some(atr_rupees),
            _ => None,
        },
    }
}

/// Every slot of `order` that holds an ATR, in slot order.
fn atr_slots(order: &OrderRequest) -> Vec<AtrSlot> {
    let mut probe = order.clone();
    let fixed = [
        AtrSlot::Stop,
        AtrSlot::Target,
        AtrSlot::TrailTrigger,
        AtrSlot::TrailStep,
        AtrSlot::BreakevenTrigger,
        AtrSlot::BreakevenOffset,
        AtrSlot::Sizer,
    ];
    let targets = (0..order.leg_rules.targets.len()).map(AtrSlot::Targets);
    let mut slots: Vec<AtrSlot> = fixed
        .into_iter()
        .chain(targets)
        .filter(|&slot| slot_mut(&mut probe, slot).is_some())
        .collect();
    slots.sort();
    slots
}

/// Check that `inputs` give exactly the ATRs `order` names, one value per bar.
///
/// Errors: `InvalidConfig` (field `conditions`) naming `name` for an ATR the template
/// names without a value, a value for a slot that holds no ATR, or a series whose
/// length is not the number of strategy bars.
pub(super) fn check_inputs(
    name: &str,
    order: &OrderRequest,
    inputs: &[AtrInput],
    bars: usize,
) -> Result<(), KernelError> {
    let mut given: Vec<AtrSlot> = inputs.iter().map(|input| input.slot).collect();
    given.sort();
    if given != atr_slots(order) {
        return Err(invalid(format!(
            "the {name} order names ATRs at {:?} but values were given for {given:?}",
            atr_slots(order)
        )));
    }
    match inputs.iter().find(|input| input.rupees.len() != bars) {
        Some(input) => Err(invalid(format!(
            "the {name} ATR of period {} has {} values for {bars} strategy bars",
            input.period,
            input.rupees.len()
        ))),
        None => Ok(()),
    }
}

/// The template with the ATRs of strategy bar `bar` (opening at `ts_open`, UTC
/// nanoseconds) put in.
///
/// Errors: `InvalidConfig` (field `conditions`) when an ATR distance has no positive
/// value at that bar, as `on_bar` refuses one; the ATR-risk sizer takes its value as it
/// is and refuses a missing one itself, as in `on_bar`.
pub(super) fn order_at(
    template: &OrderRequest,
    inputs: &[AtrInput],
    bar: usize,
    ts_open: TsNanos,
) -> Result<OrderRequest, KernelError> {
    let mut order = template.clone();
    for input in inputs {
        let value = input.rupees.get(bar).copied().unwrap_or(f64::NAN);
        let usable = value.is_finite() && value > 0.0;
        if !usable && input.slot != AtrSlot::Sizer {
            return Err(invalid(format!(
                "bt.atr(period={period}) has no positive value at the strategy bar opening at \
                 {ts_open} (UTC nanoseconds): fewer than {period} strategy bars have closed, \
                 or the series has not moved. Require the ATR in the condition, for example \
                 & (self.I(ta.atr, self.high, self.low, self.close, {period}) > 0)",
                period = input.period
            )));
        }
        if let Some(slot) = slot_mut(&mut order, input.slot) {
            *slot = value;
        }
    }
    Ok(order)
}
