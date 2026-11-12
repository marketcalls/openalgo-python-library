//! The events of a GTT's life (decision GTT-8): placed, modified, fired with its child,
//! the child filled or cancelled, and cancelled. The documents describe the life cycle
//! without naming its events; names and fields are those of the worked traces.
//!
//! | Event | Step | Fields set |
//! |---|---|---|
//! | `GttPlaced` | 2 after the fill that opened the position, 7 for a strategy GTT, 4 after a re-plan, 1 for a re-creation | quantity, triggers |
//! | `GttModified` | 4 (a trail, a breakeven, an add or a partial exit) | quantity, triggers |
//! | `GttTriggered` | 3 | quantity, triggers, fired leg, child limit |
//! | `GttChildFilled` | 3 | quantity, child limit, price |
//! | `GttChildCanceled` | 3 | quantity, child limit |
//! | `GttCanceled` | the step of the engine exit it precedes, 7 when the strategy cancels, 1 for a re-creation | quantity, triggers |

use crate::model::Product;

use super::gtt::{Gtt, GttId, GttLegRole};
use super::instrument::InstrumentId;

/// The name of a GTT event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GttEventName {
    /// The GTT was placed.
    Placed = 0,
    /// Its triggers or quantity were replaced.
    Modified = 1,
    /// A trigger fired and the child order was placed.
    Triggered = 2,
    /// The child order filled.
    ChildFilled = 3,
    /// The child order did not fill through its bar and was cancelled.
    ChildCanceled = 4,
    /// The GTT was cancelled.
    Canceled = 5,
}

impl GttEventName {
    /// The event name in results.
    pub fn code(self) -> &'static str {
        match self {
            GttEventName::Placed => "GttPlaced",
            GttEventName::Modified => "GttModified",
            GttEventName::Triggered => "GttTriggered",
            GttEventName::ChildFilled => "GttChildFilled",
            GttEventName::ChildCanceled => "GttChildCanceled",
            GttEventName::Canceled => "GttCanceled",
        }
    }
}

/// One GTT event. Prices are rupees, the quantity units; a field the event does not
/// carry is `None` (see the module table).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GttEvent {
    pub name: GttEventName,
    pub gtt: GttId,
    pub instrument: InstrumentId,
    pub product: Product,
    /// The child order's quantity.
    pub quantity_units: f64,
    pub stop_trigger_rupees: Option<f64>,
    pub target_trigger_rupees: Option<f64>,
    /// The trigger of a strategy's own GTT.
    pub order_trigger_rupees: Option<f64>,
    /// The leg that fired.
    pub fired_leg: Option<GttLegRole>,
    /// The child order's limit.
    pub child_limit_rupees: Option<f64>,
    /// The child's fill price.
    pub price_rupees: Option<f64>,
}

impl GttEvent {
    /// The event `name` for `gtt` as it stands, with the fields that name carries. For
    /// `ChildFilled` the price is left `None`; see `child_filled`.
    pub fn of(name: GttEventName, gtt: &Gtt) -> GttEvent {
        let shows_triggers = !matches!(
            name,
            GttEventName::ChildFilled | GttEventName::ChildCanceled
        );
        let shows_child = matches!(
            name,
            GttEventName::Triggered | GttEventName::ChildFilled | GttEventName::ChildCanceled
        );
        let trigger = |role| shows_triggers.then(|| gtt.trigger(role)).flatten();
        let child = gtt
            .fired
            .and_then(|role| gtt.leg(role))
            .map(|leg| leg.child_limit_rupees);
        GttEvent {
            name,
            gtt: gtt.id,
            instrument: gtt.instrument,
            product: gtt.product,
            quantity_units: gtt.quantity_units,
            stop_trigger_rupees: trigger(GttLegRole::Stop),
            target_trigger_rupees: trigger(GttLegRole::Target),
            order_trigger_rupees: trigger(GttLegRole::Order),
            fired_leg: (name == GttEventName::Triggered)
                .then_some(gtt.fired)
                .flatten(),
            child_limit_rupees: shows_child.then_some(child).flatten(),
            price_rupees: None,
        }
    }

    /// `GttChildFilled` for `gtt`, filled at `price_rupees`.
    pub fn child_filled(gtt: &Gtt, price_rupees: f64) -> GttEvent {
        GttEvent {
            price_rupees: Some(price_rupees),
            ..GttEvent::of(GttEventName::ChildFilled, gtt)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::gtt::{GttKind, GttLeg, GttPurpose, GttSlot, GttStatus};
    use crate::exec::matching::TriggerDirection;
    use crate::model::Side;

    /// Trace 06b's OCO: stop 1473.90 (child 1459.15) and target 1556.65 (child
    /// 1541.05), its stop fired.
    fn fired_oco() -> Gtt {
        let leg = |role, trigger, direction, child| GttLeg {
            role,
            trigger_rupees: trigger,
            direction,
            child_limit_rupees: child,
        };
        Gtt {
            id: GttId(1),
            instrument: InstrumentId(0),
            product: Product::Cnc,
            side: Side::Sell,
            quantity_units: 10.0,
            kind: GttKind::Oco,
            legs: vec![
                leg(GttLegRole::Stop, 1473.9, TriggerDirection::Falling, 1459.15),
                leg(
                    GttLegRole::Target,
                    1556.65,
                    TriggerDirection::Rising,
                    1541.05,
                ),
            ],
            purpose: GttPurpose::Protect(GttSlot::Target(0)),
            status: GttStatus::Triggered,
            created_on: 0,
            eligible_from: 0,
            last_modified_slice: None,
            fired: Some(GttLegRole::Stop),
        }
    }

    #[test]
    fn a_trigger_event_carries_both_triggers_the_fired_leg_and_its_child_limit() {
        let event = GttEvent::of(GttEventName::Triggered, &fired_oco());
        assert_eq!(event.stop_trigger_rupees, Some(1473.9));
        assert_eq!(event.target_trigger_rupees, Some(1556.65));
        assert_eq!(event.fired_leg, Some(GttLegRole::Stop));
        assert_eq!(event.child_limit_rupees, Some(1459.15));
        assert_eq!(event.price_rupees, None);
    }

    #[test]
    fn a_child_fill_carries_the_child_limit_and_the_price_but_no_trigger() {
        let event = GttEvent::child_filled(&fired_oco(), 1473.9);
        assert_eq!(event.name.code(), "GttChildFilled");
        assert_eq!((event.stop_trigger_rupees, event.fired_leg), (None, None));
        assert_eq!(event.child_limit_rupees, Some(1459.15));
        assert_eq!(event.price_rupees, Some(1473.9));
    }

    #[test]
    fn a_placement_carries_the_triggers_and_nothing_of_a_child() {
        let event = GttEvent::of(GttEventName::Placed, &fired_oco());
        assert_eq!(event.quantity_units, 10.0);
        assert_eq!(event.stop_trigger_rupees, Some(1473.9));
        assert_eq!((event.fired_leg, event.child_limit_rupees), (None, None));
    }
}
