//! What an order asks for, in a live session's saved state (`venue_state`): its request
//! (instrument, side, quantity, type with its prices, validity, product, flags, exit
//! reason, leg rules, leverage, tag) and the leg rules, with the instrument and position
//! keys every part of the state uses.
//!
//! Codes are the ones the bindings use: "BUY", "MIS", "SL-M", "leg_sl"; a distance is
//! its unit ("points", "pct", "ticks", "rupees", "atr") with its value, and the ATR in
//! rupees for "atr".

use crate::exec::{InstrumentId, OrderRequest, OrderType, QuantitySpec, Validity};
use crate::model::{Product, Side};
use crate::portfolio::PositionKey;
use crate::rms::{
    AfterFirstTarget, BreakevenSpec, Distance, ExitReason, LegRules, TargetSpec, TrailSpec,
};
use crate::sizing::{SizerSpec, SizingBasis};

use super::error::LiveStateError;
use super::value::{list, map, Read, StateValue};

/// The instrument a saved value names, by registration index.
///
/// Errors: `Malformed` for an index the session does not have.
pub(super) fn read_instrument(
    read: &Read<'_>,
    instruments: usize,
) -> Result<InstrumentId, LiveStateError> {
    let index: u32 = read.whole()?;
    if usize::try_from(index).map_or(true, |at| at >= instruments) {
        return Err(read.malformed(format!("names no instrument of the session ({index})")));
    }
    Ok(InstrumentId(index))
}

/// The fields "instrument" and "product" of a position key.
pub(super) fn key_fields(key: PositionKey) -> [(&'static str, StateValue); 2] {
    [
        ("instrument", key.instrument.0.into()),
        ("product", key.product.code().into()),
    ]
}

/// The position key in the fields "instrument" and "product" of `read`.
///
/// Errors: `Malformed` for a field missing or unknown.
pub(super) fn read_key(read: &Read<'_>, instruments: usize) -> Result<PositionKey, LiveStateError> {
    Ok(PositionKey {
        instrument: read_instrument(&read.field("instrument")?, instruments)?,
        product: read.field("product")?.parsed(Product::from_code)?,
    })
}

/// The side in field `key`. Errors: `Malformed` for anything but "BUY" or "SELL".
pub(super) fn read_side(read: &Read<'_>, key: &str) -> Result<Side, LiveStateError> {
    read.field(key)?.parsed(Side::from_code)
}

/// An order request.
pub(super) fn write_request(request: &OrderRequest) -> StateValue {
    let validity = match request.validity {
        Validity::Day => "day",
        Validity::Gtt => "gtt",
    };
    map([
        ("instrument", request.instrument.0.into()),
        ("side", request.side.code().into()),
        ("quantity", write_quantity(request.quantity)),
        ("order_type", request.order_type.code().into()),
        ("limit_rupees", request.order_type.limit_rupees().into()),
        ("trigger_rupees", request.order_type.trigger_rupees().into()),
        ("validity", validity.into()),
        ("product", request.product.code().into()),
        ("reduce_only", request.reduce_only.into()),
        ("reverse", request.reverse.into()),
        (
            "exit_reason",
            request.exit_reason.map(ExitReason::code).into(),
        ),
        ("leg_rules", write_rules(&request.leg_rules)),
        ("leverage", request.leverage.into()),
        ("tag", request.tag.clone().into()),
        ("rebalance", request.rebalance.into()),
    ])
}

/// The order request `write_request` wrote.
///
/// Errors: `Malformed` for a field missing or unusable.
pub(super) fn read_request(
    read: &Read<'_>,
    instruments: usize,
) -> Result<OrderRequest, LiveStateError> {
    let validity = read.field("validity")?;
    Ok(OrderRequest {
        instrument: read_instrument(&read.field("instrument")?, instruments)?,
        side: read_side(read, "side")?,
        quantity: read_quantity(&read.field("quantity")?)?,
        order_type: read_order_type(read)?,
        validity: validity.parsed(|code| match code {
            "day" => Some(Validity::Day),
            "gtt" => Some(Validity::Gtt),
            _ => None,
        })?,
        product: read.field("product")?.parsed(Product::from_code)?,
        reduce_only: read.bool_of("reduce_only")?,
        reverse: read.bool_of("reverse")?,
        exit_reason: read
            .maybe("exit_reason")?
            .map(|reason| reason.parsed(ExitReason::from_code))
            .transpose()?,
        leg_rules: read_rules(&read.field("leg_rules")?)?,
        leverage: read.opt_f64_of("leverage")?,
        tag: read
            .maybe("tag")?
            .map(|tag| tag.text().map(str::to_string))
            .transpose()?,
        rebalance: read.bool_of("rebalance")?,
    })
}

/// The order type in the fields "order_type", "limit_rupees" and "trigger_rupees".
///
/// Errors: `Malformed` for an unknown type or a price it needs missing.
pub(super) fn read_order_type(read: &Read<'_>) -> Result<OrderType, LiveStateError> {
    let code = read.field("order_type")?;
    let limit = || read.f64_of("limit_rupees");
    let trigger = || read.f64_of("trigger_rupees");
    Ok(match code.text()? {
        "MARKET" => OrderType::Market,
        "LIMIT" => OrderType::Limit {
            limit_rupees: limit()?,
        },
        "SL" => OrderType::StopLimit {
            trigger_rupees: trigger()?,
            limit_rupees: limit()?,
        },
        "SL-M" => OrderType::StopMarket {
            trigger_rupees: trigger()?,
        },
        other => return Err(code.malformed(format!("{other:?} is not an order type"))),
    })
}

fn write_quantity(quantity: QuantitySpec) -> StateValue {
    match quantity {
        QuantitySpec::Units(units) => map([("kind", "units".into()), ("value", units.into())]),
        QuantitySpec::Lots(lots) => map([("kind", "lots".into()), ("value", lots.into())]),
        QuantitySpec::PositionFraction(fraction) => {
            map([("kind", "fraction".into()), ("value", fraction.into())])
        }
        QuantitySpec::Sizer(sizer) => {
            map([("kind", "sizer".into()), ("sizer", write_sizer(sizer))])
        }
    }
}

fn read_quantity(read: &Read<'_>) -> Result<QuantitySpec, LiveStateError> {
    let kind = read.field("kind")?;
    Ok(match kind.text()? {
        "units" => QuantitySpec::Units(read.f64_of("value")?),
        "lots" => QuantitySpec::Lots(read.whole_of("value")?),
        "fraction" => QuantitySpec::PositionFraction(read.f64_of("value")?),
        "sizer" => QuantitySpec::Sizer(read_sizer(&read.field("sizer")?)?),
        other => return Err(kind.malformed(format!("{other:?} is not a quantity kind"))),
    })
}

fn basis_code(basis: SizingBasis) -> &'static str {
    match basis {
        SizingBasis::Equity => "equity",
        SizingBasis::InitialCapital => "capital",
        SizingBasis::FreeCash => "cash",
    }
}

fn write_sizer(sizer: SizerSpec) -> StateValue {
    let name = ("sizer", StateValue::from(sizer.code()));
    match sizer {
        SizerSpec::Qty(units) => map([name, ("units", units.into())]),
        SizerSpec::Lots(lots) => map([name, ("lots", lots.into())]),
        SizerSpec::Value { rupees } => map([name, ("rupees", rupees.into())]),
        SizerSpec::Percent { percent, basis } => map([
            name,
            ("percent", percent.into()),
            ("basis", basis_code(basis).into()),
        ]),
        SizerSpec::FixedRisk {
            risk_percent,
            units,
            hard_limit_units,
        } => map([
            name,
            ("risk_percent", risk_percent.into()),
            ("units", units.into()),
            ("hard_limit_units", hard_limit_units.into()),
        ]),
        SizerSpec::AtrRisk {
            risk_percent,
            multiple,
            atr_rupees,
        } => map([
            name,
            ("risk_percent", risk_percent.into()),
            ("multiple", multiple.into()),
            ("atr_rupees", atr_rupees.into()),
        ]),
        SizerSpec::Function { units } => map([name, ("units", units.into())]),
        SizerSpec::EngineDefault => map([name]),
    }
}

fn read_sizer(read: &Read<'_>) -> Result<SizerSpec, LiveStateError> {
    const BASES: [SizingBasis; 3] = [
        SizingBasis::Equity,
        SizingBasis::InitialCapital,
        SizingBasis::FreeCash,
    ];
    let name = read.field("sizer")?;
    Ok(match name.text()? {
        "qty" => SizerSpec::Qty(read.f64_of("units")?),
        "lots" => SizerSpec::Lots(read.whole_of("lots")?),
        "value" => SizerSpec::Value {
            rupees: read.f64_of("rupees")?,
        },
        "percent" => SizerSpec::Percent {
            percent: read.f64_of("percent")?,
            basis: read.field("basis")?.code(&BASES, basis_code)?,
        },
        "fixed_risk" => SizerSpec::FixedRisk {
            risk_percent: read.f64_of("risk_percent")?,
            units: read.whole_of("units")?,
            hard_limit_units: read.opt_f64_of("hard_limit_units")?,
        },
        "atr_risk" => SizerSpec::AtrRisk {
            risk_percent: read.f64_of("risk_percent")?,
            multiple: read.f64_of("multiple")?,
            atr_rupees: read.f64_of("atr_rupees")?,
        },
        "function" => SizerSpec::Function {
            units: read.f64_of("units")?,
        },
        "default" => SizerSpec::EngineDefault,
        other => return Err(name.malformed(format!("{other:?} is not a sizer"))),
    })
}

fn write_distance(distance: Distance) -> StateValue {
    let (unit, value, atr): (&str, StateValue, Option<f64>) = match distance {
        Distance::Points(value) => ("points", value.into(), None),
        Distance::Percent(value) => ("pct", value.into(), None),
        Distance::Ticks(count) => ("ticks", count.into(), None),
        Distance::Rupees(value) => ("rupees", value.into(), None),
        Distance::Atr {
            multiple,
            atr_rupees,
        } => ("atr", multiple.into(), Some(atr_rupees)),
    };
    map([
        ("unit", unit.into()),
        ("value", value),
        ("atr_rupees", atr.into()),
    ])
}

fn read_distance(read: &Read<'_>) -> Result<Distance, LiveStateError> {
    let unit = read.field("unit")?;
    Ok(match unit.text()? {
        "points" => Distance::Points(read.f64_of("value")?),
        "pct" => Distance::Percent(read.f64_of("value")?),
        "ticks" => Distance::Ticks(read.whole_of("value")?),
        "rupees" => Distance::Rupees(read.f64_of("value")?),
        "atr" => Distance::Atr {
            multiple: read.f64_of("value")?,
            atr_rupees: read.f64_of("atr_rupees")?,
        },
        other => return Err(unit.malformed(format!("{other:?} is not a distance unit"))),
    })
}

fn read_opt_distance(read: &Read<'_>, key: &str) -> Result<Option<Distance>, LiveStateError> {
    read.maybe(key)?
        .map(|found| read_distance(&found))
        .transpose()
}

/// A pair of distances, the second optional (a trail's trigger and step, a breakeven's
/// trigger and offset).
fn write_pair(first: Distance, second: Option<Distance>, names: [&str; 2]) -> StateValue {
    map([
        (names[0], write_distance(first)),
        (names[1], second.map(write_distance).into()),
    ])
}

/// Leg rules.
pub(super) fn write_rules(rules: &LegRules) -> StateValue {
    let after = match rules.after_first_target {
        AfterFirstTarget::Keep => "keep",
        AfterFirstTarget::Breakeven => "breakeven",
    };
    map([
        ("stop", rules.stop.map(write_distance).into()),
        ("target", rules.target.map(write_distance).into()),
        (
            "targets",
            list(&rules.targets, |target| {
                map([
                    ("distance", write_distance(target.distance)),
                    ("fraction", target.fraction.into()),
                ])
            }),
        ),
        (
            "trail",
            rules
                .trail
                .map(|trail| write_pair(trail.trigger, trail.step, ["trigger", "step"]))
                .into(),
        ),
        (
            "breakeven",
            rules
                .breakeven
                .map(|be| write_pair(be.trigger, be.offset, ["trigger", "offset"]))
                .into(),
        ),
        ("after_first_target", after.into()),
    ])
}

/// The leg rules `write_rules` wrote.
///
/// Errors: `Malformed` for a field missing or unusable.
pub(super) fn read_rules(read: &Read<'_>) -> Result<LegRules, LiveStateError> {
    let targets = read
        .items_of("targets")?
        .iter()
        .map(|target| {
            Ok(TargetSpec {
                distance: read_distance(&target.field("distance")?)?,
                fraction: target.f64_of("fraction")?,
            })
        })
        .collect::<Result<_, LiveStateError>>()?;
    let trail = match read.maybe("trail")? {
        Some(trail) => Some(TrailSpec {
            trigger: read_distance(&trail.field("trigger")?)?,
            step: read_opt_distance(&trail, "step")?,
        }),
        None => None,
    };
    let breakeven = match read.maybe("breakeven")? {
        Some(be) => Some(BreakevenSpec {
            trigger: read_distance(&be.field("trigger")?)?,
            offset: read_opt_distance(&be, "offset")?,
        }),
        None => None,
    };
    let after = read.field("after_first_target")?;
    Ok(LegRules {
        stop: read_opt_distance(read, "stop")?,
        target: read_opt_distance(read, "target")?,
        targets,
        trail,
        breakeven,
        after_first_target: after.parsed(|code| match code {
            "keep" => Some(AfterFirstTarget::Keep),
            "breakeven" => Some(AfterFirstTarget::Breakeven),
            _ => None,
        })?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Product;

    fn round_trip(request: &OrderRequest) -> Result<OrderRequest, LiveStateError> {
        let written = write_request(request);
        read_request(&Read::root(&written), 1)
    }

    #[test]
    fn a_request_reads_back_as_it_was_written() {
        let rules = LegRules {
            stop: Some(Distance::Atr {
                multiple: 2.0,
                atr_rupees: 3.5,
            }),
            targets: vec![TargetSpec {
                distance: Distance::Ticks(40),
                fraction: 0.5,
            }],
            trail: Some(TrailSpec {
                trigger: Distance::Percent(1.0),
                step: Some(Distance::Points(0.5)),
            }),
            breakeven: Some(BreakevenSpec {
                trigger: Distance::Rupees(500.0),
                offset: None,
            }),
            after_first_target: AfterFirstTarget::Breakeven,
            ..LegRules::default()
        };
        let sizer = SizerSpec::Percent {
            percent: 10.0,
            basis: SizingBasis::FreeCash,
        };
        let request = OrderRequest {
            order_type: OrderType::StopLimit {
                trigger_rupees: 499.0,
                limit_rupees: 474.05,
            },
            product: Product::Mis,
            reduce_only: true,
            exit_reason: Some(ExitReason::LegSl),
            leg_rules: rules,
            leverage: Some(5.0),
            tag: Some("breakout".to_string()),
            ..OrderRequest::market(InstrumentId(0), Side::Sell, QuantitySpec::Sizer(sizer))
        };
        assert_eq!(round_trip(&request), Ok(request));
    }

    #[test]
    fn every_quantity_and_sizer_reads_back() {
        let quantities = [
            QuantitySpec::Units(75.0),
            QuantitySpec::Lots(2),
            QuantitySpec::PositionFraction(0.5),
            QuantitySpec::Sizer(SizerSpec::Qty(10.0)),
            QuantitySpec::Sizer(SizerSpec::Lots(3)),
            QuantitySpec::Sizer(SizerSpec::Value { rupees: 50_000.0 }),
            QuantitySpec::Sizer(SizerSpec::FixedRisk {
                risk_percent: 1.0,
                units: 2,
                hard_limit_units: Some(500.0),
            }),
            QuantitySpec::Sizer(SizerSpec::AtrRisk {
                risk_percent: 0.5,
                multiple: 2.0,
                atr_rupees: 4.25,
            }),
            QuantitySpec::Sizer(SizerSpec::Function { units: 30.0 }),
            QuantitySpec::Sizer(SizerSpec::EngineDefault),
        ];
        for quantity in quantities {
            let request = OrderRequest::market(InstrumentId(0), Side::Buy, quantity);
            assert_eq!(round_trip(&request), Ok(request), "{quantity:?}");
        }
    }

    #[test]
    fn a_request_for_an_instrument_the_session_lacks_is_refused() {
        let request = OrderRequest::market(InstrumentId(3), Side::Buy, QuantitySpec::Units(1.0));
        let refused = round_trip(&request);
        assert!(
            matches!(&refused, Err(LiveStateError::Malformed { path, .. }) if path == "live_state.instrument"),
            "{refused:?}"
        );
    }
}
