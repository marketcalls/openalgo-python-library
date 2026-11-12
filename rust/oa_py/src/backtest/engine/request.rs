//! An order request from Python (a dict built by `openalgo/backtest/_orders.py`),
//! parsed into `oa_bt::exec::OrderRequest`.
//!
//! Distances cross as (unit, value, atr) with unit "points", "pct", "ticks", "rupees"
//! or "atr" (value is then the multiple and atr the resolved ATR in rupees).

use oa_bt::exec::{InstrumentId, OrderRequest, OrderType, QuantitySpec, Validity};
use oa_bt::model::{Product, Side};
use oa_bt::rms::{
    AfterFirstTarget, BreakevenSpec, Distance, ExitReason, LegRules, TargetSpec, TrailSpec,
};
use oa_bt::sizing::{SizerSpec, SizingBasis};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::setup::{field, optional, unknown};

type DistanceTuple = (String, f64, Option<f64>);

fn distance((unit, value, atr): DistanceTuple) -> PyResult<Distance> {
    match unit.as_str() {
        "points" => Ok(Distance::Points(value)),
        "pct" => Ok(Distance::Percent(value)),
        "ticks" if value >= 0.0 && value.fract() == 0.0 => Ok(Distance::Ticks(value as u32)),
        "rupees" => Ok(Distance::Rupees(value)),
        "atr" => Ok(Distance::Atr {
            multiple: value,
            atr_rupees: atr.unwrap_or(f64::NAN),
        }),
        _ => Err(unknown(
            "distance unit",
            &unit,
            "points, pct, ticks (whole), rupees or atr",
        )),
    }
}

fn leg_rules(dict: &Bound<'_, PyDict>) -> PyResult<LegRules> {
    let targets: Vec<(DistanceTuple, f64)> = optional(dict, "targets")?.unwrap_or_default();
    let trail: Option<(DistanceTuple, Option<DistanceTuple>)> = optional(dict, "trail")?;
    let breakeven: Option<(DistanceTuple, Option<DistanceTuple>)> = optional(dict, "breakeven")?;
    let after: Option<String> = optional(dict, "after_first_target")?;
    Ok(LegRules {
        stop: optional(dict, "stop")?.map(distance).transpose()?,
        target: optional(dict, "target")?.map(distance).transpose()?,
        targets: targets
            .into_iter()
            .map(|(d, fraction)| {
                Ok(TargetSpec {
                    distance: distance(d)?,
                    fraction,
                })
            })
            .collect::<PyResult<_>>()?,
        trail: match trail {
            Some((trigger, step)) => Some(TrailSpec {
                trigger: distance(trigger)?,
                step: step.map(distance).transpose()?,
            }),
            None => None,
        },
        breakeven: match breakeven {
            Some((trigger, offset)) => Some(BreakevenSpec {
                trigger: distance(trigger)?,
                offset: offset.map(distance).transpose()?,
            }),
            None => None,
        },
        after_first_target: match after.as_deref() {
            None | Some("keep") => AfterFirstTarget::Keep,
            Some("breakeven") => AfterFirstTarget::Breakeven,
            Some(other) => return Err(unknown("after_first_target", other, "breakeven")),
        },
    })
}

fn sizer(dict: &Bound<'_, PyDict>) -> PyResult<SizerSpec> {
    let name: String = field(dict, "sizer")?;
    let amount = || field::<f64>(dict, "amount");
    let basis: Option<String> = optional(dict, "basis")?;
    Ok(match name.as_str() {
        "qty" => SizerSpec::Qty(amount()?),
        "lots" => SizerSpec::Lots(field(dict, "amount")?),
        "value" => SizerSpec::Value { rupees: amount()? },
        "percent" => SizerSpec::Percent {
            percent: amount()?,
            basis: match basis.as_deref() {
                None | Some("equity") => SizingBasis::Equity,
                Some("capital") => SizingBasis::InitialCapital,
                Some("cash") => SizingBasis::FreeCash,
                Some(other) => {
                    return Err(unknown("sizing basis", other, "equity, capital or cash"))
                }
            },
        },
        "fixed_risk" => SizerSpec::FixedRisk {
            risk_percent: amount()?,
            units: optional(dict, "units")?.unwrap_or(1),
            hard_limit_units: optional(dict, "hard_limit")?,
        },
        "atr_risk" => SizerSpec::AtrRisk {
            risk_percent: amount()?,
            multiple: field(dict, "multiple")?,
            atr_rupees: optional(dict, "atr")?.unwrap_or(f64::NAN),
        },
        "function" => SizerSpec::Function { units: amount()? },
        "default" => SizerSpec::EngineDefault,
        other => {
            return Err(unknown(
                "sizer",
                other,
                "qty, lots, value, percent, fixed_risk, atr_risk, function or default",
            ))
        }
    })
}

fn quantity(dict: &Bound<'_, PyDict>) -> PyResult<QuantitySpec> {
    let kind: String = field(dict, "kind")?;
    match kind.as_str() {
        "units" => Ok(QuantitySpec::Units(field(dict, "value")?)),
        "lots" => Ok(QuantitySpec::Lots(field(dict, "value")?)),
        "fraction" => Ok(QuantitySpec::PositionFraction(field(dict, "value")?)),
        "sizer" => Ok(QuantitySpec::Sizer(sizer(dict)?)),
        _ => Err(unknown(
            "quantity kind",
            &kind,
            "units, lots, fraction or sizer",
        )),
    }
}

fn order_type(dict: &Bound<'_, PyDict>) -> PyResult<OrderType> {
    let code: String = field(dict, "order_type")?;
    let price = || field::<f64>(dict, "price");
    let trigger = || field::<f64>(dict, "trigger");
    Ok(match code.as_str() {
        "MARKET" => OrderType::Market,
        "LIMIT" => OrderType::Limit {
            limit_rupees: price()?,
        },
        "SL" => OrderType::StopLimit {
            trigger_rupees: trigger()?,
            limit_rupees: price()?,
        },
        "SL-M" => OrderType::StopMarket {
            trigger_rupees: trigger()?,
        },
        _ => return Err(unknown("order type", &code, "MARKET, LIMIT, SL or SL-M")),
    })
}

/// An order request from its dict.
pub(super) fn order_request(dict: &Bound<'_, PyDict>) -> PyResult<OrderRequest> {
    let side: String = field(dict, "side")?;
    let product: String = field(dict, "product")?;
    let validity: String = field(dict, "validity")?;
    let reason: Option<String> = optional(dict, "exit_reason")?;
    let rules: Option<Bound<'_, PyDict>> = optional(dict, "rules")?;
    Ok(OrderRequest {
        instrument: InstrumentId(field(dict, "instrument")?),
        side: Side::from_code(&side).ok_or_else(|| unknown("side", &side, "BUY or SELL"))?,
        quantity: quantity(&field(dict, "quantity")?)?,
        order_type: order_type(dict)?,
        validity: match validity.as_str() {
            "DAY" => Validity::Day,
            "GTT" => Validity::Gtt,
            _ => return Err(unknown("validity", &validity, "DAY or GTT")),
        },
        product: Product::from_code(&product)
            .ok_or_else(|| unknown("product", &product, "CNC, MIS or NRML"))?,
        reduce_only: field(dict, "reduce_only")?,
        reverse: field(dict, "reverse")?,
        exit_reason: match reason {
            None => None,
            Some(code) => Some(ExitReason::from_code(&code).ok_or_else(|| {
                let reasons: Vec<&str> =
                    ExitReason::ALL.iter().map(|reason| reason.code()).collect();
                unknown(
                    "exit reason",
                    &code,
                    &format!("one of {}", reasons.join(", ")),
                )
            })?),
        },
        leg_rules: rules
            .map(|rules| leg_rules(&rules))
            .transpose()?
            .unwrap_or_default(),
        leverage: optional(dict, "leverage")?,
        tag: optional(dict, "tag")?,
        // Only `Engine.rebalance_target` makes a rebalance order.
        rebalance: false,
    })
}
