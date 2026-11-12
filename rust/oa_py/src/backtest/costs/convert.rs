//! Conversions between the plain tuples that cross the Python boundary and the
//! kernel's cost types, and the mapping of `CostError` to a Python exception.
//!
//! The Python wrapper owns all parsing of user input (dicts, dates, strings); by the
//! time data reaches this module it is typed tuples, so nothing here inspects Python
//! objects. Value validation stays in the kernel.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use oa_bt::market::costs::{
    BrokerageRule, CostBreakdown, CostError, CostSchedule, RateStatus, Segment, SegmentRates,
    SegmentRow, TaxBase, TaxUnit,
};

/// A brokerage rule: `("zero", 0, 0)`, `("flat", rupees, 0)`, `("percent", percent,
/// cap)` in percent units, or `("rate", fraction, cap)`. `cap = inf` means no cap.
pub(super) type RuleTuple = (String, f64, f64);

/// One fill's cost in rupees: brokerage, stt, exchange_txn, sebi_fee, stamp_duty,
/// gst, total.
pub(super) type CostTuple = (f64, f64, f64, f64, f64, f64, f64);

/// Which schedule version priced a trade (its position) and the trust id of its
/// rates.
pub(super) type StatusTuple = (usize, u8);

/// The seven rates of a row, as fractions: stt_buy, stt_sell, exchange_txn, ipft,
/// sebi, stamp_buy, stt_exercise.
pub(super) type RatesTuple = (f64, f64, f64, f64, f64, f64, f64);

/// One rate row: segment code, rates, verified_through (IST day or None),
/// calculator tax base and unit codes, notes.
pub(super) type RowTuple = (String, RatesTuple, Option<i32>, String, String, String);

/// One schedule version: id, valid_from and valid_to (IST days; valid_to None while
/// in force), gst_rate, gst_on_sebi_fee, gst_verified_through, source, notes, rate rows.
pub(super) type ScheduleTuple = (
    String,
    i32,
    Option<i32>,
    f64,
    bool,
    Option<i32>,
    String,
    String,
    Vec<RowTuple>,
);

const RULE_ZERO: &str = "zero";
const RULE_FLAT: &str = "flat";
const RULE_PERCENT: &str = "percent";
const RULE_RATE: &str = "rate";

/// A `CostError` as a Python `ValueError` with the same message. The Python wrapper
/// re-raises it as `openalgo.backtest.errors.CostError`.
pub(super) fn cost_err(error: CostError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// A batch error as a Python `ValueError` naming the row.
pub(super) fn row_err((row, error): (usize, CostError)) -> PyErr {
    PyValueError::new_err(format!("row {row}: {error}"))
}

/// A breakdown and its reported total (`CostModel::reported_total`) as a tuple.
pub(super) fn cost_tuple(cost: &CostBreakdown, total: f64) -> CostTuple {
    let [brokerage, stt, exchange_txn, sebi_fee, stamp_duty, gst] = cost.components();
    (
        brokerage,
        stt,
        exchange_txn,
        sebi_fee,
        stamp_duty,
        gst,
        total,
    )
}

pub(super) fn status_tuple(status: RateStatus) -> StatusTuple {
    (status.schedule, status.trust as u8)
}

/// A rule from its tuple. Only the kind is checked here; the kernel validates the
/// amounts when the preset is built.
pub(super) fn rule_from_tuple(rule: &RuleTuple) -> PyResult<BrokerageRule> {
    let (kind, first, second) = rule;
    match kind.as_str() {
        RULE_ZERO => Ok(BrokerageRule::Zero),
        RULE_FLAT => Ok(BrokerageRule::Flat(*first)),
        // Python sends no cap as infinity, which the kernel reads as no cap; NaN and
        // negative caps reach the kernel's validation instead of becoming no cap.
        RULE_PERCENT => Ok(BrokerageRule::percent(*first, Some(*second))),
        RULE_RATE => Ok(BrokerageRule::Rate {
            rate: *first,
            cap: *second,
        }),
        other => Err(cost_err(CostError::InvalidBrokerage(format!(
            "unknown rule kind {other:?}: use zero, flat, percent or rate"
        )))),
    }
}

/// A rule as a tuple, always in the kernel's own terms: zero, flat or rate.
pub(super) fn rule_to_tuple(rule: &BrokerageRule) -> RuleTuple {
    match *rule {
        BrokerageRule::Zero => (RULE_ZERO.to_string(), 0.0, 0.0),
        BrokerageRule::Flat(amount) => (RULE_FLAT.to_string(), amount, 0.0),
        BrokerageRule::Rate { rate, cap } => (RULE_RATE.to_string(), rate, cap),
    }
}

fn invalid_schedule(id: &str, why: String) -> PyErr {
    cost_err(CostError::InvalidSchedule(format!("version {id:?}: {why}")))
}

fn row_from_tuple(id: &str, row: RowTuple) -> PyResult<(Segment, SegmentRow)> {
    let (code, rates, verified_through, tax_base, tax_unit, notes) = row;
    let segment = Segment::from_code(&code).map_err(cost_err)?;
    let (
        stt_buy_rate,
        stt_sell_rate,
        exchange_txn_rate,
        ipft_rate,
        sebi_rate,
        stamp_buy_rate,
        stt_exercise_rate,
    ) = rates;
    let calculator_tax_base = TaxBase::from_code(&tax_base).ok_or_else(|| {
        invalid_schedule(
            id,
            format!("calculator_tax_base {tax_base:?} must be sides or turnover"),
        )
    })?;
    let calculator_tax_unit = TaxUnit::from_code(&tax_unit).ok_or_else(|| {
        invalid_schedule(
            id,
            format!("calculator_tax_unit {tax_unit:?} must be rupee or paise"),
        )
    })?;
    let rates = SegmentRates {
        stt_buy_rate,
        stt_sell_rate,
        exchange_txn_rate,
        ipft_rate,
        sebi_rate,
        stamp_buy_rate,
        stt_exercise_rate,
        calculator_tax_base,
        calculator_tax_unit,
    };
    Ok((
        segment,
        SegmentRow {
            rates,
            verified_through,
            notes,
        },
    ))
}

/// A schedule version from its tuple. Codes are checked here; the kernel validates
/// the values and dates when the version joins a schedule book.
pub(super) fn schedule_from_tuple(schedule: ScheduleTuple) -> PyResult<CostSchedule> {
    let (
        id,
        valid_from,
        valid_to,
        gst_rate,
        gst_on_sebi_fee,
        gst_verified_through,
        source,
        notes,
        row_tuples,
    ) = schedule;
    let mut rows = std::array::from_fn(|_| None);
    for row_tuple in row_tuples {
        let (segment, row) = row_from_tuple(&id, row_tuple)?;
        rows[segment.index()] = Some(row);
    }
    Ok(CostSchedule {
        id,
        valid_from,
        valid_to,
        gst_rate,
        gst_on_sebi_fee,
        gst_verified_through,
        source,
        notes,
        rows,
    })
}

/// The inverse of `schedule_from_tuple`.
pub(super) fn schedule_to_tuple(schedule: &CostSchedule) -> ScheduleTuple {
    let rows = Segment::ALL
        .into_iter()
        .filter_map(|segment| schedule.row(segment).map(|row| (segment, row)))
        .map(|(segment, row)| {
            let r = &row.rates;
            let rates = (
                r.stt_buy_rate,
                r.stt_sell_rate,
                r.exchange_txn_rate,
                r.ipft_rate,
                r.sebi_rate,
                r.stamp_buy_rate,
                r.stt_exercise_rate,
            );
            (
                segment.code().to_string(),
                rates,
                row.verified_through,
                r.calculator_tax_base.code().to_string(),
                r.calculator_tax_unit.code().to_string(),
                row.notes.clone(),
            )
        })
        .collect();
    (
        schedule.id.clone(),
        schedule.valid_from,
        schedule.valid_to,
        schedule.gst_rate,
        schedule.gst_on_sebi_fee,
        schedule.gst_verified_through,
        schedule.source.clone(),
        schedule.notes.clone(),
        rows,
    )
}
