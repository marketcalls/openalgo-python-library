//! The read-only tables and lookups the Python wrapper reads from the kernel: the
//! segment table, component and code lists, the supported date range, the built-in
//! schedule versions and the resolution of an instrument to its segment.

use pyo3::prelude::*;

use oa_bt::market::costs::{
    builtin_schedules, CostError, NameKind, NoteGroup, Rounding, Segment, Trust, COMPONENT_NAMES,
    CONTRACT_NOTE_RULE,
};
use oa_bt::model::{Exchange, InstrumentKind, Product, MAX_DAY, MIN_DAY};

use super::convert::{cost_err, schedule_to_tuple, ScheduleTuple};

/// A segment table row: id, code, exchange, brokerage class, note group,
/// description.
type SegmentRow = (
    u8,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
);

fn unknown(kind: NameKind, name: &str) -> PyErr {
    cost_err(CostError::UnknownName {
        kind,
        name: name.to_string(),
    })
}

/// The segment table, in id order.
#[pyfunction]
fn cost_segments() -> Vec<SegmentRow> {
    Segment::ALL
        .into_iter()
        .map(|segment| {
            (
                segment as u8,
                segment.code(),
                segment.exchange().code(),
                segment.brokerage_class().code(),
                segment.note_group().code(),
                segment.description(),
            )
        })
        .collect()
}

/// Contract-note group codes, indexed by group id.
#[pyfunction]
fn cost_note_groups() -> Vec<&'static str> {
    NoteGroup::ALL.iter().map(|group| group.code()).collect()
}

/// Names of the six cost components, in output order.
#[pyfunction]
fn cost_components() -> Vec<&'static str> {
    COMPONENT_NAMES.to_vec()
}

/// Trust codes, indexed by trust id: verified, unverified, past_verification,
/// backfilled.
#[pyfunction]
fn cost_trust_codes() -> Vec<&'static str> {
    Trust::ALL.iter().map(|trust| trust.code()).collect()
}

/// Rounding mode codes, the default first.
#[pyfunction]
fn cost_rounding_codes() -> Vec<&'static str> {
    Rounding::ALL.iter().map(|mode| mode.code()).collect()
}

/// The first and last IST day number the engine supports.
#[pyfunction]
fn cost_day_range() -> (i32, i32) {
    (MIN_DAY, MAX_DAY)
}

/// The contract-note rounding rule the kernel applies, for result metadata.
#[pyfunction]
fn cost_contract_note_rule() -> &'static str {
    CONTRACT_NOTE_RULE
}

/// The id of a segment code. Errors for an unknown or unsupported code.
#[pyfunction]
fn cost_segment_id(code: &str) -> PyResult<u8> {
    Segment::from_code(code)
        .map(|segment| segment as u8)
        .map_err(cost_err)
}

/// Resolve an OpenAlgo exchange, instrument type (EQ, FUT, OPT, CE, PE) and product
/// (CNC, MIS, NRML) to a segment code.
#[pyfunction]
#[pyo3(signature = (exchange, instrument, product, bse_group=None, underlying=None))]
fn cost_resolve_segment(
    exchange: &str,
    instrument: &str,
    product: &str,
    bse_group: Option<&str>,
    underlying: Option<&str>,
) -> PyResult<&'static str> {
    let exchange_code =
        Exchange::from_code(exchange).ok_or_else(|| unknown(NameKind::Exchange, exchange))?;
    let product_code =
        Product::from_code(product).ok_or_else(|| unknown(NameKind::Product, product))?;
    let kind = InstrumentKind::from_code(instrument)
        .ok_or_else(|| unknown(NameKind::Instrument, instrument))?;
    Segment::resolve(exchange_code, kind, product_code, bse_group, underlying)
        .map(Segment::code)
        .map_err(cost_err)
}

/// The schedule versions compiled into the library, oldest first.
#[pyfunction]
fn cost_builtin_schedules() -> PyResult<Vec<ScheduleTuple>> {
    let versions = builtin_schedules().map_err(cost_err)?;
    Ok(versions.iter().map(schedule_to_tuple).collect())
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cost_segments, m)?)?;
    m.add_function(wrap_pyfunction!(cost_note_groups, m)?)?;
    m.add_function(wrap_pyfunction!(cost_components, m)?)?;
    m.add_function(wrap_pyfunction!(cost_trust_codes, m)?)?;
    m.add_function(wrap_pyfunction!(cost_rounding_codes, m)?)?;
    m.add_function(wrap_pyfunction!(cost_day_range, m)?)?;
    m.add_function(wrap_pyfunction!(cost_contract_note_rule, m)?)?;
    m.add_function(wrap_pyfunction!(cost_segment_id, m)?)?;
    m.add_function(wrap_pyfunction!(cost_resolve_segment, m)?)?;
    m.add_function(wrap_pyfunction!(cost_builtin_schedules, m)?)?;
    Ok(())
}
