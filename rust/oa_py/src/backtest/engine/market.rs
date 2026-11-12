//! The market settings of a run from Python, and the dated market tables for Python.
//!
//! Config keys read here (set by `openalgo/backtest/_market.py`):
//! - `costs`: None (cost-free) or a native `CostModel` (`openalgo.backtest.costs`);
//! - `market`: absent, or one dict per instrument with `underlying`, `bse_group`,
//!   `slippage_class` (a code of the default slippage table) and `slippage` (None, or
//!   the instrument's own slippage as (kind, amount, min_ticks)).
//!
//! The tables cross as tuples: `slippage_defaults()` gives (row_id, class, valid_from,
//! valid_to, model, amount, min_ticks, source, status, notes) and `margin_defaults()`
//! (row_id, kind, product, underlying, valid_from, valid_to, model, percent, source,
//! status, notes), with dates as IST days since 1970-01-01 or None.

use oa_bt::engine::{CostConfig, KernelError};
use oa_bt::market::facts::{InstrumentMarket, MarketFacts};
use oa_bt::market::margin::{
    margin_defaults as margin_rows, MarginBasis, MarginError, MarginModel, MarginRequest,
};
use oa_bt::market::slippage::{
    slippage_defaults as slippage_rows, SlippageClass, SlippageError, SlippageSpec,
};
use oa_bt::market::table_rows::RowMeta;
use oa_bt::model::{InstrumentKind, Product, Side};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::super::calendar::{parse_exchange, PyDay};
use super::super::costs::PyCostModel;
use super::setup::{optional, slippage, unknown};
use super::{engine_error, NativeEngineInputError};

/// (row_id, valid_from, valid_to, source, status, notes) of a row.
type MetaParts = (
    String,
    Option<i32>,
    Option<i32>,
    String,
    &'static str,
    String,
);

type SlippageRowTuple = (
    String,
    &'static str,
    Option<i32>,
    Option<i32>,
    &'static str,
    f64,
    u32,
    String,
    &'static str,
    String,
);

type MarginRowTuple = (
    String,
    &'static str,
    &'static str,
    Option<String>,
    Option<i32>,
    Option<i32>,
    &'static str,
    f64,
    String,
    &'static str,
    String,
);

/// The run's charges: None, or a native cost model.
pub(super) fn costs(dict: &Bound<'_, PyDict>) -> PyResult<CostConfig> {
    match dict.get_item("costs")? {
        None => Ok(CostConfig::Off),
        Some(value) if value.is_none() => Ok(CostConfig::Off),
        Some(value) => {
            let model: PyRef<'_, PyCostModel> = value.extract().map_err(|_| {
                NativeEngineInputError::new_err(
                    "key \"costs\": pass None or a cost model from openalgo.backtest.costs",
                )
            })?;
            Ok(CostConfig::Model(model.inner.clone()))
        }
    }
}

fn instrument_market(dict: &Bound<'_, PyDict>) -> PyResult<InstrumentMarket> {
    let class: Option<String> = optional(dict, "slippage_class")?;
    let own: Option<(String, f64, u32)> = optional(dict, "slippage")?;
    Ok(InstrumentMarket {
        underlying: optional(dict, "underlying")?,
        bse_group: optional(dict, "bse_group")?,
        slippage_class: class
            .map(|code| {
                SlippageClass::from_code(&code).ok_or_else(|| {
                    unknown(
                        "slippage class",
                        &code,
                        "a class of the default slippage table",
                    )
                })
            })
            .transpose()?,
        slippage: own.map(slippage).transpose()?,
    })
}

/// The per-instrument market facts (empty when the key is absent).
pub(super) fn market_facts(dict: &Bound<'_, PyDict>) -> PyResult<MarketFacts> {
    let rows: Option<Vec<Bound<'_, PyDict>>> = optional(dict, "market")?;
    let instruments = rows
        .unwrap_or_default()
        .iter()
        .map(instrument_market)
        .collect::<PyResult<_>>()?;
    Ok(MarketFacts { instruments })
}

fn meta(row: &RowMeta) -> MetaParts {
    (
        row.id.clone(),
        row.valid_from,
        row.valid_to,
        row.source.clone(),
        row.status.code(),
        row.notes.clone(),
    )
}

/// (kind, amount, min_ticks) of a slippage form.
pub(super) fn spec_parts(spec: SlippageSpec) -> (&'static str, f64, u32) {
    match spec {
        SlippageSpec::Zero => ("zero", 0.0, 0),
        SlippageSpec::Ticks(ticks) => ("ticks", f64::from(ticks), 0),
        SlippageSpec::Bps { bps, min_ticks } => ("bps", bps, min_ticks),
        SlippageSpec::PercentOfPrice { percent, min_ticks } => ("percent", percent, min_ticks),
        SlippageSpec::SegmentDefault => ("segment_default", 0.0, 0),
    }
}

/// The built-in default slippage rows.
#[pyfunction]
fn slippage_defaults() -> PyResult<Vec<SlippageRowTuple>> {
    let rows = slippage_rows().map_err(|e: SlippageError| PyValueError::new_err(e.to_string()))?;
    Ok(rows
        .iter()
        .map(|row| {
            let (id, from, to, source, status, notes) = meta(&row.meta);
            let (model, amount, min_ticks) = spec_parts(row.spec);
            let class = row.class.code();
            (
                id, class, from, to, model, amount, min_ticks, source, status, notes,
            )
        })
        .collect())
}

/// The built-in default margin rows.
#[pyfunction]
fn margin_defaults() -> PyResult<Vec<MarginRowTuple>> {
    let rows = margin_rows().map_err(|e: MarginError| PyValueError::new_err(e.to_string()))?;
    Ok(rows
        .iter()
        .map(|row| {
            let (id, from, to, source, status, notes) = meta(&row.meta);
            let product = row.product.map_or("*", |p| p.code());
            let (model, percent) = match row.basis {
                MarginBasis::Leverage { least_percent } => ("leverage", least_percent),
                MarginBasis::Percent { percent } => ("percent", percent),
            };
            let kind = row.kind.code();
            let underlying = row.underlying.clone();
            (
                id, kind, product, underlying, from, to, model, percent, source, status, notes,
            )
        })
        .collect())
}

/// The id of the margin row that prices opening a long position of `kind` ("EQ",
/// "FUT", "OPT") in `product` on the IST day `day` (days since 1970-01-01) on
/// `underlying` (None when it has none), so a run can check before loading any data that
/// every instrument it trades has a margin. Raises what the engine raises at the fill:
/// NotImplementedError for a futures underlying with no row, NativeEngineInputError for
/// a kind and product with no rule.
#[pyfunction]
#[pyo3(signature = (exchange, kind, product, day, underlying))]
fn margin_check(
    exchange: &str,
    kind: &str,
    product: &str,
    day: PyDay,
    underlying: Option<String>,
) -> PyResult<String> {
    let request = MarginRequest {
        exchange: parse_exchange(exchange)?,
        kind: InstrumentKind::from_code(kind)
            .ok_or_else(|| unknown("instrument kind", kind, "EQ, FUT or OPT"))?,
        product: Product::from_code(product)
            .ok_or_else(|| unknown("product", product, "CNC, MIS or NRML"))?,
        side: Side::Buy,
        quantity_units: 1.0,
        price_rupees: 1.0,
        multiplier: 1.0,
        leverage: 1.0,
        day: day.0,
        underlying: underlying.as_deref(),
    };
    let quote = MarginModel
        .initial_margin(&request)
        .map_err(|error| engine_error(KernelError::Margin(error)))?;
    Ok(quote.row.meta.id.clone())
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(slippage_defaults, m)?)?;
    m.add_function(wrap_pyfunction!(margin_defaults, m)?)?;
    m.add_function(wrap_pyfunction!(margin_check, m)?)?;
    Ok(())
}
