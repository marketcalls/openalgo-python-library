//! Bindings for `oa_bt::market::costs`.
//!
//! A thin shim: all cost arithmetic happens in `oa_bt`, and all parsing of user
//! input happens in the Python wrapper (`openalgo/backtest/costs.py`). What crosses
//! the boundary is plain: numbers, strings, the typed tuples of `convert`, the
//! preset object of `presets`, and contiguous NumPy arrays.
//!
//! Conventions: trade dates are int32 IST days since 1970-01-01; a side is uint8
//! (0 buy, 1 sell, the kernel's `Side`); in arrays a segment is its uint8 id and in
//! scalar calls its code. Every result carries which schedule version priced it and
//! a trust id (`cost_trust_codes`). A `CostError` becomes a `ValueError`.

// PyO3 0.22 expands a `#[pyfunction]` or `#[pymethods]` item that returns `PyResult`
// into an identity `PyErr` conversion, which this lint reports on every such item.
#![allow(clippy::useless_conversion)]

mod convert;
mod presets;
mod tables;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use oa_bt::market::costs::{
    contract_notes, CostColumns, CostError, CostModel, NoteLine, Rounding, ScheduleBook, Segment,
    COMPONENT_NAMES,
};
use oa_bt::model::Side;

use convert::{
    cost_err, cost_tuple, row_err, schedule_from_tuple, schedule_to_tuple, status_tuple, CostTuple,
    ScheduleTuple, StatusTuple,
};
use presets::PyBrokerage;

type FloatArray = Py<PyArray1<f64>>;

/// A costed batch: seven amount arrays, then the schedule position and trust id of
/// each row.
type BatchColumns = (Vec<FloatArray>, Py<PyArray1<u16>>, Py<PyArray1<u8>>);

/// Contract notes as columns: day, group id, fills, six charged component arrays,
/// charged total, accrued total, adjustment (charged minus accrued).
type NoteColumns = (
    Py<PyArray1<i32>>,
    Py<PyArray1<u8>>,
    Py<PyArray1<i64>>,
    Vec<FloatArray>,
    FloatArray,
    FloatArray,
    FloatArray,
);

/// The DP charge: fee, gst, total, fee source code, schedule position and trust id.
type DpChargeTuple = (f64, f64, f64, &'static str, usize, u8);

fn segment_from_code(code: &str) -> PyResult<Segment> {
    Segment::from_code(code).map_err(cost_err)
}

fn side_from_id(id: u8) -> PyResult<Side> {
    Side::from_u8(id).ok_or_else(|| cost_err(CostError::UnknownSide(id)))
}

fn float_arrays(py: Python<'_>, columns: Vec<Vec<f64>>) -> Vec<FloatArray> {
    columns
        .into_iter()
        .map(|column| column.into_pyarray_bound(py).unbind())
        .collect()
}

/// A cost model: dated schedules, a brokerage preset and a rounding mode.
#[pyclass(name = "CostModel", module = "openalgo._oaindicators.backtest", frozen)]
pub(in crate::backtest) struct PyCostModel {
    pub(in crate::backtest) inner: CostModel,
}

#[pymethods]
impl PyCostModel {
    /// With `include_builtin` the `schedules` are merged with the built-in versions,
    /// otherwise they replace them.
    #[new]
    fn new(
        brokerage: &PyBrokerage,
        rounding: &str,
        schedules: Vec<ScheduleTuple>,
        include_builtin: bool,
        backfill: bool,
    ) -> PyResult<Self> {
        let rounding = Rounding::from_code(rounding).map_err(cost_err)?;
        let user = schedules
            .into_iter()
            .map(schedule_from_tuple)
            .collect::<PyResult<Vec<_>>>()?;
        let book = match (include_builtin, user.is_empty()) {
            (true, true) => ScheduleBook::builtin(),
            (true, false) => ScheduleBook::builtin_with(user),
            (false, _) => ScheduleBook::from_versions(user),
        };
        let book = book.map_err(cost_err)?.with_backfill(backfill);
        Ok(PyCostModel {
            inner: CostModel::new(book, brokerage.inner, rounding),
        })
    }

    /// Cost in rupees of one executed order of `quantity` price units at `price` on
    /// IST day `day`, with the status of the rates used.
    fn fill(
        &self,
        segment: &str,
        side: u8,
        price: f64,
        quantity: f64,
        day: i32,
    ) -> PyResult<(CostTuple, StatusTuple)> {
        let segment = segment_from_code(segment)?;
        let cost = self
            .inner
            .fill_cost(segment, side_from_id(side)?, price, quantity, day)
            .map_err(cost_err)?;
        let status = self.inner.status(segment, day).map_err(cost_err)?;
        Ok((
            cost_tuple(&cost, self.inner.reported_total(&cost)),
            status_tuple(status),
        ))
    }

    /// Cost in rupees of buying on `buy_day` and selling on `sell_day`, with the
    /// status of the rates used for each leg.
    fn round_trip(
        &self,
        segment: &str,
        buy_price: f64,
        sell_price: f64,
        quantity: f64,
        buy_day: i32,
        sell_day: i32,
    ) -> PyResult<(CostTuple, StatusTuple, StatusTuple)> {
        let segment = segment_from_code(segment)?;
        let cost = self
            .inner
            .round_trip_cost(segment, buy_price, sell_price, quantity, buy_day, sell_day)
            .map_err(cost_err)?;
        let bought = self.inner.status(segment, buy_day).map_err(cost_err)?;
        let sold = self.inner.status(segment, sell_day).map_err(cost_err)?;
        let total = self.inner.reported_total(&cost);
        Ok((
            cost_tuple(&cost, total),
            status_tuple(bought),
            status_tuple(sold),
        ))
    }

    /// Cost of many executed orders, one per row.
    fn fill_batch<'py>(
        &self,
        py: Python<'py>,
        segments: PyReadonlyArray1<'py, u8>,
        sides: PyReadonlyArray1<'py, u8>,
        prices: PyReadonlyArray1<'py, f64>,
        quantities: PyReadonlyArray1<'py, f64>,
        days: PyReadonlyArray1<'py, i32>,
    ) -> PyResult<BatchColumns> {
        let segments = segments.as_slice()?;
        let rows = segments.len();
        let mut amounts: [Vec<f64>; 7] = std::array::from_fn(|_| vec![0.0; rows]);
        let mut schedule = vec![0u16; rows];
        let mut trust = vec![0u8; rows];
        let [brokerage, stt, exchange_txn, sebi_fee, stamp_duty, gst, total] = &mut amounts;
        let mut out = CostColumns {
            brokerage,
            stt,
            exchange_txn,
            sebi_fee,
            stamp_duty,
            gst,
            total,
            schedule: &mut schedule,
            trust: &mut trust,
        };
        self.inner
            .fill_cost_batch(
                segments,
                sides.as_slice()?,
                prices.as_slice()?,
                quantities.as_slice()?,
                days.as_slice()?,
                &mut out,
            )
            .map_err(row_err)?;
        Ok((
            float_arrays(py, amounts.into()),
            schedule.into_pyarray_bound(py).unbind(),
            trust.into_pyarray_bound(py).unbind(),
        ))
    }

    /// DP charge in rupees for one scrip sold from holdings on `day`.
    fn dp_charge(&self, day: i32) -> PyResult<DpChargeTuple> {
        let charge = self.inner.dp_charge(day).map_err(cost_err)?;
        let (schedule, trust) = status_tuple(self.inner.gst_status(day).map_err(cost_err)?);
        Ok((
            charge.fee,
            charge.gst,
            charge.total(),
            charge.fee_source.code(),
            schedule,
            trust,
        ))
    }

    /// STT in rupees on an exercised option, with the status of the rate used.
    fn exercise_stt(
        &self,
        segment: &str,
        intrinsic: f64,
        quantity: f64,
        day: i32,
    ) -> PyResult<(f64, StatusTuple)> {
        let segment = segment_from_code(segment)?;
        let stt = self
            .inner
            .exercise_stt(segment, intrinsic, quantity, day)
            .map_err(cost_err)?;
        let status = self.inner.status(segment, day).map_err(cost_err)?;
        Ok((stt, status_tuple(status)))
    }

    /// The schedule versions of this model, oldest first.
    fn schedules(&self) -> Vec<ScheduleTuple> {
        self.inner
            .schedules()
            .versions()
            .iter()
            .map(schedule_to_tuple)
            .collect()
    }

    #[getter]
    fn rounding(&self) -> &'static str {
        self.inner.rounding().code()
    }

    #[getter]
    fn backfill(&self) -> bool {
        self.inner.schedules().backfill()
    }
}

/// Round per-fill accruals into contract notes, one per trade day and note group.
///
/// `components` is six float64 arrays of unrounded rupees in component order, one
/// value per fill, alongside the fills' IST `days` and segment ids.
#[pyfunction]
fn cost_contract_notes<'py>(
    py: Python<'py>,
    days: PyReadonlyArray1<'py, i32>,
    segments: PyReadonlyArray1<'py, u8>,
    components: Vec<PyReadonlyArray1<'py, f64>>,
) -> PyResult<NoteColumns> {
    let slices = components
        .iter()
        .map(|column| column.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let slices: [&[f64]; 6] = slices.try_into().map_err(|_| {
        let (expected, got) = (COMPONENT_NAMES.len(), components.len());
        cost_err(CostError::LengthMismatch {
            name: "components",
            expected,
            got,
        })
    })?;
    let notes = contract_notes(days.as_slice()?, segments.as_slice()?, slices).map_err(row_err)?;
    let note_days: Vec<i32> = notes.iter().map(|note| note.day).collect();
    let groups: Vec<u8> = notes.iter().map(|note| note.group as u8).collect();
    let fills: Vec<i64> = notes.iter().map(|note| i64::from(note.fills)).collect();
    let column = |pick: fn(&NoteLine) -> f64| -> FloatArray {
        notes
            .iter()
            .map(pick)
            .collect::<Vec<f64>>()
            .into_pyarray_bound(py)
            .unbind()
    };
    let charged = (0..COMPONENT_NAMES.len())
        .map(|component| {
            notes
                .iter()
                .map(|note| note.charged.components()[component])
                .collect()
        })
        .collect();
    Ok((
        note_days.into_pyarray_bound(py).unbind(),
        groups.into_pyarray_bound(py).unbind(),
        fills.into_pyarray_bound(py).unbind(),
        float_arrays(py, charged),
        column(|note| note.charged_total),
        column(|note| note.accrued.total()),
        column(|note| note.adjustment()),
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCostModel>()?;
    m.add_class::<PyBrokerage>()?;
    m.add_function(wrap_pyfunction!(cost_contract_notes, m)?)?;
    tables::register(m)
}
