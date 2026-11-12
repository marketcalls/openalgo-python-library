//! The counts of an intraday check as a Python dict.

use numpy::IntoPyArray;
use oa_bt::market::calendar as cal;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// A dict keyed by field name from the listed fields of a struct, so a key can never
/// drift from the field it reports.
macro_rules! dict_of_fields {
    ($py:expr, $source:expr, [$($field:ident),+ $(,)?]) => {{
        let dict = PyDict::new_bound($py);
        $(dict.set_item(stringify!($field), $source.$field)?;)+
        dict
    }};
}

/// Every count of `check` under its field name, and the findings as five more keys:
///
/// - `offset`: the direction code ("late_5h30" or "early_5h30") when the series is
///   shifted, else None;
/// - `offset_check`: "aligned", "shifted" or "inconclusive";
/// - `offset_direction`: the direction that fits best when not aligned, else None;
/// - `offset_reason`: the outcome in words, empty when aligned;
/// - `finding`: what is wrong with the timestamps, for choosing advice: "late_5h30",
///   "early_5h30", "close_labelled" or None. The two kinds never coincide.
///
/// `placement` is a uint8 array with one code per bar (see `cal::Placement`). The
/// Python wrapper documents every key.
pub(super) fn check_report<'py>(
    py: Python<'py>,
    check: cal::IntradayCheck,
) -> PyResult<Bound<'py, PyDict>> {
    let report = dict_of_fields!(
        py,
        check,
        [
            bars,
            days,
            session_days,
            closed_days,
            first_closed_day,
            closed_day_bars,
            days_starting_at_open,
            days_starting_late,
            first_late_day,
            days_without_session_bars,
            days_with_bar_at_close,
            first_bar_at_close,
            pre_open_bars,
            post_close_bars,
            off_lattice_bars,
            first_off_lattice,
            bars_in_session,
            missing_bars,
            missing_days,
            duplicate_bars,
            template_only_days,
            unverified_hours_days,
            close_labelled,
        ]
    );
    let shifted = match check.offset {
        cal::OffsetCheck::Shifted(evidence) => Some(evidence.direction.code()),
        _ => None,
    };
    let direction = check
        .offset
        .evidence()
        .map(|evidence| evidence.direction.code());
    report.set_item("offset", shifted)?;
    report.set_item("offset_check", check.offset.code())?;
    report.set_item("offset_direction", direction)?;
    report.set_item("offset_reason", check.offset.reason())?;
    let finding = shifted.or(check.close_labelled.then_some("close_labelled"));
    report.set_item("finding", finding)?;
    let placement: Vec<u8> = check
        .placement
        .iter()
        .map(|&placement| placement as u8)
        .collect();
    report.set_item("placement", placement.into_pyarray_bound(py))?;
    Ok(report)
}
