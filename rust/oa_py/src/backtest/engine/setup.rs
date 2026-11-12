//! The run setup from Python: the configuration dict, the instrument dicts and the bar
//! arrays, parsed into `oa_bt::engine::RunSetup`.
//!
//! The dict keys are fixed by `openalgo/backtest/_engine.py`; a missing key or a value
//! of the wrong kind is a `NativeEngineInputError` that names the key.

use numpy::PyReadonlyArray1;
use oa_bt::data::resample::CalendarUnit;
use oa_bt::engine::{
    BarColumns, FillTiming, IntradayOverrides, QuantityMode, RunConfig, StrategyInterval,
};
use oa_bt::exec::InstrumentSpec;
use oa_bt::market::calendar::BaseInterval;
use oa_bt::market::margin::MarginModel;
use oa_bt::market::slippage::SlippageSpec;
use oa_bt::model::InstrumentKind;
use oa_bt::rms::{AfterRiskExit, RiskBasis, StrategyRiskConfig};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::super::calendar::{parse_exchange, PyDay};
use super::NativeEngineInputError;

/// A required key of a dict, extracted as `T`.
pub(super) fn field<'py, T: FromPyObject<'py>>(
    dict: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<T> {
    let value = dict
        .get_item(key)?
        .ok_or_else(|| NativeEngineInputError::new_err(format!("missing key {key:?}")))?;
    value
        .extract()
        .map_err(|error| NativeEngineInputError::new_err(format!("key {key:?}: {error}")))
}

/// A key that may be absent or None.
pub(super) fn optional<'py, T: FromPyObject<'py>>(
    dict: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<Option<T>> {
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => value
            .extract()
            .map(Some)
            .map_err(|error| NativeEngineInputError::new_err(format!("key {key:?}: {error}"))),
        _ => Ok(None),
    }
}

pub(super) fn unknown(what: &str, value: &str, allowed: &str) -> PyErr {
    NativeEngineInputError::new_err(format!("unknown {what} {value:?}: use {allowed}"))
}

/// The base interval: nanoseconds of an intraday interval, or None for daily.
fn base_interval(nanos: Option<i64>) -> BaseInterval {
    nanos.map_or(BaseInterval::Daily, BaseInterval::Intraday)
}

/// The strategy interval: nanoseconds, or "D", "W", "M", "Q", "Y".
fn strategy_interval(value: &Bound<'_, PyAny>) -> PyResult<StrategyInterval> {
    if let Ok(nanos) = value.extract::<i64>() {
        return Ok(StrategyInterval::Intraday(nanos));
    }
    let code: String = value.extract()?;
    if code == "D" {
        return Ok(StrategyInterval::Daily);
    }
    CalendarUnit::from_code(&code)
        .map(StrategyInterval::Period)
        .ok_or_else(|| unknown("strategy interval", &code, "nanoseconds or D, W, M, Q, Y"))
}

/// Slippage as (kind, amount, min_ticks).
pub(super) fn slippage(spec: (String, f64, u32)) -> PyResult<SlippageSpec> {
    let (kind, amount, min_ticks) = spec;
    match kind.as_str() {
        "zero" => Ok(SlippageSpec::Zero),
        "ticks" if amount >= 0.0 && amount.fract() == 0.0 => Ok(SlippageSpec::Ticks(amount as u32)),
        "bps" => Ok(SlippageSpec::Bps {
            bps: amount,
            min_ticks,
        }),
        "percent" => Ok(SlippageSpec::PercentOfPrice {
            percent: amount,
            min_ticks,
        }),
        "segment_default" => Ok(SlippageSpec::SegmentDefault),
        _ => Err(unknown(
            "slippage",
            &kind,
            "zero, ticks (a whole number), bps, percent or segment_default",
        )),
    }
}

fn risk(dict: &Bound<'_, PyDict>) -> PyResult<StrategyRiskConfig> {
    let basis: String = field(dict, "risk_basis")?;
    let after: String = field(dict, "after_risk_exit")?;
    Ok(StrategyRiskConfig {
        max_daily_loss_rupees: optional(dict, "max_daily_loss")?,
        risk_basis: match basis.as_str() {
            "gross" => RiskBasis::Gross,
            "net" => RiskBasis::Net,
            _ => return Err(unknown("risk_basis", &basis, "gross or net")),
        },
        after_risk_exit: match after.as_str() {
            "lock_session" => AfterRiskExit::LockSession,
            "allow_reentry" => AfterRiskExit::AllowReentry,
            _ => {
                return Err(unknown(
                    "after_risk_exit",
                    &after,
                    "lock_session or allow_reentry",
                ))
            }
        },
        max_positions: optional(dict, "max_positions")?,
        drawdown_halt_percent: optional(dict, "drawdown_halt")?,
        cooldown_bars: optional(dict, "cooldown_bars")?,
        max_hold_bars: optional(dict, "max_hold_bars")?,
        round_levels: field(dict, "round_levels")?,
        strict_config: field(dict, "strict_config")?,
        ..StrategyRiskConfig::default()
    })
}

/// The run configuration.
pub(super) fn run_config(dict: &Bound<'_, PyDict>) -> PyResult<RunConfig> {
    let fill: String = field(dict, "fill")?;
    let mode: String = field(dict, "quantity_mode")?;
    let first_day: PyDay = field(dict, "first_day")?;
    let last_day: PyDay = field(dict, "last_day")?;
    let interval = dict
        .get_item("strategy_interval")?
        .ok_or_else(|| NativeEngineInputError::new_err("missing key \"strategy_interval\""))?;
    Ok(RunConfig {
        first_day: first_day.0,
        last_day: last_day.0,
        first_slice: optional(dict, "first_slice")?,
        last_slice: optional(dict, "last_slice")?,
        base: base_interval(optional(dict, "base_nanos")?),
        strategy_interval: strategy_interval(&interval)?,
        fill: match fill.as_str() {
            "next_open" => FillTiming::NextOpen,
            "close" => FillTiming::Close,
            _ => return Err(unknown("fill", &fill, "next_open or close")),
        },
        quantity_mode: match mode.as_str() {
            "fractional" => QuantityMode::Fractional,
            "venue" => QuantityMode::Venue,
            _ => return Err(unknown("quantity_mode", &mode, "fractional or venue")),
        },
        capital_rupees: field(dict, "capital")?,
        costs: super::market::costs(dict)?,
        slippage: slippage(field(dict, "slippage")?)?,
        margin: MarginModel,
        market: super::market::market_facts(dict)?,
        risk: risk(&field(dict, "risk")?)?,
        intraday: IntradayOverrides {
            entry_from_minute: optional(dict, "entry_from_minute")?,
            entry_until_minute: optional(dict, "entry_until_minute")?,
            square_off_minute: optional(dict, "square_off_minute")?,
        },
        deliver_events: field(dict, "deliver_events")?,
    })
}

/// One instrument from its dict.
pub(super) fn instrument(dict: &Bound<'_, PyDict>) -> PyResult<InstrumentSpec> {
    let exchange: String = field(dict, "exchange")?;
    let kind: String = field(dict, "kind")?;
    let last_trading_day: Option<PyDay> = optional(dict, "last_trading_day")?;
    Ok(InstrumentSpec {
        symbol: field(dict, "symbol")?,
        exchange: parse_exchange(&exchange)?,
        kind: InstrumentKind::from_code(&kind)
            .ok_or_else(|| unknown("instrument kind", &kind, "EQ, INDEX, FUT or OPT"))?,
        lot_size_units: field(dict, "lot_size")?,
        tick_size_rupees: field(dict, "tick_size")?,
        multiplier: field(dict, "multiplier")?,
        freeze_quantity_units: optional(dict, "freeze_qty")?,
        cost_segment: None,
        last_trading_day: last_trading_day.map(|day| day.0),
    })
}

/// One instrument's bars: (ts_open, open, high, low, close, volume).
#[derive(FromPyObject)]
pub(super) struct BarArrays<'py>(
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
);

impl BarArrays<'_> {
    /// Copy the arrays into kernel columns.
    pub(super) fn columns(&self) -> BarColumns {
        BarColumns {
            ts_open: self.0.as_array().to_vec(),
            open: self.1.as_array().to_vec(),
            high: self.2.as_array().to_vec(),
            low: self.3.as_array().to_vec(),
            close: self.4.as_array().to_vec(),
            volume: self.5.as_array().to_vec(),
        }
    }
}
