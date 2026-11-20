//! Bindings for `oa_bt::fund`: the fund of a portfolio run as `Fund`.
//!
//! A portfolio run builds one `Engine` per strategy (each as the backtest of that strategy
//! alone, before its first `advance`) and a `Fund` over them, in call order:
//!
//! ```text
//! fund = Fund(engines, names, {"pool": "shared", "account_capital": None,
//!                              "account_max_daily_loss": 15000.0})
//! fund.add_signals(index, spec)          # a condition strategy, decided in Rust
//! while (step := fund.advance()) is not None:
//!     index, pause = step                # the pause tuple of Engine.advance
//!     ...                                # strategy `index` decides on engines[index]
//! outputs = [engine.finish() for engine in engines]
//! account = fund.finish()
//! ```
//!
//! The fund holds the engines and lends them to the kernel's fund at each `advance`; a
//! strategy decides on its own `Engine` (submit, cancel, the queries) while the fund has
//! stopped at its pause, exactly as at a pause of `Engine.advance`. The engines are
//! advanced by the fund only (an `Engine.advance` of one of them makes the next
//! `Fund.advance` raise `NativeEngineError`), and finished once `advance` has returned
//! None: a fund lends every engine at each call.
//!
//! `config` has "pool" ("shared" or "separate", `POOLS`), "account_capital" (rupees, or
//! None for the sum of the capitals; in a separate pool None or that sum) and
//! "account_max_daily_loss" (rupees above 0, or None). `names` are the strategies' names,
//! one per engine, used in messages.
//!
//! `finish()` returns the account's records as a dict: "account_capital" (rupees);
//! "account", the row of every round as NumPy arrays ("ts" int64 UTC nanoseconds of the
//! round's close, "day" int32 IST session days since 1970-01-01, "equity_rupees",
//! "free_cash_rupees", "margin_rupees", "daily_loss_rupees" float64); "breaches", each
//! breach of the account's daily loss limit as (at UTC nanoseconds, day, limit rupees,
//! measure rupees); "refused", the orders the account refused or ended as a dict of
//! lists ("strategy" index, "order_id" in that strategy's engine, "slice", "slice_ts",
//! "step" of that engine's stamp, "reason" one of `REFUSALS`: "account_margin",
//! "account_loss_gate", "account_loss" for an order a breach cancelled); and "skipped",
//! per strategy the entries its Rust driver's sizer skipped as (decision instant,
//! reason), empty for a strategy decided in Python.
//!
//! Errors: bad settings or engines raise `NativeEngineInputError`; an engine's own error
//! raises what `Engine` raises for it, its message starting with the strategy's name; a
//! call at the wrong time (an engine advanced outside the fund, `advance` after the end,
//! `finish` before it) raises `NativeEngineError`.

// See `engine`: lints that fire inside PyO3 0.22 macro expansions, not in this code.
#![allow(unexpected_cfgs, clippy::useless_conversion)]

use numpy::IntoPyArray;
use oa_bt::engine::Engine;
use oa_bt::fund::{Fund, FundConfig, FundError, FundOutput, FundPause, Pool, Refusal};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use super::engine::{error_of_kind, field, optional, pause_tuple, signal_arrays, unknown};
use super::engine::{NativeEngine, NativeEngineError, NativeEngineInputError, PauseTuple};

/// What a call on a fund that `finish` has taken raises.
const FINISHED: &str = "this portfolio run has already finished";

/// The fund of a portfolio run (see `oa_bt::fund`).
#[pyclass(name = "Fund", module = "openalgo._oaindicators.backtest")]
pub struct NativeFund {
    engines: Vec<Py<NativeEngine>>,
    names: Vec<String>,
    /// `None` once `finish` has taken it.
    inner: Option<Fund>,
}

#[pymethods]
impl NativeFund {
    /// Build the fund of a portfolio run over `engines` (backtests before their first
    /// `advance`, in call order) named by `names`, with the account of `config`: "pool",
    /// "account_capital" and "account_max_daily_loss" (see the module docs).
    #[new]
    fn new(
        py: Python<'_>,
        engines: Vec<Py<NativeEngine>>,
        names: Vec<String>,
        config: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        if names.len() != engines.len() {
            let given = (engines.len(), names.len());
            let message = format!(
                "{} engines and {} names: give one name each",
                given.0, given.1
            );
            return Err(NativeEngineInputError::new_err(message));
        }
        let shared = |(i, engine): (usize, &Py<NativeEngine>)| {
            engines[..i].iter().any(|earlier| earlier.is(engine))
        };
        if engines.iter().enumerate().any(shared) {
            return Err(NativeEngineInputError::new_err(
                "an engine is given twice: each strategy has an engine of its own",
            ));
        }
        let config = fund_config(config)?;
        let guards = borrowed(py, &engines)?;
        let lent = guards
            .iter()
            .map(|g| g.engine())
            .collect::<PyResult<Vec<_>>>()?;
        let fund = Fund::new(config, &lent).map_err(|error| fund_error(&names, error))?;
        drop(guards);
        Ok(NativeFund {
            engines,
            names,
            inner: Some(fund),
        })
    }

    /// Decide strategy `index` in Rust from its condition spec dict (as `Engine.run_signals`
    /// reads it), with no Python call per bar; before the first `advance`.
    fn add_signals(
        &mut self,
        py: Python<'_>,
        index: usize,
        spec: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let arrays = signal_arrays(spec)?;
        let engine = self.engines.get(index).ok_or_else(|| {
            NativeEngineInputError::new_err(format!("the fund runs no strategy {index}"))
        })?;
        let engine = engine.bind(py).try_borrow()?;
        let fund = self.inner.as_mut().ok_or_else(finished)?;
        let set = fund.set_signals(index, arrays, engine.engine()?);
        set.map_err(|error| fund_error(&self.names, error))
    }

    /// Run to the next pause a strategy decides in Python: (strategy index, the pause
    /// tuple as `Engine.advance` returns it), or None once every strategy's run has
    /// finished. A "finished" pause ends that strategy's run (its `on_stop`).
    fn advance<'py>(&mut self, py: Python<'py>) -> PyResult<Option<(usize, PauseTuple<'py>)>> {
        let fund = self.inner.as_mut().ok_or_else(finished)?;
        let mut guards = borrowed_mut(py, &self.engines)?;
        let lent = guards.iter_mut().map(|g| g.engine_mut());
        let mut lent = lent.collect::<PyResult<Vec<&mut Engine>>>()?;
        let step = fund
            .advance(&mut lent)
            .map_err(|error| fund_error(&self.names, error))?;
        drop(guards);
        match step {
            FundPause::Finished => Ok(None),
            FundPause::Strategy { strategy, pause } => {
                Ok(Some((strategy, pause_tuple(py, &pause)?)))
            }
        }
    }

    /// The account's records, once `advance` has returned None (see the module docs). The
    /// fund cannot be used after.
    fn finish<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let fund = self.inner.as_ref().ok_or_else(finished)?;
        if !fund.is_finished() {
            return Err(fund_error(&self.names, FundError::NotFinished));
        }
        let fund = self.inner.take().ok_or_else(finished)?;
        let output = fund
            .finish()
            .map_err(|error| fund_error(&self.names, error))?;
        output_dict(py, &output)
    }
}

/// `NativeEngineError` for a fund that `finish` has taken.
fn finished() -> PyErr {
    NativeEngineError::new_err(FINISHED)
}

/// Every engine, borrowed to read.
fn borrowed<'py>(
    py: Python<'py>,
    engines: &[Py<NativeEngine>],
) -> PyResult<Vec<PyRef<'py, NativeEngine>>> {
    engines
        .iter()
        .map(|engine| Ok(engine.bind(py).try_borrow()?))
        .collect()
}

/// Every engine, borrowed to change.
fn borrowed_mut<'py>(
    py: Python<'py>,
    engines: &[Py<NativeEngine>],
) -> PyResult<Vec<PyRefMut<'py, NativeEngine>>> {
    engines
        .iter()
        .map(|engine| Ok(engine.bind(py).try_borrow_mut()?))
        .collect()
}

/// The account of a config dict (module docs).
fn fund_config(dict: &Bound<'_, PyDict>) -> PyResult<FundConfig> {
    let pool: String = field(dict, "pool")?;
    let codes: Vec<&str> = Pool::ALL.iter().map(|pool| pool.code()).collect();
    let pool = Pool::from_code(&pool)
        .ok_or_else(|| unknown("pool", &pool, &format!("one of {}", codes.join(", "))))?;
    Ok(FundConfig {
        pool,
        account_capital_rupees: optional(dict, "account_capital")?,
        account_max_daily_loss_rupees: optional(dict, "account_max_daily_loss")?,
    })
}

/// A fund's error as the Python exception of its kind (module docs), naming the strategy.
fn fund_error(names: &[String], error: FundError) -> PyErr {
    let name = |strategy: usize| names.get(strategy).map_or("a strategy", String::as_str);
    match error {
        FundError::Engine { strategy, error } => {
            let message = format!("{}: {error}", name(strategy));
            error_of_kind(&error, message)
        }
        FundError::OutOfStep { strategy } => NativeEngineError::new_err(format!(
            "{}: its engine was advanced outside the fund: the engines of a portfolio run \
             are advanced by the fund only",
            name(strategy)
        )),
        FundError::InvalidConfig { .. } => NativeEngineInputError::new_err(error.to_string()),
        FundError::Finished | FundError::NotFinished => {
            NativeEngineError::new_err(error.to_string())
        }
    }
}

/// The account's records as a dict (module docs).
fn output_dict<'py>(py: Python<'py>, output: &FundOutput) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("account_capital", output.account_capital_rupees)?;
    let rows = &output.account;
    let account = PyDict::new_bound(py);
    account.set_item("ts", rows.ts.clone().into_pyarray_bound(py))?;
    account.set_item("day", rows.day.clone().into_pyarray_bound(py))?;
    let columns = [
        ("equity_rupees", &rows.equity_rupees),
        ("free_cash_rupees", &rows.free_cash_rupees),
        ("margin_rupees", &rows.margin_rupees),
        ("daily_loss_rupees", &rows.daily_loss_rupees),
    ];
    for (key, column) in columns {
        account.set_item(key, column.clone().into_pyarray_bound(py))?;
    }
    dict.set_item("account", account)?;
    let breaches: Vec<(i64, i32, f64, f64)> = output
        .breaches
        .iter()
        .map(|b| (b.at, b.day, b.limit_rupees, b.measure_rupees))
        .collect();
    dict.set_item("breaches", breaches)?;
    dict.set_item("refused", refused(py, output)?)?;
    let skipped: Vec<Vec<(i64, &str)>> = output
        .skipped
        .iter()
        .map(|entries| {
            entries
                .iter()
                .map(|e| (e.decided_at, e.reason.code()))
                .collect()
        })
        .collect();
    dict.set_item("skipped", skipped)?;
    Ok(dict)
}

/// The refused orders as a dict of lists (module docs).
fn refused<'py>(py: Python<'py>, output: &FundOutput) -> PyResult<Bound<'py, PyDict>> {
    let rows = &output.refused;
    let dict = PyDict::new_bound(py);
    dict.set_item(
        "strategy",
        rows.iter().map(|r| r.strategy).collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "order_id",
        rows.iter().map(|r| r.order.0).collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "slice",
        rows.iter().map(|r| r.stamp.slice).collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "slice_ts",
        rows.iter().map(|r| r.stamp.slice_ts).collect::<Vec<_>>(),
    )?;
    let steps: Vec<u8> = rows.iter().map(|r| r.stamp.step.number()).collect();
    dict.set_item("step", steps)?;
    let reasons: Vec<&str> = rows.iter().map(|r| r.refusal.code()).collect();
    dict.set_item("reason", reasons)?;
    Ok(dict)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeFund>()?;
    let py = m.py();
    m.add("POOLS", PyTuple::new_bound(py, Pool::ALL.map(Pool::code)))?;
    m.add(
        "REFUSALS",
        PyTuple::new_bound(py, Refusal::ALL.map(Refusal::code)),
    )
}
