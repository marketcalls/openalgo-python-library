//! The run a `NativeEngine` holds (the parent module), from the kernel's build until
//! `finish` takes it.

use oa_bt::engine::{Engine, KernelError, RunSetup};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::super::calendar::NativeCalendar;
use super::errors::{engine_error, NativeEngineError};
use super::setup::{run_setup, BarArrays};
use super::NativeEngine;

/// What a call on a run that `finish` has taken raises.
const FINISHED: &str = "this run has already finished";

impl NativeEngine {
    /// Build a run from the binding's inputs (`setup::run_setup`) with `build`
    /// (`Engine::new` for a backtest, `Engine::new_session` for a live session), or raise
    /// the error the inputs or the kernel refused it with.
    pub(super) fn from_inputs(
        calendar: &NativeCalendar,
        config: &Bound<'_, PyDict>,
        instruments: &[Bound<'_, PyDict>],
        bars: &[BarArrays<'_>],
        build: fn(RunSetup) -> Result<Engine, KernelError>,
    ) -> PyResult<NativeEngine> {
        let setup = run_setup(calendar, config, instruments, bars)?;
        let inner = Some(build(setup).map_err(engine_error)?);
        Ok(NativeEngine { inner })
    }

    /// The run; `NativeEngineError` once `finish` has taken it.
    pub(super) fn engine(&self) -> PyResult<&Engine> {
        let engine = self.inner.as_ref();
        engine.ok_or_else(|| NativeEngineError::new_err(FINISHED))
    }

    /// The run, to change; `NativeEngineError` once `finish` has taken it.
    pub(super) fn engine_mut(&mut self) -> PyResult<&mut Engine> {
        let engine = self.inner.as_mut();
        engine.ok_or_else(|| NativeEngineError::new_err(FINISHED))
    }

    /// Take the finished run out, for `finish`: `NativeEngineError` before the run has
    /// finished (it stays), and when it was taken already.
    pub(super) fn take_finished(&mut self) -> PyResult<Engine> {
        if !self.engine()?.is_finished() {
            return Err(engine_error(KernelError::RunNotFinished));
        }
        let engine = self.inner.take();
        engine.ok_or_else(|| NativeEngineError::new_err(FINISHED))
    }
}
