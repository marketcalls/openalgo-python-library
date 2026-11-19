//! The run a `NativeEngine` holds (the parent module), from the kernel's build until
//! `finish` takes it.

use oa_bt::engine::{Engine, KernelError};
use pyo3::prelude::*;

use super::errors::{engine_error, NativeEngineError};
use super::NativeEngine;

/// What a call on a run that `finish` has taken raises.
const FINISHED: &str = "this run has already finished";

impl NativeEngine {
    /// Hold the run the kernel built, or raise the error it refused it with.
    pub(super) fn built(engine: Result<Engine, KernelError>) -> PyResult<NativeEngine> {
        let inner = Some(engine.map_err(engine_error)?);
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

    /// Take the run out, for `finish`; `NativeEngineError` when it was taken already.
    pub(super) fn take(&mut self) -> PyResult<Engine> {
        let engine = self.inner.take();
        engine.ok_or_else(|| NativeEngineError::new_err(FINISHED))
    }
}
