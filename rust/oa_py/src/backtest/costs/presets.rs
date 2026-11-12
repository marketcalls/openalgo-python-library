//! The brokerage preset as a Python object. It wraps the kernel's `Brokerage`, so a
//! preset can only exist in a validated state on either side of the boundary.

use pyo3::prelude::*;

use oa_bt::market::costs::{Brokerage, BrokerageClass};

use super::convert::{cost_err, rule_from_tuple, rule_to_tuple, RuleTuple};

/// A validated brokerage preset: one rule per kind of trade plus the DP fee.
#[pyclass(
    name = "CostBrokerage",
    module = "openalgo._oaindicators.backtest",
    frozen
)]
#[derive(Clone)]
pub(super) struct PyBrokerage {
    pub(super) inner: Brokerage,
}

fn wrap(result: Result<Brokerage, oa_bt::market::costs::CostError>) -> PyResult<PyBrokerage> {
    result.map(|inner| PyBrokerage { inner }).map_err(cost_err)
}

#[pymethods]
impl PyBrokerage {
    /// A built-in preset by name: "zerodha" or "zero_brokerage".
    #[staticmethod]
    fn named(name: &str) -> PyResult<PyBrokerage> {
        wrap(Brokerage::from_name(name))
    }

    /// The same flat amount in rupees per executed order in every segment.
    #[staticmethod]
    fn flat(amount: f64) -> PyResult<PyBrokerage> {
        wrap(Brokerage::flat(amount))
    }

    /// A percentage of order value in every segment, in percent units, at most `cap`
    /// rupees per executed order.
    #[staticmethod]
    #[pyo3(signature = (percent, cap=None))]
    fn percent(percent: f64, cap: Option<f64>) -> PyResult<PyBrokerage> {
        wrap(Brokerage::percent(percent, cap))
    }

    /// One rule tuple per kind of trade.
    #[staticmethod]
    fn custom(
        delivery: RuleTuple,
        intraday: RuleTuple,
        futures: RuleTuple,
        options: RuleTuple,
    ) -> PyResult<PyBrokerage> {
        wrap(Brokerage::custom(
            rule_from_tuple(&delivery)?,
            rule_from_tuple(&intraday)?,
            rule_from_tuple(&futures)?,
            rule_from_tuple(&options)?,
        ))
    }

    /// The same preset with the caller's DP fee, in rupees before GST.
    fn with_dp_fee(&self, dp_fee: f64) -> PyResult<PyBrokerage> {
        wrap(self.inner.with_dp_fee(dp_fee))
    }

    /// The four rules in the order delivery, intraday, futures, options, each as
    /// ("zero", 0, 0), ("flat", rupees, 0) or ("rate", fraction, cap).
    fn rules(&self) -> Vec<RuleTuple> {
        let classes = [
            BrokerageClass::Delivery,
            BrokerageClass::Intraday,
            BrokerageClass::Futures,
            BrokerageClass::Options,
        ];
        classes
            .iter()
            .map(|class| rule_to_tuple(self.inner.rule(*class)))
            .collect()
    }

    /// DP fee in rupees per scrip per sell day, before GST.
    #[getter]
    fn dp_fee(&self) -> f64 {
        self.inner.dp_fee()
    }

    /// Where the DP fee comes from: zerodha_preset, zerodha_stand_in or supplied.
    #[getter]
    fn dp_fee_source(&self) -> &'static str {
        self.inner.dp_fee_source().code()
    }

    fn __eq__(&self, other: &PyBrokerage) -> bool {
        self.inner == other.inner
    }
}
