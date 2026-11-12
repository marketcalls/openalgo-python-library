//! The kernel's one error type, wrapping the errors of every area.
//!
//! A market rule that refuses an order is never an error: it is a `Rejected` order
//! event. An error means the run cannot go on: bad configuration or data, a call made
//! at the wrong time, a kernel defect, or a part that is not built yet. The last kind
//! is always `NotBuilt` somewhere in the chain; `KernelError::not_built_feature` finds
//! it so the bindings can raise Python's `NotImplementedError`.

use std::fmt;

use crate::data::resample::ResampleError;
use crate::exec::{ExecError, OrderId};
use crate::market::calendar::CalendarError;
use crate::market::margin::MarginError;
use crate::market::slippage::SlippageError;
use crate::portfolio::PortfolioError;
use crate::rms::RmsError;
use crate::sizing::SizingError;

/// Why the kernel stopped.
#[derive(Clone, Debug, PartialEq)]
pub enum KernelError {
    /// A run setting that cannot be used.
    InvalidConfig { field: &'static str, reason: String },
    /// Bars that cannot be used.
    InvalidData { symbol: String, reason: String },
    /// A calendar error (unsupported day, grid too large).
    Calendar(CalendarError),
    /// The strategy bars could not be built from the base bars.
    Resample(ResampleError),
    /// The execution layer.
    Exec(ExecError),
    /// Positions and the account.
    Portfolio(PortfolioError),
    /// Risk management.
    Rms(RmsError),
    /// Position sizing.
    Sizing(SizingError),
    /// Margin.
    Margin(MarginError),
    /// Slippage.
    Slippage(SlippageError),
    /// An order call made while the engine is not paused at a decision point.
    NotAtDecisionPoint,
    /// `advance` called after the run finished.
    RunFinished,
    /// `finish` called before the run finished.
    RunNotFinished,
    /// An order id the run does not know.
    UnknownOrder(OrderId),
    /// A part of the engine that is not built yet.
    NotBuilt { feature: &'static str },
}

impl KernelError {
    /// The feature that is not built yet, when this error is (or wraps) a `NotBuilt`.
    pub fn not_built_feature(&self) -> Option<&'static str> {
        match self {
            KernelError::NotBuilt { feature }
            | KernelError::Exec(ExecError::NotBuilt { feature })
            | KernelError::Rms(RmsError::NotBuilt { feature })
            | KernelError::Sizing(SizingError::NotBuilt { feature })
            | KernelError::Margin(MarginError::NotBuilt { feature })
            | KernelError::Slippage(SlippageError::NotBuilt { feature }) => Some(feature),
            _ => None,
        }
    }

    /// True for an error in what the caller supplied (configuration, data, a request),
    /// as opposed to a call at the wrong time, a defect or an unbuilt part.
    pub fn is_input_error(&self) -> bool {
        matches!(
            self,
            KernelError::InvalidConfig { .. }
                | KernelError::InvalidData { .. }
                | KernelError::Calendar(_)
                | KernelError::Resample(_)
                | KernelError::Exec(ExecError::InvalidInstrument { .. })
                | KernelError::Exec(ExecError::InvalidPrice { .. })
                | KernelError::Exec(ExecError::InvalidTick { .. })
                | KernelError::Exec(ExecError::InvalidRequest { .. })
                | KernelError::Portfolio(PortfolioError::InvalidCapital { .. })
                | KernelError::Rms(RmsError::InvalidRule { .. })
                | KernelError::Sizing(SizingError::InvalidSizer { .. })
                | KernelError::Margin(MarginError::InvalidRequest { .. })
                | KernelError::Slippage(SlippageError::InvalidSpec { .. })
        )
    }
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelError::InvalidConfig { field, reason } => write!(f, "{field}: {reason}"),
            KernelError::InvalidData { symbol, reason } => write!(f, "bars of {symbol}: {reason}"),
            KernelError::Calendar(inner) => inner.fmt(f),
            KernelError::Resample(inner) => inner.fmt(f),
            KernelError::Exec(inner) => inner.fmt(f),
            KernelError::Portfolio(inner) => inner.fmt(f),
            KernelError::Rms(inner) => inner.fmt(f),
            KernelError::Sizing(inner) => inner.fmt(f),
            KernelError::Margin(inner) => inner.fmt(f),
            KernelError::Slippage(inner) => inner.fmt(f),
            KernelError::NotAtDecisionPoint => write!(
                f,
                "orders can only be placed, changed or cancelled while the engine is paused \
                 at a decision point (in on_bar, a timer or a callback)"
            ),
            KernelError::RunFinished => write!(f, "the run has finished; nothing to advance"),
            KernelError::RunNotFinished => {
                write!(f, "the run has not finished: advance it to the end first")
            }
            KernelError::UnknownOrder(id) => write!(f, "no order {id} in this run"),
            KernelError::NotBuilt { feature } => {
                write!(
                    f,
                    "{feature} is not implemented yet (Phase 1 work in progress)"
                )
            }
        }
    }
}

impl std::error::Error for KernelError {}

macro_rules! wrap {
    ($($source:ty => $variant:ident),* $(,)?) => {
        $(impl From<$source> for KernelError {
            fn from(error: $source) -> KernelError {
                KernelError::$variant(error)
            }
        })*
    };
}

wrap!(
    CalendarError => Calendar,
    ResampleError => Resample,
    ExecError => Exec,
    PortfolioError => Portfolio,
    RmsError => Rms,
    SizingError => Sizing,
    MarginError => Margin,
    SlippageError => Slippage,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wrapped_not_built_error_names_its_feature() {
        let error = KernelError::from(RmsError::NotBuilt {
            feature: "lock profit",
        });
        assert_eq!(error.not_built_feature(), Some("lock profit"));
        assert!(!error.is_input_error());
        let config = KernelError::InvalidConfig {
            field: "capital",
            reason: "must be positive".to_string(),
        };
        assert_eq!(config.not_built_feature(), None);
        assert!(config.is_input_error());
    }
}
