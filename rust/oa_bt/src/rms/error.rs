//! Errors of the risk-management layer.

use std::fmt;

/// What went wrong in risk management. A rule that fires is not an error: it is a
/// `RiskEvent` and an exit.
#[derive(Clone, Debug, PartialEq)]
pub enum RmsError {
    /// A rule configuration that cannot be used (05-risk-management 8, strict config).
    InvalidRule { rule: &'static str, reason: String },
    /// A price or quantity handed to a leg that is not a positive finite number (a
    /// kernel defect in a backtest; an unusable tick in tick mode).
    InvalidValue { what: &'static str, value: f64 },
    /// A part of risk management that is not built yet, and what the user can do
    /// instead.
    NotBuilt {
        feature: &'static str,
        instead: &'static str,
    },
}

impl fmt::Display for RmsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RmsError::InvalidRule { rule, reason } => write!(f, "{rule}: {reason}"),
            RmsError::InvalidValue { what, value } => {
                write!(f, "{what} must be a positive finite number, got {value}")
            }
            RmsError::NotBuilt { feature, instead } => {
                crate::model::write_not_supported(f, feature, instead)
            }
        }
    }
}

impl std::error::Error for RmsError {}
