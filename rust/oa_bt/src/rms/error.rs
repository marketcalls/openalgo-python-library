//! Errors of the risk-management layer.

use std::fmt;

/// What went wrong in risk management. A rule that fires is not an error: it is a
/// `RiskEvent` and an exit.
#[derive(Clone, Debug, PartialEq)]
pub enum RmsError {
    /// A rule configuration that cannot be used (05-risk-management 8, strict config).
    InvalidRule { rule: &'static str, reason: String },
    /// A part of risk management that is not built yet.
    NotBuilt { feature: &'static str },
}

impl fmt::Display for RmsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RmsError::InvalidRule { rule, reason } => write!(f, "{rule}: {reason}"),
            RmsError::NotBuilt { feature } => {
                write!(
                    f,
                    "{feature} is not implemented yet (Phase 1 work in progress)"
                )
            }
        }
    }
}

impl std::error::Error for RmsError {}
