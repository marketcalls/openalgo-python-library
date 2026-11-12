//! Input the metrics kernel cannot use. Every message names the column and the row and
//! says what to change.

use std::fmt;

/// Why a metric could not be computed from the input.
#[derive(Clone, Debug, PartialEq)]
pub enum MetricsError {
    /// An equity series with fewer points than the calculation needs.
    TooFewPoints { needed: usize, found: usize },
    /// Two columns of the same table have different lengths.
    LengthMismatch {
        what: &'static str,
        expected: usize,
        found: usize,
    },
    /// Timestamps that do not increase at `index` (strictly for equity points; cash
    /// flows may share a timestamp).
    NotIncreasing { what: &'static str, index: usize },
    /// A value that is NaN or infinite where a number is required.
    NotFinite { what: &'static str, index: usize },
    /// A starting equity of zero or below: no return can be measured on it.
    NonPositiveStart { equity: f64 },
    /// An opening wealth (the capital before the first point) that is not a positive
    /// finite number of rupees.
    InvalidOpeningWealth { value: f64 },
    /// A cash flow stamped at or before the first equity point or after the last.
    FlowOutsideCurve { index: usize },
    /// A cash flow after which no capital is left, so later returns are undefined.
    NoCapitalAfterFlow { index: usize },
    /// A trade whose exit is before its entry.
    ExitBeforeEntry { index: usize },
    /// A trade whose gross P&L minus costs is not its net P&L.
    PnlMismatch {
        index: usize,
        gross: f64,
        costs: f64,
        net: f64,
    },
    /// Sessions per year that is not a positive finite number.
    InvalidPeriodsPerYear { value: f64 },
    /// An annual risk-free rate that is not finite or is -100 percent or below.
    InvalidRiskFreeRate { value: f64 },
    /// `combine_equity` called with no series.
    NothingToCombine,
}

impl fmt::Display for MetricsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetricsError::TooFewPoints { needed, found } => write!(
                f,
                "the equity series has {found} point(s) and needs at least {needed}: the \
                 equity at each mark, with the starting equity as the first point when \
                 no opening wealth is given"
            ),
            MetricsError::LengthMismatch {
                what,
                expected,
                found,
            } => write!(
                f,
                "{what} has {found} value(s) where {expected} are needed: every column of \
                 a table must have one value per row"
            ),
            MetricsError::NotIncreasing { what, index } => write!(
                f,
                "{what} do not increase at row {index}: sort the rows by time and remove \
                 duplicate timestamps"
            ),
            MetricsError::NotFinite { what, index } => write!(
                f,
                "{what} at row {index} is missing or infinite: supply a number for every row"
            ),
            MetricsError::NonPositiveStart { equity } => write!(
                f,
                "the starting equity is {equity}: returns need a positive starting capital"
            ),
            MetricsError::InvalidOpeningWealth { value } => write!(
                f,
                "the opening wealth is {value}: it is the capital before the first equity \
                 point, a positive number of rupees"
            ),
            MetricsError::FlowOutsideCurve { index } => write!(
                f,
                "cash flow {index} is not after the first equity point and at or before the \
                 last: fold a flow at the start into the starting equity (or the opening \
                 wealth), and drop flows after the end"
            ),
            MetricsError::NoCapitalAfterFlow { index } => write!(
                f,
                "cash flow {index} leaves no capital in the account: end the equity series \
                 at that point"
            ),
            MetricsError::ExitBeforeEntry { index } => {
                write!(f, "trade {index} exits before it enters: check its times")
            }
            MetricsError::PnlMismatch {
                index,
                gross,
                costs,
                net,
            } => write!(
                f,
                "trade {index}: gross P&L {gross} minus costs {costs} is not its net P&L \
                 {net}; net P&L must be gross P&L minus costs, in rupees"
            ),
            MetricsError::InvalidPeriodsPerYear { value } => write!(
                f,
                "sessions per year must be a positive number, got {value}"
            ),
            MetricsError::InvalidRiskFreeRate { value } => write!(
                f,
                "the risk-free rate must be an annual fraction above -1 (0.065 is 6.5 \
                 percent), got {value}"
            ),
            MetricsError::NothingToCombine => {
                write!(f, "no equity series to combine: pass at least one")
            }
        }
    }
}

impl std::error::Error for MetricsError {}
