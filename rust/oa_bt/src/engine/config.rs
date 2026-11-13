//! The run configuration: everything the Python layer resolved before the run, in
//! kernel units, and the inputs that come with it (calendar, instruments, bars).
//!
//! Defaults are resolved in Python (the documented defaults of 03-architecture 4.1 and
//! 04-execution 2) and printed in the Assumptions block; the kernel receives explicit
//! values and checks that it can honour every one of them. A setting the kernel cannot
//! honour yet is a `KernelError::NotBuilt`, never ignored.

use crate::data::resample::CalendarUnit;
use crate::exec::InstrumentSpec;
use crate::market::calendar::{BaseInterval, Calendar};
use crate::market::costs::CostModel;
use crate::market::facts::MarketFacts;
use crate::market::margin::MarginModel;
use crate::market::slippage::SlippageSpec;
use crate::model::{Day, TsNanos, NANOS_PER_DAY};
use crate::rms::StrategyRiskConfig;
use crate::sizing::SizingControls;

use super::error::KernelError;

/// The interval at whose bar closes the strategy decides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrategyInterval {
    /// An intraday interval in nanoseconds, on the session lattice.
    Intraday(i64),
    /// One bar per session.
    Daily,
    /// Weekly, monthly, quarterly or yearly bars.
    Period(CalendarUnit),
}

impl StrategyInterval {
    /// True for an intraday interval (DAY orders end in the decision session).
    pub fn is_intraday(self) -> bool {
        matches!(self, StrategyInterval::Intraday(_))
    }
}

/// When a decision fills.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FillTiming {
    /// At the next base bar's open (the default).
    #[default]
    NextOpen,
    /// At the decision bar's close (`fill="close"`, an explicit opt-in).
    Close,
}

/// The quantity rule of a backtest (03-architecture 7).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QuantityMode {
    /// Fractional cash equity (research only); derivatives stay in whole lots.
    Fractional,
    /// The venue rule: whole shares, whole lots (the backtest default, decision D26).
    #[default]
    Venue,
}

/// Charges on fills.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum CostConfig {
    /// No charges (`costs=None`, research only; every report says so).
    #[default]
    Off,
    /// The Indian cost model of `market::costs`.
    Model(CostModel),
}

/// Overrides of the default intraday times, minutes since IST midnight. `None` keeps
/// the default: the session open for `entry_from`, the calendar's dated defaults for
/// the other two (`market::calendar::default_intraday_times`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IntradayOverrides {
    /// `entry_from`.
    pub entry_from_minute: Option<u32>,
    /// `entry_until`.
    pub entry_until_minute: Option<u32>,
    /// `square_off`.
    pub square_off_minute: Option<u32>,
}

/// Every resolved setting of a run.
#[derive(Clone, Debug, PartialEq)]
pub struct RunConfig {
    /// First IST date of the run.
    pub first_day: Day,
    /// Last IST date of the run (inclusive).
    pub last_day: Day,
    /// Optional first slice instant (UTC nanoseconds): slices opening earlier are not
    /// run.
    pub first_slice: Option<TsNanos>,
    /// Optional last slice instant (UTC nanoseconds): slices opening later are not run.
    pub last_slice: Option<TsNanos>,
    /// The present, UTC nanoseconds: slices closing later are not run (`run_end`). The
    /// caller resolves it; the kernel reads no clock. `None` sets no such bound.
    pub present: Option<TsNanos>,
    /// The base interval of 03-architecture 4.1.
    pub base: BaseInterval,
    /// The strategy interval.
    pub strategy_interval: StrategyInterval,
    /// When decisions fill.
    pub fill: FillTiming,
    /// Quantity rule.
    pub quantity_mode: QuantityMode,
    /// Capital allocated to the strategy, rupees.
    pub capital_rupees: f64,
    /// Charges.
    pub costs: CostConfig,
    /// Slippage on every MARKET fill.
    pub slippage: SlippageSpec,
    /// Margin rules.
    pub margin: MarginModel,
    /// Per-instrument facts of the market rules (underlying, BSE group, slippage class
    /// and override), in registration order; empty derives them from exchange and kind.
    pub market: MarketFacts,
    /// Strategy and session risk rules.
    pub risk: StrategyRiskConfig,
    /// `reinvest` and the caps shared by every sizer.
    pub sizing: SizingControls,
    /// Intraday time overrides.
    pub intraday: IntradayOverrides,
    /// Pause at step 7 when events are waiting even if no strategy bar closes (the
    /// strategy has `on_fill`, `on_order`, `on_position` or `on_risk_event`).
    pub deliver_events: bool,
}

/// One instrument's base bars, as columns. `ts_open` holds UTC nanoseconds, strictly
/// increasing; prices are rupees and volume is units.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BarColumns {
    pub ts_open: Vec<TsNanos>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
}

/// Everything a run needs: the configuration, the calendar, the instruments in
/// registration order and their base bars in the same order.
#[derive(Clone, Debug)]
pub struct RunSetup {
    pub config: RunConfig,
    pub calendar: Calendar,
    pub instruments: Vec<InstrumentSpec>,
    pub bars: Vec<BarColumns>,
}

fn invalid(field: &'static str, reason: &str) -> KernelError {
    KernelError::InvalidConfig {
        field,
        reason: reason.to_string(),
    }
}

fn not_built(feature: &'static str, instead: &'static str) -> KernelError {
    KernelError::NotBuilt { feature, instead }
}

/// Strategy intervals longer than a day: what is not supported and what to do instead.
/// The clock and the market data refuse them with the same words.
pub(super) const PERIOD_INTERVALS: &str = "a strategy interval longer than a day (W, M, Q or Y)";
pub(super) const PERIOD_INTERVALS_INSTEAD: &str =
    "run the strategy on interval=\"D\" and read the longer bars with \
     self.data[symbol, \"W\"] (or \"M\", \"Q\", \"Y\")";

impl RunConfig {
    /// Check that the kernel can run this configuration as given.
    ///
    /// Errors: `InvalidConfig` for a reversed date range, capital that is not positive,
    /// or a strategy interval that is not a whole multiple of the base; `Rms` for an
    /// unusable strategy or session risk rule; `Sizing` for an unusable cap; `NotBuilt`
    /// for `fill="close"` and period intervals.
    pub fn validate(&self) -> Result<(), KernelError> {
        if self.first_day > self.last_day {
            return Err(invalid("start", "the first day is after the last day"));
        }
        if !self.capital_rupees.is_finite() || self.capital_rupees <= 0.0 {
            return Err(invalid("capital", "must be a positive amount of rupees"));
        }
        self.check_intervals()?;
        if self.fill == FillTiming::Close {
            return Err(not_built(
                "fill=\"close\"",
                "use fill=\"next_open\" (the default): an order fills at the next bar's open",
            ));
        }
        self.risk.validate()?;
        self.sizing.validate()?;
        Ok(())
    }

    fn check_intervals(&self) -> Result<(), KernelError> {
        match (self.base, self.strategy_interval) {
            (BaseInterval::Intraday(base), StrategyInterval::Intraday(strategy)) => {
                let usable = base > 0 && strategy >= base && strategy % base == 0;
                if usable && strategy < NANOS_PER_DAY {
                    Ok(())
                } else {
                    Err(invalid(
                        "interval",
                        "the strategy interval must be a whole multiple of the base interval",
                    ))
                }
            }
            (BaseInterval::Daily, StrategyInterval::Intraday(_)) => Err(invalid(
                "base",
                "an intraday strategy needs an intraday base interval",
            )),
            (_, StrategyInterval::Period(_)) => {
                Err(not_built(PERIOD_INTERVALS, PERIOD_INTERVALS_INSTEAD))
            }
            _ => Ok(()),
        }
    }
}
