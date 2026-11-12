//! The Rust driver of a condition strategy: at each pause where the strategy's
//! instrument has a closed strategy bar, read what is held, decide (`decide`) and
//! submit, exactly where the event driver would call `on_bar`.

use crate::exec::{OrderRequest, QuantitySpec};
use crate::model::Side;

use super::super::error::KernelError;
use super::super::kernel::Engine;
use super::super::pause::{Driver, Pause};
use super::super::submit::Submission;
use super::decide::{decide, Decision, Holding};
use super::{EntrySignal, SignalArrays, SkippedEntry};

/// Drives one condition strategy through `pause::run`.
pub(super) struct SignalDriver<'a> {
    signals: &'a SignalArrays,
    /// Entries a sizer skipped, in decision order.
    pub(super) skipped: Vec<SkippedEntry>,
}

impl<'a> SignalDriver<'a> {
    pub(super) fn new(signals: &'a SignalArrays) -> SignalDriver<'a> {
        SignalDriver {
            signals,
            skipped: Vec::new(),
        }
    }

    /// What the strategy holds in its product, read in step 7 as `self.position` reads
    /// it; a position never opened is flat.
    fn holding(&self, engine: &Engine) -> Holding {
        let (held, adds) = engine
            .position(self.signals.instrument, self.signals.product)
            .map_or((0.0, 0), |position| {
                (position.quantity_units, position.adds)
            });
        if held > 0.0 {
            Holding::Long { adds }
        } else if held < 0.0 {
            Holding::Short { adds }
        } else if engine.has_pending_entry(self.signals.instrument) {
            Holding::EntryPending
        } else {
            Holding::Flat
        }
    }

    /// The order `exit()` places: the whole position, reduce-only, reason `signal`.
    fn exit_order(&self, holding: Holding) -> OrderRequest {
        let side = match holding {
            Holding::Short { .. } => Side::Buy,
            _ => Side::Sell,
        };
        OrderRequest {
            product: self.signals.product,
            reduce_only: true,
            ..OrderRequest::market(
                self.signals.instrument,
                side,
                QuantitySpec::PositionFraction(1.0),
            )
        }
    }
}

impl Driver for SignalDriver<'_> {
    /// Decide at a strategy-bar close of the instrument; every other pause (events
    /// only, timers, the end) needs no decision. A rejected order is not an error, as in
    /// `on_bar`; an entry a sizer skipped is kept for the result's skipped orders.
    fn on_pause(&mut self, engine: &mut Engine, pause: &Pause) -> Result<(), KernelError> {
        let Pause::BarClose { bars, .. } = pause else {
            return Ok(());
        };
        let instrument = self.signals.instrument;
        let Some(bar) = bars.iter().find(|bar| bar.instrument == instrument) else {
            return Ok(());
        };
        let index = self.signals.bar_index(bar.ts_open)?;
        let holding = self.holding(engine);
        let entry_at = |signal: Option<&EntrySignal>| {
            signal
                .map(|signal| signal.order_at(index, bar.ts_open))
                .transpose()
        };
        let order = match decide(holding, self.signals.at(index), self.signals.pyramiding) {
            Decision::Hold => None,
            Decision::EnterLong => entry_at(self.signals.entry.as_ref())?,
            Decision::EnterShort => entry_at(self.signals.short.as_ref())?,
            Decision::Exit => Some(self.exit_order(holding)),
        };
        let Some(order) = order else {
            return Ok(());
        };
        if let Submission::Skipped(reason) = engine.submit(order)? {
            self.skipped.push(SkippedEntry {
                decided_at: engine.decision_instant().unwrap_or(bar.ts_close),
                reason,
            });
        }
        Ok(())
    }
}
