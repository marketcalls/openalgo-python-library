//! The strategy's daily limits (18-multi-strategy 2.5a), as the engine applies them: the
//! trade caps (rules 1 to 4 and 7), the daily profit target (rule 5) and the order value
//! cap (rule 11). Each is off unless the run sets it, and then changes nothing.
//!
//! - **The trade count** (`rms::TradeCount`). An entry is an order that opens a position
//!   from flat, or a reversal (its opening part); it is counted on the IST session day of
//!   its decision once it passes its checks (`counted`), while it works and once any part
//!   of it fills. A modification is not a new trade.
//! - **The trade caps**, at submission after every other check (in a live session after
//!   the gates of a shared instrument, which stay first): an entry when the strategy's
//!   count is at `max_trades_per_day` is rejected `trade_limit`, and when the account's
//!   count (the strategy's and the other strategies', `rms::AccountTrades`, set by a
//!   portfolio run's fund or a live session) is at `account_max_trades_per_day`,
//!   `account_trade_limit`. Nothing is exited and nothing is locked.
//! - **The order value cap**, beside them: an entry or add whose value is above
//!   `max_order_value` is rejected `order_value_limit`. The value is the order's units
//!   (its opening part's for a reversal) times the price it is judged at times the
//!   instrument's multiplier: a LIMIT or stop-limit order at its limit, an SL-M at its
//!   trigger, a MARKET order at the last close, the price the sizers size it at. It is
//!   judged at submission, in a backtest as live (where it is the price a live session's
//!   margin check quotes, `venue_checks`, and no fill price is known before the order is
//!   sent), so both refuse the same orders. A MARKET order of an instrument with no bar
//!   yet has no price to judge: under a cap it is refused, never taken as worth nothing.
//!   An order the engine default sizes at the fill (part shares, `fractional`) is judged at
//!   its fill, at the price it is sized at there (`sizing::size_at_fill`). Exits, stops,
//!   targets, the square-off, risk exits and rolls are never refused by it.
//! - **A reversal at a cap** (rule 4) is cut to its close: an order that only reduces, for
//!   the units held, goes through every check as an exit, and an order for the opening
//!   part is recorded rejected with the cap's code (`cut_reversal`). A trader who caps
//!   trades still has the flip signal take the position off.
//! - **The daily profit target**, in step 6 after the daily loss limit (`aggregate`), on
//!   the same measure: at or past `+daily_profit_target` while ACTIVE, the risk event
//!   `daily_profit_hit` (scope session, the target as the level, the measure as observed),
//!   REDUCING until the next session (entries then rejected `daily_profit_gate`,
//!   `rms::TradingLock`), working entries and adds cancelled and every position exited
//!   with the reason `daily_profit_target`, exactly as the daily loss limit's.

use crate::exec::{OrderId, OrderRequest, OrderType, QuantitySpec, RejectCode};
use crate::model::{Day, Side};
use crate::portfolio::PositionKey;
use crate::rms::{
    is_target_reached, order_value_gate, trade_cap_gate, LegRules, RiskEvent, RiskEventName,
    RiskScope, TradingState,
};

use super::clock::SliceInfo;
use super::error::KernelError;
use super::kernel::Engine;
use super::submit::Submission;

/// What the daily caps make of an order that passed every other submission check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Capped {
    /// It passes; `true` when it is an entry the trade count counts.
    Pass(bool),
    /// It is rejected with this code.
    Refuse(RejectCode),
    /// A reversal whose opening part a cap refuses with this code: cut to its close.
    CutReversal(RejectCode),
}

impl Engine {
    /// The entries the strategy has counted on the session day of the slice it last
    /// opened (module docs): those that filled and those that still work. 0 before its
    /// first slice.
    pub fn trades_today(&self) -> u32 {
        self.risk.session_day.map_or(0, |day| self.trades_on(day))
    }

    /// The entries counted on session day `day`.
    pub(super) fn trades_on(&self, day: Day) -> u32 {
        let works = |id: OrderId| {
            self.book
                .get(id)
                .is_some_and(|order| order.status.is_working())
        };
        self.risk.trades.on(day, works)
    }

    /// The daily caps of `request` for `units` (0 when it is sized at its fill), decided
    /// in `slice`; `modifying` for a modification, which the trade caps do not count
    /// (module docs).
    ///
    /// Errors: `Exec` for an instrument the run does not have.
    pub(super) fn daily_caps(
        &self,
        request: &OrderRequest,
        units: f64,
        slice: &SliceInfo,
        modifying: bool,
    ) -> Result<Capped, KernelError> {
        let key = PositionKey {
            instrument: request.instrument,
            product: request.product,
        };
        let held = self.held_units(key);
        let reduces = held != 0.0 && (held > 0.0) != (request.side == Side::Buy);
        if request.reduce_only || (reduces && units <= held.abs()) {
            return Ok(Capped::Pass(false));
        }
        let refuse = |code| match reduces {
            true => Capped::CutReversal(code),
            false => Capped::Refuse(code),
        };
        let entry = (held == 0.0 || reduces) && !modifying;
        if entry {
            let own = self.trades_on(slice.day);
            let caps = trade_cap_gate(
                self.config.risk.max_trades_per_day,
                own,
                self.account.link.trades,
            );
            if let Err(code) = caps {
                return Ok(refuse(code));
            }
        }
        let opening = if reduces { units - held.abs() } else { units };
        let value = match self.judged_price(request) {
            Some(price) => self.order_value(request, opening, price)?,
            // No price to judge it at (module docs): above any cap, so refused under one.
            None if opening > 0.0 => f64::INFINITY,
            None => 0.0,
        };
        Ok(
            match order_value_gate(self.config.risk.max_order_value_rupees, value) {
                Err(code) => refuse(code),
                Ok(()) => Capped::Pass(entry),
            },
        )
    }

    /// Count `order`, an entry decided in `slice` that passed its checks.
    pub(super) fn counted(&mut self, slice: &SliceInfo, order: OrderId) {
        self.risk.trades.entered(slice.day, order);
    }

    /// The price an order is judged at for its value (module docs): its limit, an SL-M's
    /// trigger, else the last close; None for a MARKET order of an instrument with no bar
    /// yet.
    fn judged_price(&self, request: &OrderRequest) -> Option<f64> {
        match request.order_type {
            OrderType::Limit { limit_rupees } | OrderType::StopLimit { limit_rupees, .. } => {
                Some(limit_rupees)
            }
            OrderType::StopMarket { trigger_rupees } => Some(trigger_rupees),
            OrderType::Market => self.last_close(request.instrument),
        }
    }

    /// The value of `units` of `request`'s instrument at `price_rupees`, rupees.
    ///
    /// Errors: `Exec` for an instrument the run does not have.
    fn order_value(
        &self,
        request: &OrderRequest,
        units: f64,
        price_rupees: f64,
    ) -> Result<f64, KernelError> {
        let multiplier = self.spec(request.instrument)?.multiplier;
        Ok(units * price_rupees * multiplier)
    }

    /// The order value cap of an entry or add the engine default sized at its fill, for
    /// `units` at `price_rupees` (module docs).
    ///
    /// Errors: `Ok(Err(OrderValueLimit))` above the cap; `Exec` for an instrument the run
    /// does not have.
    pub(super) fn order_value_at_fill(
        &self,
        request: &OrderRequest,
        units: f64,
        price_rupees: f64,
    ) -> Result<Result<(), RejectCode>, KernelError> {
        let value = self.order_value(request, units, price_rupees)?;
        Ok(order_value_gate(
            self.config.risk.max_order_value_rupees,
            value,
        ))
    }

    /// Submit a reversal whose opening part a cap refuses with `code` (module docs): the
    /// close, as an order that only reduces the `held` units through every check, then the
    /// opening part's order, rejected with `code`. Returns the close's submission.
    ///
    /// Errors: those of `submit`.
    pub(super) fn cut_reversal(
        &mut self,
        request: OrderRequest,
        units: f64,
        code: RejectCode,
    ) -> Result<Submission, KernelError> {
        let key = PositionKey {
            instrument: request.instrument,
            product: request.product,
        };
        let held = self.held_units(key).abs();
        let close = OrderRequest {
            quantity: QuantitySpec::Units(held),
            reduce_only: true,
            reverse: false,
            leg_rules: LegRules::default(),
            ..request.clone()
        };
        let submitted = self.submit(close)?;
        let rest = OrderRequest {
            quantity: QuantitySpec::Units(units - held),
            ..request
        };
        self.record_submission(rest, units - held, Err(code))?;
        Ok(submitted)
    }

    /// The daily profit target's event when the strategy's measure reaches it while
    /// ACTIVE (module docs).
    pub(super) fn daily_profit_reached(&self) -> Option<RiskEvent> {
        let target = self.config.risk.daily_profit_target_rupees?;
        if self.risk.lock.state() != TradingState::Active {
            return None;
        }
        let measure = self.risk.daily_loss.measure_rupees(
            self.config.risk.risk_basis,
            self.account_totals(),
            self.portfolio.unrealized_rupees(),
        );
        is_target_reached(measure, target).then_some(RiskEvent {
            name: RiskEventName::DailyProfitHit,
            scope: RiskScope::Session,
            level_rupees: Some(target),
            target_index: None,
            observed_rupees: Some(measure),
        })
    }
}

#[cfg(test)]
mod tests;
