//! Registering futures series before the run, and planning their switches from the
//! bars (`futures` states the rules). Also the data's settlement prices.

use crate::exec::InstrumentId;
use crate::market::calendar::format_day;
use crate::model::{Day, InstrumentKind};

use super::error::KernelError;
use super::futures::{invalid, FuturesSeries, RollRule, Switch};
use super::kernel::{Engine, Phase};

impl Engine {
    /// Register a futures series (module docs). Call before the first `advance`.
    /// Returns the series' index.
    ///
    /// Errors: `InvalidConfig` (field `roll`) after the run started, for an empty
    /// series, an instrument that is not a future with a last trading day or that is in
    /// another series, hand-over sessions out of order, before the run or on the wrong
    /// side of their contract's last trading day, and a hand-over whose bars do not
    /// allow a switch (no slice in which both contracts have a bar).
    pub fn add_futures_series(&mut self, series: FuturesSeries) -> Result<u32, KernelError> {
        if self.phase != Phase::Ready(0) {
            return Err(invalid(
                "futures series are registered before the run starts".to_string(),
            ));
        }
        self.check_series(&series)?;
        let index = u32::try_from(self.futures.series.len()).unwrap_or(u32::MAX);
        let mut switches = Vec::new();
        for pair in series.contracts.windows(2) {
            let (near, next) = (pair[0], pair[1]);
            let Some(hand_over) = near.hand_over else {
                continue;
            };
            if let Some(switch) = self.plan_switch(
                index,
                near.instrument,
                next.instrument,
                hand_over,
                series.rule,
            )? {
                switches.push(switch);
            }
        }
        self.futures.switches.extend(switches);
        self.futures
            .switches
            .sort_by_key(|switch| (switch.slice, switch.series));
        self.futures.series.push(series);
        Ok(index)
    }

    /// Every switch of the run's series, in slice order (known before the run, fixed by
    /// the bars).
    pub fn futures_switches(&self) -> &[Switch] {
        &self.futures.switches
    }

    /// Settlement prices of a future by IST session date, rupees, from the data: the
    /// daily mark to market and the expiry settlement use them instead of the last
    /// close. Call before the first `advance`.
    ///
    /// Errors: `InvalidConfig` (field `settlement`) after the run started, for an
    /// instrument that is not a future, days out of order or a price that is not a
    /// positive amount.
    pub fn set_settlement_prices(
        &mut self,
        instrument: InstrumentId,
        prices: Vec<(Day, f64)>,
    ) -> Result<(), KernelError> {
        let refuse = |reason: &str| KernelError::InvalidConfig {
            field: "settlement",
            reason: reason.to_string(),
        };
        if self.phase != Phase::Ready(0) {
            return Err(refuse("settlement prices are given before the run starts"));
        }
        if self.spec(instrument)?.kind != InstrumentKind::Future {
            return Err(refuse("settlement prices are for futures"));
        }
        if prices.windows(2).any(|pair| pair[0].0 >= pair[1].0) {
            return Err(refuse("settlement days must be strictly increasing"));
        }
        if prices.iter().any(|(_, p)| !(p.is_finite() && *p > 0.0)) {
            return Err(refuse(
                "a settlement price must be a positive amount of rupees",
            ));
        }
        self.futures
            .settlement_prices
            .retain(|(i, _)| *i != instrument);
        self.futures.settlement_prices.push((instrument, prices));
        Ok(())
    }

    fn check_series(&self, series: &FuturesSeries) -> Result<(), KernelError> {
        if series.contracts.is_empty() {
            return Err(invalid(
                "a futures series needs at least one contract".to_string(),
            ));
        }
        let mut previous: Option<Day> = None;
        for (position, contract) in series.contracts.iter().enumerate() {
            let spec = self.spec(contract.instrument)?;
            let Some(last_day) = spec
                .last_trading_day
                .filter(|_| spec.kind == InstrumentKind::Future)
            else {
                return Err(invalid(format!(
                    "{} is not a future with a last trading day",
                    spec.symbol
                )));
            };
            let taken = self.futures.series.iter().flat_map(|s| s.contracts.iter());
            if taken
                .into_iter()
                .any(|c| c.instrument == contract.instrument)
            {
                return Err(invalid(format!("{} is in two futures series", spec.symbol)));
            }
            let last = position + 1 == series.contracts.len();
            match (contract.hand_over, last) {
                (Some(_), true) => {
                    return Err(invalid(format!(
                        "{} is the last contract of its series and hands over to none",
                        spec.symbol
                    )))
                }
                (None, false) => {
                    return Err(invalid(format!(
                        "{} hands over to a next contract on no session",
                        spec.symbol
                    )))
                }
                _ => {}
            }
            if let Some(day) = contract.hand_over {
                let fits = match series.rule {
                    RollRule::Roll => day <= last_day,
                    RollRule::SettleAtExpiry => day > last_day,
                };
                if !fits || previous.is_some_and(|p| p >= day) {
                    return Err(invalid(format!(
                        "{} cannot hand over on {}: hand-over sessions increase, a roll comes \
                         on or before the last trading day {} and a settlement after it",
                        spec.symbol,
                        format_day(day),
                        format_day(last_day)
                    )));
                }
                previous = Some(day);
            }
        }
        Ok(())
    }

    /// The switch from `near` to `next` at `hand_over`, or `None` when the session is
    /// after the run.
    fn plan_switch(
        &self,
        series: u32,
        near: InstrumentId,
        next: InstrumentId,
        hand_over: Day,
        rule: RollRule,
    ) -> Result<Option<Switch>, KernelError> {
        let first_day = self.timeline.slice(0).map_or(hand_over, |slice| slice.day);
        if hand_over < first_day {
            return Err(invalid(format!(
                "the hand-over session {} of {} is before the run's first session; resolve \
                 the series from the run's start",
                format_day(hand_over),
                self.spec(near)?.symbol
            )));
        }
        let last_day = self.spec(near)?.last_trading_day.unwrap_or(hand_over);
        let both = |index: usize| {
            let near_bar = self.market.bar(near, index);
            let next_bar = self.market.bar(next, index);
            near_bar.zip(next_bar)
        };
        let from = (0..self.timeline.len()).find(|&index| {
            self.timeline
                .slice(index)
                .is_some_and(|s| s.day >= hand_over)
        });
        let Some(from) = from else {
            return Ok(None);
        };
        let found = match rule {
            RollRule::Roll => (from..self.timeline.len())
                .take_while(|&i| self.timeline.slice(i).is_some_and(|s| s.day <= last_day))
                .find_map(|i| both(i).map(|(n, x)| (i, x.open - n.open))),
            RollRule::SettleAtExpiry => (0..from)
                .rev()
                .take_while(|&i| self.timeline.slice(i).is_some_and(|s| s.day >= last_day))
                .find_map(|i| both(i).map(|(n, x)| (from, x.close - n.close))),
        };
        let Some((index, spread)) = found else {
            return Err(KernelError::InvalidData {
                symbol: self.spec(near)?.symbol.clone(),
                reason: format!(
                    "no slice from {} to its last trading day {} has a bar of both it and {}, \
                     so the series cannot move to the next contract",
                    format_day(hand_over.min(last_day)),
                    format_day(last_day),
                    self.spec(next)?.symbol
                ),
            });
        };
        let slice = self
            .timeline
            .slice(index)
            .ok_or_else(|| invalid("no such slice".to_string()))?;
        Ok(Some(Switch {
            series,
            from: near,
            to: next,
            slice: index,
            slice_ts: slice.open,
            day: slice.day,
            spread_rupees: spread,
            rolls_positions: rule == RollRule::Roll,
        }))
    }
}
