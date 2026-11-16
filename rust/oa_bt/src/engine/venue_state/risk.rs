//! The risk bookkeeping of a live session's saved state (`venue_state`): the trading
//! state with its cause, the daily loss totals, the drawdown peak, the slices the
//! cooldown and the time exit count from, and the engine exits waiting for an open.

use crate::rms::{
    AccountTotals, DailyLossMeasure, DailyLossTotals, DrawdownWatch, ExitReason, TradingLock,
    TradingState, TradingStateCause,
};

use super::super::strategy_risk::{PendingExit, StrategyRisk};
use super::error::LiveStateError;
use super::requests::{key_fields, read_key};
use super::value::{list, map, Read, StateValue};

fn write_totals(totals: AccountTotals) -> StateValue {
    map([
        ("realized_rupees", totals.realized_rupees.into()),
        ("costs_rupees", totals.costs_rupees.into()),
    ])
}

fn read_totals(read: &Read<'_>) -> Result<AccountTotals, LiveStateError> {
    Ok(AccountTotals {
        realized_rupees: read.f64_of("realized_rupees")?,
        costs_rupees: read.f64_of("costs_rupees")?,
    })
}

fn read_opt_totals(read: &Read<'_>, key: &str) -> Result<Option<AccountTotals>, LiveStateError> {
    read.maybe(key)?
        .map(|totals| read_totals(&totals))
        .transpose()
}

/// The strategy's risk bookkeeping (module docs).
pub(super) fn write_risk(risk: &StrategyRisk) -> StateValue {
    let daily = risk.daily_loss.totals();
    map([
        ("trading_state", risk.lock.state().code().into()),
        (
            "trading_state_cause",
            risk.lock.cause().map(TradingStateCause::code).into(),
        ),
        (
            "daily_loss",
            map([
                ("flat", write_totals(daily.flat)),
                (
                    "live_run_from",
                    daily.live_run_from.map(write_totals).into(),
                ),
                ("counted_from", daily.counted_from.map(write_totals).into()),
            ]),
        ),
        (
            "drawdown_peak_rupees",
            risk.drawdown.map(|watch| watch.peak_rupees()).into(),
        ),
        (
            "last_exit_slices",
            list(&risk.last_exit_slice, |slice| (*slice).into()),
        ),
        (
            "entered_at",
            list(&risk.entered_at, |(key, slice)| {
                let [instrument, product] = key_fields(*key);
                map([instrument, product, ("slice", (*slice).into())])
            }),
        ),
        ("roll_carry_slice", risk.roll_carry.into()),
        (
            "pending_exits",
            list(&risk.pending_exits, |pending| {
                let [instrument, product] = key_fields(pending.key);
                map([
                    instrument,
                    product,
                    ("reason", pending.reason.code().into()),
                ])
            }),
        ),
    ])
}

/// The trading lock saved as its state and cause: the one change from ACTIVE the cause
/// makes, which every state a lock can be in is.
fn read_lock(read: &Read<'_>) -> Result<TradingLock, LiveStateError> {
    let state = read.field("trading_state")?;
    let mut lock = TradingLock::default();
    if let Some(cause) = read.maybe("trading_state_cause")? {
        lock.apply(cause.code(&TradingStateCause::ALL, TradingStateCause::code)?);
    }
    let states = [
        TradingState::Active,
        TradingState::Reducing,
        TradingState::Halted,
    ];
    if state.code(&states, TradingState::code)? != lock.state() {
        return Err(state.malformed("does not follow from its cause"));
    }
    Ok(lock)
}

/// `base` (the restarted session's bookkeeping, which keeps its session day) with what
/// `write_risk` wrote; the drawdown halt keeps the session's `drawdown_limit_percent`.
///
/// Errors: `Malformed` for a field missing or unusable, a drawdown peak saved without a
/// drawdown halt set or the reverse, and a list of exit slices that is not one per
/// instrument.
pub(super) fn read_risk(
    read: &Read<'_>,
    base: &StrategyRisk,
    instruments: usize,
    drawdown_limit_percent: Option<f64>,
) -> Result<StrategyRisk, LiveStateError> {
    let daily = read.field("daily_loss")?;
    let totals = DailyLossTotals {
        flat: read_totals(&daily.field("flat")?)?,
        live_run_from: read_opt_totals(&daily, "live_run_from")?,
        counted_from: read_opt_totals(&daily, "counted_from")?,
    };
    let peak = read.field("drawdown_peak_rupees")?;
    let drawdown = match (
        drawdown_limit_percent,
        read.opt_f64_of("drawdown_peak_rupees")?,
    ) {
        (Some(limit), Some(peak)) => Some(DrawdownWatch::new(limit, peak)),
        (None, None) => None,
        _ => return Err(peak.malformed("does not match the session's drawdown halt setting")),
    };
    let exits = read.field("last_exit_slices")?;
    let last_exit_slice = exits
        .items()?
        .iter()
        .map(|slice| (!slice.is_null()).then(|| slice.whole()).transpose())
        .collect::<Result<Vec<_>, LiveStateError>>()?;
    if last_exit_slice.len() != instruments {
        return Err(exits.malformed(format!("has not one slice per instrument ({instruments})")));
    }
    let entered_at = read
        .items_of("entered_at")?
        .iter()
        .map(|entry| Ok((read_key(entry, instruments)?, entry.whole_of("slice")?)))
        .collect::<Result<_, LiveStateError>>()?;
    let pending_exits = read
        .items_of("pending_exits")?
        .iter()
        .map(|pending| {
            Ok(PendingExit {
                key: read_key(pending, instruments)?,
                reason: pending.field("reason")?.parsed(ExitReason::from_code)?,
            })
        })
        .collect::<Result<_, LiveStateError>>()?;
    Ok(StrategyRisk {
        lock: read_lock(read)?,
        daily_loss: DailyLossMeasure::resume(totals),
        drawdown,
        last_exit_slice,
        entered_at,
        roll_carry: read.opt_whole_of("roll_carry_slice")?,
        pending_exits,
        ..base.clone()
    })
}
