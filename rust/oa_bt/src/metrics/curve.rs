//! The time-weighted wealth index of an equity series, and its per-session returns.
//!
//! Definitions, for points `0..n` with equity `E_i` at `t_i`:
//!
//! - `F_i`, the cash flow booked at point `i`: the sum of the flows stamped in
//!   `(t_{i-1}, t_i]`. A flow is taken to arrive at the point that follows it, so the
//!   P&L of that interval is earned on `E_{i-1}` alone. Deposits are positive,
//!   withdrawals negative.
//! - Bar return `r_i = (E_i - F_i) / E_{i-1} - 1`, and the wealth index
//!   `W_0 = 1`, `W_i = W_{i-1} * (1 + r_i)`. Between two flows the index is computed
//!   as `W_k * E_i / E_k` from the last flow point `k`, so an equity that returns to an
//!   earlier value gives exactly the earlier index.
//! - Cumulative P&L `P_i = E_i - E_0 - (F_1 + ... + F_i)`, in rupees.
//! - Ruin: the first point at which the equity before its flow, `E_i - F_i`, is zero
//!   or below. The index is 0 from there on: the capital is lost and no later return
//!   is defined.
//! - Session return: the IST date of each point after the first is its session date.
//!   The return of a session is `W(last point of the date) / W(last point of the
//!   previous date) - 1`, with `W_0` as the base of the first date. Sessions after the
//!   ruin session have no return.

use crate::model::{ist_day, Day, TsNanos};

use super::error::MetricsError;

/// An equity series: one equity value in rupees per UTC nanosecond timestamp.
#[derive(Clone, Copy, Debug)]
pub struct EquityCurve<'a> {
    pub ts: &'a [TsNanos],
    pub equity: &'a [f64],
}

/// Deposits (positive) and withdrawals (negative) in rupees, by UTC nanosecond time.
#[derive(Clone, Copy, Debug)]
pub struct CashFlows<'a> {
    pub ts: &'a [TsNanos],
    pub amount: &'a [f64],
}

impl CashFlows<'static> {
    /// No cash flows.
    pub const NONE: CashFlows<'static> = CashFlows {
        ts: &[],
        amount: &[],
    };
}

/// The wealth index of an equity series and what it rests on.
#[derive(Clone, Debug, PartialEq)]
pub struct Wealth {
    /// Time-weighted wealth index per point: 1.0 at the start, 0.0 from ruin on.
    pub index: Vec<f64>,
    /// Cumulative P&L per point, in rupees (equity minus the start and the flows).
    pub pnl: Vec<f64>,
    /// The cash flow booked at each point, in rupees.
    pub flow: Vec<f64>,
    /// The first point at which the account was ruined, if it was.
    pub ruin: Option<usize>,
}

/// Per-session returns of a wealth index.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionReturns {
    /// IST session dates, increasing.
    pub days: Vec<Day>,
    /// Return of each session, as a fraction.
    pub returns: Vec<f64>,
}

pub(super) fn check_increasing(ts: &[TsNanos], what: &'static str) -> Result<(), MetricsError> {
    match ts.windows(2).position(|pair| pair[1] <= pair[0]) {
        Some(index) => Err(MetricsError::NotIncreasing {
            what,
            index: index + 1,
        }),
        None => Ok(()),
    }
}

pub(super) fn check_finite(values: &[f64], what: &'static str) -> Result<(), MetricsError> {
    match values.iter().position(|value| !value.is_finite()) {
        Some(index) => Err(MetricsError::NotFinite { what, index }),
        None => Ok(()),
    }
}

pub(super) fn check_length(
    values: usize,
    expected: usize,
    what: &'static str,
) -> Result<(), MetricsError> {
    if values == expected {
        Ok(())
    } else {
        Err(MetricsError::LengthMismatch {
            what,
            expected,
            found: values,
        })
    }
}

impl EquityCurve<'_> {
    /// Check the series: equal lengths, at least `needed` points, strictly increasing
    /// timestamps and finite equity.
    pub(super) fn check(&self, needed: usize) -> Result<(), MetricsError> {
        check_length(self.equity.len(), self.ts.len(), "equity")?;
        if self.ts.len() < needed {
            return Err(MetricsError::TooFewPoints {
                needed,
                found: self.ts.len(),
            });
        }
        check_increasing(self.ts, "equity timestamps")?;
        check_finite(self.equity, "equity")
    }
}

/// The cash flow booked at each equity point (see the module docs).
fn flows_per_point(curve: &EquityCurve, flows: &CashFlows) -> Result<Vec<f64>, MetricsError> {
    check_length(flows.amount.len(), flows.ts.len(), "cash flow amounts")?;
    check_finite(flows.amount, "cash flow amount")?;
    if let Some(index) = flows.ts.windows(2).position(|pair| pair[1] < pair[0]) {
        let what = "cash flow timestamps";
        return Err(MetricsError::NotIncreasing {
            what,
            index: index + 1,
        });
    }
    let (Some(&first), Some(&last)) = (curve.ts.first(), curve.ts.last()) else {
        return Ok(Vec::new());
    };
    let mut booked = vec![0.0; curve.ts.len()];
    let mut point = 1;
    for (index, (&ts, &amount)) in flows.ts.iter().zip(flows.amount).enumerate() {
        if ts <= first || ts > last {
            return Err(MetricsError::FlowOutsideCurve { index });
        }
        while curve.ts.get(point).is_some_and(|&mark| mark < ts) {
            point += 1;
        }
        if let Some(slot) = booked.get_mut(point) {
            *slot += amount;
        }
    }
    Ok(booked)
}

/// The time-weighted wealth index, cumulative P&L and ruin point of an equity series
/// (definitions in the module docs).
///
/// Errors: `TooFewPoints` below two points; `LengthMismatch`, `NotIncreasing`,
/// `NotFinite` for malformed columns; `NonPositiveStart`; `FlowOutsideCurve`;
/// `NoCapitalAfterFlow` when a withdrawal empties the account.
pub fn wealth(curve: &EquityCurve, flows: &CashFlows) -> Result<Wealth, MetricsError> {
    curve.check(2)?;
    let flow = flows_per_point(curve, flows)?;
    let start = curve.equity.first().copied().unwrap_or(0.0);
    if start <= 0.0 {
        return Err(MetricsError::NonPositiveStart { equity: start });
    }
    let n = curve.equity.len();
    let (mut index, mut pnl) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let (mut base_index, mut base_equity) = (1.0, start);
    let (mut previous_index, mut previous_equity) = (1.0, start);
    let (mut flows_so_far, mut ruin) = (0.0, None);
    for (point, (&equity, &booked)) in curve.equity.iter().zip(&flow).enumerate() {
        flows_so_far += booked;
        pnl.push(equity - start - flows_so_far);
        let current = if point == 0 {
            1.0
        } else if ruin.is_some() || equity - booked <= 0.0 {
            ruin = ruin.or(Some(point));
            0.0
        } else if booked == 0.0 {
            base_index * equity / base_equity
        } else if equity <= 0.0 {
            return Err(MetricsError::NoCapitalAfterFlow { index: point });
        } else {
            let after_flow = previous_index * (equity - booked) / previous_equity;
            (base_index, base_equity) = (after_flow, equity);
            after_flow
        };
        index.push(current);
        (previous_index, previous_equity) = (current, equity);
    }
    Ok(Wealth {
        index,
        pnl,
        flow,
        ruin,
    })
}

/// Per-session returns of a wealth index whose points are stamped `ts` (definitions in
/// the module docs). `ts` and `wealth.index` must have equal lengths; extra values of
/// the longer one are ignored. An index with fewer than two points has no session.
pub fn session_returns(ts: &[TsNanos], wealth: &Wealth) -> SessionReturns {
    let mut out = SessionReturns {
        days: Vec::new(),
        returns: Vec::new(),
    };
    let mut base = 1.0;
    let last_point = wealth.ruin.unwrap_or(usize::MAX);
    let points: Vec<(Day, f64)> = ts
        .iter()
        .zip(&wealth.index)
        .skip(1)
        .map(|(&t, &w)| (ist_day(t), w))
        .collect();
    for (position, &(day, index)) in points.iter().enumerate() {
        let next_day = points.get(position + 1).map(|&(next, _)| next);
        let ends_session = next_day != Some(day) || position + 1 == last_point;
        if !ends_session {
            continue;
        }
        out.days.push(day);
        out.returns.push(index / base - 1.0);
        base = index;
        if position + 1 >= last_point {
            break;
        }
    }
    out
}
