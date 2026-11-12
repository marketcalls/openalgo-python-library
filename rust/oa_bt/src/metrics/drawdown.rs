//! Drawdowns of a wealth index: the series, the deepest episode, the longest episode
//! and the largest fall of cumulative P&L in rupees.
//!
//! Definitions, on every point of the series (an intraday dip that recovered by the
//! close counts):
//!
//! - Drawdown `D_i = W_i / max(W_0..W_i) - 1`, a fraction from 0 down to -1.
//! - An episode starts at a peak, the last point at the running maximum before the
//!   index falls below it, and ends at its recovery, the first later point at which
//!   the index is back at or above the peak. An episode still open at the last point
//!   is unrecovered. Its trough is its deepest point (the first one on a tie) and its
//!   depth is the drawdown there.
//! - Duration: from the peak to the recovery, or to the last point when unrecovered,
//!   in nanoseconds of elapsed time.
//! - Maximum drawdown: the episode with the lowest depth (the first one on a tie).
//! - Longest drawdown: the episode with the longest duration (the first one on a tie).
//! - Maximum drawdown amount: the largest fall of cumulative P&L from its running
//!   peak, in rupees (cash flows excluded), zero or positive.

use crate::model::TsNanos;

/// One drawdown episode. Indexes are positions in the series.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrawdownEpisode {
    /// The last point at the running maximum before the fall.
    pub peak: usize,
    /// The deepest point of the episode.
    pub trough: usize,
    /// The first point back at or above the peak, or `None` when unrecovered.
    pub recovery: Option<usize>,
    /// Drawdown at the trough, a fraction in `[-1, 0)`.
    pub depth: f64,
    /// Elapsed time from the peak to the recovery or the last point, in nanoseconds.
    pub duration: TsNanos,
}

/// The drawdowns of a series (definitions in the module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct Drawdowns {
    /// Drawdown at each point, a fraction in `[-1, 0]`.
    pub series: Vec<f64>,
    /// The deepest episode, or `None` when the index never falls below its peak.
    pub max: Option<DrawdownEpisode>,
    /// The longest episode, or `None` when the index never falls below its peak.
    pub longest: Option<DrawdownEpisode>,
    /// The largest fall of cumulative P&L from its running peak, in rupees.
    pub max_amount: f64,
}

struct Tracker<'a> {
    ts: &'a [TsNanos],
    max: Option<DrawdownEpisode>,
    longest: Option<DrawdownEpisode>,
}

impl Tracker<'_> {
    fn close(&mut self, mut episode: DrawdownEpisode, end: usize) {
        let at = |i: usize| self.ts.get(i).copied().unwrap_or(0);
        episode.duration = at(end).saturating_sub(at(episode.peak));
        if self.max.map_or(true, |max| episode.depth < max.depth) {
            self.max = Some(episode);
        }
        if self
            .longest
            .map_or(true, |longest| episode.duration > longest.duration)
        {
            self.longest = Some(episode);
        }
    }
}

/// The drawdowns of the wealth index `wealth` stamped `ts`, and the largest rupee fall
/// of the cumulative P&L `pnl` (definitions in the module docs). The three slices are
/// read up to the shortest of them. An empty series has no drawdown.
pub fn drawdowns(ts: &[TsNanos], wealth: &[f64], pnl: &[f64]) -> Drawdowns {
    let n = ts.len().min(wealth.len());
    let mut series = Vec::with_capacity(n);
    let mut tracker = Tracker {
        ts,
        max: None,
        longest: None,
    };
    let mut open: Option<DrawdownEpisode> = None;
    let (mut peak, mut peak_index) = (f64::NEG_INFINITY, 0);
    for (point, &index) in wealth.iter().take(n).enumerate() {
        if index >= peak {
            if let Some(mut episode) = open.take() {
                episode.recovery = Some(point);
                tracker.close(episode, point);
            }
            (peak, peak_index) = (index, point);
            series.push(0.0);
            continue;
        }
        let depth = if peak > 0.0 { index / peak - 1.0 } else { 0.0 };
        series.push(depth);
        let episode = open.get_or_insert(DrawdownEpisode {
            peak: peak_index,
            trough: point,
            recovery: None,
            depth,
            duration: 0,
        });
        if depth < episode.depth {
            (episode.trough, episode.depth) = (point, depth);
        }
    }
    if let Some(episode) = open {
        tracker.close(episode, n.saturating_sub(1));
    }
    Drawdowns {
        series,
        max: tracker.max,
        longest: tracker.longest,
        max_amount: max_fall(pnl),
    }
}

/// The largest fall of a series from its running peak; zero for a series that never
/// falls or is empty.
fn max_fall(values: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut fall: f64 = 0.0;
    for &value in values {
        peak = peak.max(value);
        fall = fall.max(peak - value);
    }
    fall
}
