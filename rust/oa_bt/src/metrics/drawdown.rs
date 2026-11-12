//! Drawdowns of a wealth index: the series, the deepest episode, the longest episode
//! and the largest fall of cumulative P&L in rupees.
//!
//! Definitions, on every point of the series (an intraday dip that recovered by the
//! close counts). The running maximum starts at the opening, where the wealth index is
//! 1 and the cumulative P&L is 0 (`curve`): with an opening wealth a first point below
//! it is already in a drawdown, and without one the first point is the opening (index
//! exactly 1, P&L 0) and the first peak.
//!
//! - Drawdown `D_i = W_i / max(1, W_0..W_i) - 1`, a fraction from 0 down to -1.
//! - An episode starts at a peak, the last point at the running maximum before the
//!   index falls below it (the opening when no point has reached it yet), and ends at
//!   its recovery, the first later point at which the index is back at or above the
//!   peak. An episode still open at the last point is unrecovered. Its trough is its
//!   deepest point (the first one on a tie) and its depth is the drawdown there.
//! - Duration: from the peak to the recovery, or to the last point when unrecovered,
//!   in nanoseconds of elapsed time. An episode that starts at the opening is timed from
//!   the opening time when the caller gives one (the instant the run opened with its
//!   capital), and otherwise from the first point.
//! - Maximum drawdown: the episode with the lowest depth (the first one on a tie).
//! - Longest drawdown: the episode with the longest duration (the first one on a tie).
//! - Episodes: every episode in the order they start, the last one possibly
//!   unrecovered; the drawdown details table ranks them.
//! - Maximum drawdown amount: the largest fall of cumulative P&L from its running
//!   peak (0 at the opening), in rupees (cash flows excluded), zero or negative like
//!   every drawdown: a fall of 500 rupees is -500.

use crate::model::TsNanos;

/// The wealth index at the opening, where every running maximum starts.
const OPENING_INDEX: f64 = 1.0;
/// Cumulative P&L at the opening, in rupees.
const OPENING_PNL: f64 = 0.0;

/// One drawdown episode. Indexes are positions in the series.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrawdownEpisode {
    /// The last point at the running maximum before the fall, or `None` when the peak
    /// is the opening (the series fell below its opening wealth at its first point).
    pub peak: Option<usize>,
    /// The deepest point of the episode.
    pub trough: usize,
    /// The first point back at or above the peak, or `None` when unrecovered.
    pub recovery: Option<usize>,
    /// Drawdown at the trough, a fraction in `[-1, 0)`.
    pub depth: f64,
    /// Elapsed time from the peak (the opening time, or the first point without one,
    /// when the peak is the opening) to the recovery or the last point, in nanoseconds.
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
    /// Every episode, in the order they start.
    pub episodes: Vec<DrawdownEpisode>,
    /// The largest fall of cumulative P&L from its running peak, in rupees, zero or
    /// negative.
    pub max_amount: f64,
}

struct Tracker<'a> {
    ts: &'a [TsNanos],
    opening_time: Option<TsNanos>,
    max: Option<DrawdownEpisode>,
    longest: Option<DrawdownEpisode>,
    episodes: Vec<DrawdownEpisode>,
}

impl Tracker<'_> {
    fn close(&mut self, mut episode: DrawdownEpisode, end: usize) {
        let at = |i: usize| self.ts.get(i).copied().unwrap_or(0);
        let start = match episode.peak {
            Some(peak) => at(peak),
            None => self.opening_time.unwrap_or_else(|| at(0)),
        };
        episode.duration = at(end).saturating_sub(start);
        if self.max.map_or(true, |max| episode.depth < max.depth) {
            self.max = Some(episode);
        }
        if self
            .longest
            .map_or(true, |longest| episode.duration > longest.duration)
        {
            self.longest = Some(episode);
        }
        self.episodes.push(episode);
    }
}

/// The drawdowns of the wealth index `wealth` stamped `ts`, and the largest rupee fall
/// of the cumulative P&L `pnl` (definitions in the module docs). Both are measured from
/// the opening (index 1, P&L 0), as `curve::wealth` returns them. `opening_time` is the
/// UTC nanosecond instant of the opening, which times an episode that starts there;
/// `None` times it from the first point. The three slices are read up to the shortest
/// of them. An empty series has no drawdown.
pub fn drawdowns(
    ts: &[TsNanos],
    wealth: &[f64],
    pnl: &[f64],
    opening_time: Option<TsNanos>,
) -> Drawdowns {
    let n = ts.len().min(wealth.len());
    let mut series = Vec::with_capacity(n);
    let mut tracker = Tracker {
        ts,
        opening_time,
        max: None,
        longest: None,
        episodes: Vec::new(),
    };
    let mut open: Option<DrawdownEpisode> = None;
    let (mut peak, mut peak_index) = (OPENING_INDEX, None);
    for (point, &index) in wealth.iter().take(n).enumerate() {
        if index >= peak {
            if let Some(mut episode) = open.take() {
                episode.recovery = Some(point);
                tracker.close(episode, point);
            }
            (peak, peak_index) = (index, Some(point));
            series.push(0.0);
            continue;
        }
        // The running maximum never falls below the opening's index of 1.
        let depth = index / peak - 1.0;
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
        episodes: tracker.episodes,
        max_amount: max_fall(pnl),
    }
}

/// The largest fall of a cumulative P&L series from its running peak, which starts at
/// the opening's 0, as a negative amount; zero for a series that never falls below its
/// peak or is empty.
fn max_fall(values: &[f64]) -> f64 {
    let mut peak = OPENING_PNL;
    let mut fall: f64 = 0.0;
    for &value in values {
        peak = peak.max(value);
        fall = fall.min(value - peak);
    }
    fall
}
