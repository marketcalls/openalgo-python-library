//! Zero-dependency Rust core for OpenAlgo technical indicators.
//!
//! Every function here is a byte-for-byte port of the corresponding legacy
//! `@njit` kernel in `openalgo/indicators/utils.py`. The goal is exact numerical
//! parity: same seeding, same NaN placement, same accumulation order, so that the
//! `from openalgo import ta` public API returns identical values after the backend
//! swap from numba to Rust.
//!
//! Convention: `f64::NAN` represents the numpy `np.nan` warm-up region. Boolean
//! kernels return `Vec<bool>` (mapped to numpy bool arrays at the PyO3 layer).

#[inline]
fn nan_vec(n: usize) -> Vec<f64> {
    vec![f64::NAN; n]
}

#[inline]
fn max3(a: f64, b: f64, c: f64) -> f64 {
    let m = if a > b { a } else { b };
    if m > c {
        m
    } else {
        c
    }
}

// ============================================================================
// Rolling reductions
// ============================================================================

/// Simple Moving Average — O(n) rolling sum. NaN for the first `period-1` slots.
pub fn sma(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = nan_vec(n);
    if period == 0 || n < period {
        return result;
    }
    let mut rolling = 0.0;
    for &x in data.iter().take(period) {
        rolling += x;
    }
    result[period - 1] = rolling / period as f64;
    for i in period..n {
        // Match the reference left-to-right association exactly: (r + new) - old.
        rolling = rolling + data[i] - data[i - period];
        result[i] = rolling / period as f64;
    }
    result
}

/// Rolling sum over `period`. NaN for the first `period-1` slots.
pub fn rolling_sum(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = nan_vec(n);
    if period == 0 || n < period {
        return result;
    }
    let mut rolling = 0.0;
    for &x in data.iter().take(period) {
        rolling += x;
    }
    result[period - 1] = rolling;
    for i in period..n {
        rolling = rolling + data[i] - data[i - period];
        result[i] = rolling;
    }
    result
}

/// Population rolling variance (sumsq/period - mean^2). NaN warm-up.
pub fn rolling_variance(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = nan_vec(n);
    if period == 0 || n < period {
        return result;
    }
    let p = period as f64;
    let mut rsum = 0.0;
    let mut rsq = 0.0;
    for &x in data.iter().take(period) {
        rsum += x;
        rsq += x * x;
    }
    let mean = rsum / p;
    result[period - 1] = (rsq / p) - mean * mean;
    for i in period..n {
        let old = data[i - period];
        let new = data[i];
        rsum = rsum + new - old;
        rsq = rsq + new * new - old * old;
        let mean = rsum / p;
        result[i] = (rsq / p) - mean * mean;
    }
    result
}

/// Population rolling standard deviation = sqrt(max(0, variance)). NaN warm-up.
pub fn stdev(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = nan_vec(n);
    if period == 0 || n < period {
        return result;
    }
    let p = period as f64;
    let mut rsum = 0.0;
    let mut rsq = 0.0;
    for &x in data.iter().take(period) {
        rsum += x;
        rsq += x * x;
    }
    let mean = rsum / p;
    result[period - 1] = (rsq / p - mean * mean).max(0.0).sqrt();
    for i in period..n {
        let old = data[i - period];
        let new = data[i];
        rsum = rsum + new - old;
        rsq = rsq + new * new - old * old;
        let mean = rsum / p;
        result[i] = (rsq / p - mean * mean).max(0.0).sqrt();
    }
    result
}

// ============================================================================
// Moving averages
// ============================================================================

/// Exponential Moving Average — first-value seed, alpha = 2/(period+1), full length.
pub fn ema(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![0.0f64; n];
    if n == 0 {
        return result;
    }
    let alpha = 2.0 / (period as f64 + 1.0);
    result[0] = data[0];
    for i in 1..n {
        result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
    }
    result
}

/// Weighted Moving Average — linear weights 1..period. NaN for first `period-1`.
pub fn wma(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = nan_vec(n);
    if period == 0 || n < period {
        return result;
    }
    let weight_sum = (period * (period + 1) / 2) as f64;
    let p = period as f64;
    // O(n) rolling: maintain window sum and weighted sum (weights 1..period).
    // wsum_i = wsum_{i-1} + period*x_new - sum_{i-1}; then slide sum.
    let mut sum = 0.0;
    let mut wsum = 0.0;
    for j in 0..period {
        sum += data[j];
        wsum += data[j] * (j + 1) as f64;
    }
    result[period - 1] = wsum / weight_sum;
    for i in period..n {
        wsum = wsum + p * data[i] - sum;
        sum = sum + data[i] - data[i - period];
        result[i] = wsum / weight_sum;
    }
    result
}

/// SMA-seeded EMA: alpha = 2/(period+1), seed = SMA at index period-1, NaN warm-up.
/// (Distinct from `ema` which first-value-seeds, and `ema_wilder` which uses 1/period.)
pub fn ema_sma(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut r = nan_vec(n);
    if period == 0 || n < period {
        return r;
    }
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut s = 0.0;
    for &x in data.iter().take(period) {
        s += x;
    }
    r[period - 1] = s / period as f64;
    for i in period..n {
        r[i] = alpha * data[i] + (1.0 - alpha) * r[i - 1];
    }
    r
}

/// Wilder EMA (alpha = 1/period), SMA-seeded from the first valid index. NaN warm-up.
/// NaN inputs after the seed carry the previous value forward.
pub fn ema_wilder(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = nan_vec(n);
    if period == 0 {
        return result;
    }
    let mut first_valid = 0usize;
    while first_valid < n && data[first_valid].is_nan() {
        first_valid += 1;
    }
    if first_valid + period > n {
        return result;
    }
    let mut sum_val = 0.0;
    for i in first_valid..first_valid + period {
        if data[i].is_nan() {
            return result;
        }
        sum_val += data[i];
    }
    let start = first_valid + period - 1;
    result[start] = sum_val / period as f64;
    let pm1 = (period - 1) as f64;
    let p = period as f64;
    for i in start + 1..n {
        if data[i].is_nan() {
            result[i] = result[i - 1];
        } else {
            result[i] = (result[i - 1] * pm1 + data[i]) / p;
        }
    }
    result
}

// ============================================================================
// True range / ATR
// ============================================================================

/// True Range. tr[0] = high[0]-low[0]; thereafter max(h-l, |h-c[-1]|, |l-c[-1]|).
pub fn true_range(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let n = high.len();
    let mut tr = vec![0.0f64; n];
    if n == 0 {
        return tr;
    }
    tr[0] = high[0] - low[0];
    for i in 1..n {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        tr[i] = max3(hl, hc, lc);
    }
    tr
}

/// ATR with Wilder smoothing. Seed = simple average of first `period` TRs.
pub fn atr_wilder(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = high.len();
    let tr = true_range(high, low, close);
    let mut atr = nan_vec(n);
    if period == 0 || n < period {
        return atr;
    }
    let mut sum_tr = 0.0;
    for &t in tr.iter().take(period) {
        sum_tr += t;
    }
    atr[period - 1] = sum_tr / period as f64;
    let pm1 = (period - 1) as f64;
    let p = period as f64;
    for i in period..n {
        atr[i] = (atr[i - 1] * pm1 + tr[i]) / p;
    }
    atr
}

/// ATR using a simple moving average of True Range.
pub fn atr_sma(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let tr = true_range(high, low, close);
    sma(&tr, period)
}

// ============================================================================
// Differences
// ============================================================================

/// `data[i] - data[i-length]`. NaN for the first `length` slots.
pub fn change(data: &[f64], length: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = nan_vec(n);
    for i in length..n {
        result[i] = data[i] - data[i - length];
    }
    result
}

/// Rate of change as a percent: ((d[i]-d[i-length])/d[i-length])*100, NaN if divisor==0.
pub fn roc(data: &[f64], length: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = nan_vec(n);
    for i in length..n {
        if data[i - length] != 0.0 {
            result[i] = ((data[i] - data[i - length]) / data[i - length]) * 100.0;
        }
    }
    result
}

// ============================================================================
// Rolling extrema (monotonic deque, O(n))
// ============================================================================

/// Monotonic-deque rolling extremum over `period`, O(n). `want_max` selects max vs
/// min. The deque of indices lives in a small power-of-two ring buffer (capacity
/// >= period), indexed by masked monotonic head/tail counters - no heap alloc per
/// op, no modulo, cache-resident. `want_max` is loop-invariant (predicted perfectly).
#[inline]
fn _roll_extreme(data: &[f64], period: usize, want_max: bool) -> Vec<f64> {
    let n = data.len();
    let mut result = nan_vec(n);
    if period == 0 {
        return result;
    }
    let cap = period.next_power_of_two();
    let mask = cap - 1;
    let mut ring: Vec<usize> = vec![0usize; cap];
    let mut head = 0usize; // front counter
    let mut tail = 0usize; // back counter (len = tail - head)
    for i in 0..n {
        if head < tail && ring[head & mask] + period <= i {
            head += 1;
        }
        let x = data[i];
        while tail > head {
            let b = ring[(tail - 1) & mask];
            let drop = if want_max { data[b] <= x } else { data[b] >= x };
            if drop {
                tail -= 1;
            } else {
                break;
            }
        }
        ring[tail & mask] = i;
        tail += 1;
        if i + 1 >= period {
            result[i] = data[ring[head & mask]];
        }
    }
    result
}

/// Rolling maximum over `period`. NaN for the first `period-1` slots.
pub fn highest(data: &[f64], period: usize) -> Vec<f64> {
    _roll_extreme(data, period, true)
}

/// Rolling minimum over `period`. NaN for the first `period-1` slots.
pub fn lowest(data: &[f64], period: usize) -> Vec<f64> {
    _roll_extreme(data, period, false)
}

// ============================================================================
// Volume / adaptive kernels
// ============================================================================

/// Volume Weighted Moving Average. Falls back to price when window volume is 0.
pub fn vwma(data: &[f64], volume: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = nan_vec(n);
    if period == 0 || n < period {
        return result;
    }
    let mut sum_pv = 0.0;
    let mut sum_v = 0.0;
    for i in 0..period {
        sum_pv += data[i] * volume[i];
        sum_v += volume[i];
    }
    result[period - 1] = if sum_v > 0.0 {
        sum_pv / sum_v
    } else {
        data[period - 1]
    };
    for i in period..n {
        let new_pv = data[i] * volume[i];
        let old_pv = data[i - period] * volume[i - period];
        sum_pv = sum_pv + new_pv - old_pv;
        sum_v = sum_v + volume[i] - volume[i - period];
        result[i] = if sum_v > 0.0 { sum_pv / sum_v } else { data[i] };
    }
    result
}

/// Chande Momentum Oscillator (rolling sums of up/down changes).
pub fn cmo(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = nan_vec(n);
    if n < period + 1 {
        return result;
    }
    let mut changes = vec![0.0f64; n - 1];
    for i in 1..n {
        changes[i - 1] = data[i] - data[i - 1];
    }
    let mut up = 0.0;
    let mut down = 0.0;
    for &ch in changes.iter().take(period) {
        if ch > 0.0 {
            up += ch;
        } else if ch < 0.0 {
            down += ch.abs();
        }
    }
    let total = up + down;
    result[period] = if total > 0.0 {
        ((up - down) / total) * 100.0
    } else {
        0.0
    };
    for i in period + 1..n {
        let old = changes[i - period - 1];
        if old > 0.0 {
            up -= old;
        } else if old < 0.0 {
            down -= old.abs();
        }
        let new = changes[i - 1];
        if new > 0.0 {
            up += new;
        } else if new < 0.0 {
            down += new.abs();
        }
        let total = up + down;
        result[i] = if total > 0.0 {
            ((up - down) / total) * 100.0
        } else {
            0.0
        };
    }
    result
}

/// Kaufman Adaptive MA. Seeds result[period]=data[period]; ER over `period`.
pub fn kama(data: &[f64], period: usize, fast_sc: f64, slow_sc: f64) -> Vec<f64> {
    let n = data.len();
    let mut result = nan_vec(n);
    if n < period + 1 {
        return result;
    }
    let fast_alpha = 2.0 / (fast_sc + 1.0);
    let slow_alpha = 2.0 / (slow_sc + 1.0);
    result[period] = data[period];
    let mut vol = 0.0;
    for i in 1..period + 1 {
        vol += (data[i] - data[i - 1]).abs();
    }
    for i in period + 1..n {
        let direction = (data[i] - data[i - period]).abs();
        vol -= (data[i - period] - data[i - period - 1]).abs();
        vol += (data[i] - data[i - 1]).abs();
        let er = if vol > 0.0 { direction / vol } else { 0.0 };
        let sc = (er * (fast_alpha - slow_alpha) + slow_alpha).powi(2);
        result[i] = result[i - 1] + sc * (data[i] - result[i - 1]);
    }
    result
}

/// Arnaud Legoux MA: Gaussian-weighted window. m = offset*(period-1), s = period/sigma.
pub fn alma(data: &[f64], period: usize, offset: f64, sigma: f64) -> Vec<f64> {
    let n = data.len();
    let mut result = nan_vec(n);
    if period == 0 || n < period {
        return result;
    }
    let m = offset * (period as f64 - 1.0);
    let s = period as f64 / sigma;
    let mut weights = vec![0.0f64; period];
    for (i, w) in weights.iter_mut().enumerate() {
        *w = (-((i as f64 - m).powi(2)) / (2.0 * s * s)).exp();
    }
    let wsum: f64 = weights.iter().sum();
    for w in weights.iter_mut() {
        *w /= wsum;
    }
    for i in period - 1..n {
        let mut acc = 0.0;
        for j in 0..period {
            acc += weights[j] * data[i + 1 - period + j];
        }
        result[i] = acc;
    }
    result
}

/// McGinley Dynamic. Seed = mean(first period); MD += (src-MD)/(period*(src/MD)^4).
pub fn mcginley(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = nan_vec(n);
    if period == 0 || n < period {
        return result;
    }
    let mut s = 0.0;
    for &x in data.iter().take(period) {
        s += x;
    }
    result[period - 1] = s / period as f64;
    for i in period..n {
        if result[i - 1] != 0.0 {
            let ratio = data[i] / result[i - 1];
            let factor = period as f64 * ratio.powi(4);
            result[i] = result[i - 1] + (data[i] - result[i - 1]) / factor;
        } else {
            result[i] = data[i];
        }
    }
    result
}

/// VIDYA: CMO-scaled EMA. Inline CMO over `period`; seed result[period]=data[period].
pub fn vidya(data: &[f64], period: usize, alpha: f64) -> Vec<f64> {
    let n = data.len();
    let mut result = nan_vec(n);
    if period == 0 || n < period + 1 {
        return result;
    }
    let mut cmo = nan_vec(n);
    for i in period..n {
        let mut gains = 0.0;
        let mut losses = 0.0;
        for j in i + 1 - period..i + 1 {
            if j > 0 {
                let diff = data[j] - data[j - 1];
                if diff > 0.0 {
                    gains += diff;
                } else if diff < 0.0 {
                    losses += -diff;
                }
            }
        }
        cmo[i] = if gains + losses != 0.0 {
            100.0 * (gains - losses) / (gains + losses)
        } else {
            0.0
        };
    }
    result[period] = data[period];
    for i in period + 1..n {
        if !cmo[i].is_nan() {
            let sc = alpha * cmo[i].abs() / 100.0;
            result[i] = result[i - 1] + sc * (data[i] - result[i - 1]);
        } else {
            result[i] = result[i - 1];
        }
    }
    result
}

/// KAMA (TradingView variant). er = |chg(length)| / sum(|chg(1)|, length); each bar
/// recomputes the volatility window. prev = src when first/NaN (nz). Matches the
/// `_calculate_kama_tv` reference bit-for-bit.
pub fn kama_tv(data: &[f64], length: usize, fast_length: usize, slow_length: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = nan_vec(n);
    if length == 0 || n < length + 1 {
        return result;
    }
    let fast_alpha = 2.0 / (fast_length as f64 + 1.0);
    let slow_alpha = 2.0 / (slow_length as f64 + 1.0);
    for i in length..n {
        let mom = (data[i] - data[i - length]).abs();
        let mut volatility = 0.0;
        for j in i + 1 - length..i + 1 {
            if j > 0 {
                volatility += (data[j] - data[j - 1]).abs();
            }
        }
        let er = if volatility != 0.0 { mom / volatility } else { 0.0 };
        let alpha = (er * (fast_alpha - slow_alpha) + slow_alpha).powi(2);
        let prev = if i == length || result[i - 1].is_nan() {
            data[i]
        } else {
            result[i - 1]
        };
        result[i] = alpha * data[i] + (1.0 - alpha) * prev;
    }
    result
}

/// Supertrend (TradingView band-flip). Returns (supertrend, direction) where
/// direction is -1 uptrend / +1 downtrend. ATR via Wilder smoothing.
pub fn supertrend(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    multiplier: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = close.len();
    let mut st = nan_vec(n);
    let mut dir = nan_vec(n);
    if period == 0 || n == 0 {
        return (st, dir);
    }
    let atr = atr_wilder(high, low, close, period);
    let mut ub = vec![0.0f64; n];
    let mut lb = vec![0.0f64; n];
    for i in 0..n {
        let hl = (high[i] + low[i]) / 2.0;
        ub[i] = hl + multiplier * atr[i];
        lb[i] = hl - multiplier * atr[i];
    }
    let mut fu = nan_vec(n);
    let mut fl = nan_vec(n);
    let fv = period - 1;
    if fv >= n {
        return (st, dir);
    }
    fu[fv] = ub[fv];
    fl[fv] = lb[fv];
    dir[fv] = 1.0;
    st[fv] = fu[fv];
    for i in fv + 1..n {
        fl[i] = if lb[i] > fl[i - 1] || close[i - 1] < fl[i - 1] {
            lb[i]
        } else {
            fl[i - 1]
        };
        fu[i] = if ub[i] < fu[i - 1] || close[i - 1] > fu[i - 1] {
            ub[i]
        } else {
            fu[i - 1]
        };
        if st[i - 1] == fu[i - 1] {
            dir[i] = if close[i] > fu[i] { -1.0 } else { 1.0 };
        } else {
            dir[i] = if close[i] < fl[i] { 1.0 } else { -1.0 };
        }
        st[i] = if dir[i] == -1.0 { fl[i] } else { fu[i] };
    }
    (st, dir)
}

/// Chande Kroll Stop. Returns (long_stop, short_stop). ATR via Wilder smoothing;
/// first stops over `p`, then nested extrema of first stops over `q`.
pub fn chande_kroll_stop(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p: usize,
    x: f64,
    q: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = close.len();
    let mut long_stop = nan_vec(n);
    let mut short_stop = nan_vec(n);
    if p == 0 || q == 0 || n < p {
        return (long_stop, short_stop);
    }
    let atr = atr_wilder(high, low, close, p);
    let mut fhs = nan_vec(n);
    let mut fls = nan_vec(n);
    for i in p - 1..n {
        let mut hh = high[i + 1 - p];
        let mut ll = low[i + 1 - p];
        for k in i + 1 - p..i + 1 {
            if high[k] > hh {
                hh = high[k];
            }
            if low[k] < ll {
                ll = low[k];
            }
        }
        fhs[i] = hh - x * atr[i];
        fls[i] = ll + x * atr[i];
    }
    let start = p + q - 2;
    for i in start..n {
        let qs = i + 1 - q;
        let mut mx = f64::NEG_INFINITY;
        let mut any_h = false;
        let mut mn = f64::INFINITY;
        let mut any_l = false;
        for k in qs..i + 1 {
            if !fhs[k].is_nan() {
                if fhs[k] > mx {
                    mx = fhs[k];
                }
                any_h = true;
            }
            if !fls[k].is_nan() {
                if fls[k] < mn {
                    mn = fls[k];
                }
                any_l = true;
            }
        }
        if any_h {
            short_stop[i] = mx;
        }
        if any_l {
            long_stop[i] = mn;
        }
    }
    (long_stop, short_stop)
}

/// FRAMA (TradingView fractal adaptive MA on hl2, with trailing SMA(5) smoothing).
/// Uses log/exp so parity is transcendental-tolerance, not bit-exact.
pub fn frama(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    let n = high.len();
    let mut filt = nan_vec(n);
    let price: Vec<f64> = (0..n).map(|i| (high[i] + low[i]) / 2.0).collect();
    if n > 0 {
        filt[0] = price[0];
    }
    let half = period / 2;
    let ln2 = 2.0_f64.ln();
    for i in 1..n {
        if i >= period && half > 0 {
            let mut hmax = high[i + 1 - period];
            let mut lmin = low[i + 1 - period];
            for k in i + 1 - period..i + 1 {
                if high[k] > hmax {
                    hmax = high[k];
                }
                if low[k] < lmin {
                    lmin = low[k];
                }
            }
            let n3 = (hmax - lmin) / period as f64;
            let mut hh = high[i];
            let mut ll = low[i];
            for count in 0..half {
                let idx = i - count;
                if high[idx] > hh {
                    hh = high[idx];
                }
                if low[idx] < ll {
                    ll = low[idx];
                }
            }
            let n1 = (hh - ll) / half as f64;
            let mut hh2 = high[i - half];
            let mut ll2 = low[i - half];
            for count in half..period {
                let idx = i - count;
                if high[idx] > hh2 {
                    hh2 = high[idx];
                }
                if low[idx] < ll2 {
                    ll2 = low[idx];
                }
            }
            let n2 = (hh2 - ll2) / half as f64;
            let dimen = if n1 > 0.0 && n2 > 0.0 && n3 > 0.0 {
                ((n1 + n2).ln() - n3.ln()) / ln2
            } else {
                1.0
            };
            let mut alpha = (-4.6 * (dimen - 1.0)).exp();
            alpha = alpha.min(1.0).max(0.01);
            filt[i] = alpha * price[i] + (1.0 - alpha) * filt[i - 1];
        } else {
            filt[i] = price[i];
        }
    }
    let mut smoothed = nan_vec(n);
    for i in 0..n {
        if i < period + 1 {
            smoothed[i] = price[i];
        } else if i >= 4 {
            let ws = i - 4;
            let mut s = 0.0;
            let mut cnt = 0usize;
            for j in ws..i + 1 {
                if !filt[j].is_nan() {
                    s += filt[j];
                    cnt += 1;
                }
            }
            smoothed[i] = if cnt > 0 { s / cnt as f64 } else { filt[i] };
        } else {
            smoothed[i] = filt[i];
        }
    }
    smoothed
}

/// Ulcer Index (running-peak drawdown RMS * 100) over `period`.
pub fn ulcer_index(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = nan_vec(n);
    if period == 0 || n < period {
        return result;
    }
    for i in period - 1..n {
        let mut ssd = 0.0;
        let mut peak = data[i + 1 - period];
        for j in 0..period {
            let idx = i + 1 - period + j;
            if data[idx] > peak {
                peak = data[idx];
            }
            if peak > 0.0 {
                let dd = (data[idx] - peak) / peak;
                ssd += dd * dd;
            }
        }
        result[i] = (ssd / period as f64).sqrt() * 100.0;
    }
    result
}

// ============================================================================
// Boolean signal kernels
// ============================================================================

/// series1 crosses above series2 at i.
pub fn crossover(s1: &[f64], s2: &[f64]) -> Vec<bool> {
    let n = s1.len();
    let mut result = vec![false; n];
    for i in 1..n {
        if !s1[i].is_nan()
            && !s2[i].is_nan()
            && !s1[i - 1].is_nan()
            && !s2[i - 1].is_nan()
            && s1[i] > s2[i]
            && s1[i - 1] <= s2[i - 1]
        {
            result[i] = true;
        }
    }
    result
}

/// series1 crosses below series2 at i.
pub fn crossunder(s1: &[f64], s2: &[f64]) -> Vec<bool> {
    let n = s1.len();
    let mut result = vec![false; n];
    for i in 1..n {
        if !s1[i].is_nan()
            && !s2[i].is_nan()
            && !s1[i - 1].is_nan()
            && !s2[i - 1].is_nan()
            && s1[i] < s2[i]
            && s1[i - 1] >= s2[i - 1]
        {
            result[i] = true;
        }
    }
    result
}

/// Either-direction cross.
pub fn cross(s1: &[f64], s2: &[f64]) -> Vec<bool> {
    let n = s1.len();
    let mut result = vec![false; n];
    for i in 1..n {
        if !s1[i].is_nan() && !s2[i].is_nan() && !s1[i - 1].is_nan() && !s2[i - 1].is_nan() {
            let over = s1[i] > s2[i] && s1[i - 1] <= s2[i - 1];
            let under = s1[i] < s2[i] && s1[i - 1] >= s2[i - 1];
            result[i] = over || under;
        }
    }
    result
}

/// data[i] > data[i-length].
pub fn rising(data: &[f64], length: usize) -> Vec<bool> {
    let n = data.len();
    let mut result = vec![false; n];
    for i in length..n {
        if !data[i].is_nan() && !data[i - length].is_nan() {
            result[i] = data[i] > data[i - length];
        }
    }
    result
}

/// data[i] < data[i-length].
pub fn falling(data: &[f64], length: usize) -> Vec<bool> {
    let n = data.len();
    let mut result = vec![false; n];
    for i in length..n {
        if !data[i].is_nan() && !data[i - length].is_nan() {
            result[i] = data[i] < data[i - length];
        }
    }
    result
}

/// Excess-removal latch: fire `primary` once, re-arm on `secondary`.
pub fn exrem(primary: &[bool], secondary: &[bool]) -> Vec<bool> {
    let n = primary.len();
    let mut result = vec![false; n];
    let mut active = false;
    for i in 0..n {
        if !active && primary[i] {
            result[i] = true;
            active = true;
        } else if secondary[i] {
            active = false;
        }
    }
    result
}

/// Toggle latch: set on `primary`, clear on `secondary`, hold otherwise.
pub fn flip(primary: &[bool], secondary: &[bool]) -> Vec<bool> {
    let n = primary.len();
    let mut result = vec![false; n];
    let mut active = false;
    for i in 0..n {
        if primary[i] {
            active = true;
        } else if secondary[i] {
            active = false;
        }
        result[i] = active;
    }
    result
}

// ============================================================================
// Momentum / oscillator indicators
// ============================================================================

/// Wilder RSI. NaN until index `period`; avg over first `period` deltas, then
/// Wilder smoothing. avg_loss == 0 -> 100.
pub fn rsi(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = nan_vec(n);
    if period == 0 || n < period + 1 {
        return result;
    }
    let p = period as f64;
    let pm1 = (period - 1) as f64;
    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;
    for i in 0..period {
        let d = data[i + 1] - data[i];
        if d > 0.0 {
            avg_gain += d;
        } else if d < 0.0 {
            avg_loss += -d;
        }
    }
    avg_gain /= p;
    avg_loss /= p;
    result[period] = if avg_loss == 0.0 {
        100.0
    } else {
        100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    };
    for i in period..n - 1 {
        let d = data[i + 1] - data[i];
        let gain = if d > 0.0 { d } else { 0.0 };
        let loss = if d < 0.0 { -d } else { 0.0 };
        avg_gain = (avg_gain * pm1 + gain) / p;
        avg_loss = (avg_loss * pm1 + loss) / p;
        result[i + 1] = if avg_loss == 0.0 {
            100.0
        } else {
            100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        };
    }
    result
}

/// Stochastic oscillator. Returns (slow_k, slow_d):
/// fast_k = 100*(c-ll)/(hh-ll) (50 when hh==ll); slow_k = SMA(fast_k, smooth_k);
/// slow_d = SMA(slow_k, d_period). hh/ll are rolling extrema over k_period.
pub fn stochastic(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    k_period: usize,
    smooth_k: usize,
    d_period: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = close.len();
    let mut slow_k = nan_vec(n);
    let mut slow_d = nan_vec(n);
    if k_period == 0 || smooth_k == 0 || d_period == 0 || n == 0 {
        return (slow_k, slow_d);
    }
    let hh = highest(high, k_period);
    let ll = lowest(low, k_period);
    let mut fast_k = nan_vec(n);
    let fk_start = k_period - 1;
    for i in fk_start..n {
        fast_k[i] = if hh[i] != ll[i] {
            100.0 * (close[i] - ll[i]) / (hh[i] - ll[i])
        } else {
            50.0
        };
    }
    let sk_start = fk_start + smooth_k - 1;
    if sk_start < n {
        let mut s = 0.0;
        for j in fk_start..fk_start + smooth_k {
            s += fast_k[j];
        }
        slow_k[sk_start] = s / smooth_k as f64;
        for i in sk_start + 1..n {
            s += fast_k[i] - fast_k[i - smooth_k];
            slow_k[i] = s / smooth_k as f64;
        }
    }
    let sd_start = sk_start + d_period - 1;
    if sd_start < n {
        let mut s = 0.0;
        for j in sk_start..sk_start + d_period {
            s += slow_k[j];
        }
        slow_d[sd_start] = s / d_period as f64;
        for i in sd_start + 1..n {
            s += slow_k[i] - slow_k[i - d_period];
            slow_d[i] = s / d_period as f64;
        }
    }
    (slow_k, slow_d)
}

/// Commodity Channel Index. TP=(h+l+c)/3; (TP-SMA(TP))/(0.015*meandev); 0 if meandev==0.
pub fn cci(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut out = nan_vec(n);
    if period == 0 || n < period {
        return out;
    }
    let tp: Vec<f64> = (0..n)
        .map(|i| (high[i] + low[i] + close[i]) / 3.0)
        .collect();
    let p = period as f64;
    let mut rsum = 0.0;
    for &t in tp.iter().take(period) {
        rsum += t;
    }
    for i in period - 1..n {
        if i > period - 1 {
            rsum = rsum + tp[i] - tp[i - period];
        }
        let sma_tp = rsum / p;
        let mut md = 0.0;
        for j in 0..period {
            md += (tp[i + 1 - period + j] - sma_tp).abs();
        }
        md /= p;
        out[i] = if md != 0.0 {
            (tp[i] - sma_tp) / (0.015 * md)
        } else {
            0.0
        };
    }
    out
}

/// Williams %R. -100*(hh-close)/(hh-ll) over `period`; -50 when hh==ll.
pub fn williams_r(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut out = nan_vec(n);
    if period == 0 {
        return out;
    }
    let hh = highest(high, period);
    let ll = lowest(low, period);
    for i in period - 1..n {
        out[i] = if hh[i] != ll[i] {
            -100.0 * (hh[i] - close[i]) / (hh[i] - ll[i])
        } else {
            -50.0
        };
    }
    out
}

// ============================================================================
// Volume indicators
// ============================================================================

/// Session-anchored VWAP. `starts[i] != 0` (or i==0) resets the running sums.
/// Returns (vwap, stdev) where stdev = sqrt(max(0, E[p^2] - vwap^2)) within session.
pub fn session_vwap(source: &[f64], volume: &[f64], starts: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = source.len();
    let mut vwap = nan_vec(n);
    let mut sd = nan_vec(n);
    let mut spv = 0.0;
    let mut sv = 0.0;
    let mut spv2 = 0.0;
    for i in 0..n {
        if starts[i] != 0.0 || i == 0 {
            spv = 0.0;
            sv = 0.0;
            spv2 = 0.0;
        }
        spv += source[i] * volume[i];
        sv += volume[i];
        spv2 += source[i] * source[i] * volume[i];
        if sv > 0.0 {
            vwap[i] = spv / sv;
            let variance = spv2 / sv - vwap[i].powf(2.0);
            sd[i] = variance.max(0.0).sqrt();
        } else {
            vwap[i] = source[i];
            sd[i] = 0.0;
        }
    }
    (vwap, sd)
}

/// Parabolic SAR. Returns (sar, trend) where trend is +1 up / -1 down.
pub fn sar(high: &[f64], low: &[f64], acceleration: f64, maximum: f64) -> (Vec<f64>, Vec<f64>) {
    let n = high.len();
    let mut sar = vec![0.0f64; n];
    let mut trend = vec![0.0f64; n];
    if n == 0 {
        return (sar, trend);
    }
    sar[0] = low[0];
    trend[0] = 1.0;
    let mut af = acceleration;
    let mut ep = high[0];
    for i in 1..n {
        let prev_sar = sar[i - 1];
        let prev_trend = trend[i - 1];
        sar[i] = prev_sar + af * (ep - prev_sar);
        if prev_trend == 1.0 {
            if low[i] <= sar[i] {
                trend[i] = -1.0;
                sar[i] = ep;
                ep = low[i];
                af = acceleration;
            } else {
                trend[i] = 1.0;
                if high[i] > ep {
                    ep = high[i];
                    af = (af + acceleration).min(maximum);
                }
                if i >= 2 {
                    sar[i] = sar[i].min(low[i - 1]).min(low[i - 2]);
                } else {
                    sar[i] = sar[i].min(low[i - 1]);
                }
            }
        } else if high[i] >= sar[i] {
            trend[i] = 1.0;
            sar[i] = ep;
            ep = high[i];
            af = acceleration;
        } else {
            trend[i] = -1.0;
            if low[i] < ep {
                ep = low[i];
                af = (af + acceleration).min(maximum);
            }
            if i >= 2 {
                sar[i] = sar[i].max(high[i - 1]).max(high[i - 2]);
            } else {
                sar[i] = sar[i].max(high[i - 1]);
            }
        }
    }
    (sar, trend)
}

/// On Balance Volume. obv[0]=0; cumulative sign(change)*volume where a flat close
/// (close==prev) contributes 0 -- matching TradingView ta.obv and TA-Lib OBV.
pub fn obv(close: &[f64], volume: &[f64]) -> Vec<f64> {
    let n = close.len();
    let mut r = vec![0.0f64; n];
    for i in 1..n {
        let sign = if close[i] > close[i - 1] {
            1.0
        } else if close[i] < close[i - 1] {
            -1.0
        } else {
            0.0
        };
        r[i] = r[i - 1] + sign * volume[i];
    }
    r
}

/// Accumulation/Distribution Line. Seed 0; cumulative money-flow volume.
pub fn adl(high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
    let n = close.len();
    let mut r = nan_vec(n);
    if n == 0 {
        return r;
    }
    r[0] = 0.0;
    for i in 1..n {
        let mfm = if high[i] != low[i] {
            ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
        } else {
            0.0
        };
        r[i] = r[i - 1] + mfm * volume[i];
    }
    r
}

/// Chaikin Money Flow: sum(MFV, period) / sum(volume, period) per window.
pub fn cmf(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut r = nan_vec(n);
    if period == 0 || n < period {
        return r;
    }
    for i in period - 1..n {
        let mut smfv = 0.0;
        let mut sv = 0.0;
        for j in 0..period {
            let idx = i + 1 - period + j;
            let mfm = if high[idx] != low[idx] {
                ((close[idx] - low[idx]) - (high[idx] - close[idx])) / (high[idx] - low[idx])
            } else {
                0.0
            };
            smfv += mfm * volume[idx];
            sv += volume[idx];
        }
        r[i] = if sv > 0.0 { smfv / sv } else { 0.0 };
    }
    r
}

/// Money Flow Index (volume-weighted RSI), rolling window of pos/neg money flow.
pub fn mfi(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut r = nan_vec(n);
    if period == 0 {
        return r;
    }
    let tp: Vec<f64> = (0..n)
        .map(|i| (high[i] + low[i] + close[i]) / 3.0)
        .collect();
    let mut pos = vec![0.0f64; n];
    let mut neg = vec![0.0f64; n];
    for i in 1..n {
        let rmf = tp[i] * volume[i];
        if tp[i] > tp[i - 1] {
            pos[i] = rmf;
        } else if tp[i] < tp[i - 1] {
            neg[i] = rmf;
        }
    }
    let mut ps = 0.0;
    let mut ns = 0.0;
    for i in 1..n {
        ps += pos[i];
        ns += neg[i];
        if i >= period {
            ps -= pos[i - period];
            ns -= neg[i - period];
        }
        if i >= period - 1 {
            r[i] = if ns == 0.0 {
                100.0
            } else {
                100.0 - 100.0 / (1.0 + ps / ns)
            };
        }
    }
    r
}

/// Raw Ease of Movement: divisor * change(hl2) * (high-low) / volume; 0 if vol/range<=0.
pub fn emv_raw(high: &[f64], low: &[f64], volume: &[f64], divisor: f64) -> Vec<f64> {
    let n = high.len();
    let mut r = nan_vec(n);
    for i in 1..n {
        let chl2 = (high[i] + low[i]) / 2.0 - (high[i - 1] + low[i - 1]) / 2.0;
        let hlr = high[i] - low[i];
        r[i] = if volume[i] > 0.0 && hlr > 0.0 {
            divisor * chl2 * hlr / volume[i]
        } else {
            0.0
        };
    }
    r
}

/// EMA (alpha=2/(p+1)) seeded at the first non-NaN value; NaN inputs after are skipped
/// (value held). Matches the FI/force-index EMA helper.
pub fn ema_first_valid(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut r = nan_vec(n);
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut fv = None;
    for i in 0..n {
        if !data[i].is_nan() {
            r[i] = data[i];
            fv = Some(i);
            break;
        }
    }
    let fv = match fv {
        Some(x) => x,
        None => return r,
    };
    for i in fv + 1..n {
        if !data[i].is_nan() {
            r[i] = alpha * data[i] + (1.0 - alpha) * r[i - 1];
        }
    }
    r
}

/// Balance of Power: (close-open)/(high-low) per bar; 0 when high==low.
pub fn bop(open_: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let n = close.len();
    let mut r = nan_vec(n);
    for i in 0..n {
        r[i] = if high[i] != low[i] {
            (close[i] - open_[i]) / (high[i] - low[i])
        } else {
            0.0
        };
    }
    r
}

/// Fisher Transform (TradingView). Input is the price source (hl2). Returns
/// (fisher, trigger). Recursive value/fisher with log; window extrema over `length`.
pub fn fisher(data: &[f64], length: usize) -> (Vec<f64>, Vec<f64>) {
    let n = data.len();
    let mut fish1 = nan_vec(n);
    let mut fish2 = nan_vec(n);
    if length == 0 || n < length {
        return (fish1, fish2);
    }
    let mut value = 0.0;
    let mut fish1_prev = 0.0;
    for i in length - 1..n {
        let ws = i + 1 - length;
        let mut hi = data[ws];
        let mut lo = data[ws];
        for k in ws..i + 1 {
            if data[k] > hi {
                hi = data[k];
            }
            if data[k] < lo {
                lo = data[k];
            }
        }
        if hi != lo {
            let normalized = (data[i] - lo) / (hi - lo) - 0.5;
            let new_value = 0.66 * normalized + 0.67 * value;
            value = if new_value > 0.99 {
                0.999
            } else if new_value < -0.99 {
                -0.999
            } else {
                new_value
            };
            let log_term = 0.5 * ((1.0 + value) / (1.0 - value)).ln();
            fish1[i] = log_term + 0.5 * fish1_prev;
            fish1_prev = fish1[i];
        } else {
            fish1[i] = fish1_prev;
        }
        fish2[i] = if i > length - 1 { fish1[i - 1] } else { 0.0 };
    }
    (fish1, fish2)
}

/// Connors RSI up/down streak length (positive=up run, negative=down run, 0=flat).
pub fn updown_streak(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut s = vec![0.0f64; n];
    for i in 1..n {
        if data[i] == data[i - 1] {
            s[i] = 0.0;
        } else if data[i] > data[i - 1] {
            s[i] = if s[i - 1] <= 0.0 { 1.0 } else { s[i - 1] + 1.0 };
        } else {
            s[i] = if s[i - 1] >= 0.0 { -1.0 } else { s[i - 1] - 1.0 };
        }
    }
    s
}

/// Percent rank: pct of the trailing `period` window strictly below the current value.
pub fn percent_rank(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut r = nan_vec(n);
    if period == 0 {
        return r;
    }
    for i in period - 1..n {
        let cur = data[i];
        let mut count = 0usize;
        for j in i + 1 - period..i + 1 {
            if data[j] < cur {
                count += 1;
            }
        }
        r[i] = (count as f64 / period as f64) * 100.0;
    }
    r
}

// ============================================================================
// Statistics (per-window OLS / correlation / beta)
// ============================================================================

#[inline]
fn _linreg_endval(y: &[f64], period: usize, sx: f64, den: f64, end_x: f64) -> f64 {
    let p = period as f64;
    let mut sy = 0.0;
    let mut sxy = 0.0;
    for (j, &v) in y.iter().enumerate() {
        sy += v;
        sxy += j as f64 * v;
    }
    if den != 0.0 {
        let slope = (p * sxy - sx * sy) / den;
        let intercept = (sy - slope * sx) / p;
        slope * end_x + intercept
    } else {
        y[period - 1]
    }
}

fn _xstats(period: usize) -> (f64, f64) {
    let mut sx = 0.0;
    let mut sx2 = 0.0;
    for j in 0..period {
        let xf = j as f64;
        sx += xf;
        sx2 += xf * xf;
    }
    (sx, sx2)
}

/// Per-window mean (naive sum / period). NaN if the window contains any NaN.
/// Matches `_win_mean` (np.mean over each window) within ~1e-14.
pub fn win_mean(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = nan_vec(n);
    if period == 0 || n < period {
        return out;
    }
    let p = period as f64;
    for i in period - 1..n {
        let w = &data[i + 1 - period..i + 1];
        let mut s = 0.0;
        let mut nan = false;
        for &v in w {
            if v.is_nan() {
                nan = true;
                break;
            }
            s += v;
        }
        if !nan {
            out[i] = s / p;
        }
    }
    out
}

/// Per-window population std = sqrt(mean((w-mean)^2)). NaN if the window has any NaN.
pub fn win_std(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = nan_vec(n);
    if period == 0 || n < period {
        return out;
    }
    let p = period as f64;
    for i in period - 1..n {
        let w = &data[i + 1 - period..i + 1];
        let mut s = 0.0;
        let mut nan = false;
        for &v in w {
            if v.is_nan() {
                nan = true;
                break;
            }
            s += v;
        }
        if nan {
            continue;
        }
        let m = s / p;
        let mut acc = 0.0;
        for &v in w {
            let d = v - m;
            acc += d * d;
        }
        out[i] = (acc / p).sqrt();
    }
    out
}

/// O(n) rolling OLS over `period`. Returns (slope, intercept) per window, NaN warm-up.
/// x is the integer grid 0..period-1 (so `sx`, `sx2` are constants). Sy and the
/// weighted sum (weights 1..period) are slid in O(1) per step, exactly like `wma`;
/// Sxy = wsum - Sy. Drift vs per-window recompute is ~1e-11 (same as the prior code).
fn _ols_roll(data: &[f64], period: usize) -> (Vec<f64>, Vec<f64>) {
    let n = data.len();
    let mut slope = nan_vec(n);
    let mut intercept = nan_vec(n);
    if period == 0 || n < period {
        return (slope, intercept);
    }
    let p = period as f64;
    let (sx, sx2) = _xstats(period);
    let den = p * sx2 - sx * sx;
    let mut sy = 0.0;
    let mut wsum = 0.0;
    for j in 0..period {
        sy += data[j];
        wsum += data[j] * (j + 1) as f64;
    }
    let mut emit = |idx: usize, sy: f64, wsum: f64| {
        if den != 0.0 {
            let sxy = wsum - sy; // Σ j*y over the window (weights 0..period-1)
            let m = (p * sxy - sx * sy) / den;
            slope[idx] = m;
            intercept[idx] = (sy - m * sx) / p;
        } else {
            // period == 1: regression of a single point is the point itself.
            slope[idx] = 0.0;
            intercept[idx] = data[idx];
        }
    };
    emit(period - 1, sy, wsum);
    for i in period..n {
        wsum = wsum + p * data[i] - sy;
        sy = sy + data[i] - data[i - period];
        emit(i, sy, wsum);
    }
    (slope, intercept)
}

/// Rolling linear-regression endpoint value (slope*(period-1)+intercept).
pub fn linreg(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = nan_vec(n);
    if period == 0 || n < period {
        return out;
    }
    let (slope, intercept) = _ols_roll(data, period);
    let ex = (period - 1) as f64;
    for i in period - 1..n {
        out[i] = slope[i] * ex + intercept[i];
    }
    out
}

/// Time Series Forecast (linreg projected one bar ahead: slope*period+intercept).
pub fn tsf(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = nan_vec(n);
    if period == 0 || n < period {
        return out;
    }
    let (slope, intercept) = _ols_roll(data, period);
    let ex = period as f64;
    for i in period - 1..n {
        out[i] = slope[i] * ex + intercept[i];
    }
    out
}

/// Linear-regression slope (TradingView): (endval(cur) - endval(prev)) / interval.
pub fn lrslope(data: &[f64], period: usize, interval: f64) -> Vec<f64> {
    let n = data.len();
    let mut out = nan_vec(n);
    if period == 0 || n < period + 1 {
        return out;
    }
    let (slope, intercept) = _ols_roll(data, period);
    let ex = (period - 1) as f64;
    for i in period..n {
        let cur = slope[i] * ex + intercept[i];
        let prev = slope[i - 1] * ex + intercept[i - 1];
        out[i] = (cur - prev) / interval;
    }
    out
}

/// Rolling Pearson correlation over `period`.
pub fn correl(d1: &[f64], d2: &[f64], period: usize) -> Vec<f64> {
    let n = d1.len();
    let mut out = nan_vec(n);
    if period == 0 || n < period {
        return out;
    }
    let p = period as f64;
    for i in period - 1..n {
        let x = &d1[i + 1 - period..i + 1];
        let y = &d2[i + 1 - period..i + 1];
        let mut mx = 0.0;
        let mut my = 0.0;
        for j in 0..period {
            mx += x[j];
            my += y[j];
        }
        mx /= p;
        my /= p;
        let mut num = 0.0;
        let mut ssx = 0.0;
        let mut ssy = 0.0;
        for j in 0..period {
            let dx = x[j] - mx;
            let dy = y[j] - my;
            num += dx * dy;
            ssx += dx * dx;
            ssy += dy * dy;
        }
        let den = (ssx * ssy).sqrt();
        out[i] = if den > 0.0 { num / den } else { 0.0 };
    }
    out
}

/// Rolling Beta = cov(asset_returns, market_returns) / var(market_returns) over `period`.
pub fn beta(asset: &[f64], market: &[f64], period: usize) -> Vec<f64> {
    let n = asset.len();
    let mut out = nan_vec(n);
    if period == 0 || n < period + 1 {
        return out;
    }
    let p = period as f64;
    let mut ar = nan_vec(n);
    let mut mr = nan_vec(n);
    for i in 1..n {
        ar[i] = asset[i] - asset[i - 1];
        mr[i] = market[i] - market[i - 1];
    }
    // O(n) rolling moments over the first-difference returns. Returns are small,
    // mean ~0, so the one-pass cov/var formula does not suffer the large-magnitude
    // cancellation that price-level correlation would (verified <1e-11 vs the prior
    // per-window two-pass on 924k NIFTY bars). Window is ar/mr[i+1-period..=i].
    let (mut sa, mut sm, mut smm, mut sam) = (0.0, 0.0, 0.0, 0.0);
    for i in 1..n {
        sa += ar[i];
        sm += mr[i];
        smm += mr[i] * mr[i];
        sam += ar[i] * mr[i];
        if i >= period + 1 {
            let oa = ar[i - period];
            let om = mr[i - period];
            sa -= oa;
            sm -= om;
            smm -= om * om;
            sam -= oa * om;
        }
        if i >= period {
            let cov = sam / p - (sa / p) * (sm / p);
            let mvar = smm / p - (sm / p) * (sm / p);
            out[i] = if mvar > 0.0 { cov / mvar } else { 0.0 };
        }
    }
    out
}

/// Rolling sample variance (/(lookback-1)), threaded rolling sums. PR or LR(log*100).
pub fn variance(data: &[f64], lookback: usize, use_log: bool) -> Vec<f64> {
    let n = data.len();
    let mut source = nan_vec(n);
    if use_log {
        for i in 1..n {
            if data[i] > 0.0 && data[i - 1] > 0.0 {
                source[i] = (data[i] / data[i - 1]).ln() * 100.0;
            }
        }
    } else {
        source.copy_from_slice(data);
    }
    let mut var = nan_vec(n);
    if lookback <= 1 || n < lookback {
        return var;
    }
    let lb = lookback as f64;
    let mut rs = 0.0;
    let mut rsq = 0.0;
    for &s in source.iter().take(lookback) {
        if !s.is_nan() {
            rs += s;
            rsq += s * s;
        }
    }
    let mean = rs / lb;
    let v = (rsq - lb * mean * mean) / (lb - 1.0);
    if v >= 0.0 {
        var[lookback - 1] = v;
    }
    for i in lookback..n {
        let old = source[i - lookback];
        let new = source[i];
        if !old.is_nan() {
            rs -= old;
            rsq -= old * old;
        }
        if !new.is_nan() {
            rs += new;
            rsq += new * new;
        }
        let mean = rs / lb;
        let v = (rsq - lb * mean * mean) / (lb - 1.0);
        if v >= 0.0 {
            var[i] = v;
        }
    }
    var
}

/// Aroon up/down over `period` (lookback period+1, first-occurrence extrema).
pub fn aroon(high: &[f64], low: &[f64], period: usize) -> (Vec<f64>, Vec<f64>) {
    let n = high.len();
    let mut up = nan_vec(n);
    let mut down = nan_vec(n);
    if period == 0 {
        return (up, down);
    }
    let lookback = period + 1;
    if n < lookback {
        return (up, down);
    }
    let p = period as f64;
    // O(n): two monotonic deques tracking the absolute INDEX of the rolling max(high)
    // and min(low) over `lookback` bars. Strict comparisons on back-pop keep the
    // EARLIEST extreme at the front, matching the reference's first-occurrence tie
    // break (periods-since-extreme = i - front_index = lookback-1-position).
    let cap = lookback.next_power_of_two();
    let mask = cap - 1;
    let mut hr: Vec<usize> = vec![0usize; cap]; // high max-deque (decreasing)
    let mut lr: Vec<usize> = vec![0usize; cap]; // low  min-deque (increasing)
    let (mut hh, mut ht) = (0usize, 0usize);
    let (mut lh, mut lt) = (0usize, 0usize);
    for i in 0..n {
        if hh < ht && hr[hh & mask] + lookback <= i {
            hh += 1;
        }
        if lh < lt && lr[lh & mask] + lookback <= i {
            lh += 1;
        }
        let hx = high[i];
        while ht > hh && high[hr[(ht - 1) & mask]] < hx {
            ht -= 1;
        }
        hr[ht & mask] = i;
        ht += 1;
        let lx = low[i];
        while lt > lh && low[lr[(lt - 1) & mask]] > lx {
            lt -= 1;
        }
        lr[lt & mask] = i;
        lt += 1;
        if i + 1 >= lookback {
            up[i] = 100.0 * (p - (i - hr[hh & mask]) as f64) / p;
            down[i] = 100.0 * (p - (i - lr[lh & mask]) as f64) / p;
        }
    }
    (up, down)
}

/// Williams Fractals (up, down) booleans, TradingView frontier patterns.
pub fn williams_fractals(high: &[f64], low: &[f64], n: usize) -> (Vec<bool>, Vec<bool>) {
    let length = high.len();
    let mut fu = vec![false; length];
    let mut fd = vec![false; length];
    if n == 0 || length < 2 * n + 1 {
        return (fu, fd);
    }
    for center in n..length - n {
        let mut df = true;
        for i in 1..n + 1 {
            if high[center - i] >= high[center] {
                df = false;
                break;
            }
        }
        if df {
            let mut f = [true; 5];
            for i in 1..n + 1 {
                if center + i < length && high[center + i] >= high[center] {
                    f[0] = false;
                }
                for p in 1..5 {
                    for q in 1..p + 1 {
                        if center + q < length && high[center + q] > high[center] {
                            f[p] = false;
                        }
                    }
                    if center + i + p < length && high[center + i + p] >= high[center] {
                        f[p] = false;
                    }
                }
            }
            fu[center] = f.iter().any(|&x| x);
        }
        let mut df = true;
        for i in 1..n + 1 {
            if low[center - i] <= low[center] {
                df = false;
                break;
            }
        }
        if df {
            let mut f = [true; 5];
            for i in 1..n + 1 {
                if center + i < length && low[center + i] <= low[center] {
                    f[0] = false;
                }
                for p in 1..5 {
                    for q in 1..p + 1 {
                        if center + q < length && low[center + q] < low[center] {
                            f[p] = false;
                        }
                    }
                    if center + i + p < length && low[center + i + p] <= low[center] {
                        f[p] = false;
                    }
                }
            }
            fd[center] = f.iter().any(|&x| x);
        }
    }
    (fu, fd)
}

/// Rolling mode via histogram binning (first-max bin midpoint).
pub fn mode(data: &[f64], period: usize, bins: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = nan_vec(n);
    if period == 0 || bins == 0 || n < period {
        return out;
    }
    for i in period - 1..n {
        let w = &data[i + 1 - period..i + 1];
        let mut mn = w[0];
        let mut mx = w[0];
        for &x in w {
            if x < mn {
                mn = x;
            }
            if x > mx {
                mx = x;
            }
        }
        if mx > mn {
            let bw = (mx - mn) / bins as f64;
            let mut counts = vec![0i64; bins];
            for &x in w {
                let mut bi = ((x - mn) / bw) as i64;
                if bi < 0 {
                    bi = 0;
                }
                if bi > bins as i64 - 1 {
                    bi = bins as i64 - 1;
                }
                counts[bi as usize] += 1;
            }
            let mut mb = 0usize;
            for k in 1..bins {
                if counts[k] > counts[mb] {
                    mb = k;
                }
            }
            out[i] = mn + (mb as f64 + 0.5) * bw;
        } else {
            out[i] = w[0];
        }
    }
    out
}

/// Triangular MA: double per-window SMA. n1=(period+1)/2, n2=period-n1+1; second pass
/// averages the last n2 valid values of the first SMA. Matches TRIMA._calculate_trima.
pub fn trima(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = nan_vec(n);
    if period == 0 {
        return out;
    }
    let n1 = (period + 1) / 2;
    let n2 = period - n1 + 1;
    if n1 == 0 || n2 == 0 {
        return out;
    }
    // First pass is a plain O(n) rolling SMA (valid contiguously from index n1-1).
    let first = sma(data, n1);
    if n1 + n2 < 2 {
        return out;
    }
    let start = n1 + n2 - 2;
    if start >= n {
        return out;
    }
    // Second pass: O(n) rolling SMA over `first` of window n2. Since `first` is
    // contiguously valid from n1-1, the window for out[start] is first[n1-1..=start].
    let mut s = 0.0;
    for k in start + 1 - n2..=start {
        s += first[k];
    }
    out[start] = s / n2 as f64;
    for i in start + 1..n {
        s = s + first[i] - first[i - n2];
        out[i] = s / n2 as f64;
    }
    out
}

/// Stochastic RSI -> (%K, %D). RSI(rsi_period) -> per-window stoch(stoch_period) of RSI
/// (50 when flat) -> SMA-of-clean(%K, k_period) -> SMA-of-clean(%D, d_period).
pub fn stochrsi(
    data: &[f64],
    rsi_period: usize,
    stoch_period: usize,
    k_period: usize,
    d_period: usize,
) -> (Vec<f64>, Vec<f64>) {
    let r = rsi(data, rsi_period);
    let n = r.len();
    let mut sr = nan_vec(n);
    if stoch_period >= 1 {
        for i in stoch_period - 1..n {
            let mut hi = f64::NEG_INFINITY;
            let mut lo = f64::INFINITY;
            let mut any = false;
            for k in i + 1 - stoch_period..i + 1 {
                let v = r[k];
                if !v.is_nan() {
                    if v > hi {
                        hi = v;
                    }
                    if v < lo {
                        lo = v;
                    }
                    any = true;
                }
            }
            if any {
                if hi != lo {
                    sr[i] = (r[i] - lo) / (hi - lo) * 100.0;
                } else {
                    sr[i] = 50.0;
                }
            }
        }
    }
    let kk = _smooth_clean(&sr, k_period);
    let dd = _smooth_clean(&kk, d_period);
    (kk, dd)
}

/// Mean of the non-NaN values in each trailing `period` window (NaN if none).
fn _smooth_clean(src: &[f64], period: usize) -> Vec<f64> {
    let n = src.len();
    let mut out = nan_vec(n);
    if period == 0 {
        return out;
    }
    for i in period - 1..n {
        let mut s = 0.0;
        let mut cnt = 0usize;
        for k in i + 1 - period..i + 1 {
            if !src[k].is_nan() {
                s += src[k];
                cnt += 1;
            }
        }
        if cnt > 0 {
            out[i] = s / cnt as f64;
        }
    }
    out
}

/// Rolling median over `period` (odd -> middle, even -> mean of two middle).
pub fn median(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = nan_vec(n);
    if period == 0 || n < period {
        return out;
    }
    let mut buf = vec![0.0f64; period];
    for i in period - 1..n {
        buf.copy_from_slice(&data[i + 1 - period..i + 1]);
        buf.sort_by(|a, b| a.partial_cmp(b).unwrap());
        out[i] = if period % 2 == 1 {
            buf[period / 2]
        } else {
            (buf[period / 2 - 1] + buf[period / 2]) / 2.0
        };
    }
    out
}

/// Single-series stochastic over `period` (NaN-aware; 50 when flat). Used by STC.
pub fn stoch_single(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = nan_vec(n);
    if period == 0 {
        return out;
    }
    for i in period - 1..n {
        let w = &data[i + 1 - period..i + 1];
        let mut hi = f64::NEG_INFINITY;
        let mut lo = f64::INFINITY;
        let mut any = false;
        for &x in w {
            if !x.is_nan() {
                if x > hi {
                    hi = x;
                }
                if x < lo {
                    lo = x;
                }
                any = true;
            }
        }
        if any {
            if hi != lo && !data[i].is_nan() {
                out[i] = (data[i] - lo) / (hi - lo) * 100.0;
            } else if hi == lo {
                out[i] = 50.0;
            }
        }
    }
    out
}

/// NaN-aware weighted MA (weights 1..period over valid entries). Used by Coppock.
pub fn wma_nan(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = nan_vec(n);
    if period == 0 || n < period {
        return out;
    }
    for i in period - 1..n {
        if data[i].is_nan() {
            continue;
        }
        let mut ws = 0.0;
        let mut vw = 0.0;
        for j in 0..period {
            let v = data[i + 1 - period + j];
            if !v.is_nan() {
                let w = (j + 1) as f64;
                ws += v * w;
                vw += w;
            }
        }
        if vw > 0.0 {
            out[i] = ws / vw;
        }
    }
    out
}

#[inline]
fn swma_vec(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut out = nan_vec(n);
    for i in 3..n {
        out[i] = (data[i - 3] + 2.0 * data[i - 2] + 2.0 * data[i - 1] + data[i]) / 6.0;
    }
    out
}

/// Relative Vigor Index (vigor): sum(swma(c-o))/sum(swma(h-l)) over period; signal=swma.
pub fn rvi_vigor(
    open_: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = close.len();
    let co: Vec<f64> = (0..n).map(|i| close[i] - open_[i]).collect();
    let hl: Vec<f64> = (0..n).map(|i| high[i] - low[i]).collect();
    let sco = swma_vec(&co);
    let shl = swma_vec(&hl);
    let mut rvi = nan_vec(n);
    if n > period + 2 {
        for i in period + 2..n {
            let mut ns = 0.0;
            let mut ds = 0.0;
            for j in i + 1 - period..i + 1 {
                if !sco[j].is_nan() {
                    ns += sco[j];
                }
                if !shl[j].is_nan() {
                    ds += shl[j];
                }
            }
            rvi[i] = if ds != 0.0 { ns / ds } else { 0.0 };
        }
    }
    let signal = swma_vec(&rvi);
    (rvi, signal)
}

/// ADX system: returns (di_plus, di_minus, adx) with Wilder smoothing.
pub fn adx(high: &[f64], low: &[f64], close: &[f64], period: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = close.len();
    let mut dip = nan_vec(n);
    let mut dim = nan_vec(n);
    let mut dx = nan_vec(n);
    if period == 0 || n < period {
        return (dip, dim, nan_vec(n));
    }
    // Single fused pass: true range and +DM/-DM are computed on the fly and their
    // Wilder-smoothed values (sa/sp/sm) carried in scalars - identical seeding to
    // three ema_wilder() calls (seed = mean of first `period`, then x = (x*(p-1)+v)/p)
    // but without the six intermediate arrays.
    let p = period as f64;
    let pm1 = (period - 1) as f64;
    let (mut sa, mut sp, mut sm) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let tr_i = if i == 0 {
            high[0] - low[0]
        } else {
            max3(
                high[i] - low[i],
                (high[i] - close[i - 1]).abs(),
                (low[i] - close[i - 1]).abs(),
            )
        };
        let (dmp_i, dmm_i) = if i == 0 {
            (0.0, 0.0)
        } else {
            let up = high[i] - high[i - 1];
            let dn = low[i - 1] - low[i];
            (
                if up > dn && up > 0.0 { up } else { 0.0 },
                if dn > up && dn > 0.0 { dn } else { 0.0 },
            )
        };
        if i < period {
            sa += tr_i;
            sp += dmp_i;
            sm += dmm_i;
            if i + 1 == period {
                sa /= p;
                sp /= p;
                sm /= p;
            }
        } else {
            sa = (sa * pm1 + tr_i) / p;
            sp = (sp * pm1 + dmp_i) / p;
            sm = (sm * pm1 + dmm_i) / p;
        }
        if i + 1 >= period && sa > 0.0 {
            dip[i] = sp / sa * 100.0;
            dim[i] = sm / sa * 100.0;
            let ds = dip[i] + dim[i];
            if ds > 0.0 {
                dx[i] = (dip[i] - dim[i]).abs() / ds * 100.0;
            }
        }
    }
    let adx = ema_wilder(&dx, period);
    (dip, dim, adx)
}

/// Random Walk Index (rwi_high, rwi_low). ATR = per-window mean of True Range.
pub fn rwi(high: &[f64], low: &[f64], close: &[f64], period: usize) -> (Vec<f64>, Vec<f64>) {
    let n = close.len();
    let atr = win_mean(&true_range(high, low, close), period);
    let mut rh = nan_vec(n);
    let mut rl = nan_vec(n);
    if period > 0 && n > period {
        let sq = (period as f64).sqrt();
        for i in period..n {
            if atr[i] > 0.0 {
                rh[i] = (high[i] - low[i - period]) / (atr[i] * sq);
                rl[i] = (high[i - period] - low[i]) / (atr[i] * sq);
            }
        }
    }
    (rh, rl)
}

/// valuewhen: value of `array` when `expr` (nonzero == true) was true the n-th most
/// recent time. Mirrors the legacy kernel, including its 1000-entry lookback cap.
pub fn valuewhen(expr: &[f64], array: &[f64], n: usize) -> Vec<f64> {
    let length = expr.len();
    let mut result = nan_vec(length);
    if n == 0 {
        return result;
    }
    let max_lookback = std::cmp::min(1000, length);
    let mut idx = vec![0usize; max_lookback.max(1)];
    let mut count = 0usize;
    for i in 0..length {
        if expr[i] != 0.0 {
            if count >= max_lookback {
                for j in 0..max_lookback - 1 {
                    idx[j] = idx[j + 1];
                }
                idx[max_lookback - 1] = i;
            } else {
                idx[count] = i;
                count += 1;
            }
        }
        if count >= n {
            result[i] = array[idx[count - n]];
        }
    }
    result
}

/// Ultimate Oscillator, single pass. Maintains the three buying-pressure and
/// true-range rolling sums simultaneously (no intermediate arrays / round-trips).
/// out = 100*(4*r1 + 2*r2 + r3)/7, r = sum(bp)/sum(tr); NaN before max(p1,p2,p3)-1.
pub fn ultosc(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
) -> Vec<f64> {
    let n = close.len();
    let mut out = nan_vec(n);
    let mp = p1.max(p2).max(p3);
    if n == 0 || mp == 0 || n < mp {
        return out;
    }
    let tr = true_range(high, low, close);
    let mut bp = vec![0.0f64; n];
    bp[0] = close[0] - low[0].min(close[0]);
    for i in 1..n {
        bp[i] = close[i] - low[i].min(close[i - 1]);
    }
    let (mut b1, mut t1, mut b2, mut t2, mut b3, mut t3) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..n {
        b1 += bp[i];
        t1 += tr[i];
        b2 += bp[i];
        t2 += tr[i];
        b3 += bp[i];
        t3 += tr[i];
        if i >= p1 {
            b1 -= bp[i - p1];
            t1 -= tr[i - p1];
        }
        if i >= p2 {
            b2 -= bp[i - p2];
            t2 -= tr[i - p2];
        }
        if i >= p3 {
            b3 -= bp[i - p3];
            t3 -= tr[i - p3];
        }
        if i + 1 >= mp {
            let r1 = if t1 > 0.0 { b1 / t1 } else { 0.0 };
            let r2 = if t2 > 0.0 { b2 / t2 } else { 0.0 };
            let r3 = if t3 > 0.0 { b3 / t3 } else { 0.0 };
            out[i] = 100.0 * (4.0 * r1 + 2.0 * r2 + r3) / 7.0;
        }
    }
    out
}

/// Chaikin A/D Oscillator (TA-Lib ADOSC) = EMA(AD, fast) - EMA(AD, slow), where AD
/// is the cumulative accumulation/distribution line. Single-pass AD, then two EMAs.
pub fn adosc(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    fast: usize,
    slow: usize,
) -> Vec<f64> {
    let n = close.len();
    let mut ad = vec![0.0f64; n];
    let mut acc = 0.0;
    for i in 0..n {
        let rng = high[i] - low[i];
        let clv = if rng != 0.0 {
            ((close[i] - low[i]) - (high[i] - close[i])) / rng
        } else {
            0.0
        };
        acc += clv * volume[i];
        ad[i] = acc;
    }
    let ef = ema(&ad, fast);
    let es = ema(&ad, slow);
    let mut out = vec![0.0f64; n];
    for i in 0..n {
        out[i] = ef[i] - es[i];
    }
    out
}

// ---------------------------------------------------------------------------
// TA-Lib-faithful directional-movement family (exact TA-Lib seeding/smoothing).
// These intentionally replicate TA-Lib (NOT OpenAlgo's TradingView-seeded adx).
// ---------------------------------------------------------------------------

/// TA-Lib PLUS_DM: Wilder running sum of +DM1. First value at index period-1
/// = sum of +DM1 over bars 1..period-1; then dm = dm - dm/period + +DM1.
pub fn plus_dm_talib(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    let n = high.len();
    let mut out = nan_vec(n);
    if period == 0 || n < period {
        return out;
    }
    let mut prev = 0.0;
    for i in 1..period {
        let dp = high[i] - high[i - 1];
        let dm = low[i - 1] - low[i];
        if dp > 0.0 && dp > dm {
            prev += dp;
        }
    }
    out[period - 1] = prev;
    let pf = period as f64;
    for i in period..n {
        let dp = high[i] - high[i - 1];
        let dm = low[i - 1] - low[i];
        let add = if dp > 0.0 && dp > dm { dp } else { 0.0 };
        prev = prev - prev / pf + add;
        out[i] = prev;
    }
    out
}

/// TA-Lib MINUS_DM (symmetric to plus_dm_talib).
pub fn minus_dm_talib(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    let n = high.len();
    let mut out = nan_vec(n);
    if period == 0 || n < period {
        return out;
    }
    let mut prev = 0.0;
    for i in 1..period {
        let dp = high[i] - high[i - 1];
        let dm = low[i - 1] - low[i];
        if dm > 0.0 && dm > dp {
            prev += dm;
        }
    }
    out[period - 1] = prev;
    let pf = period as f64;
    for i in period..n {
        let dp = high[i] - high[i - 1];
        let dm = low[i - 1] - low[i];
        let add = if dm > 0.0 && dm > dp { dm } else { 0.0 };
        prev = prev - prev / pf + add;
        out[i] = prev;
    }
    out
}

/// TA-Lib DX. First value at index period (Wilder-smoothed +DI/-DI then
/// 100*|+DI - -DI|/(+DI + -DI)).
pub fn dx_talib(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = high.len();
    let mut out = nan_vec(n);
    if period == 0 || n < period + 1 {
        return out;
    }
    let tr1 = true_range(high, low, close);
    let (mut str_, mut sp, mut sm) = (0.0, 0.0, 0.0);
    for i in 1..period {
        let dp = high[i] - high[i - 1];
        let dm = low[i - 1] - low[i];
        if dp > 0.0 && dp > dm {
            sp += dp;
        } else if dm > 0.0 && dm > dp {
            sm += dm;
        }
        str_ += tr1[i];
    }
    let pf = period as f64;
    for i in period..n {
        let dp = high[i] - high[i - 1];
        let dm = low[i - 1] - low[i];
        str_ = str_ - str_ / pf + tr1[i];
        sp = sp - sp / pf + if dp > 0.0 && dp > dm { dp } else { 0.0 };
        sm = sm - sm / pf + if dm > 0.0 && dm > dp { dm } else { 0.0 };
        if str_ != 0.0 {
            let pdi = 100.0 * sp / str_;
            let mdi = 100.0 * sm / str_;
            let s = pdi + mdi;
            out[i] = if s != 0.0 { 100.0 * (pdi - mdi).abs() / s } else { 0.0 };
        } else {
            out[i] = 0.0;
        }
    }
    out
}

/// TA-Lib ADX. First value at index 2*period-1 (mean of `period` DX values),
/// then Wilder recursion adx = (adx*(period-1) + dx)/period.
pub fn adx_talib(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = high.len();
    let mut out = nan_vec(n);
    if period == 0 || n < 2 * period {
        return out;
    }
    let tr1 = true_range(high, low, close);
    let (mut str_, mut sp, mut sm) = (0.0, 0.0, 0.0);
    for i in 1..period {
        let dp = high[i] - high[i - 1];
        let dm = low[i - 1] - low[i];
        if dp > 0.0 && dp > dm {
            sp += dp;
        } else if dm > 0.0 && dm > dp {
            sm += dm;
        }
        str_ += tr1[i];
    }
    let pf = period as f64;
    let wilder = |i: usize, str_: &mut f64, sp: &mut f64, sm: &mut f64| -> f64 {
        let dp = high[i] - high[i - 1];
        let dm = low[i - 1] - low[i];
        *str_ = *str_ - *str_ / pf + tr1[i];
        *sp = *sp - *sp / pf + if dp > 0.0 && dp > dm { dp } else { 0.0 };
        *sm = *sm - *sm / pf + if dm > 0.0 && dm > dp { dm } else { 0.0 };
        if *str_ != 0.0 {
            let pdi = 100.0 * *sp / *str_;
            let mdi = 100.0 * *sm / *str_;
            let s = pdi + mdi;
            if s != 0.0 {
                return 100.0 * (pdi - mdi).abs() / s;
            }
        }
        0.0
    };
    let mut sumdx = 0.0;
    for i in period..2 * period {
        sumdx += wilder(i, &mut str_, &mut sp, &mut sm);
    }
    let mut prevadx = sumdx / pf;
    out[2 * period - 1] = prevadx;
    for i in 2 * period..n {
        let dx = wilder(i, &mut str_, &mut sp, &mut sm);
        prevadx = (prevadx * (pf - 1.0) + dx) / pf;
        out[i] = prevadx;
    }
    out
}

/// TA-Lib ADXR = (ADX[i] + ADX[i-(period-1)]) / 2.
pub fn adxr_talib(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = high.len();
    let adx = adx_talib(high, low, close, period);
    let mut out = nan_vec(n);
    if period == 0 {
        return out;
    }
    let off = period - 1;
    let start = 2 * period - 1 + off;
    for i in start..n {
        if !adx[i].is_nan() && !adx[i - off].is_nan() {
            out[i] = (adx[i] + adx[i - off]) / 2.0;
        }
    }
    out
}

/// TA-Lib STOCHF -> (fastk, fastd). fastk = 100*(c-LL)/(HH-LL) over fastk_period,
/// fastd = SMA(fastk, fastd_period). Both aligned to first index
/// (fastk_period-1)+(fastd_period-1), matching TA-Lib's output padding.
pub fn stochf(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fastk_period: usize,
    fastd_period: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = close.len();
    let mut k = nan_vec(n);
    let mut d = nan_vec(n);
    if fastk_period == 0 || fastd_period == 0 || n < fastk_period {
        return (k, d);
    }
    let hh = highest(high, fastk_period);
    let ll = lowest(low, fastk_period);
    let mut raw = nan_vec(n);
    for i in fastk_period - 1..n {
        let rng = hh[i] - ll[i];
        raw[i] = if rng != 0.0 { 100.0 * (close[i] - ll[i]) / rng } else { 0.0 };
    }
    let start = fastk_period - 1 + fastd_period - 1;
    let fp = fastd_period as f64;
    for i in start..n {
        let mut s = 0.0;
        for j in i + 1 - fastd_period..=i {
            s += raw[j];
        }
        d[i] = s / fp;
        k[i] = raw[i];
    }
    (k, d)
}

/// TA-Lib LINEARREG_ANGLE = degrees(atan(OLS slope)) over `period`. O(n).
pub fn linreg_angle(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = nan_vec(n);
    if period == 0 || n < period {
        return out;
    }
    let (slope, _) = _ols_roll(data, period);
    for i in period - 1..n {
        out[i] = slope[i].atan().to_degrees();
    }
    out
}

/// TA-Lib LINEARREG_INTERCEPT = OLS intercept b over `period`. O(n).
pub fn linreg_intercept(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = nan_vec(n);
    if period == 0 || n < period {
        return out;
    }
    let (_, intercept) = _ols_roll(data, period);
    for i in period - 1..n {
        out[i] = intercept[i];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) {
        if a.is_nan() && b.is_nan() {
            return;
        }
        assert!((a - b).abs() <= 1e-12, "expected {b}, got {a}");
    }

    #[test]
    fn sma_basic() {
        let d = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r = sma(&d, 3);
        assert!(r[0].is_nan() && r[1].is_nan());
        approx(r[2], 2.0);
        approx(r[3], 3.0);
        approx(r[4], 4.0);
    }

    #[test]
    fn sma_short() {
        let d = [1.0, 2.0];
        for v in sma(&d, 5) {
            assert!(v.is_nan());
        }
    }

    #[test]
    fn ema_seed_and_recursion() {
        let d = [1.0, 2.0, 3.0, 4.0];
        let r = ema(&d, 3);
        let alpha = 2.0 / 4.0;
        approx(r[0], 1.0);
        approx(r[1], alpha * 2.0 + (1.0 - alpha) * 1.0);
    }

    #[test]
    fn stdev_population() {
        let d = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let r = stdev(&d, 8);
        // population std of the full set is 2.0
        approx(r[7], 2.0);
    }

    #[test]
    fn true_range_first_is_hl() {
        let h = [10.0, 12.0];
        let l = [8.0, 9.0];
        let c = [9.0, 11.0];
        let tr = true_range(&h, &l, &c);
        approx(tr[0], 2.0);
        // max(12-9, |12-9|, |9-9|) = 3
        approx(tr[1], 3.0);
    }

    #[test]
    fn atr_wilder_seed() {
        let h = [10.0, 12.0, 13.0, 14.0];
        let l = [8.0, 9.0, 10.0, 11.0];
        let c = [9.0, 11.0, 12.0, 13.0];
        let atr = atr_wilder(&h, &l, &c, 3);
        assert!(atr[0].is_nan() && atr[1].is_nan());
        assert!(!atr[2].is_nan());
    }

    #[test]
    fn highest_lowest_window() {
        let d = [1.0, 3.0, 2.0, 5.0, 4.0];
        let hi = highest(&d, 3);
        let lo = lowest(&d, 3);
        approx(hi[2], 3.0);
        approx(hi[3], 5.0);
        approx(hi[4], 5.0);
        approx(lo[2], 1.0);
        approx(lo[3], 2.0);
        approx(lo[4], 2.0);
    }

    #[test]
    fn change_and_roc() {
        let d = [10.0, 11.0, 0.0, 5.0];
        let ch = change(&d, 1);
        assert!(ch[0].is_nan());
        approx(ch[1], 1.0);
        approx(ch[3], 5.0);
        let r = roc(&d, 1);
        approx(r[1], 10.0);
        // divisor data[1]=11 -> change to 0: (0-11)/11*100
        approx(r[2], (0.0 - 11.0) / 11.0 * 100.0);
        // data[2]=0 divisor -> stays NaN
        assert!(r[3].is_nan());
    }

    #[test]
    fn crossover_detect() {
        let a = [1.0, 1.0, 3.0];
        let b = [2.0, 2.0, 2.0];
        let x = crossover(&a, &b);
        assert!(!x[0] && !x[1] && x[2]);
        let y = crossunder(&b, &a);
        assert!(y[2]);
    }

    #[test]
    fn exrem_flip_latch() {
        let p = [true, true, false, true];
        let s = [false, false, true, false];
        assert_eq!(exrem(&p, &s), vec![true, false, false, true]);
        assert_eq!(flip(&p, &s), vec![true, true, false, true]);
    }

    #[test]
    fn trima_basic() {
        let d: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let r = trima(&d, 4); // n1=2, n2=3
        assert!(r[3].is_finite());
        // smooth of a linear ramp stays increasing
        assert!(r[9] > r[8]);
    }

    #[test]
    fn talib_dm_family_alignment() {
        let n = 60usize;
        let high: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.3).sin() * 5.0 + i as f64 * 0.1).collect();
        let low: Vec<f64> = high.iter().map(|h| h - 2.0).collect();
        let close: Vec<f64> = high.iter().map(|h| h - 1.0).collect();
        let p = 14;
        let pdm = plus_dm_talib(&high, &low, p);
        let mdm = minus_dm_talib(&high, &low, p);
        // first valid at period-1
        assert!(pdm[p - 2].is_nan() && pdm[p - 1].is_finite());
        assert!(mdm[p - 1].is_finite());
        let dx = dx_talib(&high, &low, &close, p);
        assert!(dx[p - 1].is_nan() && dx[p].is_finite()); // first DX at index period
        let adx = adx_talib(&high, &low, &close, p);
        assert!(adx[2 * p - 2].is_nan() && adx[2 * p - 1].is_finite()); // 2*period-1
        let adxr = adxr_talib(&high, &low, &close, p);
        assert!(adxr[3 * p - 3].is_nan() && adxr[3 * p - 2].is_finite()); // 3*period-2
        // DX bounded [0,100]
        for v in dx.iter().filter(|v| !v.is_nan()) {
            assert!(*v >= -1e-9 && *v <= 100.0 + 1e-9);
        }
    }

    #[test]
    fn stochf_and_linreg_variants() {
        let n = 40usize;
        let high: Vec<f64> = (0..n).map(|i| 50.0 + (i as f64 * 0.4).sin() * 3.0).collect();
        let low: Vec<f64> = high.iter().map(|h| h - 1.5).collect();
        let close: Vec<f64> = high.iter().map(|h| h - 0.5).collect();
        let (k, d) = stochf(&high, &low, &close, 5, 3);
        // both aligned to (5-1)+(3-1)=6
        assert!(k[5].is_nan() && k[6].is_finite() && d[6].is_finite());
        for v in k.iter().filter(|v| !v.is_nan()) {
            assert!(*v >= -1e-9 && *v <= 100.0 + 1e-9);
        }
        let ang = linreg_angle(&close, 14);
        let icpt = linreg_intercept(&close, 14);
        assert!(ang[12].is_nan() && ang[13].is_finite());
        assert!(icpt[13].is_finite());
        // angle of a rising line is positive
        let ramp: Vec<f64> = (0..30).map(|x| x as f64).collect();
        let a = linreg_angle(&ramp, 10);
        approx(a[29], 45.0); // slope 1 -> 45 degrees
    }

    #[test]
    fn stochrsi_bounds() {
        let d: Vec<f64> = (1..=60).map(|x| (x as f64 * 0.2).sin() + 100.0).collect();
        let (k, dd) = stochrsi(&d, 14, 14, 3, 3);
        let kv: Vec<f64> = k.iter().cloned().filter(|x| !x.is_nan()).collect();
        assert!(kv.iter().all(|&x| x >= 0.0 && x <= 100.0));
        assert_eq!(dd.len(), 60);
    }

    #[test]
    fn aroon_extremes() {
        let h = [1.0, 2.0, 3.0, 4.0, 5.0];
        let l = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (up, down) = aroon(&h, &l, 3);
        approx(up[4], 100.0); // highest is the current bar
        approx(down[4], 0.0); // lowest was period bars ago
    }

    #[test]
    fn adx_rwi_run() {
        let h: Vec<f64> = (1..=40).map(|x| (x as f64).sin() + 10.0).collect();
        let l: Vec<f64> = (1..=40).map(|x| (x as f64).sin() + 9.0).collect();
        let c: Vec<f64> = (1..=40).map(|x| (x as f64).sin() + 9.5).collect();
        let (dp, dm, a) = adx(&h, &l, &c, 14);
        assert_eq!(a.len(), 40);
        assert!(dp[39].is_finite() && dm[39].is_finite());
        let (rh, rl) = rwi(&h, &l, &c, 14);
        assert!(rh[39].is_finite() && rl[39].is_finite());
    }

    #[test]
    fn mode_variance_run() {
        let d: Vec<f64> = (1..=60).map(|x| (x % 7) as f64 + 100.0).collect();
        assert!(mode(&d, 20, 10)[59].is_finite());
        assert!(variance(&d, 20, false)[59] >= 0.0);
        assert!(stoch_single(&d, 10)[59] >= 0.0);
    }

    #[test]
    fn win_mean_std_basic() {
        let d = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let m = win_mean(&d, 8);
        approx(m[7], 5.0);
        let s = win_std(&d, 8);
        approx(s[7], 2.0); // population std
        assert!(m[6].is_nan() && s[6].is_nan());
    }

    #[test]
    fn linreg_perfect_line() {
        let d = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let r = linreg(&d, 3);
        approx(r[2], 3.0); // endpoint of line through (1,2,3) extended
        approx(r[5], 6.0);
        let t = tsf(&d, 3);
        approx(t[2], 4.0); // one step ahead
    }

    #[test]
    fn correl_self_is_one() {
        let d = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0];
        let r = correl(&d, &d, 4);
        approx(r[5], 1.0);
    }

    #[test]
    fn beta_self_is_one() {
        let a = [10.0, 11.0, 10.5, 12.0, 11.5, 13.0];
        let r = beta(&a, &a, 4);
        approx(r[5], 1.0);
    }

    #[test]
    fn sar_shapes() {
        let h = [10.0, 11.0, 12.0, 11.5, 13.0, 14.0];
        let l = [9.0, 10.0, 11.0, 10.5, 12.0, 13.0];
        let (s, t) = sar(&h, &l, 0.02, 0.2);
        assert_eq!(s.len(), 6);
        approx(s[0], 9.0);
        assert!(t[0] == 1.0);
    }

    #[test]
    fn obv_cumulative() {
        let c = [10.0, 11.0, 10.5, 10.5];
        let v = [100.0, 200.0, 50.0, 30.0];
        let r = obv(&c, &v);
        approx(r[0], 0.0);
        approx(r[1], 200.0); // up
        approx(r[2], 150.0); // down -50
        approx(r[3], 150.0); // equal -> 0 (matches TA-Lib / TradingView)
    }

    #[test]
    fn adl_seed_zero() {
        let h = [10.0, 12.0];
        let l = [8.0, 9.0];
        let c = [9.0, 11.0];
        let v = [100.0, 200.0];
        let r = adl(&h, &l, &c, &v);
        approx(r[0], 0.0);
        assert!(r[1].is_finite());
    }

    #[test]
    fn mfi_bounds() {
        let h: Vec<f64> = (1..=20).map(|x| x as f64 + 1.0).collect();
        let l: Vec<f64> = (1..=20).map(|x| x as f64 - 1.0).collect();
        let c: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let v = vec![100.0; 20];
        let r = mfi(&h, &l, &c, &v, 14);
        assert!(r[12].is_nan());
        assert!(r[14] >= 0.0 && r[14] <= 100.0);
    }

    #[test]
    fn ema_first_valid_skips_leading_nan() {
        let d = [f64::NAN, 2.0, 4.0, 6.0];
        let r = ema_first_valid(&d, 3);
        assert!(r[0].is_nan());
        approx(r[1], 2.0); // seed at first valid
    }

    #[test]
    fn bop_basic() {
        let o = [10.0, 10.0];
        let h = [12.0, 11.0];
        let l = [8.0, 11.0];
        let c = [11.0, 11.0];
        let r = bop(&o, &h, &l, &c);
        approx(r[0], (11.0 - 10.0) / (12.0 - 8.0));
        approx(r[1], 0.0); // high==low
    }

    #[test]
    fn updown_streak_runs() {
        let d = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0];
        let s = updown_streak(&d);
        approx(s[1], 1.0);
        approx(s[2], 2.0);
        approx(s[3], 0.0);
        approx(s[4], -1.0);
        approx(s[5], -2.0);
    }

    #[test]
    fn percent_rank_basic() {
        let d = [1.0, 2.0, 3.0, 4.0];
        let r = percent_rank(&d, 3);
        approx(r[2], 100.0 * 2.0 / 3.0); // 3 > 1,2
        approx(r[3], 100.0 * 2.0 / 3.0); // 4 > 2,3
    }

    #[test]
    fn fisher_runs() {
        let d: Vec<f64> = (1..=20).map(|x| (x as f64 * 0.3).sin() + 5.0).collect();
        let (f1, f2) = fisher(&d, 9);
        assert_eq!(f1.len(), 20);
        assert!(f1[8].is_finite());
        approx(f2[9], f1[8]);
    }

    #[test]
    fn supertrend_shapes() {
        let h = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        let l = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0];
        let c = [9.5, 10.5, 11.5, 12.5, 13.5, 14.5];
        let (st, dir) = supertrend(&h, &l, &c, 3, 3.0);
        assert!(st[1].is_nan() && dir[1].is_nan());
        assert!(st[2].is_finite() && (dir[2] == 1.0 || dir[2] == -1.0));
    }

    #[test]
    fn chande_kroll_shapes() {
        let h: Vec<f64> = (1..=30).map(|x| x as f64 + 1.0).collect();
        let l: Vec<f64> = (1..=30).map(|x| x as f64 - 1.0).collect();
        let c: Vec<f64> = (1..=30).map(|x| x as f64).collect();
        let (ls, ss) = chande_kroll_stop(&h, &l, &c, 10, 1.0, 9);
        assert_eq!(ls.len(), 30);
        assert!(ls[0].is_nan());
        assert!(ls[18].is_finite() && ss[18].is_finite());
    }

    #[test]
    fn frama_runs() {
        let h: Vec<f64> = (1..=40).map(|x| (x as f64).sin() + 10.0).collect();
        let l: Vec<f64> = (1..=40).map(|x| (x as f64).sin() + 9.0).collect();
        let r = frama(&h, &l, 16);
        assert_eq!(r.len(), 40);
        assert!(r[39].is_finite());
    }

    #[test]
    fn vidya_seed() {
        let d = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let r = vidya(&d, 3, 0.2);
        assert!(r[2].is_nan());
        approx(r[3], d[3]); // seeded with data[period]
    }

    #[test]
    fn mcginley_seed_is_sma() {
        let d = [2.0, 4.0, 6.0, 8.0, 10.0];
        let r = mcginley(&d, 3);
        approx(r[2], 4.0); // mean(2,4,6)
        assert!(!r[3].is_nan());
    }

    #[test]
    fn alma_finite_window() {
        let d: Vec<f64> = (1..=30).map(|x| x as f64).collect();
        let r = alma(&d, 9, 0.85, 6.0);
        assert!(r[7].is_nan());
        assert!(r[8].is_finite());
    }

    #[test]
    fn kama_tv_seed_and_trend() {
        let d = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let r = kama_tv(&d, 3, 2, 30);
        assert!(r[0].is_nan() && r[2].is_nan());
        // first valid at index length=3; er=1 on a clean trend -> alpha=fast_alpha^2
        let fa = 2.0 / 3.0;
        approx(r[3], fa * fa * d[3] + (1.0 - fa * fa) * d[3]); // prev=src at seed -> == d[3]
    }

    #[test]
    fn rsi_all_gains_is_100() {
        let d = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let r = rsi(&d, 3);
        assert!(r[0].is_nan() && r[2].is_nan());
        approx(r[3], 100.0);
        approx(r[5], 100.0);
    }

    #[test]
    fn williams_r_bounds() {
        let h = [2.0, 3.0, 4.0, 5.0];
        let l = [1.0, 1.0, 2.0, 3.0];
        let c = [1.5, 2.5, 3.5, 4.5];
        let r = williams_r(&h, &l, &c, 2);
        assert!(r[0].is_nan());
        // i=1: hh=3,ll=1,c=2.5 -> -100*(3-2.5)/(3-1)=-25
        approx(r[1], -25.0);
    }

    #[test]
    fn stochastic_shapes() {
        let h = [2.0, 3.0, 4.0, 5.0, 6.0];
        let l = [1.0, 1.0, 2.0, 3.0, 4.0];
        let c = [1.5, 2.5, 3.5, 4.5, 5.5];
        let (k, d) = stochastic(&h, &l, &c, 2, 2, 2);
        assert_eq!(k.len(), 5);
        assert_eq!(d.len(), 5);
        assert!(k[0].is_nan());
    }

    #[test]
    fn cci_zero_meandev() {
        let d = [5.0, 5.0, 5.0, 5.0];
        let r = cci(&d, &d, &d, 2);
        approx(r[1], 0.0);
        approx(r[3], 0.0);
    }

    #[test]
    fn wma_weights() {
        let d = [1.0, 2.0, 3.0, 4.0];
        let r = wma(&d, 3);
        assert!(r[0].is_nan() && r[1].is_nan());
        // (1*1 + 2*2 + 3*3)/6 = 14/6
        approx(r[2], 14.0 / 6.0);
        // (2*1 + 3*2 + 4*3)/6 = 20/6
        approx(r[3], 20.0 / 6.0);
    }

    #[test]
    fn valuewhen_recent() {
        let expr = [0.0, 1.0, 0.0, 1.0, 0.0];
        let arr = [10.0, 11.0, 12.0, 13.0, 14.0];
        let r = valuewhen(&expr, &arr, 1);
        assert!(r[0].is_nan());
        approx(r[1], 11.0);
        approx(r[2], 11.0);
        approx(r[3], 13.0);
        approx(r[4], 13.0);
    }

    #[test]
    fn vwma_zero_volume_fallback() {
        let d = [10.0, 20.0, 30.0];
        let v = [0.0, 0.0, 0.0];
        let r = vwma(&d, &v, 2);
        approx(r[1], 20.0);
        approx(r[2], 30.0);
    }
}
