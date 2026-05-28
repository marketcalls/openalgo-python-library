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
        rolling += data[i] - data[i - period];
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
        rolling += data[i] - data[i - period];
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
        rsum += new - old;
        rsq += new * new - old * old;
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
        rsum += new - old;
        rsq += new * new - old * old;
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

/// Rolling maximum over `period`. NaN for the first `period-1` slots.
pub fn highest(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = nan_vec(n);
    if period == 0 {
        return result;
    }
    // monotonic decreasing deque of indices
    let mut dq: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    for i in 0..n {
        while let Some(&front) = dq.front() {
            if front + period <= i {
                dq.pop_front();
            } else {
                break;
            }
        }
        while let Some(&back) = dq.back() {
            if data[back] <= data[i] {
                dq.pop_back();
            } else {
                break;
            }
        }
        dq.push_back(i);
        if i + 1 >= period {
            result[i] = data[*dq.front().unwrap()];
        }
    }
    result
}

/// Rolling minimum over `period`. NaN for the first `period-1` slots.
pub fn lowest(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = nan_vec(n);
    if period == 0 {
        return result;
    }
    let mut dq: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    for i in 0..n {
        while let Some(&front) = dq.front() {
            if front + period <= i {
                dq.pop_front();
            } else {
                break;
            }
        }
        while let Some(&back) = dq.back() {
            if data[back] >= data[i] {
                dq.pop_back();
            } else {
                break;
            }
        }
        dq.push_back(i);
        if i + 1 >= period {
            result[i] = data[*dq.front().unwrap()];
        }
    }
    result
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
        sum_pv += data[i] * volume[i] - data[i - period] * volume[i - period];
        sum_v += volume[i] - volume[i - period];
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
    fn vwma_zero_volume_fallback() {
        let d = [10.0, 20.0, 30.0];
        let v = [0.0, 0.0, 0.0];
        let r = vwma(&d, &v, 2);
        approx(r[1], 20.0);
        approx(r[2], 30.0);
    }
}
