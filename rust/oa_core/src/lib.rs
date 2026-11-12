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
//!
//! One deliberate deviation from the legacy kernels: NaN *inside* the input is
//! handled rather than fed to a running accumulator. See the "Rolling reductions"
//! header below for the contract, and issue #2029 for why. Values for NaN-free
//! input are unchanged.

#[inline]
fn nan_vec(n: usize) -> Vec<f64> {
    vec![f64::NAN; n]
}

/// NaN -> 0.0, so a NaN can enter a sliding accumulator without poisoning it.
/// The caller tracks how many NaNs are in the window and withholds the output
/// until that count is back to zero, at which point the accumulator is exact.
#[inline]
fn nan_to_zero(x: f64) -> f64 {
    if x.is_nan() {
        0.0
    } else {
        x
    }
}

/// Start index of the first NaN-free window of `period` values, if any.
#[inline]
fn first_clean_window(data: &[f64], period: usize) -> Option<usize> {
    if period == 0 || data.len() < period {
        return None;
    }
    let mut run = 0usize;
    for (i, &x) in data.iter().enumerate() {
        run = if x.is_nan() { 0 } else { run + 1 };
        if run == period {
            return Some(i + 1 - period);
        }
    }
    None
}

/// Length of the shortest of several input series: the bars every one of them has.
#[inline]
fn shortest(lengths: &[usize]) -> usize {
    lengths.iter().copied().min().unwrap_or(0)
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
//
// NaN contract (see issue #2029). Every rolling-window kernel below is
// *window-local*: `out[i]` is NaN iff `i < period-1` or the window
// `[i-period+1 ..= i]` contains a NaN, and the series recovers as soon as the
// NaN slides out — the same rule as pandas `.rolling(period)`. The O(n)
// accumulators therefore never take a NaN in (a NaN added to a running sum
// poisons it for the rest of the series); instead NaNs contribute 0.0 and a
// running `nans` count decides whether the window is publishable. Because the
// sliding recurrences are algebraically exact, the accumulator is exact again
// the moment the window is NaN-free.
//
// Recursive kernels (the EMA family) cannot be window-local: they skip leading
// NaNs, seed at the first finite value, and carry their state across an
// interior NaN rather than letting it poison the recursion.
// ============================================================================

/// Simple Moving Average — O(n) rolling sum. NaN for the first `period-1` slots.
pub fn sma(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = nan_vec(n);
    if period == 0 || n < period {
        return result;
    }
    let mut rolling = 0.0;
    let mut nans = 0usize;
    for &x in data.iter().take(period) {
        if x.is_nan() {
            nans += 1;
        } else {
            rolling += x;
        }
    }
    if nans == 0 {
        result[period - 1] = rolling / period as f64;
    }
    let p = period as f64;
    for i in period..n {
        let new = data[i];
        let old = data[i - period];
        // Match the reference left-to-right association exactly: (r + new) - old.
        rolling = rolling + nan_to_zero(new) - nan_to_zero(old);
        nans += new.is_nan() as usize;
        nans -= old.is_nan() as usize;
        result[i] = if nans == 0 { rolling / p } else { f64::NAN };
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
    let mut nans = 0usize;
    for &x in data.iter().take(period) {
        if x.is_nan() {
            nans += 1;
        } else {
            rolling += x;
        }
    }
    if nans == 0 {
        result[period - 1] = rolling;
    }
    for i in period..n {
        let new = data[i];
        let old = data[i - period];
        rolling = rolling + nan_to_zero(new) - nan_to_zero(old);
        nans += new.is_nan() as usize;
        nans -= old.is_nan() as usize;
        result[i] = if nans == 0 { rolling } else { f64::NAN };
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
    let mut nans = 0usize;
    for &x in data.iter().take(period) {
        if x.is_nan() {
            nans += 1;
        } else {
            rsum += x;
            rsq += x * x;
        }
    }
    if nans == 0 {
        let mean = rsum / p;
        result[period - 1] = (rsq / p) - mean * mean;
    }
    for i in period..n {
        let old = data[i - period];
        let new = data[i];
        let (nv, ov) = (nan_to_zero(new), nan_to_zero(old));
        rsum = rsum + nv - ov;
        rsq = rsq + nv * nv - ov * ov;
        nans += new.is_nan() as usize;
        nans -= old.is_nan() as usize;
        let mean = rsum / p;
        result[i] = if nans == 0 {
            (rsq / p) - mean * mean
        } else {
            f64::NAN
        };
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
    let mut nans = 0usize;
    for &x in data.iter().take(period) {
        if x.is_nan() {
            nans += 1;
        } else {
            rsum += x;
            rsq += x * x;
        }
    }
    if nans == 0 {
        let mean = rsum / p;
        result[period - 1] = (rsq / p - mean * mean).max(0.0).sqrt();
    }
    for i in period..n {
        let old = data[i - period];
        let new = data[i];
        let (nv, ov) = (nan_to_zero(new), nan_to_zero(old));
        rsum = rsum + nv - ov;
        rsq = rsq + nv * nv - ov * ov;
        nans += new.is_nan() as usize;
        nans -= old.is_nan() as usize;
        let mean = rsum / p;
        // `f64::max` returns the *other* operand when one side is NaN, so a
        // poisoned accumulator used to surface as a silent 0.0 here rather than a
        // NaN. The `nans` guard is what keeps that from happening.
        result[i] = if nans == 0 {
            (rsq / p - mean * mean).max(0.0).sqrt()
        } else {
            f64::NAN
        };
    }
    result
}

// ============================================================================
// Moving averages
// ============================================================================

/// Exponential Moving Average — alpha = 2/(period+1), seeded with the first
/// finite value. Leading NaNs are skipped (they come out NaN); an interior NaN
/// comes out NaN for that bar only, with the recursion resuming from the last
/// finite value instead of being poisoned. For NaN-free input this is the
/// original first-value-seeded, full-length series.
pub fn ema(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = nan_vec(n);
    if n == 0 {
        return result;
    }
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut fv = 0usize;
    while fv < n && data[fv].is_nan() {
        fv += 1;
    }
    if fv == n {
        return result;
    }
    let mut prev = data[fv];
    result[fv] = prev;
    for i in fv + 1..n {
        let x = data[i];
        if !x.is_nan() {
            prev = alpha * x + (1.0 - alpha) * prev;
            result[i] = prev;
        }
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
    let mut nans = 0usize;
    for j in 0..period {
        let v = data[j];
        if v.is_nan() {
            nans += 1;
        } else {
            sum += v;
            wsum += v * (j + 1) as f64;
        }
    }
    if nans == 0 {
        result[period - 1] = wsum / weight_sum;
    }
    for i in period..n {
        let new = data[i];
        let old = data[i - period];
        let (nv, ov) = (nan_to_zero(new), nan_to_zero(old));
        wsum = wsum + p * nv - sum;
        sum = sum + nv - ov;
        nans += new.is_nan() as usize;
        nans -= old.is_nan() as usize;
        result[i] = if nans == 0 {
            wsum / weight_sum
        } else {
            f64::NAN
        };
    }
    result
}

/// Hull Moving Average: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n)).
///
/// The final WMA is an inline O(n) rolling pass over `2*WMA(n/2) - WMA(n)` (no
/// intermediate allocation), window-local in exactly the way `wma` is: the
/// accumulators take 0.0 in place of a NaN and `nans` withholds the output until
/// the window is clean again. For NaN-free input the first valid output is at
/// `(period-1) + (sqrt(period)-1)`, i.e. WMA(n)'s warm-up plus the outer WMA's;
/// NaN in the *input* pushes that out window-locally instead of blanking the rest.
pub fn hma(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = nan_vec(n);
    if period == 0 || n < period {
        return out;
    }
    let half = period / 2;
    let sqrt_p = (period as f64).sqrt() as usize;
    if half == 0 || sqrt_p == 0 || n < sqrt_p {
        return out;
    }
    let wh = wma(data, half);
    let wf = wma(data, period);
    let diff = |idx: usize| 2.0 * wh[idx] - wf[idx];
    let weight_sum = (sqrt_p * (sqrt_p + 1) / 2) as f64;
    let sp = sqrt_p as f64;
    let mut sum = 0.0;
    let mut wsum = 0.0;
    let mut nans = 0usize;
    for j in 0..sqrt_p {
        let v = diff(j);
        if v.is_nan() {
            nans += 1;
        } else {
            sum += v;
            wsum += v * (j + 1) as f64;
        }
    }
    if nans == 0 {
        out[sqrt_p - 1] = wsum / weight_sum;
    }
    // Slide: wsum += sqrt_p*x_new - sum; sum += x_new - x_leaving (same as `wma`).
    for i in sqrt_p..n {
        let v = diff(i);
        let leaving = diff(i - sqrt_p);
        let (nv, lv) = (nan_to_zero(v), nan_to_zero(leaving));
        wsum = wsum + sp * nv - sum;
        sum = sum + nv - lv;
        nans += v.is_nan() as usize;
        nans -= leaving.is_nan() as usize;
        out[i] = if nans == 0 { wsum / weight_sum } else { f64::NAN };
    }
    out
}

/// SMA-seeded EMA: alpha = 2/(period+1), seed = SMA at index period-1, NaN warm-up.
/// (Distinct from `ema` which first-value-seeds, and `ema_wilder` which uses 1/period.)
pub fn ema_sma(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut r = nan_vec(n);
    if period == 0 || n < period {
        return r;
    }
    // Seed on the first NaN-free window rather than blindly on data[..period],
    // so leading warm-up NaNs from an upstream indicator only delay the start.
    let fv = match first_clean_window(data, period) {
        Some(x) => x,
        None => return r,
    };
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut s = 0.0;
    for &x in data.iter().skip(fv).take(period) {
        s += x;
    }
    let start = fv + period - 1;
    let mut prev = s / period as f64;
    r[start] = prev;
    for i in start + 1..n {
        if data[i].is_nan() {
            continue;
        }
        prev = alpha * data[i] + (1.0 - alpha) * prev;
        r[i] = prev;
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
    let first_valid = match first_clean_window(data, period) {
        Some(x) => x,
        None => return result,
    };
    let mut sum_val = 0.0;
    for i in first_valid..first_valid + period {
        sum_val += data[i];
    }
    let start = first_valid + period - 1;
    result[start] = sum_val / period as f64;
    let pm1 = (period - 1) as f64;
    let p = period as f64;
    // Carry the recursion in `prev` and select rather than branch: this loop is a
    // serial dependency chain, so a mispredict costs more than the extra multiply.
    let mut prev = result[start];
    for i in start + 1..n {
        let v = data[i];
        prev = if v.is_nan() { prev } else { (prev * pm1 + v) / p };
        result[i] = prev;
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
        // `max3` compares with `>`, which is false against NaN, so a NaN leg can
        // lose to a finite one and vanish. An unknown bar must stay unknown.
        tr[i] = if hl.is_nan() || hc.is_nan() || lc.is_nan() {
            f64::NAN
        } else {
            max3(hl, hc, lc)
        };
    }
    tr
}

/// ATR with Wilder smoothing. Seed = simple average of first `period` TRs.
pub fn atr_wilder(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = high.len();
    if period == 0 || n < period {
        return nan_vec(n);
    }
    // Wilder smoothing of the true range: `ema_wilder` is the same recursion with
    // the leading-NaN handling already in it, so ATR inherits that rather than
    // re-seeding blindly at index 0 and poisoning `atr[i - 1]` for the rest of
    // the series the moment a NaN reaches the true range.
    ema_wilder(&true_range(high, low, close), period)
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
///
/// All NaN when the period is 0 or longer than the series: no window exists then,
/// and nothing is allocated for one.
#[inline]
fn _roll_extreme(data: &[f64], period: usize, want_max: bool) -> Vec<f64> {
    let n = data.len();
    let mut result = nan_vec(n);
    if period == 0 || period > n {
        return result;
    }
    let cap = period.next_power_of_two();
    let mask = cap - 1;
    let mut ring: Vec<usize> = vec![0usize; cap];
    let mut head = 0usize; // front counter
    let mut tail = 0usize; // back counter (len = tail - head)
    // Window-local in NaN: every comparison below is false against a NaN, so a NaN
    // would otherwise sit in the deque undroppable and be reported as the extreme.
    // NaNs are kept out of the deque entirely and counted instead.
    let mut nans = 0usize;
    for i in 0..n {
        if i >= period && data[i - period].is_nan() {
            nans -= 1;
        }
        if head < tail && i - ring[head & mask] >= period {
            head += 1;
        }
        let x = data[i];
        if x.is_nan() {
            nans += 1;
        } else {
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
        }
        if i + 1 >= period && nans == 0 {
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
    let bad = |i: usize| data[i].is_nan() || volume[i].is_nan();
    let mut sum_pv = 0.0;
    let mut sum_v = 0.0;
    let mut nans = 0usize;
    for i in 0..period {
        if bad(i) {
            nans += 1;
        } else {
            sum_pv += data[i] * volume[i];
            sum_v += volume[i];
        }
    }
    if nans == 0 {
        result[period - 1] = if sum_v > 0.0 {
            sum_pv / sum_v
        } else {
            data[period - 1]
        };
    }
    for i in period..n {
        let (new_bad, old_bad) = (bad(i), bad(i - period));
        let new_v = if new_bad { 0.0 } else { volume[i] };
        let new_pv = if new_bad { 0.0 } else { data[i] * volume[i] };
        let old_v = if old_bad { 0.0 } else { volume[i - period] };
        let old_pv = if old_bad {
            0.0
        } else {
            data[i - period] * volume[i - period]
        };
        sum_pv = sum_pv + new_pv - old_pv;
        sum_v = sum_v + new_v - old_v;
        nans += new_bad as usize;
        nans -= old_bad as usize;
        result[i] = if nans != 0 {
            f64::NAN
        } else if sum_v > 0.0 {
            sum_pv / sum_v
        } else {
            data[i]
        };
    }
    result
}

/// Chande Momentum Oscillator: 100 * (up - down) / (up + down), where `up` and
/// `down` are the sums of the rises and of the falls among the last `period`
/// bar-to-bar changes. 0.0 when nothing moved. NaN for the first `period` bars
/// and while a NaN is among the `period + 1` bars those changes span.
pub fn cmo(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = nan_vec(n);
    if period >= n {
        return result;
    }
    let mut changes = vec![0.0f64; n - 1];
    for i in 1..n {
        changes[i - 1] = data[i] - data[i - 1];
    }
    let mut up = 0.0;
    let mut down = 0.0;
    // A change next to a NaN bar is NaN: neither a rise nor a fall. Such changes
    // are counted, and a window holding one publishes nothing.
    let mut unknown = 0usize;
    for &ch in changes.iter().take(period) {
        if ch > 0.0 {
            up += ch;
        } else if ch < 0.0 {
            down += ch.abs();
        }
        unknown += ch.is_nan() as usize;
    }
    let oscillator = |up: f64, down: f64| {
        let total = up + down;
        if total > 0.0 {
            ((up - down) / total) * 100.0
        } else {
            0.0
        }
    };
    if unknown == 0 {
        result[period] = oscillator(up, down);
    }
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
        unknown += new.is_nan() as usize;
        unknown -= old.is_nan() as usize;
        if unknown == 0 {
            result[i] = oscillator(up, down);
        }
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

/// KAMA with integer EMA lengths and a full-window efficiency ratio.
/// er = |chg(length)| / sum(|chg(1)|, length), the volatility sum recomputed over
/// the whole window at every bar (no running sum); alpha = (er * (fast_alpha -
/// slow_alpha) + slow_alpha)^2 with fast_alpha = 2 / (fast_length + 1) and
/// slow_alpha = 2 / (slow_length + 1). The previous value is the source itself at
/// the first bar and after a NaN output. Matches the
/// `_calculate_kama_full_window` reference bit-for-bit.
pub fn kama_full_window(
    data: &[f64],
    length: usize,
    fast_length: usize,
    slow_length: usize,
) -> Vec<f64> {
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

/// Supertrend (hl2 +- multiplier * ATR bands that flip on a close through the
/// active band). Returns (supertrend, direction) where
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

/// Bars of the trailing mean FRAMA applies to its filter.
const FRAMA_SMOOTHING: usize = 5;

/// FRAMA (fractal adaptive MA on hl2, with a trailing 5-bar mean).
/// Uses log/exp so parity is transcendental-tolerance, not bit-exact.
///
/// Bars are counted from the first one that has a midpoint (high + low) / 2.
/// For the first `period` bars the filter is the midpoint. After that the
/// fractal dimension D = (ln(N1 + N2) - ln(N3)) / ln 2 compares the high-low
/// range of the last `period` bars (N3, per bar) with the ranges of their newer
/// and older halves (N1, N2, per bar of a half), D = 1 when a range is not
/// positive, and filter = alpha * midpoint + (1 - alpha) * previous filter with
/// alpha = exp(-4.6 * (D - 1)) held between 0.01 and 1. The output is the
/// midpoint for the first `period + 1` bars and the 5-bar mean of the filter
/// from then on.
///
/// The filter is NaN while a NaN bar is among the `period` bars of its ranges
/// and resumes from its last value afterwards; the output is NaN while a NaN
/// filter value is among the five it averages. Bars before the first midpoint
/// and beyond the end of `low` are NaN.
pub fn frama(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    let mut out = nan_vec(high.len());
    let bars = high.len().min(low.len());
    let price: Vec<f64> = (0..bars).map(|i| (high[i] + low[i]) / 2.0).collect();
    let Some(first) = price.iter().position(|p| !p.is_nan()) else {
        return out;
    };
    // NaN midpoints before each bar, so a window is checked with one subtraction.
    let mut unknown_before = vec![0usize; bars + 1];
    for i in 0..bars {
        unknown_before[i + 1] = unknown_before[i] + price[i].is_nan() as usize;
    }
    let half = period / 2;
    let ln2 = 2.0_f64.ln();
    let mut filt = nan_vec(bars);
    let mut last_filt = f64::NAN;
    for i in first..bars {
        if i - first >= period && half > 0 {
            if unknown_before[i + 1] != unknown_before[i + 1 - period] {
                continue;
            }
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
            filt[i] = alpha * price[i] + (1.0 - alpha) * last_filt;
        } else {
            filt[i] = price[i];
        }
        if !filt[i].is_nan() {
            last_filt = filt[i];
        }
    }
    for i in first..bars {
        let age = i - first;
        if age <= period {
            out[i] = price[i];
        } else if age >= FRAMA_SMOOTHING - 1 {
            // A NaN among the five makes the sum, and so the bar, NaN.
            let mut s = 0.0;
            for value in &filt[i + 1 - FRAMA_SMOOTHING..=i] {
                s += value;
            }
            out[i] = s / FRAMA_SMOOTHING as f64;
        } else {
            out[i] = filt[i];
        }
    }
    out
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
    // Seed on the first NaN-free run of period+1 values (= period deltas). A NaN
    // delta is neither > 0 nor < 0, so seeding blindly at index 0 used to fold an
    // upstream indicator's warm-up into avg_gain == avg_loss == 0 and publish the
    // avg_loss == 0 branch: a confident, wrong RSI of 100 over the whole prefix.
    let fv = match first_clean_window(data, period + 1) {
        Some(x) => x,
        None => return result,
    };
    let p = period as f64;
    let pm1 = (period - 1) as f64;
    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;
    for i in fv..fv + period {
        let d = data[i + 1] - data[i];
        if d > 0.0 {
            avg_gain += d;
        } else if d < 0.0 {
            avg_loss += -d;
        }
    }
    avg_gain /= p;
    avg_loss /= p;
    let start = fv + period;
    result[start] = if avg_loss == 0.0 {
        100.0
    } else {
        100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    };
    for i in start..n - 1 {
        let d = data[i + 1] - data[i];
        if d.is_nan() {
            // Bar is unknown, not flat: leave NaN and keep the averages intact.
            continue;
        }
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

/// Anchored VWAP. `starts[i] != 0` (or i == 0) opens a new period and resets the
/// running sums. Returns (vwap, stdev): vwap = sum(price * volume) / sum(volume)
/// over the period so far, stdev = sqrt(max(0, E[price^2] - vwap^2)) with the
/// same volume weights.
///
/// Both are NaN on a bar whose price or volume is NaN (the sums keep their state
/// and the next bar continues from them) and while the period has no volume yet.
/// The output has the length of `source`; bars beyond the end of `volume` are
/// NaN and a missing `starts` entry does not open a period.
pub fn session_vwap(source: &[f64], volume: &[f64], starts: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = source.len();
    let mut vwap = nan_vec(n);
    let mut sd = nan_vec(n);
    let mut spv = 0.0;
    let mut sv = 0.0;
    let mut spv2 = 0.0;
    for i in 0..n.min(volume.len()) {
        if i == 0 || starts.get(i).is_some_and(|&s| s != 0.0) {
            spv = 0.0;
            sv = 0.0;
            spv2 = 0.0;
        }
        if source[i].is_nan() || volume[i].is_nan() {
            continue;
        }
        spv += source[i] * volume[i];
        sv += volume[i];
        spv2 += source[i] * source[i] * volume[i];
        if sv > 0.0 {
            vwap[i] = spv / sv;
            let variance = spv2 / sv - vwap[i].powf(2.0);
            sd[i] = variance.max(0.0).sqrt();
        }
    }
    (vwap, sd)
}

/// Parabolic SAR. Returns (sar, trend), trend +1 while long (stop below price)
/// and -1 while short.
///
/// State: the stop, the extreme point EP of the current trend and the
/// acceleration factor AF. The first bar that has both a high and a low starts a
/// long trend with stop = low, EP = high, AF = `acceleration`. No close series
/// is taken, so the first trend is not read off the first two closes: it is
/// always long, and a falling market reverses it on the next bar.
///
/// Every later bar, in this order:
/// 1. stop = stop + AF * (EP - stop).
/// 2. Reversal, tested against that unclamped stop with strict comparisons.
///    Long and stop > low: go short, stop = max(high, EP), EP = low, AF =
///    `acceleration`. Short and stop < high: go long, stop = min(low, EP),
///    EP = high, AF = `acceleration`. A bar that only touches the stop does not
///    reverse.
/// 3. Without a reversal, a new extreme (high > EP when long, low < EP when
///    short) becomes EP and raises AF by `acceleration`, up to `maximum`.
/// 4. Clamp, on reversal bars too: when long the stop is at most the lows of the
///    two previous bars, when short at least their highs.
///
/// A bar with a NaN high or low is skipped: its outputs are NaN, the state is
/// kept, and it does not count as one of the two previous bars. The output has
/// the length of `high`; bars beyond the end of `low` are NaN.
pub fn sar(high: &[f64], low: &[f64], acceleration: f64, maximum: f64) -> (Vec<f64>, Vec<f64>) {
    let n = high.len();
    let mut sar = nan_vec(n);
    let mut trend = nan_vec(n);
    let mut long = true;
    let mut stop = f64::NAN;
    let mut ep = f64::NAN;
    let mut af = acceleration;
    // (high, low) of the last bar that had both, and of the one before it.
    let mut prev: Option<(f64, f64)> = None;
    let mut prev2: Option<(f64, f64)> = None;
    for i in 0..n.min(low.len()) {
        let (h, l) = (high[i], low[i]);
        if h.is_nan() || l.is_nan() {
            continue;
        }
        let Some((prev_high, prev_low)) = prev else {
            stop = l;
            ep = h;
            sar[i] = stop;
            trend[i] = 1.0;
            prev = Some((h, l));
            continue;
        };
        stop += af * (ep - stop);
        let mut reversed = false;
        if long {
            if stop > l {
                reversed = true;
                long = false;
                stop = h.max(ep);
                ep = l;
                af = acceleration;
            }
        } else if stop < h {
            reversed = true;
            long = true;
            stop = l.min(ep);
            ep = h;
            af = acceleration;
        }
        if !reversed {
            if long {
                if h > ep {
                    ep = h;
                    af = (af + acceleration).min(maximum);
                }
            } else if l < ep {
                ep = l;
                af = (af + acceleration).min(maximum);
            }
        }
        if long {
            stop = stop.min(prev_low);
            if let Some((_, low2)) = prev2 {
                stop = stop.min(low2);
            }
        } else {
            stop = stop.max(prev_high);
            if let Some((high2, _)) = prev2 {
                stop = stop.max(high2);
            }
        }
        sar[i] = stop;
        trend[i] = if long { 1.0 } else { -1.0 };
        prev2 = prev;
        prev = Some((h, l));
    }
    (sar, trend)
}

/// On Balance Volume. obv[0]=0; cumulative sign(change)*volume where a flat close
/// (close==prev) contributes 0.
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

/// Chaikin Money Flow: sum(money flow volume, period) / sum(volume, period),
/// where money flow volume is ((close - low) - (high - close)) / (high - low) *
/// volume, zero on a bar without range. 0.0 when the window's volume is not
/// positive. NaN for the first `period - 1` bars and while a NaN high, low,
/// close or volume is in the window. Bars beyond the shortest input are NaN.
pub fn cmf(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], period: usize) -> Vec<f64> {
    let mut r = nan_vec(close.len());
    let bars = shortest(&[close.len(), high.len(), low.len(), volume.len()]);
    if period == 0 || bars < period {
        return r;
    }
    for i in period - 1..bars {
        let mut smfv = 0.0;
        let mut sv = 0.0;
        let mut known = true;
        for idx in i + 1 - period..=i {
            let (h, l, c, v) = (high[idx], low[idx], close[idx], volume[idx]);
            if h.is_nan() || l.is_nan() || c.is_nan() || v.is_nan() {
                known = false;
                break;
            }
            let mfm = if h != l { ((c - l) - (h - c)) / (h - l) } else { 0.0 };
            smfv += mfm * v;
            sv += v;
        }
        if known {
            r[i] = if sv > 0.0 { smfv / sv } else { 0.0 };
        }
    }
    r
}

/// Money Flow Index: 100 - 100 / (1 + positive flow / negative flow) over the
/// last `period` bars, 100 when the window has no negative flow. A bar's flow is
/// typical price * volume, positive when the typical price rose from the bar
/// before, negative when it fell, and zero when it did not move or when there is
/// no bar before to compare with (bar 0, or the bar after a NaN one).
///
/// NaN on bar 0, for the first `period - 1` bars, and while a bar with a NaN
/// price or volume is in the window. Bars beyond the shortest input are NaN.
pub fn mfi(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], period: usize) -> Vec<f64> {
    let mut r = nan_vec(close.len());
    let bars = shortest(&[close.len(), high.len(), low.len(), volume.len()]);
    if period == 0 {
        return r;
    }
    let tp: Vec<f64> = (0..bars)
        .map(|i| (high[i] + low[i] + close[i]) / 3.0)
        .collect();
    let unknown = |i: usize| tp[i].is_nan() || volume[i].is_nan();
    let mut pos = vec![0.0f64; bars];
    let mut neg = vec![0.0f64; bars];
    for i in 1..bars {
        if unknown(i) || tp[i - 1].is_nan() {
            continue;
        }
        let rmf = tp[i] * volume[i];
        if tp[i] > tp[i - 1] {
            pos[i] = rmf;
        } else if tp[i] < tp[i - 1] {
            neg[i] = rmf;
        }
    }
    let mut ps = 0.0;
    let mut ns = 0.0;
    let mut unknown_bars = if bars > 0 { unknown(0) as usize } else { 0 };
    for i in 1..bars {
        ps += pos[i];
        ns += neg[i];
        unknown_bars += unknown(i) as usize;
        if i >= period {
            ps -= pos[i - period];
            ns -= neg[i - period];
            unknown_bars -= unknown(i - period) as usize;
        }
        if i >= period - 1 && unknown_bars == 0 {
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
    // Carry the recursion in `prev`, not in `r[i - 1]`: an interior NaN leaves a
    // NaN in `r` and reading it back would poison every later bar.
    let mut prev = r[fv];
    for i in fv + 1..n {
        if !data[i].is_nan() {
            prev = alpha * data[i] + (1.0 - alpha) * prev;
            r[i] = prev;
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

/// Fisher Transform. Input is the price source (hl2). Returns
/// (fisher, trigger). Per bar, with the highest and lowest of the last `length`
/// values: value = 0.66 * ((price - lowest) / (highest - lowest) - 0.5) + 0.67 *
/// value[1], held at +-0.999 beyond +-0.99, and fisher = 0.5 * ln((1 + value) /
/// (1 - value)) + 0.5 * fisher[1]. A window without range repeats the previous
/// fisher. The trigger is the fisher value before the bar, 0.0 at the start.
///
/// Both are NaN for the first `length - 1` bars and while a NaN is in the
/// window. `value` and `fisher` keep their state across such bars, so the first
/// bar after a NaN warm-up starts exactly as bar `length - 1` of a clean series.
pub fn fisher(data: &[f64], length: usize) -> (Vec<f64>, Vec<f64>) {
    let n = data.len();
    let mut fish = nan_vec(n);
    let mut trigger = nan_vec(n);
    if length == 0 || n < length {
        return (fish, trigger);
    }
    let mut value = 0.0;
    let mut fish_prev = 0.0;
    let mut nans = data[..length - 1].iter().filter(|x| x.is_nan()).count();
    for i in length - 1..n {
        nans += data[i].is_nan() as usize;
        if i >= length {
            nans -= data[i - length].is_nan() as usize;
        }
        if nans > 0 {
            continue;
        }
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
        trigger[i] = fish_prev;
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
            fish_prev = log_term + 0.5 * fish_prev;
        }
        fish[i] = fish_prev;
    }
    (fish, trigger)
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

/// Linear-regression slope as the change of the regression end value between two
/// consecutive windows: (endval(cur) - endval(prev)) / interval.
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

/// Williams Fractals (up, down) booleans. An up fractal needs the `n` highs before
/// the centre strictly below its high; after it, any of five patterns qualifies:
/// `n` strictly lower highs, or 1 to 4 highs that do not exceed the centre high
/// followed by `n` strictly lower highs. Down fractals mirror this on the lows.
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
        let w = &data[i + 1 - period..i + 1];
        if w.iter().any(|x| x.is_nan()) {
            // Window-local, and NaN has no total order: `partial_cmp` returns None
            // for it, so sorting a window that contains one used to panic straight
            // through the PyO3 boundary (a BaseException, not a catchable error).
            continue;
        }
        buf.copy_from_slice(w);
        buf.sort_by(f64::total_cmp);
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
/// Definition: John Ehlers, "Relative Vigor Index", 2002:
/// sum(swma(close - open), len) / sum(swma(high - low), len), where swma is the
/// symmetric 4-bar weighted mean (1, 2, 2, 1) / 6.
///
/// Window-local in NaN, like the rolling reductions above: `rvi[i]` is NaN while
/// any bar feeding its window is NaN. All NaN when the window does not fit the
/// series. Bars beyond the shortest input are NaN.
pub fn rvi_vigor(
    open_: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = close.len();
    let bars = shortest(&[n, open_.len(), high.len(), low.len()]);
    let co: Vec<f64> = (0..bars).map(|i| close[i] - open_[i]).collect();
    let hl: Vec<f64> = (0..bars).map(|i| high[i] - low[i]).collect();
    let sco = swma_vec(&co);
    let shl = swma_vec(&hl);
    let mut rvi = nan_vec(n);
    // The first window ends at bar period + 2 (the swma needs three bars before
    // it). A period too large for that has no window, and the loop does not run.
    for i in period.saturating_add(2)..bars {
        let mut ns = 0.0;
        let mut ds = 0.0;
        for j in i + 1 - period..i + 1 {
            ns += sco[j];
            ds += shl[j];
        }
        // A NaN sum means a NaN bar fed the window: leave the bar NaN.
        if !ns.is_nan() && !ds.is_nan() {
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

/// Largest `n` that `valuewhen` resolves.
const VALUEWHEN_LOOKBACK_CAP: usize = 1000;

/// valuewhen: value of `array` when `expr` (nonzero == true) was true the n-th most
/// recent time. NaN until `expr` has been true `n` times, and for every bar when
/// `n` is 0 or above `min(VALUEWHEN_LOOKBACK_CAP, len)`. A bar whose value lies
/// beyond the end of `array` is NaN.
pub fn valuewhen(expr: &[f64], array: &[f64], n: usize) -> Vec<f64> {
    let length = expr.len();
    let mut result = nan_vec(length);
    if n == 0 || n > std::cmp::min(VALUEWHEN_LOOKBACK_CAP, length) {
        return result;
    }
    // Only the last `n` true bars matter, so they live in a ring of `n` slots.
    let mut ring = vec![0usize; n];
    let mut count = 0usize; // true bars seen so far
    for i in 0..length {
        if expr[i] != 0.0 {
            ring[count % n] = i;
            count += 1;
        }
        if count >= n {
            // The n-th most recent true bar is the oldest of the last n, which
            // is the slot the next true bar will overwrite.
            if let Some(&value) = array.get(ring[count % n]) {
                result[i] = value;
            }
        }
    }
    result
}

/// T3 (Tillson) = gd(gd(gd(data))), gd(x) = (1+v)*EMA(x) - v*EMA(EMA(x)). Computed
/// entirely in Rust (6 EMA passes, no per-stage numpy temps / PyO3 round-trips).
pub fn t3(data: &[f64], period: usize, v: f64) -> Vec<f64> {
    fn gd(x: &[f64], period: usize, v: f64) -> Vec<f64> {
        let e1 = ema(x, period);
        let e2 = ema(&e1, period);
        let mut out = vec![0.0f64; x.len()];
        for i in 0..x.len() {
            out[i] = (1.0 + v) * e1[i] - v * e2[i];
        }
        out
    }
    let g1 = gd(data, period, v);
    let g2 = gd(&g1, period, v);
    gd(&g2, period, v)
}

/// MIDPRICE = (highest(high,period) + lowest(low,period)) / 2, fused single pass.
pub fn midprice(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    let hi = _roll_extreme(high, period, true);
    let lo = _roll_extreme(low, period, false);
    let n = high.len();
    let mut out = nan_vec(n);
    for i in 0..n {
        if !hi[i].is_nan() {
            out[i] = (hi[i] + lo[i]) / 2.0;
        }
    }
    out
}

/// MIDPOINT = (highest(data,period) + lowest(data,period)) / 2 over one series.
pub fn midpoint(data: &[f64], period: usize) -> Vec<f64> {
    let hi = _roll_extreme(data, period, true);
    let lo = _roll_extreme(data, period, false);
    let n = data.len();
    let mut out = nan_vec(n);
    for i in 0..n {
        if !hi[i].is_nan() {
            out[i] = (hi[i] + lo[i]) / 2.0;
        }
    }
    out
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

/// Chaikin A/D Oscillator (ADOSC) = EMA(AD, fast) - EMA(AD, slow), where AD
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
// Directional movement as Wilder running sums. Each sum is seeded with the plain
// sum of the first `period - 1` one-bar movements (bars 1..period-1) and then
// updated as sum = sum - sum / period + today. This is a different seed and warm-up
// from `adx`, which averages the first `period` bars (bar 0 included).
// ---------------------------------------------------------------------------

/// PLUS_DM: Wilder running sum of +DM1. First value at index period-1
/// = sum of +DM1 over bars 1..period-1; then dm = dm - dm/period + +DM1.
pub fn plus_dm_wilder_sum(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
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

/// MINUS_DM (symmetric to plus_dm_wilder_sum).
pub fn minus_dm_wilder_sum(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
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

/// DX. First value at index period (Wilder-summed +DI/-DI then
/// 100*|+DI - -DI|/(+DI + -DI)).
pub fn dx_wilder_sum(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
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

/// ADX over Wilder-summed DX. First value at index 2*period-1 (mean of `period`
/// DX values), then Wilder recursion adx = (adx*(period-1) + dx)/period.
pub fn adx_wilder_sum(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
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

/// ADXR = (ADX[i] + ADX[i-(period-1)]) / 2, over `adx_wilder_sum`.
pub fn adxr_wilder_sum(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = high.len();
    let adx = adx_wilder_sum(high, low, close, period);
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

/// STOCHF -> (fastk, fastd). fastk = 100*(c-LL)/(HH-LL) over fastk_period,
/// fastd = SMA(fastk, fastd_period). Both start at the same first index
/// (fastk_period-1)+(fastd_period-1): fastk is NaN until fastd exists.
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

/// LINEARREG_ANGLE = degrees(atan(OLS slope)) over `period`. O(n).
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

/// LINEARREG_INTERCEPT = OLS intercept b over `period`. O(n).
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
    fn hma_nan_safe_and_aligned() {
        let n = 100usize;
        let d: Vec<f64> = (0..n)
            .map(|i| i as f64 + 1.0 + (i as f64).sin() * 5.0)
            .collect();
        let period = 16usize;
        let sqrt_p = (period as f64).sqrt() as usize; // 4
        let r = hma(&d, period);
        // First valid output at (period-1)+(sqrt_p-1); everything before is NaN.
        let first = (period - 1) + (sqrt_p - 1);
        assert!(r[first - 1].is_nan());
        assert!(r[first].is_finite());
        // The rolling pass must not get poisoned: the whole valid tail stays finite.
        assert!(r[first..].iter().all(|x| x.is_finite()));

        // Brute-force reference: full-window WMA over the finite diff region only.
        let wh = wma(&d, period / 2);
        let wf = wma(&d, period);
        let ws = (sqrt_p * (sqrt_p + 1) / 2) as f64;
        for i in first..n {
            let mut acc = 0.0;
            for j in 0..sqrt_p {
                let idx = i - sqrt_p + 1 + j;
                acc += (2.0 * wh[idx] - wf[idx]) * (j + 1) as f64;
            }
            assert!((r[i] - acc / ws).abs() < 1e-9);
        }
    }

    #[test]
    fn wilder_sum_dm_family_alignment() {
        let n = 60usize;
        let high: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.3).sin() * 5.0 + i as f64 * 0.1).collect();
        let low: Vec<f64> = high.iter().map(|h| h - 2.0).collect();
        let close: Vec<f64> = high.iter().map(|h| h - 1.0).collect();
        let p = 14;
        let pdm = plus_dm_wilder_sum(&high, &low, p);
        let mdm = minus_dm_wilder_sum(&high, &low, p);
        // first valid at period-1
        assert!(pdm[p - 2].is_nan() && pdm[p - 1].is_finite());
        assert!(mdm[p - 1].is_finite());
        let dx = dx_wilder_sum(&high, &low, &close, p);
        assert!(dx[p - 1].is_nan() && dx[p].is_finite()); // first DX at index period
        let adx = adx_wilder_sum(&high, &low, &close, p);
        assert!(adx[2 * p - 2].is_nan() && adx[2 * p - 1].is_finite()); // 2*period-1
        let adxr = adxr_wilder_sum(&high, &low, &close, p);
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
        approx(r[3], 150.0); // equal close -> contributes 0
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
    fn kama_full_window_seed_and_trend() {
        let d = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let r = kama_full_window(&d, 3, 2, 30);
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

    // ------------------------------------------------------------------
    // NaN contract (issue #2029)
    //
    // Rolling-window kernels are window-local: output[i] is NaN iff i < period-1
    // or the window [i-period+1 ..= i] contains a NaN, and the series recovers
    // once the NaN slides out. Recursive kernels skip leading NaNs, seed at the
    // first finite value, and carry their state across an interior NaN.
    // ------------------------------------------------------------------

    /// 1.0 ..= 40.0 with NaN punched in at `holes`.
    fn ramp_with_nan(holes: &[usize]) -> Vec<f64> {
        let mut d: Vec<f64> = (1..=40).map(|x| x as f64).collect();
        for &h in holes {
            d[h] = f64::NAN;
        }
        d
    }

    fn nan_count(v: &[f64]) -> usize {
        v.iter().filter(|x| x.is_nan()).count()
    }

    /// Indices whose `period`-wide window contains one of `holes`, plus warm-up.
    fn expected_nan_idx(n: usize, period: usize, holes: &[usize]) -> Vec<usize> {
        (0..n)
            .filter(|&i| {
                i + 1 < period
                    || holes
                        .iter()
                        .any(|&h| h + period > i && h <= i)
            })
            .collect()
    }

    fn assert_window_local(out: &[f64], period: usize, holes: &[usize]) {
        let want = expected_nan_idx(out.len(), period, holes);
        let got: Vec<usize> = (0..out.len()).filter(|&i| out[i].is_nan()).collect();
        assert_eq!(got, want, "NaN positions differ");
    }

    #[test]
    fn sma_nan_is_window_local() {
        for holes in [vec![0usize], vec![0, 1, 2], vec![20]] {
            let d = ramp_with_nan(&holes);
            assert_window_local(&sma(&d, 9), 9, &holes);
        }
        // Values after recovery match the clean series exactly.
        let clean = ramp_with_nan(&[]);
        let holed = ramp_with_nan(&[20]);
        approx(sma(&holed, 9)[29], sma(&clean, 9)[29]);
    }

    #[test]
    fn wma_nan_is_window_local() {
        for holes in [vec![0usize], vec![0, 1, 2], vec![20]] {
            let d = ramp_with_nan(&holes);
            assert_window_local(&wma(&d, 9), 9, &holes);
        }
        let clean = ramp_with_nan(&[]);
        let holed = ramp_with_nan(&[20]);
        approx(wma(&holed, 9)[29], wma(&clean, 9)[29]);
    }

    #[test]
    fn rolling_sum_nan_is_window_local() {
        let holes = [20usize];
        let d = ramp_with_nan(&holes);
        assert_window_local(&rolling_sum(&d, 9), 9, &holes);
        approx(rolling_sum(&d, 9)[29], rolling_sum(&ramp_with_nan(&[]), 9)[29]);
    }

    #[test]
    fn stdev_nan_is_window_local_not_zero() {
        // Regression: `NaN.max(0.0)` returns 0.0 in Rust, so a poisoned accumulator
        // silently produced 0.0 for the rest of the series instead of NaN.
        let holes = [0usize];
        let d = ramp_with_nan(&holes);
        let r = stdev(&d, 9);
        assert_window_local(&r, 9, &holes);
        approx(r[9], stdev(&ramp_with_nan(&[]), 9)[9]);
        assert!(r[9] > 0.0, "stdev collapsed to {}", r[9]);
    }

    #[test]
    fn rolling_variance_nan_is_window_local() {
        let holes = [20usize];
        let d = ramp_with_nan(&holes);
        assert_window_local(&rolling_variance(&d, 9), 9, &holes);
    }

    #[test]
    fn vwma_nan_is_window_local() {
        let holes = [20usize];
        let d = ramp_with_nan(&holes);
        let v = vec![100.0f64; 40];
        assert_window_local(&vwma(&d, &v, 9), 9, &holes);
        // A NaN in the volume leg is window-local too.
        let mut v2 = vec![100.0f64; 40];
        v2[5] = f64::NAN;
        assert_window_local(&vwma(&ramp_with_nan(&[]), &v2, 9), 9, &[5]);
    }

    #[test]
    fn hma_nan_is_window_local() {
        // HMA(9) = WMA(2*WMA(.,4) - WMA(.,9), 3): first valid at (9-1)+(3-1) = 10,
        // and a hole at h blanks every output whose combined window touches it.
        let d = ramp_with_nan(&[0]);
        let r = hma(&d, 9);
        assert_eq!(nan_count(&r), 11);
        assert!(!r[11].is_nan());
        approx(r[20], hma(&ramp_with_nan(&[]), 9)[20]);
    }

    #[test]
    fn ema_skips_leading_nan_and_survives_a_hole() {
        let d = ramp_with_nan(&[0, 1, 2]);
        let r = ema(&d, 9);
        assert!(r[0].is_nan() && r[1].is_nan() && r[2].is_nan());
        approx(r[3], 4.0); // seeded with the first finite value
        assert_eq!(nan_count(&r), 3);

        let mid = ramp_with_nan(&[20]);
        let rm = ema(&mid, 9);
        assert_eq!(nan_count(&rm), 1, "an interior NaN must not poison the state");
        assert!(rm[20].is_nan() && !rm[21].is_nan());
    }

    #[test]
    fn ema_sma_skips_leading_nan() {
        let d = ramp_with_nan(&[0, 1, 2]);
        let r = ema_sma(&d, 9);
        // Seed is the SMA of data[3..12], first output at index 11.
        assert_eq!(nan_count(&r), 11);
        approx(r[11], (4..=12).map(|x| x as f64).sum::<f64>() / 9.0);
    }

    #[test]
    fn ema_first_valid_survives_an_interior_nan() {
        let d = ramp_with_nan(&[20]);
        let r = ema_first_valid(&d, 9);
        assert_eq!(nan_count(&r), 1);
        assert!(r[20].is_nan() && !r[21].is_nan());
    }

    #[test]
    fn chained_indicators_produce_signal() {
        // The reported reproduction: wma -> rsi -> wma must not blank out.
        let n = 400;
        let src: Vec<f64> = (0..n)
            .map(|i| 100.0 + 10.0 * ((i as f64) / 7.0).sin() + (i as f64) * 0.01)
            .collect();
        let base = wma(&src, 55);
        let r = rsi(&base, 14);
        let avg = wma(&r, 9);
        let crossings = (1..n)
            .filter(|&i| {
                let (a0, b0, a1, b1) = (r[i - 1], avg[i - 1], r[i], avg[i]);
                a0.is_finite() && b0.is_finite() && a1.is_finite() && b1.is_finite()
                    && a0 <= b0 && a1 > b1
            })
            .count();
        assert!(crossings > 0, "expected crossovers, found none");
    }

    #[test]
    fn rsi_skips_upstream_warmup_instead_of_printing_100() {
        // A NaN delta is neither a gain nor a loss, so the old seeding turned an
        // upstream indicator's warm-up into avg_loss == 0 -> a flat RSI of 100.
        let mut d: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        for v in d.iter_mut().take(20) {
            *v = f64::NAN;
        }
        let r = rsi(&d, 14);
        for (i, v) in r.iter().enumerate().take(34) {
            assert!(v.is_nan(), "rsi[{i}] = {v}, expected NaN inside the warm-up");
        }
        assert!(!r[34].is_nan());
        // Seeded at the first clean window, it matches the same slice run alone.
        let tail: Vec<f64> = d[20..].to_vec();
        approx(r[34], rsi(&tail, 14)[14]);
    }

    #[test]
    fn median_window_with_nan_is_nan_not_a_panic() {
        let mut d: Vec<f64> = (1..=30).map(|x| x as f64).collect();
        d[5] = f64::NAN;
        let r = median(&d, 5);
        // Warm-up 0..=3, then the five windows that span the hole at index 5.
        for i in (0..4).chain(5..10) {
            assert!(r[i].is_nan(), "median[{i}] = {}", r[i]);
        }
        approx(r[4], 3.0);
        approx(r[10], 9.0);
    }

    // ------------------------------------------------------------------
    // Parabolic SAR. Expected values are worked out by hand from the order
    // documented on `sar`, with acceleration 0.02 and maximum 0.2 unless a
    // test says otherwise.
    // ------------------------------------------------------------------

    const SAR_STEP: f64 = 0.02;
    const SAR_MAX: f64 = 0.2;
    const UP: f64 = 1.0;
    const DOWN: f64 = -1.0;

    fn approx_all(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g.is_nan() && w.is_nan()) || (g - w).abs() <= 1e-12,
                "bar {i}: expected {w}, got {g}"
            );
        }
    }

    /// Five rising bars, then bar 5 is an outside bar: a new high (15, above the
    /// extreme point 14) and a low (9.5) below the projected stop 9.93704.
    fn outside_bar_ends_an_uptrend() -> ([f64; 9], [f64; 9]) {
        (
            [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 14.5, 14.2, 14.0],
            [9.0, 10.0, 11.0, 12.0, 13.0, 9.5, 13.5, 13.0, 12.5],
        )
    }

    /// Bar 1 reverses the initial uptrend, bars 2 to 4 fall, then bar 5 is an
    /// outside bar: a new low (14, below the extreme point 15) and a high (21.5)
    /// above the projected stop 19.275424.
    fn outside_bar_ends_a_downtrend() -> ([f64; 7], [f64; 7]) {
        (
            [20.0, 19.0, 18.0, 17.0, 16.0, 21.5, 20.0],
            [19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 14.6],
        )
    }

    #[test]
    fn sar_uptrend_follows_the_hand_computed_path() {
        let (h, l) = outside_bar_ends_an_uptrend();
        let (s, t) = sar(&h[..5], &l[..5], SAR_STEP, SAR_MAX);
        // bar 1: 9 + 0.02*(10-9) = 9.02, capped at low[0] = 9
        // bar 2: 9 + 0.04*(11-9) = 9.08, capped at low[0] = 9
        // bar 3: 9 + 0.06*(12-9) = 9.18
        // bar 4: 9.18 + 0.08*(13-9.18) = 9.4856
        approx_all(&s, &[9.0, 9.0, 9.0, 9.18, 9.4856]);
        approx_all(&t, &[UP; 5]);
    }

    #[test]
    fn sar_reversal_to_short_starts_from_the_higher_of_the_high_and_the_extreme_point() {
        let (h, l) = outside_bar_ends_an_uptrend();
        let (s, t) = sar(&h, &l, SAR_STEP, SAR_MAX);
        assert!(t[4] == UP && t[5] == DOWN);
        approx(s[5], 15.0); // max(high 15, extreme point 14), above high[4] and high[3]
    }

    #[test]
    fn sar_stays_short_while_the_high_is_below_the_projected_stop() {
        let (h, l) = outside_bar_ends_an_uptrend();
        let (s, t) = sar(&h, &l, SAR_STEP, SAR_MAX);
        approx_all(&t[5..], &[DOWN; 4]);
        // bar 6: 15 + 0.02*(9.5-15) = 14.89, raised to high[5] = 15
        // bar 7: 14.89 again, raised to high[5] = 15
        // bar 8: 14.89, above high[7] and high[6], so it stands
        approx_all(&s[6..], &[15.0, 15.0, 14.89]);
    }

    #[test]
    fn sar_downtrend_follows_the_hand_computed_path() {
        let (h, l) = outside_bar_ends_a_downtrend();
        let (s, t) = sar(&h[..5], &l[..5], SAR_STEP, SAR_MAX);
        // bar 1: 19.02 is above low 18: reverse, max(high 19, extreme point 20) = 20
        // bar 2: 20 + 0.02*(18-20) = 19.96, raised to high[0] = 20
        // bar 3: 20 + 0.04*(17-20) = 19.88
        // bar 4: 19.88 + 0.06*(16-19.88) = 19.6472
        approx_all(&s, &[19.0, 20.0, 20.0, 19.88, 19.6472]);
        approx_all(&t, &[UP, DOWN, DOWN, DOWN, DOWN]);
    }

    #[test]
    fn sar_reversal_to_long_starts_from_the_lower_of_the_low_and_the_extreme_point() {
        let (h, l) = outside_bar_ends_a_downtrend();
        let (s, t) = sar(&h, &l, SAR_STEP, SAR_MAX);
        assert!(t[4] == DOWN && t[5] == UP);
        approx(s[5], 14.0); // min(low 14, extreme point 15), below low[4] and low[3]
    }

    #[test]
    fn sar_stays_long_while_the_low_is_above_the_projected_stop() {
        let (h, l) = outside_bar_ends_a_downtrend();
        let (s, t) = sar(&h, &l, SAR_STEP, SAR_MAX);
        assert!(t[6] == UP);
        approx(s[6], 14.0); // 14 + 0.02*(21.5-14) = 14.15, capped at low[5] = 14
    }

    #[test]
    fn sar_reversal_stop_is_clamped_to_the_bar_two_back() {
        // Two reversals in a row: at bar 4 the new short stop is
        // max(high 19.0, extreme point 19.6) = 19.6, then raised to high[2] = 19.7.
        let h = [20.0, 19.5, 19.7, 19.6, 19.0];
        let l = [19.0, 10.0, 9.0, 19.0, 9.0];
        let (s, t) = sar(&h, &l, SAR_STEP, SAR_MAX);
        // bar 1: 19.02 > low 10: short, max(19.5, 20) = 20
        // bar 2: 19.8, not below high 19.7; raised to high[0] = 20
        // bar 3: 20 + 0.04*(9-20) = 19.56 < high 19.6: long, min(low 19, extreme 9) = 9
        // bar 4: 9 + 0.02*(19.6-9) = 9.212 > low 9: short, 19.6 raised to 19.7
        approx_all(&s, &[19.0, 20.0, 20.0, 9.0, 19.7]);
        approx_all(&t, &[UP, DOWN, DOWN, UP, DOWN]);
    }

    #[test]
    fn sar_does_not_reverse_when_the_low_only_touches_the_stop() {
        // Step 0.25 keeps the arithmetic exact: 8 + 0.25*(12-8) = 9 = low[1].
        let (s, t) = sar(&[12.0, 13.0], &[8.0, 9.0], 0.25, 0.5);
        approx_all(&s, &[8.0, 8.0]); // 9 capped at low[0]
        approx_all(&t, &[UP, UP]);
    }

    #[test]
    fn sar_does_not_reverse_when_the_high_only_touches_the_stop() {
        // bar 1: 9 > low 7: short, stop max(high 11, extreme point 12) = 12
        // bar 2: 12 + 0.25*(7-12) = 10.75 = high[2]: no reversal, raised to high[0]
        let (s, t) = sar(&[12.0, 11.0, 10.75], &[8.0, 7.0, 9.0], 0.25, 0.5);
        approx_all(&s, &[8.0, 12.0, 12.0]);
        approx_all(&t, &[UP, DOWN, DOWN]);
    }

    #[test]
    fn sar_of_a_constant_series_never_reverses() {
        let (s, t) = sar(&[5.0; 6], &[5.0; 6], SAR_STEP, SAR_MAX);
        approx_all(&s, &[5.0; 6]);
        approx_all(&t, &[UP; 6]);
    }

    #[test]
    fn sar_starts_at_the_first_bar_that_has_a_high_and_a_low() {
        let nan = f64::NAN;
        let h = [nan, 11.0, 12.0, 13.0, 9.0, 8.0];
        let l = [9.0, 10.0, 11.0, 12.0, 7.0, 6.0];
        let (s, t) = sar(&h, &l, SAR_STEP, SAR_MAX);
        // Seeded at bar 1 (stop 10, extreme point 11).
        // bar 2: 10.02 capped at low[1] = 10;  bar 3: 10.08 capped at 10
        // bar 4: 10.18 > low 7: short, max(high 9, extreme 13) = 13
        // bar 5: 13 + 0.02*(7-13) = 12.88, raised to high[3] = 13
        approx_all(&s, &[nan, 10.0, 10.0, 10.0, 13.0, 13.0]);
        approx_all(&t, &[nan, UP, UP, UP, DOWN, DOWN]);
    }

    #[test]
    fn sar_skips_a_nan_bar_as_if_it_were_not_there() {
        let (h, l) = outside_bar_ends_an_uptrend();
        let (want_s, want_t) = sar(&h, &l, SAR_STEP, SAR_MAX);
        // The same bars with an unknown bar inserted after bar 2.
        let at = 3;
        let mut gap_h = h.to_vec();
        let mut gap_l = l.to_vec();
        gap_h.insert(at, f64::NAN);
        gap_l.insert(at, 11.5);
        let (s, t) = sar(&gap_h, &gap_l, SAR_STEP, SAR_MAX);
        assert!(s[at].is_nan() && t[at].is_nan());
        approx_all(&[&s[..at], &s[at + 1..]].concat(), &want_s);
        approx_all(&[&t[..at], &t[at + 1..]].concat(), &want_t);
    }

    #[test]
    fn sar_reads_no_further_than_the_shorter_of_its_inputs() {
        let (h, l) = outside_bar_ends_an_uptrend();
        let (s, t) = sar(&h, &l[..7], SAR_STEP, SAR_MAX);
        let (want_s, _) = sar(&h[..7], &l[..7], SAR_STEP, SAR_MAX);
        assert_eq!(s.len(), h.len());
        approx_all(&s[..7], &want_s);
        assert!(s[7].is_nan() && s[8].is_nan() && t[7].is_nan() && t[8].is_nan());
    }

    #[test]
    fn sar_keeps_clear_of_the_two_previous_bars_on_a_choppy_series() {
        let n = 400usize;
        let mid: Vec<f64> = (0..n)
            .map(|i| 100.0 + 8.0 * ((i as f64) / 5.0).sin() + 3.0 * ((i as f64) / 1.7).cos())
            .collect();
        let h: Vec<f64> = (0..n).map(|i| mid[i] + 1.0 + ((i * 7) % 5) as f64 * 0.6).collect();
        let l: Vec<f64> = (0..n).map(|i| mid[i] - 1.0 - ((i * 3) % 4) as f64 * 0.7).collect();
        let (s, t) = sar(&h, &l, SAR_STEP, SAR_MAX);
        let mut reversals = 0usize;
        for i in 2..n {
            reversals += (t[i] != t[i - 1]) as usize;
            if t[i] == DOWN {
                assert!(s[i] >= h[i - 1] && s[i] >= h[i - 2], "bar {i}: short stop {}", s[i]);
            } else {
                assert!(s[i] <= l[i - 1] && s[i] <= l[i - 2], "bar {i}: long stop {}", s[i]);
            }
        }
        assert!(reversals > 10, "the series should reverse often, got {reversals}");
    }

    // ------------------------------------------------------------------
    // Relative Vigor Index
    // ------------------------------------------------------------------

    /// Bars with close - open = 1 and high - low = 2: the RVI is exactly 0.5.
    fn flat_bars(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        (vec![10.0; n], vec![12.0; n], vec![10.0; n], vec![11.0; n])
    }

    #[test]
    fn rvi_vigor_is_one_half_on_bars_whose_body_is_half_their_range() {
        let period = 10;
        let (o, h, l, c) = flat_bars(30);
        let (r, s) = rvi_vigor(&o, &h, &l, &c, period);
        // The 4-bar swma starts at bar 3, the sum of `period` of them at bar
        // period + 2, and the signal (a swma of the RVI) three bars later.
        for i in 0..30 {
            if i < period + 2 {
                assert!(r[i].is_nan(), "rvi[{i}] = {}", r[i]);
            } else {
                approx(r[i], 0.5);
            }
            if i < period + 5 {
                assert!(s[i].is_nan(), "signal[{i}] = {}", s[i]);
            } else {
                approx(s[i], 0.5);
            }
        }
    }

    #[test]
    fn rvi_vigor_equals_the_formula_evaluated_term_by_term() {
        let o = [10.0, 10.5, 11.0, 10.8, 11.4, 11.9, 12.1, 11.7];
        let h = [10.9, 11.4, 11.6, 11.7, 12.2, 12.6, 12.5, 12.4];
        let l = [9.8, 10.2, 10.6, 10.5, 11.1, 11.6, 11.5, 11.3];
        let c = [10.6, 11.1, 10.9, 11.5, 12.0, 12.2, 11.8, 12.3];
        let (r, _) = rvi_vigor(&o, &h, &l, &c, 3);
        let co: Vec<f64> = (0..8).map(|i| c[i] - o[i]).collect();
        let hl: Vec<f64> = (0..8).map(|i| h[i] - l[i]).collect();
        let swma = |x: &[f64], i: usize| (x[i - 3] + 2.0 * x[i - 2] + 2.0 * x[i - 1] + x[i]) / 6.0;
        for v in r.iter().take(5) {
            assert!(v.is_nan());
        }
        for (i, got) in r.iter().enumerate().skip(5) {
            let num = swma(&co, i - 2) + swma(&co, i - 1) + swma(&co, i);
            let den = swma(&hl, i - 2) + swma(&hl, i - 1) + swma(&hl, i);
            approx(*got, num / den);
        }
    }

    #[test]
    fn rvi_vigor_is_nan_while_a_nan_bar_is_in_its_window() {
        let period = 10;
        let (mut o, h, l, c) = flat_bars(45);
        o[20] = f64::NAN;
        let (r, _) = rvi_vigor(&o, &h, &l, &c, period);
        // The NaN is inside the swma of bars 20..=23 and so inside every sum
        // window that touches one of them: bars 20..=23+period-1.
        for (i, v) in r.iter().enumerate().skip(period + 2) {
            if (20..=32).contains(&i) {
                assert!(v.is_nan(), "rvi[{i}] = {v}, expected NaN");
            } else {
                approx(*v, 0.5);
            }
        }
    }

    #[test]
    fn rvi_signal_recovers_three_bars_after_the_rvi() {
        let (mut o, h, l, c) = flat_bars(45);
        o[20] = f64::NAN;
        let (_, s) = rvi_vigor(&o, &h, &l, &c, 10);
        for (i, v) in s.iter().enumerate().take(36).skip(20) {
            assert!(v.is_nan(), "signal[{i}] = {v}, expected NaN");
        }
        approx(s[19], 0.5);
        approx(s[36], 0.5);
    }

    #[test]
    fn rvi_vigor_waits_for_a_nan_warmup_to_leave_its_window() {
        let (mut o, h, l, c) = flat_bars(30);
        for v in o.iter_mut().take(4) {
            *v = f64::NAN;
        }
        let (r, _) = rvi_vigor(&o, &h, &l, &c, 10);
        // swma is NaN up to bar 6, so the first clean 10-bar window ends at 16.
        for (i, v) in r.iter().enumerate().take(16) {
            assert!(v.is_nan(), "rvi[{i}] = {v}, expected NaN");
        }
        for v in r.iter().skip(16) {
            approx(*v, 0.5);
        }
    }

    // ------------------------------------------------------------------
    // A period no window can satisfy: all NaN, and no allocation sized by it
    // ------------------------------------------------------------------

    #[test]
    fn rolling_extremes_are_all_nan_when_the_period_exceeds_the_series() {
        let d = [0.0, 1.0, 2.0, 3.0, 4.0];
        for period in [6, 1_000_000_000_000, usize::MAX / 2 + 1, usize::MAX] {
            assert!(highest(&d, period).iter().all(|v| v.is_nan()), "highest, period {period}");
            assert!(lowest(&d, period).iter().all(|v| v.is_nan()), "lowest, period {period}");
        }
    }

    #[test]
    fn window_kernels_are_all_nan_when_the_period_exceeds_the_series() {
        let (o, h, l, c) = flat_bars(6);
        let v = vec![100.0; 6];
        for period in [7, usize::MAX - 1, usize::MAX] {
            assert!(cmo(&c, period).iter().all(|x| x.is_nan()), "cmo, period {period}");
            assert!(cmf(&h, &l, &c, &v, period).iter().all(|x| x.is_nan()), "cmf, period {period}");
            assert!(fisher(&c, period).0.iter().all(|x| x.is_nan()), "fisher, period {period}");
            let (r, s) = rvi_vigor(&o, &h, &l, &c, period);
            assert!(r.iter().chain(&s).all(|x| x.is_nan()), "rvi_vigor, period {period}");
        }
    }

    #[test]
    fn frama_is_the_midpoint_when_the_period_exceeds_the_series() {
        let (_, h, l, _) = flat_bars(6);
        for period in [8, usize::MAX - 1, usize::MAX] {
            approx_all(&frama(&h, &l, period), &[11.0; 6]);
        }
    }

    // ------------------------------------------------------------------
    // NaN contract of the window and recursive kernels
    // ------------------------------------------------------------------

    /// Eight rising bars: high 10..17, low two below, close half a point below
    /// the high, so the money-flow multiplier is 0.5 on every bar.
    fn rising_bars() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let h: Vec<f64> = (10..18).map(|x| x as f64).collect();
        let l: Vec<f64> = h.iter().map(|x| x - 2.0).collect();
        let c: Vec<f64> = h.iter().map(|x| x - 0.5).collect();
        (h, l, c, vec![100.0; 8])
    }

    #[test]
    fn cmf_is_nan_while_a_nan_volume_is_in_its_window() {
        let nan = f64::NAN;
        let (h, l, c, mut v) = rising_bars();
        v[3] = nan;
        approx_all(&cmf(&h, &l, &c, &v, 3), &[nan, nan, 0.5, nan, nan, nan, 0.5, 0.5]);
    }

    #[test]
    fn cmf_is_nan_for_a_nan_close_on_a_bar_without_range() {
        let nan = f64::NAN;
        let (mut h, l, mut c, v) = rising_bars();
        h[3] = l[3]; // no range: the multiplier would be 0 whatever the close
        c[3] = nan;
        let r = cmf(&h, &l, &c, &v, 3);
        assert!(r[3].is_nan() && r[4].is_nan() && r[5].is_nan());
        approx(r[6], 0.5);
    }

    #[test]
    fn mfi_is_nan_while_a_nan_price_is_in_its_window() {
        let nan = f64::NAN;
        let p = [10.0, 11.0, 10.0, nan, 12.0, 11.0, 13.0]; // high = low = close
        let v = [1.0; 7];
        // Flow per bar: +11, -10, unknown, none (no earlier price), -11, +13.
        // Two-bar windows:
        //   bar 1: +11                 -> 100
        //   bar 2: +11, -10            -> 100 - 100/(1 + 11/10)
        //   bars 3 and 4 hold the NaN  -> NaN
        //   bar 5: none, -11           -> 0
        //   bar 6: -11, +13            -> 100 - 100/(1 + 13/11)
        let want =
            [nan, 100.0, 100.0 - 100.0 / 2.1, nan, nan, 0.0, 100.0 - 100.0 / (1.0 + 13.0 / 11.0)];
        approx_all(&mfi(&p, &p, &p, &v, 2), &want);
    }

    #[test]
    fn mfi_recovers_after_a_nan_volume() {
        let nan = f64::NAN;
        let p = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let v = [1.0, 1.0, nan, 1.0, 1.0, 1.0, 1.0];
        // Prices only rise, so the index is 100 wherever its 2-bar window is known.
        approx_all(&mfi(&p, &p, &p, &v, 2), &[nan, 100.0, nan, nan, 100.0, 100.0, 100.0]);
    }

    #[test]
    fn cmo_is_nan_while_a_nan_is_among_the_bars_of_its_window() {
        let nan = f64::NAN;
        let d = [10.0, 11.0, 12.0, nan, 14.0, 15.0, 16.0, 17.0, 18.0];
        // Three changes span four bars; the first window clear of bar 3 ends at 7.
        approx_all(&cmo(&d, 3), &[nan, nan, nan, nan, nan, nan, nan, 100.0, 100.0]);
    }

    #[test]
    fn fisher_carries_its_state_across_a_nan_bar() {
        let nan = f64::NAN;
        let d = [1.0, 2.0, nan, 2.0, 1.0, 2.0];
        let (f, t) = fisher(&d, 2);
        // bar 1: window (1, 2), position 1 -> value 0.33
        let v1: f64 = 0.66 * 0.5;
        let f1 = 0.5 * ((1.0 + v1) / (1.0 - v1)).ln();
        // bars 2 and 3 hold the NaN. bar 4: window (2, 1), position 0
        let v4 = 0.66 * -0.5 + 0.67 * v1;
        let f4 = 0.5 * ((1.0 + v4) / (1.0 - v4)).ln() + 0.5 * f1;
        // bar 5: window (1, 2), position 1
        let v5 = 0.66 * 0.5 + 0.67 * v4;
        let f5 = 0.5 * ((1.0 + v5) / (1.0 - v5)).ln() + 0.5 * f4;
        approx_all(&f, &[nan, f1, nan, nan, f4, f5]);
        // The trigger is the Fisher value before the bar: 0 at the start.
        approx_all(&t, &[nan, 0.0, nan, nan, f1, f4]);
    }

    #[test]
    fn fisher_after_a_nan_warmup_equals_the_series_without_it() {
        let d: Vec<f64> = (0..30).map(|x| ((x as f64) * 0.7).sin() + 5.0).collect();
        let (want_f, want_t) = fisher(&d, 9);
        let mut padded = vec![f64::NAN; 3];
        padded.extend_from_slice(&d);
        let (f, t) = fisher(&padded, 9);
        assert!(f[..3].iter().all(|x| x.is_nan()));
        approx_all(&f[3..], &want_f);
        approx_all(&t[3..], &want_t);
    }

    /// A straight line: low = bar number, high one above. Its fractal dimension
    /// is 1, the filter weight 1, so FRAMA is the 5-bar mean of the midpoint:
    /// the midpoint two bars back.
    fn straight_line(n: usize) -> (Vec<f64>, Vec<f64>) {
        ((0..n).map(|i| i as f64 + 1.0).collect(), (0..n).map(|i| i as f64).collect())
    }

    #[test]
    fn frama_of_a_straight_line_is_the_five_bar_mean_of_the_midpoint() {
        let period = 4;
        let (h, l) = straight_line(12);
        let want: Vec<f64> =
            (0..12).map(|i| if i <= period { i as f64 + 0.5 } else { i as f64 - 1.5 }).collect();
        approx_all(&frama(&h, &l, period), &want);
    }

    #[test]
    fn frama_is_nan_only_while_a_nan_bar_is_in_its_windows() {
        let period = 4;
        let (mut h, l) = straight_line(20);
        h[10] = f64::NAN;
        let r = frama(&h, &l, period);
        // The filter is unknown for the 4 bars whose range window holds bar 10,
        // and its 5-bar mean for 4 bars more: bars 10..=17.
        for (i, v) in r.iter().enumerate() {
            if (10..=17).contains(&i) {
                assert!(v.is_nan(), "frama[{i}] = {v}, expected NaN");
            } else if i <= period {
                approx(*v, i as f64 + 0.5);
            } else {
                approx(*v, i as f64 - 1.5);
            }
        }
    }

    #[test]
    fn frama_after_a_nan_warmup_equals_the_series_without_it() {
        let h: Vec<f64> = (0..40).map(|x| ((x as f64) * 0.9).sin() + 10.0).collect();
        let l: Vec<f64> = h.iter().map(|x| x - 1.0).collect();
        let want = frama(&h, &l, 8);
        let mut pad_h = vec![f64::NAN; 3];
        let mut pad_l = vec![f64::NAN; 3];
        pad_h.extend_from_slice(&h);
        pad_l.extend_from_slice(&l);
        let r = frama(&pad_h, &pad_l, 8);
        assert!(r[..3].iter().all(|x| x.is_nan()));
        approx_all(&r[3..], &want);
    }

    #[test]
    fn session_vwap_is_nan_until_the_period_has_volume() {
        let nan = f64::NAN;
        let (vwap, sd) =
            session_vwap(&[9.0, 19.0, 29.0, 39.0], &[0.0, 0.0, 5.0, 5.0], &[1.0, 0.0, 0.0, 0.0]);
        approx_all(&vwap, &[nan, nan, 29.0, 34.0]);
        approx_all(&sd, &[nan, nan, 0.0, 5.0]); // sqrt((29^2 + 39^2)/2 - 34^2) = 5
    }

    #[test]
    fn session_vwap_skips_a_nan_bar_and_keeps_its_sums() {
        let nan = f64::NAN;
        let src = [9.0, 19.0, 29.0, 39.0];
        let (vwap, _) = session_vwap(&src, &[1.0, nan, 1.0, 1.0], &[1.0, 0.0, 0.0, 0.0]);
        approx_all(&vwap, &[9.0, nan, 19.0, 77.0 / 3.0]); // (9+29)/2, (9+29+39)/3
    }

    #[test]
    fn session_vwap_restarts_on_a_start_bar_that_is_itself_nan() {
        let nan = f64::NAN;
        let (vwap, _) =
            session_vwap(&[9.0, nan, 29.0, 39.0], &[1.0, 1.0, 1.0, 1.0], &[1.0, 1.0, 0.0, 0.0]);
        approx_all(&vwap, &[9.0, nan, 29.0, 34.0]);
    }

    #[test]
    fn session_vwap_reads_no_further_than_its_shortest_input() {
        let (vwap, sd) = session_vwap(&[9.0, 19.0, 29.0], &[1.0, 1.0], &[1.0]);
        approx(vwap[0], 9.0);
        approx(vwap[1], 14.0);
        assert!(vwap[2].is_nan() && sd[2].is_nan());
    }

    // ------------------------------------------------------------------
    // Signal helpers that the public `ta` utilities call
    // ------------------------------------------------------------------

    #[test]
    fn crossover_and_crossunder_need_a_strict_cross_from_the_other_side() {
        let a = [1.0, 3.0, 2.0, 2.5, 1.0, 1.0, 5.0];
        let b = [2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0];
        // 1: up through 2. 2: a touch from above is not a cross under.
        // 3: up from equal. 4: down through 2. 5: equal. 6: up from equal.
        assert_eq!(crossover(&a, &b), vec![false, true, false, true, false, false, true]);
        assert_eq!(crossunder(&a, &b), vec![false, false, false, false, true, false, false]);
    }

    #[test]
    fn cross_is_crossover_or_crossunder() {
        let a = [1.0, 3.0, 2.0, 2.5, 1.0, 1.0, 5.0];
        let b = [2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0];
        let (over, under) = (crossover(&a, &b), crossunder(&a, &b));
        let either: Vec<bool> = (0..a.len()).map(|i| over[i] || under[i]).collect();
        assert_eq!(cross(&a, &b), either);
    }

    #[test]
    fn a_nan_on_the_bar_or_the_bar_before_blocks_every_cross() {
        let a = [1.0, f64::NAN, 3.0, 1.0];
        let b = [2.0, 2.0, 2.0, 2.0];
        // Bar 2 is above after bar 0 was below, but bar 1 in between is unknown.
        assert_eq!(crossover(&a, &b), vec![false, false, false, false]);
        assert_eq!(crossunder(&a, &b), vec![false, false, false, true]);
        assert_eq!(cross(&a, &b), vec![false, false, false, true]);
    }

    #[test]
    fn rising_and_falling_compare_with_the_bar_length_back() {
        let d = [1.0, 2.0, f64::NAN, 4.0, 3.0, 3.0];
        assert_eq!(rising(&d, 1), vec![false, true, false, false, false, false]);
        assert_eq!(falling(&d, 1), vec![false, false, false, false, true, false]);
        assert_eq!(rising(&d, 2), vec![false, false, false, true, false, false]);
    }

    #[test]
    fn rising_and_falling_are_all_false_when_the_lookback_exceeds_the_series() {
        let d = [1.0, 2.0, 3.0];
        assert_eq!(rising(&d, 10), vec![false; 3]);
        assert_eq!(falling(&d, 10), vec![false; 3]);
    }

    #[test]
    fn valuewhen_returns_the_value_at_the_nth_most_recent_true_bar() {
        let expr = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let arr = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        let latest = valuewhen(&expr, &arr, 1);
        assert!(latest[0].is_nan());
        for (i, want) in [(1, 11.0), (2, 11.0), (3, 13.0), (4, 13.0), (5, 15.0)] {
            approx(latest[i], want);
        }
        let second = valuewhen(&expr, &arr, 2);
        assert!(second[1].is_nan() && second[2].is_nan());
        for (i, want) in [(3, 11.0), (4, 11.0), (5, 13.0)] {
            approx(second[i], want);
        }
    }

    /// Two bars in three are true, so 3000 bars hold 2000 true bars: twice the cap.
    fn dense_condition() -> (Vec<f64>, Vec<f64>) {
        let n = 3000usize;
        (
            (0..n).map(|i| if i % 3 != 0 { 1.0 } else { 0.0 }).collect(),
            (0..n).map(|i| i as f64 * 0.5).collect(),
        )
    }

    #[test]
    fn valuewhen_matches_a_direct_scan_beyond_the_lookback_cap() {
        let (expr, arr) = dense_condition();
        for &k in &[1usize, 2, 7, VALUEWHEN_LOOKBACK_CAP - 1, VALUEWHEN_LOOKBACK_CAP] {
            let got = valuewhen(&expr, &arr, k);
            let mut hits: Vec<usize> = Vec::new();
            for i in 0..expr.len() {
                if expr[i] != 0.0 {
                    hits.push(i);
                }
                if hits.len() >= k {
                    approx(got[i], arr[hits[hits.len() - k]]);
                } else {
                    assert!(got[i].is_nan(), "n={k} bar {i}: expected NaN, got {}", got[i]);
                }
            }
        }
    }

    #[test]
    fn valuewhen_never_resolves_an_n_of_zero_or_above_the_cap() {
        let (expr, arr) = dense_condition();
        let all_nan = |v: Vec<f64>| v.iter().all(|x| x.is_nan());
        assert!(all_nan(valuewhen(&expr, &arr, 0)));
        assert!(all_nan(valuewhen(&expr, &arr, VALUEWHEN_LOOKBACK_CAP + 1)));
        // The cap is also the series length when that is shorter.
        assert!(all_nan(valuewhen(&expr[..5], &arr[..5], 6)));
    }

    #[test]
    fn valuewhen_leaves_nan_where_the_value_series_is_too_short() {
        let expr = [1.0, 0.0, 1.0, 1.0];
        let arr = [10.0, 11.0, 12.0]; // one value short of the condition
        let got = valuewhen(&expr, &arr, 1);
        approx(got[0], 10.0);
        approx(got[1], 10.0);
        approx(got[2], 12.0);
        assert!(got[3].is_nan());
    }
}
