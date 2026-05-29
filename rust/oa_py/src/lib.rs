//! PyO3 bindings: expose the zero-dependency `oa_core` indicator kernels to Python
//! as the `openalgo._oaindicators` extension module. The Python indicator wrappers
//! call these in place of the legacy numba kernels, keeping the public `ta` API
//! byte-identical.
//!
//! All numeric kernels take/return contiguous float64 numpy arrays. The Python side
//! already passes C-contiguous float64 (via `validate_input`), so `as_slice` succeeds.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// f64 slice -> f64 numpy array, for a kernel taking (data, period).
macro_rules! wrap_period {
    ($name:ident, $core:path) => {
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>,
            data: PyReadonlyArray1<'py, f64>,
            period: usize,
        ) -> PyResult<Py<PyArray1<f64>>> {
            let out = $core(data.as_slice()?, period);
            Ok(out.into_pyarray_bound(py).unbind())
        }
    };
}

wrap_period!(sma, oa_core::sma);
wrap_period!(wma, oa_core::wma);
wrap_period!(rolling_sum, oa_core::rolling_sum);
wrap_period!(rolling_variance, oa_core::rolling_variance);
wrap_period!(stdev, oa_core::stdev);
wrap_period!(ema, oa_core::ema);
wrap_period!(ema_wilder, oa_core::ema_wilder);
wrap_period!(ema_sma, oa_core::ema_sma);
wrap_period!(highest, oa_core::highest);
wrap_period!(lowest, oa_core::lowest);
wrap_period!(change, oa_core::change);
wrap_period!(roc, oa_core::roc);
wrap_period!(cmo, oa_core::cmo);
wrap_period!(ulcer_index, oa_core::ulcer_index);
wrap_period!(rsi, oa_core::rsi);
wrap_period!(win_mean, oa_core::win_mean);
wrap_period!(stoch_single, oa_core::stoch_single);
wrap_period!(median, oa_core::median);
wrap_period!(trima, oa_core::trima);

#[pyfunction]
#[pyo3(signature = (data, rsi_period, stoch_period, k_period, d_period))]
fn stochrsi<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    rsi_period: usize,
    stoch_period: usize,
    k_period: usize,
    d_period: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (k, d) = oa_core::stochrsi(data.as_slice()?, rsi_period, stoch_period, k_period, d_period);
    Ok((k.into_pyarray_bound(py).unbind(), d.into_pyarray_bound(py).unbind()))
}
wrap_period!(wma_nan, oa_core::wma_nan);

#[pyfunction]
#[pyo3(signature = (data, lookback, use_log))]
fn variance<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    lookback: usize,
    use_log: bool,
) -> PyResult<Py<PyArray1<f64>>> {
    Ok(oa_core::variance(data.as_slice()?, lookback, use_log).into_pyarray_bound(py).unbind())
}

#[pyfunction]
#[pyo3(signature = (high, low, period))]
fn aroon<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    period: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (u, d) = oa_core::aroon(high.as_slice()?, low.as_slice()?, period);
    Ok((u.into_pyarray_bound(py).unbind(), d.into_pyarray_bound(py).unbind()))
}

#[pyfunction]
#[pyo3(signature = (high, low, n))]
fn williams_fractals<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<(Py<PyArray1<bool>>, Py<PyArray1<bool>>)> {
    let (u, d) = oa_core::williams_fractals(high.as_slice()?, low.as_slice()?, n);
    Ok((u.into_pyarray_bound(py).unbind(), d.into_pyarray_bound(py).unbind()))
}

#[pyfunction]
#[pyo3(signature = (data, period, bins))]
fn mode<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    bins: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    Ok(oa_core::mode(data.as_slice()?, period, bins).into_pyarray_bound(py).unbind())
}

#[pyfunction]
#[pyo3(signature = (open, high, low, close, period))]
fn rvi_vigor<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    period: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (r, s) = oa_core::rvi_vigor(open.as_slice()?, high.as_slice()?, low.as_slice()?, close.as_slice()?, period);
    Ok((r.into_pyarray_bound(py).unbind(), s.into_pyarray_bound(py).unbind()))
}

#[pyfunction]
#[pyo3(signature = (high, low, close, period))]
fn adx<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    period: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (dp, dm, a) = oa_core::adx(high.as_slice()?, low.as_slice()?, close.as_slice()?, period);
    Ok((
        dp.into_pyarray_bound(py).unbind(),
        dm.into_pyarray_bound(py).unbind(),
        a.into_pyarray_bound(py).unbind(),
    ))
}

#[pyfunction]
#[pyo3(signature = (high, low, close, period))]
fn rwi<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    period: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (rh, rl) = oa_core::rwi(high.as_slice()?, low.as_slice()?, close.as_slice()?, period);
    Ok((rh.into_pyarray_bound(py).unbind(), rl.into_pyarray_bound(py).unbind()))
}
wrap_period!(win_std, oa_core::win_std);
wrap_period!(linreg, oa_core::linreg);
wrap_period!(tsf, oa_core::tsf);
wrap_period!(linreg_angle, oa_core::linreg_angle);
wrap_period!(linreg_intercept, oa_core::linreg_intercept);

#[pyfunction]
#[pyo3(signature = (high, low, period))]
fn plus_dm_talib<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    Ok(oa_core::plus_dm_talib(high.as_slice()?, low.as_slice()?, period)
        .into_pyarray_bound(py)
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (high, low, period))]
fn minus_dm_talib<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    Ok(oa_core::minus_dm_talib(high.as_slice()?, low.as_slice()?, period)
        .into_pyarray_bound(py)
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (high, low, close, period))]
fn dx_talib<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    Ok(oa_core::dx_talib(high.as_slice()?, low.as_slice()?, close.as_slice()?, period)
        .into_pyarray_bound(py)
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (high, low, close, period))]
fn adx_talib<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    Ok(oa_core::adx_talib(high.as_slice()?, low.as_slice()?, close.as_slice()?, period)
        .into_pyarray_bound(py)
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (high, low, close, period))]
fn adxr_talib<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    Ok(oa_core::adxr_talib(high.as_slice()?, low.as_slice()?, close.as_slice()?, period)
        .into_pyarray_bound(py)
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (high, low, close, fastk_period, fastd_period))]
fn stochf<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    fastk_period: usize,
    fastd_period: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (k, d) = oa_core::stochf(
        high.as_slice()?,
        low.as_slice()?,
        close.as_slice()?,
        fastk_period,
        fastd_period,
    );
    Ok((k.into_pyarray_bound(py).unbind(), d.into_pyarray_bound(py).unbind()))
}

#[pyfunction]
#[pyo3(signature = (data, period, interval))]
fn lrslope<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    interval: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    Ok(oa_core::lrslope(data.as_slice()?, period, interval).into_pyarray_bound(py).unbind())
}

#[pyfunction]
#[pyo3(signature = (d1, d2, period))]
fn correl<'py>(
    py: Python<'py>,
    d1: PyReadonlyArray1<'py, f64>,
    d2: PyReadonlyArray1<'py, f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    Ok(oa_core::correl(d1.as_slice()?, d2.as_slice()?, period).into_pyarray_bound(py).unbind())
}

#[pyfunction]
#[pyo3(signature = (asset, market, period))]
fn beta<'py>(
    py: Python<'py>,
    asset: PyReadonlyArray1<'py, f64>,
    market: PyReadonlyArray1<'py, f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    Ok(oa_core::beta(asset.as_slice()?, market.as_slice()?, period).into_pyarray_bound(py).unbind())
}

/// (high, low, close, period) -> f64 array, for hlc kernels.
macro_rules! wrap_hlc_period {
    ($name:ident, $core:path) => {
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>,
            high: PyReadonlyArray1<'py, f64>,
            low: PyReadonlyArray1<'py, f64>,
            close: PyReadonlyArray1<'py, f64>,
            period: usize,
        ) -> PyResult<Py<PyArray1<f64>>> {
            let out = $core(high.as_slice()?, low.as_slice()?, close.as_slice()?, period);
            Ok(out.into_pyarray_bound(py).unbind())
        }
    };
}
wrap_hlc_period!(cci, oa_core::cci);
wrap_hlc_period!(williams_r, oa_core::williams_r);

wrap_period!(percent_rank, oa_core::percent_rank);
wrap_period!(ema_first_valid, oa_core::ema_first_valid);

#[pyfunction]
#[pyo3(signature = (high, low, acceleration, maximum))]
fn sar<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    acceleration: f64,
    maximum: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (s, t) = oa_core::sar(high.as_slice()?, low.as_slice()?, acceleration, maximum);
    Ok((s.into_pyarray_bound(py).unbind(), t.into_pyarray_bound(py).unbind()))
}

#[pyfunction]
#[pyo3(signature = (source, volume, starts))]
fn session_vwap<'py>(
    py: Python<'py>,
    source: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    starts: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (v, s) = oa_core::session_vwap(source.as_slice()?, volume.as_slice()?, starts.as_slice()?);
    Ok((v.into_pyarray_bound(py).unbind(), s.into_pyarray_bound(py).unbind()))
}

#[pyfunction]
#[pyo3(signature = (close, volume))]
fn obv<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    Ok(oa_core::obv(close.as_slice()?, volume.as_slice()?).into_pyarray_bound(py).unbind())
}

#[pyfunction]
#[pyo3(signature = (high, low, close, volume))]
fn adl<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let out = oa_core::adl(high.as_slice()?, low.as_slice()?, close.as_slice()?, volume.as_slice()?);
    Ok(out.into_pyarray_bound(py).unbind())
}

#[pyfunction]
#[pyo3(signature = (high, low, close, volume, period))]
fn cmf<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let out = oa_core::cmf(high.as_slice()?, low.as_slice()?, close.as_slice()?, volume.as_slice()?, period);
    Ok(out.into_pyarray_bound(py).unbind())
}

#[pyfunction]
#[pyo3(signature = (high, low, close, volume, period))]
fn mfi<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let out = oa_core::mfi(high.as_slice()?, low.as_slice()?, close.as_slice()?, volume.as_slice()?, period);
    Ok(out.into_pyarray_bound(py).unbind())
}

#[pyfunction]
#[pyo3(signature = (high, low, volume, divisor))]
fn emv_raw<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    divisor: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let out = oa_core::emv_raw(high.as_slice()?, low.as_slice()?, volume.as_slice()?, divisor);
    Ok(out.into_pyarray_bound(py).unbind())
}

#[pyfunction]
fn updown_streak<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    Ok(oa_core::updown_streak(data.as_slice()?).into_pyarray_bound(py).unbind())
}

#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
fn bop<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let out = oa_core::bop(open.as_slice()?, high.as_slice()?, low.as_slice()?, close.as_slice()?);
    Ok(out.into_pyarray_bound(py).unbind())
}

#[pyfunction]
#[pyo3(signature = (data, length))]
fn fisher<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (f1, f2) = oa_core::fisher(data.as_slice()?, length);
    Ok((
        f1.into_pyarray_bound(py).unbind(),
        f2.into_pyarray_bound(py).unbind(),
    ))
}

#[pyfunction]
#[pyo3(signature = (high, low, period))]
fn frama<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let out = oa_core::frama(high.as_slice()?, low.as_slice()?, period);
    Ok(out.into_pyarray_bound(py).unbind())
}

#[pyfunction]
#[pyo3(signature = (high, low, close, period, multiplier))]
fn supertrend<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    period: usize,
    multiplier: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (st, dir) = oa_core::supertrend(
        high.as_slice()?,
        low.as_slice()?,
        close.as_slice()?,
        period,
        multiplier,
    );
    Ok((
        st.into_pyarray_bound(py).unbind(),
        dir.into_pyarray_bound(py).unbind(),
    ))
}

#[pyfunction]
#[pyo3(signature = (high, low, close, p, x, q))]
fn chande_kroll_stop<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    p: usize,
    x: f64,
    q: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (ls, ss) = oa_core::chande_kroll_stop(
        high.as_slice()?,
        low.as_slice()?,
        close.as_slice()?,
        p,
        x,
        q,
    );
    Ok((
        ls.into_pyarray_bound(py).unbind(),
        ss.into_pyarray_bound(py).unbind(),
    ))
}

#[pyfunction]
#[pyo3(signature = (high, low, close, k_period, smooth_k, d_period))]
fn stochastic<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    k_period: usize,
    smooth_k: usize,
    d_period: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (k, d) = oa_core::stochastic(
        high.as_slice()?,
        low.as_slice()?,
        close.as_slice()?,
        k_period,
        smooth_k,
        d_period,
    );
    Ok((
        k.into_pyarray_bound(py).unbind(),
        d.into_pyarray_bound(py).unbind(),
    ))
}

#[pyfunction]
fn true_range<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let out = oa_core::true_range(high.as_slice()?, low.as_slice()?, close.as_slice()?);
    Ok(out.into_pyarray_bound(py).unbind())
}

#[pyfunction]
fn atr_wilder<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let out = oa_core::atr_wilder(high.as_slice()?, low.as_slice()?, close.as_slice()?, period);
    Ok(out.into_pyarray_bound(py).unbind())
}

#[pyfunction]
fn atr_sma<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let out = oa_core::atr_sma(high.as_slice()?, low.as_slice()?, close.as_slice()?, period);
    Ok(out.into_pyarray_bound(py).unbind())
}

#[pyfunction]
fn vwma<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let out = oa_core::vwma(data.as_slice()?, volume.as_slice()?, period);
    Ok(out.into_pyarray_bound(py).unbind())
}

#[pyfunction]
#[pyo3(signature = (data, period, offset, sigma))]
fn alma<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    offset: f64,
    sigma: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let out = oa_core::alma(data.as_slice()?, period, offset, sigma);
    Ok(out.into_pyarray_bound(py).unbind())
}

wrap_period!(mcginley, oa_core::mcginley);

#[pyfunction]
#[pyo3(signature = (data, period, alpha))]
fn vidya<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    alpha: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let out = oa_core::vidya(data.as_slice()?, period, alpha);
    Ok(out.into_pyarray_bound(py).unbind())
}

#[pyfunction]
#[pyo3(signature = (data, length, fast_length, slow_length))]
fn kama_tv<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
    fast_length: usize,
    slow_length: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let out = oa_core::kama_tv(data.as_slice()?, length, fast_length, slow_length);
    Ok(out.into_pyarray_bound(py).unbind())
}

#[pyfunction]
#[pyo3(signature = (data, period, fast_sc, slow_sc))]
fn kama<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    fast_sc: f64,
    slow_sc: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let out = oa_core::kama(data.as_slice()?, period, fast_sc, slow_sc);
    Ok(out.into_pyarray_bound(py).unbind())
}

// --- boolean kernels (inputs treated as truthy; outputs are numpy bool arrays) ---

macro_rules! wrap_cross {
    ($name:ident, $core:path) => {
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>,
            s1: PyReadonlyArray1<'py, f64>,
            s2: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<Py<PyArray1<bool>>> {
            let out = $core(s1.as_slice()?, s2.as_slice()?);
            Ok(out.into_pyarray_bound(py).unbind())
        }
    };
}
wrap_cross!(crossover, oa_core::crossover);
wrap_cross!(crossunder, oa_core::crossunder);
wrap_cross!(cross, oa_core::cross);

macro_rules! wrap_riserun {
    ($name:ident, $core:path) => {
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>,
            data: PyReadonlyArray1<'py, f64>,
            length: usize,
        ) -> PyResult<Py<PyArray1<bool>>> {
            let out = $core(data.as_slice()?, length);
            Ok(out.into_pyarray_bound(py).unbind())
        }
    };
}
wrap_riserun!(rising, oa_core::rising);
wrap_riserun!(falling, oa_core::falling);

/// exrem/flip accept float "boolean-like" arrays (nonzero == true), matching the
/// legacy numba kernels which were called with float64 signal arrays.
macro_rules! wrap_latch {
    ($name:ident, $core:path) => {
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>,
            primary: PyReadonlyArray1<'py, f64>,
            secondary: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<Py<PyArray1<bool>>> {
            let p: Vec<bool> = primary.as_slice()?.iter().map(|&x| x != 0.0).collect();
            let s: Vec<bool> = secondary.as_slice()?.iter().map(|&x| x != 0.0).collect();
            let out = $core(&p, &s);
            Ok(out.into_pyarray_bound(py).unbind())
        }
    };
}
wrap_latch!(exrem, oa_core::exrem);
wrap_latch!(flip, oa_core::flip);

#[pyfunction]
#[pyo3(signature = (expr, array, n))]
fn valuewhen<'py>(
    py: Python<'py>,
    expr: PyReadonlyArray1<'py, f64>,
    array: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let out = oa_core::valuewhen(expr.as_slice()?, array.as_slice()?, n);
    Ok(out.into_pyarray_bound(py).unbind())
}

#[pymodule]
fn _oaindicators(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sma, m)?)?;
    m.add_function(wrap_pyfunction!(wma, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_sum, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_variance, m)?)?;
    m.add_function(wrap_pyfunction!(stdev, m)?)?;
    m.add_function(wrap_pyfunction!(ema, m)?)?;
    m.add_function(wrap_pyfunction!(ema_wilder, m)?)?;
    m.add_function(wrap_pyfunction!(ema_sma, m)?)?;
    m.add_function(wrap_pyfunction!(highest, m)?)?;
    m.add_function(wrap_pyfunction!(lowest, m)?)?;
    m.add_function(wrap_pyfunction!(change, m)?)?;
    m.add_function(wrap_pyfunction!(roc, m)?)?;
    m.add_function(wrap_pyfunction!(cmo, m)?)?;
    m.add_function(wrap_pyfunction!(ulcer_index, m)?)?;
    m.add_function(wrap_pyfunction!(rsi, m)?)?;
    m.add_function(wrap_pyfunction!(win_mean, m)?)?;
    m.add_function(wrap_pyfunction!(stoch_single, m)?)?;
    m.add_function(wrap_pyfunction!(median, m)?)?;
    m.add_function(wrap_pyfunction!(trima, m)?)?;
    m.add_function(wrap_pyfunction!(stochrsi, m)?)?;
    m.add_function(wrap_pyfunction!(wma_nan, m)?)?;
    m.add_function(wrap_pyfunction!(variance, m)?)?;
    m.add_function(wrap_pyfunction!(aroon, m)?)?;
    m.add_function(wrap_pyfunction!(williams_fractals, m)?)?;
    m.add_function(wrap_pyfunction!(mode, m)?)?;
    m.add_function(wrap_pyfunction!(rvi_vigor, m)?)?;
    m.add_function(wrap_pyfunction!(adx, m)?)?;
    m.add_function(wrap_pyfunction!(rwi, m)?)?;
    m.add_function(wrap_pyfunction!(win_std, m)?)?;
    m.add_function(wrap_pyfunction!(linreg, m)?)?;
    m.add_function(wrap_pyfunction!(tsf, m)?)?;
    m.add_function(wrap_pyfunction!(linreg_angle, m)?)?;
    m.add_function(wrap_pyfunction!(linreg_intercept, m)?)?;
    m.add_function(wrap_pyfunction!(plus_dm_talib, m)?)?;
    m.add_function(wrap_pyfunction!(minus_dm_talib, m)?)?;
    m.add_function(wrap_pyfunction!(dx_talib, m)?)?;
    m.add_function(wrap_pyfunction!(adx_talib, m)?)?;
    m.add_function(wrap_pyfunction!(adxr_talib, m)?)?;
    m.add_function(wrap_pyfunction!(stochf, m)?)?;
    m.add_function(wrap_pyfunction!(lrslope, m)?)?;
    m.add_function(wrap_pyfunction!(correl, m)?)?;
    m.add_function(wrap_pyfunction!(beta, m)?)?;
    m.add_function(wrap_pyfunction!(cci, m)?)?;
    m.add_function(wrap_pyfunction!(williams_r, m)?)?;
    m.add_function(wrap_pyfunction!(stochastic, m)?)?;
    m.add_function(wrap_pyfunction!(bop, m)?)?;
    m.add_function(wrap_pyfunction!(ema_first_valid, m)?)?;
    m.add_function(wrap_pyfunction!(sar, m)?)?;
    m.add_function(wrap_pyfunction!(session_vwap, m)?)?;
    m.add_function(wrap_pyfunction!(obv, m)?)?;
    m.add_function(wrap_pyfunction!(adl, m)?)?;
    m.add_function(wrap_pyfunction!(cmf, m)?)?;
    m.add_function(wrap_pyfunction!(mfi, m)?)?;
    m.add_function(wrap_pyfunction!(emv_raw, m)?)?;
    m.add_function(wrap_pyfunction!(fisher, m)?)?;
    m.add_function(wrap_pyfunction!(updown_streak, m)?)?;
    m.add_function(wrap_pyfunction!(percent_rank, m)?)?;
    m.add_function(wrap_pyfunction!(frama, m)?)?;
    m.add_function(wrap_pyfunction!(supertrend, m)?)?;
    m.add_function(wrap_pyfunction!(chande_kroll_stop, m)?)?;
    m.add_function(wrap_pyfunction!(true_range, m)?)?;
    m.add_function(wrap_pyfunction!(atr_wilder, m)?)?;
    m.add_function(wrap_pyfunction!(atr_sma, m)?)?;
    m.add_function(wrap_pyfunction!(vwma, m)?)?;
    m.add_function(wrap_pyfunction!(kama, m)?)?;
    m.add_function(wrap_pyfunction!(kama_tv, m)?)?;
    m.add_function(wrap_pyfunction!(alma, m)?)?;
    m.add_function(wrap_pyfunction!(mcginley, m)?)?;
    m.add_function(wrap_pyfunction!(vidya, m)?)?;
    m.add_function(wrap_pyfunction!(crossover, m)?)?;
    m.add_function(wrap_pyfunction!(crossunder, m)?)?;
    m.add_function(wrap_pyfunction!(cross, m)?)?;
    m.add_function(wrap_pyfunction!(rising, m)?)?;
    m.add_function(wrap_pyfunction!(falling, m)?)?;
    m.add_function(wrap_pyfunction!(exrem, m)?)?;
    m.add_function(wrap_pyfunction!(flip, m)?)?;
    m.add_function(wrap_pyfunction!(valuewhen, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
