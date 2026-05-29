# Rust Indicator Migration — Ralph Loop Plan

**Goal:** Replace the numba+numpy backend of `openalgo/indicators/` with an in-tree
Rust core (PyO3/maturin), removing the `numba`/`llvmlite` dependency entirely, while
keeping the public API (`from openalgo import ta`, every indicator name + parameter)
**byte-identical** for the 10,000+ existing users. Faster, no LLVM install.

This file is the loop's persistent memory. Each iteration: read it, do the next
unchecked step, update statuses, commit locally. **Never push** until everything is
migrated + tested + benchmarked + parity-verified.

## Hard constraints
- Local commits only on branch `rust-indicators-migration`. **No `git push`.**
- No icons/emoji in code or logger output (user global rule).
- Public API unchanged: signatures, defaults, return shapes, NaN placement identical.
- Parity target: bit-exact for integer/min-max kernels; <=1e-9 rel for recursive
  (EMA-family) kernels; cross-checked vs **TA-Lib** and vs the **Python reference**
  (run with `NUMBA_DISABLE_JIT=1`, which executes the exact same kernel code).
- Remove `pip install openalgo[indicators]` extra (the numba/llvmlite optional dep).

## Environment facts (discovered)
- Python 3.14.4, numpy 2.4.2, pandas 3.0.0, TA-Lib 0.6.8, yfinance 1.1.0.
- Rust 1.92.0, cargo 1.92.0, maturin 1.8.6. Host: x86_64-pc-windows-msvc.
- **Legacy numba stack is BROKEN here**: numba 0.63 references `np.trapz` (removed in
  numpy 2.x) -> `from openalgo import ...` crashes unless `NUMBA_DISABLE_JIT=1`.
  This is the strongest justification for the migration.
- Dhan broker market-data is **unsubscribed / under maintenance**, and Historify/DuckDB
  is empty -> real OpenAlgo data unavailable. **Using yfinance (RELIANCE.NS, SBIN.NS)**
  for now; `benchmark/fetch_data.py` auto-switches to OpenAlgo db/api when they return.

## Layout
```
rust/
  Cargo.toml            # workspace (members: oa_core; oa_py added when wiring PyO3)
  oa_core/              # zero-dep Rust kernels, cargo-testable  [DONE: primitives]
  oa_py/                # PyO3 cdylib -> openalgo._oaindicators  [TODO]
benchmark/
  fetch_data.py         # db->api->yfinance fallback; writes data/<SYM>_<IV>.csv  [DONE]
  data/                 # cached CSVs (gitignored)  [DONE: RELIANCE/SBIN D+1m]
  parity.py             # Rust vs TA-Lib vs Python-ref  [TODO]
  speed.py              # Rust vs numba vs TA-Lib timings  [TODO]
RUST_MIGRATION_TRACKER.csv  # 108-row indicator inventory + per-indicator status
```

## Phases (see RUST_MIGRATION_TRACKER.csv for the full per-indicator list)
- **Phase 0 — Foundation** [DONE]
  - [x] Branch, toolchain check, data harness, real CSVs fetched.
  - [x] `oa_core` primitives ported + unit-tested (11/11 green):
        sma, rolling_sum, rolling_variance, stdev, ema, ema_wilder, true_range,
        atr_wilder, atr_sma, change, roc, highest, lowest, vwma, cmo, kama,
        ulcer_index, crossover, crossunder, cross, rising, falling, exrem, flip.
  - [x] `oa_py` PyO3 crate (cdylib `openalgo._oaindicators`) exposing all 24 kernels.
        Built with pyo3 0.22 + numpy 0.22, abi3-py39. Python 3.14 needs
        `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` at build time.
  - [x] `benchmark/parity.py`: ALL HARD GATES PASS on RELIANCE/SBIN D+1m — Rust ==
        Python reference (max diff 0.0; a few at ~1e-13, well under atol=1e-12/
        rtol=1e-9). TA-Lib cross-check matches for sma/stdev/sum/var/roc/highest/
        lowest/true_range; ema/atr/cmo differ only by documented convention.
  - [x] valuewhen + wma kernels ported (+cargo tests, 13/13 green) and exposed.

- **Phase 1 — Backend seam + first wrappers** [IN PROGRESS]
  - [x] `openalgo/indicators/_backend.py`: Rust-backed kernels with pure-numpy
        fallback (no numba in either path). Holds sma/wma/ema so far.
  - [x] SMA/EMA/WMA classes swapped to `_backend` -> public `ta.sma/ema/wma` now
        numba-free. `benchmark/wrapper_parity.py`: ta.sma==pandas, ta.wma==TA-Lib,
        ta.ema==manual (all 0.00 diff), Series/index/list typing preserved.
  - [x] `_warmup()` made defensive: **`import openalgo` now succeeds WITHOUT
        NUMBA_DISABLE_JIT** (the numpy>=2 numba crash no longer breaks import).
  - [x] Phase-1 batch migrated: RSI, MACD, BBands, ATR, Stochastic, CCI, Williams %R.
        Kernels rsi/cci/williams_r/stochastic added to oa_core (cargo 17/17 green);
        macd/bbands/atr composed in _backend from ema/sma/stdev/atr_wilder. Public
        ta.rsi/cci/williams_r/bbands match TA-Lib (<=1.7e-11); atr/macd match
        OpenAlgo's own prior output bit-for-bit (TA-Lib differs only by seed).
  - [x] FP-association fix: running-sum kernels (sma/rolling_sum/stdev/variance/
        vwma/cci) rewritten to Python's exact left-to-right association -> now
        BIT-EXACT (0.0) vs the reference, not just within tolerance.
  - [x] Trend MAs migrated: DEMA, TEMA, TRIMA, HMA, VWMA, ZLEMA, T3, KAMA.
        New rust kernel kama_tv (TradingView ER variant; cargo 18/18). vwma/zlema/
        t3/dema/tema/hma bit-exact (0.0); trima via numpy per-window np.mean
        (bit-exact, numba-free); kama_tv ~4.5e-13 abs (recursive, within 1e-12 rel).
        benchmark/trend_parity.py added (all PASS). 18 indicators migrated total.
  - [x] Migrated: ALMA, McGinley, VIDYA, Alligator, MA Envelopes (rust kernels alma/
        mcginley/vidya; alligator=ema_wilder+shift; ma_env=window-mean/ema; cargo
        21/21). VIDYA/Alligator/MA-Env bit-exact; ALMA/McGinley <=1e-12 (exp/pow).
  - [x] Complex trend migrated: Supertrend, Ichimoku, FRAMA, ChandeKrollStop
        (rust kernels supertrend/chande_kroll_stop/frama multi-output; ichimoku
        composed from highest/lowest; cargo 24/24). ALL bit-exact (0.0) incl FRAMA
        log/exp. **TREND MODULE COMPLETE (20/20).**
  - [x] Momentum complete (9/9): + BOP, ElderRay, Fisher, CRSI. New kernels bop/
        fisher/updown_streak/percent_rank (cargo 28/28); elderray/crsi composed.
        All bit-exact (benchmark/momentum_parity.py). **MOMENTUM MODULE COMPLETE.**
  - [x] Volatility batch 1: Keltner, Donchian, Chaikin, NATR, ULTOSC, RVI(vol).
        New kernel ema_sma (SMA-seeded 2/(p+1) EMA, used by Keltner+Chaikin); rest
        composed (donchian=hi/lo, natr=atr/close, ultosc=rolling_sum). All bit-exact
        except ULTOSC 3.2e-13 (rolling_sum vs np.sum). benchmark/volatility_parity.py.
  - [x] Volatility batch 2: TRANGE, MASS, BBPercent, BBWidth, ChandelierExit,
        HistoricalVolatility, UlcerIndex, STARC. Per-window mean/std helpers
        (_win_mean/_win_std) in _backend for bit-exactness; mass/ulcer compose rust.
        ALL bit-exact (0.0) incl HV log. **VOLATILITY MODULE COMPLETE.**
  - [x] Volume batch 1: OBV, ADL, CMF, MFI, EMV, FI. New kernels obv/adl/cmf/mfi/
        emv_raw/ema_first_valid (cargo 32/32). All bit-exact (volume_parity.py).
  - [x] Volume batch 2: NVI, PVI, VOLOSC, VROC, KVO, PVT, RVOL. Composed in _backend
        (cumsum/cumprod + ema/ema_first_valid). PVI fixed via threaded cumprod
        (prepend initial) for bit-exactness. All bit-exact (volume_parity.py).
  - [x] Volume remaining: OBVSmoothed (composes migrated OBV/SMA/EMA/WMA/BB +
        rma_smma/vwma_strict helpers), VWAP (rust session_vwap kernel; was already
        numba-free). All bit-exact. **VOLUME MODULE COMPLETE (15/15).**
  - [x] Oscillators batch 1: ROC, CMO, TRIX, AO, AC, PPO, PO, DPO, AROONOSC.
        Composed in _backend (roc_osc/trix/ao/ac/ppo/price_oscillator/dpo/aroon_osc;
        cmo via rust). All bit-exact except CMO 2.7e-13 (rolling vs per-window, within
        1e-12 rel). benchmark/oscillator_parity.py.
  - [x] Oscillators batch 2a: UO, StochRSI, CHO, CHOP (composed; all bit-exact incl
        UO per-window np.sum and CHOP log10).
  - [x] Oscillators batch 2b: RVI(vigor, swma), KST, TSI. All bit-exact (TSI fixed
        via 100*(pcs2/apcs2) association).
  - [x] Oscillators batch 2c: VI, GatorOscillator, STC, Coppock. All bit-exact.
        **OSCILLATORS MODULE COMPLETE (20/20).**
  - [x] Statistics module: LINREG, LRSLOPE, CORREL, BETA, VAR, TSF, MEDIAN,
        MedianBands, MODE. Per-window numpy in _backend (bit-exact, numba-free).
        All bit-exact (statistics_parity.py). **STATISTICS MODULE COMPLETE (9/9).**
  - [x] Hybrid module: ADX, Aroon, PivotPoints, SAR (rust kernel), DMI, ZigZag,
        WilliamsFractals, RWI. All bit-exact (hybrid_parity.py). **HYBRID COMPLETE.**

  *** ALL INDICATOR MODULES MIGRATED. Every public ta.* routes through the Rust
  backend (numpy fallback for per-window kernels). 7 parity suites + wrapper gate
  all green; `import openalgo` works without numba. ***

- **Phase 3 — Cleanup / dependency removal** [DONE]
  - [x] numba_shim.py rewritten to a pure no-op (NO `from numba import`); jit/njit/
        prange are passthroughs. `@jit` decorators now run functions interpreted.
  - [x] openalgo/__init__.py: removed the `import numba` monkey-patch block.
  - [x] _warmup() made a no-op (early return).
  - [x] setup.py: removed `extras_require={"indicators": [numba]}`. install_requires
        has no numba. README + docs: dropped `pip install openalgo[indicators]` and
        "JIT/Numba" wording -> "Rust core".
  - [x] `grep import numba|from numba|llvmlite openalgo/` -> NONE. `import openalgo`
        works WITHOUT numba (numba never loaded). All 9 parity gates green.
  NOTE: the dead legacy `_calculate_*` numba staticmethods + utils.py kernels are
  intentionally KEPT (run interpreted) because the parity suites use them as the
  reference. They are never called by the public API. A later optional pass can delete
  them and freeze references to saved golden arrays.
- **Phase 4 — Packaging + CI/CD** [DONE]
  - [x] pyproject.toml: build-backend maturin, python-source=".",
        module-name="openalgo._oaindicators", manifest-path="rust/oa_py/Cargo.toml",
        features=["pyo3/extension-module"]; metadata migrated from setup.py.
  - [x] Local `maturin build --release` -> abi3 wheel
        openalgo-1.0.51-cp39-abi3-win_amd64.whl that bundles the full openalgo python
        package + openalgo/_oaindicators.pyd (28 files, verified).
  - [x] .github/workflows/CI.yml mirroring opengreeks: cargo test (oa_core) -> Linux
        wheel smoke test (benchmark/ci_smoke.py) -> abi3 wheels matrix (linux
        x86_64+aarch64 manylinux 2_28, macos-14 x86_64+arm64, windows msvc) -> sdist ->
        PyPI publish on v* via OIDC. PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 env set.
  - [x] benchmark/ci_smoke.py: synthetic-data end-to-end check (no committed data),
        10/10 PASS; asserts numba never imported.
  NOTE: setup.py/setup.cfg kept but superseded by pyproject (pip uses the maturin
  backend). Can be deleted in a final tidy. Wheels build under rust/target (gitignored).
- **Phase 5 — Benchmark report + final verification** [DONE]
  - [x] benchmark/speed_bench.py -> benchmark/SPEED.md: common indicators ~50x-240x
        vs interpreted-python; on par with TA-Lib for pure kernels.
  - [x] Final verification: all 9 parity suites + ci_smoke + cargo (33) green.
  - [x] MIGRATION_SUMMARY.md written; RUST_MIGRATION_TRACKER.csv all migrated.

*** Phases 0-5 COMPLETE. Migration + tests + parity + benchmark + CI/CD done. ***

- **Phase 6 — Rust-port the remaining numpy per-window kernels** [IN PROGRESS]
  Goal: the ~20 indicators still computed in numpy (per-window loops) show ~1x speedup
  in the NIFTY 924k benchmark. Port each to oa_core Rust (naive per-window sums; these
  differ from numpy pairwise summation by ~1e-14, well within 1e-12 rel - relax those
  parity gates from bit-exact to tol=1e-9 and document).
  - [x] Batch 1 (statistics regressions): linreg, tsf, lrslope, correl, beta -> Rust.
        ~360x-810x vs old interpreted; on par with TA-Lib. statistics_parity green (tol).
  - [x] Batch 2: Rust win_mean/win_std (per-window) kernels; _backend._win_mean/
        _win_std route through them (cargo 37). Massive wins: bbpercent 42.9->0.13ms,
        bbwidth 43.4->0.12, hv 31.7->0.08, starc 23.3->0.04, awesome 23.3->0.07,
        accel 34.4->0.08, kst 57.6->0.15, rvol ~200x, chandelier 0.11. Affected parity
        gates relaxed to tol=1e-9 (~1e-14 rust-vs-numpy-pairwise); all green.
  - [x] Batch 3 (bespoke loops): aroon, aroon_osc, williams_fractals, mode, median,
        stoch_single, wma_nan, rvi_vigor, adx, rwi, variance -> Rust; dpo vectorized.
        cargo 40; all 9 parity gates green. adx 188x, rwi 356x, median 121x, etc.
  - [x] NIFTY 924k full benchmark generated -> benchmark/FULL_BENCHMARK.md. Most
        indicators 50x-725x vs interpreted, accuracy 0.0.
  - [x] Batch 4 (stragglers): trima 5775->33.6ms (171x), roc 269->1.66ms (163x,
        ta.roc now routes to rust), stochrsi 8044->20.3ms. cargo 42; all 9 gates green
        (trima relaxed to 1e-9). 924k benchmark refreshed.

*** PHASE 6 COMPLETE. Every indicator is now Rust-fast - no ~1x numpy stragglers
remain (bbands shows 0.9x only because its benchmark "old" ref is itself the rust
backend). Slowest New on 924k: lrslope 136ms / mode 99ms / ichimoku 70ms (all
O(n*period)/multi-pass, comparable to TA-Lib). All 9 parity suites green; numba-free;
abi3 wheel + CI/CD in place. Benchmarks: benchmark/FULL_BENCHMARK.md (924k) + SPEED.md.
NOTHING PUSHED - awaiting user decision (push / PR / tag). Real-data parity/benchmark
still pending Dhan/Historify (currently yfinance + NIFTY CSV). ***

NOTHING PUSHED. Awaiting user decision on push / PR / tag. Real-data parity/benchmark
still pending Dhan/Historify (currently yfinance + NIFTY CSV).
        NOTE: PVI._with_signal secondary method still references numba EMA - swap in
        Phase 3 cleanup along with all remaining _calculate_* numba staticmethods.
        NOTE: _backend.frama and _backend.fisher numpy fallbacks raise (rust-only);
        add numpy fallbacks in a later polish pass.

  LOCAL BUILD METHOD (until maturin/CI phase): from rust/ run
  `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build --release -p oa_py`
  then copy `rust/target/release/_oaindicators.dll` -> `openalgo/_oaindicators.pyd`.
  (`*.pyd` is gitignored; the built artifact is never committed.)
  maturin/pyproject packaging is intentionally deferred to Phase 4 to avoid
  disrupting the live setuptools build mid-migration.

  NOTE: kernels are parity-verified but the Python indicator CLASSES still call the
  numba kernels. Swapping each wrapper to `_oaindicators` is Phase 3 (that is when a
  tracker row becomes truly "done" = public `ta.*` output unchanged, numba gone).
- **Phase 1 — Simple indicators** (RSI, MACD, ATR, BBands, ADX, Stochastic, CCI, ...)
- **Phase 2 — Complex/stateful** (Supertrend, SAR, Ichimoku, KAMA-MA, VIDYA, FRAMA,
  ZigZag, VWAP, Fisher, STC, Connors RSI, ...)
- **Phase 3 — Backend swap**: rewrite `indicators/*.py` to call `_oaindicators`;
  delete `numba_shim.py`; drop numba/llvmlite from deps + remove `[indicators]` extra.
- **Phase 4 — CI/CD**: `.github/workflows` mirroring opengreeks (cargo test ->
  parity -> abi3 wheels linux x86_64/aarch64 + macos x86_64/arm64 + windows + sdist ->
  PyPI on `v*` via OIDC). Wheels for py3.9..3.13 abi3.
- **Phase 5 — Final gate**: full parity + benchmark report; only then ask user to push.

## Parity/benchmark protocol
- Datasets: `benchmark/data/{RELIANCE,SBIN}_{D,1m}.csv` (real) + a deterministic
  synthetic 100k series for scaling tests.
- For each indicator: assert max abs/rel diff vs reference within tolerance, identical
  NaN mask, identical length. Record pass/fail + timing into the tracker + a report.

## Next action for the loop
Wrapper-swap pattern proven (SMA/EMA/WMA migrated; `import openalgo` fixed). Scale it:
  1. Add Phase-1 kernels to oa_core (+cargo tests) and oa_py: rsi (Wilder), macd
     (ema fast/slow + signal), bbands (sma +/- k*stdev), atr (alias atr_wilder),
     stochastic, cci, williams_r. READ each class in momentum.py / volatility.py
     FIRST to copy exact seeding/NaN semantics before porting.
  2. Add `_backend.py` entries (rust + numpy fallback) for each.
  3. Swap the matching classes to `_backend`; extend parity.py + wrapper_parity.py.
  4. Rebuild ext, re-run BOTH gates; update tracker/plan; commit.
Backlog: trend MAs reusing existing kernels (dema/tema/trima/hma/vwma/zlema/kama/t3),
then statistics/oscillators/volume/hybrid; finally Phase 3 cleanup (delete numba_shim,
drop numba dep + [indicators] extra) and Phase 4 maturin/CI.

## Gap-fill loop progress (post-migration)
- Batch A1 (committed b5f194f): price transforms + ROC/MOM/MIDPOINT/APO variants
  (avgprice/medprice/typprice/wclprice/midpoint/midprice/mom/rocp/rocr/rocr100/apo).
- Batch A2 (this commit): TA-Lib-faithful directional-movement family + linreg
  variants — ta.plus_dm/minus_dm/dx/adxr/stochf/linregangle/linregintercept. New
  Rust kernels (plus_dm_talib/minus_dm_talib/dx_talib/adx_talib/adxr_talib/stochf/
  linreg_angle/linreg_intercept) with exact TA-Lib Wilder seeding; all match TA-Lib
  <=1e-12. Also HARDENED all oa_core window indexing to underflow-safe `i+1-period`
  form so `cargo test` passes in DEBUG (was only passing in release; CI runs debug).
  44 cargo tests + 9 parity gates + talib_extra_parity (19) + ci_smoke all green.
- NEXT (batch B): Hilbert Transform suite (HT_DCPERIOD, HT_DCPHASE, HT_PHASOR,
  HT_SINE, HT_TRENDMODE, HT_TRENDLINE) — port TA-Lib's exact DSP, parity <=1e-6.
- Still NOT pushed. Candlesticks (61 CDL*) intentionally out of scope unless asked.

## Performance pass P1 (O(n*period) -> O(n))
User redirect: skip Hilbert; close the TA-Lib speed gap on large series by turning
O(n*period) per-window kernels into O(n) rolling accumulators.
- wma: rolling weighted-sum (TA-Lib periodSum/periodSub trick). 12.8ms -> 2.95ms on
  924k bars; now matches talib.WMA EXACTLY (0.0 diff).
- trima: was O(n*period) AND allocated a Vec per bar; now two O(n) SMA passes
  (sma-of-sma). 33.6ms -> 4.9ms; flat at period 100 (was O(n*p)). parity 2.3e-12.
- linreg/tsf/lrslope/linregangle/linregintercept: shared O(n) rolling-OLS helper
  (_ols_roll) maintaining Sy + weighted-sum, Sxy = wsum - Sy on the integer x-grid.
  lrslope(100) 136ms -> 6.5ms (21x). linreg/tsf/linregangle now BEAT talib.
  Parity vs reference 0.0; vs talib still <=1e-12.
- highest/lowest already O(n) (monotonic deque); stochastic/rsi already O(n) - their
  benchmark gap to talib is fixed Python FFI/wrapper overhead, not the kernel.
- NEXT (P2): correl + beta are the last O(n*period) stats (one-pass rolling moments
  is cancellation-prone at price^2 scale; needs careful tolerance). Also investigate
  the benchmark's correlation max|d|=1.0 New-vs-Old (degenerate flat-window: 0.0 vs
  perfect-corr). Then refresh benchmark/FULL_BENCHMARK.md with the new numbers.
