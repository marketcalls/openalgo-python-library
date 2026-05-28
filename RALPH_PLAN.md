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
- **Phase 0 — Foundation** [IN PROGRESS]
  - [x] Branch, toolchain check, data harness, real CSVs fetched.
  - [x] `oa_core` primitives ported + unit-tested (11/11 green):
        sma, rolling_sum, rolling_variance, stdev, ema, ema_wilder, true_range,
        atr_wilder, atr_sma, change, roc, highest, lowest, vwma, cmo, kama,
        ulcer_index, crossover, crossunder, cross, rising, falling, exrem, flip.
  - [ ] `oa_py` PyO3 crate (cdylib `openalgo._oaindicators`) exposing the kernels.
  - [ ] maturin wiring in pyproject.toml (mixed Python/Rust, python-source=".",
        module-name="openalgo._oaindicators"); keep setuptools build working until
        the extension is proven, then switch backend.
  - [ ] `benchmark/parity.py`: load CSV -> compare oa_core (via _oaindicators) vs
        TA-Lib vs Python-ref for every ported kernel. Gate: all within tolerance.
  - [ ] valuewhen kernel (port + test) — deferred from primitive batch.
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
Build `rust/oa_py` (PyO3) + maturin config, `maturin develop`, then write and run
`benchmark/parity.py` for the Phase-0 primitives against TA-Lib + Python reference.
