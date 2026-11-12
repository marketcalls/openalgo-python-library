# OpenAlgo Python SDK

Python client library (`pip install openalgo`) for OpenAlgo's trading platform.
It wraps the OpenAlgo REST API, the WebSocket market-data feed, and ships a
technical-indicator library powered by a Rust core (`openalgo._oaindicators`).

- Requires Python >= 3.12.
- Runtime deps: `httpx`, `pandas`, `numpy`, `websocket-client`. The indicators are
  built into the package (Rust/PyO3); there is no optional extra and no numba.
- The SDK talks to a running OpenAlgo server (default REST `http://127.0.0.1:5000`,
  WebSocket proxy `ws://127.0.0.1:8765`). It is a client only; it does not run a
  server itself.

## Engineering standards (apply to every change)

Code quality, code hygiene, modularity, readability and maintainability are
requirements for every change in this library, throughout its development. They are a
gate on each unit of work, not a final polish. Before a change is committed:

- **Quality.** It is correct and tested, and it passes the checks for its area. A
  defect is reproduced by a failing test before it is fixed.
- **Hygiene.** No dead code, commented-out code, debug prints, hardcoded secrets or
  unexplained constants. Errors are never swallowed.
- **Modularity.** Each module, class and function has one job. A new concern goes in
  a new, focused module, not in more branches of a long function. Every shared
  definition has one source of truth.
- **Readability.** Code is written for the next maintainer: plain names in the
  trader's vocabulary, units stated, a docstring on every public item, and comments
  that explain why.
- **Maintainability.** A small public surface, backward compatible changes to
  published APIs, every behaviour change recorded in `CHANGELOG.md`, and no
  abstraction for a feature that is not being built.

In the existing SDK modules keep the diff minimal: change only what the fix needs and
do not reformat or restyle the surrounding code. New code under the backtest paths
follows the detailed rules in "Code standards for the backtest package" below. Work
produced by agents is reviewed independently before it is committed.

## Architecture

The public entry point is `openalgo.api`, which composes every REST mixin via
multiple inheritance and calls `BaseAPI.__init__` once:

    api(OrderAPI, GTTAPI, DataAPI, AccountAPI, FeedAPI, OptionsAPI,
        StrategyAPI, TelegramAPI, WhatsAppAPI, UtilitiesAPI)

`Strategy` (the strategy module's public webhook sender) and `ta` (indicators) are
separate top-level exports.

| File | Responsibility |
|------|----------------|
| `openalgo/base.py` | `BaseAPI`: holds config and the shared pooled `httpx.Client` (`self.client`), the shared `_post()` request helper, plus `close()` / context-manager support |
| `openalgo/orders.py` | place / modify / cancel / smart / basket / split orders, order status, positions |
| `openalgo/gtt.py` | `GTTAPI`: `placegttorder` / `modifygttorder` / `cancelgttorder` / `gttorderbook` (SINGLE and OCO triggers) |
| `openalgo/data.py` | quotes, depth, history, intervals, symbol/instrument lookups (returns pandas where relevant) |
| `openalgo/account.py` | funds, holdings, orderbook, tradebook, positionbook, margin, `analyzerstatus` / `analyzertoggle` |
| `openalgo/options.py` | option-chain and options order helpers |
| `openalgo/telegram.py`, `openalgo/whatsapp.py` | notification endpoints |
| `openalgo/utilities.py` | market holidays / timings |
| `openalgo/strategy.py` | `Strategy`: standalone webhook poster for `/strategy/webhook/<token>` (own pooled client) |
| `openalgo/strategy_api.py` | `StrategyAPI`: the API-key surface under `/api/v1/strategy/` (list, status, start, stop, close_all, close_leg, runs, orders, events) |
| `openalgo/feed.py` | `FeedAPI`: WebSocket market-data feed (uses `websocket-client`, not httpx) |
| `openalgo/indicators/` | `ta` technical indicators (Rust core via `_oaindicators`) |

## HTTP layer (connection pooling)

All REST calls go through the single `httpx.Client` created in `BaseAPI.__init__`
and reused via `self.client.post(...)` / `self.client.get(...)`. Do NOT reintroduce
module-level `httpx.post` / `httpx.get` — that opens and tears down a fresh TCP
connection per call, which over a trading session leaves thousands of sockets in
TIME_WAIT and exhausts ephemeral ports. The client mirrors the server's own pooling
in `utils/httpx_client.py`.

Keep-alive reuse only happens when the server supports it. Production OpenAlgo runs
under gunicorn+eventlet (keep-alive supported). The local Werkzeug dev server sends
`Connection: close`, so connections cannot be pooled in local dev even though the
client is correct.

## WebSocket feed

`FeedAPI.connect()` opens one WebSocket and authenticates. `subscribe_ltp` /
`subscribe_quote` / `subscribe_depth` accept `[{"exchange", "symbol"}]` and an
optional callback; polled snapshots are available via `get_ltp` / `get_quotes` /
`get_depth`. Auto-reconnect (on by default) replays active subscriptions after a
drop; `ping_interval`/`ping_timeout` guard against zombie sockets.

## Analyzer (sandbox) mode

The server has an analyzer/sandbox mode where orders are simulated instead of being
sent to the broker. Responses carry `"mode": "analyze"`, and that field is the only
reliable marker: sandbox order IDs are plain numeric strings (`YYMMDD` plus an 8-digit
sequence, with no `SB-` prefix), so they cannot be told apart from broker order IDs.
Sandbox GTT IDs look like `GTT-YYMMDD-xxxxxxxx`.
Check it with `client.analyzerstatus()` and switch with `client.analyzertoggle(True/False)`.
Always confirm analyze mode before placing test orders. Note: simulated MARKET orders
need a live price, so they fail when the market is closed (use LIMIT for testing on
holidays).

## Backtest package (`openalgo.backtest`, in progress)

A backtesting, paper and live trading engine for Indian markets is being built inside
the SDK. Only the foundations exist so far; there is no event kernel, Strategy class or
run function yet.

| Path | Responsibility |
|------|----------------|
| `rust/oa_bt/` | Zero-dependency Rust kernel crate: `model` (shared types and IST time helpers), `market` (costs, calendars), `data` (resampling) |
| `rust/oa_py/src/backtest/` | PyO3 bindings, one file per area, exposed as `openalgo._oaindicators.backtest` |
| `openalgo/backtest/` | Python package: costs, calendar, resample, data layer, reference tables. `import openalgo` does not import it |
| `test/backtest/` | pytest suite. It never contacts a server; anything that needs a client uses an in-process fake |

### Code standards for the backtest package

These apply to every change under the four paths above.

- One responsibility per module. Split a file before it grows past roughly 400 lines of
  non-test code and a function before roughly 60 lines. Data tables, parsing and rules
  live in separate modules.
- Small, explicit public surface: `__all__` in Python, `pub use` in Rust. Everything
  else is private.
- One source of truth: shared types in `rust/oa_bt/src/model.rs`, the bars layout in
  `openalgo/backtest/_layout.py`, exceptions in `openalgo/backtest/errors.py`. Do not
  redefine them locally.
- Names use the trader's vocabulary and carry the unit where it matters (UTC
  nanoseconds versus IST, quantity in units never lots, percent versus fraction). No
  magic numbers: use a named constant and say where the value comes from.
- Every public function and class has a docstring that states units, conventions and
  the errors it raises. Comments explain why, not what.
- Nothing falls back silently. Unknown input raises a typed error whose message says
  what to do. Library code never prints.
- Market facts (rates, lot sizes, holidays, session times) are data with a source and a
  verification status, never constants scattered through the code.
- Rust: `cargo fmt` clean, `cargo clippy -- -D warnings` clean, no `unsafe`, and no
  `unwrap`, `expect` or panic on a path that depends on input.
- Python: `ruff check` and `ruff format --check` clean (configuration in
  `pyproject.toml`), type hints on public functions, no bare `except`, no mutable
  module state.
- Tests read as specifications: one behaviour per test, the name states the behaviour,
  table-driven where rows repeat, and expected values are never produced by copying the
  implementation.
- No dead code, no commented-out code, and no abstraction for a feature that is not
  being built yet.
- Original code only, described in its own terms. Never name another backtesting or
  trading engine or library in code, comments, tests, data files, docs, the changelog
  or commit messages, whether as a comparison, an inspiration or a cautionary example.
  Say what this engine does and why. (OpenAlgo, exchanges, brokers and regulators are
  named freely.)

Checks to run before committing:

    cargo fmt -p oa_bt --check --manifest-path rust/Cargo.toml
    cargo clippy -p oa_bt --all-targets --manifest-path rust/Cargo.toml -- -D warnings
    cargo test -p oa_bt --manifest-path rust/Cargo.toml
    ruff check openalgo/backtest test/backtest
    ruff format --check openalgo/backtest test/backtest
    pytest test/backtest

## Bumping the version

The version string lives in three files and they must stay in sync:

1. `pyproject.toml`: `version = "x.y.z"` (source of truth: maturin builds the wheel)
2. `setup.py`: `version="x.y.z"` (legacy setuptools metadata, still committed)
3. `openalgo/__init__.py`: `__version__ = "x.y.z"`

`setup.cfg` contains no version, and there is no committed `PKG-INFO`. After bumping,
grep the repo for the old version to confirm nothing was missed:

    Grep pattern "x\.y\.z" across the repo

Add the release notes to `CHANGELOG.md` under `## [x.y.z] - YYYY-MM-DD` at the top.

Commit message convention (see git log): `vX.Y.Z: short summary`.

## Releasing to PyPI

Do not upload from a developer machine. `.github/workflows/CI.yml` builds abi3 wheels
for five targets (linux x86_64 + aarch64, macOS x86_64 + arm64, windows x86_64) plus
an sdist, and publishes them on a `v*` tag push through PyPI trusted publishing (OIDC,
no API token). A local `maturin publish` would upload one platform's wheel, and the
version number cannot be reused afterwards, so macOS and Linux users would be left
compiling the Rust core from the sdist.

    git tag vX.Y.Z && git push origin vX.Y.Z

## Running / testing locally

- Live tests need a running OpenAlgo server on `:5000` (and the WS proxy on `:8765`)
  plus a valid API key. Do not hardcode API keys in committed files.
- Importing `openalgo` needs no Numba and no environment variable. Numba was removed in
  2.0.0, so `NUMBA_DISABLE_JIT` has no effect on the SDK (`openalgo/numba_shim.py` only
  keeps the old `@jit` / `@njit` decorators importable as no-ops). The indicators run in
  the Rust core (`openalgo._oaindicators`). In a source checkout without a built
  extension, `openalgo/indicators/_backend.py` falls back to NumPy where a fallback
  exists; build the extension with maturin (see `pyproject.toml`) for the full set.
