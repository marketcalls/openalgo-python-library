# Changelog

All notable changes to the OpenAlgo Python Library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.2.0] - 2026-11-13

### Added

- **`Result.trade_list()`**: the closed trades as a pandas DataFrame a trader reads,
  one row each, indexed by trade number: symbol, direction ("long" or "short"),
  product, entry and exit time (IST), quantity in units, entry and exit price, gross
  P&L, charges and net P&L in rupees, `return_pct` (net P&L over the entry value, in
  percent), `bars_held`, `days_held`, `exit_reason` and `r_multiple`. The tearsheet's
  trade table shows the same rows. `Result.performance.trades` gains a `bars_held`
  column. `bars_held` counts bars of the strategy's own interval that closed while the
  trade was open, not base bars: those whose close is after `entry_time` and at or
  before `exit_time`, so a 5-minute strategy on 1-minute data held from 09:40 to 15:00
  holds 64 bars, not 320, and a trade opened and closed inside one bar holds 0. A fill
  at a slice's close (an expiry settlement) is timed at that slice's open, as
  `exit_time` shows, so the bar it closes is not counted. A run has one strategy
  interval for all its symbols (a futures series' contracts included), and each trade
  counts the bars of its own instrument.
- **`Result.summary()`**: a short plain-text summary to print after a run:

      Net profit   : Rs 2,00,000.00
      Return       : 4.00%
      Max drawdown : -0.97%
      Trades       : 1 (100% won)
      Benchmark    : 3.52% return (NIFTY)

  Net profit in rupees with Indian digit grouping, the return and the maximum drawdown
  in percent, the closed trades with their win rate and any positions still open, and
  the benchmark's return for a run with `benchmark=` ("given closes" for closes given
  as a Series). A run with no closed trade (a buy-and-hold or rebalanced portfolio, or
  one whose entries were all rejected or skipped) says "no closed trades" where the usual
  `f"{m['win_rate']:.0%}"` printed "nan%"; the `win_rate` metric itself is unchanged
  (undefined, NaN, with no trades). The win rate is in whole percent but never reads 0%
  or 100% unless it is exactly so: 1 winner in 250 reads "0.4% won" and 249 in 250
  "99.6% won" (the fewest decimals that keep it off the bound). A loss below half a
  paisa reads "Rs 0.00", not "Rs -0.00" (in the tearsheet's rupee figures too).
  `print(result)` is still the full Assumptions block.
- **Orders that were not placed are told on screen.** A `bt.backtest` run in which
  orders were rejected or skipped for a reason the run did not ask for ends with one
  `OrdersNotPlacedWarning` (a `UserWarning` in `openalgo.backtest.errors`, so it can be
  filtered by its class), pointed at the line that ran the backtest: how many orders,
  the most common reason in plain words and its code, for example "3 of 3 orders were
  rejected because a delivery (CNC) short is not allowed in Indian cash markets
  (reject_code 'product_rule'). See result.warnings." or, for a share that costs more
  than the capital in whole shares, "249 orders were skipped because the sized quantity
  was less than one share or, for a derivative, one lot (skip reason 'rounded_to_zero').
  See result.warnings." Before, a reader who printed only a few metrics saw nothing:
  such a run just showed 0 trades. Orders a rule of the run refused as it intends do not
  raise it: the MIS entry window (`entry_from`, `entry_until` or the default cut-off),
  `max_daily_loss`, `drawdown_halt`, `cooldown`, `max_positions` and a size cap
  (`size_caps`); a run with both kinds counts only the others and then gives no total
  of orders ("1 order was rejected because ..."). A sizing function that returns 0 is
  counted, as the kernel gives it the skip reason `rounded_to_zero`. Once per run, not
  per order; `bt.optimize` does not issue it for its combinations, in its own process
  or in its workers.
- **Skipped orders are reported like rejected ones.** `result.warnings` gains one line
  per skip reason that occurred (first, before the other warnings), with how many orders
  it skipped, when the first was decided and what to do, for example "249 orders were
  skipped with skip reason 'rounded_to_zero' (result.skipped), the first decided at
  2025-01-01 15:30 IST: the sized quantity was less than one share or, for a
  derivative, one lot; give the strategy more capital, size with a larger percent, trade
  a cheaper share, or pass quantity_mode="fractional" for part shares ...", and
  `print(result)` a "skipped orders: 249 (rounded_to_zero 249)" line.
  The line for orders refused by the MIS entry window now says they were outside the
  window, gives its times and says they were not placed, as the window intends: "4 of 4
  orders were outside the MIS entry window, 09:15 to 09:20 (entry_from to entry_until),
  and were not placed, as the window intends (reject_code 'entry_window' in
  result.orders), the first decided at 2026-11-04 09:30 IST".
- **The Assumptions block states how orders are sized** (`sizing`): with whole shares,
  an order with no quantity is sized at the decision close and keeps 2 percent of its
  budget in cash for the price's move to the next open (the engine's cash buffer, the
  same in live trading; its value is unchanged), then takes its estimated charges off
  the rest, which is why 10,00,000 at 1,000.00 with no charges buys 980 shares and not
  1,000; `bt.size` sizers and `target()` keep no buffer and `rebalance` keeps its own
  `cash_buffer`. With part shares it is sized at the fill price and spends the whole
  budget less the charges. `Strategy.buy` and `sell` and `bt.size` document the same.

### Changed

- **Indicators warm up automatically** (changes results). A backtest loads history
  before `start` from the run's own source (the source its bars came from, Historify
  or the broker, or the bars of `data=` before `start`): up to 2,048 strategy bars at
  the run's base interval. Every `self.I` indicator, every condition and every
  `from_signals` function is computed over it (`from_arrays` arrays are read for the
  run's bars only). The strategy makes no decision before `start` and none before its
  `self.I` indicators are ready: an indicator is ready once its series has closed its
  warm-up of bars, the length the window-stability check already measured and printed
  in the Assumptions block (for example 128 bars for a 20-day EMA, 512 for a 50-day
  one, 16 for a 3-bar average), the same window live trading keeps. `on_bar`, timers,
  conditions and arrays wait alike. Equity, trades, charges, metrics and the benchmark
  still start at `start`. When the source has fewer bars before
  `start` (own data that begins at `start`, a new listing, a futures series, which is
  built from `start`), the strategy waits inside the run until its indicators are
  ready, and a warning says how many strategy bars it waited and why. The Assumptions
  block gains `warm_up` (the rule, each series' need, the history loaded and the first
  decision), the manifest records the warm-up used under `arguments["warm_up"]` and the
  history's partition hashes under `data["<symbol> history"]`, and `bt.optimize`,
  `rerun()` and the single runs of a sweep's combinations use the same history. A
  strategy's own warm-up guard (`if len(self.close) < self.slow: return`) still works;
  it is no longer needed. An indicator no window settles (a cumulative series such as
  on-balance volume) is not waited for, with a warning. Each indicator's warm-up is now
  measured with its parameters: `ema(close, 20)` and `ema(close, 50)` each get their
  own (the truncation check still runs once per function and data). The first trades
  of runs on short own data move later; the golden pins were regenerated for this.
- **Whole shares by default** (changes results): `quantity_mode` defaults to "venue"
  (whole shares of cash equity, whole lots of derivatives) instead of "fractional", so
  every sizer, an order with no quantity, `target()` and `rebalance` round down to whole
  shares unless a run passes `quantity_mode="fractional"`, which stays for research.
  A `target(value=...)` or `target(percent=...)`, or a rebalance weight, that rounds down
  to no share while nothing is held is now a skipped order (`result.skipped`, reason
  `rounded_to_zero`, and the warnings above) instead of passing silently; `target()`
  still returns None for it.
- **Daily and longer bars indexed by naive midnight dates need no `tz=`.** The daily
  bars `client.history(...)` returns are indexed that way, and passing them as
  `bt.backtest(..., data=df)` stopped with "the timestamps of the supplied data carry
  no time zone". A naive index (or timestamp column, or CSV dates) whose times are all
  midnight is a list of session dates, which no time zone can shift, so for D, W, M, Q
  and Y bars it is read as IST session dates, as `tz="Asia/Kolkata"` reads it, and the
  Assumptions block says so in one line ("timestamps: SBIN: naive midnight timestamps
  read as IST session dates (daily bars need no tz=)"). Intraday naive timestamps, and
  daily ones stamped at a time of day, still need `tz=`: a wrong guess there moves
  every bar by 5 hours 30 minutes.
- **The HTML tearsheet is a dashboard in the layout of OpenStatz's dashboard**
  (`result.tearsheet(path)`, `report.tearsheet`): a sticky header with section links,
  Export PDF and a light and dark toggle, then numbered sections: Overview (headline
  tiles), Performance (cumulative return against the benchmark, rolling Sharpe,
  volatility and win rate, CAGR, maximum drawdown and Sharpe by horizon, period
  returns), Risk (drawdown chart, risk tiles, the benchmark tiles and rolling beta, the
  ten worst drawdowns, losing streaks), Monthly (monthly and weekly heatmaps, monthly
  tiles, yearly returns against the benchmark), Distribution (tiles, histogram, box
  plot), Metrics (every metric, the benchmark's beside it, sortable), Trades (trade
  tiles, the sortable and paged trade list, charges by component, the R-multiples) and
  Notes (how the figures were computed, the assumptions and warnings). A tile's
  subline is the benchmark's figure measured as the strategy's is, where a benchmark
  has one. It is still one offline file: inline CSS and SVG, plus a small inline script
  for the toggle, sorting, paging, the weekly year picker and a chart crosshair; it
  reads in full with script turned off. `mode="basic" | "metrics" | "charts"` and
  `log_scale=` show subsets of the same layout.
- The tearsheet no longer draws the Monte Carlo band and histograms, the session
  return bars, the trade P&L histogram or the trade losing-streak table:
  `Result.monte_carlo()`, `performance.session_returns`, `Result.trade_list()` and
  `performance.losing_streaks` keep them.

## [2.1.1] - 2026-11-13

### Added

- **`Result.final_positions` shows each position's `value_rupees` and `weight_pct`** (its
  market value at the mark, in percent of the run's final equity), so a portfolio's end
  weights read off directly; an optimisation table counts `trades` in whole numbers. A
  future has neither (NaN): it is not bought with cash, and its contract value, often
  several times the equity, would read as a share of the portfolio;
  `equity["exposure_rupees"]` counts it.
- **`self.rebalance(weights, cash_buffer=2)` in `openalgo.backtest` (preview).** Trades a
  portfolio to target weights, in percent of equity by symbol, at one decision: a
  symbol of the strategy left out of the weights is sold to flat, `cash_buffer`
  percent of equity stays in cash for charges and the move to the next open, and
  each symbol gets one `target(value=...)` order. Every sale fills before every buy
  at the same open, whatever order the symbols were registered in, so the buys can use
  the sales' proceeds. Its buys are intended adds: the averaging-down refusal
  (`allow_averaging_down = False`, the default) does not apply to them; the other
  scale-in limits do. Every order of a rebalance is tagged "rebalance" in
  `result.orders`. Scheduling stays in the strategy (`on_bar`, or `self.every`). A
  timer that rebalances at an open where an earlier rebalance's buys are due trades
  from the position those buys make: they fill before the timer runs. Not
  built yet: scaling the buys to the cash left at the fill, no-trade rules, weight
  tables and functions, drift bands, the rebalance log and `bt.rebalance`.
- **The credit for selling delivery holdings is a dated table** (`sale_credit_defaults.csv`
  in the kernel, each row with its source and status): the whole sale value is cash
  for trading on the day of the sale before 2020-12-01 (unverified: observed for
  October 2020 and applied to earlier dates) and from 2024-10-07, and 80 percent from
  2020-12-01 to 2024-10-06 with the rest on the next trading day (under SEBI's peak
  margin rules 20 percent stayed blocked until the shares were delivered; the broker
  sources are in the table). It applies to shares (ETFs included) sold on CNC; a future
  or an option sold on CNC withholds nothing. Units bought and sold on one day are an
  intraday trade and are credited in full. A run that uses the unverified row warns.
  Not modelled: the next-day credit of shares sold before their delivery (bought the
  previous day).
- **`Result.to_csv(folder)` and `Result.to_excel(path)`** write every table of
  `to_dataframes()`: one CSV file per table, or one workbook with a sheet per table;
  both create the folder they write into.
  Times are written as plain IST date-times (a spreadsheet cell holds no time zone, so
  `to_excel` on a table failed before); the `*_ts` columns stay UTC nanoseconds. Excel
  needs openpyxl, which is not installed with openalgo: without it `to_excel` raises
  `MissingDependencyError` saying `pip install openpyxl`.
- **`SymbolNotFoundError`** (a `DataError`, in `openalgo.backtest.errors`): the OpenAlgo
  server answered that the symbol is not in its master contract. `symbol` and
  `exchange` name what was asked for.
- **Charges from 2020-07-01 (`openalgo.backtest.costs`).** Eleven earlier schedule
  versions, from IN-2020-07-01 (the day stamp duty became one rate across states) to
  IN-2024-05-13, so a backtest from mid 2020 is charged the rates that applied on each
  trade date instead of stopping with "no cost schedule covers ...". They cover every
  segment (NSE and BSE equity, NFO, BFO, MCX futures) and each change has its own
  version: NSE transaction charges (2021-01-01, 2023-04-01, with the NSE IPFT levy),
  BSE group A and B charges (2021-03-01, 2022-12-01), the SEBI fee (half rate until
  2021-03-31), MCX KAPAS (2021-05-01), F&O STT (2023-04-01), BSE options charges
  (2022-05-02, 2023-11-01, 2024-05-13) and GST on the SEBI fee (2022-07-18: the fee was
  exempt from GST before, so earlier trades pay GST on brokerage and the exchange charge
  only). Every version names its sources (exchange circulars and notices, SEBI
  notifications and circulars, the Finance Acts, the GST rate notification) and a row is
  verified only where a primary source states its rates and their dates; the other rows
  are unverified, say why in their notes and warn: BSE groups X and SS, MCX RBDPMOLEIN,
  MCX in 2020, Sensex options from 2023-11-01 to 2024-05-12, NSE equity from 2020-08-01
  to 2021-03-31 (NSE charged shares outside the NIFTY 50 and NIFTY Next 50 a lower rate,
  and the engine does not know an index's members) and the MCX futures at Rs 2.60 per
  lakh until 2021-03-31 (the third expiry and later paid half; the rate is right for the
  current_month and next_month contracts a futures strategy holds, but the engine does
  not know the expiry rank of a contract traded on its own). `costs.schedules()` lists
  them; each version's `gst_on_sebi_fee` says whether GST falls on the SEBI fee, and a
  user schedule may set it (True by default).

### Changed

- **Delivery (CNC) cash in a backtest is what a cash account holds.** The free cash
  that the margin check at a fill, `self.cash` and sizers on free cash use now counts
  a delivery holding at the price paid for it: its open profit is no longer spendable
  and its open loss no longer blocks cash already left over. Before, a portfolio of
  holdings that had fallen had its buys refused for margin although the cash was
  there (the rejections seen when rebalancing a buy-and-hold portfolio), and one that
  had risen could buy with its paper profit. This applies to shares (ETFs included):
  intraday (MIS) positions, and futures and options on any product, CNC included,
  block their margin as before.
- **Every time in a `Result` table is an IST datetime.** `orders["decided_at"]` and
  `orders["good_until"]` (NaT for a GTT order) and `skipped["decided_at"]` were UTC
  nanoseconds; `stale_marks` and `settlements` gain a `time` column; `rolls` has an
  IST `date` in place of its day number. The `time` column of `rolls` and
  `settlements` is an IST datetime column in a run without futures too, where the
  tables are empty. The kernel's UTC nanoseconds stay only in the `*_ts` columns.
- **Rejected orders are reported.** A backtest now adds one warning per reject code
  that occurred: how many of the run's orders it refused, when the earliest was decided
  (IST), what the code means and what to do (for `product_rule` on a stock: a delivery
  short is not allowed, short intraday with MIS or trade futures; on an index: an index
  cannot be traded, trade its futures or options). `print(result)` has a
  `rejected orders:` line with the count by code. Before, a run whose entries were all
  rejected ended with no trade and no warning. The warning for orders a sizer sized and
  the margin refused is now part of the margin warning, which also covers orders given
  in units or lots.
- **An expired futures contract is named as such.** When a run needs the history of a
  contract that expired before today and the server no longer lists it, the error says
  that the contract expired on its date (its last trading day, which a holiday can
  move before the expiry its symbol names), that brokers stop serving an expired
  contract's history, and to start after that date or run on stored bars with `data=`.
- **The underlying of a contract is read from its symbol for the charges.** An MCX
  future (or a BFO option) given as an `Instrument` without `underlying=` is charged
  by the commodity its OpenAlgo symbol names, as the futures margin already was;
  `CostError` remains for a symbol that names none.
- **Unverified charge rates are announced once.** A run charged at rates that were not
  verified warns with a short `UnverifiedScheduleWarning` once per process for each
  cost schedule, segment and reason, and puts the details (the date and the schedule's
  notes) in `result.warnings`. The worker processes of `bt.optimize` no longer repeat
  it. The cost model's own functions still warn with the full text, which now ends "so
  the charges may be off".
- **`bt.from_arrays` errors say what to pass.** A missing `symbol=`, `exchange=` or
  `interval=`, or any value given by position after `exits` (the message names
  `symbol=`, `exchange=`, `interval=`, `shorts=` and `covers=`), raise `ConfigError`
  with a one-line example instead of a `TypeError`.
- The `buy` and `sell` docstrings state that with `reverse=True` only the quantity
  beyond the open position opens the other side (flipping one lot takes two lots), and
  `target` is documented as the one-call way to flip.
- A trade dated before the first schedule version (now 2020-07-01) still raises
  CostError unless `backfill=True`, and the message now says from which date charges
  are available. Trades from 2020-07-01 to 2024-09-30 that a model with `backfill=True`
  used to charge at the 2024-10-01 rates are now charged at the rates of their own
  dates. A known limit of these versions, stated in their notes: exchange charges,
  tiered by a member's turnover before 2024-10-01, are the first-slab rates.

### Fixed

- **CTT on MCX agri futures outside the exemption.** Commodities transaction tax (0.01
  percent of the sell value) is exempt only for the agricultural commodities listed in
  rule 3 of the CTT Rules 2013, as substituted by notification 13/2015 (Finance Act 2013
  sections 116(7) and 117). Every charge schedule exempted all MCX agri contracts, so
  sales of CPO, MENTHAOIL, RUBBER, COTTONOIL, SUNOIL and RBDPMOLEIN futures were charged
  no CTT. They now pay it in every schedule version, through a new segment
  `MCX_FUT_AGRI_CTT` (CPO, MENTHAOIL, RUBBER, COTTONOIL, SUNOIL; the agri exchange charge
  and SEBI fee, plus CTT) and CTT on the `MCX_FUT_RBDPMOLEIN` rows. `MCX_FUT_AGRI` now
  holds CARDAMOM and COTTONCNDY only, and CASTORSEED, PEPPER and KAPAS stay exempt; all
  five are in rule 3. A user schedule that costs CPO, MENTHAOIL, RUBBER, COTTONOIL or
  SUNOIL needs a `MCX_FUT_AGRI_CTT` row. With the reason for their status gone, the
  MCX agri rows of 2021-01-01 to 2024-09-30 are verified like the other MCX rows.
- **Charge rates from 2024-10-01 are checked against primary sources.** The
  IN-2024-10-01 version (to 2026-03-31) was unverified in full, so every run dated in it
  warned. Its GST and its NSE, BSE group A, B and M, NFO, BFO and MCX rows are now
  verified through 2026-03-31 against NSE circulars NSE/FA/73061 and NSE/FATAX/63809
  and 73524, the BSE fee pages and notice 20240927-37, MCX circular MCX/F&A/631/2024,
  the SEBI fee regulations (2019 and the Stock Brokers Regulations 2026), the stamp
  duty schedule and the GST rate notification; both versions now name these sources.
  Still unverified, with the reason in the row's notes, in both IN-2024-10-01 and
  IN-2026-04-01: BSE groups X (the same row charges ZP, which pays Rs 1,00,000 per
  crore) and SS (no longer on the BSE fee page). In IN-2026-04-01 these four rows were
  marked verified before and now warn; BSE options other than Sensex and Bankex stay
  unverified there.
- **MCX CASTORSEED, PEPPER, KAPAS and RBDPMOLEIN futures pay the uniform exchange charge
  from 2024-10-01.** MCX circular MCX/F&A/631/2024 charges every futures contract Rs 2.10
  per lakh of turnover from that date. The schedules kept the contract-specific rates of
  the circulars it replaced (CASTORSEED Rs 0.50, PEPPER Rs 0.05, KAPAS Rs 2.60 and
  RBDPMOLEIN Rs 1 per lakh) in IN-2024-10-01 and IN-2026-04-01; both now charge Rs 2.10
  per lakh, verified against the circular. Earlier versions keep the old rates.

## [2.1.0] - 2026-11-12

### Added

- **WebSocket connection hooks.** `on_connect`, `on_disconnect`, `on_reconnect` and
  `on_error` register listeners on the feed (`FeedAPI` and `api`); `remove_hook`
  removes one. A connect fires after authentication, a disconnect carries the close
  code, reason and whether a reconnect will follow, and a reconnect fires after the
  server has answered the replayed subscriptions and lists what it accepted. Events
  arrive in order, outside every SDK lock, and each payload has a connection
  sequence number `seq`.
- **`get_subscriptions()`** returns the set of subscriptions the feed is asked to
  hold, which is exactly what a reconnect replays.
- **`FeedServerError`** (`.code`, `.message`) describes a rejection sent by the server.
- **`ta.rvi_volatility(data, stdev_period=10, rsi_period=14)`**: the Relative
  Volatility Index (Dorsey). `ta.rvi` remains the Relative Vigor Index.
- **`ta.psar(..., return_trend=True)`** also returns the trend (+1 long, -1 short).
- **`ta.vwap(..., timestamps=None, session_starts=None)`** and
  **`ta.pivot_points(..., anchor=None, timestamps=None)`**: pass bar times explicitly
  when the input is a plain array. `pivot_points(anchor="D")` (or W, M) gives the
  previous completed period's levels on every bar.
- **`openalgo.indicators.causality`**: `is_causal(name, output=None)`,
  `non_causal()`, `outputs(name)` and `causality(name=None)` report which indicator
  outputs use only data up to the current bar. The outputs that look ahead are
  `ichimoku` lagging span, both `fractals` outputs and the `ZigZag` class. A test
  across the whole library keeps this list complete.
- **`openalgo.backtest.calendar` and `openalgo.backtest.resample` (preview).** Session
  calendars for every OpenAlgo exchange with built-in holiday lists for 2019 to 2026,
  dated regular hours (including the closing auction changes of 2026) and default
  intraday entry and square-off times, each marked by how well it is sourced; a year
  without a holiday list is reported as such, never treated as holiday-free.
  `validate_intraday` catches series shifted by 5 hours 30 minutes, bars stamped at
  their close, bars off the session lattice and bars on closed days. `resample_bars`
  builds 5m, 15m, 1h, daily, weekly and monthly bars from 1-minute or daily bars on
  the same lattice as Historify, and drops and counts bars outside the session. Part
  of the backtest engine being built; the interface may change before the backtest
  beta.
- **`openalgo.backtest.data` (preview).** `load_bars(symbol, exchange, interval, start,
  end, ...)` loads bars from an OpenAlgo server (`source="api"` for broker history,
  `"db"` for Historify, `"auto"` to use Historify only when it covers the whole range)
  or from your own DataFrame or CSV, Parquet or Feather files with no server at all.
  Every source is normalised to one layout with UTC nanosecond bar open and close
  times, checked against the session calendar, and quality checked; a short history,
  dropped bars and every other adjustment are reported in the returned metadata. A
  local cache keeps the raw rows each source returned, verifies them on every read,
  and treats a session as complete only well after its close. No market-data
  downloader is bundled. Part of the backtest engine being built; the interface may
  change before the backtest beta.
- **`openalgo.backtest.reference` (preview).** Instruments and dated reference tables
  for Indian derivatives: OpenAlgo symbol parsing and formatting, lot size, freeze
  quantity, strike step, tick size, settlement and expiry dates by trade date for the
  NSE and BSE index underlyings, and the main MCX contracts. Each row cites its exchange
  circular or file and is marked verified or unverified; using an unverified row warns,
  and a date the tables do not cover raises instead of guessing. Contracts re-dated by
  the exchange after listing keep their listed expiry and their real last trading day.
  The tables reproduce the listed contracts, last trading days and lot sizes of 92
  days of NSE and BSE bhavcopies. Part of the backtest engine being built; the
  interface may change before the backtest beta.
- **`openalgo.backtest.backtest` and `openalgo.backtest.Strategy` (early preview).** The
  backtest kernel runs a strategy class through the Rust event loop and returns
  fills, positions, equity, trades and events as DataFrames. The entries below cover
  each part. What is not built yet (walk-forward, `bt.live`) raises
  `NotImplementedError` naming its phase.
- **Strategy and session risk rules in `bt.backtest` (early preview).** `max_daily_loss`
  measures the day's loss on `risk_basis` as the realized P&L of the strategy's runs
  that ended this session plus the live run's MTM since its first fill; a breach exits
  every position (at the next base-bar open, or at the day's close on a daily base),
  writes `daily_loss_limit_hit` and leaves the strategy REDUCING until the next
  session, rejecting new entries with `daily_loss_gate`. `drawdown_halt=bt.pct(x)`
  latches HALTED for the rest of the run (reject code `halted`); `cooldown` and
  `max_positions` reject entries with `cooldown` and `max_positions`; the class
  attribute `max_hold` exits a position after that many base bars (reason
  `time_exit`). Trading-state changes reach `on_risk_event` as `trading_state_changed`,
  and the Assumptions block lists every risk setting. A class attribute wins over the
  run argument of the same name. The combined rules (`combined_stop`,
  `combined_target`, `lock_profit`, `trail_sl_to_entry`) raise `NotImplementedError`
  until Phase 3.
- **Leg rules in `openalgo.backtest` (preview).** `buy` and `sell` on MIS take `stop=`,
  `target=`, `targets=[(distance, fraction), ...]`, `trail=` (`bt.trail(x, y)`,
  `bt.trail_points`, `bt.trail_pct`, `bt.trail_atr`), `breakeven=` and
  `after_first_target="breakeven"`. Levels come from the fill average, round to the
  tick, and are tested on every base bar from the entry bar on: a level crossed at the
  open fills at the open, the stop beats a target inside one bar, several targets can
  fill in one bar, and a trail or breakeven moved by a bar is in force from the next.
  Adds recompute the stop (never loosening it) and the unfilled targets from the new
  average. Exit fills carry the reason `leg_sl` or `leg_target` and the target index,
  with the matching risk events. `bt.atr` distances use the ATR from `openalgo.ta` at
  the decision. `reverse=True` closes a position and opens the rest on the other side;
  without it an order larger than the opposite position is rejected with
  `reverse_not_allowed`. Leg rules on CNC and NRML positions are held as GTT orders
  (see below); `max_hold` is the strategy-level time exit.
- **Backtest accounting and results (early preview).** `bt.backtest` now runs on its
  defaults, `costs="zerodha"` and `slippage="default"`, with no server: every fill is
  charged by `openalgo.backtest.costs` at the rates of its trade date, the day's
  contract notes are rounded at the session close (`Result.costs`,
  `Result.contract_notes`), delivery sells pay the DP charge, and `costs=` also takes a
  preset (`bt.costs.flat(20)`) or a `bt.costs.CostModel` (calculator rounding, own
  schedules). Slippage comes from a dated per-class table (`bt.slippage.defaults()`):
  2 bps for cash equity, 1 or 2 ticks for futures, a percent of premium for options,
  overridable with `bt.ticks`, `bt.pct`, `bt.slippage.bps` or `bt.slippage.table(...)`.
  Margin comes from a dated table (delivery in full, intraday 1x unless the order asks
  for up to 5x, long options the premium, NSE index futures a percent of value); a row
  that is not verified is named in `Result.warnings`. Orders with no quantity take the
  engine default (1x the capital net of estimated costs, one lot for F&O);
  `bt.size.fixed_risk` and `bt.size.atr_risk` size to the stop; sized quantities are
  capped at the freeze quantity. `Result.performance`, `Result.metrics` and
  `Result.tearsheet()` report the run. A delivery (CNC) buy and sell of one scrip on the
  same day is charged as intraday with no DP charge: the units a CNC fill matches
  against the day's fills on the other side are charged at the intraday rates and the
  earlier fills charged again. Short-option margin is not built yet.
- **LIMIT, stop and GTT orders in `openalgo.backtest` (preview).** `buy` and `sell` take
  `price=` (LIMIT) and `trigger_price=` (SL with a price, SL-M without), OpenAlgo's
  order types. A LIMIT or stop order rests from the next base bar until it fills or its
  session closes, and is matched on every base bar: the open first, then the range. A
  LIMIT fills at the open or at its limit with no slippage; an SL-M fills at its
  trigger, or at the open on a gap, plus slippage; an SL fills like a stop whose limit
  then holds the price, and a stop-limit exit that triggers without filling becomes a
  market order at the next bar's open. On options an SL-M is sent as an SL with its
  limit 5 percent beyond the trigger. A limit or trigger off the tick grid is rejected
  with `tick`. On CNC and NRML, `stop=`, `target=`, `targets=`, `trail=` and
  `breakeven=` are now held as GTT orders: one SINGLE or OCO per target with a stop for
  the rest, placed with the fill that opens the position, modified at most once per
  base bar when a trail, breakeven or add moves a level by at least a tick, cancelled
  before any other exit, and re-created when older than 360 days. A fired GTT places a
  LIMIT child 1 percent beyond its trigger (5 percent for options); a child that a gap
  leaves unfilled is cancelled and the engine exits at the next base-bar open, flagged
  `gtt_child_unfilled`. `validity="gtt"` rests an order as a GTT across sessions (MIS
  is rejected with `product_rule`). The event stream carries the GTT events
  (`GttPlaced`, `GttModified`, `GttTriggered`, `GttChildFilled`, `GttChildCanceled`,
  `GttCanceled`), fills carry `gtt_id`, and the orders table shows the type, prices and
  validity. The kernel's `Engine.modify` replaces a working order in full.
- **The strategy surface of `openalgo.backtest` (early preview).** `self.I(ta.x, ...)`
  registers an indicator from `openalgo.ta`, computed once in Rust and revealed bar by
  bar, with `[-1]`, slices, comparisons, arithmetic, `.crossed_above` and
  `.crossed_below`; outputs that look ahead (`ta.fractals`, the Ichimoku lagging span)
  are rejected by name, and each indicator passes a truncation check and a
  window-stability check (warm-up length, tolerance in the Assumptions block) once per
  strategy and data. `self.data[symbol, "15m"]` shows another interval, each bar only
  once it has closed. `self.at("09:20", fn)` and `self.every("month", fn)` are timers
  whose orders fill at that same open. Without `base=`, an intraday run uses 1-minute
  bars when 1-minute data covers the whole run and the strategy interval otherwise, and
  says which and why. Futures and options take their lot size, tick and freeze quantity
  from the reference tables. `bt.optimize` sweeps a parameter grid in spawned worker
  processes (Windows included) and returns `.table`, `.best`, `.select()`,
  `.heatmap()` and `.path`; `Result.rerun()` repeats a run bit for bit or with
  overrides, and `Result.manifest` records what is needed to reproduce it.
- **Condition and array strategies in `openalgo.backtest` (preview).** A strategy with
  no `on_bar` declares `entry_when(cond, **order_args)`, `exit_when(cond)`,
  `short_when(cond, **order_args)` and `cover_when(cond)` in `init`; the conditions are
  read once per run over the strategy bars and the whole run then goes through the Rust
  kernel with no Python call per bar. The decisions are exactly those of the equivalent
  `on_bar` of the strategy contract: level-triggered, long entry before short, exit
  before adding, `pyramiding` adds (default 0), the position read after the fills of the
  bar, a queued entry counting as not flat. A condition is a view of booleans built from
  series views and `self.I` indicators (comparisons, arithmetic, `.crossed_above`,
  `.crossed_below`, and new on every view: `&`, `|`, `~`, `.shift(n)`,
  `.rolling_max(n)`, `.rolling_min(n)`), a clock condition (`bt.time_between("09:30",
  "14:30")`, `bt.weekday("Mon", ...)`, combined with views by `&` and `|`), or an array
  with one boolean per strategy bar; anything else (a number, a plain `True`, text, a
  look-ahead indicator output, a negative shift) raises `ConfigError` saying to use
  `on_bar`. A view of another interval counts with its latest closed bar.
  `bt.from_arrays(entries, exits, symbol=..., exchange=..., interval=...)` and
  `bt.from_signals(func, symbol, exchange, interval)` build such a strategy from boolean
  arrays or from a function of the strategy bars; a `from_signals` function is re-run on
  cut data before the run and refused if its signals look ahead. The order arguments are
  `qty`, `size`, `lots`, `stop`, `target`, `targets`, `trail`, `breakeven`,
  `after_first_target`, `product`, `leverage` and `tag`; ATR distances and
  `bt.size.atr_risk` in them take the ATR of the deciding bar. A condition strategy
  trades one symbol and has no callbacks, timers or sizing functions.
- **Futures strategies in `openalgo.backtest` (preview).** A strategy names an
  underlying (`underlying = "NIFTY"` with `exchange = "NFO"`, or underlying names in
  `symbols`) and the contract rank in `expiry` (`"current_month"`, the default, or
  `"next_month"`); the contracts come from `instrument=` or the reference tables (lot
  size by contract expiry, final last trading days). `bt.backtest(roll=1)` (the
  default) rolls a position N sessions before expiry at the roll session's first open:
  near leg first, both legs charged, reason `roll`, stop, target, trail, breakeven and
  the favourable extreme shifted by the spread of the two raw opens, a level crossed at
  the near contract's open exits there instead, GTTs cancelled and placed again on the
  next contract; a roll does not end a run for the daily loss limit. `roll=None` holds
  the position to expiry and settles it at the close of the last trading day (reason
  `expiry`, flagged, not charged). `on_bar` and every series view and `self.I`
  indicator read one back-adjusted series per underlying, re-adjusted at each roll as
  known at the decision point (point-in-time causality checks); the bar `on_bar` gets is
  the contract's own. `Result.settlements` holds the daily futures mark to market (the
  last close, flagged, where the data gives no settlement price) and `Result.rolls`
  every switch with its spread. Trades are split at a roll (exit reason `roll`).
  Condition strategies on an underlying and other intervals of a futures series raise
  `NotImplementedError`.
- **A missing margin rule stops a run before any data is loaded,** with the engine's
  message, instead of at the first entry.
- **Futures margin in backtests, and `margin=` in `bt.backtest` and `bt.optimize`
  (preview).** Every futures contract, on any exchange, underlying and date, blocks 20
  percent of its contract value (5x) by default, since real margins differ by underlying
  and change every day; a run on the default does not warn, and the Assumptions block
  states it. The futures margin of a backtest can be the user's own: one value for every
  futures contract
  (`margin=bt.pct(15)`), or a dict by underlying symbol (`margin={"NIFTY": bt.pct(12),
  "CRUDEOIL": bt.rupees(450_000)}`) whose missing names keep the default. `bt.pct(x)` is
  x percent of the contract value (at most 100); `bt.rupees(x)` is x rupees per lot of
  the contract traded, the figure a broker's margin calculator shows, so after a roll
  that changed the lot size the next contract's lots count. A bare number, a percent
  above 100, a contract symbol or a lower-case name raises `ConfigError` saying what to
  write; a name the run does not trade, or `margin=` on a run without futures, is a
  warning. The Assumptions block states the futures margin in force for each underlying
  traded and whether it is the default or `margin=`, the manifest records it,
  `Result.rerun()` repeats it and `bt.optimize` applies it to every combination. It is a
  backtest assumption only: in live trading the broker computes margin.
- **Ticks chosen by a price slab come from the run's data.** An NFO or BFO contract whose
  tick follows a reference close (an index future's by the index close on the previous
  month's last trading day, a stock derivative's by the share's) no longer asks for
  `instrument=`: the backtest loads that close from its own source, the underlying's
  when the source has it, else the contract's own close that day (said in the
  Assumptions). It still asks when the data has neither, and refuses a tick that would
  change during the run.
- **Scale-in limits in `bt.backtest` (preview).** `max_adds`, `max_position` (units,
  lots, rupees of value or percent of equity), `min_spacing` (base bars, or a price
  distance in points, percent or ticks), `max_risk` (rupees or percent of equity) and
  `allow_averaging_down` (default False) are checked on every add, at the decision
  price, after the cooldown and maximum positions; a broken limit rejects the add
  with code `scale_in_limit`. An entry from flat is never held to them. A class-level
  `leverage` is the default of every order; `leverage=` on an order wins.
- **Target positions, sizing functions, reinvest and size caps in `bt.backtest`
  (preview).** `self.target(qty=, lots=, value=, percent=)` trades to a target position
  (negative is short): units and lots as given, value and percent of equity converted
  at the decision price and rounded towards zero to whole shares or lots. It sends one
  market order for the difference to the position the strategy will hold once its
  queued market orders fill (a reduction only reduces, a move to the other side
  reverses in that order), returns None when there is none, and leaves resting limit,
  stop and GTT orders working. `size=` (on an order or as the strategy's `size`)
  accepts a function `size(ctx, order)` that returns units: `ctx` has the equity,
  capital, cash, decision price, stop distance, lot size, multiplier and position,
  `order` the symbol, side, product and tag; 0 skips the order. `reinvest = False`
  sizes every sizer, percent cap and `target(percent=)` from the initial capital
  instead of the current equity. `size_caps = bt.size.caps(position=, leverage=,
  risk=)` cuts what any sizer gives to a largest position (units, lots, rupees or
  percent of equity, counting what is held), a leverage on equity, and a largest loss
  of the order at its stop; an order with no room left is skipped with reason `cap`.
  Both are in the Assumptions block.
- **R-multiples (preview).** `Result.trades` gains `initial_risk_rupees` (what the
  opening fill would have lost at the stop its entry set) and `r_multiple` (net P&L
  over it; NaN for a trade opened without a stop). `Result.r_summary` gives their count,
  mean, median, standard deviation, best, worst, total, share above zero and the
  distribution in one-R bins, and the full tearsheet has an R-multiples section.
- **ATR rules in condition strategies (preview).** `bt.atr` distances, `bt.trail_atr`
  and `bt.size.atr_risk` now work in `entry_when` and `short_when` (they raised
  NotImplementedError): each entry takes the `openalgo.ta` ATR of the deciding bar, so
  a condition strategy and the equivalent `on_bar` give the same fills. An entry
  decided before the ATR has a value stops the run with a message, as in `on_bar`.
- **No strategy setting or run argument is accepted and then ignored.** Each one
  works, warns when it cannot apply to the run (`pyramiding` on an `on_bar`
  strategy, intraday times on a delivery run, `client=` given with own data) or
  raises; a test pins the audited lists. The Assumptions block states `entry_from`,
  `leverage`, the scale-in limits, and the intraday times as they applied over the
  whole run (one time, the rule they follow, or dated periods). Own data with naive
  timestamps and `tz=` gives the run its dates; epoch numbers with `unit=` are UTC
  and need no `tz=`. Every Result table keeps its columns when it has no rows.
- **`openalgo.backtest.metrics` and `openalgo.backtest.report` (preview).** Performance
  metrics computed in the Rust core (returns, CAGR on calendar time, volatility, Sharpe
  and Sortino with a per-session risk-free rate, drawdowns, Calmar, exposure, trade
  statistics, costs, turnover, monthly and yearly returns, benchmark alpha, beta and
  correlation), annualised with the real number of sessions per year from the session
  calendar. Several strategies combine into one equity curve with a correlation
  matrix. `analyse(..., opening_wealth=capital)` reads every equity point as a mark
  after the capital, so the P&L of the first point, the return of its session and a
  fall below the capital all count; `Result.performance` passes the run's capital
  (before, a backtest's first slice and, on daily runs, its whole first session were
  left out), and `combine_equity(..., opening_wealth=capitals)` holds each strategy at
  its capital until its first mark. `report.tearsheet()` writes a self-contained HTML
  report with inline charts that follows the reader's light or dark setting and needs
  no plotting package. Part of the backtest engine being built; the interface may
  change before the backtest beta.
- **Complete performance reporting in `openalgo.backtest` (preview).** Every figure is
  computed in the Rust core and defined, with its source, in `metrics.DEFINITIONS`:
  smart, adjusted and probabilistic Sharpe and Sortino, the autocorrelation penalty,
  Omega, gain to pain, the risk-return ratio, return over exposure, ulcer index,
  ulcer performance index, serenity index, recovery factor; the distribution of
  session returns (expected and average return, best and worst session, month and
  year, session win rate and streaks, skew, excess kurtosis, historical value at risk,
  expected shortfall, conditional drawdown, tail and outlier ratios); trade ratios
  (win/loss count ratio, Kelly criterion, risk of ruin as the Lundberg bound, CPC
  index, common sense ratio); and against a benchmark R squared, tracking error,
  information ratio and Treynor ratio. `Performance` adds `wealth`,
  `drawdown_episodes` with `drawdown_details(top)`, `rolling` (volatility, Sharpe,
  Sortino, win rate and beta over a window of sessions, `rolling_window=`), `horizons`
  (1M, 3M, 6M, 1Y, 3Y, 5Y and since start), `losing_streaks`, `benchmark_returns` and
  `benchmark_comparison(period)`. `metrics.monte_carlo()` resamples session returns or
  trades with a seeded generator (the same seed gives the same simulations on every
  platform) and reports the percentiles of final return, CAGR, maximum drawdown and
  Sharpe, a cumulative-return band and the share of paths that reach a bust or goal
  level. `report.tearsheet(..., mode=)` renders a "full", "basic", "metrics" or
  "charts" page, still self-contained (inline SVG, no script, no external URL), with
  the new charts: cumulative return against the benchmark (`log_scale=True` for a log
  axis), session return bars, return distribution by period and its histogram, the
  longest drawdowns shaded on the equity curve, rolling statistics and Monte Carlo.
  `report.metrics_table()` gives the metrics as a DataFrame; `Result.tearsheet(path,
  mode=, log_scale=)`, `Result.metrics_table()` and `Result.monte_carlo()` do the same
  for a run. `analyse(..., opening_time=)` passes the instant the run opened with its
  capital: the start, elapsed time and CAGR are counted from it, exposure counts the
  first slice, and a drawdown from the opening is timed from it; `Result.performance`
  passes the open of the run's first slice. Every drawdown figure is now zero or
  negative: `max_drawdown_amount` changed sign (a fall of 500 rupees is -500, as
  `max_drawdown` is a negative fraction). `Result.returns` measures the first slice
  from the capital instead of reporting 0. Equal returns now have a standard
  deviation of exactly 0, so Sharpe is undefined rather than enormous when a series
  of equal nonzero returns averages to a rounding unit away from them.
- **A benchmark for a backtest (preview).** `bt.backtest(benchmark=...)` takes an index
  symbol (`"NIFTY"`, read on NSE_INDEX for an NSE or NFO strategy and on BSE_INDEX for
  BSE or BFO), a `(symbol, exchange)` pair on NSE_INDEX or BSE_INDEX, or your own
  closes as a pandas Series (or a DataFrame with a close column; a date index is
  stamped at each session's close). An index's daily closes come through the same data
  layer and source rules as the strategy's bars (a dict `data=` holding the symbol,
  else Historify or the broker through `client`), on the run's calendar, up to `now=`,
  starting at the session before the run so its first session has a benchmark return.
  `Result.performance`, `metrics`, `metrics_table()` and `tearsheet()` then report
  alpha, beta, correlation, R squared, tracking error, information ratio, Treynor
  ratio, rolling beta, the horizon summary and the yearly comparison against it; the
  Assumptions block names it and where its closes came from. It never trades and never
  changes a fill. Missing benchmark data is a warning, never a stop: a source that
  cannot serve it leaves the run without benchmark figures, and the warnings name every
  session without a close. The manifest records it (symbol, exchange, source and the
  hashes of its closes), `rerun()` reproduces it and says when only its closes changed,
  and `bt.optimize` measures every combination against it. The default is no benchmark.
  Behaviour change in `metrics.analyse`: a missing benchmark session stays missing and
  is never filled from a neighbouring day. Before, a session without a benchmark close
  took the last earlier close (a zero return) and the next session took the whole move;
  now a session has a benchmark return only when it and the session before it both
  have a close of their own (for the first session, the close of the calendar session
  before it), so both of those sessions are NaN in `benchmark_returns` and the
  benchmark level is NaN on the missing session. Alpha, beta, correlation, R squared,
  tracking error, information ratio, Treynor ratio and rolling beta use only the
  sessions with both a strategy and a benchmark return (`benchmark_sessions`), and the
  horizon summary gains `benchmark_sessions`, shown in the tearsheet next to the
  benchmark return.
- **`openalgo.backtest.costs` (preview).** Indian trading charges for NSE and BSE
  equity (delivery and intraday), NFO and BFO futures and options, and MCX futures:
  brokerage presets (`zerodha`, `flat`, `percent`, `zero_brokerage`, `custom`), STT or
  CTT, exchange and IPFT charges, SEBI fee, stamp duty, GST, the depository charge and
  STT on exercised options. Rates are dated data (`schedules()` lists each version and
  whether it is verified), every trade is charged at the rates of its trade date, and
  anything charged at unverified rates is flagged and warned about. `rounding="calculator"`
  reproduces the broker calculator to the paisa for equity and F&O; the default
  `"contract_note"` rounds per trade day and segment. Part of the backtest engine
  being built; the interface may change before the backtest beta.
- **Split and bonus detection in `openalgo.backtest` (preview).** OpenAlgo history may
  not be adjusted for splits and bonuses, so the data quality check now reads them from
  the bars of NSE and BSE shares: an overnight gap that is within 5 percent of a usual
  factor (a 1:3 bonus up to a 10 for 1 split, or a consolidation), lies beyond the widest
  daily price band and still holds at the session's close is a suspected unadjusted
  split or bonus, an error that stops `load_bars` and `bt.backtest` with the date, the
  ratio, the prices and how to go on. Circuit-limit days and ordinary large moves are not
  reported; indices and derivatives are listed as not checked. `corporate_actions=`
  (on `load_bars` and `bt.backtest`) takes declarations made with
  `data.CorporateAction.split(ex_date, from_face_value, to_face_value)`, `.bonus(ex_date,
  bonus, held)` or `.none(ex_date)` for a genuine move: a declared split or bonus the
  bars are not adjusted for becomes a warning in the report and the result, one the
  bars are already adjusted for is noted, and a declared factor that does not match
  the gap is an error. Adjusting a holding across a corporate action is planned for a
  later phase: a backtest that holds a position or a GTT across a declared, unadjusted
  split or bonus raises NotImplementedError. `QualityReport.price_gaps` carries every
  reported gap with its evidence, and the run manifest records the declarations. Part
  of the backtest engine being built; the interface may change before the backtest
  beta.

### Security

- The feed no longer prints the first and last characters of the API key when
  `verbose` is 1 or 2.
- The example, playground, audit, benchmark and test scripts no longer carry API key
  literals. They read the key from the `OPENALGO_API_KEY` environment variable.

### Changed

- Backtest preview: adding to a losing position is refused unless the strategy sets
  `allow_averaging_down = True`; `state` as a class attribute raises `ConfigError`
  (use `self.state`); `after_risk_exit="allow_reentry"` raises `NotImplementedError`
  until the combined risk rules it governs are built; `roll=1` is the default of
  `bt.backtest`.
- **Backtest messages say what to do.** A part of the backtest engine that is not built
  yet raises `NotImplementedError` with "... is not supported in this version: ..." and
  what to do instead (for example "bt.live is not supported in this version: test the
  strategy with bt.backtest"). No exception, warning, quality report line or
  Assumptions text cites an internal planning reference (a requirement or decision id,
  a phase number) any more, and the source column of the default slippage table names
  its origin plainly. Counts read in the right number ("1 declared corporate action",
  "1 row has a price that is zero or negative"). A weekly or longer strategy interval
  is refused with how to read longer bars instead of a message about base intervals.
- **Sized orders rejected for margin are explained.** An order whose sizer asked for
  more than the margin allows is still rejected with reject code `margin` (the sizer is
  not cut down to the cash); the result now warns that the sizer asked for more than
  the margin allows and names `size_caps` (`bt.size.caps(leverage=1)`) as the bound.
  `result.orders` has a new `sizer` column naming the sizer that set each order's
  quantity.
- `ReferenceTables.expiry_schedule` takes `record_use=False` to list contracts without
  recording them as used or warning about them.
- **Backtest speed.** A run's time no longer grows with its bars times its orders (every
  slice walked every order and GTT of the run) nor with its bars times its months (the
  data hash); results are unchanged bit for bit. An event strategy makes about 550k
  `on_bar` calls a second (was about 120k) and an array strategy over 1M one-minute bars
  with charges, slippage and every metric runs in 1.1 to 1.5 seconds (was about 10.6).
  `Bar`, the argument of `on_bar`, is now an immutable named tuple that also carries
  `ts_open` and `ts_close` (UTC nanoseconds), and makes `time` and `close_time` when
  they are read. `benchmark/backtest_speed.py --check` checks the speed targets.
- `closeposition()` and `cancelallorder()` are documented as what they are:
  account-wide. The server squares off every open position, or cancels every open
  order, of the account that owns the API key; the `strategy` argument is a label
  and does not scope the call. Behaviour is unchanged, only the docstrings.
- `history()`, `intervals()` and `depth()` docstrings now match what the server
  accepts and returns: the exact set of intervals, the two shapes of a "no data"
  reply, and the REST depth layout (`bids` and `asks` lists).
- Sandbox order IDs are described correctly. They are plain numeric strings with no
  `SB-` prefix, so the `"mode": "analyze"` field is the only reliable marker of a
  simulated order.
- Source comments, docstrings, tests, benchmarks, audit notes and documentation describe
  this library in its own terms and name no third-party product: each indicator is
  described by its formula, seed and window. The benchmark scripts that compared against
  an outside library are removed, and private kernels and one internal module are renamed
  (`kama_full_window`, the `*_wilder_sum` kernels, `openalgo/indicators/classic.py`). The
  public API and every indicator output are unchanged.
- **Market tables say which figures are assumptions.** The rows of the default slippage
  table (`bt.slippage.defaults()`) have status `default` instead of `verified`: no
  exchange or broker publishes slippage, so they are OpenAlgo's starting assumptions,
  overridable with `slippage=`. As with the default futures margin, a run does not warn
  about them and the Assumptions block names the default table. The NSE and BSE intraday
  times from 2026-12-07 (entries until 14:50, square-off at 15:00, kept from the earlier
  row) are unverified until brokers publish their times for the 15:30 closing auction,
  so `Calendar.intraday_times` reports `unverified=True` for those sessions. Fills,
  charges and every other result are unchanged.
- **Indicator docs and audit notes match the code.** `docs/INDICATOR_CONVENTIONS.md`
  now states the conventions of WMA, FRAMA, the Fisher Transform, CMF, MFI, RVI, RWI,
  the Parabolic SAR and VWAP and how NaN input is treated; it no longer promises
  identical values across releases (a change is listed here) or that every function is
  O(n) (KAMA and `ta.correlation` recompute each window). `docs/PARAMETERS_REFERENCE.md`
  gives the real signatures of `ta.frama(high, low, period=26)` and
  `ta.fisher(high, low, length=9)`, and the RWI formula as computed: a simple mean of
  the true range, NaN for the first `period` bars. The audit notes under `audit/` name,
  for every verdict, the test or script that checks it and its tolerance, mark MACD,
  `validate_input` and Bollinger Bands as a whole as unchecked, and correct the EMA,
  TRIX, Mass Index and Relative Volatility Index rows.
- **Indicator checks that can fail.** `test/indicator_references.py` implements WMA,
  FRAMA, the Fisher Transform, CMO, CMF, MFI, RVI, VWAP and the Parabolic SAR from
  their published formulas with plain Python loops, and `test/test_indicators_fixes.py`
  compares each with `ta` on both backends in CI. The parity scripts in `benchmark/`
  compare against these references, or against pandas and NumPy expressions, wherever
  their old reference had come to call the same kernel it was checking.
- **Feed errors are never silent.** An exception raised inside an LTP, quote, depth
  or order callback, a frame that cannot be parsed, and a rejection from the server
  (for example an invalid API key) now reach `on_error` listeners, or an ERROR record
  on the `openalgo.feed` logger when none is registered, at every `verbose` level.
  Repeated failures of a tick callback are rate limited; order-update failures are not.
- **Subscriptions survive a reconnect as last requested.** A `subscribe_*` or
  `unsubscribe_*` call made while the socket is down is remembered and applied when
  the connection next authenticates (an unsubscribe during a reconnect used to be
  lost and the instrument came back). The return value is unchanged: it still says
  whether the frame was sent at that moment. The order-update stream follows the
  same rule.
- `disconnect()`, and a drop with `auto_reconnect=False`, forget the requested
  subscriptions, so a later connection starts clean.
- `connect()` on a feed that is already connected and authenticated returns True
  without opening a second socket.
- `disconnect()` called from inside a callback no longer stalls for two seconds.

- **Condition declarations fail loudly.** A truth test of a view inside `init` (`and`,
  `or`, `not`, `if`, or a chained comparison such as `a < b < c`) raises `ConfigError`
  saying to combine conditions with `&`, `|` and `~`; before, `a and b` silently became
  `a`. `bt.optimize` reports the path a sweep really ran (`"conditions"` or
  `"event"`) with a matching duration estimate, and sweeps strategies made by
  `bt.from_arrays` and `bt.from_signals` in this process instead of failing to pickle
  them.

#### Indicators: results that change on clean data

- **`ta.vwap` honours `anchor`.** In 2.0.5 the anchor was ignored and the VWAP ran
  from the first bar. It now restarts each anchor period (default `"Session"`) when
  bar times are known: a pandas Series whose index holds dates, or `timestamps=`.
  Plain arrays are unchanged. On daily, weekly or monthly bars every bar is its own
  session, so the VWAP equals the bar's typical price, and a warning says so. Pass
  `anchor=None` for the 2.0.5 result. An unknown anchor now raises `ValueError`, and
  zero volume no longer raises `RuntimeError` (the bar is NaN until volume arrives).
- **`ta.psar` and `SAR`**: on an outside bar that reverses the trend, the new stop
  starts at that bar's extreme and is clamped to the two previous bars (2.0.5 used the
  previous extreme point and did not clamp), and a bar that only touches the stop no
  longer reverses. Values change from such a bar to the next reversal; ordinary bars
  are unchanged. Each bar computes the new stop, tests the bar against it, then
  clamps it to the two previous bars.
- **`volatility.RVI` class** now computes the published Relative Volatility Index;
  2.0.5 took an RSI of the standard deviation. Nearly every value changes.
- **Invalid helper arguments**: a negative length in `change`, `rising` and
  `falling` raises `ValueError` (was `IndexError`); `highest` and `lowest` with period
  0, and `valuewhen` with n below 1, return all NaN.

#### Indicators: results that change only around NaN input

The 2.0.5 NaN contract now holds in every kernel these changes touch: a rolling
window is NaN while it holds a NaN, and a recursive indicator carries its state
across a NaN bar.

- `cmf`, `mfi`, `cmo`, `rvi`, `highest`, `lowest`: only NaN positions move.
- `fisher`, `frama`, `psar`, `vwap`: NaN while a NaN is in the window, then recover
  (2.0.5 stayed NaN to the end of the series).
- `dema`, `tema`: only the NaN bar is NaN. 2.0.5 `tema` dropped NaN bars before its
  third average, so values after a gap depended on later bars (a look-ahead).

- **WebSocket URL for servers hosted on a domain.** When `host` is an `https://` URL
  and neither `ws_url` nor `ws_port` is passed, the feed now connects to
  `wss://<host>/ws`, the address the OpenAlgo installers serve behind TLS, instead of
  `ws://<host>:8765`. Nothing changes for `http://` hosts or hosts without a scheme
  (`ws://<host>:<ws_port>`, port 8765 by default), for an explicit `ws_url`, or for
  an `https://` host with `ws_port` passed, which keeps `ws://<host>:<ws_port>`. If
  your https deployment publishes the plain WebSocket port, pass `ws_port=8765` or
  `ws_url="ws://<host>:8765"` to keep the old address. The `ws_port` parameter
  default is now `None` so the SDK can tell a chosen port from the default;
  `client.ws_port` still reads 8765 when none is passed. At `verbose=1` a failed
  connection names the URL it tried.

### Fixed

- A data quality error in own 1-minute data now stops `bt.backtest` with that error;
  before, the base-interval probe moved on to the strategy interval and the run stopped
  with a confusing `TimestampError` about the interval.
- `bt.backtest(data="bars.csv")` (a file path, a `pathlib.Path` or `bt.data.Files`)
  without `start=` and `end=` takes the run's dates from the file, as it does for a
  DataFrame.
- A futures run that ends within about 75 days of the reference tables' check date no
  longer warns about the contracts listed past its end that it never uses. It warns
  only when an expiry it relies on (one that sets a roll inside the run) is past the
  check date.
- The ATR error of a condition strategy and the kernel's errors about a bar name the
  bar's open in IST, not in UTC nanoseconds.
- `openalgo.backtest.reference.round_to_tick` (and `ReferenceTables.round_price`) took
  a real price within about 1e-12 of its value of a tick as on that tick, so on a
  0.0025 tick a price of several lakh rupees could round down to the tick above it. Only
  floating-point noise (64 units in the last place) is now treated as on a tick. The
  backtest kernel rounds on its own price units and was not affected.
- The WebSocket URL derived from an IPv6 literal host or from an upper-case scheme
  was invalid.
- `ta.highest` and `ta.lowest` with a period far larger than the series tried to
  allocate memory for the whole period and could abort the interpreter; they now
  return NaN.
- `ta.ulcerindex` returned NaN instead of 0 where rounding made the rolling mean of
  squared drawdowns a hair below zero.
- Without the compiled extension, `fisher`, `cmf`, `mfi`, `psar`, `cmo` and `frama`
  raised an error; they now compute with NumPy, and `stochastic` returns one value per
  bar on short input.
- Crossover helpers (`crossover`, `crossunder`, `cross`, `valuewhen` and the others)
  run in the Rust core: about 100 times faster on long series.
- **A backtest no longer runs past its data or the present.** `bt.backtest` ends a run
  at the earliest of `end`, the close of the last bar any instrument has, and the
  present (the new `now=` argument, the clock by default), with a warning that names
  the end used when it is before `end`. Equity, trades and metrics stop there, a
  position still open is reported open, and the contract notes of the last session are
  settled at the run's end as at a session close, so charges do not depend on where
  the run ends after its last fill; before, a run for today continued on empty
  slices to the session close and squared off an MIS position at a price from hours
  earlier.
- **Quote ticks carry the average price and last quantity the server sends.**
  `avg_trade_price` and `last_trade_quantity` were always 0 in quote callbacks and
  `get_quotes()` because the server sends them as `average_price` and `last_quantity`;
  quotes now also carry `ltt`, `total_buy_quantity`, `total_sell_quantity` and `oi`.
- The `serenity_index` definition cites the published serenity ratio without naming a
  company.
- With an opening wealth, the benchmark return of the horizon summary's "Since start"
  row and of the first period of `benchmark_comparison` was NaN when the benchmark had
  no value before the first equity point, while `benchmark_total_return` started at the
  first session end. All three now follow one rule, stated in the metric definitions:
  from the benchmark's first value in the span, NaN when the span has no benchmark
  return.
- The `broker_units` reference table records that a live OpenAlgo server's Zerodha
  master contract gives MCX `lotsize` in units (observed 2026-11-12); the broker-side
  conversion to contracts remains unconfirmed.
- `result.orders` marks an order without a `reject_code`, `tag` or `sizer` with NaN in
  that column in every run; before, it was None when no order of the run had one and
  NaN when another did (an exit after an entry with the default size).
- A CNC sell with `reverse=True` that meets a long reduced after it was placed (by
  another sale filling first) closes the long and rejects the rest with `product_rule`;
  before, the rest opened a delivery short.
- A condition entry (`entry_when` or `short_when`) whose `product=` differs from the
  run's product raises ConfigError before the run. Before, it filled into a position the
  conditions never read, so the strategy stayed flat and entered again at every bar its
  condition held, whatever `pyramiding` said.
- `self.target(percent=...)` while the sizing equity is below zero asks for a flat
  position; before, the negative equity flipped the side, so a long target became a
  short one.
- The report's rolling statistics name the session of an infinite benchmark return in
  their error; before, they named the return's place in the first window that held it.
- `costs.custom()` refuses a percent rule whose cap is NaN or negative infinity, as
  `costs.percent()` does; before, such a cap silently meant no cap.
- A user cost schedule that gives one segment twice under codes differing only in case
  or spaces (`"nse_eq_delivery"` and `"NSE_EQ_DELIVERY"`) raises CostError; before, one
  row silently replaced the other. A rate row's `notes`, `calculator_tax_base` or
  `calculator_tax_unit` given as None takes its default; before, it was stored as the
  text "None".
- `costs.trade_costs` and `CostModel.trades` refuse None, booleans and text among the
  prices and quantities, as a single trade does; before, a float cast took None as NaN,
  True as 1 and "1100" as 1100.
- `Result.rerun(params={...})` runs with the parameters given; before, it kept those of
  the original run.
- A truth test (`if`, `and`, `or`, `not`) of a futures strategy's series in `init`
  raises the ConfigError that says to combine conditions with `&`, `|` and `~`, as it
  does for a symbol's series; before, it read False.
- `instrument=` is checked before the run: each dict key must be its Instrument's
  symbol, an instrument of a symbol the strategy does not trade is refused, and a
  futures strategy refuses anything but futures of its underlyings on its exchange (a
  list, a value that is not an Instrument). Before, such entries were dropped and the
  run used the generic or reference-table terms instead.
- A futures run whose last given contract reaches its roll session (`roll=N`) inside
  the run says that the next contract is needed; before, it held the last contract
  past its roll.
- `bt.from_signals` refuses a function whose outputs change in number, or become None,
  once later bars are removed: they could not be checked for look-ahead. The docstring
  now says the check samples three cut points.
- An indicator that no cut of the truncation check is long enough for is reported as
  "truncation not checked" in the Assumptions block, not "passed at 0 cuts".
- `margin=` by underlying applies to a future given without `underlying=` whose symbol
  names it (NIFTY24NOV26FUT is a NIFTY future); before, it kept the default margin and a
  warning called NIFTY untraded.
- The refusal of `after_risk_exit="allow_reentry"` names the settings it concerns,
  `combined_stop` and `lock_profit`, and the `size_caps` line of the Assumptions block
  shows every digit (1000000, not 1e+06).
- `metrics.return_correlation` refuses a Series with a missing (NaN) return instead of
  counting it as a return of 0.
- With an installed extension older than a metric, the metrics raise the ImportError
  that says to reinstall openalgo, not an AttributeError.
- `bt.optimize` refuses, before running anything, an `objective` metric that is not a
  number (a flag, a time or text, such as `"ruined"`).
- A base of seconds (`base="30s"`) no longer stops an MIS run with ZeroDivisionError,
  and the Assumptions block calls its bars 30-second bars (it said 0-hour).
- A session time or an interval written with a digit that is not decimal (a superscript
  two, for example) raises ConfigError instead of ValueError.
- `benchmark/backtest_speed.py --rows` refuses an unknown row group, so `--check` cannot
  pass without measuring anything.
- When a reconnected socket opened, went live and dropped again before the reconnect
  thread looked at it, the feed stayed down for the 5-second connect timeout before the
  next attempt. A connection attempt now ends as soon as its socket is refused or
  closes, so the next attempt follows its backoff, and `connect()` to a server that
  refuses returns False at once instead of after the timeout.
- A depth callback that edited a level of its update changed what `get_depth()`
  returns. Data callbacks now receive their own copy of the update, nested levels
  included.
- An error reported while the last `on_error` hook was being removed reached neither a
  hook nor the `openalgo.feed` logger. It is now logged.
- An order callback received the very dict kept in the order history, and
  `get_orders()` returned those dicts, so editing one changed what later
  `get_orders()` calls returned. Both now hand out copies.

## [2.0.5] - 2026-09-11

### Fixed

#### NaN handling in the indicator kernels ([#2029])

Chaining one indicator into another silently produced nothing. Every indicator
emits warm-up NaNs, and the moving-average kernels let a single NaN into their
running accumulator, which stayed poisoned for the rest of the series:

```python
df["RSI_wma"] = ta.rsi(ta.wma(df["close"], 55), 14)
df["RSI_avg"] = ta.wma(df["RSI_wma"], 9)             # was: NaN on every row
ta.crossover(df["RSI_wma"], df["RSI_avg"]).sum()     # was: 0 across 1600 bars
```

`ta.crossover` / `ta.crossunder` were never at fault and are unchanged. Affected
kernels: `sma`, `wma`, `ema`, `hma`, `zlema`, `t3`, `vwma`, `stdev`,
`rolling_sum`, `rolling_variance`, `ema_sma`, `ema_wilder`, `ema_first_valid`,
`true_range`, `atr_wilder` and the rolling extremes behind `stochastic`,
`williams_r`, `aroon`, `donchian`, `midprice` and `midpoint`.

Three failed silently, returning plausible numbers instead of NaN:

- **`stdev` returned `0.0`** for the rest of the series after a NaN rather than
  NaN, because `f64::max` returns the *other* operand when one side is NaN and
  `(NaN).max(0.0).sqrt()` is `0.0`. Bollinger width collapsed to zero instead of
  failing visibly.
- **`rsi` returned a flat `100`** across an upstream indicator's warm-up. A NaN
  delta is neither `> 0` nor `< 0`, so `avg_gain` and `avg_loss` both seeded to 0
  and the `avg_loss == 0` branch published 100. This one predates the Rust core,
  which is why pinning back to 1.0.50 did not help.
- **`median` panicked** through the PyO3 boundary (`partial_cmp().unwrap()`) on
  any window containing a NaN - a `PanicException`, which is a `BaseException`
  and so escapes a normal `except Exception`.

`ta.vi` and `ta.ulcerindex` returned all-NaN even on clean OHLCV; both now match
their textbook definitions.

### Changed

#### The NaN contract is now stated and enforced

Previously unstated, and the two backends had silently drifted apart, so the same
script gave different answers from a wheel and from a source checkout:

- **Rolling-window kernels are window-local.** `out[i]` is NaN iff `i < period-1`
  or the window `[i-period+1 .. i]` contains a NaN; the series recovers as soon as
  the NaN slides out. This is the same rule as pandas `.rolling(period)`.
- **Recursive kernels (the EMA family) skip leading NaNs**, seed at the first
  finite value, and carry their state across an interior NaN.

The Rust core and the pure-NumPy fallback are now checked against each other, and
against pandas, in `benchmark/ci_smoke.py` on every CI run. The Rust core carries
13 new unit tests for the contract.

**For NaN-free input every other indicator is unchanged, bit for bit** - verified
across 155 output series against 2.0.4. Performance is at or better than 2.0.4 for
the hot kernels on 2M rows (`sma` 3.36 -> 2.43 ms, `wma` 3.38 -> 2.62 ms,
`stdev` 3.53 -> 2.96 ms, `atr` 9.30 -> 8.87 ms); `vwma` is the one regression,
2.31 -> 3.20 ms.

[#2029]: https://github.com/marketcalls/openalgo/issues/2029

## [2.0.4] - 2026-09-08

### New Features

#### GTT (Good Till Triggered) Orders
- **`placegttorder(...)`**: Place a SINGLE or OCO price trigger that sits with
  the broker until LTP crosses the level, then places the underlying order.
  SINGLE takes exactly one of `triggerprice_sl` / `triggerprice_tg`; OCO takes
  all four of `triggerprice_sl`, `stoploss`, `triggerprice_tg` and `target`,
  with the stoploss trigger below the target trigger.
- **`modifygttorder(...)`**: Replace an active trigger's spec. Modify is a full
  replacement, not a patch, so every field to keep must be sent again.
- **`cancelgttorder(trigger_id=...)`**: Cancel an active trigger. Cancelling an
  OCO removes both legs atomically.
- **`gttorderbook()`**: List active triggers. Triggered, cancelled, expired and
  rejected GTTs are filtered out at the broker layer.
- Trigger specs are validated locally before a request leaves the machine: a
  SINGLE with no trigger or with both, an OCO missing a field or with its
  stoploss trigger at or above its target trigger, and an unknown
  `trigger_type` all return `{'status': 'error', 'error_type':
  'validation_error'}` rather than reaching the broker.
- GTT numeric fields are sent as JSON numbers, not strings, matching what the
  server validates. GTT accepts CNC and NRML only; MIS is refused server-side.

#### Strategy Module API (API-key surface)
Nine methods against `/api/v1/strategy/`, for the multi-leg options strategy
engine with end-to-end risk management:
- **`strategylist(status=None, q=None)`**: Strategies this key owns, newest first.
- **`strategystatus(strategy_id=...)`**: One strategy's full config including
  legs, plus its current run or `null`.
- **`strategystart(strategy_id=..., mode=...)`**: Start a batch strategy. `mode`
  is a required keyword argument with no default, in the SDK as on the server:
  omitting it is a `TypeError`, never a live order. The value is passed through
  unnormalised, so a near miss such as `"LIVE"` is refused rather than quietly
  read as sandbox.
- **`strategystop(strategy_id=...)`**: Exit every owned position at market. Read
  `stop_pending` and the per-leg outcomes; an accepted stop is not proof of
  flatness.
- **`strategycloseall(strategy_id=...)`**: Same stop mechanics, plus a
  `close_all_manual` audit event proving an operator asked for the flatten.
- **`strategycloseleg(strategy_id=..., leg_id=...)`**: Exit one leg; the run
  continues with the rest.
- **`strategyruns(strategy_id=..., limit=None)`**: Run history, newest first.
- **`strategyorders(strategy_id=..., run_id=None)`**: Order history, oldest
  first, so an entry always precedes its exit.
- **`strategyevents(strategy_id=..., run_id=None, kind=None, severity=None, limit=None)`**:
  The append-only risk-event audit trail, newest first.

#### Strategy Webhook Client Revamped
`Strategy` now speaks the strategy module's webhook protocol:
- **Batch**: `start(mode)` and `stop()`. `mode` is a required positional
  argument with no default.
- **Signal**: `long_entry()`, `long_exit()`, `short_entry()`, `short_exit()`,
  each naming the leg by `leg_id` or by `symbol` plus `exchange`, and a generic
  `signal(action, ...)` plus a raw `send(payload)`.
- **`webhook_token` constructor keyword** alongside the original `webhook_id`,
  which still works. A missing token is a `ValueError` at construction.
- **The token is never rendered.** `repr()` redacts it, because the token is the
  whole credential and anywhere it is written down is a second copy of it.
- **Every documented outcome is returned, not raised.** A 200
  `rejected_dedupe`, a 409 `rejected_cooling_off` and a 403
  `rejected_live_disabled` all come back as dicts carrying their `result` label
  and the HTTP `code`, because that label is the contract. Only a transport
  failure or a non-JSON answer produces a locally-built error dict.

### Breaking Changes

- **`Strategy.strategyorder(symbol, action, position_size)` was removed.** The
  webhook it posted to no longer exists. It now raises `NotImplementedError`
  naming the replacements rather than failing at the network. There is no
  automatic mapping: `BUY` is `long_entry` on a flat leg and `short_exit` on a
  short one, and guessing which would be guessing at an order. Use
  `start(mode)` / `stop()` for a batch strategy, or `long_entry()` /
  `long_exit()` / `short_entry()` / `short_exit()` for a signal strategy.

### Internal

- **`BaseAPI._post(endpoint, payload)`**: one shared pooled-client request
  helper for the new surfaces. Unlike the older per-mixin `_make_request`, a
  non-200 answer with a JSON body is returned as-is with `code` attached
  instead of being flattened into an `HTTP 409: {...}` string - the strategy
  and GTT surfaces put the actionable detail inside that body (a 409 stop
  carries `stop_pending` and the per-leg outcomes, a 400 carries a `message`
  object keyed by field name). Existing methods are untouched.

## [2.0.3] - 2026-07-15

### New Features

#### WebSocket Feed API - Real-Time Order Updates
- **`subscribe_orders(on_order_update=None)`**: Account-level real-time order
  status stream over the existing OpenAlgo WebSocket connection — fills,
  partial fills, rejections and cancellations pushed by the broker (live
  mode) or the sandbox engine (analyze mode), for orders from any origin
  (API, broker app/web, engine square-offs).
- **`unsubscribe_orders()`**: Stop the order-update stream.
- **`get_orders()`**: Cached order updates received so far (most recent
  first, capped at 500).
- **`[ORDER]` log category**: order updates printed at `verbose=2`.
- Order-update subscription survives reconnects — replayed automatically
  after re-authentication, like market-data subscriptions.
- Message fields use OpenAlgo common order constants (`symbol` in OpenAlgo
  format, `action`, `pricetype`, `product`, lowercase `order_status`,
  `rejection_reason` with the broker's full RMS/OMS text).
- Requires an OpenAlgo server with order-update streaming (the
  `subscribe_orders` WebSocket action).

## [1.0.40] - 2025-11-25

### New Features

#### WebSocket Feed API - Verbose Control
- **`verbose` parameter**: Added verbosity control for WebSocket feed operations (LTP, Quote, Depth)
  - `verbose=False` (default): Silent mode - no SDK output, only errors
  - `verbose=True` or `1`: Basic info - connection, auth, subscription status
  - `verbose=2`: Full debug - all market data updates from SDK
- **Cleaner output format**: Aligned log categories `[WS]`, `[AUTH]`, `[SUB]`, `[LTP]`, `[QUOTE]`, `[DEPTH]`, `[ERROR]`
- **Trader-friendly logging**: Concise, aligned output for easy troubleshooting

### Usage Example
```python
from openalgo import api

# Silent mode (default) - no SDK output
client = api(api_key="...", host="...", ws_url="...", verbose=False)

# Basic logging - connection/subscription info
client = api(api_key="...", host="...", ws_url="...", verbose=True)

# Full debug - all data updates
client = api(api_key="...", host="...", ws_url="...", verbose=2)
```

### Output Format (verbose=True)
```
[WS]    Connected to ws://127.0.0.1:8765
[AUTH]  Success | Broker: upstox | User: rajandran
[SUB]   NSE_INDEX:NIFTY | Mode: Quote | Status: success
```

---

## [1.0.36] - 2025-01-20

### ✨ New Features

#### Data API - instruments() Function Enhancements
- **Download ALL Exchanges**: `instruments()` now supports downloading all exchanges when no parameter is specified
  - `client.instruments()` - Downloads **ALL exchanges** and combines into single DataFrame
  - Downloads from: NSE, BSE, NFO, BFO, MCX, CDS, BCD, NSE_INDEX, BSE_INDEX
  - Returns **148,000+ total instruments** across all exchanges
  - Gracefully handles exchanges that fail or return no data
  - Backward compatible - `exchange` parameter still works for specific exchanges

### 🐛 Bug Fixes

#### Data API - instruments() Function
- **Fixed HTTP 400 Error**: Removed `Content-Type: application/json` header from GET request
  - GET requests don't have a request body, so this header was causing server rejection
  - Server was returning: `"The browser (or proxy) sent a request that this server could not understand"`

### 📚 Documentation Updates
- Updated `instruments()` docstring with enhanced usage examples
- Added documentation for ALL exchanges download feature
- Updated examples in documentation

### ✅ Verified Functionality
- **ALL Exchanges**: Successfully retrieves 148,208 instruments from 8 exchanges
  - NFO: 88,140 instruments (PE, CE, FUT)
  - MCX: 26,375 instruments
  - BSE: 12,698 instruments (EQ)
  - CDS: 11,019 instruments
  - BFO: 6,754 instruments
  - NSE: 3,046 instruments (EQ)
  - NSE_INDEX: 120 instruments
  - BSE_INDEX: 56 instruments
- **Specific Exchange**: NSE, NFO, and other exchanges work correctly
- Returns clean pandas DataFrame with all instrument details
- CSV export functionality working correctly

### 💡 Usage Examples
```python
# Download ALL exchanges (new feature!)
all_instruments = client.instruments()
print(f"Total instruments: {len(all_instruments)}")  # 148,208+

# Filter by exchange from combined data
nse_only = all_instruments[all_instruments['exchange'] == 'NSE']
nfo_options = all_instruments[
    (all_instruments['exchange'] == 'NFO') &
    (all_instruments['instrumenttype'].isin(['CE', 'PE']))
]

# Download specific exchange (still supported)
nse_df = client.instruments(exchange="NSE")
nfo_df = client.instruments(exchange="NFO")

# Filter and analyze
equities = nse_df[nse_df['instrumenttype'] == 'EQ']
df.to_csv('nse_instruments.csv', index=False)
```

---

## [1.0.35] - 2025-01-20

### ✨ New Features

#### Data API Enhancements
- **`instruments()` function**: Download all trading symbols and instruments with exchange-wise filtering
  - Returns data as pandas DataFrame for easy analysis and manipulation
  - Supports filtering by exchange (NSE, BSE, NFO, BFO, BCD, CDS, MCX, NSE_INDEX, BSE_INDEX)
  - Returns comprehensive instrument details: symbol, name, token, lot size, tick size, instrument type, etc.
  - Enables quick symbol lookup, filtering, and export capabilities

- **`syntheticfuture()` function**: Calculate synthetic futures price using ATM options
  - Implements synthetic future formula: `Strike Price + Call Premium - Put Premium`
  - Automatically determines ATM strike from available options
  - Useful for arbitrage opportunities and pricing verification
  - Supports indices (NIFTY, BANKNIFTY) and equity stocks
  - Returns underlying LTP, ATM strike, and calculated synthetic future price

### 🔄 Deprecations

#### Options API Parameter Changes
- **`strike_int` parameter**: Made optional and marked for deprecation in `optionsorder()` and `optionsymbol()`
  - Now optional with default value `None`
  - Deprecation warning issued when parameter is used
  - Will be removed in future versions

- **`strategy` parameter**: Made optional and marked for deprecation in `optionsymbol()`
  - Changed default from `"Python"` to `None`
  - Deprecation warning issued when parameter is used
  - Will be removed in future versions

### 📚 Usage Examples

```python
from openalgo import api

client = api(api_key="your_key", host="http://127.0.0.1:5000")

# Download all NSE instruments
nse_df = client.instruments(exchange="NSE")
print(f"Total NSE instruments: {len(nse_df)}")

# Filter for equities only
equities = nse_df[nse_df['instrumenttype'] == 'EQ']

# Calculate synthetic future price
synthetic = client.syntheticfuture(
    underlying="NIFTY",
    exchange="NSE_INDEX",
    expiry_date="28NOV25"
)
print(f"Synthetic Future: {synthetic['synthetic_future_price']}")
print(f"Spot Price: {synthetic['underlying_ltp']}")
print(f"Basis: {synthetic['synthetic_future_price'] - synthetic['underlying_ltp']}")
```

### 🛠️ Technical Improvements
- Added proper deprecation warnings using Python's `warnings` module
- Backward compatible - existing code continues to work with warnings
- Enhanced documentation for new functions
- Improved error handling for GET requests in Data API

---

## [1.0.25] - 2025-01-14

### 🚀 MAJOR PERFORMANCE OPTIMIZATIONS & 100% SUCCESS RATE

This release delivers **massive performance improvements** and achieves **100% indicator functionality** through comprehensive optimization.

### ✅ Enhanced
- **🎯 PERFECT SUCCESS: 100% Indicator Coverage** - All 103 technical indicators now working flawlessly
- **🚀 Major Performance Improvements**:
  - **Ichimoku**: 4600% faster execution (1.4s → 0.03s)
  - **ZLEMA**: 54% better scaling ratio (104× → 48×)  
  - **MODE**: Optimized binning algorithm implementation
  - **NATR**: Vectorized operations for linear performance
- **⚡ O(n) Algorithm Implementation**: Linear complexity for critical indicators
- **🛠️ Consolidated Utilities**: Unified EMA, ATR, SMA, STDEV kernels across all modules
- **🔧 Numba JIT Optimization**: Consistent compilation with caching for maximum performance

### 🔧 Fixed
- **DEMA/TEMA**: Resolved missing `_calculate_ema` method with consolidated utility integration
- **RVI**: Fixed parameter signature to properly handle OHLC data (open, high, low, close, period)
- **ckstop**: Resolved Numba compilation issue by replacing class method calls
- **true_range**: Corrected parameter count (high, low, close)
- **roc_oscillator**: Fixed to single parameter interface

### 🏗️ Technical Improvements
- **Consistent API**: Perfect `from openalgo import ta` usage pattern
- **Memory Optimization**: Linear scaling maintained across all dataset sizes
- **Production Ready**: Sub-millisecond performance for typical trading datasets
- **Code Quality**: Eliminated redundant implementations and improved maintainability

### 📊 Performance Metrics
- **Average Execution**: 4.5ms per indicator (10K dataset)
- **Best Performance**: 0.022ms for fastest indicators
- **System Capacity**: 20,000+ indicators/second for medium datasets
- **Scaling**: Linear O(n) behavior for all optimized indicators

## [1.0.24] - 2025-01-14

### 🎉 Major Technical Indicators Enhancement

This release brings **complete technical analysis capabilities** to the OpenAlgo Python Library with **100% functional technical indicators**.

### ✅ Added
- **Complete Technical Indicators Library**: All 102 technical analysis functions now working perfectly
- **High-Performance Implementation**: NumPy and Numba optimization for fast calculations
- **Function-call Syntax**: Easy-to-use `ta.function()` interface
- **Comprehensive Coverage**:
  - **19 Trend Indicators**: SMA, EMA, Supertrend, Ichimoku, HMA, etc.
  - **9 Momentum Indicators**: RSI, MACD, Stochastic, CCI, Williams %R, etc.
  - **18 Volatility Indicators**: ATR, Bollinger Bands, Keltner Channels, etc.
  - **13 Volume Indicators**: OBV, VWAP, MFI, ADL, CMF, etc.
  - **20 Oscillators**: ROC, TRIX, Awesome Oscillator, PPO, etc.
  - **8 Statistical Indicators**: Correlation, Beta, Linear Regression, etc.
  - **11 Hybrid Indicators**: ADX, Aroon, Pivot Points, SAR, etc.
  - **5 Utility Functions**: Crossover/Crossunder detection, Highest/Lowest, etc.

### 🔧 Fixed
- **Parameter Signature Issues**: Fixed 4 indicators with incorrect parameter counts
  - `alligator()`: Fixed parameter signature to use single data input
  - `gator_oscillator()`: Corrected parameter count and removed unnecessary shift parameters
  - `fractals()`: Removed incorrect period parameter
  - `zigzag()`: Added missing close parameter
- **Numba Compilation Issues**: Resolved 5 indicators with self-reference compilation errors
  - `vidya()`: Inlined CMO calculation to remove self-reference
  - `rvol()`: Fixed RVI class confusion and parameter signature
  - `chandelier_exit()`: Inlined ATR calculation to remove self-reference
  - `stochrsi()`: Inlined RSI calculation to remove self-reference
  - `chop()`: Inlined ATR sum calculation to remove self-reference
- **RWI Implementation**: Fixed undefined class reference error

### 📚 Documentation Updates
- **FUNCTION_ABBREVIATIONS_LIST.md**: Updated with all 102 correct function names and abbreviations
- **FINAL_INDICATOR_VALIDATION_REPORT.md**: Complete validation report showing 100% success rate
- **Comprehensive Testing**: All indicators validated with generated test data

### 🎯 Technical Details
- **Input Flexibility**: All indicators accept numpy arrays, pandas Series, or Python lists
- **Output Consistency**: Returns same format as input (numpy/pandas preservation)
- **Error Handling**: Robust validation for periods, data length, and parameter ranges
- **Performance Optimized**: Numba JIT compilation for mathematical operations
- **Memory Efficient**: Optimized array operations and memory usage

### 🚀 Usage Examples
```python
from openalgo import ta
import numpy as np

# Sample price data
close = np.array([100, 101, 99, 102, 98, 105, 107, 103, 106, 108])
high = close * 1.02
low = close * 0.98
volume = np.random.randint(1000, 5000, len(close))

# Trend indicators
sma_20 = ta.sma(close, 20)
ema_50 = ta.ema(close, 50)
supertrend, direction = ta.supertrend(high, low, close, 10, 3)

# Momentum indicators  
rsi = ta.rsi(close, 14)
macd_line, signal_line, histogram = ta.macd(close, 12, 26, 9)

# Volatility indicators
atr = ta.atr(high, low, close, 14)
upper, middle, lower = ta.bbands(close, 20, 2)

# Volume indicators
obv = ta.obv(close, volume)
vwap = ta.vwap(high, low, close, volume)

# Oscillators
stoch_k, stoch_d = ta.stochastic(high, low, close, 14, 3)
williams_r = ta.williams_r(high, low, close, 14)

# Utility functions
cross_above = ta.crossover(close, sma_20)
cross_below = ta.crossunder(close, sma_20)
```

### 🏆 Quality Metrics
- **Success Rate**: 100% (102/102 indicators working)
- **Test Coverage**: Comprehensive validation with synthetic and real market data
- **Performance**: Optimized for high-frequency trading applications
- **Reliability**: Production-ready with extensive error handling

---

## [1.0.23] - 2024-XX-XX
### Previous Release
- Core trading API functionality
- WebSocket market data feeds
- Order management system
- Account operations