# Changelog

All notable changes to the OpenAlgo Python Library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Groundwork for running several strategies on one account (2.4.0, kernel unit M1; `bt.backtest`
  does not take a list of strategies yet, and nothing changes for a single strategy). The kernel
  runs each time slice as separate phases, so a portfolio run can step several strategies' engines
  together; every existing backtest, result and live session gives the same figures at the same
  speed. New codes a portfolio run will report: the order rejection `account_margin` (the
  strategy's own free cash covers the order's margin, the shared account's does not), the entry
  rejection `account_loss_gate` (entries after the account's daily loss limit is breached, until
  the next session), the exit reason `account_loss_limit` and the risk event `account_loss_hit`
  (scope `account`). Result warnings explain the new rejection codes and what to do.
- Custom tearsheets (2.4.0, unit C1). `result.tearsheet(layout=...)` and
  `report.tearsheet(..., layout=...)` take a `report.Layout`, which can hide or show built-in
  sections and parts (`report.PARTS` lists them), add sections of your own with
  `report.section` (charts and tables drawn in the tearsheet's look: `line`, `bar`,
  `histogram`, `scatter`, `heatmap`, `table`, `tiles`, `text`), add blocks to the end of a
  built-in section (name your section like it), place and order sections (`after=`,
  `order=`) and set the page's light and dark colours (`report.Theme`). A block's data is a
  pandas object or a function of the result, so one layout serves every run. `report.html`
  and `report.svg` embed your own HTML or SVG, checked against an allow list (no script, no
  style sheet, nothing fetched, links only inside the page, images only inline as PNG, JPEG
  or SVG data), so the page stays one offline file; anything off the list raises a
  `ConfigError` naming it and its block. The built-in tearsheet stays the default and is the
  same byte for byte: no layout, `layout=None` and `Layout(base=mode)` give today's page,
  pinned by twelve golden pages in the tests.
- The fund: several strategies stepped together on one account in a backtest (2.4.0, kernel
  unit M2; `bt.backtest` takes a list of strategies from a later unit, and nothing changes for a
  single strategy). Each strategy keeps its own engine, instrument and bar interval, and the fund
  advances them in order on one merged time line, across exchanges and intervals (an NSE minute
  strategy beside an MCX one or a daily one). With a shared pool, an opening order also needs its
  margin within the account's free cash, and a sizer sees at most the account's free cash; with
  separate pools each strategy keeps its own capital and the account is only recorded. The
  account's daily loss limit counts each strategy's loss of the round's session day only, and
  fires once a day: every strategy's entries are locked, its working entries cancelled and its
  positions exited (exit reason `account_loss_limit`). The account's figures each round (equity,
  free cash, margin, daily loss), its breaches and the orders it refused or cancelled are kept for
  the portfolio result. A strategy in a fund gives exactly the result it gives alone whenever the
  account does not bind. Speed rows `portfolio-3x1M`, `fund round` and `portfolio-callbacks` in
  `benchmark/backtest_speed.py`.
- Live engine groundwork for two strategies on one symbol (2.4.0, kernel unit K5; the session
  uses it from a later unit). On an instrument several strategies of a session trade, an
  entry, add or reversal on the other side of the side the others hold or are entering is
  rejected `group_other_side` (a broker nets the two into one position), and a strategy's
  resting order (LIMIT, SL, SL-M) is rejected `shared_resting`; the leg's broker-held stop,
  its targets and the engine's own exits are not affected. A closing order the broker
  rejects in such a group holds the position: no market exit, no re-placed stop and no
  square-off sent by the engine until the session has checked the broker's books. Both
  codes have a meaning and what to do in the rejection warnings.
- Two or more strategies on one symbol in a live session (2.4.0, units M6 and M7). `bt.live` used to
  refuse a second strategy on a symbol and product ("run one strategy per symbol"); now each
  strategy keeps its own position and its own stop at the broker while OpenAlgo holds one net
  position for the symbol. The symbol is subscribed and checked against broker candles once,
  and every tick, quote and candle reaches each of its strategies. On such a symbol:
  - An entry, add or reversal that would put a strategy on the other side of another
    strategy's position or pending entry is rejected `group_other_side`; a strategy's own
    LIMIT, SL and SL-M orders are rejected `shared_resting` (its stop and targets still work);
    while another strategy's reversal works, its entries are locked (`group_reversing`).
  - The reconciler, the restart, the exit guard and the kill switch judge the strategies
    together: the sum of their positions against OpenAlgo's net. A trade by hand that the
    trade book explains is shared out among the strategies it closes, the most recently opened
    position first, and each strategy's stop is cut or cancelled to what it still holds. Any
    other difference locks every strategy's entries and holds their exits until the books
    agree, with one alert that names the group.
  - A stop or exit the broker rejects holds the position: no market exit, no stop placed
    again and no exit retry. The strategy's entries stay locked for the session
    (`margin_on_close`), OpenAlgo's books are read at once, and one ERROR alert says to end
    the session, check the position book and close by hand. The session's own exits (Ctrl+C,
    the kill switch, the square-off) go only after a book read agrees.
  - The start banner names each shared symbol and its strategies and says never to trade it
    by hand while the session runs. A new unprotected reason, "instrument held by another
    strategy's order", comes with an ERROR alert naming that order. An order the order book
    cannot tell apart locks the entries of every strategy on its symbol.
  - The reports show the group. `SessionResult.open_positions` gains a row per shared symbol
    whose OpenAlgo net is a position, named by its strategies ("A and B"), with that net and
    whether it equals their sum, and the DAILY_SUMMARY a "Shared" line per shared symbol
    (`SessionSummary.shared`). In a session with a shared symbol,
    `SessionResult.reconciliations` gains the columns `group`, `group_units`,
    `member_units`, `external_units`, `allocation` and `stops_cut`. A difference of the group
    counts as one reconcile mismatch and is named once at the close.
  - Units beyond every strategy's position, or a difference on the other side of the
    strategies, are taken in by none: nothing is sent, every strategy on the symbol is
    locked, and the alert says to end the session and close by hand. On the other side every
    strategy's stop is cancelled, with one ERROR alert for the symbol.
  - A stop's cancel whose fills the trade book has not yet confirmed is decided from the
    position book and then the order book when the order book counts every other strategy's
    fills (sandbox books do). A live broker's order book shows a resting stop without a fill
    count, so there the cancel waits for the trade book, reading both books every 2 seconds.
  A strategy alone on its symbol keeps 2.3.0's behaviour throughout, its reports included.
- `bt.backtest` takes a list of strategies and runs them together on one account (2.4.0, unit
  M3), returning a `bt.PortfolioResult`. Each strategy keeps its own `Result`
  (`pf.strategy(name)`), measured on its own capital. The portfolio result gives the account's
  figures and tables: `account` (equity, free cash, margin and daily loss each round), equity,
  returns, performance and metrics, every strategy's tables in one with a `strategy` column,
  `trade_list()`, `refused` (the orders the account refused), `net_positions`, `correlation`,
  `contribution`, assumptions, warnings, the manifest, and CSV and Excel export.
  - Per-strategy settings (`capital`, `params`, `product`, `base`, `max_daily_loss`,
    `after_risk_exit`, `max_positions`, `drawdown_halt`, `cooldown`, `entry_from`,
    `entry_until`, `square_off`) take one value for every strategy or a dict by strategy class
    name. A `params` dict may leave a strategy out (it keeps its class defaults); the other
    dicts may not. Every other argument is the run's: `data=` as a dict by symbol feeds every
    strategy trading that symbol, and `instrument=` and `corporate_actions=` reach only the
    strategies they apply to.
  - New arguments, used only with a list: `pool` ("shared" by default, "separate" for fixed
    allocations), `account_capital` (the account's funds, by default the sum of the capitals;
    with that default a shared account refuses an order only when other strategies' open
    losses leave too little free cash) and `account_max_daily_loss` (at or past it every
    strategy exits with reason `account_loss_limit`, and later entries that day are refused
    `account_loss_gate`).
  - Two strategies may trade one symbol, on either side. A warning names an instrument held
    on opposite sides (`bt.live` would refuse the later entry), and the days on which a
    broker would net one strategy's delivery trade against another's in the same scrip,
    which changes the charges.
  - A list of one strategy also returns a `PortfolioResult`. A strategy class on its own runs
    exactly as before, with the same figures.
- Combined reports of a portfolio run (2.4.0, unit M4).
  - `PortfolioResult.summary()`: the account's net profit, return, maximum drawdown, closed
    trades and benchmark, as `Result.summary()` gives them for one strategy, then the quick
    overfitting checks of each strategy's own trades on a line each. None are run on the
    account's trades together, which overlap in time.
  - `PortfolioResult.tearsheet(path=None, *, mode="full", log_scale=False, layout=None)`: the
    account's page, with the Strategies section (the strategies head to head and the
    correlation of their returns), the overfitting checks in a row per strategy, a trade
    list with a Strategy column, and a new Portfolio section (full mode): each strategy's
    contribution and its P&L by month, stacked; the account's highest margin and lowest
    free cash in each session and each strategy's margin at the account's peak; the orders
    the account refused, with times; and the net position of each instrument two strategies
    held, at every change. `report.tearsheet(pf)` gives the same page. Each strategy's own
    page is still `pf.strategy(name).tearsheet()`.
    On this page the hover title of each MAE and MFE scatter mark names its strategy
    ("LongDaily trade 1: MAE ..."), since each strategy numbers its own trades.
  - Custom tearsheets: `report.PARTS` gains the "portfolio" section with the parts
    `contribution`, `account_margin`, `refused` and `net_positions`, which a layout hides,
    shows and orders like any other part. On a portfolio a layout's block functions receive
    the `PortfolioResult`, and a Theme's series colours colour the strategies.
- A live session's strategies as one account, and every per-strategy live setting by name
  (2.4.0, unit M5).
  - `SessionResult.portfolio`: a `PortfolioResult` of the session's strategies, holding the
    same `results`, so it covers this start only. It has the account on the sum of the
    strategies' capitals (its equity as `metrics.combine_equity` adds theirs, the margin the
    engines block, free cash and this start's daily loss measure), the metrics, `summary()`,
    the combined `tearsheet()`, every strategy's tables in one with a `strategy` column, and
    `net_positions` for an instrument several strategies share. Its Assumptions name the
    funds OpenAlgo reported at the start and say the margin and free cash are the engines'
    model, not the broker's. `refused` lists the entries the account's trade cap refused
    (`account_trade_limit`); entries locked out by the account's loss limit or profit target
    are in each strategy's orders as `live_lock`.
  - `bt.live`: `after_risk_exit`, `max_positions`, `drawdown_halt`, `cooldown`, `entry_from`,
    `entry_until` and `square_off` each take one value for every strategy, as before, or a
    dict naming each strategy by class name, as `bt.backtest` takes them for a list of
    strategies. A dict that names a strategy the session does not run, or leaves one out, is
    refused before anything is sent. A class attribute of the same name still wins. Every
    existing `bt.live` call works unchanged.
  - `PortfolioResult` takes an optional keyword `notes`: Assumptions lines a run words
    itself, in place of the block's own lines of the same key.
- Daily limits per strategy and for the account, in `bt.backtest` and `bt.live` (2.4.0, unit
  M10). Each is off unless set, and takes one value or a dict by strategy class name; a class
  attribute of the same name wins.
  - `max_trades_per_day`: the entries of a session day. An entry is an order that opens a
    position from flat, or a reversal; adds, exits and the engine's own exits never count.
    An entry counts from submission while it works, and for good once any part fills. At the
    cap an entry is rejected `trade_limit`, and a reversal is cut to its close (the close is
    sent as an order that only reduces the units held, and the opening part is recorded
    rejected). Nothing is exited.
  - `daily_profit_target`: when the day's profit (the measure `max_daily_loss` uses) reaches
    it, every position exits with the reason `daily_profit_target` and entries are rejected
    `daily_profit_gate` until the next session day, whatever `after_risk_exit` says.
  - `max_order_value`: an entry or add worth more is rejected `order_value_limit`. Its worth
    is the units times the limit price, an SL-M's trigger or the last close, times the
    multiplier, judged at submission; a MARKET entry with no last close yet is refused. A
    modify is checked too and, when refused, leaves the order as it was. Exits are never
    refused.
  - For a list of strategies in `bt.backtest`, and in `bt.live`: `account_max_trades_per_day`
    (the entries of every strategy together; at the cap entries are rejected
    `account_trade_limit` and nothing is exited) and `account_daily_profit_target` (works as
    `account_max_daily_loss` does: exit reason `account_profit_target`, entries rejected
    `account_profit_gate` in a backtest, the kill switch fired for `account_profit` live).
    `pf.refused` lists the account's refusals; a strategy's own cap codes are in its
    `result.orders` and in `pf.orders`.
  - Each limit is named in the Assumptions block, in the live start banner and in the daily
    summary's Limits line. Refusals by these limits are told once in `result.warnings` and
    raise no OrdersNotPlacedWarning. A live restart of the day keeps the trade counts
    (the account's include the entries of strategies an earlier start ran) and the profit
    locks.
- `bt.live(max_orders_per_second=8)`: at most this many order calls (place, modify and cancel
  together) in any second, OpenAlgo's limit of 10 less 2 for your own orders (2.4.0, unit
  M10). A call beyond the limit waits, stops and exits first. A call that waits 2 s writes the
  event `order_throttled` and sends one alert per session start. An entry or add that waits
  5 s is not sent (reason `order_rate`); an exit is never dropped. An entry still waiting
  when its strategy is locked (a profit target, the kill switch, a safe stop) is withdrawn,
  not placed.
- Live sessions write a decisions file per day beside the event stream,
  `<live_dir>/<account>/logs/YYYY-MM-DD.decisions.jsonl` (`SessionResult.paths["decisions"]`):
  one line per strategy bar, off the console and the event stream. A decision line gives the
  bar as the strategy saw it and whether it came from ticks or a broker candle, the latest
  value of each `self.I` output, the position in units, and what `on_bar` submitted (each
  order with its side, units, type, status and reject code). A skipped line says why no
  decision was made: missing data, indicators not ready, trading not begun, the session
  stopping, or the strategy failed. A line that cannot be written is logged and dropped.
  - Each `drift` event carries the tick bar's and the broker candle's open, high, low, close
    and volume beside their differences.
  - An order the engine rejects at submission is logged at INFO with the strategy, side,
    units and reject code, and so is the first decision per strategy skipped because its
    indicators are not ready (later ones at DEBUG).
- An audit record of every live order (2.4.0, unit M11; design 18 section 12).
  - Every order carries its strategy's version id: 12 hexadecimal characters of the SHA-256
    of its source and parameter values, the same across days, restarts and processes. Each
    start writes every strategy's run manifest and version id to the journal.
    `SessionResult.versions` maps each strategy name to its manifest, `SessionResult.orders`
    has a `version_id` column, the start banner has a Versions line, and the daily summary's
    Strategies line names each version. A restart whose version differs from the day's
    earlier start is allowed: the banner says so and the event `version_changed` records
    both ids. The `strategy` field sent to OpenAlgo is still the plain class name.
  - The live journal is schema version 2: a `versions` table and `version_id` on every
    intent. A 2.3.0 (version 1) journal is upgraded in place when opened, its rows kept; a
    newer file is still refused.
  - Live preflight checks each futures and options contract's expiry in OpenAlgo's master
    contract against the reference tables. A disagreement refuses the start, naming both
    dates; a missing expiry is a banner warning.
  - Live preflight compares OpenAlgo's market holidays (one `holidays` call, in every
    session) with the package calendar for the session's exchanges, from today to the
    year's end. A disagreement about today refuses the start; one about a later day is a
    banner warning and the event `calendar_disagrees`.
  - A backtest with a client checks each futures or options contract the server still
    lists against the master contract's expiry, with a ReferenceDataWarning on a
    disagreement or when the lookup fails. Expired contracts, clients without `symbol` and
    cash equity make no call.
  - `SessionResult.fill_comparison`: each fill of the day (every start) against the price
    a backtest of the session's own bars fills the same order at. It names the strategy,
    version, purpose (target and hand included), price type, live and backtest time and
    price, the difference per unit against the trader, rupees, and whether it was compared
    and why not. A MARKET order is set against the open of the bar that opens at its
    decision, an engine target exit against its level, and a stop against its trigger (or the open of a
    bar that gapped beyond it) in the bar that reached it. LIMIT orders, repriced stops and
    fills adopted by hand are not compared. `summary.stop_slippage` totals the stop rows,
    and the daily summary has a Stop fills line on a day a stop filled; a stop fill not
    compared is counted there, with its reason in its `fill_comparison` row.
- MCX silver mini (SILVERM) and silver micro (SILVERMIC) can be backtested and traded: their
  freeze quantity, 600 kg per order, is in the contract table from MCX circulars
  MCX/TRD/114/2025, MCX/TRD/091/2026, MCX/TRD/275/2025 and MCX/TRD/302/2026 (contracts
  outside those circulars carry the same terms as unverified, with a warning). Before, both
  were refused for want of a freeze quantity.

### Changed

- `Result.strategy()`, `Result.combined` and `Result.correlation` still raise
  NotImplementedError on a single strategy's result; the message now points to `bt.backtest`
  with a list of strategies.
- `bt.optimize`, `bt.robustness` and `bt.walk_forward` given a list of strategies raise
  NotImplementedError: run them for each strategy. Given one strategy, they and `bt.backtest`
  refuse `pool`, `account_capital` and `account_max_daily_loss` with a `ConfigError`, and
  `bt.live` refuses `pool` and `account_capital` as it refuses the other backtest-only
  arguments.
- Live sessions send at most 8 order calls a second by default (it was 5), set by
  `max_orders_per_second`.
- Live sessions save their state in layout 5 (the day's trade counts), so a restart cannot
  resume a state that 2.3.0 saved earlier the same day: start it fresh.
- `Fund.advance` called after a portfolio run has ended raises the fund's "has finished"
  error, and each of the fund's breach records carries its rule (`account_loss` or
  `account_profit`).
- Live sessions: the strategy decides, and keeps deciding, on the bars built from ticks. A
  broker candle read after a decision (2 seconds after the close, often before the broker's
  final candle) no longer replaces the tick bar in the strategy's series or its indicators,
  so a crossing between the tick bar and a later, different candle of the same minute is
  seen by exactly one decision, never by none and never twice. On the 2026-11-20 sandbox
  day a reversal on a 1-minute EMA crossing was seen by no decision this way. Candles are
  still checked against every tick bar for the drift log. A candle is the bar only of a
  minute with no ticks (the feed was dead, or watching began in it after a start or
  restart): it refreshes the indicators and no decision is made on it afterwards. The
  engine and the strategy now hold the same bars. Within one run only: after a restart the
  minutes before it come from broker candles, as before.
- Live sessions: a trade closed by the account's daily loss limit (`account_max_daily_loss`,
  through the kill switch) now records the exit reason `account_loss_limit` instead of
  `manual`, the reason a portfolio backtest records for the same breach.
- `Strategy.exit(reason=...)` refuses `"leg_sl"` and `"leg_target"` with a `ConfigError`:
  they are the reasons of the engine's own exits at a leg's stop and target, and a live
  session timed a strategy exit that named them as the engine's tick exit.
- A restart that finds a strategy's saved state is of another symbol or contract (the same
  class name moved to a new contract) still refuses it before sending anything, but now says
  what to do: give the class a new name for the new contract, or keep its symbol. Every
  strategy so moved is named at once.
- Tearsheets: a figure that rounds to zero no longer shows a minus sign ("-0.0%" now reads
  "0.0%", in the weekly cells and titles, the yearly axis and the charges table), a yearly
  returns axis that steps by half a percent labels its ticks with one decimal, a histogram
  no longer shows an empty bin of zero width at its low end, the noise check's tile gives the
  count of noise runs above the run as well as the share, and the optimizer page no longer
  prints the sweep's measured times, so two runs of one sweep give the same page.
- The robustness check of the parameter neighbourhood names the range a `params=` value lies
  outside and what to pass, where it said only that it could not judge.
- `bt.optimize` uses less memory on large sweeps: the returns of each combination are copied
  into the overfitting figures' array as they come back, not after all of them.
- Live sessions no longer send a resting reversal (`reverse=True` on a LIMIT, SL or SL-M
  order): the session rejects it with `venue` before it reaches the broker, the strategy sees
  it rejected in `on_order` before its next decision, and a warning says to submit the
  reversal as a MARKET order or to exit and enter separately. At the broker it worked beside
  the position's stop, both on the closing side, and both could fill: a short of 10 with a
  LIMIT buy of 20 resting ended long 20, not long 10. A MARKET reversal is sent as before,
  and backtests are unchanged.
  The reason a live result gives for the `venue` code now names both causes (the broker
  refused the order, or the session refused a resting reversal) with the advice for each.
- Live sessions: the day summary's slippage sets the engine's target exits against the
  target's level with the run's slippage, as a backtest fills them, instead of the next
  bar's open. This corrects the live figure (2.4.0, unit M11). It still equals the sum of
  the compared MARKET rows of `SessionResult.fill_comparison`.
- Live sessions: EXIT alerts and the event of each closing request name the exit reason,
  for example "(exit: target level, order N)", "(square_off: square-off time, ...)" or
  "(risk_exit: kill switch at the account daily loss limit, ...)" (2.4.0, unit M11). The
  other alert texts are unchanged.
- Live sessions: the combined page of `SessionResult.portfolio` no longer says "The
  account refused no order and ended none." after the account's kill switch fired. When
  the session set an account limit, an empty Refused card says what the account's trade
  cap refused, that the entries its daily loss limit or profit target locked out are in
  the strategies' orders as live_lock, and that the entries and adds its kill switch
  cancelled are in their orders as cancelled (2.4.0, unit M11). The account note names the
  kill switch's cancels. A backtest portfolio's page is unchanged.

### Fixed

- Live sessions: a reversal (`reverse=True`) waits for its position's stop to be cancelled
  before it is sent; if entries were locked meanwhile (a dead feed, a reconciliation
  mismatch, a failed exit, the account loss limit), it was still sent and opened the other
  side. It is now refused whole with `live_lock`, as it is when it is submitted under a lock,
  and the position gets its stop back. A reversal no longer raises a false "had no stop at
  the broker" alert between the stop's cancel and the order's fill.
- Live sessions: a decision is skipped for missing data when the bar before its own holds a
  minute with no data, as it already was when its own bar held one. After a start or restart
  in the middle of a minute, that minute's bar comes from its broker candle; when the candle
  came after the first whole bar closed, a 1-minute strategy's first decision read a series
  without the bar before its own, and a crossing (or `shift(1)`, `[-2]`) compared it with the
  bar two minutes back. On the 2026-11-20 sandbox day the first decision after a restart sold
  on an EMA crossing that the full series shows one minute earlier, before that decision. The
  strategy now decides at the next close, with the minute's bar in its series once its
  candle is read. A minute whose candle never comes holds up one more decision, no more.
- Live sessions: an entry or add held in the order worker's queue by
  `max_orders_per_second` is now withdrawn once its strategy's entries lock for any reason (a
  dead feed, a reconciliation mismatch, failed exits, a closing order refused on a shared
  instrument), and the strategy sees it rejected with `live_lock`. Before, only the daily
  profit target, the kill switch and the safe stop withdrew it, so an add could be bought up
  to 5 s after such a lock.
- Live sessions: a position left with no stop at the broker and nothing to close it (for
  example after one of two targets that fired together was refused) gets its stop placed
  again and an alert even when the strategy rests a reversal, such as a LIMIT buy below the
  market against a short. Only a MARKET reversal, which closes the position at once, now
  counts as covering it; before, a resting reversal hid the missing stop.
- Live sessions: an entry or add the order worker had already taken off its queue when its
  strategy's entries locked is no longer sent. The worker reads the strategy's locks just
  before it sends an entry or add, and the strategy sees it rejected with `live_lock`; exits
  and stops are never held by it. Before, such an order could still go out a few
  milliseconds after the lock.
- Live sessions: on an instrument held for the day (an order the order book cannot tell
  apart from another, so nothing more can be sent for the instrument), the square-off no
  longer reports a stop that still works at the broker as being cancelled, nor its position
  as unprotected from 15:00. The stop is reported as working, and one ERROR alert per
  strategy at its square-off says that nothing is sent for the position and what to do:
  cancel its stop at the broker, then close the position by hand in OpenAlgo, the broker's
  own MIS square-off remaining the backstop. The note of the position left open says the
  same.

## [2.3.0] - 2026-11-20

### Added

- Phase 2 (live trading) groundwork, no change to backtests: the kernel can hold before
  the slices not released yet (`Engine::hold`, `Engine::release`, `Pause::Waiting`), the
  base of the live clock. A backtest released one slice at a time gives the same result
  as one run straight through.
- A live session in the kernel (`Engine::new_session`, `Engine::add_bar`, `BaseBar`): built
  before its bars from the calendar's grid of its day, given each instrument's base bar
  as its slice closes (checked: on the grid, for a slice that has not run, in order, a
  valid OHLC bar), its strategy bars rebuilt by the backtest's resampler, and ending at
  the end of its day. A session given its bars minute by minute gives the same result as
  the backtest of those bars.
- A live session's venue in the kernel (no change to backtests): a session fills and
  expires no order itself. Each order that passes the backtest's submission checks is
  handed to the live layer at once as a command (place, modify, cancel:
  `Engine::take_commands`), and the broker's results come back: `Engine::apply_fill` with
  the units newly filled at the broker's price (no slippage model; the cost model's
  charges, an order filled in parts charged once on its total; recorded all or nothing),
  `apply_reject`, `apply_cancel`, and the broker's answers `apply_cancel_refused`,
  `apply_modify` and `apply_modify_refused`. A cancelled order works on until the broker's
  result, and a cancellation the broker refuses is sent again in a later slice, at most
  once in each; a modified one keeps its old terms until the broker confirms the new
  ones, and one cut to the units it has filled meanwhile completes. Every fill of an
  order comes before its terminal result.
- Cancel, then exit: an exit (`exit()`, `reduce()`, the daily loss limit, the drawdown
  halt, the square-off, the time exit) first cancels every other working order of the
  position on the closing side and is sent once they have ended. The strategy's exits of
  a position add up as a backtest fills them: each is sized when it is sent, on the
  position less what the earlier ones still have to fill, so two `reduce(0.5)` close
  three quarters, `reduce()` of 50 then of 10 units close 60, and `reduce()` then
  `exit()` close it all. The engine's exits take over from the strategy's (their exits
  are cancelled first) and close the whole position. Every exit goes out above the
  freeze quantity as several orders of at most it. The engine's exits are sent where a
  backtest makes them (the square-off and time exit as the session reaches their open,
  with or without a bar), and again at the next open while they apply (a rejected exit,
  a position a cancel race reopened). MIS entries still working when entries stop are
  cancelled. Timers fire as the session reaches their open. At a session close the DAY
  orders are cancelled through the broker and the session does not finish while an
  order works; a position still open is reported open (no closing fill is invented).
- Reduce-only orders never reverse a position in a session: the reduce-only orders
  resting on its closing side may not hold more units than it has (new reject code
  `exit_cover`: one stop for the whole position rests, a stop and a target for it both
  do not), they are cut down or cancelled when the position shrinks, and a broker's fill
  beyond the position of an order that may not reverse it (reduce-only, or without
  `reverse=True`) is recorded up to the position and refused for the rest
  (`VenueRefusalError` of kind `would_reverse`).
- Refused at submission in a session, with the backtest's codes and errors, what a fill
  could not record: a leverage the margin table does not allow, a short option, a margin
  above the free cash net of what the session's working entries would block (`margin`),
  a fill the cost model cannot price. Not supported in a
  session yet (NotImplementedError): `stop=`, `target=`, trailing and breakeven on CNC
  and NRML (MIS legs are below), GTT orders, futures series and `rebalance()`; fractional
  quantities are refused. A broker's
  own rejection has the new reject code `venue`; a result the kernel refuses raises
  `openalgo.backtest.errors.VenueRefusalError` with its `kind` (`VENUE_REFUSAL_KINDS`,
  the same list the native module exports). A
  session whose orders are filled where the simulator fills them gives the same result as
  the backtest, field for field. The native `Engine` gains `session`, `add_bar`,
  `release`, `venue`, `take_commands`, `modify` and the `apply_*` calls.
- A live session's controls in the kernel (no change to backtests). Every order sent
  carries its role (`OrderRole`: entry, add, the strategy's exit, the engine's risk exit
  with its reason, the square-off; `Engine::order_role`), and the native "place" command
  gains `role` and `reason` (`ORDER_ROLES`, `RISK_EXIT_REASONS`). `request_exit` exits a
  whole position for the session's own reason (an operator stop, the kill switch, the
  account's loss limit, an unprotected leg, a stop rejected or unfilled, or the
  square-off) through the position's exit owner, cancelling the orders on its closing
  side first; `cancel` is taken while the session waits and sends a cancellation again
  after the broker refused it. `lock_entries` and `unlock_entries` keep entry locks as
  reasons in a set: while any is set, entries and adds are rejected with the new reject
  code `live_lock`, and exits are not. A session also rejects an entry or add while
  another order of the symbol and product works (new reject code `live_in_flight`) and one
  that would take the position above the freeze quantity (`freeze_qty`), so a session's
  position fits in one order. The orders working on a position's closing side never hold
  more units than it: a plain sale or buy against the position is an exit only within
  what the others working to close it leave, and beyond that an entry, which those gates
  judge (so two plain sales of a long whose first fill is reported late, or a sale during
  the session's own exit, can no longer open a short); a reduce-only order beyond it is
  rejected `exit_cover`, counting plain closing orders too; the fill of an order sent to
  close is never recorded beyond the position, even from a flat one; and a modification
  that would change an order's role is refused (cancel it and place a new one).
  `request_exit` returns what it did (`sent`, `waits_for_cancels`, `already_exiting`,
  `flat`, `market_closed`; native `EXIT_REQUESTS`), and an exit asked for when the market
  has closed sends nothing. A lock reason with spaces around it is refused.
  `cap_sizing_equity` makes the engine default size from the lesser of the equity and the
  account's available cash; `daily_loss_measure` reads the strategy's daily loss measure
  for the account's limit; `stop_session` ends a session where it stands, sending nothing
  and reporting its working orders and open positions as they are (a strategy exit that
  waited for its cancels, never sent, ends canceled), with the day's contract note
  settled, so `finish` can be called.
- A leg's stop, targets, trail and breakeven in a live session (MIS; backtests change only
  for option stops, below). The broker holds the stop (decision D31): the fill that opens
  a leg with a stop level places it for the units filled, SL-M for cash equity and
  futures, and for NFO and BFO options an SL with its limit 5 percent beyond the trigger;
  each further fill or add modifies its units at once, and a trail or breakeven moves its
  trigger once the move is at least 0.25 percent of the trigger the broker last confirmed,
  at most once every 5 seconds and one modification at a time (the newest terms go once
  the answer has come and 5 seconds have passed). The kernel executes nothing when the
  price reaches the stop: the stop's own
  fill closes the leg (`leg_sl_hit`, reason `leg_sl`), and a stop's fill beyond the
  position records the units that closed it and refuses the rest (`would_reverse`; the
  session design asked for such a fill to be refused whole, and the closing part is kept
  instead, so the session's books hold every unit the broker closed). The stop never
  works beside more than the position: a resting reduce-only order beside it is rejected
  `exit_cover` and a plain resting sale beside it is an entry, as beside any order
  closing the position, and a stop placed later (a trail or breakeven arming, `protect`,
  a re-placed stop) cancels the resting orders sent only to close the position and covers
  their units once they have ended. A partial exit (`reduce()`, a partial target) cuts the
  stop down to what stays open first and is sent once the broker confirms that (a refusal
  leaves it waiting, the cut sent again 5 seconds later); a full exit cancels the stop
  first, and so does a partial exit followed by exits that close the rest (`reduce()`
  then `exit()`), which then go one after the other; until that cancellation goes
  through (a refused one is sent again) the stop covers the whole position. A plain MARKET order against the
  position (`sell()` of the long, a reversal, a `target()` that flips) goes the same way,
  the stop cut down or cancelled first, so it and the stop never both fill at the
  broker; a reversal whose stop fills before its cancellation ends canceled and opens
  nothing, and the strategy decides again at its next bar. A stop the broker rejects while its leg is open exits the
  leg at market (`stop_rejected`); one the broker cancels of its own accord, or that could
  not be placed (reported with `apply_cancel`), is placed again once, and a re-placed stop
  that ends without a fill exits at market (`stop_rejected`, or `unprotected`); a stop that
  filled while its leg holds more (an add not covered yet, or a cut the broker applied
  before it filled) exits the rest (`stop_unfilled`). A stop a closed leg left working
  (its cancellation refused) keeps new entries of the instrument out (`live_in_flight`).
  The kernel holds the targets, the trail and the breakeven on the session's ticks:
  `push_tick(instrument, price, at)` ratchets the legs, exits a target it reaches through
  the position's exit owner (once per leg: an exit the broker rejects is not made again
  on the next tick, the session retries it), marks the position and tests the daily loss
  limit and the drawdown halt on that mark; a session no longer ratchets on its bars. New
  session calls: `protect` places a leg's stop again (native `PROTECT_OUTCOMES`, with
  `waits_for_cancels` when resting orders sent to close are cancelled for its room),
  `reprice_exit` modifies a stop the price has passed (or the engine's exit) to a
  marketable LIMIT for the order's total units, and `request_exit(..., limit_rupees=)`
  exits with LIMIT orders. A session on recorded ticks whose broker fills where the
  simulator does, and applies a confirmed stop modification only from the next bar, trades
  as the backtest of its bars (the same fills, trades, equity, position history and
  charges) as long as no daily loss limit or drawdown halt is breached by a tick inside a
  bar where no bar close breaches it, and the strategy rests no order to close a leg that
  has a stop, a trail or a breakeven (use `target=` instead; a session rejects or cancels
  such an order for the stop's room, where a backtest keeps it): a real broker fills a
  moved stop within the bar that moved it, and the session tests its loss limits on every
  tick.
- The live layer's mode guard (private, not used by `bt.live` yet): OpenAlgo's sandbox or
  live mode read at the start of a session, checked on every order answer and update of
  the session's own orders, and polled every 5 seconds and before each entry. A change of
  mode, or OpenAlgo's semi-auto mode, locks the session (nothing more is sent) and sends
  one alert listing the orders sent and the orders in doubt after the change. New errors
  in `openalgo.backtest.errors`: `LiveError`, `PreflightError`, `ModeChangedError`,
  `JournalError`.
- The live layer's journal (private): a SQLite file per account, bound to its broker,
  with one record for sandbox and one for live. Each order intent is written and synced
  before the order is sent, so a restart knows every order whose outcome is unknown; it
  also keeps each order's updates, the broker-held stop of each position, the risk
  state of each strategy and trading day, and the strategy's own `self.state` across
  days. The first final update of an order stands, except that a fill with a price
  replaces one reported at price 0.
- The live layer's bars (private): 1-minute bars built from OpenAlgo's WebSocket ticks on
  the same calendar slices a live session runs, stamped at their close and closed by the
  clock one second after it. A tick counts as a trade only when the day's volume rose or
  the last traded price changed (every print for an index), so a minute without trades
  has no bar, as in the broker's history. After a mid-session start or a feed outage the
  partial minutes only set the starting level. Broker candles are compared with the
  bars minute by minute, and feed health tells a dead feed from a quiet symbol.
- The live layer's order path (private): each order is written to the journal before it
  is sent, sent once and never retried, at most 5 orders a second. Fills come from
  OpenAlgo's order-updates stream, with polling only as a fallback, and are priced and
  confirmed from the tradebook where a status answer cannot be trusted; a fill is never
  taken at price 0. An order whose outcome is unknown (a timeout, a dropped connection,
  a gateway error) is resolved against the order book after the server's broker window
  before anything more is sent for that instrument, or for any instrument while the
  server is silent. An order queued for approval in OpenAlgo's Action Center stays
  queued across a restart. fivepaisa is not supported for live trading until its
  tradebook order IDs match its orders'. The SDK's `api` gains no method in this change.
- `api.ping()`: checks the API key and returns the broker OpenAlgo is logged in to
  (`{"status": "success", "data": {"broker": ..., "message": "pong"}}`).
- The live layer's preflight (private): the checks before a live session. One session per
  account and per OpenAlgo server; the broker login proven by ping, funds and one quote
  (the quote reaches the broker in sandbox mode too); a supported broker; funds against
  capital (available cash); each symbol's lot size, tick size and freeze quantity; the
  interval and 1-minute candles offered; MIS only; the risk limits; the trading day and
  the MIS entry window; the computer's clock against a remote server's; and a delivered
  test alert in LIVE mode. Every failed check is reported together, with what to fix.
- The live layer's alerts, daily summary and session logs (private): alerts through
  OpenAlgo's Telegram endpoint, each starting with the session's mode (SANDBOX or LIVE)
  and strategies. Sending never blocks or stops a session: one background sender posts
  at most 20 alerts a minute (25 for the most urgent) under the server's limit of 30,
  most urgent first, merges routine alerts that pile up into one message, retries a
  rate limit, timeout or server error for up to 5 minutes, and counts every alert not
  delivered. The same alert is held back for a minute only for conditions that repeat
  (feed dead, reconciliation mismatch, errors). The daily summary reads like
  `Result.summary()`; the session's log file and JSON-lines event stream redact any
  field named like a key or token.
- The live layer's tick adapter and tick recorder (private; `bt.record` is wired with
  `bt.live`): each WebSocket message becomes a tick, with the exchange's last trade time
  when the broker's feed gives one in a readable unit (otherwise the receive time is
  used, and the tick is counted). A recording writes its ticks to disk at least once a
  second in checksummed blocks, so a crash or power loss costs at most the last second
  and never leaves a wrong tick; each finished recording is saved once per instrument
  and never rewritten, and a day is read by merging its recordings in the exact order
  they were received. Feed outages and each recording's start and stop are kept, so a
  replay can tell a dead feed from a gap in the recording.
- The live layer's order path, preflight and journal gain what the session needs (private):
  a check of one order's status, a probe of a silent server, polls of the orders a
  stream outage may have missed, the release of a held cancel at the fills found; the
  preflight checks every strategy's interval, sends its test alert once, and waits for
  the broker login only for the failures a login can fix (a refused API key, no funds or
  no quote), failing at once when OpenAlgo cannot be reached; the journal's wait for a
  lock is set by `busy_timeout_s`.
- The live session's building blocks (private): its clients, each with its own timeout;
  an inbox and a feed link that take ticks and order updates from the WebSocket without
  ever blocking it and tell a feed outage from a quiet symbol; a strategy's live series,
  whose bars, views and indicators equal a backtest's at every decision; and a data
  worker that reads quotes and broker candles on its own thread under the server's rate
  limits.
- The live session's plan (private): `bt.live`'s arguments are checked and each strategy
  is set up the way a backtest sets it up for the same day, with OpenAlgo's lot size,
  tick size and freeze quantity in place of the reference tables'; the session then
  catches up from the open without deciding or sending anything. `max_daily_loss` must
  be given in the `bt.live` call: a limit set only on the strategy class is not taken as
  the live limit, and a class limit that differs from the call's stops the start.
- The live session's order worker (private): the kernel's orders turned into OpenAlgo
  requests and each answer into the one kernel call it calls for (the venue map); every
  fill of the session's own orders reaches the kernel once, as the units newly filled
  and their price, however many sources report it (the stream, a poll, the tradebook);
  and one worker thread that sends the orders by priority (stops first), settles
  unanswered or refused modifications and cancellations by reading the order, and keeps
  the journal's record of each leg's stop.
- The live session itself, first version (private; the public `bt.live` comes with the
  session's result): the start sequence (mode, locks, logs, preflight, the order worker,
  warm-up and catch-up, the countdown), the loop that closes bars by the clock, drives
  each strategy's kernel session and hands its orders to the worker, the square-off by
  the clock, the end of the day, and the common close. Ctrl+C is the safe stop: entries
  locked and working entries cancelled, then with `on_stop="flatten"` (the default)
  every open position exited and waited for, or with `on_stop="keep"` the positions left
  open; a second Ctrl+C waits briefly for what is in flight and lists what is left open.
  A journal failure stops the session at once; a strategy error stops it the safe way.
  A strategy's leg rules (stop, targets, trailing, breakeven) run in the session through
  the stops supervisor, below.
- The live session's stops supervisor (private): every leg with `stop=` has its stop held
  by the broker, placed when the entry fills (SL-M for cash equity and futures, a
  stop-limit 5 percent beyond the trigger for NFO and BFO options), confirmed only when
  the broker answers that it works, and moved by modifying it for a trail or a breakeven
  (at most once per 5 seconds and 0.25 percent). Targets and the daily loss limit are held
  by the engine on ticks and quotes. An exit cancels or cuts the stop first, so a stop and
  an exit never both fill. A time without a confirmed stop is counted as unprotected, and
  a stop accepted but not confirmed within 10 seconds exits the leg (never while the order
  worker is busy or its answers wait to be read). An option stop-limit the price runs through is repriced to a
  marketable limit every 2 seconds, up to 5 times, then exited. A rejected exit is asked
  for again after 1, 2, 5 and 10 seconds; after the fifth rejection entries are locked,
  the stop is placed again and the alert says to end the session before any close by
  hand. A refused cancel is sent again; held fills and held cancels are decided from the
  broker's books. While OpenAlgo is silent and a stop or an exit waits, it is probed every
  2 seconds, with an alert after 10 seconds. The safe stop and every close protect each
  leg by its state first, and the lines printed as a session ends name each open leg's
  stop and orders and what to do by hand. Entries without a stop are allowed, with a
  warning in the start banner and the result.
- The live session's risk watch and kill switch (private): a feed with no tick on any
  instrument for 10 seconds is dead, with one alert, entries locked, the strategies'
  prices read by quote every 2 seconds and the slices it covered marked missing (no bar
  is made from part of a slice); entries open again at the close of the first whole
  slice after ticks return. A quiet instrument with an open position is priced by quote
  without a lock. A refused quote read alerts that the broker's login may have expired.
  The clock skew between this computer and the exchange is estimated from the trades'
  exchange times in the first minute and alerted beyond 2 seconds; a broker whose ticks
  carry no exchange time is named in the start banner or the result's warnings. The
  account's daily loss limit, across every strategy of the session, fires the kill
  switch: entries locked for the day, working entries cancelled, then each position
  exited only when OpenAlgo's position book holds at least the strategy's units on the
  same side (otherwise nothing is sent and the disagreement is alerted), and each
  strategy reported against the broker's book once flat.
- A live session's state saved and restored in the kernel (no change to backtests):
  `Engine::live_state` writes the session's state as plain data that survives a JSON round
  trip (in Python a dict of None, bool, int, float, str, list and dict, written by
  `json.dumps` and read back equal): where it stands, the cash, the positions with their
  tranches, the legs, every working order with its role, request, units, filled units and
  what its fills were charged, the cancellations and modifications waiting for the
  broker's answer, the exits and the legs' broker-held stops, the targets fired, the
  daily loss totals, the trading state, the cooldown and time exit counters, the entry
  locks and the commands not yet taken. Each saved order says whether it was sent: a
  strategy exit that waits for its cancels was not, and the restart's order check skips
  it. `Engine::restore_live_state` gives it to a session built for a restart, once, after
  its catch-up and before any order; the saved working orders are numbered from o1 again
  and the call returns each one's saved and new id. Later orders take the ids after them,
  which the earlier start may have given to orders that have since ended, so the live
  layer rebuilds its map of broker orders from the returned pairs alone. A session restored
  at a point of its trading day goes on as the uninterrupted one in its commands, states,
  fills, trades and P&L, with these limits: the events recorded after the last pause and
  not yet delivered are not saved, so the restarted strategy's `on_fill`, `on_order` and
  `on_position` never run for them; the entry locks are not restored (the session keeps
  its own, and the live layer sets again, from the saved state, those that last the day);
  a sizing cap the restarted session has set already is kept, being the fresher figure;
  and a state saved while the session closed is refused, since no session resumes at its
  close. Each start's result covers that start: the contract notes are settled per start,
  and a CNC sale after a restart is not matched with a same-day CNC buy before it for the
  intraday rates. A state of another day, other instruments or another capital, of
  another layout, one that cannot be read, or one whose parts do not hold together (a
  position its tranches or its leg do not match, an open position without its leg, an
  order filled beyond its units, a leg's stop that is not one; the error names the field)
  is refused whole, and so is a restore into a session that has traded, was restored
  already, or stands before the saved point. Through the binding, a state nested deeper
  than any saved one is refused before it is read.
- A live session hands out no command of an order that has ended: `Engine::take_commands`
  drops the Place, Modify or Cancel of an order that was rejected, cancelled or filled
  whole since the command was made, such as a restored order the restart's order check
  ends before its commands are taken.
- A live session's fill with no order of its own in the kernel (no change to backtests):
  `Engine::apply_external_fill` records the broker closing or reducing a position by
  itself (its square-off, a position closed by hand) through the one fill path, at the
  broker's price, with the exit reason given and the fill flag `external`; the orders
  working to close the position follow it (a leg's stop is cut down, or cancelled once the
  position is flat). A fill that would not only close or reduce the position (on its
  side, of a flat position, through zero) is refused whole: `VenueRefusalError` of the
  new kind `beyond_position`.
- The live session's reconciler and restart (private). OpenAlgo's position, order and trade
  books and the funds are read when trading begins, every 30 seconds, 3 seconds after a
  fill, 3 seconds after a comparison that found a new difference, around each square-off,
  before the safe stop's flatten and at the close, and compared leg by leg with the engine
  once the leg has no order in flight (a leg in flight for more than 3 minutes is compared
  anyway, with an alert and no action). A fill of the session's own orders that the stream
  and the polls missed is taken from the trade book. A difference seen in two comparisons
  in a row locks the strategy's entries, sends an alert, and allows only a safe
  correction: a position closed or reduced by other orders whose trades come to the
  difference exactly (a sale by hand, the broker's square-off) is adopted, with its stop
  cancelled or cut to what is left; when the broker holds the other side the leg's own
  stop is cancelled and the alert says to close it by hand; a broker holding fewer units
  that no trade explains, or more units, is left alone and alerted (an ERROR alert to
  check the account once it lasts 3 minutes). Nothing is ever traded to make the broker
  match the engine, and no difference exits a position or fires the kill switch. From the
  first comparison showing the broker with fewer units, none or the other side, no exit,
  flatten, square-off or stop of that leg is sent for more than the broker holds on that
  side. Orders of the session working at the broker that the engine does not hold, and
  working orders naming the strategy that the session did not send, are reported. The
  funds read caps each strategy's sizing by the available cash. After the session's last
  lines the close sends nothing those lines did not name; a difference only its final read
  sees is reported at OpenAlgo's units in the result and the daily summary, with one ERROR
  alert, never as flat, and a leg no read could judge (its own fill landed inside the
  read) is reported as not confirmed, with an alert to check OpenAlgo's position book. A
  restart the same day first resolves the journal's unsent or unanswered orders against
  the order book, then reads the books: a broker position on a traded symbol that the
  journal does not hold, or a journal position the broker does not hold and no trade
  explains, refuses the start with nothing sent. Each strategy's saved state is restored
  after its catch-up, every restored order is checked against the journal and the books,
  the entry locks that last the day are set again, and before the banner every open leg
  has its stop confirmed or placed again (or is exited when past its square-off).
- **Live trading: `bt.live`, `bt.record` and `bt.SessionResult` are public.**
  `bt.live(strategy, client, *, capital, max_daily_loss, account_max_daily_loss=None,
  alerts_to=None, account=None, params=None, ..., live_dir=None, on_stop="flatten")`
  trades one strategy class or a list of them through OpenAlgo for today, in the mode
  set by OpenAlgo's analyzer switch (sandbox or live: no argument chooses or confirms
  it), and returns a `SessionResult` when the session ends (the end of the day, Ctrl+C,
  or after the switch changed). The settings it shares with `bt.backtest` take the same
  defaults; a `bt.backtest` argument with no meaning live raises ConfigError saying why,
  and what this version does not run live (condition, signal and array strategies,
  several symbols, futures series, daily intervals, CNC and NRML) raises
  NotImplementedError before any call to OpenAlgo. `SessionResult` holds the mode,
  broker, account and day, how the session stopped and its restarts, each strategy's
  `bt.Result` (built by the backtest's own builder, with its metrics and tearsheet, for
  this start of the session), the day's summary (`summary`, `summary_text()`), and
  tables of orders, fills, drift, seconds unprotected, locks, reconciliations, skipped
  decisions and open positions, each instant with an IST column; `flat` says whether
  the day ended flat. `bt.record(client, instruments, *, mode="quote", cache_dir=None,
  until=None)` records the feed's ticks for a later replay and returns a
  `RecordSummary`. Importing `openalgo.backtest` loads no module of the live layer.
- Live sessions, before the first public release: the day's summary counts the whole day
  across restarts (closed round trips, rejected orders, mismatches, seconds unprotected,
  undelivered alerts), so a day split by Ctrl+C or a crash reads like the day run
  unbroken; every alert is also printed on the console, whether or not `alerts_to` is
  set; the start banner states the stop rule once, and the first position opened with no
  stop at the broker prints a warning at once and is listed in the result; a comparison
  of a leg in flight for more than 3 minutes that finds a position difference alerts at
  once, and the close names that leg in the summary and the result, never as flat; a
  fill priced from an average is flagged only when its price may lie on another tick.
- Live sessions, from the fault suite (unit 14): a position difference that comparisons
  in time already acted on and that the final read sees again is named at the close (an
  ERROR alert, the summary's Reconcile and End state lines, `SessionResult.flat` False)
  and counted once; the units of an entry the order book cannot tell apart from another
  order are listed in `open_positions` with no confirmed stop, so such a day no longer
  ends "flat"; after a restart, a position carried open across it keeps its entry's
  charges in the restarted start's `bt.Result`, so its trades and its metrics agree. The
  session's behaviour is now checked after every test against the design's safety rules
  (no orphan order, one stop per leg, exits within the position, nothing sent after a
  lock), over every change of every order at the simulated broker.
- `bt.read_ticks(day, instruments=None, *, cache_dir=None)` reads the ticks `bt.record` (or
  a session) recorded on a day back as one DataFrame, merged in the exact order they were
  received: the exchange and symbol, the exchange's last trade time (or the receive time
  where the feed gave none, with a column saying which), the receive time, the last price,
  and the volume, bid and ask where the recording kept them. A day with no recording
  raises DataError naming the folder.
- Live sessions: another client's fill on a symbol the session trades (a sale by hand, for
  example) now prompts a comparison with OpenAlgo's books 3 seconds later, so the session
  fits its stop to the sale within seconds instead of at its next 30-second read; the
  preflight warns, in the start banner and the result, that the session's matching of the
  trade book to its orders by order ID has not been checked for pocketful and mstock. The
  loop's time from a tick's arrival to the kernel is measured by
  `benchmark/live_loop_speed.py` (median about 7 microseconds, 99th percentile about 19,
  against a target under 500).
- Live sessions, figures a trader reads at the end: a strategy's `bt.Result` now adds up
  to its trades. A session's stop (Ctrl+C, the end of the day after the square-off, or an
  error) settles the day's contract notes and records the minute it stopped in, so the
  Result's equity, net profit, returns and drawdown include the session's last fills and
  the notes' charges: its net profit equals the sum of its trades' net, plus an open
  position's value at its mark less its charges. The day summary's charges stay each
  fill's charges rounded to the paisa (they add up across restarts) and differ from the
  Results' by exactly the contract notes' rounding, shown in `Result.contract_notes`. The
  day summary's slippage is now measured: each MARKET fill against the price a backtest
  fills the same order at (the open of the bar after the decision, with the run's
  modelled slippage), signed against the trader and times the units, for example
  "Slippage : Rs 2.50 better than the backtest"; a fill with no bar to compare it with
  (a Ctrl+C sale before the next minute's first trade) is counted as not compared. A live
  trade's MAE and MFE judge the ticks after a confirmed change of a stop or target, and
  the next minute's, against the terms it replaced too, so a fill on the old terms is
  never counted with the whole position. Backtest figures do not change.
- Reference table `price_protection` (`openalgo/backtest/reference/data/price_protection.csv`,
  read with `tables.find("price_protection", (exchange, segment, check, underlying, days),
  date)`): the price bands NSE and BSE apply to an option's limit price, by date, each row
  with its exchange circular or notice and its verification status. `underlying` is ALL or
  one underlying with a rule of its own; `days` is every_day, or expiry_day for a rule that
  holds only on a day an option on the underlying expires. Limit price protection, met by
  a stop-limit order when it triggers: the limit within 40 percent (20 rupees minimum) of
  the contract's 30-second average trade price on NSE options, BSE index options from
  2025-12-01 and BSE stock options from 2026-07-06; 60 percent (30 rupees) on BSE index
  options from 2024-04-23 to 2025-11-30 and BSE stock options from 2024-04-23 to
  2024-06-30; 40 percent with a 10 rupee minimum on every SENSEX option on a SENSEX expiry
  day from 2026-03-12; 80 percent (80 rupees) on BSE FOCUSED IT options from 2026-05-11.
  From 2024-07-01 to 2026-07-05 BSE stock options were under BSE's Price Reasonability
  Check instead, whose rupee slabs are never narrower than 10 percent of the premium. The
  stop-limit range, met at entry and on every modification: the limit within 20 percent
  (10 rupees) of the trigger, on NSE options and BSE index options from 2026-02-09. The
  stock option floor and ceiling during an underlying's cooling-off: 15 percent (7.5
  rupees) on NSE, 30 percent with no rupee minimum on BSE. Every row is verified: the BSE
  notices were read to 2026-11-16. A kernel test reads the table and finds the 5 percent
  option buffer inside every band at every premium on the 0.05 and 0.01 tick grids, for a
  stop-limit's limit, a GTT child's and the reprice of a stop the price has passed, except
  where a band is narrower than one tick: the BSE cooling-off band at a premium of three
  ticks or less, where no limit off the premium fits. The buffer is unchanged. No change in
  backtest results; the reference data's version string now carries 2026-11-17, the date
  this table was checked.
- MAE and MFE of every trade, computed by the kernel for backtests and live sessions
  alike. `Result.trades` gains `mae_rupees` (the lowest running P&L the trade showed while
  open, before charges: zero or negative), `mfe_rupees` (the highest: zero or positive),
  `mae_pct` and `mfe_pct` (as percents of the trade's largest capital at work: average
  entry price times the largest quantity held times the contract multiplier) and
  `slippage_rupees`; `result.trade_list()` gains `mae`, `mfe`, `mae_pct`, `mfe_pct` and
  `slippage`. Only prices the trade could have seen while it was open count: the high and
  low of each base bar it held through, its fill prices, the open of a bar it exited in,
  and a live session's ticks. In a live session a trade whose closing order may have
  filled unseen (a bar reached its stop, or a stop change waits for the broker) counts
  only its fills until the broker reports, so its MAE and MFE can be smaller than the
  backtest's, never larger. Every existing column and figure is unchanged.
- Slippage paid: `Result.fills` gains `slippage_rupees`, each fill's price against its
  price before slippage times units and multiplier, positive when worse. It is already in
  the fill prices, so no P&L changes; a fill the broker priced in a live session has none
  (NaN), and so has its trade.
- New metrics, group "Excursions and trading cost": `average_mae` and `average_mfe` (as
  shares of capital at work), `worst_mae` and `best_mfe` (rupees), `edge_ratio` (mean MFE
  over mean MAE), `profit_kept` (the winners' gross P&L over their MFE),
  `closed_turnover`, `total_slippage`, `trading_cost` (charges plus slippage),
  `trading_cost_share_of_gross_pnl` and `trading_cost_bps` (basis points of the closed
  trades' turnover). A trades table without the new columns gives NaN for them.
- Tearsheet, Trades section: a Trading Cost row (turnover of closed trades, charges,
  slippage, trading cost, its share of gross P&L and its basis points of turnover); a MAE
  and MFE block with six tiles, histograms of MAE and MFE percent and scatter charts of
  each against the trade's return (winners as circles, the others as crosses); MAE and
  MFE columns in the trade list; slippage and trading cost rows under the Charges table.
  Money tiles show paise below Rs 1,000. Mode "basic" shows the trading cost row and the
  histograms, "charts" the histograms and scatter charts (the Trades section now appears
  in that mode), "metrics" the new metrics.
- Robustness and overfitting checks, first part: `bt.robustness(strategy, client=None, *,
  checks="all", objective="sharpe", runs=1000, seed=0, workers=None, progress=None,
  others=None, **settings)` runs the backtest once and then the checks, and `result.robustness()` runs
  them on a finished backtest. Both return a `RobustnessReport`: `verdicts`, a table of all
  seventeen checks with each one's question, figure, rule and verdict; `check(name)`, a
  `CheckResult` with every run, the summary, the settings and the seed; `passed`; `plan`;
  and `tearsheet(path)`, a self-contained HTML report with a scorecard, a section of
  charts per check and the notes. Four checks of the trade list are built: random out of
  sample (whole sessions put out of sample at random, 30 percent by default; passes when
  at least 80 percent of the assignments that can be judged keep at least half the
  in-sample figure), trade
  reshuffle (the drawdowns of the trades in random orders; informational unless
  `max_drawdown` is given), trade resample (trades drawn with replacement; passes with at
  least 95 percent of paths profitable and at most 5 percent ruined, a fall of 50 percent
  from a peak) and variance (the same with the win rate 5 points lower; passes with at
  least 80 percent of paths profitable). A check with fewer than 30 closed trades, too few
  sessions, or a run that did not make money says "not judged" and why, and never passes.
  Each check draws from its own random stream (the seed plus the check's number), so the
  same seed gives the same report on every platform. A check not built yet is listed as
  such, and asking for one raises NotImplementedError saying what it needs. Every verdict
  is worded the same way: passing proves nothing on its own; failing says the edge did
  not hold up under that check (fitted to the past, or too small to tell from noise),
  never that it is proof of overfitting; and the report says which checks can see a
  strategy picked as the best of many tries.
- Quick overfitting checks on every backtest: the four trade-list checks at 200 runs, seed
  0, computed when first read and never during `bt.backtest` or `bt.optimize`. The
  tearsheet has a new "Overfitting checks" section after Trades (full and basic modes),
  `result.summary()` a new "Overfitting" line before Benchmark, and `result.warnings` one
  line for each quick check that fails. No existing figure changes.
- Robustness and overfitting checks, second part: five checks that rerun the backtest on
  other bars. Generated bars keep every bar a valid OHLC on the tick grid and are the same
  for the same seed: permuted bars (each bar's move kept whole, the moves shuffled within
  each session and the sessions' opening moves with their overnight gaps; several symbols
  share one order) and noisy bars (each price moved, with a stated chance, by up to 0.20 of
  the bar's ATR 14). The reruns are ordinary backtests of the same kernel, with the
  warm-up read from the generated bars, in this process or in `workers` processes (the
  same result either way), and the plan states their count and an estimated time first.
  Versus random: three families (random entries on the run's bars, the strategy on
  permuted bars, random entries on permuted bars); passes when the run is above the best
  of every random run (`margin` above it if set), and reports the run's percentile.
  Permutation test: p = (permuted runs at or above the run + 1) / (runs + 1); passes at p
  of 0.05 or less. Noise: passes when the median noisy run keeps at least half the run's
  figure, at least 80 percent of noisy runs make money, and the run is not above their
  95th percentile. Other markets: the strategy unchanged on `others=` (symbols or
  (symbol, exchange) pairs); passes when at least half make money; not judged without
  `others`. Synthetic stress: noisy and permuted candidates kept as shocks by rules on
  their own prices (a deeper drawdown, a longer recovery, a deeper fall below the
  200-session average, returns little correlated with the real ones, or a function of
  the real and candidate bars); each rule's candidates are judged against the share of
  their generators' candidates that made money. Each check has its charts (equity fans,
  histograms with the run marked, real against noisy candles, tables by family, symbol
  or rule) in the robustness report. The random out-of-sample chart now draws the price
  when the report has the run's bars. A check of these five with fewer than 30 closed
  trades, or on a run that did not make money, is not judged.
- Robustness and overfitting checks, third part: four checks that rerun the backtest on
  the run's own bars with one thing changed. Shifted bars: the strategy bars built with
  their boundaries moved (an intraday strategy's bars ending 1 to n-1 base bars earlier; a
  daily strategy's days ending 15, 30, 45 and 60 minutes earlier); passes when every shift
  makes money and the median keeps at least half the run's figure. Random exits: each
  trade's stop, target and trailing distances scaled at random within 25 percent (and
  any `exit_params` drawn from the values given); passes when at least 70 percent of runs
  make money. Delayed orders: entries and exits 0 to 2 strategy bars late in every pair;
  passes when entries one bar late still make money and keep half the run's figure (a
  late run that took more entries than the run is not judged). Liquidity: the fills
  priced by six other models; passes when every model makes money and no drawdown is
  deeper than 1.5 times the run's.
- `bt.backtest(fill=...)` takes six more fill models for these checks and for your own
  tests: "next_close", "next_mid" (the high and low's midpoint), "next_ohlc4" (the average
  of open, high, low and close) of the next base bar, and the same averaged with the bar
  after it ("next_close_2", "next_mid_2", "next_ohlc4_2"). `bt.backtest(delay_entries=0,
  delay_exits=0)` holds a decided order that opens or adds (or that closes, reduces or
  reverses) for that many strategy bars; the engine's own stops, targets and exits are
  never held. The default run is unchanged (the golden runs give the same figures);
  `bt.live` refuses both settings, saying why.
- Robustness checks on generated bars (second part, refined): a synthetic stress rule
  that keeps more than half its candidates is not judged ("it does not single out
  shocks"); permutation, synthetic stress and the strategy-on-permuted family of versus
  random are not judged for `bt.from_arrays` strategies, whose decisions are fixed in
  advance; the plan counts and times only the reruns that will run; the random entries
  of an all-intraday run stay within one session each.
- Robustness and overfitting checks, last part: the four checks that can see a strategy
  picked as the best of many settings, so all seventeen are built. Give `bt.robustness`
  (or `result.robustness`) parameter ranges as keywords (`fast=range(10, 31, 5)`) and the
  chosen settings in `params=`. Out of sample: the first 70 percent of sessions searched,
  the best combination rerun on the last 30; passes when it makes money out of sample and
  keeps half its in-sample figure (totals compared per year). Walk-forward: five rolling
  folds (or anchored), each searched on its in-sample part and traded on the next; passes
  when the stitched out-of-sample run makes money, its efficiency (out-of-sample figure
  over the folds' in-sample figures) is at least 0.5 and 60 percent of folds make money.
  Parameter neighbourhood: every combination one grid step from the chosen one (or a 10
  percent cube around it); passes when the 25th percentile of the neighbours makes money
  and their median keeps half the chosen figure; with a heatmap, a surface view, a
  histogram and a blend of the neighbours. Noise-robust optimization: every combination
  scored on the median of 50 noisy copies of the bars; informational: its figure says
  whether the robust choice sits on a plateau (the neighbourhood rule on the noisy
  scores), and it counts neither as a pass nor as a failure.
- `bt.walk_forward(strategy, client=None, *, folds=5, anchored=False, in_sample=0.7,
  objective="net_profit", workers=None, progress=None, **settings_and_ranges)` is public
  (it raised NotImplementedError before) and returns a `WalkForwardResult`: `folds` (each
  fold's windows, chosen parameters, in-sample and out-of-sample figures and efficiency),
  `equity` (the stitched out-of-sample curve), `performance`, `efficiency`,
  `profitable_share`, `net_profit` and `drift` (how far each parameter moved between
  folds).
- `bt.optimize` results gain the overfitting measures of a search: `neighbourhood` (the
  best combination's neighbours, with a warning in `warnings` when the best is a lone
  spike), `deflated_sharpe` (the best Sharpe ratio corrected for the number of
  combinations tried), `overfitting_probability` (the probability of backtest overfitting,
  by combinatorially symmetric cross-validation over 16 blocks of sessions),
  `overfitting` (both figures with their inputs) and `tearsheet(path, x=, y=)`, a heatmap
  page. No existing column or figure changes.
- Tearsheet: the trade list's entry and exit times wrap between the date and the time, so
  every column fits a narrower window; the MAE and MFE histograms have a count axis.
  `bt.backtest`'s delays: a reversal counts as an exit and takes `delay_exits`.
- Polish before the release: `bt.walk_forward` given a client now runs on its worker
  processes (it read the bars once and ran in one process, giving the wrong reason); when
  a setting truly cannot be sent to workers (a cost model, a calendar), the plan names it.
  The robustness report, the tearsheet and the optimizer's page fit their window at 1024,
  1280 and 1440 pixels with Export PDF and the theme switch in view (the section links
  scroll inside the header); the optimizer page's axis labels and its best-setting tile
  fit their charts, and its cross-validation histogram counts splits. Percentiles read as
  ordinals ("the 66th percentile"); `WalkForwardResult.folds["trades"]` is an integer
  column; the trade reshuffle says its end figure is the closed trades'.

### Changed

- Backtests: an option leg's stop (`stop=` on an MIS option position) is now the
  stop-limit order the exchange takes for it, its limit 5 percent beyond the stop (NFO and
  BFO options accept no SL-M), as a strategy's own SL-M order on an option already was. A
  stop reached inside a bar still fills at the stop, and a bar that opens between the stop
  and its limit at the open; a bar that opens beyond the limit fills at the limit when it
  trades back to it, and otherwise the leg exits at the next bar's open, where it used to
  fill at the gapped open. Slippage on an option stop's fill is held at its limit. Change
  in results: only backtests of MIS option legs with a stop that a bar opens more than 5
  percent through (or whose slippage would pass the limit) change; cash equity and
  futures stops, and CNC and NRML legs (their stops are GTTs with the same buffer), are
  unchanged. None of the four golden runs holds an option leg with a stop, so no pin
  changed.
- An option sell stop or GTT child priced from a trigger of one tick (0.05, or 0.01 on a
  one-paisa contract) keeps a limit of one tick, where 5 percent below it used to round
  down to 0.00, a price no order can carry. One rule now prices every option limit beyond
  a price (`exec::price::limit_beyond`): a stop-limit's limit, a GTT child's limit and the
  reprice of a stop the price has passed. Change in results: only option sell stops at
  one tick, which used to rest with a limit of 0.00.

### Fixed

- MCX futures whose contract multiplier is not 1 are now charged on their traded value
  (price x units x multiplier), in `bt.backtest` and in `bt.live`. Before, each unit was
  charged as if it were worth one quoted price: GOLDM and GOLDTEN (multiplier 0.1) and
  GOLDGUINEA (0.125) were charged too much (about eight times on a GOLDM lot), GOLD (100)
  about 100 times too little, and ZINC, LEAD, ALUMINIUM and their minis (1,000) about
  1,000 times too little. P&L, margin and turnover were already right. Change in results,
  for these contracts only: each fill's charges, the contract notes, the trades' costs and
  net P&L, the equity curve and every metric built on it, the tearsheet's Charges table, a
  live session's day summary, and the size the fixed-risk and ATR-risk sizers choose
  (their cost estimate changes too). Example: one GOLDM lot bought at 1,20,000 and sold at
  1,21,000 is charged Rs 254.76, not Rs 2,122.84. Every NSE and BSE instrument, CRUDEOIL,
  CRUDEOILM, NATURALGAS, SILVER, SILVERM, SILVERMIC, COPPER, NICKEL and GOLDPETAL give
  the same figures as before; the `bt.costs` calculators are unchanged.

- The `turnover` metric (and the turnover ratio built on it) and the trade list's
  `return_pct` now count the contract multiplier. For a contract whose multiplier is not
  1 (an MCX gold future quoted per 10 grams and traded in 1 kg units has 100) the
  turnover read 100 times too small and the return 100 times too large. A trades table
  given to `metrics.analyse` may carry a `multiplier` column (1 when absent). Change in
  results: only instruments with a multiplier other than 1; shares, NSE and BSE
  derivatives and most MCX contracts are unchanged to the bit.

## [2.2.1] - 2026-11-15

### Fixed

- **A server that cannot answer stops the run.** When the OpenAlgo server fails a
  request for the bars before `start` that the indicators need (it cannot be reached,
  keeps answering HTTP 429 after the retries, or answers with an error, such as a
  broker login that has expired), the run asks it once more for the symbol's most recent
  sessions (the week up to the present), without the local cache. If that fails too,
  `bt.backtest` and `bt.optimize` raise the new `ServerError`: "SBIN: the OpenAlgo
  server cannot answer history requests. The bars before start that the indicators
  need failed (...), and so did a request for its most recent sessions (2026-11-08 to
  2026-11-15), asked without the local cache. Check that OpenAlgo is running and logged
  in to the broker, then run the backtest again." Before, a run whose own bars came
  from the local cache went on without its history and waited inside its bars for its
  indicators: a NIFTY futures run with an expired broker login finished with 0 trades
  and only a `WarmUpWarning`. A broker that refuses only older dates (before a listing,
  or beyond the recent history it keeps) still answers its recent sessions, so that run
  keeps the history it could load and says why, as before. A server that keeps
  answering HTTP 429 stops the run at once, with its own advice to raise `pacing=` or
  try again later. A run that needs no history still runs from the cache.
- **A trade's charges are what the account paid for its fills.** Each day's contract
  note rounds STT (CTT on MCX) and stamp duty to the rupee, and the ledger is charged
  the note, but the trades table, `trade_list()` and the trade metrics added up each
  fill's unrounded charges, so the trades did not add up to the account. In the SBIN
  EMA 20/50 run of the tutorial (daily, 2021-01-01 to 2026-11-11, Rs 1,00,000) the
  trades' net P&L came to Rs 53,942.35 against a net profit of Rs 53,941.09, and their
  charges to Rs 3,321.65 against Rs 3,322.91. Now each line of a settled note is shared
  over the fills on it in proportion to what each fill accrued of it (the STT rounding
  goes to the fills that paid STT, the stamp duty rounding to the buys), in whole paise
  that add up to the note. A trade's charges are the sum over its fills, and its net
  P&L its gross P&L less that, so the trades add up to the account's charges and, with
  no position open at the end, to its net profit, to the paisa: Rs 3,322.91 and
  Rs 53,941.09 in that run. The account, the equity curve and every return and
  drawdown figure are unchanged, but for a paisa in a run where a CNC sale and a buy of
  the same share fall on one day (see Changed). The trade figures (`total_costs`,
  `net_pnl`, average win and loss, profit factor, expectancy and the rest) move by the
  rounding: up to about Rs 1.50 per trade, the rupee rounding of STT and stamp duty on
  its notes. A trade's gross P&L now adds each closing fill's realized P&L rounded to
  the paisa, as the ledger posts it.

### Added

- **`ServerError`** (`openalgo.backtest.errors`), a `DataError`: raised for every
  history request the OpenAlgo server could not answer (no response, HTTP 429 after
  the retries, or an error reply), where a plain `DataError` was raised before. Code
  that catches `DataError` still catches it. `rate_limited` is True when the server
  kept answering HTTP 429.
- **`Result.costs` gains `charged`**: what the account paid for the fill, whole paise:
  its share of its contract note plus its DP charge. The column adds up to the run's
  charges. `cost` and `total` stay the unrounded accrual.

### Changed

- `result.fills["costs_rupees"]` is what the account paid for the fill (`charged`).
  Before, it was the fill's unrounded charges, and a CNC sale that turned the day's
  earlier buys into intraday trades also carried their change in charges.
- The tearsheet's charges table and `Performance.cost_breakdown` itemise the
  contract-note rounding ("Contract-note rounding", column `note_rounding` of
  `result.performance.trades` in contract-note rounding), so the components add up to
  the total charges instead of leaving a "Not itemised" row.
- `Result.trade_list()` gives `gross_pnl`, `charges` and `net_pnl` rounded to the
  paisa, and `return_pct` from the rounded net P&L.
- The ledger posts a fill's charges and its DP charge each rounded to the paisa, where
  it rounded their sum once, so it equals the contract notes plus the DP charges
  exactly. A run's charges can differ by a paisa from 2.2.0 only where a CNC sale and
  a buy of the same share fell on one day.

## [2.2.0] - 2026-11-15

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
- **A wait for the indicators is told on screen.** A `bt.backtest` run in which the
  strategy waited inside the run for its indicators to warm up ends with one
  `WarmUpWarning` (a `UserWarning` in `openalgo.backtest.errors`), pointed at the line
  that ran the backtest: the first decision and the strategy bars each symbol waited,
  for example "The strategy waited inside the run for its indicators to warm up and
  made its first decision at 2026-01-22 15:30 IST (SBIN waited 15 strategy bars):
  signals before then were not taken. See result.warnings for why.", that it made no
  decision at all, or that it made none for a symbol whose indicators were never ready.
  Once per run; `bt.optimize` does not issue it. `result.warnings` keeps the full line,
  with why and what to do.
- **The causality registry records anchored indicators** (`causality.Anchor`, the
  `anchor` of an `Indicator` row): the argument that names the anchor period, the whole
  periods before the current one a value reads (0 for `ta.vwap`, 1 for
  `ta.pivot_points`) and the arguments that supply bar times of the user's own.
  `causality.causality()` records gain an "anchor" key, None for every other
  indicator; a test checks that every `ta` function with an `anchor` parameter has one.

### Changed

- **Indicators warm up automatically** (changes results). The strategy makes no
  decision before `start` and none before its `self.I` indicators are ready: an
  indicator is ready once its series has closed its warm-up of bars. The warm-up is the
  window-stability length measured on the engine's own reference bars (65,536 bars made
  from integers, the same on every platform, with long trends, ranges, reversals and
  opening gaps; each window compared at 1,024 bars, and at the events of a condition),
  never on the run's data: it depends on the indicator call alone (the function, its
  parameters and the bar fields it reads) and is counted in the bars of the indicator's
  own series, so no decision depends on a later bar, a different `end=` leaves the early
  trades alone, and the same call needs the same bars on daily and on 15-minute data.
  For example 256 bars for a 20-bar EMA, 512 for a 50-bar one, 2,048 for a 200-bar one,
  64 for a 50-bar simple average, 256 for a 14-bar RSI, 512 for MACD 12/26/9 and for
  Supertrend 10/3, 128 for the parabolic SAR, 16 for a 3-bar average; a test checks the
  warm-up of each common indicator against eight differently shaped price series. An
  indicator of other views needs theirs before its own (a crossing over
  `self.close.rolling_max(100).shift(1)` needs at least 102 bars). An anchored
  indicator, as the causality registry marks it (`ta.vwap` with anchor Session, Week,
  Month, Quarter or Year, and `ta.pivot_points` with an anchor), needs the bars since
  the start of the first anchor period it reads: a VWAP its own period, floor pivots the
  whole period before their own as well (monthly pivots on 15-minute bars keep the
  whole previous month, and daily pivots after a 14:00 datetime `start` keep the
  previous session). They are loaded, and when the data begin inside the period the
  strategy waits for the next one. One that needs more than the longest
  window (an EMA of 10,000 bars) says so, gets up to 30 years of history and waits that
  window at least. Live trading keeps the same windows. A backtest learns what its
  indicators need before it loads any data, by running `init` once on the reference
  bars, so `init` runs twice: there, then once on the data. It then loads, from the
  run's own source (Historify or the broker its bars came from, or the bars of
  `data=`), the bars before `start` each series needs, in that series' own interval (a
  weekly EMA(50) gets its 512 weeks, a monthly EMA(10) its 128 months), at most 30 years
  back. They are asked for in steps of a fixed ladder of spans, from the span the need
  fills at full session density, and further back while the bars returned fall short
  of a need and a step returned bars older than the one before it (a thinly traded
  share whose minutes without a trade have no candle, a suspension across the first day
  of a step), so a need is met whenever the source holds the bars. Each step asks only
  for the days before those already loaded, so a symbol whose bars never meet a need
  asks for the thirty years once in all, in at most 31 requests. A step that fails
  (a bar off the lattice) ends the steps but keeps the bars of the steps before it and
  says why loading stopped, and when own data falls short the warning names the first
  bar the data holds and, when the history stopped at a gap in it, the gap. With a
  datetime `start` the bars of that day before it are history too, so a
  daily view and a strategy bar cut by `start` hold the whole day. Every series holds
  the history before `init` runs on the data, so every indicator is computed over it
  (`init` itself sees no bar, so `len(self.close)` is 0 there and on the probe)
  and one longer than the run (a 200-day EMA on a run of 60 days, or in a
  `from_signals` function) works. The history kept and the run's first base bar are
  checked as one series, and no later bar, so the afternoon of the start day cannot
  decide whether its morning trades: a suspected unadjusted split at the join leaves
  the run without history and says why. A split or bonus declared with
  `corporate_actions=` that the history is not adjusted for is adjusted for in it, as a
  split-adjusted chart shows the bars before it (prices before the ex-date divided by
  its factor, volume multiplied by it), so the indicators at `start` are those of the
  adjusted series, and the Assumptions line says so ("adjusted for a split (2 shares
  for every 1) on 2024-12-17: prices before it divided by 2, volume multiplied by 2");
  the run's own bars and an ex-date inside the run are handled as before. `on_bar`,
  timers, conditions and arrays wait alike;
  `from_arrays` arrays are read for the run's bars only, an indicator over an array of
  the user's own is computed on the run's bars, and a `from_signals` function is
  computed over the history a 200-bar EMA needs (its indicators are its own, so the
  strategy does not wait for them). Equity, trades, charges, metrics and the benchmark
  still start at `start`. Readiness is per symbol and causal: at each strategy bar a
  symbol (or futures underlying) is ready once its own indicators have closed their
  warm-up, counting only the bars up to that bar, so it never depends on a later bar or
  on how many bars follow (a later `end=` cannot stop another symbol from trading in
  January). A symbol not ready yet gets no bar in `on_bar` and its orders are skipped
  (skip reason "not_ready"; a rebalance that gives it no weight, or a `target()` of
  flat, sends nothing while it is flat and is no skip), it joins at the first bar it is
  ready, and the symbols
  that are ready trade; nothing is decided before the first symbol is ready, so a run
  of one symbol simply waits. An indicator of a symbol that is not ready yet reads NaN,
  or false for a signal, wherever it is read (deciding for another symbol, in a timer,
  through another interval of it or a futures series), as does every other view
  computed from its bars (a comparison, a shift, a crossing); its bar fields (open,
  high, low, close, volume) and `self.history` stay readable. A filter on a symbol not
  ready yet (trade SBIN while NIFTYBEES closes above its EMA) therefore does not trade
  on it: before, its values were shown, and they could depend on how many bars followed
  (a 20-bar EMA is not computed on fewer than 20 bars and is computed from the first bar
  on 20 or more), so a later `end=` could add a trade in January. The values are kept:
  once the symbol is ready, its lookbacks (`line[-2]`) read them. When the source has
  fewer bars before `start` (own data
  that begins at `start`, a new listing, a futures series rolled from contract to
  contract, which is built from `start`), the symbol waits inside the run until its
  indicators are ready; a warning counts the strategy bars each symbol waited and says
  why (for own data, to give earlier bars, naming the day they begin; for a source, "the
  source has no bars of BBB before 2026-02-02"; for an indicator the run's `init`
  registers but did not register on the reference bars, that no history was loaded for
  it), and `bt.backtest` ends with a `WarmUpWarning` that gives the first decision of
  each symbol that waited. A symbol whose series has fewer bars than a need of its own
  makes no decision, and the manifest lists it under `no_decision`. An indicator whose
  series is too short for its function is not computed, and its values are unknown: a
  call that raises on its series and computes on the same values with more bars in
  front was refused for the length (a 100-bar average on 60 bars of own data, a 20-bar
  ulcer index on fewer than the 52 bars it checks for, a 20-bar KAMA, CMO, historical
  volatility, regression slope or VIDYA on exactly 20 bars stopped the run with a
  ConfigError before, and now make no decision for that symbol with a warning, as an
  RSI did). Its warm-up is longer than the series, so the symbol is not ready at any of
  its bars, as it would not be by then on a longer series. One over values of the
  user's own is not waited for on a series of any length, so on one too short for it its
  values are unknown and a warning says so. A call that fails on the longer series too
  (a period of 0) still stops the run. A run on one explicit futures contract
  loads that contract's own bars before `start`. The Assumptions block gains `warm_up`
  (the rule, each series' need in its own bars, the history kept with the day asked
  from and, when later, the day the source's bars began, for a symbol with none the day
  its bars begin ("the bars given for BBB begin on 2026-02-02"), and the first decision
  of each symbol), the manifest records the warm-up under `arguments["warm_up"]` (the
  needs, with `since` for an anchor period, `periods_before` for one that reads whole periods
  before its own and `more_than` for a need longer than the longest window, the
  indicators not waited for, the run's first decision and each symbol's
  (`first_decisions`, None for a symbol ready at start), the strategy bars waited by
  each symbol that waited, and the symbols never ready in a run whose others decided) and
  the partition hashes of the history kept, as the source gave the bars, under
  `data["<symbol> history"]`; the history load's own warnings join the run's.
  `bt.optimize` learns each combination's needs on the reference bars before any data is
  loaded, loads the history once for the largest, in which each combination finds the
  request its single run makes, and each combination keeps and checks exactly the bars
  its single run does (a data problem outside its own history does not cost it that
  history; when a step of the larger load fails, as one bar off the lattice far before
  `start` makes it, a combination whose steps stop short of it keeps its own, and one
  that reaches it keeps the steps before it, as its single run does), so the table,
  `select()`
  and `rerun()` reproduce single runs. Needs are keyed on the parameter values frozen
  in order, so a list, tuple or dict value works as a number does (a list raised
  TypeError). A strategy's
  own warm-up guard (`if len(self.close) < self.slow: return`) still works; it is no
  longer needed. An indicator no window settles (a running sum such as on-balance
  volume), one over an array of the user's own, and one the reference bars cannot
  compute are not waited for, with a warning that says which. `self.I` called after
  `init` raises `IndicatorOutsideInitError` (a `ConfigError`, in
  `openalgo.backtest.errors`): indicators are registered in `init`, where their history
  and readiness are settled. The truncation check still runs on the run's bars, once
  per function and data. The first trades of runs on short own data move later; the
  intraday golden run gains two sessions of history and the condition golden run
  thirteen months instead of seven, and their pins were regenerated.
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
- **Candles of a special session that opens off the bar lattice are told by the
  session.** Intraday bars sit on a lattice from each date's regular open (09:15 IST on
  NSE), special sessions included, and a source that starts a special session's candles
  at the session's own open (15-minute broker candles of Muhurat trading from 14:20)
  stopped the load with the general off-lattice message. When every off-lattice bar lies
  in such a session, the TimestampError now names it from the calendar data and says
  what to do: "NSE:SBIN 15m: 5 bars are not on the NSE bar lattice: the session of
  2026-11-08 (Diwali Laxmi Pujan (Muhurat trading)) opens at 14:20 IST, off the 09:15
  bar lattice, so 15-minute candles of that session cannot be placed. Pass 1-minute data
  (the engine builds the session's 15-minute bars from it), or leave that session's bars
  out." Own data and OpenAlgo data alike; nothing is loaded differently.

### Fixed

- **A broker history load no longer stops at a holiday the calendar does not know, and
  loses no older bars.** The calendar's holiday lists begin in 2019, so an earlier
  holiday such as 2018-12-25 counts as a session. The cache recorded only dates that
  returned rows, so such a holiday was asked for again on every load as a range of its
  own, and the broker walk (newest first) stopped at the first empty window that had a
  session, before the older range was asked for: a load of SBIN daily bars from
  2016-10-27 to 2018-12-31 over a cache holding 2018 returned the 245 sessions of 2018
  instead of the broker's 537 bars, with a short-history warning that blamed
  2018-12-25. An empty window now ends the walk only in the oldest range asked for (a
  later range lies after sessions the cache has settled, so an empty window there is a
  gap), and the reason says what was concluded: "the broker returned no bars for
  2024-02-02 to 2024-04-01, so the history is taken to start after 2024-04-01; nothing
  older was asked for". A session the broker answered with no bars, later than the
  oldest session with bars, is now recorded in the cache as fetched with none and is
  not asked for again once settled. Sessions before the first bar (where the broker's
  history starts) and every empty answer from Historify (which downloads fill later)
  are still asked for again. The cache format is unchanged, so caches written by
  earlier versions are read as they are, with no migration: a holiday they lack is
  asked for once more, then recorded. `FetchReport` gains `answered`, the windows
  answered as ISO date pairs.
- **No false "The run ends at" warning when every session's bars stop at 15:15.** From
  2026-08-03 continuous trading in a stock with derivatives ends at 15:15 IST (the
  closing auction makes no candle), while the calendar keeps 15:30 for every stock, so
  every intraday run that ended on such a stock warned that its data ended early, its
  equity stopped at 15:15 and an MIS position still open was reported open instead of
  being squared off at the session close as on every other day. A run now ends at the
  close of its last bar only when the last session's bars stop earlier, before its
  session close, than those of the latest earlier session with bars; otherwise the empty
  slots stay slices and the run ends at the end of its range or at the present. A run of
  one session, or whose last session is cut short by the range, keeps the plain rule.
- **A `start=` or `end=` string with a time of day is a datetime bound.**
  `start="2026-03-10 11:00"` was read as the plain date 2026-03-10, so the run began at
  the open and the morning was traded instead of being history. A string with a time of
  day other than midnight is now read as the datetime it names (naive in IST, or with
  its offset), as `datetime(2026, 3, 10, 11, 0)` is, and both run the same; a date
  string, or one at midnight, is still the date. The date of a datetime bound with an
  offset is taken in IST.

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
  status stream over the existing OpenAlgo WebSocket connection: fills,
  partial fills, rejections and cancellations pushed by the broker (live
  mode) or the sandbox engine (analyze mode), for orders from any origin
  (API, broker app/web, engine square-offs).
- **`unsubscribe_orders()`**: Stop the order-update stream.
- **`get_orders()`**: Cached order updates received so far (most recent
  first, capped at 500).
- **`[ORDER]` log category**: order updates printed at `verbose=2`.
- Order-update subscription survives reconnects: replayed automatically
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