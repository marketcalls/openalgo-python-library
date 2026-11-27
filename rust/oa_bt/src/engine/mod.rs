//! The event kernel: one deterministic, single-threaded state machine that runs every
//! backtest (and, from Phase 2, live trading) with the same rules.
//!
//! # Data flow
//!
//! ```text
//!  Python (openalgo.backtest)                     Rust (oa_bt)
//!  -------------------------                      ------------
//!  bt.backtest(Strategy, ...)
//!    resolve settings (defaults, Assumptions)
//!    load bars (data layer), resolve instruments
//!    (reference tables), build the calendar   --> RunSetup { RunConfig, Calendar,
//!                                                   [InstrumentSpec], [BarColumns] }
//!                                                 Engine::new: validate, build the
//!                                                 grid (Timeline), align bars and build
//!                                                 strategy bars (MarketData)
//!    driver loop:                                 Engine::advance: run slices until a
//!      pause = engine.advance()               <--   decision point -> Pause
//!      callbacks(pause.events)
//!      on_bar(pause.bars) -> buy/sell/exit    --> Engine::submit / cancel (step 7)
//!      ... until Pause::Finished
//!    engine.finish()                          <-- RunOutput: fills, levels (position
//!    -> Result (DataFrames)                         history), equity, trades, orders,
//!                                                   events, stale marks, final positions
//! ```
//!
//! Python declares and decides; Rust executes. Event strategies (`on_bar` in Python)
//! are driven by the binding through the stepping API, one pause per strategy-bar
//! close. Condition and array strategies run a Rust driver (`signals`) over the same
//! API with no Python call per bar. Both paths reach the same `Engine` and the same
//! steps, so they cannot drift apart.
//!
//! # The stepping API
//!
//! | Call | When | What |
//! |---|---|---|
//! | `Engine::new(RunSetup)` | once | validate and prepare; the engine stands before slice 0 |
//! | `Engine::advance()` | until `Pause::Finished` | run steps to the next decision point |
//! | `Engine::submit(OrderRequest)` | while paused | submission checks, `Submitted` or `Rejected` (in a live session sent, or an exit that cancels first, `venue_owner`) |
//! | `Engine::cancel(OrderId)` | while paused (a live session: or waiting) | cancel a working order (in a live session sent, the order works until the venue's result; sent again after the venue refused it, `venue_session`) |
//! | `Engine::modify(OrderId, OrderRequest)` | while paused | a full replacement of a working order (`modify`; in a live session sent, the old terms stand until the venue confirms, and a replacement that would change the order's role is refused) |
//! | `Engine::submit_target(TargetRequest)` | while paused | the order for the difference to a target position (`target`) |
//! | `Engine::submit_rebalance(TargetRequest)` | while paused | the same order as part of a rebalance (`rebalance`) |
//! | `position`, `leg`, `order`, `working_orders`, `gtts`, `has_pending_entry`, `account`, `last_close`, `decision_instant`, `trading_state`, `strategy_bars`, `timer_counts` | any time | queries |
//! | `Engine::add_timer(TimerSpec)` | before the first `advance` | register `self.at` or `self.every` |
//! | `Engine::new_session(RunSetup)` | once | live clock (`live_clock`): a session built before its bars, ending at the end of its range, holding; its venue is external (`venue`) |
//! | `Engine::add_bar(InstrumentId, BaseBar)`, `Engine::release(slices)`, `Engine::hold()` | any time | live clock: hand over a closed slice's bars, then release it; `advance` returns `Pause::Waiting` before the first slice not released (a backtest never holds) |
//! | `Engine::take_commands()` | any time | a live session's orders for its venue (`venue`): `VenueCommand::Place` (with the order's role, `venue_roles`), `Modify`, `Cancel`, in the order they were made (none in a backtest), but none of an order that has ended since; `venue()` says which venue fills; `order_role(OrderId)` reports an accepted order's role |
//! | `Engine::apply_fill(OrderId, units, price, at)`, `Engine::apply_fill_flagged(OrderId, units, price, at, FillFlags)`, `Engine::apply_reject(OrderId, RejectCode)`, `Engine::apply_cancel(OrderId)` | while waiting, at a timer pause, or while a session close waits | a live session's results from its venue (`venue_fills`): the units newly filled at the venue's price through `execute_fill` (all or nothing, costed once on the order's total; units beyond the position of an order that may not reverse it are refused, `would_reverse`), with the flags the venue reports (`price_unverified`, `missed_day` only), a rejection, a cancellation; each returns the stamp it was recorded with. Every fill of an order comes before its terminal result |
//! | `Engine::apply_cancel_refused(OrderId)`, `Engine::apply_modify(OrderId)`, `Engine::apply_modify_refused(OrderId)` | as `apply_fill` | the venue's answers (`venue_answers`) to a cancellation (the order works on; the engine's own withdrawals send it again in a later slice, `cancel` at once) and to a modification (confirmed: the new terms, complete when cut to what has filled; refused: the old ones stand) |
//! | `Engine::request_exit(InstrumentId, Product, EngineExit)` | while paused or waiting | a live session's exit of a whole position for its own reason (`venue_session`), through the position's exit owner (`venue_owner`); returns what it did (`ExitRequest`: sent, waits for cancels, already exiting, flat, market closed) |
//! | `Engine::request_limit_exit(InstrumentId, Product, EngineExit, limit)` | as `request_exit` | the same exit as LIMIT orders at `limit` rupees, the price rule's marketable fallback (`venue_protect`) |
//! | `Engine::push_tick(InstrumentId, price, at)` | as `apply_fill` | tick mode (`venue_ticks`): the session's last price at `at` ratchets the instrument's legs (the broker's stop follows under its throttle, `venue_legs`), exits a target it reaches through the exit owner, marks the position and runs the daily loss limit and drawdown halt on that mark; a stop it reaches is the broker's |
//! | `Engine::protect(InstrumentId, Product)` | while paused or waiting | a live session places a leg's stop again at its level for what its exits leave, when none works and no exit holds the position (`venue_protect`; a CNC or NRML leg's stop GTT also while the market is closed, or the leg exited when the price has crossed its stop, `venue_gtt_protect`); returns what it did (`Protect`: placed, waits for cancels, stop works, exiting, no stop level, flat, market closed, stop crossed) |
//! | `Engine::apply_gtt_placed(GttId)`, `apply_gtt_refused`, `apply_gtt_modified`, `apply_gtt_modify_refused`, `apply_gtt_cancelled`, `apply_gtt_cancel_refused`, `apply_gtt_ended(GttId, GttEnd)` | as `apply_fill` | a live session's answers about its GTTs (`venue_gtt_answers`): a CNC or NRML leg's stop GTT and a strategy's GTT order, placed with `VenueCommand::PlaceGtt` and kept with `ModifyGtt` and `CancelGtt` (`venue_gtts`) |
//! | `Engine::apply_gtt_triggered(GttId, at)` | as `apply_fill` | the venue fired a GTT: returns the kernel order its child is followed as (`venue_gtt_child`) |
//! | `Engine::gtt_view(GttId)`, `gtt_views(InstrumentId, Product)` | any time | a GTT's state, terms, pending modification and child (`venue_gtts`) |
//! | `Engine::reprice_exit(OrderId, limit)` | while paused or waiting | a live session modifies a leg's stop (or the engine's exit) to a LIMIT at `limit` rupees for the order's total units, partly filled or not; sent now (true) or with the answer to the modification that waits (false); the repriced stop exits the position (`venue_protect`) |
//! | `Engine::lock_entries(reason)`, `Engine::unlock_entries(reason)`, `entry_locks` | any time | a live session's entry locks, reasons in a set (names with no spaces around them): while any is set, entries and adds are rejected `live_lock` (`venue_gates`) |
//! | `Engine::cap_sizing_equity(rupees)`, `Engine::daily_loss_measure()` | any time | a live session's cap on the engine default's sizing equity; the strategy's daily loss measure (`venue_session`) |
//! | `Engine::set_group_side(InstrumentId, Product, Option<GroupSide>)`, `Engine::set_shared(InstrumentId, Product, bool)` | any time | a live session's gates on an instrument several of its strategies trade (`venue_group`): an entry, add or reversal on the other side of the side the others hold or are entering is rejected `group_other_side`; while shared, a strategy order that rests is rejected `shared_resting` |
//! | `Engine::apply_margin_reject(OrderId)` | as `apply_fill` | a live session's closing order whose rejection says the broker may not hold its units: it ends rejected and holds its position, so the kernel sends nothing that closes it by itself (no exit at market, no stop placed again or raised, no waiting exit, none at later opens) until the session asks with `request_exit` or `protect` (`venue_group`) |
//! | `Engine::apply_external_fill(InstrumentId, Product, Side, units, price, at, ExitReason)` | as `apply_fill` | a live session's fill with no order of its own (`venue_external`): the broker closed or reduced the position by itself; recorded through `execute_fill`, flagged `external`, then the closing orders are fitted to what is left; refused whole unless it only closes or reduces (`beyond_position`) |
//! | `Engine::live_state()` | any time | a live session's state as plain data that survives a JSON round trip (`venue_state`): positions with tranches, legs, working orders with roles and terms, pending modifies and cancels, exits and leg stops, the daily loss totals, the trading state, counters, cash |
//! | `Engine::restore_live_state(&StateValue)`, `Engine::restore_live_state_at(&StateValue, now)` | once, in a session built for a restart, after its catch-up and before any order | the saved state put back as it stood (`venue_state`; at the present `now`, a pre-open decision's orders waiting for the open go at once, flagged `late_decision`, once the open has passed), but for the entry locks (the session keeps its own) and a sizing cap the session has set; returns (saved id, new id) of each restored working order; refused whole (`LiveStateError`), also for parts that do not hold together and for a state saved while the session closed |
//! | `Engine::carried_state()` | once the day's session has closed (after its close, or `stop_session`) | `live_state` at the close without its DAY parts, for the next trading day's session (`venue_state::carry`): positions with tranches, legs, the GTTs at the venue and the GTT orders, cash, the trading state, the daily loss measure's totals, the slices held |
//! | `Engine::restore_stopped_state(&StateValue, Day, &[GttId])` | once, in a session built for an earlier day whose live session stopped without its close, before any of its slices | that session's last saved state put back where it stood (`venue_state::stopped`), to take the venue's results after the save, then `stop_session` and `carried_state`; a command never handed out is dropped; refused whole (`LiveStateError`), also for a state of the given day or later |
//! | `Engine::restore_carried_state(&StateValue)` | once, in a session of a later trading day, after its catch-up and before any order | the carried state put back, the GTTs and GTT orders under new ids (`CarriedIds`); the daily loss measure and the trading state start the session as a backtest's do at a session boundary; a position on a contract that expired before the session's day recorded exited at its carried last close (`CarriedIds::expired`, `expiry`); refused whole (`LiveStateError`) |
//! | `Engine::pre_open_decision(&[StrategyBar], last_decided, now)` | once, in a daily or longer strategy's session, after its catch-up and before it runs live | the latest completed bar of the strategy's interval decided before the open (`PreOpen`): a `Pause::BarClose` at which the strategy submits; its orders go as the session reaches its open, or at once flagged `late_decision` after it (`pre_open`) |
//! | `Engine::order_flags(OrderId)` | any time | the flags a working order's fills carry: `late_decision`, `gtt_child_unfilled`, `stop_crossed` (`pre_open`) |
//! | `Engine::market_fill_price(InstrumentId, Side, bar_open, open)` | any time | the price a backtest fills a MARKET order at, at the open of a bar the caller names, by the run's slippage rule; records nothing (`market_fill`): what a live session's day summary sets its MARKET fills against |
//! | `Engine::stop_session()` | any time | a live session ends where it stands: nothing is sent, working orders and open positions are reported as they are (a strategy exit never sent ends `Canceled`), the slice in progress is closed for the records as step 8 closes a session (the day's contract notes settled, then its position-history and equity rows), the undelivered events are returned (`venue_session`) |
//! | `Engine::finish()` | after `Pause::Finished` or `stop_session` | the `RunOutput` |
//! | `pause::run(engine, driver)` | Rust drivers | the loop above, for a `Driver` |
//! | `Engine::set_account_link(AccountLink)` | before each of the engine's turns in a portfolio run | the room a shared account leaves the engine: the second margin check of its opening fills (`account_margin`) and its sizing cap (`account_link`) |
//! | `Engine::account_breach(AccountBreach)` | at a round of a portfolio run | the account's daily loss limit or daily profit target (`AccountRule`): entries locked for the session day (`account_loss_gate`, `account_profit_gate`), the breach recorded where the engine stands, its entries cancelled and its positions exited (`account_breach`) |
//! | `Engine::set_account_trades(Option<AccountTrades>)`, `Engine::trades_today()` | before a strategy's decisions in a live session (the fund sets it with the link); any time | the account's trade cap and the other strategies' entries of the day, which the account's trade gate counts with the engine's own; the entries the strategy has counted on its session day (`daily_caps`) |
//!
//! `advance` runs each slice as its phases in sequence (18-multi-strategy 2.3): the open
//! phase (steps 1 and 2, `open_phase`, resumed by `resume_open_phase` after a timer pause),
//! `Phase::Opened`, the levels phase (steps 3 to 6, `levels_phase`), the decision of step 7
//! (`decision_phase`) and the close of step 8 (`end_slice`). They are the crate's entry
//! points for a portfolio run's fund, which runs each phase across its engines before the
//! next; a run of one engine runs them back to back, exactly as before.
//!
//! A pause is a decision point: `Pause::BarClose` in step 7 of a slice where a strategy
//! bar closes by the clock (or where events wait and the strategy has callbacks), and
//! `Pause::Timers` in step 2 for `self.at` and `self.every` (`timers`; orders it
//! receives fill at that same open). A step 7 pause carries the events since the
//! previous one, so callbacks see everything of steps 2 to 6 before `on_bar` runs; a
//! timer pause carries none (SPEC-31). Events of step 8 and of submissions are
//! delivered with the next step 7 pause.
//!
//! In live trading (Phase 2) the same machine runs with a live clock: the engine holds
//! (`hold`) and the adapter releases each slice once it has closed and its bars are in,
//! and tick mode runs steps 3 to 6 for one instrument per tick (03-architecture 4.2, last
//! paragraph; PRD 15-phase2-plan section 2). A run released one slice at a time is the
//! same run. Its venue is external (`venue`): the engine hands every order to the adapter
//! as a command and records the broker's fills, rejections and cancellations; it fills
//! and expires nothing itself. What a backtest does at the open of a slice (timers, the
//! engine's exits) happens as the session reaches that open, the close of the slice
//! before, and a session close waits for the venue to end its orders (`venue_clock`).
//! A leg's stop is the broker's, as an order the kernel keeps on the leg (`venue_legs`,
//! decision D31); its targets, trail and breakeven are the kernel's, on ticks
//! (`venue_ticks`): a session runs steps 3 and 4 on ticks, not on its bars. The order and
//! risk rules do not change.
//!
//! # The eight steps of a slice (03-architecture 4.2)
//!
//! The grid is the calendar's base-interval lattice: every slot is a slice, with or
//! without bars (SPEC-07). Each step runs across all instruments in registration order
//! before the next step starts.
//!
//! | Step | Code | Built |
//! |---|---|---|
//! | 1 session open, reference events | `steps::open_phase` (resumed by `steps::resume_open_phase` after a timer pause); intraday times resolved lazily by `session::session_times`; GTTs older than 360 days re-created (`gtt::renew_old_gtts`, GTT-2); futures switches are planned before the run (`futures`); then, under a delay, the cancellations and modifications whose delay has ended (`held_changes::apply_held_changes`, stamped in step 2) | times, GTTs and rolls yes; lot-size change events no |
//! | 2 fills at the open | `steps::step_open_fills`: per instrument the risk exits of step 6 (`risk_exits_at_open`) and the engine's exits after unfilled GTT children (`gtt_fire::gtt_exits_at_open`), the rearm_daily stops an instrument's first bar of a session opens beyond (`rearm::rearm_stops_at_open`), then `fills::fill_queued` (eligible when the open is at or after `decided_at`, SPEC-35, or the end of the order's delay, `delay`; a missing bar keeps the order queued, SPEC-06; entry window and margin checked at the fill bar; under a fill model they fill in step 3 instead, `fill_model`), the time exit, then `session::square_off_at_open` (the square-off timer, SPEC-05), then the exit on a contract's positional exit day (`expiry::expiry_exit_at_open`); after every instrument, `rebalance::fill_rebalance_buys` (the buys of a rebalance, after its sales), then `roll::roll_due_futures` (the futures auto-roll, near leg first, SPEC-19, GTT-9). In a slice where a timer is due, `timers::fill_due_orders` runs the engine exits and queued orders first, `timers::timers_pause` returns `Pause::Timers`, and `step_open_fills` then fills what the timers placed | MARKET orders, converted stop-limit exits, timers, square-off, rolls |
//! | 3 resting orders, leg levels | first a portfolio run's account breach found while the slice was open cancels the working entries and adds (`account_breach`); then `steps::step_levels` -> `legs::test_leg_levels` -> `rms::LegState::test_bar` (MIS; an option's stop as the segment's stop-limit, R12; a CNC or NRML leg's targets alike, its stop fired as its GTT, `gtt_levels`), `gtt_fire::test_gtts` (strategy GTT orders), `resting::test_resting_orders` (LIMIT, SL, SL-M), exits through `fills::execute_fill`; then, under a fill model, the queued orders at the model's price (`fill_model::fill_queued_at_model`); a live session on ticks (`venue_ticks`) | yes |
//! | 4 ratchets | `steps::step_ratchets` -> `legs::ratchet_legs` -> `rms::LegState::ratchet`, then `gtt::sync_gtts` (GTTs follow the levels, GTT-3); a live session on ticks | yes |
//! | 5 mark | `steps::step_mark`: close, or last close and stale; the open trades see the slice's bar (`excursion`) | yes |
//! | 6 aggregate rules | `aggregate::step_aggregate_rules`: daily loss limit, daily profit target (`daily_caps`), drawdown halt; their exits at the next open (`risk_exits_at_open`, step 2) or the last close; the session boundary in step 1 (`risk_session_open`); a portfolio run's account breach after them, or in step 1 when it was found between slices (`account_breach`) | yes (combined rules: Phase 3) |
//! | 7 callbacks and `on_bar` | `steps::callbacks_pause` -> `Pause::BarClose`; `submit::submit` | yes |
//! | 8 session close | `steps::close_slice`: expire DAY orders, close open MIS positions at the last close (reason `eod`, flagged) and the positions on a contract whose positional exit day it is (reason `expiry_exit`, flagged, `expiry`), settle expiring futures and post the futures mark to market (`settlement::settle_futures`), settle the day's contract notes (`accounting::settle_contract_notes`), credit the day's withheld sale proceeds (`sale_credit::credit_withheld_sales`) | yes (options expiry: Phase 3) |
//!
//! Every fill takes one path, `fills::execute_fill`: slippage (`market::slippage`, the
//! run's or the dated default table's), position and tranches (`portfolio`), the ledger,
//! the charges (`charges`, posted to the ledger; contract notes settle in step 8), the
//! leg state (`rms::leg`), then the records: the fill with what its slippage cost,
//! `Filled` (when it has an order), the position event, the trade and its excursion
//! (`excursion`: the MAE and MFE, from every fill, the bars in step 5 and the ticks of a
//! live session). The glue to the market rules is in `accounting`, the sizers' in
//! `sizing`.
//!
//! # Determinism
//!
//! One thread; no clock, no randomness, no hash-ordered container. Instruments are
//! processed in registration order, orders in id order (ids are dense and in
//! submission order), legs in the order their positions opened. Prices are on the tick
//! grid (integer price units, `exec::price`) and ledger postings in paise
//! (`portfolio::ledger`), so the same inputs give bit-identical results.
//!
//! # Modules
//!
//! | Module | Responsibility |
//! |---|---|
//! | `config` | `RunConfig`, `RunSetup`, `BarColumns`, interval and mode enums; what is refused as not built |
//! | `clock` | `Timeline`: the grid, slice facts, strategy-bar closes by the clock |
//! | `run_end` | Where the run ends: the end of its range, its last bar or the present |
//! | `market_data` | Base bars aligned to the grid; strategy bars via `data::resample` |
//! | `kernel` | `Engine`, `advance` (as the phases of a slice), `finish` |
//! | `account_link` | A portfolio run's account in one engine: the room it leaves the engine, the second margin check of opening fills, the sizing cap, the account's entry gate |
//! | `account_breach` | The account's daily loss limit breached in one engine: the lock, the record stamped where the engine stands, the cancels and exits |
//! | `fund_view` | What a portfolio run's fund (`crate::fund`) reads of one engine between its turns: where it stands, its grid, the ledger's figures, its events, the orders a breach cancelled |
//! | `live_clock` | A live session: built before its bars, given each bar as its slice closes, holding before the slices not released |
//! | `venue` | Who fills the orders (`Simulated`, `External`); the venue's types and its desk of what was asked and not answered; what a live session refuses as not built |
//! | `venue_orders` | A live session's commands out: the units an order is sent with, the order sent or made an exit, a modification held until answered, a cancellation (the engine's own sent at most once in a slice) |
//! | `venue_owner` | The exits of a position: cancel the orders on its closing side, then exit (a partial exit cuts the leg's stop down instead); the strategy's exits add up, the engine's takes over |
//! | `venue_exit_orders` | The orders an exit goes out in: a strategy exit sized on what the earlier ones leave, the engine's for the whole position at market or at a limit, in orders of at most the freeze quantity |
//! | `venue_legs` | A leg's broker-held stop: placed in the segment's shape, its units and trigger kept on the leg's under the throttle, the leg protected again when the stop ends without a fill |
//! | `venue_ticks` | Tick mode: `push_tick` ratchets the legs, exits the targets reached, marks and runs the aggregate rules |
//! | `venue_protect` | The session's calls for a leg: `protect`, `reprice_exit`, `request_limit_exit` (`Protect`) |
//! | `venue_exits` | A live session's own exits: made where a backtest makes them, sent again while they apply; the entries withdrawn when the entry window closes |
//! | `venue_cover` | Reduce-only orders never reverse a position: the room on its closing side, the cover rule at submission (`exit_cover`), cut down or cancelled when the position shrinks |
//! | `venue_checks` | What the fill path could not record, refused at submission with the backtest's codes and errors; the free cash net of the working orders |
//! | `venue_fills` | A live session's fills from its venue (none beyond the position of an order that may not reverse it, nor of one sent to close) |
//! | `venue_external` | A live session's fill with no order: the broker closed or reduced a position by itself |
//! | `venue_state` | A live session's state saved as plain data and restored after a restart; its parts in `venue_state/` (`value`, `requests`, `orders`, `book`, `legs`, `desk`, `risk`, `standing`, `checks`, `error`), the next day's carried state (`carry`) and a stopped session's rebuilt (`stopped`) |
//! | `venue_ends` | Its rejections and cancellations, and the slice every venue result is recorded in |
//! | `venue_answers` | The venue's answers to the cancellations and modifications it was sent |
//! | `venue_clock` | The end of a slice: a session close that waits for the venue to end its orders, the next open reached at the close of the slice before |
//! | `venue_command` | `VenueCommand`: what a live session asks of its venue, for an order or a GTT |
//! | `venue_error` | `VenueError`: a venue result that cannot be recorded, with its kind |
//! | `venue_roles` | What each order of a live session is for: `OrderRole`, `EngineExit`, `RiskExitReason`; an order against the position beyond its closing side's room is an entry (a MARKET one is judged in the room the leg's stop makes for it, and the exit owner sends it), and a modification keeps the role |
//! | `venue_session` | A live session's controls: an exit it asks for and what came of it (`ExitRequest`), the entry locks, the sizing cap, the daily loss measure, the stop where it stands |
//! | `venue_gates` | The entry gates a live session adds at submission: the entry locks, nothing in flight, a position at most the freeze quantity |
//! | `venue_group` | A live session's strategy on an instrument others of its strategies trade: the side gate, market orders only, a closing order's margin rejection (`GroupSide`) |
//! | `venue_gtts` | A live session's GTTs: what it follows of each (`GttTerms`, `GttView`), the place, modify and cancel commands, the queries |
//! | `venue_gtt_sync` | A CNC or NRML leg's stop GTT kept on its leg: placed, cut and moved under the one-minute throttle, cancelled |
//! | `venue_gtt_answers` | The venue's answers about a GTT: placed, refused, modified or not, cancelled or not, ended (`GttEnd`) |
//! | `venue_gtt_child` | A fired GTT's child followed as an order; a child not filled through a base bar cancelled, then the engine's exit |
//! | `venue_gtt_protect` | A CNC or NRML leg protected again: a new stop GTT, or an exit when the price has crossed the stop |
//! | `rearm` | `positional_stops="rearm_daily"`: a CNC or NRML leg's stop held as an MIS leg's DAY stop, placed again at each session's open, or the leg exited there when the open is beyond it |
//! | `expiry` | A derivatives position exited on its positional exit day at the square-off time, before the exchange settles it by delivery; an index future's stop GTT cancelled at the close of its last day |
//! | `pre_open` | A daily or longer strategy's decision before a live session's open, on the latest completed bar of its interval; its orders sent at the open, or at once flagged `late_decision` |
//! | `queries` | `AccountView` and the queries of positions, legs, orders and the account |
//! | `steps` | Steps 1 to 8 |
//! | `fills` | Step 2 fills of queued orders, `execute_fill` |
//! | `fill_model` | Fill models: queued orders at another price of their bar, in step 3 (`fill="next_close"` and the others) |
//! | `delay` | Decided orders held for a number of strategy bars before they can fill (`delay_entries`, `delay_exits`) |
//! | `held_changes` | The strategy's cancellations and modifications held as long as its orders under a delay |
//! | `exit_scale` | Leg rules scaled per trade by given factors (the randomized-exits check) |
//! | `shifted_bars` | The clock's strategy bars when they end earlier (`bar_shift`) |
//! | `fill_bar` | The checks of an order at its fill bar: the position (reduce, reverse), the entry window, no delivery short |
//! | `legs` | Leg rules in steps 3 and 4, the add rule of SPEC-14 |
//! | `reverse` | `reverse=True`: close, then open the rest |
//! | `resting` | Step 3: resting LIMIT, SL and SL-M DAY orders, the stop-limit exit conversion |
//! | `modify` | `modify`: a full replacement of a working order |
//! | `order_fill` | Where and how an order fills; the price of any fill (slippage, limits) |
//! | `market_fill` | The price a backtest fills a MARKET order at, for a bar the caller names, recording nothing (a live session's slippage against the backtest) |
//! | `gtt` | The GTT holding a CNC or NRML position's stop: placement, cancel-then-exit, step-4 sync, renewal |
//! | `gtt_plan` | The stop GTT a positional leg needs (design 20-positional-live P6) and when it is modified (GTT-3) |
//! | `gtt_levels` | A CNC or NRML leg against a bar: its targets as an MIS leg's, its stop by its GTT, in the order of step 3 |
//! | `gtt_fire` | Step 3 for GTTs, the child fill, the engine's exit after an unfilled child |
//! | `gtt_orders` | Strategy GTT orders (`validity="gtt"`), the per-segment order shape |
//! | `accounting` | Slippage, margin and charges of a fill; contract notes at the session close |
//! | `sale_credit` | Delivery shares, paid in full: the free cash that counts them at cost, and the part of their sale credited only on the next trading day (`market::sale_credit`) |
//! | `futures` | Futures series: the series, the switches and their spreads, the roll session, the settlement records |
//! | `futures_plan` | Registering a series and planning its switches from the bars; settlement prices |
//! | `roll` | Step 2: the futures auto-roll, the level shift, the GTTs at a roll |
//! | `settlement` | Step 8: futures expiry settlement and the daily mark to market |
//! | `sizing` | What a sizer sees, cost estimates in its budget, the engine default |
//! | `target` | `submit_target`: the order that takes a position to a target (`target()`) |
//! | `rebalance` | `submit_rebalance`: target orders of a rebalance; its buys fill after the slice's sales |
//! | `initial_risk` | What a trade's opening fill would lose at its entry stop (R-multiples) |
//! | `session` | Intraday times per session, the entry window, the square-off timer |
//! | `submit` | `submit`, `cancel`, the submission checks, DAY validity and market closed |
//! | `aggregate` | Step 6 (daily loss limit, drawdown halt), the step-1 session boundary, the exits they queue |
//! | `entry_rules` | Entry gates at submission (trading state, cooldown, maximum positions), the time exit, the fill hook |
//! | `scale_in` | The scale-in limits at submission: the facts of an add for `rms::scale_in_gate` |
//! | `strategy_risk` | `StrategyRisk`: trading state and the bookkeeping of the strategy and session rules |
//! | `pause` | `Pause`, `Driver`, `run` |
//! | `timers` | `self.at` and `self.every`: registration, when they are due, the timer pause |
//! | `history` | The strategy bars of the whole run, for indicators computed ahead |
//! | `signals` | Condition and array strategies: the Rust driver of 14-strategy-api 4.1 over the stepping API |
//! | `event` | `Event`, `EventStamp`, `Step` |
//! | `excursion` | A trade's MAE and MFE (the running P&L at the prices it saw) and the slippage of a fill |
//! | `excursion_hold` | What holds a live trade's MAE and MFE back: the closing orders the venue may have filled unreported |
//! | `output` | `RunOutput` and its tables |
//! | `error` | `KernelError` |
//!
//! The areas below the engine: `exec` (orders, fills, checks, GTT), `portfolio`
//! (positions, ledger), `rms` (rules, leg state, trading state, vocabulary), `sizing`,
//! `market::{calendar, costs, margin, slippage}`, `data::resample`, `model`.
//!
//! # Phase 1 scope of this skeleton
//!
//! Built: one exchange, any number of instruments on it, intraday or daily strategy
//! intervals on an intraday or daily base, MARKET, LIMIT, SL and SL-M DAY orders
//! (entry, add, reduce, exit by fraction, `reverse=True`, modification), GTT orders,
//! the quantity, tick, freeze, product, entry-window, market-closed and reverse checks,
//! leg rules (stop, targets, every trail form, breakeven, `after_first_target`)
//! engine-held, with a CNC or NRML leg's stop held as a GTT, the square-off, timers
//! (`self.at`, `self.every`), charges on every fill with contract-note rounding, the
//! dated default slippage and margin tables (cash, MIS with leverage on request, long
//! options, futures at the default of D23 or the run's own `margin=`), delivery cash
//! (holdings at their cost, the dated credit for selling them), a CNC buy and
//! sell of one scrip on one day charged as intraday, the sizers (with a strategy's own
//! sizing function, `reinvest` and the size caps), the engine default size, target
//! positions, the initial risk of each trade, the position
//! history, the equity series, trades and events, the whole-run strategy bars for
//! indicators; the daily loss limit, the drawdown halt, the trading state, the
//! cooldown, maximum positions, the scale-in limits and the time exit; condition and
//! array strategies on one instrument (`signals`); futures series with the auto-roll
//! or settlement at expiry, and the daily futures mark to market (`futures`, `roll`,
//! `settlement`). Everything else returns `KernelError::NotBuilt` naming the feature.

mod account_breach;
mod account_link;
#[cfg(test)]
mod account_test_kit;
mod accounting;
mod aggregate;
pub mod clock;
pub mod config;
mod daily_caps;
mod delay;
mod entry_rules;
pub mod error;
pub mod event;
mod excursion;
mod excursion_hold;
mod exit_scale;
mod expiry;
mod fill_bar;
mod fill_model;
mod fills;
mod fund_view;
mod futures;
mod futures_plan;
mod gtt;
mod gtt_fire;
mod gtt_levels;
mod gtt_orders;
mod gtt_plan;
mod held_changes;
mod history;
mod initial_risk;
pub mod kernel;
mod legs;
mod live_clock;
pub mod market_data;
mod market_fill;
mod modify;
mod order_fill;
pub mod output;
pub mod pause;
mod pre_open;
mod queries;
mod rearm;
mod rebalance;
mod resting;
mod reverse;
mod roll;
pub mod run_end;
mod sale_credit;
mod scale_in;
mod session;
mod settlement;
mod shifted_bars;
pub mod signals;
mod sizing;
mod steps;
mod strategy_risk;
pub mod submit;
mod target;
pub mod timers;
mod venue;
mod venue_answers;
mod venue_checks;
mod venue_clock;
mod venue_command;
mod venue_cover;
mod venue_ends;
mod venue_error;
mod venue_exit_orders;
mod venue_exits;
mod venue_external;
mod venue_fills;
mod venue_gates;
mod venue_group;
mod venue_gtt_answers;
mod venue_gtt_child;
mod venue_gtt_protect;
mod venue_gtt_sync;
mod venue_gtts;
mod venue_legs;
mod venue_orders;
mod venue_owner;
mod venue_protect;
mod venue_requests;
mod venue_roles;
mod venue_session;
mod venue_state;
mod venue_stop_terms;
#[cfg(test)]
mod venue_test_kit;
mod venue_ticks;

pub use account_breach::{AccountBreach, AccountRule};
pub use account_link::AccountLink;
pub use clock::{SliceInfo, Timeline};
pub use config::{
    BarColumns, BarPrice, CostConfig, FillModel, FillTiming, IntradayOverrides, Perturbation,
    QuantityMode, RunConfig, RunSetup, StrategyInterval,
};
pub use error::KernelError;
pub use event::{Event, EventStamp, OrderEventKind, Step};
pub(crate) use fund_view::EngineFigures;
pub use futures::{
    roll_session_day, session_after, FuturesOutput, FuturesSeries, RollRule, SeriesContract,
    SettlementKind, SettlementRecord, Switch,
};
pub use kernel::Engine;
pub use market_data::{BaseBar, StrategyBar};
pub use output::{
    EquityColumns, FillRecord, FinalPosition, GrossPnl, LevelRow, RunOutput, StaleMark, TradeRecord,
};
pub use pause::{run, Driver, Pause, TimerId};
pub use pre_open::PreOpen;
pub use queries::AccountView;
pub use run_end::{RunEnd, RunEndCause};
pub use signals::{run_signals, AtrInput, AtrSlot, EntrySignal, SignalArrays, SkippedEntry};
pub use submit::Submission;
pub use target::TargetRequest;
pub use timers::{TimerCount, TimerSpec};
pub use venue::{Venue, VenueCommand};
pub use venue_error::{Pending, VenueError, UNKNOWN_ORDER_KIND, VENUE_REFUSAL_KINDS};
pub use venue_group::GroupSide;
pub use venue_gtt_answers::{GttEnd, GTT_REFUSED_LOCK};
pub use venue_gtts::{GttTerms, GttView};
pub use venue_protect::Protect;
pub use venue_roles::{EngineExit, OrderRole, RiskExitReason};
pub use venue_session::ExitRequest;
pub use venue_state::{CarriedIds, LiveStateError, StateValue, LIVE_STATE_LAYOUT};
