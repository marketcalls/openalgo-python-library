//! Shared harness of the venue tests: one NSE session in 1-minute bars, a scripted
//! strategy, and an "echo" venue that answers every order of a live session the way the
//! backtest's simulator fills it, so that a session run with it is the backtest.
//!
//! The echo venue keeps the orders sent to it and, in each slice before the engine runs
//! it, fills what the simulator fills in that slice (03-architecture 4.2): MARKET orders
//! at the slice's open (step 2), then LIMIT, SL-M and SL orders where the bar reaches
//! them (`exec::matching`, step 3), each in id order, none before the slice after its
//! decision (SPEC-35) and none in a slice without a bar (SPEC-06). Slippage is zero, so
//! the simulator's price is the matched price itself. A fill at the open is reported at
//! the slice's open instant, one inside the bar one nanosecond after it. It confirms
//! every modification and cancellation as soon as it takes them, and every GTT's place,
//! modification and cancellation too (it fires no GTT: the GTT tests script that).
//!
//! Every test file that includes this module uses a different part of it, so unused
//! items are allowed here rather than duplicated per file.
#![allow(dead_code)]

use oa_bt::engine::{
    run, BarColumns, BaseBar, CostConfig, Driver, Engine, Event, FillTiming, IntradayOverrides,
    KernelError, OrderEventKind, OrderRole, Pause, Perturbation, QuantityMode, RunConfig,
    RunOutput, RunSetup, StrategyInterval, Submission, VenueCommand,
};
use oa_bt::exec::matching::{limit_after_trigger, limit_point, trigger_point};
use oa_bt::exec::{
    BarPoint, FillAt, FillPricing, InstrumentId, InstrumentSpec, OrderId, OrderRequest, OrderType,
    QuantitySpec, RejectCode, TriggerDirection,
};
use oa_bt::market::calendar::{BaseInterval, Calendar};
use oa_bt::market::costs::CostModel;
use oa_bt::market::facts::MarketFacts;
use oa_bt::market::margin::MarginModel;
use oa_bt::market::slippage::{slipped_price, SlippageSpec};
use oa_bt::model::{days_from_civil, ts_from_ist, Day, Exchange, InstrumentKind, Product, Side};
use oa_bt::rms::{BarPrices, Distance, LegRules, StrategyRiskConfig};
use oa_bt::sizing::SizingControls;

pub const MINUTE: i64 = 60_000_000_000;
/// 09:15 IST, the NSE session open, in minutes since midnight.
pub const NSE_OPEN_MINUTE: u32 = 9 * 60 + 15;
/// Slices of the 2025-07-01 NSE session on a 1-minute grid (09:15 to 15:30).
pub const SESSION_SLICES: u32 = 375;
pub const CAPITAL: f64 = 10_000_000.0;
pub const SBIN: InstrumentId = InstrumentId(0);

/// One bar: (open, high, low, close), rupees.
pub type Ohlc = (f64, f64, f64, f64);

pub fn day() -> Day {
    days_from_civil(2025, 7, 1)
}

/// The open of minute `k` of the session (k = 0 opens at 09:15), UTC nanoseconds.
pub fn minute(k: u32) -> i64 {
    ts_from_ist(day(), NSE_OPEN_MINUTE + k)
}

/// The decision instant of the strategy bar of minute `k`: its close.
pub fn close_of(k: u32) -> i64 {
    minute(k + 1)
}

/// The bar of minute `k` on a steady rise: open 500.00 + 0.10 k, high the open plus
/// 0.50, low the open less 0.50, close the open plus 0.05.
pub fn rising(k: u32) -> Ohlc {
    let open = 500.0 + 0.10 * f64::from(k);
    (open, open + 0.50, open - 0.50, open + 0.05)
}

/// Bars of the session's minutes from 09:15: `rising` unless overridden, none for the
/// minutes in `missing`.
pub fn bars(overrides: &[(u32, Ohlc)], missing: &[u32]) -> BarColumns {
    let mut columns = BarColumns::default();
    for k in (0..SESSION_SLICES).filter(|k| !missing.contains(k)) {
        let (open, high, low, close) = overrides
            .iter()
            .find(|(m, _)| *m == k)
            .map_or_else(|| rising(k), |(_, bar)| *bar);
        columns.ts_open.push(minute(k));
        columns.open.push(open);
        columns.high.push(high);
        columns.low.push(low);
        columns.close.push(close);
        columns.volume.push(1_000.0);
    }
    columns
}

pub fn sbin() -> InstrumentSpec {
    InstrumentSpec {
        symbol: "SBIN".to_string(),
        exchange: Exchange::Nse,
        kind: InstrumentKind::Equity,
        lot_size_units: 1,
        tick_size_rupees: 0.05,
        multiplier: 1.0,
        freeze_quantity_units: None,
        cost_segment: None,
        last_trading_day: None,
        positional_exit_day: None,
    }
}

/// An NFO call option: lot 75, tick 0.05.
pub fn nifty_call() -> InstrumentSpec {
    InstrumentSpec {
        symbol: "NIFTY31JUL2525000CE".to_string(),
        exchange: Exchange::Nfo,
        kind: InstrumentKind::Option,
        lot_size_units: 75,
        last_trading_day: Some(days_from_civil(2025, 7, 31)),
        ..sbin()
    }
}

/// SBIN with a freeze quantity of `units`.
pub fn frozen_at(units: u32) -> InstrumentSpec {
    InstrumentSpec {
        freeze_quantity_units: Some(units),
        ..sbin()
    }
}

/// A 1-minute strategy on 1-minute bars on 2025-07-01, with no costs and no slippage.
pub fn config() -> RunConfig {
    RunConfig {
        first_day: day(),
        last_day: day(),
        first_slice: None,
        last_slice: None,
        present: None,
        base: BaseInterval::Intraday(MINUTE),
        strategy_interval: StrategyInterval::Intraday(MINUTE),
        fill: FillTiming::NextOpen,
        quantity_mode: QuantityMode::Venue,
        capital_rupees: CAPITAL,
        costs: CostConfig::Off,
        slippage: SlippageSpec::Zero,
        margin: MarginModel,
        market: MarketFacts::default(),
        risk: StrategyRiskConfig::default(),
        sizing: SizingControls::default(),
        intraday: IntradayOverrides::default(),
        deliver_events: false,
        perturbation: Perturbation::default(),
    }
}

/// `config` with the built-in charges of the default broker.
pub fn charged_config() -> RunConfig {
    let model = CostModel::builtin().expect("the built-in schedules parse");
    RunConfig {
        costs: CostConfig::Model(model),
        ..config()
    }
}

fn setup(config: RunConfig, bars: BarColumns) -> RunSetup {
    RunSetup {
        config,
        calendar: Calendar::new().expect("the embedded calendar tables parse"),
        instruments: vec![sbin()],
        bars: vec![bars],
    }
}

/// A backtest of `bars`.
pub fn backtest(config: RunConfig, bars: BarColumns) -> Engine {
    Engine::new(setup(config, bars)).expect("a valid backtest")
}

/// A live session of the test day, built before its bars.
pub fn session(config: RunConfig) -> Engine {
    Engine::new_session(setup(config, BarColumns::default())).expect("a valid session")
}

/// A live session of the test day trading `instrument`.
pub fn session_of(config: RunConfig, instrument: InstrumentSpec) -> Engine {
    let setup = RunSetup {
        instruments: vec![instrument],
        ..setup(config, BarColumns::default())
    };
    Engine::new_session(setup).expect("a valid session")
}

/// A request for `units` of SBIN on `side` in `product`, priced as `order_type`.
pub fn order(side: Side, units: f64, order_type: OrderType, product: Product) -> OrderRequest {
    OrderRequest {
        order_type,
        product,
        ..OrderRequest::market(SBIN, side, QuantitySpec::Units(units))
    }
}

/// An MIS MARKET order.
pub fn market(side: Side, units: f64) -> OrderRequest {
    order(side, units, OrderType::Market, Product::Mis)
}

/// An MIS order that only reduces: `units` on `side`, priced as `order_type`.
pub fn exit(side: Side, units: f64, order_type: OrderType) -> OrderRequest {
    OrderRequest {
        reduce_only: true,
        ..order(side, units, order_type, Product::Mis)
    }
}

pub fn limit(limit_rupees: f64) -> OrderType {
    OrderType::Limit { limit_rupees }
}

pub fn stop_market(trigger_rupees: f64) -> OrderType {
    OrderType::StopMarket { trigger_rupees }
}

pub fn stop_limit(trigger_rupees: f64, limit_rupees: f64) -> OrderType {
    OrderType::StopLimit {
        trigger_rupees,
        limit_rupees,
    }
}

/// What the script does at a decision instant.
#[derive(Clone, Debug)]
pub enum Act {
    Submit(OrderRequest),
    Modify(OrderId, OrderRequest),
    Cancel(OrderId),
}

/// A driver that acts at given decision instants (bar closes and timer opens, UTC
/// nanoseconds) and keeps what each call returned.
#[derive(Clone, Debug)]
pub struct Script {
    acts: Vec<(i64, Act)>,
    pub submissions: Vec<Submission>,
}

impl Script {
    pub fn new(acts: Vec<(i64, Act)>) -> Script {
        Script {
            acts,
            submissions: Vec::new(),
        }
    }
}

impl Driver for Script {
    fn on_pause(&mut self, engine: &mut Engine, pause: &Pause) -> Result<(), KernelError> {
        let decided_at = match pause {
            Pause::BarClose { decided_at, .. } | Pause::Timers { decided_at, .. } => *decided_at,
            Pause::Finished { .. } | Pause::Waiting { .. } => return Ok(()),
        };
        let due: Vec<Act> = self
            .acts
            .iter()
            .filter(|(instant, _)| *instant == decided_at)
            .map(|(_, act)| act.clone())
            .collect();
        for act in due {
            match act {
                Act::Submit(request) => self.submissions.push(engine.submit(request)?),
                Act::Modify(id, request) => self.submissions.push(engine.modify(id, request)?),
                Act::Cancel(id) => {
                    engine.cancel(id)?;
                }
            }
        }
        Ok(())
    }
}

/// Run a backtest of `bars` with `script` to its end.
pub fn run_backtest(config: RunConfig, bars: &BarColumns, script: &mut Script) -> RunOutput {
    let mut kernel = backtest(config, bars.clone());
    run(&mut kernel, script).expect("the backtest completes");
    kernel.finish().expect("the backtest finished")
}

/// A backtest's output as a live session reports the same fills: its venue priced them,
/// so no fill and no trade has slippage modelled (`engine::excursion`), where the
/// backtest's fills with zero slippage record 0.
pub fn venue_priced(mut output: RunOutput) -> RunOutput {
    for record in &mut output.fills {
        record.slippage_rupees = None;
    }
    for trade in &mut output.trades {
        trade.slippage_rupees = None;
    }
    output
}

/// An order the echo venue holds.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Held {
    order: OrderId,
    side: Side,
    units: f64,
    order_type: OrderType,
    /// The first slice in which it may fill.
    from_slice: usize,
}

/// The echo venue (module docs). `rejects` names orders it rejects instead of filling.
#[derive(Clone, Debug, Default)]
pub struct Echo {
    held: Vec<Held>,
    /// Every command the session sent, in order.
    pub commands: Vec<VenueCommand>,
    /// Orders the venue rejects when it would fill them.
    pub rejects: Vec<OrderId>,
}

/// The bar of the slice opening at `open`, if `bars` has one.
pub fn bar_at(bars: &BarColumns, open: i64) -> Option<BarPrices> {
    let row = bars.ts_open.iter().position(|&ts| ts == open)?;
    Some(BarPrices {
        open: bars.open[row],
        high: bars.high[row],
        low: bars.low[row],
        close: bars.close[row],
    })
}

fn base_bar(bars: &BarColumns, open: i64) -> Option<BaseBar> {
    let row = bars.ts_open.iter().position(|&ts| ts == open)?;
    Some(BaseBar {
        ts_open: open,
        open: bars.open[row],
        high: bars.high[row],
        low: bars.low[row],
        close: bars.close[row],
        volume: bars.volume[row],
    })
}

/// Where the simulator fills an order in `bar` and how it prices it, if it fills.
fn match_in_bar(held: &Held, bar: &BarPrices) -> Option<(BarPoint, FillPricing)> {
    match held.order_type {
        OrderType::Market => {
            let open = BarPoint {
                at: FillAt::Open,
                price_rupees: bar.open,
            };
            Some((open, FillPricing::Market))
        }
        OrderType::Limit { limit_rupees } => {
            limit_point(held.side, limit_rupees, bar).map(|point| (point, FillPricing::Resting))
        }
        OrderType::StopMarket { trigger_rupees } => {
            let direction = TriggerDirection::of_stop(held.side);
            trigger_point(direction, trigger_rupees, bar).map(|point| (point, FillPricing::Market))
        }
        OrderType::StopLimit {
            trigger_rupees,
            limit_rupees,
        } => {
            let direction = TriggerDirection::of_stop(held.side);
            let fired = trigger_point(direction, trigger_rupees, bar)?;
            limit_after_trigger(held.side, limit_rupees, fired, bar).fill(limit_rupees)
        }
    }
}

/// The simulator's price of a fill referenced at `reference_rupees` with no slippage: a
/// fill that takes the market is put on the tick grid, a resting limit is not.
fn simulated_price(side: Side, reference_rupees: f64, pricing: FillPricing) -> f64 {
    if !pricing.takes_slippage() {
        return reference_rupees;
    }
    let tick = sbin().tick_size_rupees;
    let slipped = slipped_price(SlippageSpec::Zero, side, reference_rupees, tick)
        .expect("a reference price on the grid");
    pricing.price(side, reference_rupees, slipped)
}

impl Echo {
    /// An echo venue that rejects `orders` when it would fill them.
    pub fn rejecting(orders: impl IntoIterator<Item = OrderId>) -> Echo {
        Echo {
            rejects: orders.into_iter().collect(),
            ..Echo::default()
        }
    }

    /// Take the session's commands, until it makes no more: what they place or replace
    /// may fill from `slice`; a modification and a cancellation are confirmed at once.
    pub fn take(&mut self, engine: &mut Engine, slice: usize) {
        loop {
            let commands = engine.take_commands();
            if commands.is_empty() {
                return;
            }
            for command in commands {
                self.answer(engine, command, slice);
            }
        }
    }

    fn answer(&mut self, engine: &mut Engine, command: VenueCommand, slice: usize) {
        self.commands.push(command);
        match command {
            VenueCommand::Place {
                order,
                side,
                units,
                order_type,
                ..
            } => self.held.push(Held {
                order,
                side,
                units,
                order_type,
                from_slice: slice,
            }),
            VenueCommand::Modify {
                order,
                units,
                order_type,
            } => {
                engine
                    .apply_modify(order)
                    .expect("the venue confirms a modify");
                if let Some(held) = self.held.iter_mut().find(|h| h.order == order) {
                    *held = Held {
                        units,
                        order_type,
                        from_slice: slice,
                        ..*held
                    };
                }
            }
            VenueCommand::Cancel { order } => {
                self.held.retain(|held| held.order != order);
                engine
                    .apply_cancel(order)
                    .expect("the venue confirms a cancel");
            }
            VenueCommand::PlaceGtt { gtt, .. } => {
                engine
                    .apply_gtt_placed(gtt)
                    .expect("the venue confirms a GTT's place");
            }
            VenueCommand::ModifyGtt { gtt, .. } => {
                engine
                    .apply_gtt_modified(gtt)
                    .expect("the venue confirms a GTT's modify");
            }
            VenueCommand::CancelGtt { gtt, .. } => {
                engine
                    .apply_gtt_cancelled(gtt)
                    .expect("the venue confirms a GTT's cancel");
            }
        }
    }

    /// Fill what the simulator fills in slice `slice` opening at `open` (module docs).
    pub fn fill(&mut self, engine: &mut Engine, slice: usize, open: i64, bar: Option<BarPrices>) {
        self.held.retain(|held| {
            engine
                .order(held.order)
                .is_some_and(|order| order.status.is_working())
        });
        let Some(bar) = bar else {
            return;
        };
        let due = |held: &Held, market: bool| {
            held.from_slice <= slice && (held.order_type == OrderType::Market) == market
        };
        for market in [true, false] {
            let mut ready: Vec<Held> = self
                .held
                .iter()
                .copied()
                .filter(|h| due(h, market))
                .collect();
            ready.sort_by_key(|held| held.order);
            for held in ready {
                let Some((point, pricing)) = match_in_bar(&held, &bar) else {
                    continue;
                };
                self.held.retain(|h| h.order != held.order);
                if self.rejects.contains(&held.order) {
                    engine
                        .apply_reject(held.order, RejectCode::Venue)
                        .expect("the session takes the rejection");
                    continue;
                }
                let at = if point.at == FillAt::Open {
                    open
                } else {
                    open + 1
                };
                let price = simulated_price(held.side, point.price_rupees, pricing);
                engine
                    .apply_fill(held.order, held.units, price, at)
                    .expect("the session takes the fill");
            }
        }
    }
}

/// Run a live session on `bars` with `script` and the echo venue to its end: at each
/// wait the venue takes the commands and fills the slice, then the slice's bar is added
/// and the slice released. The venue answers only while the session waits, as a real
/// one's answers arrive between slices.
pub fn run_echo(
    config: RunConfig,
    bars: &BarColumns,
    script: &mut Script,
    echo: &mut Echo,
) -> RunOutput {
    drive_echo(session(config), bars, script, echo)
}

/// `run_echo` on a session already built (and given its timers).
pub fn drive_echo(
    mut kernel: Engine,
    bars: &BarColumns,
    script: &mut Script,
    echo: &mut Echo,
) -> RunOutput {
    echo_until(&mut kernel, bars, script, echo, usize::MAX);
    kernel.finish().expect("the session finished")
}

/// Run an echoed session (as `run_echo`) until it waits before slice `until`, or to its
/// end. The commands made before that wait are left for the test.
pub fn echo_until(
    kernel: &mut Engine,
    bars: &BarColumns,
    script: &mut Script,
    echo: &mut Echo,
    until: usize,
) {
    loop {
        match kernel.advance().expect("the slice runs") {
            Pause::Waiting { next_slice } if next_slice >= until => return,
            Pause::Waiting { next_slice } => {
                let open = slice_open(next_slice);
                echo.take(kernel, next_slice);
                echo.fill(kernel, next_slice, open, bar_at(bars, open));
                if let Some(bar) = base_bar(bars, open) {
                    kernel.add_bar(SBIN, bar).expect("a bar of a closed slice");
                }
                kernel.release(next_slice + 1);
            }
            Pause::Finished { .. } => return,
            pause => script.on_pause(kernel, &pause).expect("the script acts"),
        }
    }
}

/// The open of slice `index` of the session's 1-minute grid.
pub fn slice_open(index: usize) -> i64 {
    minute(u32::try_from(index).expect("a slice of the session"))
}

/// An act decided at the close of minute `k`.
pub fn at_close(k: u32, act: Act) -> (i64, Act) {
    (close_of(k), act)
}

/// Advance a session, adding `bars` and releasing each slice but filling nothing, until
/// it waits before slice `slice`. The venue's commands are left for the test.
pub fn wait_before(kernel: &mut Engine, bars: &BarColumns, script: &mut Script, slice: usize) {
    loop {
        match kernel.advance().expect("the slice runs") {
            Pause::Waiting { next_slice } if next_slice >= slice => return,
            Pause::Waiting { next_slice } => {
                if let Some(bar) = base_bar(bars, slice_open(next_slice)) {
                    kernel.add_bar(SBIN, bar).expect("a bar of a closed slice");
                }
                kernel.release(next_slice + 1);
            }
            Pause::Finished { .. } => panic!("the session finished before slice {slice}"),
            pause => script.on_pause(kernel, &pause).expect("the script acts"),
        }
    }
}

/// Advance a session (as `wait_before`) to its end, the venue confirming the cancels
/// of the session close; it is finished but not consumed.
pub fn run_to_end(kernel: &mut Engine, bars: &BarColumns, script: &mut Script) {
    loop {
        match kernel.advance().expect("the slice runs") {
            Pause::Waiting { next_slice } if next_slice >= SESSION_SLICES as usize => {
                for command in kernel.take_commands() {
                    let VenueCommand::Cancel { order } = command else {
                        continue;
                    };
                    if kernel.order(order).is_some_and(|o| o.status.is_working()) {
                        kernel
                            .apply_cancel(order)
                            .expect("the venue confirms a cancel");
                    }
                }
            }
            Pause::Waiting { next_slice } => {
                if let Some(bar) = base_bar(bars, slice_open(next_slice)) {
                    kernel.add_bar(SBIN, bar).expect("a bar of a closed slice");
                }
                kernel.release(next_slice + 1);
            }
            Pause::Finished { .. } => return,
            pause => script.on_pause(kernel, &pause).expect("the script acts"),
        }
    }
}

/// Advance a session (as `wait_before`) to its next bar-close pause, before slice
/// `slice` has run: the engine is left paused at step 7.
pub fn pause_at_close(kernel: &mut Engine, bars: &BarColumns, slice: usize) {
    loop {
        match kernel.advance().expect("the slice runs") {
            Pause::BarClose { stamp, .. } if stamp.slice as usize >= slice => return,
            Pause::Waiting { next_slice } => {
                if let Some(bar) = base_bar(bars, slice_open(next_slice)) {
                    kernel.add_bar(SBIN, bar).expect("a bar of a closed slice");
                }
                kernel.release(next_slice + 1);
            }
            Pause::Finished { .. } => panic!("the session finished before slice {slice}"),
            _ => {}
        }
    }
}

/// `kernel` holding `units` of SBIN in MIS, bought by a MARKET order decided at the close
/// of minute 0 (o1) and filled by the venue at 500.10 at the open of minute 1, then paused
/// at the close of minute 2 with no command left to take.
pub fn holding_on(mut kernel: Engine, units: f64) -> Engine {
    let acts = vec![at_close(0, Act::Submit(market(Side::Buy, units)))];
    wait_before(&mut kernel, &bars(&[], &[]), &mut Script::new(acts), 1);
    kernel
        .apply_fill(OrderId(1), units, 500.1, minute(1))
        .expect("the buy fills");
    pause_at_close(&mut kernel, &bars(&[], &[]), 2);
    kernel.take_commands();
    kernel
}

/// The (order, side, units, role) of the orders placed by `commands`.
pub fn placed_roles(commands: &[VenueCommand]) -> Vec<(OrderId, Side, f64, OrderRole)> {
    commands
        .iter()
        .filter_map(|command| match *command {
            VenueCommand::Place {
                order,
                side,
                units,
                role,
                ..
            } => Some((order, side, units, role)),
            _ => None,
        })
        .collect()
}

/// One event as (slice open, step number, name).
pub fn events(output: &RunOutput) -> Vec<(i64, u8, String)> {
    output
        .events
        .iter()
        .map(|event| {
            let stamp = event.stamp();
            let name = match event {
                Event::Order {
                    kind: OrderEventKind::Rejected(code),
                    order,
                    ..
                } => format!("{order} Rejected {}", code.code()),
                Event::Order { kind, order, .. } => format!("{order} {}", kind.code()),
                Event::Position { change, .. } => change.code().to_string(),
                Event::Risk { event, .. } => event.name.code().to_string(),
                Event::TradingState { from, to, .. } => {
                    format!("state {} -> {}", from.code(), to.code())
                }
                Event::Gtt { event, .. } => format!("{} {}", event.name.code(), event.gtt),
            };
            (stamp.slice_ts, stamp.step.number(), name)
        })
        .collect()
}

/// One second, UTC nanoseconds.
pub const SECOND: i64 = 1_000_000_000;

/// `seconds` after the open of minute `k`, UTC nanoseconds.
pub fn at_second(k: u32, seconds: i64) -> i64 {
    minute(k) + seconds * SECOND
}

/// A stop `points` from the leg's average price, on its losing side.
pub fn stop_points(points: f64) -> LegRules {
    LegRules {
        stop: Some(Distance::Points(points)),
        ..LegRules::default()
    }
}

/// A session trading `instrument` that bought `units` of it with `rules` (o1, an MIS
/// MARKET order decided at the close of minute 0), waiting before minute 1 with the buy's
/// command taken and the buy not filled.
pub fn entry_waiting(instrument: InstrumentSpec, units: f64, rules: LegRules) -> Engine {
    let mut kernel = session_of(config(), instrument);
    let request = OrderRequest {
        leg_rules: rules,
        ..market(Side::Buy, units)
    };
    let acts = vec![at_close(0, Act::Submit(request))];
    wait_before(&mut kernel, &bars(&[], &[]), &mut Script::new(acts), 1);
    kernel.take_commands();
    kernel
}

/// `entry_waiting` on SBIN with the buy filled whole at 500.10 at the open of minute 1;
/// the commands the fill made are left for the test.
pub fn holding_with(units: f64, rules: LegRules) -> Engine {
    let mut kernel = entry_waiting(sbin(), units, rules);
    kernel
        .apply_fill(OrderId(1), units, 500.1, minute(1))
        .expect("the buy fills");
    kernel
}

/// The (order, units, order type) of the modifications in `commands`.
pub fn modified(commands: &[VenueCommand]) -> Vec<(OrderId, f64, OrderType)> {
    commands
        .iter()
        .filter_map(|command| match *command {
            VenueCommand::Modify {
                order,
                units,
                order_type,
            } => Some((order, units, order_type)),
            _ => None,
        })
        .collect()
}
