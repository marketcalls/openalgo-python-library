#![cfg(all(feature = "costs", feature = "time"))]
//! `positional_stops="rearm_daily"` in a backtest (design 20-positional-live section 6,
//! decisions P5 and 11.4; unit N4 item N4-1): a CNC or NRML leg's stop held as an MIS
//! leg's engine-held stop, no GTT ever placed, and at a session's first base bar a stop
//! the open is already at or beyond exits the leg at that open plus slippage, flagged
//! `stop_crossed`, reason `leg_sl`.
//!
//! Daily bars of SBIN (NSE, tick 0.05) from Mon 2025-06-30 to Thu 2025-07-03, no charges,
//! slippage one tick on a MARKET fill. A LIMIT buy of 100 at 1,000.00 is decided at the
//! close of 06-30 and fills at the 07-01 open of 1,000.00 (a resting limit takes no
//! slippage), stop 50 points: 950.00.

use oa_bt::engine::{
    run, BarColumns, CostConfig, Engine, FillTiming, IntradayOverrides, Perturbation, QuantityMode,
    RunConfig, RunOutput, RunSetup, StrategyInterval,
};
use oa_bt::exec::{
    FillAt, FillFlags, FillReason, InstrumentId, InstrumentSpec, OrderRequest, OrderType,
    QuantitySpec,
};
use oa_bt::market::calendar::{BaseInterval, Calendar};
use oa_bt::market::facts::MarketFacts;
use oa_bt::market::margin::MarginModel;
use oa_bt::market::slippage::SlippageSpec;
use oa_bt::model::{days_from_civil, ts_from_ist, Day, Exchange, InstrumentKind, Product, Side};
use oa_bt::rms::{Distance, ExitReason, LegRules, PositionalStops, StrategyRiskConfig};
use oa_bt::sizing::SizingControls;

mod venue_support;

use venue_support::{Act, Script};

const OPEN_MINUTE: u32 = 9 * 60 + 15;
const CLOSE_MINUTE: u32 = 15 * 60 + 30;
const STOCK: InstrumentId = InstrumentId(0);

/// One bar: (open, high, low, close), rupees.
type Ohlc = (f64, f64, f64, f64);

/// June 30 is day 0; July 1 to 3 are days 1 to 3.
fn session(day: u32) -> Day {
    days_from_civil(2025, 6, 30) + day as i32
}

fn sbin() -> InstrumentSpec {
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

/// A stock future on NFO, lot 750, whose contract runs past the test's days.
fn sbin_future() -> InstrumentSpec {
    InstrumentSpec {
        symbol: "SBIN31JUL25FUT".to_string(),
        exchange: Exchange::Nfo,
        kind: InstrumentKind::Future,
        lot_size_units: 750,
        last_trading_day: Some(days_from_civil(2025, 7, 31)),
        ..sbin()
    }
}

/// Daily bars of days 0 to 3: `(1000, 1005, 995, 1000)` unless written.
fn bars(written: &[(u32, Ohlc)]) -> BarColumns {
    let mut columns = BarColumns::default();
    for day in 0..4 {
        let (open, high, low, close) = written
            .iter()
            .find(|(d, _)| *d == day)
            .map_or((1000.0, 1005.0, 995.0, 1000.0), |(_, bar)| *bar);
        columns.ts_open.push(ts_from_ist(session(day), OPEN_MINUTE));
        columns.open.push(open);
        columns.high.push(high);
        columns.low.push(low);
        columns.close.push(close);
        columns.volume.push(1_000.0);
    }
    columns
}

fn config(stops: PositionalStops) -> RunConfig {
    RunConfig {
        first_day: session(0),
        last_day: session(3),
        first_slice: None,
        last_slice: None,
        present: None,
        base: BaseInterval::Daily,
        strategy_interval: StrategyInterval::Daily,
        fill: FillTiming::NextOpen,
        quantity_mode: QuantityMode::Venue,
        capital_rupees: 100_000_000.0,
        costs: CostConfig::Off,
        slippage: SlippageSpec::Ticks(1),
        margin: MarginModel,
        market: MarketFacts::default(),
        risk: StrategyRiskConfig {
            positional_stops: stops,
            ..StrategyRiskConfig::default()
        },
        sizing: SizingControls::default(),
        intraday: IntradayOverrides::default(),
        deliver_events: false,
        perturbation: Perturbation::default(),
    }
}

/// A LIMIT order of `units` on `side` in `product` at `limit` with a stop of `points`.
fn entry(side: Side, units: f64, product: Product, limit: f64, points: f64) -> OrderRequest {
    OrderRequest {
        product,
        order_type: OrderType::Limit {
            limit_rupees: limit,
        },
        leg_rules: LegRules {
            stop: Some(Distance::Points(points)),
            ..LegRules::default()
        },
        ..OrderRequest::market(STOCK, side, QuantitySpec::Units(units))
    }
}

/// The backtest of `bars` trading `instrument`, the entry decided at the close of day 0.
fn backtest(
    stops: PositionalStops,
    instrument: InstrumentSpec,
    request: OrderRequest,
    bars: BarColumns,
) -> (RunOutput, usize) {
    let setup = RunSetup {
        config: config(stops),
        calendar: Calendar::new().expect("the embedded calendar tables parse"),
        instruments: vec![instrument],
        bars: vec![bars],
    };
    let mut kernel = Engine::new(setup).expect("a valid backtest");
    let decided = ts_from_ist(session(0), CLOSE_MINUTE);
    let mut script = Script::new(vec![(decided, Act::Submit(request))]);
    run(&mut kernel, &mut script).expect("the backtest completes");
    let gtts = kernel.gtts().len();
    (kernel.finish().expect("the backtest finished"), gtts)
}

fn cnc_long(stops: PositionalStops, written: &[(u32, Ohlc)]) -> (RunOutput, usize) {
    let request = entry(Side::Buy, 100.0, Product::Cnc, 1000.0, 50.0);
    backtest(stops, sbin(), request, bars(written))
}

/// The exit fill of the run's one trade: (price, where, reason, flags).
fn exit_fill(output: &RunOutput) -> (f64, FillAt, FillReason, FillFlags) {
    assert_eq!(output.fills.len(), 2, "{:?}", output.fills);
    let exit = &output.fills[1].fill;
    (exit.price_rupees, exit.at, exit.reason, exit.flags)
}

const STOP_CROSSED: FillFlags = FillFlags::STOP_CROSSED;
const LEG_SL: FillReason = FillReason::Exit(ExitReason::LegSl);

#[test]
fn a_rearm_daily_leg_never_places_a_gtt() {
    let (output, gtts) = cnc_long(PositionalStops::RearmDaily, &[]);
    assert_eq!(gtts, 0);
    assert_eq!(output.fills[0].fill.price_rupees, 1000.0);
    assert_eq!(
        output.trades.len(),
        0,
        "the long is held over the three sessions"
    );
    let (_, gtt_run) = cnc_long(PositionalStops::Gtt, &[]);
    assert_eq!(gtt_run, 1, "the default holds the stop as one GTT");
}

#[test]
fn a_gap_open_beyond_the_stop_exits_at_the_open_plus_slippage_flagged_stop_crossed() {
    // Day 2 opens at 940.00, beyond the 950.00 stop: no stop is placed, the leg is sold at
    // the open, 940.00 less one tick of slippage on a sale: 939.95.
    let gap = [(2, (940.0, 945.0, 930.0, 942.0))];
    let (output, _) = cnc_long(PositionalStops::RearmDaily, &gap);
    let (price, at, reason, flags) = exit_fill(&output);
    assert_eq!((price, at, reason), (939.95, FillAt::Open, LEG_SL));
    assert!(flags.contains(STOP_CROSSED), "{flags:?}");
    assert_eq!(output.fills[1].stamp.slice, 2, "at day 2's open");
}

#[test]
fn the_same_gap_with_a_gtt_stop_gives_the_gtts_child_fill() {
    // The GTT fires on the gap, its child a sale 1 percent below the 950.00 trigger,
    // 940.50; the bar trades back up to 945.00, so the child fills at its limit.
    let gap = [(2, (940.0, 945.0, 930.0, 942.0))];
    let (output, _) = cnc_long(PositionalStops::Gtt, &gap);
    let (price, _, reason, flags) = exit_fill(&output);
    assert_eq!((price, reason), (940.5, LEG_SL));
    assert!(!flags.contains(STOP_CROSSED), "{flags:?}");
}

#[test]
fn a_stop_reached_inside_a_session_fills_as_an_mis_stop_does() {
    // Day 2 opens at 960.00 and trades down to 945.00: the stop fills at its level less a
    // tick, 949.95, not flagged.
    let reached = [(2, (960.0, 965.0, 945.0, 950.0))];
    let (output, _) = cnc_long(PositionalStops::RearmDaily, &reached);
    let (price, at, reason, flags) = exit_fill(&output);
    assert_eq!((price, at, reason), (949.95, FillAt::Level, LEG_SL));
    assert_eq!(flags, FillFlags::NONE);
}

#[test]
fn a_short_futures_stop_is_above_the_price_and_a_gap_up_exits_at_the_open() {
    // An NRML short of one lot at 1,000.00, stop 50 points above: 1,050.00. Day 3 opens at
    // 1,060.00: bought back at the open plus a tick, 1,060.05.
    let request = entry(Side::Sell, 750.0, Product::Nrml, 1000.0, 50.0);
    let gap = [(3, (1060.0, 1070.0, 1055.0, 1065.0))];
    let (output, gtts) = backtest(
        PositionalStops::RearmDaily,
        sbin_future(),
        request,
        bars(&gap),
    );
    assert_eq!(gtts, 0);
    let (price, at, reason, flags) = exit_fill(&output);
    assert_eq!((price, at, reason), (1060.05, FillAt::Open, LEG_SL));
    assert!(flags.contains(STOP_CROSSED));
    assert_eq!(output.fills[1].fill.side, Side::Buy);
}

#[test]
fn an_open_inside_the_stop_changes_nothing_at_the_open() {
    // Day 2 opens at 951.00, above the stop: the stop is in force again and the bar, whose
    // low is 951.00, never reaches it.
    let inside = [(2, (951.0, 960.0, 951.0, 955.0))];
    let (output, _) = cnc_long(PositionalStops::RearmDaily, &inside);
    assert_eq!(output.fills.len(), 1, "{:?}", output.fills);
}
