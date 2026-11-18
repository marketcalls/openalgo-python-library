#![cfg(all(feature = "costs", feature = "time"))]
//! What the robustness reruns change in a run (design 17, sections 2.7, 2.12, 2.14 and
//! 2.15): fill models, decided orders held for a delay, strategy bars that end earlier,
//! and leg rules scaled per trade. Every expected price and instant is worked by hand
//! from the bars written in each test (NSE, 2025-07-01, no charges, no slippage).

mod order_support;

use oa_bt::engine::{
    run, BarPrice, Driver, Engine, FillModel, FillTiming, KernelError, Pause, RunConfig,
    StrategyInterval,
};
use oa_bt::exec::{FillAt, OrderType};
use oa_bt::market::calendar::BaseInterval;
use oa_bt::model::{ts_from_ist, Product, Side};
use oa_bt::rms::{Distance, LegRules};
use order_support::{
    engine, equity, exit_all, fills, july_2025, minute, minute_bars, minute_config, order,
    run_script, Act, Ohlc, MINUTE, NSE_OPEN_MINUTE,
};

fn buy(units: f64) -> oa_bt::exec::OrderRequest {
    order(Side::Buy, units, OrderType::Market, Product::Cnc)
}

/// Bars whose minute k opens at 100 + k, high open + 2, low open - 1, close open + 1:
/// minute 1 is (101, 103, 100, 102), minute 2 (102, 104, 101, 103).
fn rising(count: u32) -> Vec<(u32, Ohlc)> {
    (0..count)
        .map(|k| {
            let open = 100.0 + f64::from(k);
            (k, (open, open + 2.0, open - 1.0, open + 1.0))
        })
        .collect()
}

fn with_fill(fill: FillTiming) -> RunConfig {
    RunConfig {
        fill,
        ..minute_config()
    }
}

fn model(price: BarPrice, two_bars: bool) -> FillTiming {
    FillTiming::Model(FillModel { price, two_bars })
}

/// The price of the one fill of a buy decided at the close of minute 0 under `fill`.
fn entry_price(fill: FillTiming) -> (f64, FillAt) {
    let bars = minute_bars(10, 100.0, &rising(10));
    let kernel = engine(with_fill(fill), equity(), bars);
    let (output, _) = run_script(kernel, vec![(minute(1), Act::Submit(buy(10.0)))]);
    let rows = fills(&output);
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(rows[0].0, minute(1), "it fills in minute 1's slice");
    (rows[0].4, rows[0].5)
}

#[test]
fn each_fill_model_prices_the_order_at_its_price_of_the_next_bar() {
    // Minute 1 is (101, 103, 100, 102), minute 2 (102, 104, 101, 103).
    let rows = [
        (FillTiming::NextOpen, 101.0, FillAt::Open),
        (model(BarPrice::Close, false), 102.0, FillAt::Level),
        (model(BarPrice::Mid, false), 101.5, FillAt::Level),
        // (101 + 103 + 100 + 102) / 4 = 101.5
        (model(BarPrice::Ohlc4, false), 101.5, FillAt::Level),
        // (102 + 103) / 2
        (model(BarPrice::Close, true), 102.5, FillAt::Level),
        // ((103 + 100) / 2 + (104 + 101) / 2) / 2 = (101.5 + 102.5) / 2
        (model(BarPrice::Mid, true), 102.0, FillAt::Level),
        // (101.5 + (102 + 104 + 101 + 103) / 4) / 2 = (101.5 + 102.5) / 2
        (model(BarPrice::Ohlc4, true), 102.0, FillAt::Level),
    ];
    for (fill, price, at) in rows {
        assert_eq!(entry_price(fill), (price, at), "{fill:?}");
    }
}

#[test]
fn a_model_price_off_the_tick_grid_is_rounded_away_from_the_trader() {
    // Minute 1 of (101, 103.05, 100, 102): its midpoint 101.525 is rounded up to 101.55
    // for a buyer.
    let mut bars = rising(10);
    bars[1] = (1, (101.0, 103.05, 100.0, 102.0));
    let kernel = engine(
        with_fill(model(BarPrice::Mid, false)),
        equity(),
        minute_bars(10, 100.0, &bars),
    );
    let (output, _) = run_script(kernel, vec![(minute(1), Act::Submit(buy(10.0)))]);
    assert_eq!(fills(&output)[0].4, 101.55);
}

#[test]
fn a_stop_hit_inside_the_fill_bar_acts_before_a_model_fill_of_the_exit() {
    // At the next close: the buy decided at minute 0's close fills at minute 1's close
    // (102), its stop 2 points below at 100. The exit decided at minute 2's close would
    // fill at minute 3's close (104), but minute 3's low (98) reaches the stop first: the
    // leg's stop exits at 100 and the exit has nothing left to close.
    let mut bars = rising(10);
    bars[3] = (3, (103.0, 105.0, 98.0, 104.0));
    let entry = oa_bt::exec::OrderRequest {
        leg_rules: LegRules {
            stop: Some(Distance::Points(2.0)),
            ..LegRules::default()
        },
        product: Product::Mis,
        ..buy(10.0)
    };
    let kernel = engine(
        with_fill(model(BarPrice::Close, false)),
        equity(),
        minute_bars(10, 100.0, &bars),
    );
    let acts = vec![
        (minute(1), Act::Submit(entry)),
        (minute(3), Act::Submit(exit_all(Side::Buy, Product::Mis))),
    ];
    let (output, _) = run_script(kernel, acts);
    let rows = fills(&output);
    assert_eq!(rows.len(), 2, "{rows:?}");
    assert_eq!((rows[0].0, rows[0].4), (minute(1), 102.0));
    assert_eq!(
        (rows[1].0, rows[1].4, rows[1].6),
        (minute(3), 100.0, "leg_sl")
    );
}

fn delayed(entries: u32, exits: u32) -> RunConfig {
    let mut config = minute_config();
    config.perturbation.delay_entries = entries;
    config.perturbation.delay_exits = exits;
    config
}

#[test]
fn a_held_entry_fills_at_the_open_after_its_delay_ends() {
    // Decided at minute 0's close (the open of minute 1); held 2 one-minute bars, to the
    // close of minute 2: it fills at minute 3's open, 103.
    let kernel = engine(delayed(2, 0), equity(), minute_bars(10, 100.0, &rising(10)));
    let (output, _) = run_script(kernel, vec![(minute(1), Act::Submit(buy(10.0)))]);
    let rows = fills(&output);
    assert_eq!((rows[0].0, rows[0].4), (minute(3), 103.0));
}

#[test]
fn entries_and_exits_are_held_apart() {
    // Entries on time, exits held one bar: the buy decided at minute 0's close fills at
    // minute 1's open (101); the exit decided at minute 4's close fills at minute 6's
    // open (106), not minute 5's.
    let kernel = engine(delayed(0, 1), equity(), minute_bars(10, 100.0, &rising(10)));
    let acts = vec![
        (minute(1), Act::Submit(buy(10.0))),
        (minute(5), Act::Submit(exit_all(Side::Buy, Product::Cnc))),
    ];
    let (output, _) = run_script(kernel, acts);
    let rows = fills(&output);
    assert_eq!((rows[0].0, rows[0].4), (minute(1), 101.0));
    assert_eq!((rows[1].0, rows[1].4), (minute(6), 106.0));
}

#[test]
fn a_legs_stop_is_never_held() {
    // Exits held 3 bars; the stop 2 points below the 101 entry (99) is the engine's and
    // fills when minute 2's low reaches it.
    let mut bars = rising(10);
    bars[2] = (2, (102.0, 103.0, 98.0, 102.5));
    let entry = oa_bt::exec::OrderRequest {
        leg_rules: LegRules {
            stop: Some(Distance::Points(2.0)),
            ..LegRules::default()
        },
        product: Product::Mis,
        ..buy(10.0)
    };
    let kernel = engine(delayed(0, 3), equity(), minute_bars(10, 100.0, &bars));
    let (output, _) = run_script(kernel, vec![(minute(1), Act::Submit(entry))]);
    let rows = fills(&output);
    assert_eq!(
        (rows[1].0, rows[1].4, rows[1].6),
        (minute(2), 99.0, "leg_sl")
    );
}

#[test]
fn an_intraday_order_held_past_its_sessions_close_is_refused() {
    // The session's last one-minute bar closes at 15:30; an entry decided at 15:29 (the
    // close of minute 373) and held one bar would wait to 15:30: market closed.
    let kernel = engine(delayed(1, 0), equity(), minute_bars(375, 100.0, &[]));
    let decided = ts_from_ist(july_2025(1), NSE_OPEN_MINUTE + 374);
    let (output, script) = run_script(kernel, vec![(decided, Act::Submit(buy(1.0)))]);
    assert!(fills(&output).is_empty());
    assert!(matches!(
        script.submissions[0],
        oa_bt::engine::Submission::Rejected(_, code) if code.code() == "market_closed"
    ));
}

/// Records every decision instant and the strategy bar each pause hands over.
#[derive(Default)]
struct Closes {
    decided: Vec<i64>,
    bars: Vec<(i64, i64, f64, f64, f64, f64)>,
}

impl Driver for Closes {
    fn on_pause(&mut self, _engine: &mut Engine, pause: &Pause) -> Result<(), KernelError> {
        if let Pause::BarClose {
            decided_at, bars, ..
        } = pause
        {
            self.decided.push(*decided_at);
            for bar in bars {
                let row = (
                    bar.ts_open,
                    bar.ts_close,
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                );
                self.bars.push(row);
            }
        }
        Ok(())
    }
}

fn five_minute(shift: u32) -> RunConfig {
    let mut config = minute_config();
    config.strategy_interval = StrategyInterval::Intraday(5 * MINUTE);
    config.perturbation.bar_shift = shift;
    config
}

fn closes_of(config: RunConfig, bars: oa_bt::engine::BarColumns) -> Closes {
    let mut kernel = engine(config, equity(), bars);
    let mut closes = Closes::default();
    run(&mut kernel, &mut closes).expect("the run completes");
    closes
}

#[test]
fn five_minute_bars_ending_one_minute_earlier() {
    // 15 one-minute bars from 09:15. Shifted by one base bar, the bars end at 09:19,
    // 09:24 and 09:29: the first holds minutes 0 to 3 and opens at 09:14, before the
    // session, the next minutes 4 to 8.
    let closes = closes_of(five_minute(1), minute_bars(15, 100.0, &rising(15)));
    let at = |m: u32| ts_from_ist(july_2025(1), NSE_OPEN_MINUTE + m);
    assert_eq!(&closes.decided[..3], &[at(4), at(9), at(14)]);
    // Minutes 0 to 3: open 100, high 103 + 2 = 105, low 99, close 104.
    assert_eq!(
        closes.bars[0],
        (at(0) - MINUTE, at(4), 100.0, 105.0, 99.0, 104.0)
    );
    // Minutes 4 to 8: open 104, high 110, low 103, close 109.
    assert_eq!(closes.bars[1], (at(4), at(9), 104.0, 110.0, 103.0, 109.0));
}

#[test]
fn without_a_shift_five_minute_bars_end_on_the_lattice() {
    let closes = closes_of(five_minute(0), minute_bars(15, 100.0, &rising(15)));
    let at = |m: u32| ts_from_ist(july_2025(1), NSE_OPEN_MINUTE + m);
    assert_eq!(&closes.decided[..3], &[at(5), at(10), at(15)]);
}

#[test]
fn a_shift_needs_strategy_bars_of_several_base_bars() {
    let one_minute = minute_config();
    let mut shifted = one_minute.clone();
    shifted.perturbation.bar_shift = 1;
    let setup = oa_bt::engine::RunSetup {
        config: shifted,
        calendar: oa_bt::market::calendar::Calendar::new().expect("calendar"),
        instruments: vec![equity()],
        bars: vec![minute_bars(5, 100.0, &[])],
    };
    let refused = Engine::new(setup).err().map(|error| error.to_string());
    assert!(refused.is_some_and(|text| text.contains("bar_shift")));
    let mut too_far = five_minute(5);
    too_far.base = BaseInterval::Intraday(MINUTE);
    let setup = oa_bt::engine::RunSetup {
        config: too_far,
        calendar: oa_bt::market::calendar::Calendar::new().expect("calendar"),
        instruments: vec![equity()],
        bars: vec![minute_bars(5, 100.0, &[])],
    };
    assert!(Engine::new(setup).is_err());
}

/// One-minute bars of the 2025-07-01 and 2025-07-02 sessions (375 each), every bar at
/// the price of its session's day number (1 or 2) times 100 plus its minute / 1000.
fn two_sessions() -> oa_bt::engine::BarColumns {
    let mut columns = oa_bt::engine::BarColumns::default();
    for (n, day) in [(1.0, july_2025(1)), (2.0, july_2025(2))] {
        for k in 0..375u32 {
            let price = 100.0 * n + f64::from(k) / 1000.0;
            columns.ts_open.push(ts_from_ist(day, NSE_OPEN_MINUTE + k));
            columns.open.push(price);
            columns.high.push(price);
            columns.low.push(price);
            columns.close.push(price);
            columns.volume.push(1.0);
        }
    }
    columns
}

#[test]
fn a_daily_strategy_whose_days_end_fifteen_minutes_earlier() {
    // Days end at 15:15: the first bar closes at 15:15 on 07-01 (minutes 0 to 359 of the
    // run), the second at 15:15 on 07-02, from 15:15 on 07-01 (07-01's last 15 minutes
    // and 07-02's first 360). It opens at 100.36, its close is 200.359.
    let mut config = minute_config();
    config.last_day = july_2025(2);
    config.strategy_interval = StrategyInterval::Daily;
    config.perturbation.bar_shift = 15;
    let closes = closes_of(config, two_sessions());
    let at = |day, m| ts_from_ist(july_2025(day), m);
    assert_eq!(
        closes.decided,
        vec![at(1, 15 * 60 + 15), at(2, 15 * 60 + 15)]
    );
    let second = closes.bars[1];
    assert_eq!(
        (second.0, second.1),
        (at(1, 15 * 60 + 15), at(2, 15 * 60 + 15))
    );
    assert_eq!((second.2, second.5), (100.36, 200.359));
}

#[test]
fn a_trades_exit_distances_take_its_factor() {
    // A stop 2 points below the 101 entry, scaled by 1.5: 3 points, at 98. Minute 2's low
    // of 98.5 does not reach it; minute 3's low of 97 does.
    let mut bars = rising(10);
    bars[2] = (2, (102.0, 103.0, 98.5, 102.5));
    bars[3] = (3, (102.0, 103.0, 97.0, 102.5));
    let entry = oa_bt::exec::OrderRequest {
        leg_rules: LegRules {
            stop: Some(Distance::Points(2.0)),
            ..LegRules::default()
        },
        product: Product::Mis,
        ..buy(10.0)
    };
    let mut config = minute_config();
    config.perturbation.exit_scales = vec![1.5];
    let kernel = engine(config, equity(), minute_bars(10, 100.0, &bars));
    let (output, _) = run_script(kernel, vec![(minute(1), Act::Submit(entry))]);
    let rows = fills(&output);
    assert_eq!(
        (rows[1].0, rows[1].4, rows[1].6),
        (minute(3), 98.0, "leg_sl")
    );
    let stop = output
        .levels
        .iter()
        .find_map(|row| row.stop.map(|s| s.level_rupees));
    assert_eq!(stop, Some(98.0));
}

#[test]
fn an_unusable_exit_factor_is_refused() {
    let mut config = minute_config();
    config.perturbation.exit_scales = vec![1.0, 0.0];
    let setup = oa_bt::engine::RunSetup {
        config,
        calendar: oa_bt::market::calendar::Calendar::new().expect("calendar"),
        instruments: vec![equity()],
        bars: vec![minute_bars(5, 100.0, &[])],
    };
    let refused = Engine::new(setup).err().map(|error| error.to_string());
    assert!(refused.is_some_and(|text| text.contains("exit_scales")));
}
