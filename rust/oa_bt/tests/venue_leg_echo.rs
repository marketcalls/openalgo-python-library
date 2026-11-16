#![cfg(all(feature = "costs", feature = "time"))]
//! A live session driven by recorded ticks, whose venue fills every order where the
//! backtest's simulator does, trades as the backtest of the bars those ticks make, for
//! the leg rules: a stop the broker holds, targets and a trail the kernel holds on ticks
//! (`engine::venue_legs`, `venue_ticks`; 16-session-design section 8, unit K3).
//!
//! The session's ticks are each minute's open, low, high and close, 15 s apart (the bar
//! the session builds from them is the backtest's bar), so a long's adverse price comes
//! before its favourable one, as a backtest takes a bar that reaches the stop and a
//! target: the stop wins. The echo venue below answers as the simulator would:
//!
//! - a MARKET order fills at the first tick it meets: one sent while the session waits for
//!   a slice at its open, one made on a tick (a target's exit) at that tick's price;
//! - a stop-market order fills on the first tick at or through its trigger: at the open
//!   when the slice opens through it, else at its trigger;
//! - a modification is confirmed at once, its new terms in force from the next slice (the
//!   backtest's ratchet takes effect from the next slice), a cancellation at once.
//!
//! What a session records differently is how the exits are made, not what they trade: a
//! leg's stop and target are orders with their own events in a session and engine-held
//! fills with none in a backtest, and a target's MARKET exit is stamped in step 2. So the
//! equality asked of the session is of what was traded: the trades, the fills (slice,
//! units, price, where in the bar, reason, target, charges), the equity series, the
//! position history, the charges, the final positions and the P&L, with the default
//! charges and no slippage.

mod venue_support;

use oa_bt::engine::{
    BaseBar, Driver, Engine, Event, FinalPosition, OrderEventKind, Pause, RunConfig, RunOutput,
    VenueCommand,
};
use oa_bt::exec::{FillAt, FillReason, OrderId, OrderRequest, OrderType};

/// The broker-held stop: the order the fill that opens the leg places, after the buy (o1).
const STOP: OrderId = OrderId(2);
use oa_bt::market::slippage::SlippageSpec;
use oa_bt::model::Side;
use oa_bt::rms::{BreakevenSpec, Distance, ExitReason, LegRules, TargetSpec, TrailSpec};
use venue_support::{
    at_close, bar_at, bars, charged_config, market, rising, run_backtest, session, slice_open,
    stop_points, Act, Ohlc, Script, SBIN, SECOND,
};

/// Seconds after a slice's open of its four ticks: open, low, high, close.
const TICK_SECONDS: [i64; 4] = [0, 15, 30, 45];

/// An order the echo venue holds.
#[derive(Clone, Copy, Debug)]
struct Held {
    order: OrderId,
    side: Side,
    order_type: OrderType,
    /// The first slice in which it may fill.
    from_slice: usize,
    /// Terms a confirmed modification gives it, and the slice they take effect from.
    next: Option<(OrderType, usize)>,
}

/// The echo venue on ticks (module docs).
#[derive(Debug, Default)]
struct TickEcho {
    held: Vec<Held>,
    /// Every command the session sent, in order.
    commands: Vec<VenueCommand>,
}

impl TickEcho {
    /// Take the session's commands until it makes no more, in slice `slice`.
    fn take(&mut self, kernel: &mut Engine, slice: usize) {
        loop {
            let commands = kernel.take_commands();
            if commands.is_empty() {
                return;
            }
            for command in commands {
                self.answer(kernel, command, slice);
            }
        }
    }

    fn answer(&mut self, kernel: &mut Engine, command: VenueCommand, slice: usize) {
        self.commands.push(command);
        match command {
            VenueCommand::Place {
                order,
                side,
                order_type,
                ..
            } => self.held.push(Held {
                order,
                side,
                order_type,
                from_slice: slice,
                next: None,
            }),
            VenueCommand::Modify {
                order, order_type, ..
            } => {
                kernel
                    .apply_modify(order)
                    .expect("the venue confirms a modify");
                if let Some(held) = self.held.iter_mut().find(|h| h.order == order) {
                    held.next = Some((order_type, slice + 1));
                }
            }
            VenueCommand::Cancel { order } => {
                self.held.retain(|held| held.order != order);
                if kernel.order(order).is_some_and(|o| o.status.is_working()) {
                    kernel
                        .apply_cancel(order)
                        .expect("the venue confirms a cancel");
                }
            }
        }
    }

    /// The terms modifications gave take effect from `slice`.
    fn reach(&mut self, slice: usize) {
        for held in self.held.iter_mut() {
            if let Some((order_type, from)) = held.next {
                if from <= slice {
                    held.order_type = order_type;
                    held.next = None;
                }
            }
        }
    }

    /// One tick of slice `slice` opening at `open`: the session takes it, then the venue
    /// fills what it reaches (module docs), one order at a time.
    fn tick(&mut self, kernel: &mut Engine, (slice, open): (usize, i64), at: i64, price: f64) {
        kernel
            .push_tick(SBIN, price, at)
            .expect("the session takes the tick");
        self.take(kernel, slice);
        loop {
            self.held.retain(|held| {
                kernel
                    .order(held.order)
                    .is_some_and(|order| order.status.is_working())
            });
            let mut ready: Vec<Held> = self
                .held
                .iter()
                .copied()
                .filter(|held| held.from_slice <= slice)
                .collect();
            ready.sort_by_key(|held| held.order);
            let first = ready
                .into_iter()
                .find_map(|held| fill_price(&held, price, at == open).map(|p| (held, p)));
            let Some((held, fill)) = first else {
                return;
            };
            self.held.retain(|h| h.order != held.order);
            let order = kernel.order(held.order).expect("a held order is known");
            let units = order.quantity_units - order.filled_units;
            kernel
                .apply_fill(held.order, units, fill, at)
                .expect("the session takes the fill");
            self.take(kernel, slice);
        }
    }
}

/// Where the simulator fills `held` on a tick at `price`, if it does (module docs).
fn fill_price(held: &Held, price: f64, at_open: bool) -> Option<f64> {
    match held.order_type {
        OrderType::Market => Some(price),
        OrderType::StopMarket { trigger_rupees } => {
            let through = match held.side {
                Side::Sell => price <= trigger_rupees,
                Side::Buy => price >= trigger_rupees,
            };
            through.then_some(if at_open { price } else { trigger_rupees })
        }
        _ => None,
    }
}

/// The ticks of a bar: open, low, high, close, at `TICK_SECONDS` after `open_ns`.
fn ticks((open, high, low, close): Ohlc, open_ns: i64) -> [(i64, f64); 4] {
    let prices = [open, low, high, close];
    let mut ticks = [(0, 0.0); 4];
    for (slot, (seconds, price)) in TICK_SECONDS.iter().zip(prices).enumerate() {
        ticks[slot] = (open_ns + seconds * SECOND, price);
    }
    ticks
}

/// The session on the ticks of `overrides`' bars, with the echo venue, to its end.
fn run_on_ticks(config: RunConfig, overrides: &[(u32, Ohlc)], script: &mut Script) -> RunOutput {
    let columns = bars(overrides, &[]);
    let mut kernel = session(config);
    let mut echo = TickEcho::default();
    loop {
        match kernel.advance().expect("the slice runs") {
            Pause::Waiting { next_slice } => {
                let open = slice_open(next_slice);
                echo.reach(next_slice);
                echo.take(&mut kernel, next_slice);
                if let Some(bar) = bar_at(&columns, open) {
                    let ohlc = (bar.open, bar.high, bar.low, bar.close);
                    for (at, price) in ticks(ohlc, open) {
                        echo.tick(&mut kernel, (next_slice, open), at, price);
                    }
                    let base = BaseBar {
                        ts_open: open,
                        open: bar.open,
                        high: bar.high,
                        low: bar.low,
                        close: bar.close,
                        volume: 1_000.0,
                    };
                    kernel.add_bar(SBIN, base).expect("a bar of a closed slice");
                }
                kernel.release(next_slice + 1);
            }
            Pause::Finished { .. } => break,
            pause => script
                .on_pause(&mut kernel, &pause)
                .expect("the script acts"),
        }
    }
    kernel.finish().expect("the session finished")
}

/// What a fill traded: (slice, units, price, where, reason, target, charges).
type Traded = (u32, f64, f64, FillAt, FillReason, Option<u16>, f64);

fn traded(output: &RunOutput) -> Vec<Traded> {
    output
        .fills
        .iter()
        .map(|record| {
            let fill = record.fill;
            (
                record.stamp.slice,
                fill.quantity_units,
                fill.price_rupees,
                fill.at,
                fill.reason,
                fill.target_index,
                fill.costs_rupees,
            )
        })
        .collect()
}

/// The session and the backtest of a buy of 100 with `rules` at the close of minute 0, on
/// the rising bars with `overrides`; asserts they traded the same and returns the session.
fn same_trades(rules: LegRules, overrides: &[(u32, Ohlc)]) -> RunOutput {
    let config = RunConfig {
        slippage: SlippageSpec::Zero,
        ..charged_config()
    };
    let buy = OrderRequest {
        leg_rules: rules,
        ..market(Side::Buy, 100.0)
    };
    let acts = vec![at_close(0, Act::Submit(buy))];
    let columns = bars(overrides, &[]);
    let backtest = run_backtest(config.clone(), &columns, &mut Script::new(acts.clone()));
    let session = run_on_ticks(config, overrides, &mut Script::new(acts));
    assert_eq!(traded(&session), traded(&backtest), "the fills");
    assert_eq!(session.trades, backtest.trades, "the trades");
    assert_eq!(session.equity, backtest.equity, "the equity");
    assert_eq!(session.levels, backtest.levels, "the position history");
    assert_eq!(session.charges, backtest.charges, "the charges");
    assert_eq!(held(&session), held(&backtest), "the final positions");
    assert_eq!(session.gross_pnl, backtest.gross_pnl);
    session
}

/// The final positions, without the last mark of a flat one: a session marks a position
/// on each tick (step 5 of tick mode), so a position that went flat keeps the price of
/// the tick before its exit, where a backtest keeps the close of its last open bar.
fn held(output: &RunOutput) -> Vec<FinalPosition> {
    output
        .final_positions
        .iter()
        .map(|position| FinalPosition {
            mark_rupees: (position.quantity_units != 0.0)
                .then_some(position.mark_rupees)
                .flatten(),
            ..*position
        })
        .collect()
}

/// The exit of the session's one trade: (exit slice, exit price, reason).
fn exit_of(output: &RunOutput) -> (u32, f64, FillReason) {
    let trade = output.trades.last().expect("a trade");
    (trade.exit_slice, trade.exit_price_rupees, trade.exit_reason)
}

fn leg_sl() -> FillReason {
    FillReason::Exit(ExitReason::LegSl)
}

/// A row of the stop table: what, the bars overridden, and the exit (slice, price).
type StopRow = (&'static str, Vec<(u32, Ohlc)>, (u32, f64));

#[test]
fn a_stop_the_broker_holds_fills_where_the_backtest_fills_the_legs_stop() {
    let rows: [StopRow; 2] = [
        (
            "reached inside a bar: at the stop",
            vec![(6, (500.6, 501.0, 496.0, 496.5))],
            (6, 497.1),
        ),
        (
            "gapped through at the open: at the open",
            vec![(6, (495.0, 495.5, 494.0, 495.2))],
            (6, 495.0),
        ),
    ];
    for (what, overrides, (slice, price)) in rows {
        let session = same_trades(stop_points(3.0), &overrides);
        assert_eq!(exit_of(&session), (slice, price, leg_sl()), "{what}");
    }
}

#[test]
fn targets_held_on_ticks_exit_where_the_backtest_fills_them_beside_the_brokers_stop() {
    // The highs of the rising bars reach 502.10 in minute 16 and 504.10 in minute 36: the
    // first target exits half at 502.10 after the stop is cut to the other half, the
    // second cancels the stop and exits the rest at 504.10.
    let target = |points| TargetSpec {
        distance: Distance::Points(points),
        fraction: 0.5,
    };
    let rules = LegRules {
        targets: vec![target(2.0), target(4.0)],
        ..stop_points(3.0)
    };
    let session = same_trades(rules, &[]);
    let exits: Vec<(u32, f64, f64, Option<u16>)> = traded(&session)
        .into_iter()
        .skip(1)
        .map(|(slice, units, price, _, _, target, _)| (slice, units, price, target))
        .collect();
    assert_eq!(
        exits,
        [(16, 50.0, 502.1, Some(0)), (36, 50.0, 504.1, Some(1))]
    );
}

/// Minutes 11 to 15 rise 2.00 a bar, each opening at 2 points above 501.00 for each minute
/// past 10 and reaching 2 above its open; minute 16 falls through the trailed stop.
fn strong_rise_then_fall() -> Vec<(u32, Ohlc)> {
    let mut overrides: Vec<(u32, Ohlc)> = (11..=15)
        .map(|k| {
            let open = 501.0 + 2.0 * f64::from(k - 10);
            (k, (open, open + 2.0, open - 0.2, open + 1.9))
        })
        .collect();
    overrides.push((16, (512.0, 512.2, 508.0, 508.5)));
    overrides
}

#[test]
fn a_trail_the_kernel_moves_on_ticks_fills_the_brokers_stop_where_the_backtest_does() {
    // 2 points behind the best price: 511.00 after minute 15's high of 513.00, in force
    // from minute 16, which falls to 508.00.
    let rules = LegRules {
        trail: Some(TrailSpec {
            trigger: Distance::Points(2.0),
            step: None,
        }),
        ..stop_points(3.0)
    };
    let session = same_trades(rules, &strong_rise_then_fall());
    assert_eq!(exit_of(&session), (16, 511.0, leg_sl()));
    // Minute 11 moves it twice (its open arms the trail at 501.00, its high takes it to
    // 503.00), minutes 12 to 15 once each at their highs: every move is at least 2.00, above
    // a quarter percent of the trigger, and 15 s or more after the one before.
    assert_eq!(brokers_stop(&session), (511.0, 6));
}

/// The trigger the broker's stop (o2) filled at, and how many modifications of it the
/// venue confirmed.
fn brokers_stop(output: &RunOutput) -> (f64, usize) {
    let stop = output.orders.iter().find(|order| order.id == STOP);
    let trigger = stop.and_then(|order| order.request.order_type.trigger_rupees());
    (trigger.unwrap_or(f64::NAN), modifications(output))
}

/// The modifications of the broker's stop the venue confirmed (`Modified` events).
fn modifications(output: &RunOutput) -> usize {
    output
        .events
        .iter()
        .filter(|event| {
            matches!(
                event,
                Event::Order {
                    order: STOP,
                    kind: OrderEventKind::Modified,
                    ..
                }
            )
        })
        .count()
}

#[test]
fn a_breakeven_on_ticks_moves_the_brokers_stop_where_the_backtest_moves_the_legs() {
    // Armed at 502.10 by minute 16's high: the stop moves to the entry, 500.10, which
    // minute 20 falls through.
    let rules = LegRules {
        breakeven: Some(BreakevenSpec {
            trigger: Distance::Points(2.0),
            offset: None,
        }),
        ..stop_points(3.0)
    };
    let fall = (20, (rising(20).0, 502.3, 499.0, 499.5));
    let session = same_trades(rules, &[fall]);
    assert_eq!(exit_of(&session), (20, 500.1, leg_sl()));
    assert_eq!(brokers_stop(&session), (500.1, 1));
}
