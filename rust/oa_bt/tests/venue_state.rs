#![cfg(all(feature = "costs", feature = "time"))]
//! A live session's state saved at any point of the day and restored into a session built
//! for a restart continues as the uninterrupted session does (`engine::venue_state`,
//! 16-session-design R14, unit K4).
//!
//! The session trades a scripted day on SBIN's rising bars, driven by ticks (each minute's
//! open, low, high and close) through a venue that lags: it fills a MARKET order sent
//! while the session waits in two halves, at the open of that slice and of the next (so
//! orders are partly filled across slices), fills a stop or a limit on the tick that
//! reaches it, and answers a modification or a cancellation only at the next slice (so
//! modifications and cancellations wait for their answers across slices). The day has a
//! long with a stop, a trail and a target (the trail modifies the broker's stop, the
//! target cancels it before the exit), a short reduced by half (its stop cut first) and
//! stopped out, a resting CNC limit that never fills, and a long closed by `exit()`.
//!
//! The session is run once uninterrupted. Then, for every wait and every bar close of
//! the trading stretch, its state is saved there, restored into a new session that has
//! caught up to the same point with the same bars and decided nothing, and that session
//! runs on with the same script and a copy of the venue as it stood. The restored state
//! equals the saved one, and at every later wait the two sessions' states are equal, the
//! venue receives the same commands, and the run ends with the same fills, trades,
//! positions and P&L from that point on (orders compared by the ids the restore returned;
//! the orders the restored session makes later are numbered after its restored ones).

mod venue_support;

use oa_bt::engine::{
    BarColumns, BaseBar, CostConfig, Driver, Engine, FillRecord, Pause, RunConfig, RunOutput,
    StateValue, TradeRecord, VenueCommand,
};
use oa_bt::exec::{OrderId, OrderRequest, OrderType, QuantitySpec};
use oa_bt::model::{Product, Side};
use oa_bt::rms::{Distance, LegRules, TrailSpec};
use venue_support::{
    at_close, bar_at, bars, charged_config, config, market, minute, sbin, slice_open, Act, Script,
    SBIN, SECOND,
};

/// Seconds after a slice's open of its four ticks: open, low, high, close.
const TICK_SECONDS: [i64; 4] = [0, 15, 30, 45];
/// The session ends at the open of this minute: the trading stretch and some quiet after.
const LAST_MINUTE: u32 = 140;
/// Saved at every wait and bar close before this slice.
const SAVED_BEFORE: usize = 132;

/// Where a session stands when its state is saved: waiting before a slice (before the
/// venue answers in it), or paused at a slice's bar close (after the script acted).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Point {
    Wait(usize),
    Close(usize),
}

impl Point {
    /// The first slice whose results come after this point.
    fn next_slice(self) -> usize {
        match self {
            Point::Wait(slice) => slice,
            Point::Close(slice) => slice + 1,
        }
    }
}

/// An order the venue holds.
#[derive(Clone, Copy, Debug)]
struct Held {
    order: OrderId,
    side: Side,
    order_type: OrderType,
    /// The first slice in which it may fill.
    from_slice: usize,
    /// A MARKET order sent while the session waits: half at the open, the rest later.
    split: bool,
}

/// What the venue answers at the wait before a slice.
#[derive(Clone, Copy, Debug)]
enum Answer {
    Modify(OrderId, OrderType),
    Cancel(OrderId),
}

/// The lagging venue (module docs).
#[derive(Clone, Debug, Default)]
struct Lagging {
    held: Vec<Held>,
    /// (the slice before which it is given, the answer), in the order taken.
    answers: Vec<(usize, Answer)>,
    /// Every command taken, in order.
    commands: Vec<VenueCommand>,
}

impl Lagging {
    /// The venue as a restored session sees it: its orders under their new ids (`ids` maps
    /// a saved id to the new one), what it held for orders that have ended dropped.
    fn renumbered(&self, ids: &dyn Fn(OrderId) -> Option<OrderId>) -> Lagging {
        let held = self
            .held
            .iter()
            .filter_map(|held| {
                Some(Held {
                    order: ids(held.order)?,
                    ..*held
                })
            })
            .collect();
        let answers = self
            .answers
            .iter()
            .filter_map(|(slice, answer)| {
                let answer = match *answer {
                    Answer::Modify(order, terms) => Answer::Modify(ids(order)?, terms),
                    Answer::Cancel(order) => Answer::Cancel(ids(order)?),
                };
                Some((*slice, answer))
            })
            .collect();
        Lagging {
            held,
            answers,
            commands: Vec::new(),
        }
    }

    fn works(kernel: &Engine, order: OrderId) -> bool {
        kernel.order(order).is_some_and(|o| o.status.is_working())
    }

    /// Take the session's commands until it makes no more, in slice `slice`.
    fn take(&mut self, kernel: &mut Engine, slice: usize, waiting: bool) {
        loop {
            let commands = kernel.take_commands();
            if commands.is_empty() {
                return;
            }
            for command in commands {
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
                        split: waiting && order_type == OrderType::Market,
                    }),
                    VenueCommand::Modify {
                        order, order_type, ..
                    } => self
                        .answers
                        .push((slice + 1, Answer::Modify(order, order_type))),
                    VenueCommand::Cancel { order } => {
                        self.answers.push((slice + 1, Answer::Cancel(order)));
                    }
                }
            }
        }
    }

    /// Give the answers due before slice `slice`, in order.
    fn answer(&mut self, kernel: &mut Engine, slice: usize) {
        let (due, later): (Vec<_>, Vec<_>) = std::mem::take(&mut self.answers)
            .into_iter()
            .partition(|(at, _)| *at <= slice);
        self.answers = later;
        for (_, answer) in due {
            match answer {
                Answer::Modify(order, order_type) if Self::works(kernel, order) => {
                    kernel
                        .apply_modify(order)
                        .expect("the venue confirms a modify");
                    if let Some(held) = self.held.iter_mut().find(|h| h.order == order) {
                        held.order_type = order_type;
                    }
                }
                Answer::Cancel(order) if Self::works(kernel, order) => {
                    self.held.retain(|held| held.order != order);
                    kernel
                        .apply_cancel(order)
                        .expect("the venue confirms a cancel");
                }
                _ => {}
            }
        }
    }

    /// The units and price at which `held` fills on a tick at `price`, if it does.
    fn fill_of(
        &self,
        kernel: &Engine,
        held: &Held,
        tick: (usize, f64, bool),
    ) -> Option<(f64, f64)> {
        let (slice, price, at_open) = tick;
        let order = kernel.order(held.order)?;
        let left = order.quantity_units - order.filled_units;
        let cancelling = self
            .answers
            .iter()
            .any(|(_, answer)| matches!(answer, Answer::Cancel(id) if *id == held.order));
        if slice < held.from_slice || left <= 0.0 || cancelling {
            return None;
        }
        match held.order_type {
            OrderType::Market if held.split && order.filled_units == 0.0 && left >= 2.0 => {
                at_open.then_some(((left / 2.0).floor(), price))
            }
            OrderType::Market => Some((left, price)),
            OrderType::StopMarket { trigger_rupees } => {
                let through = match held.side {
                    Side::Sell => price <= trigger_rupees,
                    Side::Buy => price >= trigger_rupees,
                };
                through.then_some((left, if at_open { price } else { trigger_rupees }))
            }
            OrderType::Limit { limit_rupees } => {
                let reached = match held.side {
                    Side::Buy => price <= limit_rupees,
                    Side::Sell => price >= limit_rupees,
                };
                reached.then_some((left, limit_rupees))
            }
            OrderType::StopLimit { .. } => None,
        }
    }

    /// One tick of slice `slice`: the session takes it, then the venue fills what it
    /// reaches, one order at a time in id order.
    fn tick(&mut self, kernel: &mut Engine, (slice, open): (usize, i64), at: i64, price: f64) {
        kernel
            .push_tick(SBIN, price, at)
            .expect("the session takes the tick");
        self.take(kernel, slice, false);
        loop {
            self.held.retain(|held| Self::works(kernel, held.order));
            let mut ready = self.held.clone();
            ready.sort_by_key(|held| held.order);
            let tick = (slice, price, at == open);
            let first = ready
                .iter()
                .find_map(|held| self.fill_of(kernel, held, tick).map(|fill| (*held, fill)));
            let Some((held, (units, fill_price))) = first else {
                return;
            };
            if let Some(known) = self.held.iter_mut().find(|h| h.order == held.order) {
                known.from_slice = slice + 1;
            }
            kernel
                .apply_fill(held.order, units, fill_price, at)
                .expect("the session takes the fill");
            self.take(kernel, slice, false);
        }
    }

    /// The venue's part of the wait before slice `slice`: its answers, the commands, the
    /// slice's ticks, then the slice's bar and its release.
    fn serve(&mut self, kernel: &mut Engine, columns: &BarColumns, slice: usize) {
        self.answer(kernel, slice);
        self.take(kernel, slice, true);
        let open = slice_open(slice);
        if let Some(bar) = bar_at(columns, open) {
            let prices = [bar.open, bar.low, bar.high, bar.close];
            for (seconds, price) in TICK_SECONDS.iter().zip(prices) {
                self.tick(kernel, (slice, open), open + seconds * SECOND, price);
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
        kernel.release(slice + 1);
    }
}

/// The day's script (module docs).
fn script() -> Script {
    let long = OrderRequest {
        leg_rules: LegRules {
            stop: Some(Distance::Points(2.0)),
            target: Some(Distance::Points(6.0)),
            trail: Some(TrailSpec {
                trigger: Distance::Points(1.0),
                step: None,
            }),
            ..LegRules::default()
        },
        ..market(Side::Buy, 100.0)
    };
    let short = OrderRequest {
        leg_rules: LegRules {
            stop: Some(Distance::Points(3.0)),
            ..LegRules::default()
        },
        ..market(Side::Sell, 60.0)
    };
    let close_by = |side: Side, fraction: f64| OrderRequest {
        reduce_only: true,
        ..OrderRequest {
            quantity: QuantitySpec::PositionFraction(fraction),
            ..market(side, 1.0)
        }
    };
    let resting = OrderRequest {
        order_type: OrderType::Limit {
            limit_rupees: 400.0,
        },
        product: Product::Cnc,
        ..market(Side::Buy, 20.0)
    };
    let last = OrderRequest {
        leg_rules: LegRules {
            stop: Some(Distance::Points(1.0)),
            ..LegRules::default()
        },
        ..market(Side::Buy, 40.0)
    };
    Script::new(vec![
        at_close(0, Act::Submit(long)),
        at_close(70, Act::Submit(short)),
        at_close(80, Act::Submit(close_by(Side::Buy, 0.5))),
        at_close(85, Act::Submit(resting)),
        at_close(110, Act::Submit(last)),
        at_close(115, Act::Submit(close_by(Side::Sell, 1.0))),
    ])
}

/// The day's settings: `base`, ending at the open of `LAST_MINUTE`.
fn day_config(base: RunConfig) -> RunConfig {
    RunConfig {
        last_slice: Some(minute(LAST_MINUTE)),
        ..base
    }
}

/// A new session of the day.
fn new_session(config: &RunConfig) -> Engine {
    venue_support::session_of(config.clone(), sbin())
}

/// What is seen at a point: the session's state and the venue as they stand.
#[derive(Clone, Debug)]
struct Seen {
    point: Point,
    state: StateValue,
    venue: Lagging,
    /// Orders the session has made so far.
    orders: u64,
}

/// The orders `kernel` has made so far (ids are dense from o1).
fn order_count(kernel: &Engine) -> u64 {
    (1..)
        .find(|id| kernel.order(OrderId(*id)).is_none())
        .unwrap_or(1)
        - 1
}

/// Run `kernel` to its end with `venue` and `script`, first serving the wait before
/// `serve_first` when the session stands there; `seen` is called at every point.
fn drive(
    kernel: &mut Engine,
    venue: &mut Lagging,
    script: &mut Script,
    serve_first: Option<usize>,
    seen: &mut dyn FnMut(Point, &Engine, &Lagging),
) {
    let columns = bars(&[], &[]);
    if let Some(slice) = serve_first {
        venue.serve(kernel, &columns, slice);
    }
    loop {
        match kernel.advance().expect("the slice runs") {
            Pause::Waiting { next_slice } => {
                seen(Point::Wait(next_slice), kernel, venue);
                venue.serve(kernel, &columns, next_slice);
            }
            Pause::Finished { .. } => return,
            pause => {
                script.on_pause(kernel, &pause).expect("the script acts");
                if let Pause::BarClose { stamp, .. } = pause {
                    seen(Point::Close(stamp.slice as usize), kernel, venue);
                }
            }
        }
    }
}

/// The uninterrupted session: what was seen at every point, the venue at its end and the
/// session's result.
fn uninterrupted(config: &RunConfig) -> (Vec<Seen>, Lagging, RunOutput) {
    let mut kernel = new_session(config);
    let mut venue = Lagging::default();
    let mut points = Vec::new();
    let mut record = |point: Point, kernel: &Engine, venue: &Lagging| {
        points.push(Seen {
            point,
            state: kernel.live_state(),
            venue: venue.clone(),
            orders: order_count(kernel),
        });
    };
    drive(&mut kernel, &mut venue, &mut script(), None, &mut record);
    (
        points,
        venue,
        kernel.finish().expect("the session finished"),
    )
}

/// A new session caught up to `point` with the day's bars, deciding nothing.
fn caught_up(config: &RunConfig, point: Point) -> Engine {
    let columns = bars(&[], &[]);
    let mut kernel = new_session(config);
    loop {
        match kernel.advance().expect("the slice runs") {
            Pause::Waiting { next_slice } if point == Point::Wait(next_slice) => return kernel,
            Pause::BarClose { stamp, .. } if point == Point::Close(stamp.slice as usize) => {
                return kernel;
            }
            Pause::Waiting { next_slice } => {
                if let Some(bar) = bar_at(&columns, slice_open(next_slice)) {
                    let base = BaseBar {
                        ts_open: slice_open(next_slice),
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
            Pause::Finished { .. } => panic!("the session finished before {point:?}"),
            _ => {}
        }
    }
}

/// The ids of a restored session's orders as the saved session has them: a restored
/// order by the pairs the restore returned, a later one `later` after its own.
struct Renumber {
    pairs: Vec<(OrderId, OrderId)>,
    later: u64,
}

impl Renumber {
    fn saved_id(&self, id: u64) -> u64 {
        let restored = self.pairs.iter().find(|(_, new)| new.0 == id);
        restored.map_or(id + self.later, |(saved, _)| saved.0)
    }

    fn state(&self, value: &StateValue) -> StateValue {
        self.walk(value, false)
    }

    /// `value` with every order id renumbered: the values of the keys that name an order
    /// ("order", "strategy_order") and the items of the lists of orders ("sent",
    /// "cancelling", "unconfirmed_close_orders", "pending_close_orders").
    fn walk(&self, value: &StateValue, names_order: bool) -> StateValue {
        match value {
            StateValue::Int(id) if names_order => {
                let id = u64::try_from(*id).expect("an order id");
                StateValue::Int(i64::try_from(self.saved_id(id)).expect("an order id"))
            }
            StateValue::List(items) => StateValue::List(
                items
                    .iter()
                    .map(|item| self.walk(item, names_order))
                    .collect(),
            ),
            StateValue::Map(entries) => StateValue::Map(
                entries
                    .iter()
                    .map(|(key, item)| {
                        let names = matches!(
                            key.as_str(),
                            "order"
                                | "strategy_order"
                                | "sent"
                                | "cancelling"
                                | "unconfirmed_close_orders"
                                | "pending_close_orders"
                        );
                        (key.clone(), self.walk(item, names))
                    })
                    .collect(),
            ),
            other => other.clone(),
        }
    }

    fn command(&self, command: VenueCommand) -> VenueCommand {
        let saved = |order: OrderId| OrderId(self.saved_id(order.0));
        match command {
            VenueCommand::Place {
                order,
                instrument,
                product,
                side,
                units,
                order_type,
                role,
            } => VenueCommand::Place {
                order: saved(order),
                instrument,
                product,
                side,
                units,
                order_type,
                role,
            },
            VenueCommand::Modify {
                order,
                units,
                order_type,
            } => VenueCommand::Modify {
                order: saved(order),
                units,
                order_type,
            },
            VenueCommand::Cancel { order } => VenueCommand::Cancel {
                order: saved(order),
            },
        }
    }

    fn fill(&self, record: &FillRecord) -> FillRecord {
        let mut record = *record;
        record.fill.order = record.fill.order.map(|id| OrderId(self.saved_id(id.0)));
        record
    }
}

/// A fill as the two sessions can both report it: with charges, without its share of the
/// contract note, which each start settles for its own fills (`engine::venue_state`
/// module docs). What each fill posted to the account is the same: the states compare
/// the account's charges up to the session close.
fn comparable_fill(record: &FillRecord, charged: bool) -> FillRecord {
    let mut record = *record;
    if charged {
        record.fill.costs_rupees = 0.0;
    }
    record
}

/// A trade as the two sessions can both report it: without its number (a restored
/// session numbers its trades from 1) and, with charges, without its charges (each
/// start's result charges a trade for its own fills: a trade opened before the restart
/// is charged in the restored session's result for its fills after it).
fn comparable(trade: &TradeRecord, charged: bool) -> TradeRecord {
    let unnumbered = TradeRecord {
        number: 0,
        ..*trade
    };
    if !charged {
        return unnumbered;
    }
    TradeRecord {
        costs_rupees: 0.0,
        net_pnl_rupees: 0.0,
        ..unnumbered
    }
}

/// Save the uninterrupted session at `saved`, restore it into a session caught up to the
/// same point, run that to its end, and check it against the uninterrupted one.
fn check_restart(
    config: &RunConfig,
    saved: &Seen,
    points: &[Seen],
    venue_at_end: &Lagging,
    output: &RunOutput,
) {
    let charged = matches!(config.costs, CostConfig::Model(_));
    let point = saved.point;
    let mut kernel = caught_up(config, point);
    let pairs = kernel
        .restore_live_state(&saved.state)
        .unwrap_or_else(|error| panic!("{point:?}: {error}"));
    let renumber = Renumber {
        later: saved.orders - pairs.len() as u64,
        pairs: pairs.clone(),
    };
    assert_eq!(
        renumber.state(&kernel.live_state()),
        saved.state,
        "{point:?}: the restored state"
    );
    let new_id = |saved_id: OrderId| {
        pairs
            .iter()
            .find(|(old, _)| *old == saved_id)
            .map(|(_, new)| *new)
    };
    let mut venue = saved.venue.renumbered(&new_id);
    let after: Vec<&Seen> = points
        .iter()
        .skip_while(|seen| seen.point != point)
        .skip(1)
        .collect();
    let mut at = 0;
    let mut compare = |seen_at: Point, kernel: &Engine, _: &Lagging| {
        let expected = after
            .get(at)
            .expect("a point the uninterrupted session saw");
        at += 1;
        assert_eq!(seen_at, expected.point, "{point:?}: the points");
        let state = renumber.state(&kernel.live_state());
        assert_eq!(state, expected.state, "{point:?}: the state at {seen_at:?}");
    };
    let serve_first = match point {
        Point::Wait(slice) => Some(slice),
        Point::Close(_) => None,
    };
    drive(
        &mut kernel,
        &mut venue,
        &mut script(),
        serve_first,
        &mut compare,
    );
    let taken_before = saved.venue.commands.len();
    let commands: Vec<VenueCommand> = venue
        .commands
        .iter()
        .map(|command| renumber.command(*command))
        .collect();
    assert_eq!(
        commands,
        venue_at_end.commands[taken_before..],
        "{point:?}: commands"
    );
    let restarted = kernel.finish().expect("the restored session finished");
    let from = point.next_slice() as u32;
    let fills: Vec<FillRecord> = output
        .fills
        .iter()
        .filter(|record| record.stamp.slice >= from)
        .map(|record| comparable_fill(record, charged))
        .collect();
    let restarted_fills: Vec<FillRecord> = restarted
        .fills
        .iter()
        .map(|record| comparable_fill(&renumber.fill(record), charged))
        .collect();
    assert_eq!(restarted_fills, fills, "{point:?}: the fills");
    let trades: Vec<TradeRecord> = output
        .trades
        .iter()
        .filter(|trade| trade.exit_slice >= from)
        .map(|trade| comparable(trade, charged))
        .collect();
    let restarted_trades: Vec<TradeRecord> = restarted
        .trades
        .iter()
        .map(|trade| comparable(trade, charged))
        .collect();
    assert_eq!(restarted_trades, trades, "{point:?}: the trades");
    assert_eq!(
        restarted.final_positions, output.final_positions,
        "{point:?}"
    );
    assert_eq!(restarted.gross_pnl, output.gross_pnl, "{point:?}: the P&L");
}

/// Check a restart from every saved point of the trading stretch.
fn every_restart_continues_as_the_uninterrupted_session(config: RunConfig) {
    let config = day_config(config);
    let (points, venue_at_end, output) = uninterrupted(&config);
    let saved: Vec<&Seen> = points
        .iter()
        .filter(|seen| seen.point.next_slice() <= SAVED_BEFORE)
        .collect();
    assert!(
        saved.len() > 2 * (SAVED_BEFORE - 2),
        "a wait and a bar close per slice"
    );
    for seen in saved {
        check_restart(&config, seen, &points, &venue_at_end, &output);
    }
}

#[test]
fn a_restart_from_any_point_continues_as_the_uninterrupted_session() {
    every_restart_continues_as_the_uninterrupted_session(config());
}

#[test]
fn a_restart_with_charges_continues_with_the_same_cash_until_the_session_close() {
    every_restart_continues_as_the_uninterrupted_session(charged_config());
}

/// The value of `key` in the map `value`.
fn get<'a>(value: &'a StateValue, key: &str) -> &'a StateValue {
    let StateValue::Map(entries) = value else {
        panic!("not a map: {value:?}");
    };
    let found = entries.iter().find(|(known, _)| known == key);
    &found.unwrap_or_else(|| panic!("no {key}")).1
}

/// The items of the list `value`.
fn items(value: &StateValue) -> &[StateValue] {
    let StateValue::List(items) = value else {
        panic!("not a list: {value:?}");
    };
    items
}

#[test]
fn the_saved_points_hold_partial_fills_and_unanswered_modifications_and_cancellations() {
    // What a restore must carry, somewhere in the day this suite saves at.
    let (points, _, output) = uninterrupted(&day_config(config()));
    let any = |found: &dyn Fn(&StateValue) -> bool| points.iter().any(|seen| found(&seen.state));
    let text = |value: &StateValue, expected: &str| *value == StateValue::from(expected);
    let orders = |state: &StateValue| items(get(state, "orders")).to_vec();
    let desk = |state: &StateValue, key: &str| items(get(get(state, "desk"), key)).to_vec();
    assert!(
        any(&|state| orders(state)
            .iter()
            .any(|order| text(get(order, "status"), "PartiallyFilled"))),
        "a partly filled order"
    );
    assert!(
        any(&|state| !desk(state, "modifying").is_empty()),
        "a modification waiting"
    );
    assert!(
        any(&|state| !desk(state, "cancelling").is_empty()),
        "a cancellation waiting"
    );
    assert!(
        any(&|state| desk(state, "exits")
            .iter()
            .any(|exit| items(get(exit, "sent")).is_empty())),
        "an exit waiting for the cancellation it sent"
    );
    assert!(
        any(&|state| !items(get(state, "commands")).is_empty()),
        "a command not taken"
    );
    assert!(
        any(&|state| orders(state)
            .iter()
            .any(|order| text(get(get(order, "request"), "order_type"), "LIMIT"))),
        "a resting order"
    );
    assert!(
        output.trades.len() >= 3,
        "the day's trades: {:?}",
        output.trades
    );
}
