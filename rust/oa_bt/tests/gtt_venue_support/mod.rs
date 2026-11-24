//! Shared harness of the live GTT tests (unit N1): a live session of 2025-07-01 that
//! bought 100 SBIN in CNC with leg rules (o1, decided at the 09:15 close, filled at
//! 500.10 at the 09:16 open), and readers of the GTT commands it sends.
//!
//! Every test file that includes this module uses a different part of it, so unused
//! items are allowed here rather than duplicated per file.
#![allow(dead_code)]

use oa_bt::engine::{Engine, GttTerms, VenueCommand};
use oa_bt::exec::{GttId, InstrumentSpec, OrderId, OrderRequest, OrderType};
use oa_bt::model::{days_from_civil, Exchange, InstrumentKind, Product, Side};
use oa_bt::rms::{Distance, LegRules, TargetSpec, TrailSpec};

use super::venue_support::{
    at_close, bars, config, minute, order, sbin, session, wait_before, Act, Script, SBIN,
};

/// The price the entry fills at, rupees.
pub const ENTRY: f64 = 500.1;
/// The units bought.
pub const UNITS: f64 = 100.0;
/// The leg's stop GTT, the first of the session.
pub const G1: GttId = GttId(1);
/// The entry order.
pub const BUY: OrderId = OrderId(1);

pub fn points(value: f64) -> Distance {
    Distance::Points(value)
}

/// A stop 4 points below the entry: 496.10.
pub fn stop_4() -> LegRules {
    LegRules {
        stop: Some(points(4.0)),
        ..LegRules::default()
    }
}

/// `stop_4` with a target 6 points above the entry: 506.10.
pub fn stop_and_target() -> LegRules {
    LegRules {
        target: Some(points(6.0)),
        ..stop_4()
    }
}

/// `stop_4` with half the units at 6 points (506.10) and half at 10 (510.10).
pub fn stop_and_two_targets() -> LegRules {
    let half = |distance| TargetSpec {
        distance: points(distance),
        fraction: 0.5,
    };
    LegRules {
        targets: vec![half(6.0), half(10.0)],
        ..stop_4()
    }
}

/// `stop_4` with a continuous trail of 2 points.
pub fn stop_and_trail() -> LegRules {
    LegRules {
        trail: Some(TrailSpec {
            trigger: points(2.0),
            step: None,
        }),
        ..stop_4()
    }
}

/// A CNC buy of `units` SBIN at market with `rules`.
pub fn cnc_buy(units: f64, rules: LegRules) -> OrderRequest {
    OrderRequest {
        leg_rules: rules,
        ..order(Side::Buy, units, OrderType::Market, Product::Cnc)
    }
}

/// A live session that bought 100 SBIN in CNC with `rules` (module docs), waiting before
/// minute 1 with the buy's command taken; the commands its fill made are left.
pub fn cnc_long(rules: LegRules) -> Engine {
    let mut kernel = session(config());
    let acts = vec![at_close(0, Act::Submit(cnc_buy(UNITS, rules)))];
    wait_before(&mut kernel, &bars(&[], &[]), &mut Script::new(acts), 1);
    kernel.take_commands();
    kernel
        .apply_fill(BUY, UNITS, ENTRY, minute(1))
        .expect("the buy fills");
    kernel
}

/// `cnc_long` with its stop GTT's place confirmed and no command left.
pub fn cnc_long_protected(rules: LegRules) -> Engine {
    let mut kernel = cnc_long(rules);
    kernel.take_commands();
    kernel.apply_gtt_placed(G1).expect("the venue confirms g1");
    kernel.take_commands();
    kernel
}

/// A NIFTY future: NFO, lot 75, tick 0.05.
pub fn nifty_future() -> InstrumentSpec {
    InstrumentSpec {
        symbol: "NIFTY31JUL25FUT".to_string(),
        exchange: Exchange::Nfo,
        kind: InstrumentKind::Future,
        lot_size_units: 75,
        last_trading_day: Some(days_from_civil(2025, 7, 31)),
        ..sbin()
    }
}

/// The terms of a CNC long's stop GTT of `units` at `trigger` with its child at `limit`.
pub fn long_stop(units: f64, trigger: f64, limit: f64) -> GttTerms {
    GttTerms {
        instrument: SBIN,
        product: Product::Cnc,
        side: Side::Sell,
        units,
        trigger_rupees: trigger,
        limit_rupees: limit,
        trigger_below: true,
    }
}

/// The GTT commands among `commands`, in order.
pub fn gtt_commands(commands: &[VenueCommand]) -> Vec<VenueCommand> {
    commands
        .iter()
        .filter(|command| command.gtt().is_some())
        .copied()
        .collect()
}

/// The commands for orders among `commands`, in order.
pub fn order_commands(commands: &[VenueCommand]) -> Vec<VenueCommand> {
    commands
        .iter()
        .filter(|command| command.order().is_some())
        .copied()
        .collect()
}

/// `seconds` after the open of minute `k`, UTC nanoseconds.
pub fn at(k: u32, seconds: i64) -> i64 {
    minute(k) + seconds * 1_000_000_000
}
