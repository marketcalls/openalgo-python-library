//! Pre-trade checks, the same in backtest and live (03-architecture 4.4), and the one
//! typed set of codes their rejections carry.
//!
//! A check that refuses an order produces a `Rejected` order event with a
//! `RejectCode`, never an error. Checks run at submission (step 7 of the decision
//! slice, against the decision instant) and again at the fill bar (step 2, against the
//! bar's open), as the entry window requires (decision SPEC-04).
//!
//! Market closed (lead ruling): a DAY order is rejected with `market_closed` when the
//! fill rule of SPEC-35 would place its fill in a later session than the one it was
//! decided in. That covers the last-bar signal of an intraday strategy. Orders of daily
//! and longer strategies are valid for the next session; GTT orders persist.
//!
//! Built: the quantity rule, tick validity of limit and trigger prices, the freeze
//! quantity (deny), the product rules (no delivery short, no trading an index, no GTT
//! on MIS), the entry window and market closed; the trading-state gate (daily loss
//! gate, halted), cooldown, maximum positions and the scale-in limits are risk rules
//! (`rms::TradingLock`, `rms::entry_gates`, `rms::scale_in`). Not built yet: margin at
//! submission and the order rate limit.

use crate::model::{Product, Side, TsNanos};

use super::instrument::InstrumentSpec;
use super::order::{OrderType, Validity};
use super::price::is_on_tick;

/// Why a check refused an order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RejectCode {
    /// Not a whole number of lots, or a fractional quantity where the venue rule
    /// applies.
    QuantityRule = 0,
    /// A limit or trigger price off the tick grid.
    Tick = 1,
    /// Above the exchange's freeze quantity (deny is the default).
    FreezeQty = 2,
    /// A product rule: a delivery (CNC) sale that would open or increase a short, a GTT
    /// on MIS, an order for an index.
    ProductRule = 3,
    /// Not enough free cash for the margin the order needs.
    Margin = 4,
    /// An intraday entry outside `entry_from` to `entry_until` (SPEC-04).
    EntryWindow = 5,
    /// The fill would fall in a later session than the decision (lead ruling).
    MarketClosed = 6,
    /// An entry placed after the day's loss reached the daily loss limit (SPEC-18).
    DailyLossGate = 7,
    /// An entry inside the cooldown after an exit.
    Cooldown = 8,
    /// An entry beyond the maximum number of open positions.
    MaxPositions = 9,
    /// The trading state is REDUCING or HALTED and the order would open or add.
    Halted = 10,
    /// The order rate limit (live).
    RateLimit = 11,
    /// The order names an instrument the run does not know.
    InstrumentUnknown = 12,
    /// A reducing order larger than the opposite position, without `reverse=True`
    /// (14-strategy-api 2.4).
    ReverseNotAllowed = 13,
    /// An add beyond a scale-in limit, or against the position without
    /// `allow_averaging_down=True` (12-position-sizing 2, `rms::scale_in`).
    ScaleInLimit = 14,
}

impl RejectCode {
    /// Every code, in code order.
    pub const ALL: [RejectCode; 15] = [
        RejectCode::QuantityRule,
        RejectCode::Tick,
        RejectCode::FreezeQty,
        RejectCode::ProductRule,
        RejectCode::Margin,
        RejectCode::EntryWindow,
        RejectCode::MarketClosed,
        RejectCode::DailyLossGate,
        RejectCode::Cooldown,
        RejectCode::MaxPositions,
        RejectCode::Halted,
        RejectCode::RateLimit,
        RejectCode::InstrumentUnknown,
        RejectCode::ReverseNotAllowed,
        RejectCode::ScaleInLimit,
    ];

    /// The code string used in events and results.
    pub fn code(self) -> &'static str {
        match self {
            RejectCode::QuantityRule => "quantity_rule",
            RejectCode::Tick => "tick",
            RejectCode::FreezeQty => "freeze_qty",
            RejectCode::ProductRule => "product_rule",
            RejectCode::Margin => "margin",
            RejectCode::EntryWindow => "entry_window",
            RejectCode::MarketClosed => "market_closed",
            RejectCode::DailyLossGate => "daily_loss_gate",
            RejectCode::Cooldown => "cooldown",
            RejectCode::MaxPositions => "max_positions",
            RejectCode::Halted => "halted",
            RejectCode::RateLimit => "rate_limit",
            RejectCode::InstrumentUnknown => "instrument_unknown",
            RejectCode::ReverseNotAllowed => "reverse_not_allowed",
            RejectCode::ScaleInLimit => "scale_in_limit",
        }
    }
}

/// Relative tolerance for "a whole number of lots" on f64 quantities.
const WHOLE_LOT_TOLERANCE: f64 = 1.0e-9;

/// True when `units` is a whole multiple of `step` units.
fn is_whole_multiple(units: f64, step: f64) -> bool {
    let count = units / step;
    (count - count.round()).abs() <= WHOLE_LOT_TOLERANCE * count.abs().max(1.0)
}

/// The quantity rule (03-architecture 7): derivatives always trade in whole lots;
/// cash equity trades in whole shares unless the run is in fractional mode.
pub fn quantity_rule(
    units: f64,
    spec: &InstrumentSpec,
    fractional_mode: bool,
) -> Result<(), RejectCode> {
    let lot = f64::from(spec.lot_size_units);
    let whole = if spec.is_derivative() || !fractional_mode {
        is_whole_multiple(units, lot)
    } else {
        true
    };
    if whole && units.is_finite() && units > 0.0 {
        Ok(())
    } else {
        Err(RejectCode::QuantityRule)
    }
}

/// Tick validity (03-architecture 4.4 item 2, 7): the limit and the trigger of an order
/// must be whole numbers of the instrument's tick. A price that is not a usable price at
/// all is refused before the checks (`engine::submit`), so here it counts as off the
/// grid.
pub fn tick_rule(order_type: OrderType, spec: &InstrumentSpec) -> Result<(), RejectCode> {
    let prices = match order_type {
        OrderType::Market => [None, None],
        OrderType::Limit { limit_rupees } => [Some(limit_rupees), None],
        OrderType::StopLimit {
            trigger_rupees,
            limit_rupees,
        } => [Some(trigger_rupees), Some(limit_rupees)],
        OrderType::StopMarket { trigger_rupees } => [Some(trigger_rupees), None],
    };
    let on_grid = |price: f64| is_on_tick(price, spec.tick_size_rupees).unwrap_or(false);
    if prices.into_iter().flatten().all(on_grid) {
        Ok(())
    } else {
        Err(RejectCode::Tick)
    }
}

/// The GTT product rule (03-architecture 4.4 item 3): MIS cannot hold a GTT, so a GTT
/// order on MIS is refused.
pub fn gtt_rule(validity: Validity, product: Product) -> Result<(), RejectCode> {
    if validity == Validity::Gtt && product == Product::Mis {
        Err(RejectCode::ProductRule)
    } else {
        Ok(())
    }
}

/// The freeze quantity: an order above it is denied (the default; sequential slicing
/// is an option that is not built yet).
pub fn freeze_rule(units: f64, spec: &InstrumentSpec) -> Result<(), RejectCode> {
    match spec.freeze_quantity_units {
        Some(limit) if units > f64::from(limit) => Err(RejectCode::FreezeQty),
        _ => Ok(()),
    }
}

/// Product rules: an index cannot be traded, and delivery cannot be shorted
/// (04-execution 2), so a CNC sale may only reduce a long. `position_units` is the
/// signed position before the order (positive long).
pub fn product_rule(
    spec: &InstrumentSpec,
    product: Product,
    side: Side,
    order_units: f64,
    position_units: f64,
) -> Result<(), RejectCode> {
    let opens_short = side == Side::Sell && order_units > position_units.max(0.0);
    if !spec.is_tradable() || (product == Product::Cnc && opens_short) {
        Err(RejectCode::ProductRule)
    } else {
        Ok(())
    }
}

/// The intraday entry window of one session (MIS only, SPEC-32), UTC nanoseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntryWindow {
    /// Entries are allowed from this instant (default: the session open).
    pub from: TsNanos,
    /// Entries are refused at or after this instant.
    pub until: TsNanos,
}

impl EntryWindow {
    /// Check an instant: the decision instant at submission, the fill bar's open at the
    /// fill (SPEC-04).
    pub fn check(&self, instant: TsNanos) -> Result<(), RejectCode> {
        if instant >= self.from && instant < self.until {
            Ok(())
        } else {
            Err(RejectCode::EntryWindow)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Exchange, InstrumentKind};

    fn spec(kind: InstrumentKind, lot: u32) -> InstrumentSpec {
        InstrumentSpec {
            symbol: "X".to_string(),
            exchange: Exchange::Nse,
            kind,
            lot_size_units: lot,
            tick_size_rupees: 0.05,
            multiplier: 1.0,
            freeze_quantity_units: Some(1_800),
            cost_segment: None,
            last_trading_day: None,
        }
    }

    #[test]
    fn derivatives_trade_in_whole_lots_even_in_fractional_mode() {
        let future = spec(InstrumentKind::Future, 75);
        assert_eq!(quantity_rule(150.0, &future, true), Ok(()));
        assert_eq!(
            quantity_rule(100.0, &future, true),
            Err(RejectCode::QuantityRule)
        );
    }

    #[test]
    fn cash_equity_is_fractional_only_in_fractional_mode() {
        let equity = spec(InstrumentKind::Equity, 1);
        assert_eq!(quantity_rule(2.5, &equity, true), Ok(()));
        assert_eq!(
            quantity_rule(2.5, &equity, false),
            Err(RejectCode::QuantityRule)
        );
    }

    #[test]
    fn freeze_and_product_rules_deny() {
        let future = spec(InstrumentKind::Future, 75);
        let equity = spec(InstrumentKind::Equity, 1);
        let index = spec(InstrumentKind::Index, 1);
        assert_eq!(freeze_rule(1_875.0, &future), Err(RejectCode::FreezeQty));
        let short_delivery = product_rule(&equity, Product::Cnc, Side::Sell, 10.0, 5.0);
        assert_eq!(short_delivery, Err(RejectCode::ProductRule));
        assert_eq!(
            product_rule(&equity, Product::Mis, Side::Sell, 10.0, 0.0),
            Ok(())
        );
        let index_buy = product_rule(&index, Product::Mis, Side::Buy, 1.0, 0.0);
        assert_eq!(index_buy, Err(RejectCode::ProductRule));
    }

    #[test]
    fn limit_and_trigger_prices_must_be_whole_ticks() {
        let equity = spec(InstrumentKind::Equity, 1);
        let limit = |price| OrderType::Limit {
            limit_rupees: price,
        };
        assert_eq!(tick_rule(limit(501.05), &equity), Ok(()));
        assert_eq!(tick_rule(limit(501.02), &equity), Err(RejectCode::Tick));
        let stop = OrderType::StopLimit {
            trigger_rupees: 500.0,
            limit_rupees: 499.97,
        };
        assert_eq!(tick_rule(stop, &equity), Err(RejectCode::Tick));
        let stop_market = OrderType::StopMarket {
            trigger_rupees: 500.03,
        };
        assert_eq!(tick_rule(stop_market, &equity), Err(RejectCode::Tick));
        assert_eq!(tick_rule(OrderType::Market, &equity), Ok(()));
    }

    #[test]
    fn mis_cannot_hold_a_gtt() {
        assert_eq!(
            gtt_rule(Validity::Gtt, Product::Mis),
            Err(RejectCode::ProductRule)
        );
        assert_eq!(gtt_rule(Validity::Gtt, Product::Cnc), Ok(()));
        assert_eq!(gtt_rule(Validity::Day, Product::Mis), Ok(()));
    }

    #[test]
    fn the_entry_window_is_half_open() {
        let window = EntryWindow {
            from: 100,
            until: 200,
        };
        assert_eq!(window.check(100), Ok(()));
        assert_eq!(window.check(200), Err(RejectCode::EntryWindow));
        assert_eq!(window.check(99), Err(RejectCode::EntryWindow));
    }

    #[test]
    fn codes_are_the_ruled_vocabulary() {
        let codes: Vec<&str> = RejectCode::ALL.iter().map(|code| code.code()).collect();
        assert_eq!(codes.len(), 15);
        assert!(codes.contains(&"market_closed") && codes.contains(&"instrument_unknown"));
        assert!(codes.contains(&"scale_in_limit"));
    }
}
