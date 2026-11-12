//! Orders: what a strategy asks for (`OrderRequest`) and what the book holds (`Order`).
//!
//! Every ordinary order is a DAY order (04-execution 4.4). Its validity is fixed at
//! submission as an instant, `Order::good_until`: for an intraday strategy interval,
//! the close of the session in which it was decided; an order whose fill would fall in
//! a later session (the last-bar signal) is rejected `market_closed` instead (lead
//! ruling). For a daily or longer interval it is the close of the next session of its
//! instrument, the session in which it first fills. The session-close step expires
//! every order whose `good_until` has come (an order whose fill bar is missing waits
//! for the next bar of its session, SPEC-06, until then).
//!
//! An order fills no earlier than the open of the first base bar of its instrument
//! that opens at or after `Order::decided_at` (decision SPEC-35): the order of slices
//! in the grid never moves a fill earlier than that.
//!
//! A GTT order (`Validity::Gtt`, CNC and NRML) has no session validity: its
//! `good_until` is `GTT_GOOD_UNTIL` and it rests as a GTT until it fires or is
//! cancelled (`gtt`).

use std::fmt;

use crate::model::{Product, Side, TsNanos};
use crate::rms::rules::LegRules;
use crate::rms::vocabulary::ExitReason;
use crate::sizing::SizerSpec;

use super::checks::RejectCode;
use super::error::ExecError;
use super::instrument::InstrumentId;
use super::price::MAX_PRICE_RUPEES;
use super::state::OrderStatus;

/// Identifier of an order, unique within a run, in submission order. Shown as `o1`,
/// `o2`, ... in results.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OrderId(pub u64);

impl fmt::Display for OrderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "o{}", self.0)
    }
}

/// The `good_until` of a GTT order: no session close ends it.
pub const GTT_GOOD_UNTIL: TsNanos = TsNanos::MAX;

/// How the order is priced (04-execution 4.1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OrderType {
    /// At the open of the fill bar, plus slippage.
    Market,
    /// At the limit or better; rests until filled or expired (SPEC-26).
    Limit { limit_rupees: f64 },
    /// A stop-limit (SL): a limit order placed when the trigger is reached.
    StopLimit {
        trigger_rupees: f64,
        limit_rupees: f64,
    },
    /// A stop-market (SL-M): a market order placed when the trigger is reached.
    StopMarket { trigger_rupees: f64 },
}

impl OrderType {
    /// The limit price, for LIMIT and SL, rupees.
    pub fn limit_rupees(self) -> Option<f64> {
        match self {
            OrderType::Limit { limit_rupees } | OrderType::StopLimit { limit_rupees, .. } => {
                Some(limit_rupees)
            }
            _ => None,
        }
    }

    /// The trigger price, for SL and SL-M, rupees.
    pub fn trigger_rupees(self) -> Option<f64> {
        match self {
            OrderType::StopLimit { trigger_rupees, .. }
            | OrderType::StopMarket { trigger_rupees } => Some(trigger_rupees),
            _ => None,
        }
    }

    /// The OpenAlgo `pricetype` string.
    pub fn code(self) -> &'static str {
        match self {
            OrderType::Market => "MARKET",
            OrderType::Limit { .. } => "LIMIT",
            OrderType::StopLimit { .. } => "SL",
            OrderType::StopMarket { .. } => "SL-M",
        }
    }
}

/// How long an order rests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Validity {
    /// Until the session close (see the module documentation).
    #[default]
    Day,
    /// Until it fires, is cancelled or expires (CNC and NRML only), as a GTT.
    Gtt,
}

/// How much to trade.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QuantitySpec {
    /// Units (shares, or units of the underlying for derivatives). Must satisfy the
    /// quantity rule as given; it is never rounded.
    Units(f64),
    /// Whole lots, converted with the instrument's lot size.
    Lots(u32),
    /// A fraction (above 0, at most 1) of the position at the fill, rounded down to the
    /// venue quantity (`exit(fraction=)`, `reduce(fraction=)`).
    PositionFraction(f64),
    /// A sizer evaluated at the decision price.
    Sizer(SizerSpec),
}

/// What a strategy asks for. Built by the strategy layer from `buy`, `sell`, `exit`,
/// `reduce` and `target` (14-strategy-api 2.4).
#[derive(Clone, Debug, PartialEq)]
pub struct OrderRequest {
    /// The instrument.
    pub instrument: InstrumentId,
    /// Buy or sell.
    pub side: Side,
    /// How much.
    pub quantity: QuantitySpec,
    /// How it is priced.
    pub order_type: OrderType,
    /// DAY or GTT.
    pub validity: Validity,
    /// CNC, MIS or NRML.
    pub product: Product,
    /// The order may only reduce a position (`exit`, `reduce`): it never opens one.
    pub reduce_only: bool,
    /// When the order exceeds an opposite position, close it and open the rest on the
    /// other side (two fills, two sets of costs). Without it the excess is rejected.
    pub reverse: bool,
    /// The reason recorded when the order reduces a position (default `signal`).
    pub exit_reason: Option<ExitReason>,
    /// Leg rules for the position the order opens (empty for none).
    pub leg_rules: LegRules,
    /// Intraday leverage asked for (`leverage=`); `None` is 1x.
    pub leverage: Option<f64>,
    /// The strategy's own label, carried to results.
    pub tag: Option<String>,
}

impl OrderRequest {
    /// A plain DAY market order with no rules: the shape of the walking skeleton.
    pub fn market(instrument: InstrumentId, side: Side, quantity: QuantitySpec) -> OrderRequest {
        OrderRequest {
            instrument,
            side,
            quantity,
            order_type: OrderType::Market,
            validity: Validity::Day,
            product: Product::Cnc,
            reduce_only: false,
            reverse: false,
            exit_reason: None,
            leg_rules: LegRules::default(),
            leverage: None,
            tag: None,
        }
    }

    /// Check that the request's prices can be read: a limit and a trigger are positive
    /// finite rupees up to `price::MAX_PRICE_RUPEES`, and a GTT order has a price (its
    /// trigger). Whether they lie on the tick grid is a submission check (`tick`).
    ///
    /// Errors: `InvalidRequest` saying which price is unusable.
    pub fn check_prices(&self) -> Result<(), ExecError> {
        let usable = |price: f64| price.is_finite() && price > 0.0 && price <= MAX_PRICE_RUPEES;
        let order_type = self.order_type;
        let prices = [order_type.limit_rupees(), order_type.trigger_rupees()];
        if let Some(price) = prices.into_iter().flatten().find(|price| !usable(*price)) {
            return Err(ExecError::InvalidRequest {
                reason: format!(
                    "price {price} is not usable: give a positive price in rupees (price= for \
                     the limit, trigger_price= for the trigger)"
                ),
            });
        }
        if self.validity == Validity::Gtt && order_type == OrderType::Market {
            return Err(ExecError::InvalidRequest {
                reason: "a GTT order rests until a trigger: give price= (and optionally \
                         trigger_price=)"
                    .to_string(),
            });
        }
        Ok(())
    }
}

/// An order in the book.
#[derive(Clone, Debug, PartialEq)]
pub struct Order {
    /// Its identifier.
    pub id: OrderId,
    /// What was asked for.
    pub request: OrderRequest,
    /// Where it stands.
    pub status: OrderStatus,
    /// The instant it was decided, UTC nanoseconds: it fills no earlier than the open of
    /// a base bar at or after this instant (SPEC-35).
    pub decided_at: TsNanos,
    /// The instant its DAY validity ends, UTC nanoseconds (`GTT_GOOD_UNTIL` for a GTT
    /// order).
    pub good_until: TsNanos,
    /// Quantity in units resolved at submission; 0 when it is resolved at the fill
    /// (`QuantitySpec::PositionFraction`).
    pub quantity_units: f64,
    /// Units filled so far.
    pub filled_units: f64,
    /// Average fill price in rupees (0 before any fill).
    pub average_fill_rupees: f64,
    /// Why it was rejected, when it was.
    pub reject_code: Option<RejectCode>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_ids_print_as_in_results() {
        assert_eq!(OrderId(12).to_string(), "o12");
        assert_eq!(
            OrderType::StopMarket {
                trigger_rupees: 1.0
            }
            .code(),
            "SL-M"
        );
    }

    fn priced(order_type: OrderType, validity: Validity) -> OrderRequest {
        OrderRequest {
            order_type,
            validity,
            ..OrderRequest::market(InstrumentId(0), Side::Buy, QuantitySpec::Units(1.0))
        }
    }

    #[test]
    fn limit_and_trigger_prices_must_be_positive_finite_rupees() {
        let negative = OrderType::Limit { limit_rupees: -1.0 };
        let nan_trigger = OrderType::StopMarket {
            trigger_rupees: f64::NAN,
        };
        let fine = OrderType::StopLimit {
            trigger_rupees: 100.0,
            limit_rupees: 100.5,
        };
        assert!(priced(negative, Validity::Day).check_prices().is_err());
        assert!(priced(nan_trigger, Validity::Day).check_prices().is_err());
        assert_eq!(priced(fine, Validity::Day).check_prices(), Ok(()));
    }

    #[test]
    fn a_gtt_order_needs_a_price() {
        assert!(priced(OrderType::Market, Validity::Gtt)
            .check_prices()
            .is_err());
        let limit = OrderType::Limit {
            limit_rupees: 1450.0,
        };
        assert_eq!(priced(limit, Validity::Gtt).check_prices(), Ok(()));
    }
}
