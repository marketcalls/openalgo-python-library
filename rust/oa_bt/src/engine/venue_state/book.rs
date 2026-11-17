//! The books of a live session's saved state (`venue_state`): the account (the cash), the
//! positions with their tranches, the round trips still open (so a trade closed after a
//! restart is reported with its entry, its excursion so far and its slippage) and the
//! session's CNC buys (the sale credit).

use crate::exec::{InstrumentId, OrderBook, OrderId};
use crate::portfolio::{Position, PositionKey, SubAccount, Tranche};

use super::super::excursion::Excursion;
use super::super::excursion_hold::{Hold, ReplacedTerms, SliceTicks};
use super::super::output::OpenTrade;
use super::error::LiveStateError;
use super::orders::Ids;
use super::requests::{key_fields, read_instrument, read_key, read_order_type, read_side};
use super::value::{list, map, Read, StateValue};

/// The account: capital, realized P&L, charges, margin blocked and sale proceeds withheld,
/// rupees.
pub(super) fn write_account(account: &SubAccount) -> StateValue {
    map([
        ("capital_rupees", account.capital_rupees.into()),
        ("realized_rupees", account.realized_rupees.into()),
        ("costs_rupees", account.costs_rupees.into()),
        (
            "margin_blocked_rupees",
            account.margin_blocked_rupees.into(),
        ),
        (
            "uncredited_sale_rupees",
            account.uncredited_sale_rupees.into(),
        ),
    ])
}

/// The account `write_account` wrote. Errors: `Malformed` for a field missing.
pub(super) fn read_account(read: &Read<'_>) -> Result<SubAccount, LiveStateError> {
    Ok(SubAccount {
        capital_rupees: read.f64_of("capital_rupees")?,
        realized_rupees: read.f64_of("realized_rupees")?,
        costs_rupees: read.f64_of("costs_rupees")?,
        margin_blocked_rupees: read.f64_of("margin_blocked_rupees")?,
        uncredited_sale_rupees: read.f64_of("uncredited_sale_rupees")?,
    })
}

fn write_tranche(tranche: &Tranche) -> StateValue {
    map([
        ("number", tranche.number.into()),
        ("opened_units", tranche.opened_units.into()),
        ("open_units", tranche.open_units.into()),
        ("price_rupees", tranche.price_rupees.into()),
        ("opened_at_ns", tranche.opened_at.into()),
    ])
}

fn read_tranche(read: &Read<'_>) -> Result<Tranche, LiveStateError> {
    Ok(Tranche {
        number: read.whole_of("number")?,
        opened_units: read.f64_of("opened_units")?,
        open_units: read.f64_of("open_units")?,
        price_rupees: read.f64_of("price_rupees")?,
        opened_at: read.whole_of("opened_at_ns")?,
    })
}

/// A position, flat ones included (they keep the P&L realized on them).
pub(super) fn write_position(position: &Position) -> StateValue {
    let [instrument, product] = key_fields(position.key);
    map([
        instrument,
        product,
        ("multiplier", position.multiplier.into()),
        ("quantity_units", position.quantity_units.into()),
        ("average_price_rupees", position.average_price_rupees.into()),
        ("tranches", list(&position.tranches, write_tranche)),
        ("adds", position.adds.into()),
        ("realized_rupees", position.realized_rupees.into()),
        ("margin_rupees", position.margin_rupees.into()),
        ("mark_rupees", position.mark_rupees.into()),
        ("stale", position.stale.into()),
    ])
}

/// The position `write_position` wrote. Errors: `Malformed` for a field missing or
/// unusable.
pub(super) fn read_position(
    read: &Read<'_>,
    instruments: usize,
) -> Result<Position, LiveStateError> {
    Ok(Position {
        key: read_key(read, instruments)?,
        multiplier: read.f64_of("multiplier")?,
        quantity_units: read.f64_of("quantity_units")?,
        average_price_rupees: read.f64_of("average_price_rupees")?,
        tranches: read
            .items_of("tranches")?
            .iter()
            .map(read_tranche)
            .collect::<Result<_, _>>()?,
        adds: read.whole_of("adds")?,
        realized_rupees: read.f64_of("realized_rupees")?,
        margin_rupees: read.f64_of("margin_rupees")?,
        mark_rupees: read.opt_f64_of("mark_rupees")?,
        stale: read.bool_of("stale")?,
    })
}

/// What holds a live trade back (`excursion_hold`): the orders of `book` it waits on
/// that still work, by id (those a price reached by their own terms, and those reached
/// only by the terms of a modification not answered), the slice of its last ticks with
/// their lowest and highest price, rupees, and the terms confirmed modifications replaced
/// on orders that still work, with the slice of each confirmation (the order type in the
/// fields `read_order_type` reads).
fn write_hold(book: &OrderBook, hold: &Hold) -> [(&'static str, StateValue); 4] {
    let works = |id: &&OrderId| {
        book.get(**id)
            .is_some_and(|order| order.status.is_working())
    };
    let ids = |orders: &[OrderId]| list(orders.iter().filter(works), |id| StateValue::from(id.0));
    let ticks = hold.ticks.map_or(StateValue::Null, |ticks| {
        map([
            ("slice", ticks.slice.into()),
            ("low_rupees", ticks.low_rupees.into()),
            ("high_rupees", ticks.high_rupees.into()),
        ])
    });
    let still_working = hold.replaced_terms.iter().filter(|old| works(&&old.order));
    let replaced = list(still_working, |old| {
        map([
            ("order", old.order.0.into()),
            ("order_type", old.terms.code().into()),
            ("limit_rupees", old.terms.limit_rupees().into()),
            ("trigger_rupees", old.terms.trigger_rupees().into()),
            ("slice", old.slice.into()),
        ])
    });
    [
        ("unconfirmed_close_orders", ids(&hold.unconfirmed_closes)),
        ("pending_close_orders", ids(&hold.pending_closes)),
        ("slice_ticks", ticks),
        ("replaced_terms", replaced),
    ]
}

/// The hold `write_hold` wrote, its orders named by their new ids (`ids`). Errors:
/// `Malformed` for a field missing or unusable, or an order that is not a saved working
/// order.
fn read_hold(read: &Read<'_>, ids: &Ids) -> Result<Hold, LiveStateError> {
    let read_ticks = |ticks: Read<'_>| -> Result<SliceTicks, LiveStateError> {
        Ok(SliceTicks {
            slice: ticks.whole_of("slice")?,
            low_rupees: ticks.f64_of("low_rupees")?,
            high_rupees: ticks.f64_of("high_rupees")?,
        })
    };
    let orders = |field: &str| -> Result<Vec<OrderId>, LiveStateError> {
        read.items_of(field)?
            .iter()
            .map(|order| ids.working(order))
            .collect()
    };
    let read_replaced = |old: &Read<'_>| -> Result<ReplacedTerms, LiveStateError> {
        Ok(ReplacedTerms {
            order: ids.working(&old.field("order")?)?,
            terms: read_order_type(old)?,
            slice: old.whole_of("slice")?,
        })
    };
    Ok(Hold {
        unconfirmed_closes: orders("unconfirmed_close_orders")?,
        pending_closes: orders("pending_close_orders")?,
        ticks: read.maybe("slice_ticks")?.map(read_ticks).transpose()?,
        replaced_terms: read
            .items_of("replaced_terms")?
            .iter()
            .map(read_replaced)
            .collect::<Result<_, _>>()?,
    })
}

/// A round trip still open: its entry, what it has traded so far, its gross P&L, its
/// initial risk, its excursion so far (unrounded rupees, the last slice with a fill of it
/// inside the bar, and what holds it back, `write_hold`) and its slippage so far.
pub(super) fn write_open_trade(book: &OrderBook, trade: &OpenTrade) -> StateValue {
    let [instrument, product] = key_fields(trade.key);
    let excursion = &trade.excursion;
    let [unconfirmed, pending, ticks, replaced] = write_hold(book, &excursion.hold);
    map([
        instrument,
        product,
        ("direction", trade.direction.code().into()),
        ("entry_slice", trade.entry_slice.into()),
        ("entry_ts_ns", trade.entry_ts.into()),
        ("entry_units", trade.entry_units.into()),
        ("entry_value_rupees", trade.entry_value.into()),
        ("exit_units", trade.exit_units.into()),
        ("exit_value_rupees", trade.exit_value.into()),
        ("max_units", trade.max_units.into()),
        ("gross_rupees", trade.gross.into()),
        ("fills", trade.fills.into()),
        ("initial_risk_rupees", trade.initial_risk.into()),
        ("worst_rupees", excursion.worst_rupees.into()),
        ("best_rupees", excursion.best_rupees.into()),
        ("inside_fill_slice", excursion.inside_fill_slice.into()),
        unconfirmed,
        pending,
        ticks,
        replaced,
        ("slippage_rupees", trade.slippage.into()),
    ])
}

/// The round trip `write_open_trade` wrote, its orders named by their new ids (`ids`).
/// Errors: `Malformed` for a field missing or unusable, or an order that is not a saved
/// working order.
pub(super) fn read_open_trade(
    read: &Read<'_>,
    instruments: usize,
    ids: &Ids,
) -> Result<OpenTrade, LiveStateError> {
    Ok(OpenTrade {
        key: read_key(read, instruments)?,
        direction: read_side(read, "direction")?,
        entry_slice: read.whole_of("entry_slice")?,
        entry_ts: read.whole_of("entry_ts_ns")?,
        entry_units: read.f64_of("entry_units")?,
        entry_value: read.f64_of("entry_value_rupees")?,
        exit_units: read.f64_of("exit_units")?,
        exit_value: read.f64_of("exit_value_rupees")?,
        max_units: read.f64_of("max_units")?,
        gross: read.f64_of("gross_rupees")?,
        fills: read.whole_of("fills")?,
        initial_risk: read.opt_f64_of("initial_risk_rupees")?,
        excursion: Excursion {
            worst_rupees: read.f64_of("worst_rupees")?,
            best_rupees: read.f64_of("best_rupees")?,
            inside_fill_slice: read.opt_whole_of("inside_fill_slice")?,
            hold: read_hold(read, ids)?,
        },
        slippage: read.opt_f64_of("slippage_rupees")?,
    })
}

/// The CNC units of a scrip bought in the session and not yet sold.
pub(super) fn write_buy((instrument, units): &(InstrumentId, f64)) -> StateValue {
    map([
        ("instrument", instrument.0.into()),
        ("units", (*units).into()),
    ])
}

/// The buy `write_buy` wrote. Errors: `Malformed` for a field missing or unusable.
pub(super) fn read_buy(
    read: &Read<'_>,
    instruments: usize,
) -> Result<(InstrumentId, f64), LiveStateError> {
    let instrument = read_instrument(&read.field("instrument")?, instruments)?;
    Ok((instrument, read.f64_of("units")?))
}

/// True when two saved positions or legs name the same position.
pub(super) fn repeats(keys: impl IntoIterator<Item = PositionKey>) -> bool {
    let keys: Vec<PositionKey> = keys.into_iter().collect();
    keys.iter()
        .enumerate()
        .any(|(at, key)| keys[..at].contains(key))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Product, Side};

    #[test]
    fn an_open_trade_keeps_its_excursion_and_slippage_through_a_restart() {
        // Long 100 since slice 12, a fill inside slice 40's bar, its worst running P&L
        // -310.25 and its best 120.50 so far, ticks of slice 41 from 501.20 to 503.05,
        // slippage 3.40 so far; and one whose fills the venue priced (no slippage), none
        // inside a bar and no tick. Neither waits on an order (a restart that keeps one:
        // tests/excursion_live.rs).
        let trade = OpenTrade {
            key: PositionKey {
                instrument: InstrumentId(0),
                product: Product::Mis,
            },
            direction: Side::Buy,
            entry_slice: 12,
            entry_ts: 1_751_341_500_000_000_000,
            entry_units: 100.0,
            entry_value: 50_010.0,
            exit_units: 0.0,
            exit_value: 0.0,
            max_units: 100.0,
            gross: 0.0,
            fills: 1,
            initial_risk: None,
            excursion: Excursion {
                worst_rupees: -310.25,
                best_rupees: 120.5,
                inside_fill_slice: Some(40),
                hold: Hold {
                    ticks: Some(SliceTicks {
                        slice: 41,
                        low_rupees: 501.2,
                        high_rupees: 503.05,
                    }),
                    ..Hold::default()
                },
            },
            slippage: Some(3.4),
        };
        let venue_priced = OpenTrade {
            excursion: Excursion::default(),
            slippage: None,
            ..trade.clone()
        };
        let (book, ids) = (OrderBook::default(), Ids { pairs: Vec::new() });
        for saved in [trade, venue_priced] {
            let value = write_open_trade(&book, &saved);
            assert_eq!(read_open_trade(&Read::root(&value), 1, &ids), Ok(saved));
        }
    }

    #[test]
    fn an_order_that_held_a_trade_back_and_has_ended_is_not_saved() {
        // The trade waits on o7, reached by its own terms, and on o8, reached by the terms
        // of its modification, and keeps the terms a confirmed modification replaced on o9
        // for the bar of slice 3; the book has none of them working: nothing holds it back
        // after a restart.
        let trade = OpenTrade {
            key: PositionKey {
                instrument: InstrumentId(0),
                product: Product::Mis,
            },
            direction: Side::Sell,
            entry_slice: 3,
            entry_ts: 1_751_341_500_000_000_000,
            entry_units: 50.0,
            entry_value: 10_000.0,
            exit_units: 0.0,
            exit_value: 0.0,
            max_units: 50.0,
            gross: 0.0,
            fills: 1,
            initial_risk: None,
            excursion: Excursion {
                hold: Hold {
                    unconfirmed_closes: vec![OrderId(7)],
                    pending_closes: vec![OrderId(8)],
                    ticks: None,
                    replaced_terms: vec![ReplacedTerms {
                        order: OrderId(9),
                        terms: crate::exec::OrderType::Limit {
                            limit_rupees: 503.0,
                        },
                        slice: 3,
                    }],
                },
                ..Excursion::default()
            },
            slippage: None,
        };
        let value = write_open_trade(&OrderBook::default(), &trade);
        let read = read_open_trade(&Read::root(&value), 1, &Ids { pairs: Vec::new() });
        let restored = read.expect("the trade reads back");
        assert_eq!(restored.excursion.hold, Hold::default());
    }
}
