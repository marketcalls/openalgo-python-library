//! The books of a live session's saved state (`venue_state`): the account (the cash), the
//! positions with their tranches, the round trips still open (so a trade closed after a
//! restart is reported with its entry) and the session's CNC buys (the sale credit).

use crate::exec::InstrumentId;
use crate::portfolio::{Position, PositionKey, SubAccount, Tranche};

use super::super::output::OpenTrade;
use super::error::LiveStateError;
use super::requests::{key_fields, read_instrument, read_key, read_side};
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

/// A round trip still open: its entry, what it has traded so far, its gross P&L and its
/// initial risk.
pub(super) fn write_open_trade(trade: &OpenTrade) -> StateValue {
    let [instrument, product] = key_fields(trade.key);
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
    ])
}

/// The round trip `write_open_trade` wrote. Errors: `Malformed` for a field missing or
/// unusable.
pub(super) fn read_open_trade(
    read: &Read<'_>,
    instruments: usize,
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
