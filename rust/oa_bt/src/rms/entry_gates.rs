//! Entry gates of risk management that act on new entries at submission
//! (03-architecture 4.4 item 7; 05-risk-management 3 and 4; reject codes of the lead
//! ruling): the cooldown after an exit and the maximum number of positions, and the daily
//! trade caps and the order value cap of 18-multi-strategy 2.5a (rules 3, 7 and 11). The
//! trading-state gate of item 6 is `TradingLock::entry_gate`.
//!
//! The cooldown, maximum positions and the trade caps apply to an order that would open a
//! position from flat (and the trade caps to a reversal's opening part, `trade_count`); an
//! add is governed by the scale-in limits instead. The order value cap applies to entries
//! and adds; no gate here ever refuses an exit. Bars are counted on the run's grid of
//! base slices (every slot is a slice, SPEC-07), by slice index.

use crate::exec::RejectCode;
use crate::portfolio::round_to_paise;

use super::trade_count::AccountTrades;

/// The cooldown after an exit (`cooldown` base bars).
///
/// An entry decided at the close of slice `decision_slice` is refused while fewer than
/// `cooldown_bars` slices have closed after the slice of the instrument's last exit
/// (`last_exit_slice`): with a cooldown of 1, an entry decided at the close of the exit
/// bar is refused and one decided at the next close is allowed.
///
/// Errors: `RejectCode::Cooldown` inside the window.
pub fn cooldown_gate(
    cooldown_bars: Option<u32>,
    last_exit_slice: Option<usize>,
    decision_slice: usize,
) -> Result<(), RejectCode> {
    let (Some(bars), Some(exit)) = (cooldown_bars, last_exit_slice) else {
        return Ok(());
    };
    let closed_since = decision_slice.saturating_sub(exit);
    if closed_since < bars as usize {
        Err(RejectCode::Cooldown)
    } else {
        Ok(())
    }
}

/// The maximum number of positions (`max_positions`).
///
/// `positions_in_use` counts the strategy's positions that are open or have an entry
/// order working, so entries decided together cannot pass the cap between them; the
/// caller does not count an entry for a position already in use.
///
/// Errors: `RejectCode::MaxPositions` when the cap is already reached.
pub fn max_positions_gate(limit: Option<u32>, positions_in_use: usize) -> Result<(), RejectCode> {
    match limit {
        Some(limit) if positions_in_use >= limit as usize => Err(RejectCode::MaxPositions),
        _ => Ok(()),
    }
}

/// The trade caps for an entry (`max_trades_per_day`, `account_max_trades_per_day`):
/// `own` is the strategy's entries counted on the session day, `account` the account's
/// cap and the other strategies' entries counted on that day.
///
/// Errors: `TradeLimit` when the strategy's count has reached its cap (checked first);
/// `AccountTradeLimit` when the account's count, its own and the others', has reached the
/// account's.
pub fn trade_cap_gate(
    limit: Option<u32>,
    own: u32,
    account: Option<AccountTrades>,
) -> Result<(), RejectCode> {
    if limit.is_some_and(|limit| own >= limit) {
        return Err(RejectCode::TradeLimit);
    }
    match account {
        Some(account) if own.saturating_add(account.others) >= account.limit => {
            Err(RejectCode::AccountTradeLimit)
        }
        _ => Ok(()),
    }
}

/// The order value cap (`max_order_value`) for an entry or add worth `value_rupees` (its
/// units times the price it is judged at times the instrument's multiplier). Both are
/// compared in whole paise; a value equal to the cap passes.
///
/// Errors: `OrderValueLimit` above the cap.
pub fn order_value_gate(limit_rupees: Option<f64>, value_rupees: f64) -> Result<(), RejectCode> {
    match limit_rupees {
        Some(limit) if round_to_paise(value_rupees) > round_to_paise(limit) => {
            Err(RejectCode::OrderValueLimit)
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_strategys_trade_cap_refuses_an_entry_once_its_count_reaches_it() {
        assert_eq!(trade_cap_gate(Some(3), 2, None), Ok(()));
        assert_eq!(
            trade_cap_gate(Some(3), 3, None),
            Err(RejectCode::TradeLimit)
        );
        assert_eq!(trade_cap_gate(None, 40, None), Ok(()));
    }

    #[test]
    fn the_accounts_trade_cap_counts_every_strategys_entries() {
        let account = |others| Some(AccountTrades { limit: 5, others });
        // Two of its own and two of the others: a fifth entry is allowed.
        assert_eq!(trade_cap_gate(None, 2, account(2)), Ok(()));
        assert_eq!(
            trade_cap_gate(None, 2, account(3)),
            Err(RejectCode::AccountTradeLimit)
        );
        // The strategy's own cap is named first when both are reached.
        assert_eq!(
            trade_cap_gate(Some(2), 2, account(3)),
            Err(RejectCode::TradeLimit)
        );
    }

    #[test]
    fn the_order_value_cap_refuses_only_a_value_above_it() {
        assert_eq!(order_value_gate(Some(100_000.0), 100_000.0), Ok(()));
        assert_eq!(
            order_value_gate(Some(100_000.0), 100_000.01),
            Err(RejectCode::OrderValueLimit)
        );
        assert_eq!(order_value_gate(None, 1.0e12), Ok(()));
    }

    #[test]
    fn a_cooldown_refuses_entries_until_its_bars_have_closed() {
        // Exit in slice 10, cooldown 3: decisions at the closes of slices 10, 11 and 12
        // are refused, the one at the close of slice 13 is allowed.
        let decisions = [10, 11, 12, 13, 20];
        let outcomes: Vec<_> = decisions
            .iter()
            .map(|&decision| cooldown_gate(Some(3), Some(10), decision))
            .collect();
        let refused = Err(RejectCode::Cooldown);
        assert_eq!(outcomes, vec![refused, refused, refused, Ok(()), Ok(())]);
    }

    #[test]
    fn no_cooldown_or_no_exit_lets_entries_through() {
        assert_eq!(cooldown_gate(None, Some(10), 10), Ok(()));
        assert_eq!(cooldown_gate(Some(3), None, 0), Ok(()));
    }

    #[test]
    fn the_position_cap_refuses_an_entry_once_reached() {
        assert_eq!(max_positions_gate(Some(2), 1), Ok(()));
        assert_eq!(
            max_positions_gate(Some(2), 2),
            Err(RejectCode::MaxPositions)
        );
        assert_eq!(max_positions_gate(None, 50), Ok(()));
    }
}
