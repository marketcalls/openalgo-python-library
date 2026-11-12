//! Entry gates of risk management that act on new entries at submission
//! (03-architecture 4.4 item 7; 05-risk-management 3 and 4; reject codes of the lead
//! ruling): the cooldown after an exit and the maximum number of positions. The
//! trading-state gate of item 6 is `TradingLock::entry_gate`.
//!
//! Both gates apply to an order that would open a position from flat; an add is
//! governed by the scale-in limits instead. Bars are counted on the run's grid of base
//! slices (every slot is a slice, SPEC-07), by slice index.

use crate::exec::RejectCode;

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

#[cfg(test)]
mod tests {
    use super::*;

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
