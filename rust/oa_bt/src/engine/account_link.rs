//! An engine's link to the account it shares with the other strategies of a portfolio
//! run (18-multi-strategy 2.4): the room the account leaves it, the second margin check
//! of every opening fill, and the cap the account puts on sizing.
//!
//! The fund of a portfolio run in a shared pool sets the link before each of the
//! engine's turns (`Engine::set_account_link`). `room_rupees` is the account's free cash
//! without the engine's own (`F - f_i`), so the engine's own free cash plus its room is
//! the account's free cash. While one engine runs only it changes, so the room is exact
//! for its whole turn. An engine with no link (`room_rupees: None`: a run alone, or a
//! separate pool) runs exactly as it does today.
//!
//! - **The second check.** Every opening fill checks its margin against the engine's own
//!   free cash, refused `margin` (`fills::fill_order` for an entry or add at the open, at
//!   a level, at a fill model's price or of a GTT child; `reverse` for the opening part of
//!   a reversal), then against the account's free cash, refused `account_margin`: the
//!   other strategies hold it. Closing and reducing fills are never checked, nor is the
//!   futures roll's next contract (the near contract's margin is freed at the same open).
//! - **The sizing cap.** Every sizer sees the lesser of the engine's own free cash and the
//!   account's as its free cash; the engine default also sizes from at most the account's
//!   free cash in place of the strategy's equity (and of its capital without `reinvest`),
//!   as a live session's sizing cap does with the broker's funds (R18,
//!   `venue_session::session_sizing`). An engine default sized at the fill that the cap
//!   leaves with nothing, where the engine's own figures would have sized it, is refused
//!   `account_margin`.
//!
//! The account's daily loss limit reaches the engine as a breach (`account_breach`); the
//! entry lock it sets is kept here with the link (`AccountDesk`). A live session has no
//! link: its account is the broker's, whose margin check and funds are the session's
//! (`cap_sizing_equity`).

use crate::exec::RejectCode;
use crate::model::Day;
use crate::sizing::SizingContext;

use super::clock::SliceInfo;
use super::error::KernelError;
use super::kernel::Engine;

/// What the account of a portfolio run in a shared pool leaves one engine, set by the
/// fund before each of the engine's turns (module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AccountLink {
    /// Rupees: the account's free cash without this engine's own (`F - f_i`), negative when
    /// the other strategies' open losses have taken the account's free cash below the
    /// engine's own; `None` when the engine is not linked to an account (a run alone, a
    /// separate pool), which changes nothing.
    pub room_rupees: Option<f64>,
}

/// What the account of a portfolio run has told one engine: its link, and the entry lock
/// of an account breach and the breach waiting to be recorded (`account_breach`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct AccountDesk {
    pub(super) link: AccountLink,
    /// The IST session day the account's daily loss limit locks entries for.
    pub(super) locked_day: Option<Day>,
    /// A breach the engine records at its next step 6 or step 1 (`account_breach`).
    pub(super) waiting: Option<super::account_breach::AccountBreach>,
}

impl Engine {
    /// Link the engine to the account of a portfolio run for its next turn (module docs);
    /// `AccountLink::default()` unlinks it.
    ///
    /// Errors: `InvalidConfig` (field `account_link`) for a room that is not a finite
    /// number of rupees, and for a live session, whose account is the broker's.
    pub fn set_account_link(&mut self, link: AccountLink) -> Result<(), KernelError> {
        let refuse = |reason: &str| KernelError::InvalidConfig {
            field: "account_link",
            reason: reason.to_string(),
        };
        if link.room_rupees.is_some_and(|room| !room.is_finite()) {
            return Err(refuse("the room is a finite number of rupees"));
        }
        if self.is_external() && link.room_rupees.is_some() {
            return Err(refuse(
                "a live session's account is the broker's: its margin check and its funds \
                 (cap_sizing_equity) stand for the account",
            ));
        }
        self.account.link = link;
        Ok(())
    }

    /// The margin check of an opening fill that blocks `margin_rupees` (module docs): the
    /// engine's own free cash first, then the account's.
    ///
    /// Errors: `Margin` beyond the engine's own free cash; `AccountMargin` beyond the
    /// account's.
    pub(super) fn check_opening_margin(&self, margin_rupees: f64) -> Result<(), RejectCode> {
        let own = self.free_cash_rupees();
        if margin_rupees > own {
            return Err(RejectCode::Margin);
        }
        match self.account.link.room_rupees {
            Some(room) if margin_rupees > own + room => Err(RejectCode::AccountMargin),
            _ => Ok(()),
        }
    }

    /// `ctx` as the account leaves it to a sizer (module docs): the free cash at most the
    /// account's, and for the engine default (`engine_default`) the equity and the capital
    /// too. Unchanged without a link.
    pub(super) fn account_sizing(&self, ctx: SizingContext, engine_default: bool) -> SizingContext {
        let Some(room) = self.account.link.room_rupees else {
            return ctx;
        };
        let account_free = ctx.free_cash_rupees + room;
        let free_cash_rupees = ctx.free_cash_rupees.min(account_free);
        if !engine_default {
            return SizingContext {
                free_cash_rupees,
                ..ctx
            };
        }
        SizingContext {
            equity_rupees: ctx.equity_rupees.min(account_free),
            initial_capital_rupees: ctx.initial_capital_rupees.min(account_free),
            free_cash_rupees,
            ..ctx
        }
    }

    /// The account's entry gate for an order that opens or adds, decided in `slice`.
    ///
    /// Errors: `AccountLossGate` in a slice of the session day an account breach locked.
    pub(super) fn account_entry_gate(&self, slice: &SliceInfo) -> Result<(), RejectCode> {
        if self.account.locked_day == Some(slice.day) {
            Err(RejectCode::AccountLossGate)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests;
