//! The settings of a portfolio run's account (18-multi-strategy 1.1, 2.4, 2.5 and 2.5a):
//! how its strategies share it (`Pool`), its capital and its limits (`FundConfig`), and
//! their checks when the fund is made (`FundConfig::checked_capital`).

use crate::portfolio::round_to_paise;

use super::FundError;

/// How the strategies of a portfolio run share the account (18-multi-strategy 1.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pool {
    /// One account: an opening fill needs its margin within the strategy's own free cash
    /// and within the account's, and sizers see at most the account's free cash
    /// (`engine::AccountLink`).
    Shared,
    /// Fixed allocations: each strategy limited by its own capital only; the account is
    /// their sum, recorded.
    Separate,
}

impl Pool {
    /// Every pool, in the order of their codes.
    pub const ALL: [Pool; 2] = [Pool::Shared, Pool::Separate];

    /// The argument's value: "shared" or "separate".
    pub fn code(self) -> &'static str {
        match self {
            Pool::Shared => "shared",
            Pool::Separate => "separate",
        }
    }

    /// The pool a code names.
    pub fn from_code(code: &str) -> Option<Pool> {
        Pool::ALL.into_iter().find(|pool| pool.code() == code)
    }
}

/// The account of a portfolio run (18-multi-strategy 1.1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FundConfig {
    /// One account shared by the strategies, or fixed allocations (`Pool`).
    pub pool: Pool,
    /// Rupees: the account's funds; `None` for the sum of the strategies' capitals. In a
    /// separate pool it is that sum.
    pub account_capital_rupees: Option<f64>,
    /// Rupees, above 0: the account's daily loss limit (`account_max_daily_loss`); `None`
    /// for none.
    pub account_max_daily_loss_rupees: Option<f64>,
    /// Rupees, above 0: the account's daily profit target (`account_daily_profit_target`);
    /// `None` for none.
    pub account_daily_profit_target_rupees: Option<f64>,
    /// At least 1: the most entries of every strategy together in one session day
    /// (`account_max_trades_per_day`); `None` for no cap.
    pub account_max_trades_per_day: Option<u32>,
}

impl FundConfig {
    /// The account's capital, rupees, once every setting is checked: the one given, or the
    /// sum of the strategies' `capitals` (rupees, in call order).
    ///
    /// Errors: `InvalidConfig` for an account capital that is not a positive number of
    /// rupees (or, in a separate pool, not the sum of the capitals), a daily loss limit or
    /// daily profit target that is not a positive number of rupees, and a trade cap of 0.
    pub(super) fn checked_capital(&self, capitals: &[f64]) -> Result<f64, FundError> {
        let account_capital = account_capital(self, capitals)?;
        let amounts = [
            (
                "account_max_daily_loss",
                self.account_max_daily_loss_rupees,
                "daily loss limit",
            ),
            (
                "account_daily_profit_target",
                self.account_daily_profit_target_rupees,
                "daily profit target",
            ),
        ];
        for (field, amount, what) in amounts {
            if let Some(amount) = amount.filter(|amount| !(amount.is_finite() && *amount > 0.0)) {
                return Err(FundError::InvalidConfig {
                    field,
                    strategy: None,
                    reason: format!("{amount}: give the account's {what} in rupees, above 0"),
                });
            }
        }
        if self.account_max_trades_per_day == Some(0) {
            return Err(FundError::InvalidConfig {
                field: "account_max_trades_per_day",
                strategy: None,
                reason: "0: give the account's trade cap as at least 1 entry a day, or None"
                    .to_string(),
            });
        }
        Ok(account_capital)
    }
}

/// The account's capital (`FundConfig::account_capital_rupees`), rupees.
///
/// Errors: `InvalidConfig` (field `account_capital`) as `FundConfig::checked_capital` says.
fn account_capital(config: &FundConfig, capitals: &[f64]) -> Result<f64, FundError> {
    let sum = round_to_paise(capitals.iter().sum());
    let Some(given) = config.account_capital_rupees else {
        return Ok(sum);
    };
    let refuse = |reason: String| FundError::InvalidConfig {
        field: "account_capital",
        strategy: None,
        reason,
    };
    if !(given.is_finite() && given > 0.0) {
        return Err(refuse(format!(
            "{given}: give the account's funds in rupees, above 0"
        )));
    }
    if config.pool == Pool::Separate && round_to_paise(given) != sum {
        return Err(refuse(format!(
            "a separate pool's account is the sum of the capitals, Rs {sum}: leave \
             account_capital out, or give pool=\"shared\" for an account of Rs {given}"
        )));
    }
    Ok(given)
}
