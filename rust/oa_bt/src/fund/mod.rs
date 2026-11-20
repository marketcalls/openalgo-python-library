//! The fund of a portfolio run (18-multi-strategy section 2): the engines of several
//! strategies advanced in lockstep over one account, so a trade of one strategy changes
//! what the others can use, and the account's daily loss limit stops them all.
//!
//! Each strategy keeps its own `Engine`, unchanged in its rules: its own exchange, base
//! interval, grid, positions, orders and records. The fund owns only its place on the
//! merged time line, the account's ledger and rules, its records and the Rust drivers of
//! condition and array strategies; the caller owns the engines and lends them to every
//! `advance` in call order (the order of the strategies the caller gave), so each
//! strategy's driver keeps its own engine and decides on it while it is paused, exactly as
//! at a pause of the engine's own `advance`.
//!
//! # The time line (2.3, 2.6)
//!
//! Each slice of each engine brings two instants: its open and its close. At one instant
//! the fund first runs the round of the slices that close there, then the open phases of
//! the slices that open there (a slice ending at t comes before the next one starting at
//! t). An engine on a coarser base (a daily strategy beside 1-minute ones) takes its open
//! fills at its slice's open and stays opened through the finer rounds until its slice
//! closes; engines on other exchanges take only the instants of their own grids.
//!
//! | Phase | When | For each engine, in call order |
//! |---|---|---|
//! | O, the open | an open instant, the engines whose slice opens there | steps 1 and 2 (`open_phase`); a timer pause goes to the strategy's driver, and the rest of step 2 runs when the fund is advanced again (`resume_open_phase`) |
//! | L, the levels | a close instant, the engines whose slice closes there (the round's participants) | steps 3 to 6 (`levels_phase`) |
//! | A, the account | once per round | the account's rules on the ledger (`rules`); at a breach every engine still running takes it (`Engine::account_breach`) |
//! | C, the decisions | the participants | step 7 (`decision_phase`): the pause goes to the strategy's driver; every participant stays paused until S, so a later strategy's decision cannot change an earlier one's slice |
//! | S, the close | the participants | step 8 (`end_slice`), then the account's row of the round |
//!
//! Before each engine's turn in a shared pool the fund sets its link to the account from
//! the ledger (`Engine::set_account_link`), and after it the ledger takes the engine's
//! new figures; while one engine runs only it changes, so the room it was given is exact
//! for its whole turn. An engine whose grid ends at a round is finished right after it
//! (its `Pause::Finished` goes to its driver) and takes no more phases; its last figures
//! stay in the ledger, and the fund goes on while any engine runs (2.6).
//!
//! # The account (2.4, 2.5)
//!
//! `ledger` holds the account's capital and each engine's capital and last figures, in
//! paise; `rules` the account's daily loss limit, measured on the round's IST session day
//! only (H3), once per session day. In a separate pool no engine is linked: each runs as
//! it would alone, and the account's figures are only recorded.
//!
//! # Drivers
//!
//! A condition or array strategy is decided in Rust at its pauses, as `run_signals`
//! decides it (`set_signals`). Every other strategy's pause is returned by `advance` as
//! `FundPause::Strategy` with the strategy's index; its driver decides on that engine and
//! calls `advance` again.
//!
//! # Determinism
//!
//! One thread; the time line in instant order, closes before opens at one instant,
//! engines in call order at every phase; the ledger's figures in paise, summed in call
//! order; no clock and no hash-ordered container. The same engines and decisions give
//! the same outputs, bit for bit.

mod ledger;
mod output;
mod rounds;
mod rules;
mod timeline;

#[cfg(test)]
mod tests;

use std::fmt;

use crate::engine::{Engine, KernelError, Pause, SignalArrays, SkippedEntry};
use crate::portfolio::round_to_paise;

use ledger::AccountBook;
use output::Recorder;
use rounds::{Round, Stage};
use rules::AccountRules;
use timeline::Lane;

pub use output::{AccountRows, FundOutput, Refusal, RefusedOrder};

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
    pub pool: Pool,
    /// Rupees: the account's funds; `None` for the sum of the strategies' capitals. In a
    /// separate pool it is that sum.
    pub account_capital_rupees: Option<f64>,
    /// Rupees, above 0: the account's daily loss limit (`account_max_daily_loss`); `None`
    /// for none.
    pub account_max_daily_loss_rupees: Option<f64>,
}

/// Where `advance` stopped.
#[derive(Clone, Debug, PartialEq)]
pub enum FundPause {
    /// The engine of strategy `strategy` (its index in call order) stands at `pause`, a
    /// decision point its driver decides: the driver acts on that engine as at a pause of
    /// the engine's own `advance`, then advances the fund again. A `Pause::Finished`
    /// says the strategy's run has ended; its driver then calls `on_stop`.
    Strategy { strategy: usize, pause: Pause },
    /// Every engine has finished: `finish` each engine, then the fund.
    Finished,
}

/// Why the fund stopped.
#[derive(Clone, Debug, PartialEq)]
pub enum FundError {
    /// A setting of the fund, or the engines given to it, that cannot be used.
    InvalidConfig { field: &'static str, reason: String },
    /// The engine of strategy `strategy` (its index in call order) refused a call.
    Engine { strategy: usize, error: KernelError },
    /// The engine of strategy `strategy` was advanced outside the fund.
    OutOfStep { strategy: usize },
    /// `advance` called after the fund finished.
    Finished,
    /// `finish` called before the fund finished.
    NotFinished,
}

impl FundError {
    /// True for an error in what the caller supplied (the fund's settings, the engines,
    /// an engine's input), as opposed to a call at the wrong time or a defect.
    pub fn is_input_error(&self) -> bool {
        match self {
            FundError::InvalidConfig { .. } => true,
            FundError::Engine { error, .. } => error.is_input_error(),
            FundError::OutOfStep { .. } | FundError::Finished | FundError::NotFinished => false,
        }
    }
}

impl fmt::Display for FundError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FundError::InvalidConfig { field, reason } => write!(f, "{field}: {reason}"),
            FundError::Engine { strategy, error } => write!(f, "strategy {strategy}: {error}"),
            FundError::OutOfStep { strategy } => write!(
                f,
                "the engine of strategy {strategy} was advanced outside its fund: the engines \
                 of a portfolio run are advanced by the fund only"
            ),
            FundError::Finished => write!(f, "the portfolio run has finished; nothing to advance"),
            FundError::NotFinished => write!(
                f,
                "the portfolio run has not finished: advance the fund to the end first"
            ),
        }
    }
}

impl std::error::Error for FundError {}

/// What decides a strategy's pauses.
#[derive(Clone, Debug, PartialEq)]
enum Decider {
    /// The caller, through `FundPause::Strategy` (an event strategy).
    Caller,
    /// The Rust driver of a condition or array strategy, and the entries its sizer skipped.
    Signals {
        arrays: Box<SignalArrays>,
        skipped: Vec<SkippedEntry>,
    },
}

/// The fund of a portfolio run (module docs).
#[derive(Clone, Debug)]
pub struct Fund {
    pool: Pool,
    lanes: Vec<Lane>,
    deciders: Vec<Decider>,
    book: AccountBook,
    rules: AccountRules,
    record: Recorder,
    stage: Stage,
    /// True once `advance` has been called.
    started: bool,
    /// The round in progress, between its levels and its close.
    round: Option<Round>,
    /// The engines of the current phase, in call order: those opening at the instant, or
    /// the round's participants, or those whose run ended at the round.
    turns: Vec<usize>,
}

impl Fund {
    /// The fund over `engines`, in call order: backtests before their first `advance`.
    ///
    /// Errors: `InvalidConfig` for no engine, a live session (its account is the
    /// broker's), an engine that has advanced, an account capital that is not a positive
    /// number of rupees (or, in a separate pool, not the sum of the capitals), and a daily
    /// loss limit that is not a positive number of rupees.
    pub fn new(config: FundConfig, engines: &[&Engine]) -> Result<Fund, FundError> {
        let lanes = timeline::lanes(engines)?;
        let capitals: Vec<f64> = engines.iter().map(|e| e.capital_rupees()).collect();
        let account_capital = account_capital(&config, &capitals)?;
        let limit = config.account_max_daily_loss_rupees;
        if let Some(limit) = limit.filter(|limit| !(limit.is_finite() && *limit > 0.0)) {
            return Err(FundError::InvalidConfig {
                field: "account_max_daily_loss",
                reason: format!("{limit}: give the account's daily loss limit in rupees, above 0"),
            });
        }
        let figures = engines.iter().map(|e| e.figures()).collect();
        Ok(Fund {
            pool: config.pool,
            deciders: vec![Decider::Caller; engines.len()],
            book: AccountBook::new(account_capital, &capitals, figures),
            rules: AccountRules::new(limit),
            record: Recorder::new(account_capital, engines.len()),
            stage: Stage::Next,
            started: false,
            round: None,
            turns: Vec::with_capacity(engines.len()),
            lanes,
        })
    }

    /// Decide strategy `strategy`'s pauses in Rust from its condition arrays, as
    /// `run_signals` decides them; `engine` is that strategy's engine. Before the first
    /// `advance`.
    ///
    /// Errors: `InvalidConfig` (field `conditions`) for a strategy the fund does not run,
    /// one given arrays already, and a call after the first `advance`; `InvalidConfig`
    /// (field `engines`) when `engine` is not that strategy's; `Engine` for arrays the
    /// engine refuses (`SignalArrays::validate`).
    pub fn set_signals(
        &mut self,
        strategy: usize,
        arrays: SignalArrays,
        engine: &Engine,
    ) -> Result<(), FundError> {
        let refuse = |reason: String| FundError::InvalidConfig {
            field: "conditions",
            reason,
        };
        if self.started {
            return Err(refuse(
                "give a strategy's conditions before the first advance".into(),
            ));
        }
        match self.deciders.get(strategy) {
            None => Err(refuse(format!("the fund runs no strategy {strategy}"))),
            Some(Decider::Signals { .. }) => Err(refuse(format!(
                "strategy {strategy} was given its conditions already"
            ))),
            Some(Decider::Caller) if !self.lanes[strategy].holds(engine) => {
                Err(FundError::InvalidConfig {
                    field: "engines",
                    reason: format!(
                        "the engine given is not the fund's engine of strategy {strategy}"
                    ),
                })
            }
            Some(Decider::Caller) => {
                let check = arrays.validate(engine.instruments().len());
                check.map_err(|error| FundError::Engine { strategy, error })?;
                self.deciders[strategy] = Decider::Signals {
                    arrays: Box::new(arrays),
                    skipped: Vec::new(),
                };
                Ok(())
            }
        }
    }

    /// Run to the next pause the caller decides (module docs), lending the engines in
    /// call order.
    ///
    /// Errors: `InvalidConfig` (field `engines`) when the engines are not the fund's, in
    /// call order; `OutOfStep` when one was advanced outside the fund; `Engine` for an
    /// error of an engine's phase or of a Rust driver; `Finished` after
    /// `FundPause::Finished`.
    pub fn advance(&mut self, engines: &mut [&mut Engine]) -> Result<FundPause, FundError> {
        timeline::check_engines(&self.lanes, engines)?;
        self.started = true;
        self.run(engines)
    }

    /// True once `advance` has returned `FundPause::Finished`.
    pub fn is_finished(&self) -> bool {
        self.stage == Stage::Finished
    }

    /// The account's records, once `advance` has returned `FundPause::Finished`.
    ///
    /// Errors: `NotFinished` before.
    pub fn finish(self) -> Result<FundOutput, FundError> {
        if !self.is_finished() {
            return Err(FundError::NotFinished);
        }
        let skipped = self
            .deciders
            .into_iter()
            .map(|decider| match decider {
                Decider::Caller => Vec::new(),
                Decider::Signals { skipped, .. } => skipped,
            })
            .collect();
        Ok(self.record.into_output(skipped))
    }
}

/// The account's capital (`FundConfig::account_capital_rupees`), rupees.
///
/// Errors: `InvalidConfig` (field `account_capital`) as `Fund::new` says.
fn account_capital(config: &FundConfig, capitals: &[f64]) -> Result<f64, FundError> {
    let sum = round_to_paise(capitals.iter().sum());
    let Some(given) = config.account_capital_rupees else {
        return Ok(sum);
    };
    let refuse = |reason: String| FundError::InvalidConfig {
        field: "account_capital",
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
