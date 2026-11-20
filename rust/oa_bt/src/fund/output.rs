//! `FundOutput`: what a portfolio run's fund records of the account (18-multi-strategy KR7):
//! the account's row of every round, the account's breaches, the orders the account
//! refused or ended, and the entries the Rust drivers' sizers skipped. Each strategy's
//! own result is its engine's `RunOutput` (`Engine::finish`).
//!
//! Kernel order ids and event stamps are each engine's own (every engine numbers its
//! orders from o1), so a refused order is named by its strategy and its id together.

use crate::engine::{AccountBreach, Event, EventStamp, OrderEventKind, SkippedEntry};
use crate::exec::{OrderId, RejectCode};
use crate::model::{Day, TsNanos};

/// The account's row of every round, as columns, in round order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AccountRows {
    /// The round's close instant, UTC nanoseconds.
    pub ts: Vec<TsNanos>,
    /// The round's IST session day.
    pub day: Vec<Day>,
    /// The account's equity: its capital less the strategies' capitals plus their equity,
    /// rupees.
    pub equity_rupees: Vec<f64>,
    /// The account's free cash: its capital less the strategies' capitals plus their free
    /// cash, rupees.
    pub free_cash_rupees: Vec<f64>,
    /// The margin the strategies' open positions block, rupees.
    pub margin_rupees: Vec<f64>,
    /// The account's daily loss measure on the round's session day (`rules`), rupees,
    /// negative for a loss.
    pub daily_loss_rupees: Vec<f64>,
}

/// Why the account refused or ended an order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// Rejected `account_margin`: the account's free cash did not cover its margin, or
    /// left its sizer nothing.
    AccountMargin,
    /// Rejected `account_loss_gate`: entries were locked by the account's daily loss limit.
    AccountLossGate,
    /// Cancelled working by a breach of the account's daily loss limit (the lock
    /// `account_loss`).
    AccountLoss,
}

impl Refusal {
    /// Every refusal, in the order of their codes.
    pub const ALL: [Refusal; 3] = [
        Refusal::AccountMargin,
        Refusal::AccountLossGate,
        Refusal::AccountLoss,
    ];

    /// The reason's code: the reject code of a rejection, `account_loss` for a cancel.
    pub fn code(self) -> &'static str {
        match self {
            Refusal::AccountMargin => RejectCode::AccountMargin.code(),
            Refusal::AccountLossGate => RejectCode::AccountLossGate.code(),
            Refusal::AccountLoss => "account_loss",
        }
    }

    /// The refusal of a rejection with `code`, when the account made it.
    fn of_reject(code: RejectCode) -> Option<Refusal> {
        match code {
            RejectCode::AccountMargin => Some(Refusal::AccountMargin),
            RejectCode::AccountLossGate => Some(Refusal::AccountLossGate),
            _ => None,
        }
    }
}

/// An order the account refused or ended.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefusedOrder {
    /// The strategy's index in call order.
    pub strategy: usize,
    /// The order, in the strategy's own engine.
    pub order: OrderId,
    /// When, in the strategy's own engine: the rejection's or the cancellation's stamp.
    pub stamp: EventStamp,
    pub refusal: Refusal,
}

/// The account's records of a portfolio run (module docs).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FundOutput {
    /// The account's capital, rupees.
    pub account_capital_rupees: f64,
    pub account: AccountRows,
    /// Every breach of the account's daily loss limit, in time order.
    pub breaches: Vec<AccountBreach>,
    /// The orders the account refused or ended, in the order the fund read them: by turn,
    /// then a turn's rejections before its cancels.
    pub refused: Vec<RefusedOrder>,
    /// Per strategy in call order, the entries its Rust driver's sizer skipped (none for a
    /// strategy its caller decides).
    pub skipped: Vec<Vec<SkippedEntry>>,
}

/// The records as the fund writes them.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Recorder {
    output: FundOutput,
}

impl Recorder {
    /// Nothing recorded yet, for an account of `capital_rupees` over `strategies`.
    pub(super) fn new(capital_rupees: f64, strategies: usize) -> Recorder {
        Recorder {
            output: FundOutput {
                account_capital_rupees: capital_rupees,
                skipped: vec![Vec::new(); strategies],
                ..FundOutput::default()
            },
        }
    }

    /// The account's row of the round closing at `at` on session day `day`:
    /// (equity, free cash, margin, daily loss measure), rupees.
    pub(super) fn row(&mut self, at: TsNanos, day: Day, figures: [f64; 4]) {
        let rows = &mut self.output.account;
        let [equity, free_cash, margin, daily_loss] = figures;
        rows.ts.push(at);
        rows.day.push(day);
        rows.equity_rupees.push(equity);
        rows.free_cash_rupees.push(free_cash);
        rows.margin_rupees.push(margin);
        rows.daily_loss_rupees.push(daily_loss);
    }

    /// A breach of the account's daily loss limit.
    pub(super) fn breach(&mut self, breach: AccountBreach) {
        self.output.breaches.push(breach);
    }

    /// The account's rejections among `events` of strategy `strategy`, then the orders a
    /// breach cancelled, `cancels`.
    pub(super) fn refusals(
        &mut self,
        strategy: usize,
        events: &[Event],
        cancels: Vec<(EventStamp, OrderId)>,
    ) {
        for event in events {
            let Event::Order {
                stamp,
                order,
                kind: OrderEventKind::Rejected(code),
            } = *event
            else {
                continue;
            };
            if let Some(refusal) = Refusal::of_reject(code) {
                self.output.refused.push(RefusedOrder {
                    strategy,
                    order,
                    stamp,
                    refusal,
                });
            }
        }
        let cancelled = cancels.into_iter().map(|(stamp, order)| RefusedOrder {
            strategy,
            order,
            stamp,
            refusal: Refusal::AccountLoss,
        });
        self.output.refused.extend(cancelled);
    }

    /// The records, with each strategy's skipped entries.
    pub(super) fn into_output(self, skipped: Vec<Vec<SkippedEntry>>) -> FundOutput {
        FundOutput {
            skipped,
            ..self.output
        }
    }
}
