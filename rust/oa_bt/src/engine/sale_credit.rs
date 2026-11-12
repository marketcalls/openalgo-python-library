//! The credit for selling delivery holdings (04-execution 5): the part of a CNC sale's
//! proceeds a broker credits for trading only on the next trading day.
//!
//! The rule is one of shares (cash equity, ETFs included) bought on delivery, which are
//! paid in full and held in the demat account (`paid_in_full`). A future or an option
//! bought on CNC is a margined or premium position like any other product's: its sale
//! withholds nothing and its cost is not taken from the free cash.
//!
//! On every CNC fill of a share (`execute_fill`): a buy adds its units to the scrip's
//! buys of the session; a sale first matches those buys (units bought and sold on one
//! day are an intraday trade, credited in full) and the rest is a sale of holdings,
//! whose proceeds are credited as the dated row of `market::sale_credit` says. The part
//! credited only on the next trading day is withheld from the free cash (`SubAccount::
//! withhold_sale_proceeds`) until the session close (step 8), which credits it and starts
//! a new session's buys.
//!
//! Not modelled (stated in the table file): units bought on the previous trading day and
//! sold before delivery (T1 holdings), whose proceeds brokers credit only the next day,
//! are credited as holdings; and a buy back later on the day of a sale does not release
//! the part withheld.

use crate::charges::MarketTable;
use crate::exec::InstrumentId;
use crate::market::sale_credit::sale_credit_row;
use crate::model::{Day, InstrumentKind, Product, Side};
use crate::portfolio::{FillKind, PositionKey};

use super::error::KernelError;
use super::fills::FillRequest;
use super::kernel::Engine;

/// The CNC units of each scrip bought in the current session and not yet sold, in the
/// order the scrips were first bought.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct SessionBuys {
    bought: Vec<(InstrumentId, f64)>,
}

impl SessionBuys {
    /// Units of `instrument` bought this session and not yet matched by a sale.
    fn slot(&mut self, instrument: InstrumentId) -> &mut f64 {
        let index = match self.bought.iter().position(|(i, _)| *i == instrument) {
            Some(index) => index,
            None => {
                self.bought.push((instrument, 0.0));
                self.bought.len() - 1
            }
        };
        &mut self.bought[index].1
    }
}

impl Engine {
    /// True for a position paid for in full and held on delivery: CNC on a share. Its
    /// cost leaves the free cash and the sale credit applies to its sales.
    fn paid_in_full(&self, key: PositionKey) -> bool {
        // Every position is of an instrument of the run, so the lookup finds it.
        let share = self
            .instruments
            .get(key.instrument.index())
            .is_some_and(|spec| spec.kind == InstrumentKind::Equity);
        share && key.product == Product::Cnc
    }

    /// What the account can trade with, rupees (`Portfolio::free_cash_rupees`), with
    /// the positions `paid_in_full` counted at their cost.
    pub(super) fn free_cash_rupees(&self) -> f64 {
        self.portfolio
            .free_cash_rupees(|key| self.paid_in_full(key))
    }

    /// Note a fill (`request`, filled at `price_rupees` on the IST trade date `day`, with
    /// the instrument's `multiplier`, as an entry, add or reduction `kind`) for the sale
    /// credit: a CNC buy of a share joins the session's buys; a CNC sale of a share
    /// beyond them withholds the part of its proceeds the dated row credits only on the
    /// next trading day, and records the row as used. Other fills are not noted.
    ///
    /// Errors: `SaleCredit` (`InvalidTable`) when the embedded table is malformed.
    pub(super) fn credit_delivery_sale(
        &mut self,
        request: &FillRequest,
        price_rupees: f64,
        multiplier: f64,
        day: Day,
        kind: FillKind,
    ) -> Result<(), KernelError> {
        if !self.paid_in_full(request.key) {
            return Ok(());
        }
        let bought = self.session_buys.slot(request.key.instrument);
        if request.side == Side::Buy {
            *bought += request.units;
            return Ok(());
        }
        if kind != FillKind::Reduce {
            return Ok(());
        }
        let intraday = request.units.min(*bought);
        *bought -= intraday;
        let from_holdings = request.units - intraday;
        if from_holdings <= 0.0 {
            return Ok(());
        }
        let row = sale_credit_row(day)?;
        self.charges
            .record_use(MarketTable::SaleCredit, &row.meta, day);
        let sale_rupees = from_holdings * price_rupees * multiplier;
        self.portfolio
            .account
            .withhold_sale_proceeds(row.next_day_rupees(sale_rupees));
        Ok(())
    }

    /// Step 8 at a session close: credit the proceeds withheld in the session and start
    /// the next session with no buys.
    pub(super) fn credit_withheld_sales(&mut self) {
        self.portfolio.account.credit_sale_proceeds();
        self.session_buys = SessionBuys::default();
    }
}
