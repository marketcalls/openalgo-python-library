//! What the fund refuses: settings and engines it cannot run (`Fund::new`), engines that
//! are not its own or were advanced outside it (`advance`), conditions given too late or
//! twice (`set_signals`), and calls out of order.

use crate::engine::{Engine, KernelError, Pause, RunSetup, SignalArrays};
use crate::model::{Product, Side};

use super::super::{Fund, FundConfig, FundError, FundPause, Pool};
use super::kit::{account, flat, july, market, Run, Script, Symbol, ONE};

fn sbin(capital: f64) -> Engine {
    Run::minutes(Symbol::Sbin, july(7), july(7), capital).engine(|_, _| flat(100.0))
}

fn new_fund(config: FundConfig, engines: &[Engine]) -> Result<Fund, FundError> {
    Fund::new(config, &engines.iter().collect::<Vec<_>>())
}

/// The field a refusal of `new` names.
fn refused_field(result: Result<Fund, FundError>) -> &'static str {
    match result {
        Err(FundError::InvalidConfig { field, .. }) => field,
        other => panic!("not a refused setting: {other:?}"),
    }
}

#[test]
fn a_fund_refuses_settings_it_cannot_run() {
    let shared = |capital, limit| account(Pool::Shared, capital, limit);
    let engines = [sbin(100_000.0), sbin(200_000.0)];
    let cases: [(FundConfig, &str); 6] = [
        (shared(Some(0.0), None), "account_capital"),
        (shared(Some(f64::NAN), None), "account_capital"),
        // A separate pool's account is the sum of the capitals, Rs 3,00,000.
        (
            account(Pool::Separate, Some(250_000.0), None),
            "account_capital",
        ),
        (shared(None, Some(0.0)), "account_max_daily_loss"),
        (shared(None, Some(-5_000.0)), "account_max_daily_loss"),
        (shared(None, Some(f64::INFINITY)), "account_max_daily_loss"),
    ];
    for (config, field) in cases {
        assert_eq!(
            refused_field(new_fund(config, &engines)),
            field,
            "{config:?}"
        );
    }
    let sum = account(Pool::Separate, Some(300_000.0), None);
    assert!(new_fund(sum, &engines).is_ok());
}

#[test]
fn a_fund_refuses_no_engine_a_live_session_and_an_engine_that_has_advanced() {
    let config = account(Pool::Shared, None, None);
    assert_eq!(refused_field(new_fund(config, &[])), "engines");
    let mut advanced = sbin(100_000.0);
    assert!(matches!(advanced.advance(), Ok(Pause::BarClose { .. })));
    assert_eq!(
        refused_field(new_fund(config, &[sbin(1.0), advanced])),
        "engines"
    );
    let session = Run::minutes(Symbol::Sbin, july(7), july(7), 100_000.0);
    let setup: RunSetup = session.setup(|_, _| flat(100.0));
    let live = Engine::new_session(setup).expect("a session");
    assert_eq!(refused_field(new_fund(config, &[live])), "engines");
}

#[test]
fn advance_refuses_engines_that_are_not_the_fund_own_in_call_order() {
    let engines = vec![sbin(100_000.0), sbin(200_000.0)];
    let mut fund = new_fund(account(Pool::Shared, None, None), &engines).expect("a fund");
    let mut swapped: Vec<Engine> = engines.into_iter().rev().collect();
    let mut lent: Vec<&mut Engine> = swapped.iter_mut().collect();
    assert!(matches!(
        fund.advance(&mut lent),
        Err(FundError::InvalidConfig {
            field: "engines",
            ..
        })
    ));
    lent.pop();
    assert!(matches!(
        fund.advance(&mut lent),
        Err(FundError::InvalidConfig {
            field: "engines",
            ..
        })
    ));
}

#[test]
fn advance_refuses_an_engine_advanced_outside_the_fund() {
    let mut engines = vec![sbin(100_000.0)];
    let mut fund = new_fund(account(Pool::Shared, None, None), &engines).expect("a fund");
    let mut lent: Vec<&mut Engine> = engines.iter_mut().collect();
    let first = fund.advance(&mut lent).expect("the first pause");
    assert!(matches!(first, FundPause::Strategy { strategy: 0, .. }));
    // The driver advances its engine itself instead of deciding.
    assert!(lent[0].advance().is_ok());
    assert_eq!(
        fund.advance(&mut lent),
        Err(FundError::OutOfStep { strategy: 0 })
    );
}

#[test]
fn the_fund_finishes_once_and_its_records_wait_for_the_end() {
    let mut engines = vec![sbin(100_000.0)];
    let fund = new_fund(account(Pool::Shared, None, None), &engines).expect("a fund");
    assert_eq!(fund.clone().finish(), Err(FundError::NotFinished));
    let script = Script::new(&[(0, market(Side::Buy, 10.0, Product::Mis))]);
    let mut fund = super::kit::drive(fund, &mut engines, &[script]);
    let mut lent: Vec<&mut Engine> = engines.iter_mut().collect();
    assert_eq!(fund.advance(&mut lent), Err(FundError::Finished));
    assert!(fund.finish().is_ok());
}

/// Arrays of SBIN's run that never enter.
fn quiet_arrays(engine: &Engine) -> SignalArrays {
    let bar_opens: Vec<i64> = engine
        .strategy_bars(ONE)
        .iter()
        .map(|b| b.ts_open)
        .collect();
    SignalArrays {
        instrument: ONE,
        product: Product::Mis,
        exit: Some(vec![false; bar_opens.len()]),
        bar_opens,
        entry: None,
        short: None,
        cover: None,
        pyramiding: 0,
    }
}

#[test]
fn conditions_are_given_once_before_the_first_advance() {
    let mut engines = vec![sbin(100_000.0)];
    let mut fund = new_fund(account(Pool::Shared, None, None), &engines).expect("a fund");
    let arrays = quiet_arrays(&engines[0]);
    let refused = |result| {
        matches!(
            result,
            Err(FundError::InvalidConfig {
                field: "conditions",
                ..
            })
        )
    };
    assert!(refused(fund.set_signals(1, arrays.clone(), &engines[0])));
    let another = sbin(200_000.0);
    assert!(matches!(
        fund.set_signals(0, arrays.clone(), &another),
        Err(FundError::InvalidConfig {
            field: "engines",
            ..
        })
    ));
    let mut short = arrays.clone();
    short.exit = Some(vec![false; 3]);
    assert!(matches!(
        fund.set_signals(0, short, &engines[0]),
        Err(FundError::Engine {
            strategy: 0,
            error: KernelError::InvalidConfig {
                field: "conditions",
                ..
            }
        })
    ));
    assert_eq!(fund.set_signals(0, arrays.clone(), &engines[0]), Ok(()));
    assert!(refused(fund.set_signals(0, arrays.clone(), &engines[0])));
    let mut lent: Vec<&mut Engine> = engines.iter_mut().collect();
    assert_eq!(fund.advance(&mut lent), Ok(FundPause::Finished));
    assert!(refused(fund.set_signals(0, arrays, &engines[0])));
}
