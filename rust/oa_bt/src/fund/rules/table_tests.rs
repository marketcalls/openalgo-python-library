//! The account rule's shared table of cases (`test/backtest/data/account_rule_cases.json`,
//! 18-multi-strategy design rule 4), run through `account_measure` and `AccountRules`:
//! each case's measure counted, whether it breaches the daily loss limit and whether it
//! reaches the daily profit target (2.5a rule 6), worked by hand in the table.

use crate::model::{days_from_civil, ts_from_ist, Day};
use crate::test_json::{parse, Json};

use super::{account_measure, AccountRules};
use crate::engine::AccountRule;

const TABLE: &str = include_str!("../../../../../test/backtest/data/account_rule_cases.json");

/// An IST date "YYYY-MM-DD" as days since 1970-01-01.
fn date(text: &str) -> Day {
    let parts: Vec<u32> = text
        .split('-')
        .map(|part| part.parse().expect("a date"))
        .collect();
    let year = i32::try_from(parts[0]).expect("a year");
    days_from_civil(year, parts[1], parts[2])
}

/// An IST time "HH:MM" as minutes since midnight.
fn minute_of_day(text: &str) -> u32 {
    let (hour, minute) = text.split_once(':').expect("a time HH:MM");
    hour.parse::<u32>().expect("an hour") * 60 + minute.parse::<u32>().expect("a minute")
}

/// A case's engines as (measure, session day), in call order.
fn engines(case: &Json) -> Vec<(f64, Option<Day>)> {
    let engine = |row: &Json| {
        let measure = row.get("measure_rupees").number().expect("a measure");
        (measure, row.get("date").text().map(date))
    };
    case.get("engines").list().iter().map(engine).collect()
}

/// The case's measure counted, whether its round breaches the loss limit and whether it
/// reaches the profit target, by the fund's rules.
fn evaluate(case: &Json) -> (f64, bool, bool) {
    let round = case.get("round");
    let day = date(round.get("date").text().expect("the round's date"));
    let at = ts_from_ist(
        day,
        minute_of_day(round.get("time").text().expect("a time")),
    );
    let measure = account_measure(day, engines(case).into_iter());
    let mut rules = AccountRules {
        limit_rupees: case.get("limit_rupees").number(),
        breached_on: case.get("breached_on").text().map(date),
        target_rupees: case.get("profit_target_rupees").number(),
        reached_on: case.get("profit_reached_on").text().map(date),
    };
    let rule = rules.breach(at, day, measure).map(|breach| breach.rule);
    (
        measure,
        rule == Some(AccountRule::Loss),
        rule == Some(AccountRule::Profit),
    )
}

#[test]
fn every_case_of_the_shared_table_gives_its_measure_and_breach() {
    let table = parse(TABLE).expect("the table parses");
    let cases = table.get("cases").list();
    assert!(cases.len() >= 10, "the table holds its cases");
    for case in cases {
        let name = case.get("name").text().expect("a name");
        let expected = case.get("expected");
        let measure = expected.get("measure_rupees").number().expect("a measure");
        let breach = expected.get("breach").boolean().expect("a breach flag");
        let profit = expected.get("profit").boolean().unwrap_or(false);
        assert_eq!(evaluate(case), (measure, breach, profit), "{name}");
    }
}

#[test]
fn a_breach_carries_the_round_the_limit_and_the_measure() {
    let day = days_from_civil(2025, 7, 8);
    let at = ts_from_ist(day, 11 * 60);
    let mut rules = AccountRules::new(Some(10_000.0), None);
    let breach = rules.breach(at, day, -10_000.0).expect("at the limit");
    assert_eq!(
        (
            breach.at,
            breach.day,
            breach.rule,
            breach.limit_rupees,
            breach.measure_rupees
        ),
        (at, day, AccountRule::Loss, 10_000.0, -10_000.0)
    );
    assert_eq!(rules.breach(at + 60_000_000_000, day, -12_000.0), None);
}

#[test]
fn a_profit_target_reached_carries_its_rule_and_fires_once_a_day() {
    let day = days_from_civil(2025, 7, 8);
    let at = ts_from_ist(day, 13 * 60);
    let mut rules = AccountRules::new(Some(10_000.0), Some(8_000.0));
    let reached = rules.breach(at, day, 8_250.0).expect("past the target");
    assert_eq!(
        (reached.rule, reached.limit_rupees, reached.measure_rupees),
        (AccountRule::Profit, 8_000.0, 8_250.0)
    );
    assert_eq!(rules.breach(at + 60_000_000_000, day, 9_000.0), None);
    // The loss limit keeps its own day: a loss past it later that day still breaches.
    let loss = rules.breach(at + 120_000_000_000, day, -10_500.0);
    assert_eq!(loss.map(|breach| breach.rule), Some(AccountRule::Loss));
    let next_day = rules.breach(at + 86_400_000_000_000, day + 1, 8_000.0);
    assert_eq!(
        next_day.map(|breach| breach.rule),
        Some(AccountRule::Profit)
    );
}
