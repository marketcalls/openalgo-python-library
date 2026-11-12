//! Reference note section 3, "Commodity derivatives": the MCX futures column, the
//! contract-specific rates and the agri list. Amounts are rupees on a fill of Rs 1
//! crore unless a row says otherwise.

use crate::model::{Exchange, InstrumentKind, Product, Side};

use super::super::{CostError, NameKind, Segment};
use super::{accrued, assert_close, assert_paise, calculator_model, snapshot_day, SameDay, CRORE};

const BOTH_SIDES: [Side; 2] = [Side::Buy, Side::Sell];

const MCX_SEGMENTS: [Segment; 7] = [
    Segment::McxFut,
    Segment::McxFutAgri,
    Segment::McxFutAgriCtt,
    Segment::McxFutCastorseed,
    Segment::McxFutRbdpmolein,
    Segment::McxFutPepper,
    Segment::McxFutKapas,
];

const AGRI_SEGMENTS: [Segment; 6] = [
    Segment::McxFutAgri,
    Segment::McxFutAgriCtt,
    Segment::McxFutCastorseed,
    Segment::McxFutRbdpmolein,
    Segment::McxFutPepper,
    Segment::McxFutKapas,
];

/// The agri segments whose commodities are in rule 3 of the CTT Rules 2013.
const CTT_EXEMPT_SEGMENTS: [Segment; 4] = [
    Segment::McxFutAgri,
    Segment::McxFutCastorseed,
    Segment::McxFutPepper,
    Segment::McxFutKapas,
];

#[test]
fn brokerage_row_is_the_lower_of_3_bps_and_20_rupees() {
    // The calculator's "flat Rs 20 above Rs 2 lakh" is the same rule: 0.03 percent
    // of Rs 2 lakh is already Rs 60.
    let rows = [
        (50_000.0, 15.0),
        (150_000.0, 20.0),
        (200_001.0, 20.0),
        (5_000_000.0, 20.0),
    ];
    for segment in MCX_SEGMENTS {
        for side in BOTH_SIDES {
            for (value, brokerage) in rows {
                assert_close(accrued(segment, side, value).brokerage, brokerage);
            }
        }
    }
}

#[test]
fn ctt_row_non_agri_futures_pay_one_bp_on_sells() {
    assert_eq!(accrued(Segment::McxFut, Side::Buy, CRORE).stt, 0.0);
    assert_close(accrued(Segment::McxFut, Side::Sell, CRORE).stt, 1_000.0);
}

#[test]
fn ctt_row_agri_futures_outside_rule_3_pay_one_bp_on_sells() {
    for segment in [Segment::McxFutAgriCtt, Segment::McxFutRbdpmolein] {
        assert_eq!(accrued(segment, Side::Buy, CRORE).stt, 0.0, "{segment:?}");
        assert_close(accrued(segment, Side::Sell, CRORE).stt, 1_000.0);
    }
}

#[test]
fn ctt_row_agri_futures_in_rule_3_pay_no_ctt() {
    for segment in CTT_EXEMPT_SEGMENTS {
        for side in BOTH_SIDES {
            assert_eq!(accrued(segment, side, CRORE).stt, 0.0, "{segment:?}");
        }
    }
}

#[test]
fn exchange_txn_row_standard_and_contract_specific_rates() {
    // 0.0021 percent standard; CASTORSEED 0.0005, RBDPMOLEIN 0.001, PEPPER 0.00005
    // and KAPAS 0.0026 percent.
    let rows = [
        (Segment::McxFut, 210.0),
        (Segment::McxFutAgri, 210.0),
        (Segment::McxFutAgriCtt, 210.0),
        (Segment::McxFutCastorseed, 50.0),
        (Segment::McxFutRbdpmolein, 100.0),
        (Segment::McxFutPepper, 5.0),
        (Segment::McxFutKapas, 260.0),
    ];
    for (segment, per_crore) in rows {
        for side in BOTH_SIDES {
            assert_close(accrued(segment, side, CRORE).exchange_txn, per_crore);
        }
    }
}

#[test]
fn sebi_fee_row_is_ten_rupees_per_crore_and_one_rupee_for_agri() {
    for side in BOTH_SIDES {
        assert_close(accrued(Segment::McxFut, side, CRORE).sebi_fee, 10.0);
        for segment in AGRI_SEGMENTS {
            assert_close(accrued(segment, side, CRORE).sebi_fee, 1.0);
        }
    }
}

#[test]
fn stamp_duty_row_is_two_hundred_rupees_per_crore_on_buys() {
    for segment in MCX_SEGMENTS {
        assert_close(accrued(segment, Side::Buy, CRORE).stamp_duty, 200.0);
        assert_eq!(accrued(segment, Side::Sell, CRORE).stamp_duty, 0.0);
    }
}

#[test]
fn gst_row_is_18_percent_of_brokerage_exchange_charge_and_sebi_fee() {
    // On a Rs 1 crore sell. CTT is not in the base.
    //   non-agri: 0.18 x (20 + 210 + 10) = 43.20
    //   agri:     0.18 x (20 + 210 + 1) = 41.58
    //   PEPPER:   0.18 x (20 + 5 + 1) = 4.68
    let rows = [
        (Segment::McxFut, 43.20),
        (Segment::McxFutAgri, 41.58),
        (Segment::McxFutAgriCtt, 41.58),
        (Segment::McxFutPepper, 4.68),
    ];
    for (segment, gst) in rows {
        assert_close(accrued(segment, Side::Sell, CRORE).gst, gst);
    }
}

#[test]
fn agri_list_commodities_resolve_to_agri_segments() {
    // Rule 3 of the CTT Rules 2013 (as substituted in 2015) lists cardamom (iii),
    // cotton (viii), castor seed (iv), kapas (xii) and pepper (xiv); crude palm oil,
    // mentha oil, rubber, cotton seed wash oil, sunflower oil and RBD palmolein are
    // not in it.
    let rows = [
        ("CARDAMOM", Segment::McxFutAgri),
        ("COTTONCNDY", Segment::McxFutAgri),
        ("COTTONOIL", Segment::McxFutAgriCtt),
        ("CPO", Segment::McxFutAgriCtt),
        ("MENTHAOIL", Segment::McxFutAgriCtt),
        ("RUBBER", Segment::McxFutAgriCtt),
        ("SUNOIL", Segment::McxFutAgriCtt),
        ("CASTORSEED", Segment::McxFutCastorseed),
        ("RBDPMOLEIN", Segment::McxFutRbdpmolein),
        ("PEPPER", Segment::McxFutPepper),
        ("KAPAS", Segment::McxFutKapas),
    ];
    for (commodity, segment) in rows {
        assert_eq!(Segment::mcx_future(commodity), Ok(segment), "{commodity}");
    }
}

#[test]
fn bullion_metals_energy_and_indices_resolve_to_the_non_agri_segment() {
    let commodities = [
        "CRUDEOIL",
        "CRUDEOILM",
        "NATURALGAS",
        "NATGASMINI",
        "GOLD",
        "GOLDM",
        "GOLDPETAL",
        "GOLDGUINEA",
        "GOLDTEN",
        "SILVER",
        "SILVERM",
        "SILVERMIC",
        "SILVER100",
        "COPPER",
        "ZINC",
        "ZINCMINI",
        "LEAD",
        "LEADMINI",
        "ALUMINIUM",
        "ALUMINI",
        "NICKEL",
        "MCXBULLDEX",
        "MCXMETLDEX",
        "MCXENRGDEX",
        "ELECDMBL",
    ];
    for commodity in commodities {
        assert_eq!(
            Segment::mcx_future(commodity),
            Ok(Segment::McxFut),
            "{commodity}"
        );
    }
}

#[test]
fn commodity_names_ignore_case_and_surrounding_spaces() {
    assert_eq!(Segment::mcx_future(" silver "), Ok(Segment::McxFut));
    assert_eq!(Segment::mcx_future("kapas"), Ok(Segment::McxFutKapas));
}

#[test]
fn an_unknown_commodity_is_an_error_not_a_guess() {
    let error = Segment::mcx_future("UNOBTAINIUM").unwrap_err();
    assert_eq!(
        error,
        CostError::UnknownName {
            kind: NameKind::Commodity,
            name: "UNOBTAINIUM".to_string()
        }
    );
    for code in ["MCX_FUT for", "MCX_FUT_AGRI for", "MCX_FUT_AGRI_CTT for"] {
        assert!(error.to_string().contains(code), "{error}");
    }
}

#[test]
fn an_mcx_future_needs_its_commodity_to_be_resolved() {
    let resolve = |commodity| {
        Segment::resolve(
            Exchange::Mcx,
            InstrumentKind::Future,
            Product::Nrml,
            None,
            commodity,
        )
    };
    assert_eq!(resolve(Some("GOLD")), Ok(Segment::McxFut));
    assert_eq!(resolve(Some("CARDAMOM")), Ok(Segment::McxFutAgri));
    assert_eq!(resolve(Some("CPO")), Ok(Segment::McxFutAgriCtt));
    assert!(matches!(
        resolve(None),
        Err(CostError::InvalidCombination { .. })
    ));
}

#[test]
fn calculator_mode_non_agri_round_trip() {
    // CRUDEOIL, one lot of 100 barrels, buy 5500 and sell 5520, by the calculator's
    // rules for commodity futures:
    //   turnover 11,02,000; brokerage 20 + 20 = 40
    //   CTT 0.01 percent of the sell value 5,52,000 = 55.20, kept in paise
    //   exchange 0.0021 percent of turnover = 23.142, rounded 23.14
    //   SEBI Rs 10 per crore = 1.102, rounded 1.10
    //   GST 0.18 x (40 + 23.14 + 1.10) = 11.5632, rounded 11.56
    //   stamp duty 0.002 percent of the buy value 5,50,000 = 11
    //   total 142.00
    let cost = calculator_model()
        .round_trip_on(Segment::McxFut, 5_500.0, 5_520.0, 100.0, snapshot_day())
        .unwrap();
    assert_paise(cost.brokerage, 40.0);
    assert_paise(cost.stt, 55.20);
    assert_paise(cost.exchange_txn, 23.14);
    assert_paise(cost.sebi_fee, 1.10);
    assert_paise(cost.gst, 11.56);
    assert_paise(cost.stamp_duty, 11.0);
    assert_paise(cost.total(), 142.00);
}

#[test]
fn calculator_mode_ctt_exempt_agri_round_trip() {
    // A CTT-exempt agri future, 360 units, buy 900 and sell 905:
    //   turnover 6,49,800; brokerage 20 + 20 = 40; no CTT
    //   exchange 0.0021 percent = 13.6458, rounded 13.65
    //   SEBI Rs 1 per crore = 0.06498, rounded 0.06
    //   GST 0.18 x (40 + 13.65 + 0.06) = 9.6678, rounded 9.67
    //   stamp duty 0.002 percent of 3,24,000 = 6.48, rounded to Rs 6
    //   total 69.38
    let cost = calculator_model()
        .round_trip_on(Segment::McxFutAgri, 900.0, 905.0, 360.0, snapshot_day())
        .unwrap();
    assert_paise(cost.brokerage, 40.0);
    assert_eq!(cost.stt, 0.0);
    assert_paise(cost.exchange_txn, 13.65);
    assert_paise(cost.sebi_fee, 0.06);
    assert_paise(cost.gst, 9.67);
    assert_paise(cost.stamp_duty, 6.0);
    assert_paise(cost.total(), 69.38);
}

#[test]
fn calculator_mode_ctt_paying_agri_round_trip() {
    // MENTHAOIL, one lot of 360 kg, buy 900 and sell 905: the exempt round trip above
    // plus CTT of 0.01 percent of the sell value 3,25,800 = 32.58, kept in paise.
    //   total 69.38 + 32.58 = 101.96
    let cost = calculator_model()
        .round_trip_on(Segment::McxFutAgriCtt, 900.0, 905.0, 360.0, snapshot_day())
        .unwrap();
    assert_paise(cost.stt, 32.58);
    assert_paise(cost.sebi_fee, 0.06);
    assert_paise(cost.total(), 101.96);
}

#[test]
fn calculator_mode_is_within_one_paisa_of_the_calculator_for_mcx() {
    // CRUDEOILM (multiplier 10), 5 lots, buy 5403 and sell 5485. The calculator
    // forms CTT as 0.0001 x 5485 x 5 x 10 and gets 27.42, total 93.74; this engine
    // takes 50 units, forms 0.0001 x (5485 x 50) and may land a paisa away.
    let model = calculator_model();
    let cost = model
        .round_trip_on(Segment::McxFut, 5_403.0, 5_485.0, 50.0, snapshot_day())
        .unwrap();
    assert!((cost.stt - 27.42).abs() <= 0.01 + 1e-9, "{}", cost.stt);
    assert!((model.reported_total(&cost) - 93.74).abs() <= 0.01 + 1e-9);
}
