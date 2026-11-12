//! The segment table and the resolution of instruments to segments.

use crate::model::{Exchange, InstrumentKind, Product};

use super::super::{BrokerageClass, CostError, NameKind, NoteGroup, Segment, SEGMENT_COUNT};

/// Resolve with every detail given: BSE group A and underlying GOLD.
fn resolve(
    exchange: Exchange,
    kind: InstrumentKind,
    product: Product,
) -> Result<Segment, CostError> {
    Segment::resolve(exchange, kind, product, Some("A"), Some("GOLD"))
}

fn bse_equity(product: Product, group: Option<&str>) -> Result<Segment, CostError> {
    Segment::resolve(Exchange::Bse, InstrumentKind::Equity, product, group, None)
}

fn bfo_option(underlying: Option<&str>) -> Result<Segment, CostError> {
    Segment::resolve(
        Exchange::Bfo,
        InstrumentKind::Option,
        Product::Nrml,
        None,
        underlying,
    )
}

#[test]
fn segment_ids_are_their_positions_in_the_table() {
    assert_eq!(Segment::ALL.len(), SEGMENT_COUNT);
    for (index, segment) in Segment::ALL.into_iter().enumerate() {
        assert_eq!(segment as usize, index);
        assert_eq!(Segment::from_u8(index as u8), Some(segment));
    }
}

#[test]
fn an_id_past_the_table_is_not_a_segment() {
    assert_eq!(Segment::from_u8(SEGMENT_COUNT as u8), None);
    assert_eq!(Segment::from_u8(u8::MAX), None);
}

#[test]
fn segment_codes_round_trip_and_are_unique() {
    let mut codes: Vec<&str> = Segment::ALL.iter().map(|segment| segment.code()).collect();
    for segment in Segment::ALL {
        assert_eq!(Segment::from_code(segment.code()), Ok(segment));
    }
    codes.sort_unstable();
    codes.dedup();
    assert_eq!(codes.len(), SEGMENT_COUNT);
}

#[test]
fn segment_codes_start_with_the_exchange_code() {
    for segment in Segment::ALL {
        assert!(
            segment.code().starts_with(segment.exchange().code()),
            "{segment:?}"
        );
    }
}

#[test]
fn options_segments_are_exactly_the_options_brokerage_class() {
    let options = [Segment::NfoOpt, Segment::BfoOpt, Segment::BfoOptOther];
    for segment in Segment::ALL {
        let is_option = options.contains(&segment);
        assert_eq!(segment.is_option(), is_option, "{segment:?}");
        assert_eq!(
            segment.brokerage_class() == BrokerageClass::Options,
            is_option
        );
    }
}

#[test]
fn each_exchange_has_its_own_contract_note() {
    let rows = [
        (Segment::NseEqDelivery, NoteGroup::NseCash),
        (Segment::NseEqIntraday, NoteGroup::NseCash),
        (Segment::BseEqDelivery, NoteGroup::BseCash),
        (Segment::BseEqIntradaySs, NoteGroup::BseCash),
        (Segment::NfoFut, NoteGroup::NseFo),
        (Segment::NfoOpt, NoteGroup::NseFo),
        (Segment::BfoFut, NoteGroup::BseFo),
        (Segment::BfoOpt, NoteGroup::BseFo),
        (Segment::BfoOptOther, NoteGroup::BseFo),
        (Segment::McxFut, NoteGroup::Mcx),
        (Segment::McxFutKapas, NoteGroup::Mcx),
    ];
    for (segment, group) in rows {
        assert_eq!(segment.note_group(), group, "{segment:?}");
    }
}

#[test]
fn exchange_kind_and_product_resolve_to_the_segment() {
    use InstrumentKind::{Equity, Future, Option as Opt};
    let rows = [
        (Exchange::Nse, Equity, Product::Cnc, Segment::NseEqDelivery),
        (Exchange::Nse, Equity, Product::Mis, Segment::NseEqIntraday),
        (Exchange::Bse, Equity, Product::Cnc, Segment::BseEqDelivery),
        (Exchange::Bse, Equity, Product::Mis, Segment::BseEqIntraday),
        (Exchange::Nfo, Future, Product::Nrml, Segment::NfoFut),
        (Exchange::Nfo, Future, Product::Mis, Segment::NfoFut),
        (Exchange::Nfo, Opt, Product::Nrml, Segment::NfoOpt),
        (Exchange::Nfo, Opt, Product::Mis, Segment::NfoOpt),
        (Exchange::Bfo, Future, Product::Nrml, Segment::BfoFut),
        (Exchange::Mcx, Future, Product::Nrml, Segment::McxFut),
        (Exchange::Mcx, Future, Product::Mis, Segment::McxFut),
    ];
    for (exchange, kind, product, segment) in rows {
        assert_eq!(resolve(exchange, kind, product), Ok(segment));
    }
}

#[test]
fn a_bse_equity_trade_without_its_scrip_group_is_an_error() {
    // The charge runs from Rs 275 to Rs 1,00,000 per crore by group, so no group is
    // assumed: a Z-group scrip costed as group A would pay 3.75 instead of 100 on
    // Rs 1 lakh.
    for product in [Product::Cnc, Product::Mis] {
        assert_eq!(bse_equity(product, None), Err(CostError::MissingBseGroup));
    }
}

#[test]
fn the_missing_group_message_lists_the_groups_and_where_the_group_comes_from() {
    let message = CostError::MissingBseGroup.to_string();
    for expected in ["instrument master", "A, B, E", "ZP", "SS, ST"] {
        assert!(message.contains(expected), "{message}");
    }
}

#[test]
fn the_bse_group_selects_the_bse_equity_segment() {
    // Section 1, BSE group table. XC and XD are the former names of group X.
    let rows = [
        (
            Segment::BseEqDelivery,
            Segment::BseEqIntraday,
            &["A", "B", "E", "F", "FC", "G", "GC", "W", "T"][..],
        ),
        (
            Segment::BseEqDeliveryM,
            Segment::BseEqIntradayM,
            &["M", "MT", "TS", "MS"][..],
        ),
        (
            Segment::BseEqDeliveryX,
            Segment::BseEqIntradayX,
            &["X", "XC", "XD", "XT", "Z", "ZP"][..],
        ),
        (
            Segment::BseEqDeliverySs,
            Segment::BseEqIntradaySs,
            &["SS", "ST"][..],
        ),
    ];
    for (delivery, intraday, groups) in rows {
        for group in groups {
            assert_eq!(
                bse_equity(Product::Cnc, Some(group)),
                Ok(delivery),
                "{group}"
            );
            assert_eq!(
                bse_equity(Product::Mis, Some(group)),
                Ok(intraday),
                "{group}"
            );
        }
    }
}

#[test]
fn bse_scrip_groups_ignore_case_and_surrounding_spaces() {
    assert_eq!(
        bse_equity(Product::Cnc, Some(" zp ")),
        Ok(Segment::BseEqDeliveryX)
    );
}

#[test]
fn an_unknown_bse_group_is_an_error_not_group_a() {
    assert_eq!(
        bse_equity(Product::Cnc, Some("QQ")),
        Err(CostError::UnknownName {
            kind: NameKind::BseGroup,
            name: "QQ".to_string()
        })
    );
}

#[test]
fn the_bse_group_is_ignored_off_bse() {
    let resolved = Segment::resolve(
        Exchange::Nse,
        InstrumentKind::Equity,
        Product::Cnc,
        Some("Z"),
        None,
    );
    assert_eq!(resolved, Ok(Segment::NseEqDelivery));
}

#[test]
fn sensex_and_bankex_options_are_one_segment_and_other_bse_options_another() {
    // BSE notice of 2024-09-27: Rs 3,250 per crore of premium for Sensex and Bankex
    // options, Rs 500 for Sensex 50 and stock options.
    let rows = [
        ("SENSEX", Segment::BfoOpt),
        ("BANKEX", Segment::BfoOpt),
        (" sensex ", Segment::BfoOpt),
        ("SENSEX50", Segment::BfoOptOther),
        ("RELIANCE", Segment::BfoOptOther),
    ];
    for (underlying, segment) in rows {
        assert_eq!(bfo_option(Some(underlying)), Ok(segment), "{underlying}");
    }
}

#[test]
fn a_bfo_option_without_its_underlying_is_an_error() {
    let error = bfo_option(None).unwrap_err();
    assert!(matches!(error, CostError::InvalidCombination { .. }));
    assert!(error.to_string().contains("pass the underlying"), "{error}");
}

#[test]
fn an_unknown_segment_code_is_an_error() {
    // Bare exchange names, other spellings and lower case do not resolve.
    for code in [
        "NSE",
        "NFO",
        "NSE-INTRADAY",
        "NFO-OPT",
        "nse_eq_delivery",
        "NSE_EQ",
        "",
        "MCX",
    ] {
        assert_eq!(
            Segment::from_code(code),
            Err(CostError::UnknownName {
                kind: NameKind::Segment,
                name: code.to_string()
            }),
            "{code:?}"
        );
    }
}

#[test]
fn currency_and_mcx_option_codes_are_unsupported() {
    // Section 2 (currency) and MCX options are out of scope.
    for code in ["CDS_FUT", "CDS_OPT", "BCD_FUT", "BCD_OPT", "MCX_OPT"] {
        let result = Segment::from_code(code);
        assert!(
            matches!(result, Err(CostError::UnsupportedSegment(_))),
            "{code}"
        );
    }
}

#[test]
fn currency_exchanges_are_unsupported() {
    for exchange in [Exchange::Cds, Exchange::Bcd] {
        for kind in [InstrumentKind::Future, InstrumentKind::Option] {
            let result = resolve(exchange, kind, Product::Nrml);
            assert!(
                matches!(result, Err(CostError::UnsupportedSegment(_))),
                "{exchange:?}"
            );
        }
    }
}

#[test]
fn mcx_options_are_unsupported() {
    let result = resolve(Exchange::Mcx, InstrumentKind::Option, Product::Nrml);
    assert_eq!(
        result,
        Err(CostError::UnsupportedSegment("MCX options".to_string()))
    );
}

#[test]
fn index_exchanges_are_not_tradable() {
    for exchange in [Exchange::NseIndex, Exchange::BseIndex] {
        let result = resolve(exchange, InstrumentKind::Index, Product::Mis);
        assert_eq!(result, Err(CostError::NotTradable(exchange)));
    }
}

#[test]
fn combinations_that_do_not_exist_are_errors() {
    use InstrumentKind::{Equity, Future, Option as Opt};
    let rows = [
        (Exchange::Nse, Equity, Product::Nrml),
        (Exchange::Bse, Equity, Product::Nrml),
        (Exchange::Nse, Future, Product::Nrml),
        (Exchange::Nse, Opt, Product::Mis),
        (Exchange::Bse, Opt, Product::Nrml),
        (Exchange::Nfo, Future, Product::Cnc),
        (Exchange::Nfo, Equity, Product::Mis),
        (Exchange::Bfo, Opt, Product::Cnc),
        (Exchange::Mcx, Equity, Product::Mis),
        (Exchange::Mcx, Future, Product::Cnc),
    ];
    for (exchange, kind, product) in rows {
        let result = resolve(exchange, kind, product);
        assert!(
            matches!(result, Err(CostError::InvalidCombination { .. })),
            "{exchange:?} {kind:?} {product:?}: {result:?}"
        );
    }
}
