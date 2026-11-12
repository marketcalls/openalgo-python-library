"""Indian trading costs: brokerage presets plus statutory and exchange charges.

Every charge on an Indian trade is itemised: brokerage (from a preset), STT or CTT,
the exchange transaction charge (including the IPFT levy), the SEBI fee, stamp duty
and GST. The rates live in schedule versions with effective dates, and a trade is
charged at the rates in force on its trade date. All arithmetic runs in the Rust
kernel; this module is the Python face of it.

Quick use:

    from openalgo.backtest import costs

    c = costs.trade_cost("SELL", 1100, 400, "2026-11-12", exchange="NSE", product="MIS")
    c.total, c.stt, c.as_dict()

    costs.round_trip_cost(1000, 1100, 400, "2026-11-12", segment="NFO_FUT",
                          rounding="calculator").total        # 294.33

    model = costs.CostModel(brokerage=costs.flat(10))
    trades = model.trades(side, price, quantity, dates, "NSE_EQ_DELIVERY")   # vectorised
    costs.contract_note(trades)                                              # per day

    costs.schedules()     # the built-in versions and how far each row is verified

Units. Prices and costs are rupees. A quantity is the number of price units, never
lots, so that price times quantity is the traded value: shares for an equity, units
of premium for an option, lots times the contract multiplier for an MCX future (GOLD
is quoted per 10 grams with a 1 kg lot, so one lot is 100 units). A key or field
ending in `_rate` is a fraction (0.001 is 0.1 percent); anything called `percent`
takes percent units (`costs.percent(0.03, 20)` is 0.03 percent, at most Rs 20).
Trade dates are IST calendar dates; a date string must be ISO, year first.

One trade is one executed order: brokerage with a per-order cap or flat amount is
charged once per trade. Cost an order that filled in several parts once, on its
total.

Segments: NSE and BSE equity (delivery and intraday; BSE needs the scrip group),
NFO and BFO futures and options (BFO options need the underlying), and MCX futures
(which need the commodity). Currency derivatives (CDS, BCD) and MCX options are not
costed yet and raise CostError. Nothing falls back silently: an unknown or missing
segment, exchange, preset, scrip group, commodity or date raises CostError with a
message that says what to do.

Verification. Each schedule row records the last date its rates were checked
against their source. A cost from rates that are not verified for the trade date
(an older period, a date after the last check, an unconfirmed row) is still
returned, flagged `verified=False` and announced with UnverifiedScheduleWarning.

Rounding: `contract_note` (default) returns unrounded accruals per trade; pass the
frame to `contract_note()` to round once per trade day and contract note, which is
what a broker does (the exact rule is an assumption, reported with the result).
`calculator` reproduces the Zerodha brokerage calculator, to the paisa for equity and
equity derivatives and within one paisa for MCX futures, and is meant for parity
checks.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from ._costs.inputs import DateLike
from ._costs.kernel import (
    COMPONENTS,
    CONTRACT_NOTE_RULE,
    NOTE_GROUPS,
    ROUNDING_MODES,
    SEGMENT_TABLE,
    SEGMENTS,
    call,
    native,
)
from ._costs.model import CostModel
from ._costs.notes import contract_note
from ._costs.presets import Brokerage, custom, flat, percent, preset, zero_brokerage, zerodha
from ._costs.results import CostBreakdown, DpCharge, UnverifiedScheduleWarning
from ._costs.schedules import builtin_schedule_views
from .errors import CostError

__all__ = [
    "COMPONENTS",
    "CONTRACT_NOTE_RULE",
    "NOTE_GROUPS",
    "ROUNDING_MODES",
    "SEGMENTS",
    "Brokerage",
    "CostBreakdown",
    "CostModel",
    "DpCharge",
    "UnverifiedScheduleWarning",
    "contract_note",
    "custom",
    "dp_charge",
    "exercise_stt",
    "flat",
    "percent",
    "preset",
    "round_trip_cost",
    "schedules",
    "segment_for",
    "segments",
    "trade_cost",
    "trade_costs",
    "zero_brokerage",
    "zerodha",
]

_DEFAULT_BROKERAGE = "zerodha"

# The default model of each rounding mode, built once: a model is immutable, and
# building one per call cost ten times the call itself.
_DEFAULT_MODELS: Mapping[str, CostModel] = MappingProxyType(
    {mode: CostModel(_DEFAULT_BROKERAGE, mode) for mode in ROUNDING_MODES}
)


def _model(brokerage: str | Brokerage, rounding: str) -> CostModel:
    """The shared default model when the arguments are the defaults, else a new one."""
    is_default = isinstance(brokerage, str) and brokerage == _DEFAULT_BROKERAGE
    if is_default and rounding in _DEFAULT_MODELS:
        return _DEFAULT_MODELS[rounding]
    return CostModel(brokerage, rounding)


def segments() -> list[dict[str, Any]]:
    """The segment table: one dict per cost segment, in id order.

    Keys: id, code, exchange, brokerage_class (delivery, intraday, futures or
    options), note_group (the contract note its trades appear on) and description.
    """
    return [dict(row) for row in SEGMENT_TABLE]


def segment_for(
    exchange: str,
    product: str,
    instrument: str = "EQ",
    bse_group: str | None = None,
    underlying: str | None = None,
) -> str:
    """The cost segment code of an instrument, from OpenAlgo exchange and product codes.

    - `exchange`: NSE, BSE, NFO, BFO or MCX.
    - `product`: CNC (delivery), MIS (intraday) or NRML (carry-forward F&O).
    - `instrument`: EQ, FUT or OPT (CE and PE are accepted for options).
    - `bse_group`: the BSE scrip group from the instrument master (A, B, M, X, XT, Z,
      ZP, SS, ST and so on). Required for BSE equity, because the exchange charge
      runs from Rs 275 to Rs 1,00,000 per crore depending on it; ignored elsewhere.
    - `underlying`: required for an MCX future (the commodity, for example
      "CRUDEOIL", which decides agri or non-agri rates) and for a BFO option
      ("SENSEX", "BANKEX", "SENSEX50" or the stock symbol: Sensex and Bankex
      options pay a different exchange charge from other BSE options).

    Raises CostError for currency derivatives (CDS, BCD) and MCX options, which are
    not costed yet, for a combination that does not exist, for a missing BSE group
    or underlying, and for any name it does not recognise.
    """
    texts = {"exchange": exchange, "product": product, "instrument": instrument}
    for name, value in texts.items():
        if not isinstance(value, str):
            raise CostError(f"{name} must be a string, got {value!r}")
    for name, value in (("bse_group", bse_group), ("underlying", underlying)):
        if value is not None and not isinstance(value, str):
            raise CostError(f"{name} must be a string or None, got {value!r}")
    codes = [texts[name].strip().upper() for name in ("exchange", "instrument", "product")]
    return call(native.cost_resolve_segment, *codes, bse_group, underlying)


def _segment(
    segment: str | None,
    exchange: str | None,
    product: str | None,
    instrument: str,
    bse_group: str | None,
    underlying: str | None,
) -> str:
    """The segment code from either `segment` or an exchange and product."""
    if segment is not None:
        if exchange is not None or product is not None:
            raise CostError("pass either segment, or exchange and product, not both")
        return segment
    if exchange is None or product is None:
        raise CostError(
            "say what is being traded: pass segment='NSE_EQ_DELIVERY' (see costs.SEGMENTS), "
            "or exchange and product such as exchange='NSE', product='CNC'"
        )
    return segment_for(exchange, product, instrument, bse_group, underlying)


def trade_cost(
    side: str | int,
    price: float,
    quantity: float,
    date: DateLike,
    *,
    segment: str | None = None,
    exchange: str | None = None,
    product: str | None = None,
    instrument: str = "EQ",
    bse_group: str | None = None,
    underlying: str | None = None,
    brokerage: str | Brokerage = "zerodha",
    rounding: str = "contract_note",
) -> CostBreakdown:
    """The cost of one trade (one executed order) under the built-in schedules.

    `side` is "BUY" or "SELL" (or +1 or -1), `price` is rupees per price unit,
    `quantity` the number of price units (never lots) and `date` the IST trade date.
    Say what is traded either with `segment` (a code from `costs.SEGMENTS`) or with
    `exchange` and `product` (plus `instrument`, `bse_group` and `underlying` where
    they are needed, see `segment_for`).

        costs.trade_cost("BUY", 1000, 400, "2026-11-12", exchange="NSE", product="CNC")

    Raises CostError for unknown or invalid input and for a date before the first
    schedule version. Warns when the rates are not verified for `date`. See
    `CostModel.trade`.
    """
    code = _segment(segment, exchange, product, instrument, bse_group, underlying)
    return _model(brokerage, rounding).trade(side, price, quantity, date, code)


def round_trip_cost(
    buy_price: float,
    sell_price: float,
    quantity: float,
    date: DateLike,
    *,
    sell_date: DateLike | None = None,
    segment: str | None = None,
    exchange: str | None = None,
    product: str | None = None,
    instrument: str = "EQ",
    bse_group: str | None = None,
    underlying: str | None = None,
    brokerage: str | Brokerage = "zerodha",
    rounding: str = "contract_note",
) -> CostBreakdown:
    """The cost of a buy and a sell of the same quantity under the built-in schedules.

        costs.round_trip_cost(1000, 1100, 400, "2026-11-12", segment="NSE_EQ_INTRADAY",
                              rounding="calculator").total     # 200.62

    Raises and warns as `trade_cost` does. See `CostModel.round_trip`.
    """
    code = _segment(segment, exchange, product, instrument, bse_group, underlying)
    model = _model(brokerage, rounding)
    return model.round_trip(buy_price, sell_price, quantity, date, code, sell_date=sell_date)


def trade_costs(
    side: Any,
    price: Any,
    quantity: Any,
    date: Any,
    segment: Any,
    *,
    brokerage: str | Brokerage = "zerodha",
    rounding: str = "contract_note",
    as_frame: bool = True,
) -> pd.DataFrame | dict[str, np.ndarray]:
    """Cost many trades at once from scalars and arrays, under the built-in schedules.

    Returns a DataFrame with one row per trade: its date and segment, the rupee
    amounts, and the schedule version and verification flags of its rates. Raises
    CostError naming the first bad row. See `CostModel.trades`.
    """
    model = _model(brokerage, rounding)
    return model.trades(side, price, quantity, date, segment, as_frame=as_frame)


def dp_charge(date: DateLike, brokerage: str | Brokerage = "zerodha") -> DpCharge:
    """The DP charge for selling one scrip from holdings on `date`, in rupees.

    Charged per scrip per sell day, not per trade; the caller decides when it applies.
    The result says whose fee it is. Raises CostError when no schedule version covers
    `date`. See `CostModel.dp_charge`.
    """
    return _model(brokerage, ROUNDING_MODES[0]).dp_charge(date)


def exercise_stt(
    intrinsic: float,
    quantity: float,
    date: DateLike,
    segment: str = "NFO_OPT",
    *,
    rounding: str = "contract_note",
) -> float:
    """STT in rupees on an exercised option, at the rate in force on `date`.

    `intrinsic` is the intrinsic value in rupees per unit and `quantity` is in units.
    Raises CostError for a segment that is not an options segment. See
    `CostModel.exercise_stt`.
    """
    return _model(_DEFAULT_BROKERAGE, rounding).exercise_stt(intrinsic, quantity, date, segment)


def schedules() -> list[dict[str, Any]]:
    """The built-in schedule versions, oldest first, with their verification status.

    The first version starts on 2020-07-01, the day stamp duty on exchange trades
    became one rate across states.

    Each is a dict with id, valid_from, valid_to and verified_through (dates;
    valid_to is None while in force, verified_through None when never verified),
    gst_rate, source, notes and rates. Each row of rates carries its own
    verified_through and notes. A trade dated before the first version raises
    CostError unless the model was built with `backfill=True`. To add or change a
    version, copy one of these dicts, edit it and pass it to
    `CostModel(schedules=[...])`.
    """
    return builtin_schedule_views()
