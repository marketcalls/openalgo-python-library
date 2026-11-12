"""Schedule versions as dicts, to and from the tuples the kernel exchanges.

A schedule version as users see it:

    {"id": "IN-2026-04-01", "valid_from": date(2026, 4, 1), "valid_to": None,
     "verified_through": date(2026, 11, 12), "gst_rate": 0.18, "gst_on_sebi_fee": True,
     "source": "...", "notes": "...",
     "rates": {"NSE_EQ_DELIVERY": {
         "stt_buy_rate": 0.001, "stt_sell_rate": 0.001, "exchange_txn_rate": 0.0000297,
         "ipft_rate": 0.000001, "sebi_rate": 0.000001, "stamp_buy_rate": 0.00015,
         "stt_exercise_rate": 0.0, "verified_through": date(2026, 11, 12),
         "calculator_tax_base": "turnover", "calculator_tax_unit": "rupee",
         "notes": ""}, ...}}

Units: every key ending in `_rate` is a fraction (0.001 is 0.1 percent).
`verified_through` is the last trade date the figures were checked for (None means
never); the schedule's own value covers GST and is the default for its rows. GST falls
on brokerage and the exchange charge, and on the SEBI fee too when `gst_on_sebi_fee` is
True (from 2022-07-18; the SEBI fee was exempt before).

This module checks the shape of a user's dict (fields present, nothing misspelt);
the kernel checks the values, the plausibility of each rate and the date ranges.
"""

from types import MappingProxyType
from typing import Any

from ..errors import CostError
from .inputs import day_to_date, number, to_day
from .kernel import native

__all__ = ["RATE_FIELDS", "builtin_schedule_views", "schedule_spec", "schedule_view"]

#: The rate fields of one segment row, in kernel order. All are fractions.
RATE_FIELDS = (
    "stt_buy_rate",
    "stt_sell_rate",
    "exchange_txn_rate",
    "ipft_rate",
    "sebi_rate",
    "stamp_buy_rate",
    "stt_exercise_rate",
)

# A segment without an IPFT levy or exercise STT may leave these out.
_OPTIONAL_RATES = ("ipft_rate", "stt_exercise_rate")
# How the Zerodha calculator taxes a row, used only in calculator mode. A user row
# that does not say is treated like most rows: taxed per side, rounded to the rupee.
_ROW_DEFAULTS = MappingProxyType(
    {"calculator_tax_base": "sides", "calculator_tax_unit": "rupee", "notes": ""}
)
_ROW_FIELDS = (*RATE_FIELDS, "verified_through", *_ROW_DEFAULTS)
_SCHEDULE_FIELDS = (
    "id",
    "valid_from",
    "valid_to",
    "verified_through",
    "gst_rate",
    "gst_on_sebi_fee",
    "source",
    "notes",
    "rates",
)
_REQUIRED_FIELDS = ("id", "valid_from", "gst_rate", "rates")
# A user version that does not say taxes the SEBI fee, as GST has since 2022-07-18.
_DEFAULT_GST_ON_SEBI_FEE = True


def _row_view(row: tuple) -> dict[str, Any]:
    _, rates, verified_through, tax_base, tax_unit, notes = row
    return {
        **dict(zip(RATE_FIELDS, rates, strict=True)),
        "verified_through": day_to_date(verified_through),
        "calculator_tax_base": tax_base,
        "calculator_tax_unit": tax_unit,
        "notes": notes,
    }


def schedule_view(raw: tuple) -> dict[str, Any]:
    """A kernel schedule tuple as a dict, with dates in place of day numbers."""
    identifier, valid_from, valid_to, gst_rate, gst_on_sebi_fee, verified_through = raw[:6]
    source, notes, rows = raw[6:]
    return {
        "id": identifier,
        "valid_from": day_to_date(valid_from),
        "valid_to": day_to_date(valid_to),
        "verified_through": day_to_date(verified_through),
        "gst_rate": gst_rate,
        "gst_on_sebi_fee": gst_on_sebi_fee,
        "source": source,
        "notes": notes,
        "rates": {row[0]: _row_view(row) for row in rows},
    }


def builtin_schedule_views() -> list[dict[str, Any]]:
    """The schedule versions compiled into the kernel, oldest first, as dicts."""
    return [schedule_view(raw) for raw in native.cost_builtin_schedules()]


def _reject_unknown(fields: Any, allowed: tuple[str, ...], where: str) -> None:
    unknown = [field for field in fields if field not in allowed]
    if unknown:
        raise CostError(f"{where}: unknown field {unknown[0]!r}; allowed fields are {allowed}")


def _optional_day(value: Any, name: str) -> int | None:
    return None if value is None else to_day(value, name)


def _rate_row(code: Any, row: Any, where: str, verified_through: int | None) -> tuple:
    if not isinstance(code, str):
        raise CostError(f"{where}: segment codes must be strings, got {code!r}")
    where = f"{where}, segment {code}"
    if not isinstance(row, dict):
        raise CostError(f"{where}: the rate row must be a dict of {RATE_FIELDS}")
    _reject_unknown(row, _ROW_FIELDS, where)
    missing = [field for field in RATE_FIELDS if field not in row and field not in _OPTIONAL_RATES]
    if missing:
        raise CostError(f"{where}: missing rate {missing[0]!r}")
    rates = tuple(number(f"{where}: {field}", row.get(field, 0.0)) for field in RATE_FIELDS)
    if "verified_through" in row:
        verified_through = _optional_day(row["verified_through"], f"{where}: verified_through")
    # None means not given, as for the schedule's own text fields.
    texts = [
        str(default if row.get(field) is None else row[field])
        for field, default in _ROW_DEFAULTS.items()
    ]
    return (code.strip().upper(), rates, verified_through, *texts)


def _rate_rows(rates: dict, where: str, verified_through: int | None) -> list[tuple]:
    """The rate rows of a schedule, each segment once.

    Raises CostError when two codes name the same segment once normalised (for
    example "nse_eq_delivery" and "NSE_EQ_DELIVERY"), so neither row is dropped
    silently.
    """
    rows, given_as = [], {}
    for code, row in rates.items():
        converted = _rate_row(code, row, where, verified_through)
        segment = converted[0]
        if segment in given_as:
            raise CostError(
                f"{where}: rates {given_as[segment]!r} and {code!r} are both segment "
                f"{segment}; give each segment once"
            )
        given_as[segment] = code
        rows.append(converted)
    return rows


def schedule_spec(schedule: Any) -> tuple:
    """A user's schedule dict as the tuple the kernel takes.

    `id`, `valid_from`, `gst_rate` and `rates` are required. `valid_to` defaults to
    None (still in force), `verified_through` to None (never verified),
    `gst_on_sebi_fee` to True (GST on the SEBI fee, the rule from 2022-07-18), `source`
    to "user supplied" and `notes` to "". Each rate row needs every field of
    `RATE_FIELDS` except `ipft_rate` and `stt_exercise_rate`, which default to zero;
    it may carry its own `verified_through`, `notes`, `calculator_tax_base` ("sides"
    by default, or "turnover") and `calculator_tax_unit` ("rupee" by default, or
    "paise"); a text field given as None takes its default.

    Raises CostError for a missing or unknown field, so a misspelt rate cannot be
    dropped silently, for a `gst_on_sebi_fee` that is not True or False, and for a
    segment given twice under codes that differ only in case or spaces. Rate values and
    date ranges are validated by the kernel.
    """
    if not isinstance(schedule, dict):
        raise CostError(
            "a schedule must be a dict with id, valid_from, gst_rate and rates; start from an "
            "entry of costs.schedules()"
        )
    where = f"schedule {schedule.get('id')!r}"
    _reject_unknown(schedule, _SCHEDULE_FIELDS, where)
    missing = [field for field in _REQUIRED_FIELDS if schedule.get(field) is None]
    if missing:
        raise CostError(f"{where}: missing field {missing[0]!r}")
    rates = schedule["rates"]
    if not isinstance(rates, dict):
        raise CostError(f"{where}: rates must be a dict of segment code to rate row")
    verified = _optional_day(schedule.get("verified_through"), f"{where}: verified_through")
    gst_on_sebi_fee = schedule.get("gst_on_sebi_fee", _DEFAULT_GST_ON_SEBI_FEE)
    if not isinstance(gst_on_sebi_fee, bool):
        raise CostError(f"{where}: gst_on_sebi_fee must be True or False, got {gst_on_sebi_fee!r}")
    return (
        str(schedule["id"]),
        to_day(schedule["valid_from"], f"{where}: valid_from"),
        _optional_day(schedule.get("valid_to"), f"{where}: valid_to"),
        number(f"{where}: gst_rate", schedule["gst_rate"]),
        gst_on_sebi_fee,
        verified,
        str(schedule.get("source") or "user supplied"),
        str(schedule.get("notes") or ""),
        _rate_rows(rates, where, verified),
    )
