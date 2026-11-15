"""The REST layer's checks of an order request, before any service runs: the
schemas of restx_api/schemas.py (`OrderSchema`, `ModifyOrderSchema`, `CancelOrderSchema`)
and restx_api/account_schema.py (`OrderStatusSchema`), with their words. A refused
request is answered HTTP 400 with `str(err.messages)` as its message, a dict of each
field's errors in the schemas' order; an unknown field is refused, as the schemas
declare no other policy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

#: utils/constants.py, VALID_EXCHANGES.
VALID_EXCHANGES = (
    "NSE", "NFO", "CDS", "BSE", "BFO", "BCD", "MCX", "NCDEX", "NCO",
    "NSE_INDEX", "BSE_INDEX", "MCX_INDEX", "GLOBAL_INDEX", "CRYPTO",
)  # fmt: skip
_ACTIONS = ("BUY", "SELL", "buy", "sell")
_PRICE_TYPES = ("MARKET", "LIMIT", "SL", "SL-M")
_PRODUCTS = ("MIS", "NRML", "CNC")
_MISSING = "Missing data for required field."
_KEY_LENGTH = "Length must be between 1 and 256."


@dataclass(frozen=True)
class Field:
    """One field of a schema: its kind ("str", "float" or "int"), whether it is required,
    its default when not, the values it may take, and the lower bound of a number with
    the words of its refusal (`floor_open`: the bound itself is refused)."""

    name: str
    kind: str
    required: bool = True
    default: Any = None
    choices: tuple[str, ...] = ()
    floor: float | None = None
    floor_open: bool = False
    floor_words: str = ""


_KEY = Field("apikey", "str")
_STRATEGY = Field("strategy", "str")
_EXCHANGE = Field("exchange", "str", choices=VALID_EXCHANGES)
_SYMBOL = Field("symbol", "str")
_ORDER_ID = Field("orderid", "str")
_ACTION = Field("action", "str", choices=_ACTIONS)
_QUANTITY = Field(
    "quantity", "float", floor=0, floor_open=True, floor_words="Quantity must be a positive number."
)
_PRICE_WORDS = "Price must be a non-negative number."
_TRIGGER_WORDS = "Trigger price must be a non-negative number."
_DISCLOSED_WORDS = "Disclosed quantity must be a non-negative integer."

PLACE = (
    _KEY, _STRATEGY, _EXCHANGE, _SYMBOL, _ACTION, _QUANTITY,
    Field("pricetype", "str", required=False, default="MARKET", choices=_PRICE_TYPES),
    Field("product", "str", required=False, default="MIS", choices=_PRODUCTS),
    Field("price", "float", required=False, default=0.0, floor=0, floor_words=_PRICE_WORDS),
    Field("trigger_price", "float", required=False, default=0.0, floor=0,
          floor_words=_TRIGGER_WORDS),
    Field("disclosed_quantity", "int", required=False, default=0, floor=0,
          floor_words=_DISCLOSED_WORDS),
    Field("underlying_ltp", "float", required=False),
)  # fmt: skip
MODIFY = (
    _KEY, _STRATEGY, _EXCHANGE, _SYMBOL, _ORDER_ID, _ACTION,
    Field("product", "str", choices=_PRODUCTS),
    Field("pricetype", "str", choices=_PRICE_TYPES),
    Field("price", "float", floor=0, floor_words=_PRICE_WORDS),
    _QUANTITY,
    Field("disclosed_quantity", "int", floor=0, floor_words=_DISCLOSED_WORDS),
    Field("trigger_price", "float", floor=0, floor_words=_TRIGGER_WORDS),
)  # fmt: skip
CANCEL = (_KEY, _STRATEGY, _ORDER_ID)
STATUS = (_KEY, _STRATEGY, _ORDER_ID)


def load(fields: tuple[Field, ...], payload: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """The request as the schema loads it, and "" when it passed; else ({}, the REST
    layer's message)."""
    errors: dict[str, list[str]] = {}
    loaded: dict[str, Any] = {}
    for spec in fields:
        if spec.name not in payload:
            if spec.required:
                errors[spec.name] = [_MISSING]
            elif spec.default is not None:
                loaded[spec.name] = spec.default
            continue
        value, problem = _field(spec, payload[spec.name])
        if problem:
            errors[spec.name] = [problem]
        else:
            loaded[spec.name] = value
    known = {spec.name for spec in fields}
    for name in payload:
        if name not in known:
            errors[name] = ["Unknown field."]
    if not errors and "quantity" in loaded:
        quantity = loaded["quantity"]
        if quantity != int(quantity) and loaded.get("exchange") != "CRYPTO":
            words = f"Fractional quantity ({quantity}) is not allowed for non-crypto exchanges."
            errors["quantity"] = [words]
        loaded["quantity"] = int(quantity)
    return ({}, str(errors)) if errors else (loaded, "")


def _field(spec: Field, raw: Any) -> tuple[Any, str]:
    if spec.kind == "str":
        if not isinstance(raw, str):
            return None, "Not a valid string."
        if spec.name == "apikey" and not 1 <= len(raw) <= 256:
            return None, _KEY_LENGTH
        if spec.choices and raw not in spec.choices:
            return None, f"Must be one of: {', '.join(spec.choices)}."
        return raw, ""
    number = _number(raw, integer=spec.kind == "int")
    if number is None:
        return None, "Not a valid integer." if spec.kind == "int" else "Not a valid number."
    if spec.floor is not None and (
        number < spec.floor or (spec.floor_open and number == spec.floor)
    ):
        return None, spec.floor_words
    return number, ""


def _number(raw: Any, *, integer: bool) -> float | int | None:
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        number = int(raw) if integer else float(raw)
    except (TypeError, ValueError):
        return None
    if not integer and not math.isfinite(number):
        return None
    return number


__all__ = ["CANCEL", "MODIFY", "PLACE", "STATUS", "VALID_EXCHANGES", "Field", "load"]
