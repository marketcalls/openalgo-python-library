"""Sizers: `bt.size.qty`, `lots`, `value`, `percent`, `fixed_risk` and `atr_risk`, a
strategy's own sizing function, and the caps shared by every sizer, `bt.size.caps`.

A sizer is passed as `size=` on an order or set as the strategy's default (the class
attribute `size`). The kernel turns it into a quantity in units at the decision price,
cuts it to the strategy's caps (`size_caps`), rounds it down to the venue rule (whole
lots; whole shares, the default, unless the run is `quantity_mode="fractional"`), caps
it at the freeze quantity (whole lots) and skips the order with a logged reason when
nothing is left ("rounded_to_zero", or "cap" when a cap left no room). Percent
arguments are percent units (`percent(50)` is 50 percent). Raises ConfigError for a
value that is not positive.

A sizing function is any callable `size(ctx, order) -> units`, called at the decision.
`ctx` holds what the built-in sizers see (`_sizing_function.SizingInputs`: equity,
capital, cash, price, stop_distance, lot_size, multiplier, fractional, position) and
`order` the order being sized (symbol, side, product, tag). It returns a quantity in
units, never lots: 0 skips the order, anything negative or not a number is a
ConfigError. Condition strategies run in Rust and cannot call it.

Equity means the sizing equity: the current equity (compounding), or the capital
allocated at the start when the strategy sets `reinvest = False`.

An order with no `qty`, `lots` or `size` (and no strategy `size`) takes the engine
default: one lot for futures and options; for cash equity 1x the
strategy's equity (never more than its free cash, split equally across its
instruments) net of the order's estimated costs. In venue mode (the default, whole
shares) it is sized at the decision close and keeps 2 percent of that budget in cash for
the price's move to the next open (the kernel's `VENUE_CASH_BUFFER`, the same in live
trading; the Assumptions block's `sizing` line states it), the estimated costs coming off
the rest: with no charges, 10,00,000 at 1,000.00 buys 980 shares, not 1,000. In
fractional mode it is sized at the fill price and spends the whole budget less the
costs.

The sizers here keep no cash buffer. They size at the decision close, so a budget of the
whole equity (`percent(100)`) can need more than the cash when the next open is higher,
and the order is then rejected for margin; leave room (`percent(95)`) or use the engine
default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._atr import sizer_atr
from ._series import SeriesStore
from .errors import ConfigError

BASES = ("equity", "capital", "cash")


def _positive(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
        raise ConfigError(f"{what} must be a positive number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class Sizer:
    """A sizer: its kind and arguments, as the kernel reads them."""

    kind: str
    amount: float | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def to_native(self) -> dict[str, Any]:
        """The quantity dict of an order request."""
        return {"kind": "sizer", "sizer": self.kind, "amount": self.amount, **self.options}


def qty(units: float) -> Sizer:
    """A fixed quantity in units, rounded down to the venue rule."""
    return Sizer("qty", _positive(units, "qty"))


def lots(count: int) -> Sizer:
    """A fixed number of lots."""
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ConfigError(f"lots must be a positive whole number, got {count!r}")
    # The kernel reads a lot count as a whole number, so it crosses as an int.
    return Sizer("lots", count)


def value(rupees: float) -> Sizer:
    """Rupees of notional: rupees / (price x multiplier) at the decision close, with no
    cash buffer."""
    return Sizer("value", _positive(rupees, "value"))


def percent(p: float, basis: str = "equity") -> Sizer:
    """p percent of equity (default), initial capital ("capital") or free cash ("cash")
    at the decision close, with no cash buffer: `percent(100)` can be rejected for margin
    when the next open is higher (see the module docstring)."""
    if basis not in BASES:
        raise ConfigError(f"unknown sizing basis {basis!r}: use one of {BASES}")
    return Sizer("percent", _positive(p, "percent"), {"basis": basis})


def _whole(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{what} must be a whole number, 1 or more, got {value!r}")
    return value


def fixed_risk(risk_pct: float, units: int = 1, hard_limit: float | None = None) -> Sizer:
    """Risk `risk_pct` percent of equity, less the estimated costs of the round trip, to
    the order's stop (needs `stop=`): quantity = budget / (stop distance x multiplier),
    capped at `hard_limit` units and split into `units` equal tranches (the order is
    one of them)."""
    limit = None if hard_limit is None else _positive(hard_limit, "hard_limit")
    options = {"units": _whole(units, "units"), "hard_limit": limit}
    return Sizer("fixed_risk", _positive(risk_pct, "risk_pct"), options)


def atr_risk(risk_pct: float, mult: float, period: int = 14) -> Sizer:
    """Fixed risk with the stop distance taken as `mult` x the ATR of `period` strategy
    bars (`openalgo.ta.atr`) at the decision."""
    options = {"multiple": _positive(mult, "mult"), "period": _whole(period, "period")}
    return Sizer("atr_risk", _positive(risk_pct, "risk_pct"), options)


@dataclass(frozen=True)
class SizeCaps:
    """Caps on what every sizer gives (`caps`): each as the kernel reads it, None for no
    cap. `position` and `risk` are (kind, amount) pairs: kind "units", "lots",
    "rupees" or "percent" (of the sizing equity) for the position, "rupees" or
    "percent" for the risk; `leverage` is a multiple of the sizing equity."""

    position: tuple[str, float] | None
    leverage: float | None
    risk: tuple[str, float] | None

    def to_native(self) -> dict[str, Any]:
        """The `size_caps` entry of the kernel's risk dict."""
        return {"position": self.position, "leverage": self.leverage, "risk": self.risk}


def caps(position: Any = None, leverage: float | None = None, risk: Any = None) -> SizeCaps:
    """Caps on the quantity every sizer gives, for the class attribute `size_caps`; the
    tightest wins:

    * `position`: the largest position on the order's side after it, counting what is
      held: units (a number or `bt.size.qty(n)`), `bt.size.lots(n)`, rupees of value at
      the decision price (`bt.rupees(x)` or `bt.size.value(x)`) or a percent of equity
      (`bt.pct(x)` or `bt.size.percent(x)`);
    * `leverage`: the largest value of that position as a multiple of equity (1 is 1x);
    * `risk`: the largest loss of the order at its stop: rupees (a number or
      `bt.rupees(x)`) or `bt.pct(x)` of equity, below 100. An order with no stop has no
      bounded loss, so under a risk cap it is skipped.

    A cap cuts the sized quantity, which is then rounded down to the venue rule; when
    nothing is left the order is skipped with reason "cap". Caps bound sizers only: an
    explicit `qty=` or `lots=` and `target()` trade what they ask for. Raises
    ConfigError for a value of the wrong kind or not positive, or when no cap is
    given."""
    # `_scale_in` reads `Sizer` from this module, so its parsers are imported here.
    from ._scale_in import position_limit, risk_limit

    if position is None and leverage is None and risk is None:
        raise ConfigError("bt.size.caps needs at least one of position=, leverage= or risk=")
    return SizeCaps(
        position=position_limit(position, "caps position"),
        leverage=None if leverage is None else _positive(leverage, "caps leverage"),
        risk=risk_limit(risk, "caps risk"),
    )


def resolve_atr(quantity: dict[str, Any], store: SeriesStore) -> dict[str, Any]:
    """The quantity dict of a request with the ATR of an `atr_risk` sizer filled in from
    the strategy bars revealed so far (`_atr.sizer_atr`, rupees). Any other quantity is
    returned as it is. With fewer bars than the period the ATR is NaN and the kernel
    refuses the sizer with a message."""
    if quantity.get("sizer") != "atr_risk" or quantity.get("atr") is not None:
        return quantity
    return {**quantity, "atr": sizer_atr(store, int(quantity["period"]))}


__all__ = [
    "BASES",
    "SizeCaps",
    "Sizer",
    "atr_risk",
    "caps",
    "fixed_risk",
    "lots",
    "percent",
    "qty",
    "resolve_atr",
    "value",
]
