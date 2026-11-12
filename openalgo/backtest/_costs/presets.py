"""Brokerage presets: the broker's part of a trade's cost.

Statutory and exchange charges are not part of a preset; they are the same for every
broker and come from the dated schedules. A preset is validated and held by the
kernel, and the Zerodha figures live there, so no rate is written here.

Units: amounts and caps are rupees per executed order. Anything called `percent`
takes percent units (0.03 means 0.03 percent of order value). The kernel stores a
percentage as a `rate`, a fraction (0.0003), and that is what `Brokerage.rules`
reports.

The DP fee (depository participant fee on a delivery sell, rupees per scrip per sell
day before GST): Zerodha publishes Rs 13 (Rs 3.5 CDSL plus Rs 9.5 Zerodha). No other
preset has a published figure, so until you pass `dp_fee` Zerodha's Rs 13 stands in
and `dp_fee_source` says "zerodha_stand_in"; every DP charge result repeats it.
"""

import math
from typing import Any

from ..errors import CostError
from .inputs import number
from .kernel import call, native

__all__ = ["Brokerage", "custom", "flat", "percent", "preset", "zero_brokerage", "zerodha"]

_CLASSES = ("delivery", "intraday", "futures", "options")
_RULE_FORMS = (
    "None or 'zero', a flat amount in rupees, ('flat', rupees) or ('percent', percent, cap) "
    "in percent units"
)


class Brokerage:
    """A validated brokerage preset: one rule per kind of trade, plus the DP fee.

    Build one with `zerodha()`, `flat()`, `percent()`, `zero_brokerage()` or
    `custom()`. Instances are immutable; `with_dp_fee` returns a new one.
    """

    __slots__ = ("_kernel", "_name")

    def __init__(self, name: str, kernel: Any) -> None:
        self._name = name
        self._kernel = kernel

    @property
    def name(self) -> str:
        """A label for reports, such as "zerodha" or "flat(10)"."""
        return self._name

    @property
    def rules(self) -> dict[str, tuple]:
        """The rule of each kind of trade: delivery, intraday, futures, options.

        Each is `("zero",)`, `("flat", rupees)` or `("rate", rate, cap)`, where `rate`
        is a fraction of order value (0.0003 is 0.03 percent) and `cap` is in rupees
        per executed order (`inf` for no cap).
        """
        shown = {"zero": 1, "flat": 2, "rate": 3}
        rules = self._kernel.rules()
        return {
            name: tuple(rule[: shown[rule[0]]]) for name, rule in zip(_CLASSES, rules, strict=True)
        }

    @property
    def dp_fee(self) -> float:
        """The DP fee in rupees per scrip per sell day, before GST."""
        return self._kernel.dp_fee

    @property
    def dp_fee_source(self) -> str:
        """Whose DP fee it is: "zerodha_preset", "zerodha_stand_in" or "supplied"."""
        return self._kernel.dp_fee_source

    def with_dp_fee(self, dp_fee: float) -> "Brokerage":
        """The same preset with your broker's DP fee (rupees per scrip per sell day, before GST).

        Raises CostError for a fee that is negative, infinite or not a number.
        """
        return Brokerage(self._name, call(self._kernel.with_dp_fee, number("dp_fee", dp_fee)))

    def __repr__(self) -> str:
        return f"Brokerage({self._name!r}, dp_fee={self.dp_fee:g} ({self.dp_fee_source}))"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Brokerage) and self._kernel == other._kernel

    def __hash__(self) -> int:
        return hash((tuple(self._kernel.rules()), self.dp_fee, self.dp_fee_source))


def _built(name: str, constructor: Any, *args: Any, dp_fee: float | None = None) -> Brokerage:
    brokerage = Brokerage(name, call(constructor, *args))
    return brokerage if dp_fee is None else brokerage.with_dp_fee(dp_fee)


def _cap(name: str, cap: Any) -> float:
    return math.inf if cap is None else number(f"{name} cap", cap)


def _rule_from_sequence(name: str, spec: tuple | list) -> tuple | None:
    kind, arguments = spec[0], list(spec[1:])
    if kind == "zero" and not arguments:
        return ("zero", 0.0, 0.0)
    if kind == "flat" and len(arguments) == 1:
        return ("flat", number(f"{name} flat amount", arguments[0]), 0.0)
    if kind == "percent" and len(arguments) in (1, 2):
        cap = _cap(name, arguments[1] if len(arguments) == 2 else None)
        return ("percent", number(f"{name} percent", arguments[0]), cap)
    return None


def _rule(name: str, spec: Any) -> tuple:
    """A user rule as the kernel's (kind, first, second) tuple."""
    rule = None
    if spec is None or (isinstance(spec, str) and spec == "zero"):
        rule = ("zero", 0.0, 0.0)
    elif isinstance(spec, (tuple, list)) and spec:
        rule = _rule_from_sequence(name, spec)
    elif not isinstance(spec, (str, bool)):
        rule = ("flat", number(f"{name} flat amount", spec), 0.0)
    if rule is None:
        raise CostError(f"{name}: cannot read brokerage rule {spec!r}; use {_RULE_FORMS}")
    return rule


def zerodha() -> Brokerage:
    """Zerodha, the default preset.

    Delivery is free; intraday equity and futures pay 0.03 percent or Rs 20 per
    executed order, whichever is lower; options pay Rs 20 per executed order. The DP
    fee is Zerodha's published Rs 13 (Rs 15.34 with GST) per scrip per sell day.
    """
    return _built("zerodha", native.CostBrokerage.named, "zerodha")


def zero_brokerage(dp_fee: float | None = None) -> Brokerage:
    """No brokerage in any segment. Statutory charges and the DP fee still apply.

    `dp_fee` is your broker's fee in rupees per scrip per sell day before GST. When
    it is None, Zerodha's Rs 13 stands in and results say so (see the module
    docstring). Raises CostError for an invalid fee.
    """
    return _built("zero_brokerage", native.CostBrokerage.named, "zero_brokerage", dp_fee=dp_fee)


def flat(amount: float, dp_fee: float | None = None) -> Brokerage:
    """The same flat `amount` in rupees per executed order in every segment.

    `dp_fee` as for `zero_brokerage`: None means Zerodha's Rs 13 stands in. Raises
    CostError for an amount or fee that is negative, infinite or not a number.
    """
    amount = number("amount", amount)
    return _built(f"flat({amount:g})", native.CostBrokerage.flat, amount, dp_fee=dp_fee)


def percent(percent: float, cap: float | None = None, dp_fee: float | None = None) -> Brokerage:
    """A percentage of order value in every segment, at most `cap` rupees per order.

    `percent` is in percent units: `percent(0.03, 20)` is 0.03 percent of the order
    value or Rs 20, whichever is lower (Zerodha's intraday rule). `cap=None` means no
    cap. `dp_fee` as for `zero_brokerage`.

    Raises CostError for a percentage outside 0.001 to 5 (zero is allowed), which
    catches a fraction such as 0.0003 passed by mistake, and for a negative cap.
    """
    percent = number("percent", percent)
    label = f"percent({percent:g}, {'None' if cap is None else format(number('cap', cap), 'g')})"
    return _built(label, native.CostBrokerage.percent, percent, cap, dp_fee=dp_fee)


def custom(
    delivery: Any = None,
    intraday: Any = None,
    futures: Any = None,
    options: Any = None,
    dp_fee: float | None = None,
    name: str = "custom",
) -> Brokerage:
    """User-defined brokerage: one rule per kind of trade.

    Each rule is None or "zero" (no brokerage), a number (flat rupees per executed
    order), `("flat", rupees)` or `("percent", percent, cap)` in percent units with
    `cap` optional.

        costs.custom(delivery=("percent", 0.1, 30), intraday=("percent", 0.03, 20),
                     futures=20, options=20, dp_fee=15)

    `dp_fee` as for `zero_brokerage`: None means Zerodha's Rs 13 stands in. Raises
    CostError for a rule that cannot be read or has an implausible amount.
    """
    specs = zip(_CLASSES, (delivery, intraday, futures, options), strict=True)
    rules = [_rule(kind, spec) for kind, spec in specs]
    return _built(str(name), native.CostBrokerage.custom, *rules, dp_fee=dp_fee)


def preset(brokerage: "str | Brokerage") -> Brokerage:
    """Resolve a brokerage argument: a Brokerage, or the name of a built-in preset.

    Built-in names are "zerodha" and "zero_brokerage". Raises CostError for any other
    name or type: there is no default to fall back on.
    """
    if isinstance(brokerage, Brokerage):
        return brokerage
    if isinstance(brokerage, str):
        return _built(brokerage, native.CostBrokerage.named, brokerage)
    raise CostError(
        "brokerage must be a preset name or a Brokerage from costs.zerodha(), costs.flat(), "
        f"costs.percent(), costs.zero_brokerage() or costs.custom(), got {brokerage!r}"
    )
