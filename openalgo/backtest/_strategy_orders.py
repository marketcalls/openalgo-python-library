"""The order methods of `Strategy` (14-strategy-api 2.4), as a mixin.

`buy` and `sell` open, add or reduce; `exit` and `reduce` only reduce. Every method takes
`symbol=None` (required with several instruments) and returns an `Order` handle, or None
when there is nothing to do (an exit while flat) or a sizer skipped the order. A MARKET
order fills at the next base-bar open at or after the decision; a LIMIT or stop order
(`price=`, `trigger_price=`) rests from that bar until it fills or its session closes;
a GTT order (`validity="gtt"`) rests until it fires or is cancelled. The kernel refuses
what it cannot run yet with NotImplementedError.
"""

from __future__ import annotations

from typing import Any

from . import _orders
from ._atr import decision_atr
from ._orders import Order
from .errors import ConfigError

NOT_BUILT = "is not implemented yet (Phase 1 work in progress)"


class OrderMethods:
    """`buy`, `sell`, `exit`, `reduce`, `target`, `cancel`, `cancel_all`."""

    _ctx: Any

    def buy(
        self,
        qty: float | None = None,
        *,
        size: Any = None,
        lots: int | None = None,
        price: float | None = None,
        trigger_price: float | None = None,
        stop: Any = None,
        target: Any = None,
        targets: Any = None,
        trail: Any = None,
        breakeven: Any = None,
        product: str | None = None,
        leverage: float | None = None,
        symbol: str | None = None,
        tag: str | None = None,
        reverse: bool = False,
        after_first_target: str | None = None,
        validity: str = "day",
    ) -> Order | None:
        """Buy: open a long, add to it, or reduce a short (the excess over the short is
        rejected unless `reverse=True`). `price=None` is MARKET, a price makes it LIMIT;
        `trigger_price` makes it a stop order, SL with a price and SL-M without (on
        options an SL-M is sent as an SL with its limit 5 percent beyond the trigger).
        A LIMIT or stop order rests from the next base bar until it fills or its session
        closes. Quantity: `qty` units or `lots`, else `size=`, else the strategy's
        `size`, else the engine default. `leverage=` asks for intraday leverage, else the
        strategy's `leverage`, else 1x. Risk arguments attach to the position it opens
        (an add may repeat them but not change them, SPEC-14); on CNC and NRML they are
        held as GTT orders. `validity="gtt"` rests it as a GTT until it fires or is
        cancelled (CNC and NRML only; it needs `price`, its trigger)."""
        return self._open("BUY", **locals_without_self(locals()))

    def sell(
        self,
        qty: float | None = None,
        *,
        size: Any = None,
        lots: int | None = None,
        price: float | None = None,
        trigger_price: float | None = None,
        stop: Any = None,
        target: Any = None,
        targets: Any = None,
        trail: Any = None,
        breakeven: Any = None,
        product: str | None = None,
        leverage: float | None = None,
        symbol: str | None = None,
        tag: str | None = None,
        reverse: bool = False,
        after_first_target: str | None = None,
        validity: str = "day",
    ) -> Order | None:
        """Sell: open a short, add to it, or reduce a long. Arguments as `buy`."""
        return self._open("SELL", **locals_without_self(locals()))

    def _open(self, side: str, **args: Any) -> Order | None:
        ctx = self._ctx
        instrument = ctx.instrument_index(args["symbol"])
        store = ctx.stores[ctx.symbol_of(instrument)]
        rules = _orders.leg_rules(
            args["stop"],
            args["target"],
            args["targets"],
            args["trail"],
            args["breakeven"],
            args["after_first_target"],
            atr_at=lambda period: decision_atr(store, period),
        )
        request = _orders.request(
            instrument,
            side,
            args["product"] or ctx.product,
            quantity=_orders.quantity(args["qty"], args["lots"], args["size"], ctx.default_size),
            validity=_orders.validity(args["validity"]),
            reverse=bool(args["reverse"]),
            rules=rules,
            leverage=order_leverage(args["leverage"], getattr(self, "leverage", None)),
            tag=args["tag"],
            **_orders.order_shape(args["price"], args["trigger_price"]),
        )
        return ctx.submit(request)

    def exit(
        self,
        symbol: str | None = None,
        *,
        fraction: float = 1.0,
        qty: float | None = None,
        reason: str | None = None,
    ) -> Order | None:
        """Close the position, or `fraction` of it, or `qty` units, at the next open.
        Returns None when flat. The fill's reason is "signal" unless `reason=` names
        another of 05-risk-management 7."""
        return self._reduce(symbol, fraction=fraction, qty=qty, reason=reason)

    def reduce(
        self,
        qty: float | None = None,
        *,
        lots: int | None = None,
        fraction: float | None = None,
        symbol: str | None = None,
    ) -> Order | None:
        """Reduce the position by `qty` units, `lots`, or a `fraction` of it."""
        given = [value for value in (qty, lots, fraction) if value is not None]
        if len(given) != 1:
            raise ConfigError("reduce takes exactly one of qty=, lots= or fraction=")
        if lots is not None:
            qty = lots * self._ctx.instruments[self._symbol_name(symbol)].lot_size
        return self._reduce(symbol, fraction=fraction, qty=qty, reason=None)

    def _symbol_name(self, symbol: str | None) -> str:
        return self._ctx.symbol_of(self._ctx.instrument_index(symbol))

    def _reduce(
        self, symbol: str | None, *, fraction: float | None, qty: float | None, reason: str | None
    ) -> Order | None:
        ctx = self._ctx
        name = self._symbol_name(symbol)
        held = ctx.position(name).qty
        if held == 0:
            return None
        quantity = (
            {"kind": "units", "value": float(qty)}
            if qty is not None
            else {"kind": "fraction", "value": float(fraction if fraction is not None else 1.0)}
        )
        request = _orders.request(
            ctx.instrument_index(name),
            "SELL" if held > 0 else "BUY",
            ctx.product,
            quantity=quantity,
            reduce_only=True,
            exit_reason=reason,
        )
        return ctx.submit(request)

    def target(
        self,
        *,
        qty: float | None = None,
        percent: float | None = None,
        value: float | None = None,
        symbol: str | None = None,
    ) -> Order | None:
        """Trade to a target position: `qty` units, `percent` of equity or `value` rupees."""
        raise NotImplementedError(f"target() resizing {NOT_BUILT}")

    def cancel(self, order: Order) -> str:
        """Cancel a working order; returns its state afterwards."""
        return order.cancel()

    def cancel_all(self, symbol: str | None = None) -> None:
        """Cancel every working order, or those of one symbol."""
        wanted = None if symbol is None else self._ctx.instrument_index(symbol)
        for order in self._ctx.working_orders():
            if wanted is None or self._ctx.order(order.number)["instrument"] == wanted:
                order.cancel()


def locals_without_self(values: dict[str, Any]) -> dict[str, Any]:
    """A method's arguments without `self` (and `__class__`)."""
    return {key: value for key, value in values.items() if key not in ("self", "__class__")}


def order_leverage(given: float | None, strategy_leverage: float | None) -> float | None:
    """The intraday leverage an order asks for: its own `leverage=`, else the strategy's
    class attribute `leverage` (14-strategy-api 2.4); None is 1x."""
    return given if given is not None else strategy_leverage


__all__ = ["OrderMethods", "order_leverage"]
