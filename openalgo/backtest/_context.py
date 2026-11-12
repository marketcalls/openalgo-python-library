"""The run context: what binds a strategy instance to the native engine of its run.

A `Strategy` reads data, positions and the account and places orders only through its
context, so the same strategy class runs against any driver (backtest now, live in
Phase 2). Orders can be placed only while the engine is paused at a decision point; the
kernel refuses them anywhere else.
"""

from __future__ import annotations

from typing import Any

from ._engine import call
from ._orders import Order
from ._series import Bar, BarStore, PositionView, ist_time
from .errors import ConfigError
from .reference import Instrument
from .size import Sizer, resolve_atr


class RunContext:
    """One run's engine, instruments, revealed bars and decision instant."""

    def __init__(
        self,
        engine: Any,
        instruments: list[Instrument],
        product: str,
        default_size: Sizer | None,
    ) -> None:
        self.engine = engine
        self.instruments = {instrument.symbol: instrument for instrument in instruments}
        self.symbols = [instrument.symbol for instrument in instruments]
        self.index = {symbol: number for number, symbol in enumerate(self.symbols)}
        self.stores = {symbol: BarStore(symbol) for symbol in self.symbols}
        self.product = product
        self.default_size = default_size
        self.decided_at: int | None = None
        # (decision instant, symbol, reason) of every order a sizer skipped.
        self.skipped: list[tuple[int | None, str, str]] = []
        self._final_orders: dict[int, dict] | None = None

    # ----- the clock and the bars ------------------------------------------------

    def reveal(self, pause: dict) -> dict[str, Bar]:
        """Reveal the strategy bars of a pause; returns them by symbol."""
        self.decided_at = pause.get("decided_at")
        revealed = {}
        for row in pause.get("bars", ()):
            symbol = self.symbols[row[0]]
            revealed[symbol] = self.stores[symbol].append(row)
        return revealed

    @property
    def now(self):
        """The decision instant in IST, or None before the first decision."""
        return None if self.decided_at is None else ist_time(self.decided_at)

    def symbol_of(self, index: int) -> str:
        return self.symbols[index]

    def instrument_index(self, symbol: str | None) -> int:
        """The registration index of a symbol; None means the only instrument.

        Raises ConfigError for an unknown symbol, or for None with several instruments.
        """
        if symbol is None:
            if len(self.symbols) != 1:
                raise ConfigError("this strategy trades several symbols: pass symbol=")
            return 0
        if symbol not in self.index:
            raise ConfigError(f"{symbol!r} is not traded by this strategy: {self.symbols}")
        return self.index[symbol]

    # ----- orders ----------------------------------------------------------------

    def submit(self, request: dict) -> Order | None:
        """Place a request; returns its handle, or None when a sizer skipped it."""
        store = self.stores[self.symbols[request["instrument"]]]
        request = {**request, "quantity": resolve_atr(request["quantity"], store)}
        outcome, number, code = call(self.engine.submit, request)
        if outcome == "skipped":
            symbol = self.symbols[request["instrument"]]
            self.skipped.append((self.decided_at, symbol, code))
            return None
        return Order(number, self.order, self.cancel)

    def order(self, number: int) -> dict:
        """An order's state, from the engine or, after the run, from the result."""
        if self._final_orders is not None:
            return self._final_orders[number]
        return call(self.engine.order, number)

    def cancel(self, number: int) -> str:
        """Cancel a working order; returns its state afterwards."""
        if self._final_orders is not None:
            return self._final_orders[number]["status"]
        return call(self.engine.cancel, number)

    def working_orders(self) -> list[Order]:
        """Handles of the orders that can still fill."""
        if self._final_orders is not None:
            return []
        numbers = call(self.engine.working_orders)
        return [Order(number, self.order, self.cancel) for number in numbers]

    def close(self, orders: dict[str, list]) -> None:
        """Freeze order lookups on the result's orders table once the engine is gone."""
        keys = list(orders)
        rows = zip(*(orders[key] for key in keys), strict=True)
        self._final_orders = {row[0]: dict(zip(keys, row, strict=True)) for row in rows}

    # ----- positions and the account ----------------------------------------------

    def position(self, symbol: str, product: str | None = None) -> PositionView:
        """The position of a symbol in a product (the strategy's product by default)."""
        index = self.instrument_index(symbol)
        native = call(self.engine.position, index, product or self.product)
        return PositionView.from_native(symbol, native)

    def has_pending_entry(self, symbol: str) -> bool:
        """True while an order that would open or add works on the symbol."""
        return call(self.engine.has_pending_entry, self.instrument_index(symbol))

    def account(self) -> tuple[float, float, float, float, float, float]:
        """(equity, free cash, realized, unrealized, costs, margin) in rupees."""
        return call(self.engine.account)


__all__ = ["RunContext"]
