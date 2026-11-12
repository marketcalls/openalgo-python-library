"""The run context: what binds a strategy instance to the native engine of its run.

A `Strategy` reads data, positions and the account and places orders only through its
context, so the same strategy class runs against any driver (backtest now, live in
Phase 2). Orders can be placed only while the engine is paused at a decision point; the
kernel refuses them anywhere else.

A run has two phases. It is declaring while the strategy's `init` runs, from the
context's creation until the driver starts the run (the first pause of the event driver,
or the start of the condition driver, `begin_run`). The context marks the phase on every
store it makes, so a view can refuse a truth test while conditions are declared
(`_views`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from . import _sizing_function as sizing_function
from ._engine import call
from ._futures_context import SeriesContext
from ._orders import Order
from ._series import Bar, BarStore, PositionView, SeriesStore, ist_time
from ._timers import Timers
from .errors import ConfigError
from .reference import Instrument
from .size import resolve_atr


@dataclass(frozen=True)
class RunData:
    """The run's facts a strategy's data views need: the exchange, the strategy and base
    intervals, the base bars by symbol (canonical layout, cut to the run's slices), the
    calendar, and the indicator checks of the run (`_causality_checks.IndicatorChecks`)."""

    exchange: str
    interval: str
    base: str
    base_bars: dict[str, pd.DataFrame]
    calendar: Any
    checks: Any


class RunContext:
    """One run's engine, instruments, bars, timers and decision instant."""

    def __init__(
        self,
        engine: Any,
        instruments: list[Instrument],
        product: str,
        default_size: Any,
        run: RunData,
        futures: tuple[Any, list[dict]] | None = None,
    ) -> None:
        self.engine = engine
        self.instruments = {instrument.symbol: instrument for instrument in instruments}
        self.symbols = [instrument.symbol for instrument in instruments]
        self.index = {symbol: number for number, symbol in enumerate(self.symbols)}
        self.run = run
        # True while `init` declares the strategy; see the module docstring.
        self.declaring = True
        self.stores = {
            symbol: self._marked(BarStore(symbol, run.interval, call(engine.strategy_bars, number)))
            for number, symbol in enumerate(self.symbols)
        }
        self.timeframes: dict[tuple[str, str], SeriesStore] = {}
        self.timers = Timers(engine)
        self.product = product
        self.default_size = default_size
        self.decided_at: int | None = None
        # (decision instant, symbol, reason) of every order a sizer skipped.
        self.skipped: list[tuple[int | None, str, str]] = []
        # The result's orders table once the run is over (`close`), and the row of each
        # order id in it, made on the first lookup.
        self._final_orders: dict[str, list] | None = None
        self._final_rows: dict[int, int] | None = None
        # Futures series by underlying, from (`_futures.FuturesPlan`, the kernel's switches).
        self.futures = None
        if futures is not None:
            plan, switches = futures
            self.futures = SeriesContext(plan, switches, self.stores, self.symbols, run.interval)

    @property
    def names(self) -> tuple[str, ...]:
        """What the strategy addresses: its symbols, or its futures underlyings."""
        return self.futures.names if self.futures is not None else tuple(self.symbols)

    # ----- the phase of the run ----------------------------------------------------

    def _marked(self, store: SeriesStore) -> SeriesStore:
        """`store`, marked with the run's phase for its views."""
        store.declaring = self.declaring
        return store

    def begin_run(self) -> None:
        """End the declaration phase: `init` has returned and the driver runs. From here
        a view answers a truth test (in timers, callbacks, `on_bar` and `on_stop`)."""
        self.declaring = False
        for store in (*self.stores.values(), *self.timeframes.values()):
            self._marked(store)

    # ----- the clock and the bars ------------------------------------------------

    def reveal(self, decided_at: int | None, rows: list[tuple]) -> dict[str, Bar]:
        """Take the decision instant of a pause (None at the end of the run keeps the
        last one) and reveal its strategy bars, the kernel's rows; returns them by symbol
        (none at a timer pause). The first pause begins the run (`begin_run`)."""
        if self.declaring:
            self.begin_run()
        if decided_at is not None:
            self.decided_at = decided_at
        revealed = {}
        for row in rows:
            symbol = self.symbols[row[0]]
            revealed[symbol] = self.stores[symbol].reveal(row)
        if self.futures is not None:
            return self.futures.reveal(revealed, self.stores, self.decided_at)
        return revealed

    @property
    def now(self):
        """The decision instant in IST, or None before the first decision."""
        return None if self.decided_at is None else ist_time(self.decided_at)

    def store(self, symbol: str | None, interval: str | None = None) -> SeriesStore:
        """The bars of a symbol at the strategy interval, or at `interval` (built on first
        use, visible once closed). Raises ConfigError for an unknown symbol or an
        interval the base bars cannot give."""
        if self.futures is not None and (symbol is None or symbol in self.futures.names):
            return self.futures.store(symbol, interval)
        name = self.symbol_of(self.instrument_index(symbol))
        if interval is None or interval == self.run.interval:
            return self.stores[name]
        key = (name, interval)
        if key not in self.timeframes:
            from ._timeframes import timeframe_store

            run = self.run
            store = timeframe_store(
                name,
                run.exchange,
                interval,
                run.base,
                run.base_bars[name],
                run.calendar,
                lambda: self.decided_at,
            )
            self.timeframes[key] = self._marked(store)
        return self.timeframes[key]

    def symbol_of(self, index: int) -> str:
        return self.symbols[index]

    def instrument_index(self, symbol: str | None) -> int:
        """The registration index of a symbol; None means the only instrument.

        Raises ConfigError for an unknown symbol, or for None with several instruments.
        A futures underlying (or None for the only one) gives the contract it trades now.
        """
        if self.futures is not None and (symbol is None or symbol in self.futures.names):
            return self.futures.contract_index(symbol, self.decided_at)
        if symbol is None:
            if len(self.symbols) != 1:
                raise ConfigError("this strategy trades several symbols: pass symbol=")
            return 0
        if symbol not in self.index:
            raise ConfigError(f"{symbol!r} is not traded by this strategy: {self.symbols}")
        return self.index[symbol]

    # ----- orders ----------------------------------------------------------------

    def submit(self, request: dict) -> Order | None:
        """Place a request; returns its handle, or None when a sizer skipped it. A sizing
        function is called here, at the decision (`_sizing_function`)."""
        store = self.sizing_store(request["instrument"])
        symbol = self.symbols[request["instrument"]]
        request = {**request, "quantity": resolve_atr(request["quantity"], store)}
        request = sizing_function.resolve(request, self.engine, symbol)
        outcome, number, code = call(self.engine.submit, request)
        if outcome == "skipped":
            self.skipped.append((self.decided_at, symbol, code))
            return None
        return Order(number, self.order, self.cancel)

    def submit_target(self, request: dict) -> Order | None:
        """Trade to a target position (`_orders.target` builds the request); returns the
        handle of the order for the difference, or None when the position already is the
        target."""
        submitted = call(self.engine.target, request)
        if submitted is None:
            return None
        _, number, _ = submitted
        return Order(number, self.order, self.cancel)

    def sizing_store(self, instrument: int) -> SeriesStore:
        """The bars an order's ATR distances and sizers read: the instrument's strategy
        bars, or the spliced series of the futures underlying it belongs to."""
        symbol = self.symbols[instrument]
        spliced = None if self.futures is None else self.futures.store_of(symbol)
        return self.stores[symbol] if spliced is None else spliced

    def order(self, number: int) -> dict:
        """An order's state, from the engine or, after the run, from the result."""
        if self._final_orders is not None:
            return self._final_order(number)
        return call(self.engine.order, number)

    def cancel(self, number: int) -> str:
        """Cancel a working order; returns its state afterwards."""
        if self._final_orders is not None:
            return self._final_order(number)["status"]
        return call(self.engine.cancel, number)

    def working_orders(self) -> list[Order]:
        """Handles of the orders that can still fill."""
        if self._final_orders is not None:
            return []
        numbers = call(self.engine.working_orders)
        return [Order(number, self.order, self.cancel) for number in numbers]

    def close(self, orders: dict[str, list]) -> None:
        """Freeze order lookups on the result's orders table (columns by key) once the
        engine is gone. A row is read only when a handle asks for it after the run, so a
        run with many orders does not build a dict for each."""
        self._final_orders = orders

    def _final_order(self, number: int) -> dict:
        """One order of the result's orders table as a dict. Raises KeyError for an id
        the run did not give."""
        orders = self._final_orders
        if self._final_rows is None:
            self._final_rows = {order_id: row for row, order_id in enumerate(orders["order_id"])}
        row = self._final_rows[number]
        return {key: column[row] for key, column in orders.items()}

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


__all__ = ["RunContext", "RunData"]
