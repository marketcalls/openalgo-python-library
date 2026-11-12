"""What a futures strategy's run context adds: the series it addresses by underlying,
the contract each one trades at a decision point, and their spliced bars (`_spliced`).

The kernel's switches are the one source of the contract in force: from a switch's
slice on, `self.buy()` and `self.position` refer to the next contract (entries trade the
next contract from the roll session on, 04-execution 7). Fills, positions and the trade
list carry the actual contracts; `on_bar` and the series views see the spliced series.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._messages import not_supported
from ._series import Bar, BarStore
from ._spliced import SplicedStore
from .errors import ConfigError


class SeriesContext:
    """The futures series of one run.

    plan: the run's `_futures.FuturesPlan`; switches: the kernel's switches (dicts);
    stores: the contracts' strategy-bar stores by symbol; symbols: every registered
    symbol, in registration order; interval: the strategy interval."""

    def __init__(
        self,
        plan: Any,
        switches: list[dict],
        stores: dict[str, BarStore],
        symbols: list[str],
        interval: str,
    ) -> None:
        self.names = tuple(series.name for series in plan.series)
        self._interval = interval
        index_of = {symbol: number for number, symbol in enumerate(symbols)}
        self._contracts: dict[str, list[int]] = {}
        self._switch_at: dict[str, np.ndarray] = {}
        self._stores: dict[str, SplicedStore] = {}
        # (series name, position in its series) of every contract, by symbol.
        self._place: dict[str, tuple[str, int]] = {}
        for number, series in enumerate(plan.series):
            own = [switch for switch in switches if switch["series"] == number]
            symbols_of = [instrument.symbol for instrument in series.contracts]
            self._contracts[series.name] = [index_of[symbol] for symbol in symbols_of]
            self._switch_at[series.name] = np.asarray(
                [int(switch["slice_ts"]) for switch in own], dtype=np.int64
            )
            sources = [stores[symbol] for symbol in symbols_of]
            self._stores[series.name] = SplicedStore(series.name, interval, sources, own)
            for position, symbol in enumerate(symbols_of):
                self._place[symbol] = (series.name, position)

    def _name(self, name: str | None) -> str:
        if name is not None:
            return name
        if len(self.names) != 1:
            raise ConfigError(
                f"this strategy trades several underlyings {self.names}: pass symbol="
            )
        return self.names[0]

    def contract_index(self, name: str | None, decided_at: int | None) -> int:
        """The registration index of the contract a series trades at `decided_at` (its
        first contract before the first decision)."""
        name = self._name(name)
        switched = 0
        if decided_at is not None:
            switched = int(np.searchsorted(self._switch_at[name], decided_at, "right"))
        return self._contracts[name][switched]

    def store(self, name: str | None, interval: str | None) -> SplicedStore:
        """The spliced strategy bars of a series. Raises NotImplementedError for another
        interval."""
        if interval is not None and interval != self._interval:
            raise not_supported(
                "a futures series at another interval than the strategy's",
                "read the series at the strategy interval",
            )
        return self._stores[self._name(name)]

    def store_of(self, symbol: str) -> SplicedStore | None:
        """The spliced store of the series a contract belongs to, if any."""
        place = self._place.get(symbol)
        return None if place is None else self._stores[place[0]]

    def reveal(
        self, revealed: dict[str, Bar], stores: dict[str, BarStore], decided_at: int | None
    ) -> dict[str, Bar]:
        """Move every series to the epoch of `decided_at`, then reveal the spliced bars
        among the contracts' bars just revealed; returns them by series name."""
        for store in self._stores.values():
            store.advance(decided_at)
        bars: dict[str, Bar] = {}
        for symbol, bar in revealed.items():
            place = self._place.get(symbol)
            if place is None:
                continue
            name, position = place
            row = stores[symbol].visible() - 1
            spliced = self._stores[name].reveal_from(position, row, bar)
            if spliced is not None:
                bars[name] = spliced
        return bars


__all__ = ["SeriesContext"]
