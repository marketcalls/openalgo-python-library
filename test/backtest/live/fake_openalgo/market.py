"""The instruments the fake broker knows and the price path the test pushes for each.

A trade print (`MarketState.trade`) sets the last price, the day's open, high, low and
cumulative volume, and the 1-minute candle it falls in, which `history` answers. Candles
start on the minute (UTC and IST minutes coincide: IST is UTC plus 5 h 30 min), as the
broker's do, and the minute in progress is a forming candle that brokers answer too.

Prices are rupees, quantities units, times UTC nanoseconds.
"""

from __future__ import annotations

from dataclasses import dataclass

from openalgo.backtest._layout import IST_OFFSET_NANOS, NANOS_PER_MINUTE

from .clock import ist_moment

#: The segments whose orders must be whole lots (sandbox/order_manager.py, place_order).
DERIVATIVE_EXCHANGES = frozenset({"NFO", "BFO", "CDS", "BCD", "MCX", "NCDEX", "CRYPTO"})
#: The segments of exchange-traded options, which accept no SL-M order (D31).
OPTION_EXCHANGES = frozenset({"NFO", "BFO"})
#: The instrument types of an option contract in OpenAlgo's symbol table.
OPTION_TYPES = frozenset({"CE", "PE"})
_NANOS_PER_DAY = 24 * 60 * NANOS_PER_MINUTE


@dataclass(frozen=True)
class Instrument:
    """An instrument the broker trades: `lot_size` and `freeze_qty` in units (0: no
    freeze limit), `tick_size` in rupees, `instrument_type`, `expiry` ("DD-MMM-YY") and
    `strike` (rupees) as OpenAlgo's symbol table writes them ("EQ", "FUT", "CE", "PE";
    no expiry and strike 0 for cash equity)."""

    symbol: str
    exchange: str
    lot_size: int = 1
    tick_size: float = 0.05
    freeze_qty: int = 0
    instrument_type: str = "EQ"
    expiry: str = ""
    strike: float = 0.0

    def __post_init__(self) -> None:
        if not self.symbol or not self.exchange:
            raise ValueError("an instrument needs a symbol and an exchange")
        if self.lot_size < 1 or self.freeze_qty < 0 or self.tick_size <= 0:
            raise ValueError(
                f"{self.exchange}:{self.symbol}: lot_size is at least 1 unit, freeze_qty "
                "0 or more units and tick_size more than 0 rupees"
            )

    @property
    def key(self) -> tuple[str, str]:
        """(symbol, exchange), the key the fake files the instrument under."""
        return (self.symbol, self.exchange)

    @property
    def is_option(self) -> bool:
        """True for an NFO or BFO option, which refuses SL-M orders at a live broker."""
        return self.exchange in OPTION_EXCHANGES and self.instrument_type in OPTION_TYPES

    @property
    def trades_in_lots(self) -> bool:
        """True where an order's quantity must be a whole number of lots."""
        return self.exchange in DERIVATIVE_EXCHANGES


@dataclass
class Candle:
    """A 1-minute candle starting at `start_ns` (UTC nanoseconds, on the minute)."""

    start_ns: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def take(self, price: float, volume: float) -> None:
        """Fold one trade print into the candle."""
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += volume


class MarketState:
    """The price path of one instrument: its last trade, the day's statistics, the best
    bid and ask when the test gave them, and its 1-minute candles."""

    def __init__(self, instrument: Instrument) -> None:
        self.instrument = instrument
        self.last_price: float | None = None
        self.last_trade_ns: int | None = None
        self.bid: float | None = None
        self.ask: float | None = None
        self.day_open = self.day_high = self.day_low = 0.0
        self.day_volume = 0.0
        self.prev_close = 0.0
        self._day: int | None = None
        self._candles: dict[int, Candle] = {}

    def trade(
        self,
        ts_ns: int,
        price: float,
        volume: float,
        bid: float | None = None,
        ask: float | None = None,
    ) -> None:
        """Take in a trade print at `ts_ns`. Raises ValueError for a price that is not
        positive or a negative volume."""
        if price <= 0 or volume < 0:
            raise ValueError(
                f"a trade needs a price above 0 and a volume of 0 or more units, not "
                f"{price} rupees and {volume} units"
            )
        day = (ts_ns + IST_OFFSET_NANOS) // _NANOS_PER_DAY
        if day != self._day:
            if self.last_price is not None:
                self.prev_close = self.last_price
            self._day = day
            self.day_open = self.day_high = self.day_low = price
            self.day_volume = 0.0
        self.day_high = max(self.day_high, price)
        self.day_low = min(self.day_low, price)
        self.day_volume += volume
        self.last_price, self.last_trade_ns = price, ts_ns
        self.bid, self.ask = bid, ask
        start = ts_ns - ts_ns % NANOS_PER_MINUTE
        candle = self._candles.get(start)
        if candle is None:
            self._candles[start] = Candle(start, price, price, price, price, volume)
        else:
            candle.take(price, volume)

    def seed(self, candle: Candle) -> None:
        """Put in a broker candle of the past (warm-up history). Raises ValueError for a
        start that is not on the minute or a candle whose high and low do not hold its
        open and close."""
        if candle.start_ns % NANOS_PER_MINUTE:
            raise ValueError(f"a 1-minute candle starts on the minute, not at {candle.start_ns}")
        if (
            not candle.low
            <= min(candle.open, candle.close)
            <= max(candle.open, candle.close)
            <= candle.high
        ):
            raise ValueError(f"candle at {candle.start_ns}: high and low must hold open, close")
        self._candles[candle.start_ns] = candle

    def candles(self, first_day: str, last_day: str) -> list[Candle]:
        """The candles of the IST dates `first_day` to `last_day` ("YYYY-MM-DD", both
        included), oldest first, the forming one included."""
        return [
            candle
            for start, candle in sorted(self._candles.items())
            if first_day <= ist_moment(start).date().isoformat() <= last_day
        ]

    @property
    def best_bid(self) -> float:
        """The best bid: the one given with the last trade, else the last price (0
        before any trade)."""
        return self.bid if self.bid is not None else (self.last_price or 0.0)

    @property
    def best_ask(self) -> float:
        """The best ask: the one given with the last trade, else the last price (0
        before any trade)."""
        return self.ask if self.ask is not None else (self.last_price or 0.0)


__all__ = [
    "DERIVATIVE_EXCHANGES",
    "OPTION_EXCHANGES",
    "OPTION_TYPES",
    "Candle",
    "Instrument",
    "MarketState",
]
