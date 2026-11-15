"""Order and trade IDs in the formats the sandbox and a broker issue them.

The sandbox's order ID is the IST date (YYMMDD) and the millisecond of the IST day, raised
to one past the last ID issued (sandbox/order_manager.py, `_generate_order_id`); its trade
ID is "TRADE-<YYYYMMDD-HHMMSS>-<8 characters>" (sandbox/execution_engine.py,
`_generate_trade_id`). A broker's IDs are plain digits; the fake numbers its live orders
and trades in sequence after the IST date. Nothing is random, so a test can name the IDs
it expects.
"""

from __future__ import annotations

from openalgo.backtest._layout import NANOS_PER_SECOND

from .clock import ist_moment

#: Nanoseconds in a millisecond, the step of the sandbox's order ID sequence.
_NANOS_PER_MILLI = 1_000_000


class SandboxIds:
    """The sandbox's order and trade IDs."""

    def __init__(self) -> None:
        self._last_day = ""
        self._last_seq = -1
        self._trades = 0

    def order_id(self, now_ns: int) -> str:
        """A new order ID at `now_ns` (UTC nanoseconds)."""
        moment = ist_moment(now_ns)
        day = moment.strftime("%y%m%d")
        if day != self._last_day:
            self._last_day, self._last_seq = day, -1
        midnight = moment.replace(hour=0, minute=0, second=0)
        ms_of_day = (now_ns - int(midnight.timestamp()) * NANOS_PER_SECOND) // _NANOS_PER_MILLI
        self._last_seq = max(ms_of_day, self._last_seq + 1)
        return f"{day}{self._last_seq:08d}"

    def trade_id(self, now_ns: int) -> str:
        """A new trade ID at `now_ns`."""
        self._trades += 1
        return f"TRADE-{ist_moment(now_ns).strftime('%Y%m%d-%H%M%S')}-{self._trades:08x}"


class BrokerIds:
    """A broker's order and trade IDs: digits, numbered in sequence after the IST date."""

    def __init__(self) -> None:
        self._orders = 0
        self._trades = 0

    def order_id(self, now_ns: int) -> str:
        """A new order ID at `now_ns` (UTC nanoseconds): 15 digits."""
        self._orders += 1
        return f"{ist_moment(now_ns).strftime('%y%m%d')}{self._orders:09d}"

    def trade_id(self, now_ns: int) -> str:
        """A new trade ID at `now_ns`: 8 digits."""
        del now_ns  # a broker's trade IDs do not carry the time
        self._trades += 1
        return f"{self._trades:08d}"


__all__ = ["BrokerIds", "SandboxIds"]
