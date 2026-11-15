"""An in-process fake OpenAlgo for the live layer's tests (plan step 8, the fault suite).

The real SDK runs against it unchanged: an `openalgo.api` from `FakeOpenAlgo.client()`
sends its REST calls through its own HTTP client to an in-process transport, and its
`FeedAPI` connects to an in-process WebSocket proxy. Behind them sit a broker with a
manual clock, the price path the test pushes, the live book and the sandbox's, the
analyzer and semi-auto switches, and every fault of the fault suite (`Faults`). Nothing
opens a socket: the `offline` fixture fails a test that tries.

Modules, one job each: `clock` (the manual clock), `market` (instruments and prices),
`orders`, `book` and `positions` (the broker), `ids` (order and trade IDs), `venue` (the
fake's state), `schema` and the `routes_*` (the server's checks and answers), `rows`,
`messages` and `answers` (the server's shapes), `rest` (the HTTP transport), `feed_server`
and `feed` (the WebSocket), `faults`, `log`, `fake` (the facade) and `fixtures`.

Times are UTC nanoseconds, prices rupees, quantities units (never lots).
"""

from .clock import ManualClock, ist_ns
from .fake import API_KEY, BROKER, HOST, LIVE_CASH, SANDBOX_CASH, USERNAME, FakeOpenAlgo
from .faults import Faults
from .log import Entry, FakeLog
from .orders import FakeOrder, Fill

__all__ = [
    "API_KEY",
    "BROKER",
    "HOST",
    "LIVE_CASH",
    "SANDBOX_CASH",
    "USERNAME",
    "Entry",
    "FakeLog",
    "FakeOpenAlgo",
    "FakeOrder",
    "Faults",
    "Fill",
    "ManualClock",
    "ist_ns",
]
