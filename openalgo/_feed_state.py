"""Connection state and constants of the feed (private to openalgo.feed)."""

import enum
from types import MappingProxyType
from typing import NamedTuple


class Mode(NamedTuple):
    """The names that differ between the three market-data modes."""

    label: str  # as printed in the verbose log lines
    key: str  # key in get_subscriptions()
    callback_attr: str  # FeedAPI attribute that holds the user callback
    cache_attr: str  # FeedAPI attribute that holds the latest values


MODES = MappingProxyType(
    {
        1: Mode("LTP", "ltp", "ltp_callback", "ltp_data"),
        2: Mode("Quote", "quote", "quote_callback", "quotes_data"),
        3: Mode("Depth", "depth", "depth_callback", "depth_data"),
    }
)


class Phase(enum.Enum):
    """Where the connection of the feed stands."""

    IDLE = "idle"  # no socket: never connected, between attempts, or dropped and not retried
    STOPPED = "stopped"  # no socket, and disconnect() asked for none: attempts are refused
    CONNECTING = "connecting"  # a socket is being opened
    OPEN = "open"  # the socket is open and the API key was sent
    LIVE = "live"  # the server accepted the API key


class Link(NamedTuple):
    """The connection state. Replaced as a whole, with FeedAPI._state_lock held."""

    phase: Phase = Phase.IDLE
    seq: int = 0  # number of the current connection attempt: it grows with every attempt
    restoring: bool = False  # an authenticated session was lost and is being restored

    @property
    def connected(self) -> bool:
        """The socket is open."""
        return self.phase in (Phase.OPEN, Phase.LIVE)

    @property
    def live(self) -> bool:
        """The socket is open and the server accepted the API key."""
        return self.phase is Phase.LIVE


class Timing(NamedTuple):
    """How long the feed waits, in seconds (see FeedAPI._timing)."""

    poll: float = 0.1  # between two looks at the state while connect() waits
    connect: float = 5  # for the socket to open
    authenticate: float = 5  # for the server to accept the API key
    disconnect: float = 2  # for the socket thread to finish in disconnect()
    backoff: tuple[float, ...] = (1, 2, 5, 10, 30, 60)  # before each reconnect attempt
    backoff_slice: float = 0.2  # one sleep within a backoff
    replay_ack: float = 10  # for the server to answer a replay
    error_log: float = 10  # between two log records of one failing tick callback
