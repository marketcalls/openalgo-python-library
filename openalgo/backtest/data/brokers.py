"""Per-broker facts the OpenAlgo provider needs for `source="api"`. Data, not logic.

Each profile gives the longest date range to ask for in one history request (the
chunk window), the pacing between requests and whether the broker's intraday
timestamps are known to be shifted.

Source of the numbers: the OpenAlgo server's broker adapters
(`broker/<name>/api/data.py`, read at server HEAD d8cc13f24), as summarised in the
reference note `openalgo-historify-and-history.md` section 2, and PRD
`06-data-layer.md` section 4. The server adapters chunk on their own as well, so
these windows only bound how much one SDK request asks the server to assemble.

Verification status: `verified=False` marks chunk windows, and
`timestamps_verified=False` a timestamp convention, that could not be confirmed in
those sources. The provider says so in the warnings of every load that uses them.
"""

from dataclasses import dataclass

from .._layout import parse_interval

# The server spaces broker history calls 0.35 s apart, about 3 per second
# (services/history_service.py, _MIN_HISTORY_INTERVAL).
SERVER_HISTORY_SPACING_SECONDS = 0.35
# Every API route is limited to 10 requests per second by default (API_RATE_LIMIT).
DB_SPACING_SECONDS = 0.1
# One request per second, the slowest pacing any confirmed broker needs (Kotak).
CONSERVATIVE_SPACING_SECONDS = 1.0


@dataclass(frozen=True)
class BrokerProfile:
    """Chunk windows, pacing and timestamp flags of one broker.

    name: the broker name as the OpenAlgo server reports it (lower case).
    intraday_windows: ((interval_seconds, days), ...) ascending. The window of an
        interval is the days of the largest entry whose interval_seconds is not
        above it, so an interval the table does not list gets the window of the
        next finer listed interval (the cautious choice).
    daily_days: window in days for D and longer.
    spacing_seconds: least time between the starts of two history requests.
    intraday_shift_seconds: seconds by which the broker's intraday stamps are ahead
        of the true UTC instant of the bar open (19800 when IST clock time is
        delivered as UTC). The provider subtracts it and the calendar check must
        then pass. 0 means the stamps are taken as true UTC.
    verified: the chunk windows were confirmed in the server source.
    timestamps_verified: the timestamp convention was confirmed.
    notes: where the numbers come from, in words.
    """

    name: str
    intraday_windows: tuple[tuple[int, int], ...]
    daily_days: int
    spacing_seconds: float = SERVER_HISTORY_SPACING_SECONDS
    intraday_shift_seconds: int = 0
    verified: bool = True
    timestamps_verified: bool = True
    notes: str = ""

    def window_days(self, interval: str) -> int:
        """Longest inclusive date range, in days, to request at once for `interval`.

        Raises ConfigError for an unknown interval string.
        """
        kind, seconds = parse_interval(interval)
        if kind != "intraday":
            return self.daily_days
        days = self.intraday_windows[0][1]
        for listed_seconds, listed_days in self.intraday_windows:
            if listed_seconds <= seconds:
                days = listed_days
        return days


_XTS_NOTE = (
    "XTS family: the server adapter subtracts 5 hours 30 minutes from the vendor epoch "
    "for intraday bars. Whether the epochs it then delivers are true UTC could not be "
    "confirmed without a live account, so the flag is 0 and unverified; the calendar "
    "check decides at load time and names the shift if one is present."
)
_XTS_FAMILY = ("compositedge", "fivepaisaxts", "ibulls", "iifl", "jainamxts", "rmoney", "wisdom")

DEFAULT_PROFILE = BrokerProfile(
    name="default",
    intraday_windows=((1, 25), (60, 30)),
    daily_days=180,
    spacing_seconds=CONSERVATIVE_SPACING_SECONDS,
    verified=False,
    timestamps_verified=False,
    notes=(
        "Conservative default for brokers without a confirmed entry: the smallest window "
        "of the confirmed brokers for each interval class and one request per second."
    ),
)

_CONFIRMED_PROFILES = (
    BrokerProfile(
        name="zerodha",
        intraday_windows=((1, 60),),
        daily_days=2000,
        notes="60 days per request for intraday, 2000 for daily.",
    ),
    BrokerProfile(
        name="angel",
        intraday_windows=((1, 30), (180, 60), (300, 100), (900, 200), (3600, 400)),
        daily_days=2000,
        spacing_seconds=0.5,
        notes=(
            "1m 30 days, 3m 60, 5m and 10m 100, 15m and 30m 200, 1h 400, D 2000; about "
            "2 requests per second."
        ),
    ),
    BrokerProfile(
        name="upstox",
        intraday_windows=((1, 30), (1800, 90)),
        daily_days=3650,
        notes="30 days up to 15m, 90 days from 30m to 4h, 3650 days for D.",
    ),
    BrokerProfile(
        name="fyers",
        intraday_windows=((1, 25), (60, 60)),
        daily_days=300,
        notes=(
            "60 days for minute and hour bars, 300 for daily; seconds bars 25 days and "
            "only for the last 30 days."
        ),
    ),
    BrokerProfile(
        name="dhan",
        intraday_windows=((1, 90),),
        daily_days=DEFAULT_PROFILE.daily_days,
        notes=(
            "90 days per intraday request. The daily window is not confirmed (the adapter "
            "sends one request), so the conservative default is used for D."
        ),
    ),
    BrokerProfile(
        name="kotak",
        intraday_windows=((1, 30), (600, 60), (1800, 90)),
        daily_days=180,
        spacing_seconds=1.0,
        notes=(
            "1m, 3m and 5m 30 days, 10m and 15m 60, 30m and 1h 90, D and W 180; 1 request "
            "per second; history reaches back 5 years at most."
        ),
    ),
)

_XTS_PROFILES = tuple(
    BrokerProfile(
        name=name,
        intraday_windows=DEFAULT_PROFILE.intraday_windows,
        daily_days=DEFAULT_PROFILE.daily_days,
        spacing_seconds=CONSERVATIVE_SPACING_SECONDS,
        verified=False,
        timestamps_verified=False,
        notes=_XTS_NOTE,
    )
    for name in _XTS_FAMILY
)

BROKER_PROFILES = {profile.name: profile for profile in _CONFIRMED_PROFILES + _XTS_PROFILES}

# Chunk windows for source="db". Historify has no range limit; these only keep one
# JSON response to a size the server can build well inside the request timeout.
# They are an engine choice, not a server fact.
DB_PROFILE = BrokerProfile(
    name="historify",
    intraday_windows=((1, 5), (60, 60), (300, 365), (3600, 730)),
    daily_days=36_500,
    spacing_seconds=DB_SPACING_SECONDS,
    notes="Engine choice for response size; the Historify store itself has no range limit.",
)


def broker_profile(broker: str | None) -> BrokerProfile:
    """The profile for a broker name. Unknown or missing names get DEFAULT_PROFILE,
    whose `verified` flags are False so that the caller can say so."""
    if not broker:
        return DEFAULT_PROFILE
    return BROKER_PROFILES.get(str(broker).strip().lower(), DEFAULT_PROFILE)


__all__ = [
    "BROKER_PROFILES",
    "BrokerProfile",
    "DB_PROFILE",
    "DEFAULT_PROFILE",
    "broker_profile",
]
