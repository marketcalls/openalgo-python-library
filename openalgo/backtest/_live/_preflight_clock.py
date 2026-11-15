"""The clock check of a live session's preflight (`preflight`, 08-live-trading section 4):
the local clock against the OpenAlgo server's, read from the HTTP `Date` header.

No answer of OpenAlgo's REST API carries the server's time in its body, but every
response of its WSGI server carries a `Date` header. The check sends one HEAD request to
the API's URL through the SDK's own pooled client (any status will do, and no API key is
sent), noting the local time before (t0) and after (t1). `Date` holds whole seconds, so
the server's time when it answered was in [D, D + 1 s), at some local instant in
[t0, t1]: the local clock minus the server's lies in (t0 - D - 1 s, t1 - D].

The check fails only when that whole interval is beyond 2 seconds either way (the clock
is certainly off), passes when it lies within, and passes with a warning, saying why,
when it straddles the limit, when the header is missing or unreadable, or when the
request fails. Instants are UTC nanoseconds.
"""

from __future__ import annotations

import datetime as dt
import email.utils
import enum
from dataclasses import dataclass

import httpx

from .._layout import NANOS_PER_SECOND
from ._preflight_plan import WARNING, Check
from ._preflight_server import PreflightClient
from ._types import Now

#: The most the local clock may differ from the server's, in seconds (08-live-trading
#: section 4).
MAX_CLOCK_SKEW_SECONDS = 2
_MAX_SKEW_NS = MAX_CLOCK_SKEW_SECONDS * NANOS_PER_SECOND
# An HTTP date has whole seconds (RFC 9110, section 5.6.7).
_DATE_RESOLUTION_NS = NANOS_PER_SECOND
# The longest wait for the HEAD request, seconds: over a longer round trip the interval is
# too wide to show a skew within the limit, so waiting longer could tell nothing more.
_REQUEST_TIMEOUT_SECONDS = 5.0
_SYNC_ADVICE = "synchronise this computer's clock with an internet time server"


class Skew(enum.Enum):
    """What an interval of skews says of the local clock."""

    #: Within the limit, whatever the skew in the interval is.
    WITHIN = "within"
    #: Beyond the limit, whatever the skew in the interval is: the clock is off.
    BEYOND = "beyond"
    #: The interval straddles the limit.
    UNSURE = "unsure"


@dataclass(frozen=True)
class SkewBounds:
    """The local clock minus the server's lies in (`low_ns`, `high_ns`], nanoseconds."""

    low_ns: int
    high_ns: int

    @classmethod
    def measured(cls, t0_ns: int, t1_ns: int, date_ns: int) -> SkewBounds:
        """The bounds from the local instants before (`t0_ns`) and after (`t1_ns`) the
        request and the server's `Date` (`date_ns`, whole seconds), UTC nanoseconds."""
        return cls(t0_ns - date_ns - _DATE_RESOLUTION_NS, t1_ns - date_ns)

    @property
    def verdict(self) -> Skew:
        """BEYOND when no skew of the interval is within `MAX_CLOCK_SKEW_SECONDS`, WITHIN
        when every one is, UNSURE otherwise."""
        if self.low_ns >= _MAX_SKEW_NS or self.high_ns < -_MAX_SKEW_NS:
            return Skew.BEYOND
        if self.low_ns >= -_MAX_SKEW_NS and self.high_ns <= _MAX_SKEW_NS:
            return Skew.WITHIN
        return Skew.UNSURE


def clock_check(client: PreflightClient, now: Now) -> Check:
    """Measure the local clock against the server's with one HEAD request to
    `client.base_url` through `client.client`, timed by `now` (UTC nanoseconds), and
    judge it (see the module docstring)."""
    t0_ns = now()
    try:
        response = client.client.head(client.base_url, timeout=_REQUEST_TIMEOUT_SECONDS)
    except httpx.HTTPError as error:
        return _unmeasured(f"the request for its time failed ({type(error).__name__}: {error})")
    t1_ns = now()
    date = response.headers.get("date")
    if date is None:
        return _unmeasured("its answer carried no Date header")
    date_ns = _date_ns(date)
    if date_ns is None:
        return _unmeasured(f"its Date header {date!r} could not be read")
    if t1_ns < t0_ns:
        return _unmeasured("the local clock went back while the request was answered")
    return _judged(SkewBounds.measured(t0_ns, t1_ns, date_ns), t1_ns - t0_ns)


def _date_ns(date: str) -> int | None:
    """An HTTP date as UTC nanoseconds, or None when it cannot be read."""
    try:
        moment = email.utils.parsedate_to_datetime(date)
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        # The "-0000" zone of RFC 5322: a time in UTC whose local zone is not told.
        moment = moment.replace(tzinfo=dt.UTC)
    return int(moment.timestamp()) * NANOS_PER_SECOND


def _judged(bounds: SkewBounds, round_trip_ns: int) -> Check:
    low, high = bounds.low_ns / NANOS_PER_SECOND, bounds.high_ns / NANOS_PER_SECOND
    verdict = bounds.verdict
    if verdict is Skew.WITHIN:
        return Check(
            "clock",
            True,
            f"the local clock is within {MAX_CLOCK_SKEW_SECONDS} seconds of OpenAlgo's server "
            f"(it is {low:+.1f} to {high:+.1f} seconds off)",
        )
    if verdict is Skew.BEYOND:
        if low > 0:
            off = f"{low:.1f} to {high:.1f} seconds ahead of"
        else:
            off = f"{-high:.1f} to {-low:.1f} seconds behind"
        return Check(
            "clock",
            False,
            f"the local clock is {off} OpenAlgo's server, more than the "
            f"{MAX_CLOCK_SKEW_SECONDS} seconds allowed: {_SYNC_ADVICE}, then start the "
            "session again",
        )
    return Check(
        "clock",
        True,
        f"{WARNING}the local clock is {low:+.1f} to {high:+.1f} seconds off OpenAlgo's "
        f"server (the request took {round_trip_ns / NANOS_PER_SECOND:.1f} seconds), so it "
        f"may be more than the {MAX_CLOCK_SKEW_SECONDS} seconds allowed: {_SYNC_ADVICE}",
    )


def _unmeasured(why: str) -> Check:
    return Check(
        "clock",
        True,
        f"{WARNING}the local clock could not be compared with OpenAlgo's server, as {why}: "
        f"{_SYNC_ADVICE}, since the session needs it within {MAX_CLOCK_SKEW_SECONDS} "
        "seconds of the exchange's",
    )


__all__ = ["MAX_CLOCK_SKEW_SECONDS", "Skew", "SkewBounds", "clock_check"]
