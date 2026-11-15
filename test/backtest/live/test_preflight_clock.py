"""The clock check of a live session's preflight (08-live-trading section 4): the local
clock against the server's, read from the `Date` header of one HEAD request.

`Date` holds whole seconds D, and the request runs from local t0 to t1, so the local
clock minus the server's lies in (t0 - D - 1 s, t1 - D]. The check fails only when that
whole interval is beyond 2 seconds either way, and warns when it straddles the limit or
nothing could be measured. The fake server dates its answers by its own clock
(`FakeOpenAlgo.server_ns`) and answers 404, as any status will do.
"""

from __future__ import annotations

import httpx
import pytest
from preflight_fakes import BASE_URL, NOW_NS, NS, FakeOpenAlgo, make_plan

from openalgo.backtest._live._preflight_clock import Skew, SkewBounds, clock_check
from openalgo.backtest._live.preflight import WARNING, run_preflight
from openalgo.backtest.errors import PreflightError

MS = NS // 1000

SYNC = "synchronise this computer's clock with an internet time server"


def ticking(*instants: int):
    """A clock that gives `instants` in turn: the local time before and after a request."""
    given = iter(instants)
    return lambda: next(given)


def check_at(client: FakeOpenAlgo, *instants: int):
    """The clock check against `client`, the local clock reading `instants` in turn (one
    instant for both readings when only one is given)."""
    now = ticking(*instants) if len(instants) > 1 else (lambda: instants[0])
    return clock_check(client, now)


# ----- the interval arithmetic -------------------------------------------------------


@pytest.mark.parametrize(
    ("t0_after_date_ms", "t1_after_date_ms", "low_ms", "high_ms"),
    [
        (0, 0, -1000, 0),
        (300, 500, -700, 500),
        (999, 999, -1, 999),
        (-2000, -1500, -3000, -1500),
        (4000, 4200, 3000, 4200),
        (0, 4000, -1000, 4000),
    ],
)
def test_the_skew_lies_after_t0_minus_the_date_minus_a_second_up_to_t1_minus_the_date(
    t0_after_date_ms, t1_after_date_ms, low_ms, high_ms
):
    date_ns = NOW_NS

    bounds = SkewBounds.measured(
        date_ns + t0_after_date_ms * MS, date_ns + t1_after_date_ms * MS, date_ns
    )

    assert bounds == SkewBounds(low_ms * MS, high_ms * MS)


@pytest.mark.parametrize(
    ("low_ms", "high_ms", "verdict"),
    [
        (-1000, 0, Skew.WITHIN),
        (-2000, 2000, Skew.WITHIN),
        (-700, 500, Skew.WITHIN),
        (2000, 3000, Skew.BEYOND),
        (3000, 4200, Skew.BEYOND),
        (-3500, -2001, Skew.BEYOND),
        (-3000, -2000, Skew.UNSURE),
        (1999, 2999, Skew.UNSURE),
        (1500, 2500, Skew.UNSURE),
        (-2100, -1100, Skew.UNSURE),
        (-5000, 5000, Skew.UNSURE),
    ],
)
def test_an_interval_is_beyond_the_limit_only_when_every_skew_in_it_is(low_ms, high_ms, verdict):
    assert SkewBounds(low_ms * MS, high_ms * MS).verdict is verdict


# ----- the check ---------------------------------------------------------------------


def test_the_check_sends_one_head_request_to_the_api_url_with_no_key_and_a_short_wait():
    client = FakeOpenAlgo()

    check_at(client, NOW_NS)

    (request,) = client.requests
    assert (request.method, str(request.url)) == ("HEAD", BASE_URL)
    assert request.content == b""
    assert "apikey" not in str(request.url)
    assert request.extensions["timeout"] == {"connect": 5.0, "read": 5.0, "write": 5.0, "pool": 5.0}


def test_a_clock_in_step_with_the_server_passes_without_a_warning():
    check = check_at(FakeOpenAlgo(server_ns=NOW_NS), NOW_NS)

    assert check.ok
    assert check.message == (
        "the local clock is within 2 seconds of OpenAlgo's server (it is -1.0 to +0.0 seconds off)"
    )


def test_a_clock_certainly_ahead_of_the_server_fails_saying_by_how_much():
    check = check_at(FakeOpenAlgo(server_ns=NOW_NS - 5 * NS), NOW_NS)

    assert not check.ok
    assert check.message == (
        "the local clock is 4.0 to 5.0 seconds ahead of OpenAlgo's server, more than the 2 "
        f"seconds allowed: {SYNC}, then start the session again"
    )


def test_a_clock_certainly_behind_the_server_fails_saying_by_how_much():
    check = check_at(FakeOpenAlgo(server_ns=NOW_NS + 5 * NS), NOW_NS)

    assert not check.ok
    assert check.message.startswith(
        "the local clock is 5.0 to 6.0 seconds behind OpenAlgo's server, more than the 2 "
        "seconds allowed"
    )


def test_a_clock_that_may_be_off_by_more_than_the_limit_passes_with_a_warning():
    check = check_at(FakeOpenAlgo(server_ns=NOW_NS - 2 * NS), NOW_NS + 500 * MS)

    assert check.ok
    assert check.message == (
        f"{WARNING}the local clock is +1.5 to +2.5 seconds off OpenAlgo's server (the request "
        f"took 0.0 seconds), so it may be more than the 2 seconds allowed: {SYNC}"
    )


def test_a_slow_answer_widens_the_interval_into_a_warning_that_names_the_wait():
    check = check_at(FakeOpenAlgo(server_ns=NOW_NS), NOW_NS, NOW_NS + 4 * NS)

    assert check.ok
    assert check.message.startswith(
        f"{WARNING}the local clock is -1.0 to +4.0 seconds off OpenAlgo's server (the "
        "request took 4.0 seconds)"
    )


def test_a_date_in_the_unknown_zone_of_internet_mail_is_read_as_utc():
    client = FakeOpenAlgo(head_headers={"Date": "Mon, 16 Nov 2026 04:45:20 -0000"})

    check = check_at(client, NOW_NS)

    assert check.message.startswith("the local clock is within 2 seconds")


@pytest.mark.parametrize(
    ("server", "why"),
    [
        ({"head_headers": {}}, "its answer carried no Date header"),
        (
            {"head_headers": {"Date": "yesterday"}},
            "its Date header 'yesterday' could not be read",
        ),
        (
            {"head_error": httpx.ConnectError("connection refused")},
            "the request for its time failed (ConnectError: connection refused)",
        ),
        (
            {"head_error": httpx.ReadTimeout("timed out")},
            "the request for its time failed (ReadTimeout: timed out)",
        ),
    ],
)
def test_a_clock_that_cannot_be_measured_passes_with_a_warning_saying_why(server, why):
    check = check_at(FakeOpenAlgo(**server), NOW_NS)

    assert check.ok
    assert check.message == (
        f"{WARNING}the local clock could not be compared with OpenAlgo's server, as {why}: "
        f"{SYNC}, since the session needs it within 2 seconds of the exchange's"
    )


def test_a_local_clock_that_goes_back_during_the_request_passes_with_a_warning():
    check = check_at(FakeOpenAlgo(), NOW_NS + 3 * NS, NOW_NS)

    assert check.ok
    assert "the local clock went back while the request was answered" in check.message


# ----- in the preflight --------------------------------------------------------------


def test_a_clock_certainly_off_fails_the_preflight():
    client = FakeOpenAlgo(server_ns=NOW_NS - 10 * NS)

    with pytest.raises(PreflightError) as raised:
        run_preflight(client, make_plan(), now=lambda: NOW_NS)

    (failure,) = raised.value.failures
    assert failure.startswith("the local clock is 9.0 to 10.0 seconds ahead of OpenAlgo's server")


def test_the_clock_is_not_read_when_ping_fails():
    client = FakeOpenAlgo(
        ping={"status": "error", "message": "down", "error_type": "connection_error"}
    )

    with pytest.raises(PreflightError):
        run_preflight(client, make_plan(), now=lambda: NOW_NS)

    assert client.requests == []
