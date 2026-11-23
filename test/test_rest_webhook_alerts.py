"""`api.discord` (openalgo/discord.py) and `api.slack` (openalgo/slack.py): the request
each sends to OpenAlgo's notify endpoint and the answers it returns. Both channels follow
the same rules, so each behaviour is checked for both. The client's transport answers
in-process, so nothing reaches a server.

The server's answers are those of the API pages (api-documentation/v1/utilities-api/
discord.md and slack.md) and of restx_api/webhook_alerts.py with services/alerts.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import numpy
import pytest
from rest_fakes import (
    API_KEY,
    answering,
    client_answering,
    failing,
    recording,
)


@dataclass(frozen=True)
class Channel:
    """One webhook channel as its API page describes it."""

    method: str
    url: str
    label: str
    max_characters: int


DISCORD = Channel("discord", "http://openalgo.invalid/api/v1/discord/notify", "Discord", 2000)
SLACK = Channel("slack", "http://openalgo.invalid/api/v1/slack/notify", "Slack", 4000)
CHANNELS = pytest.mark.parametrize("channel", [DISCORD, SLACK], ids=lambda c: c.method)

QUEUED = {"status": "success", "message": "Notification queued."}


def send(channel: Channel, handler, *args, **kwargs) -> dict:
    """Call the channel's method on a client answered by `handler`; its return value."""
    with client_answering(handler) as client:
        return getattr(client, channel.method)(*args, **kwargs)


@CHANNELS
def test_a_message_is_posted_with_the_api_key_and_queued_by_default(channel):
    seen = []

    send(channel, recording(seen, QUEUED), "NIFTY breakout above 24,200")

    assert seen == [
        (
            "POST",
            channel.url,
            {
                "apikey": API_KEY,
                "message": "NIFTY breakout above 24,200",
                "wait_for_delivery": False,
            },
        )
    ]


@CHANNELS
def test_wait_for_delivery_asks_the_server_to_post_before_it_answers(channel):
    seen = []

    send(channel, recording(seen, QUEUED), "Stop-loss hit", wait_for_delivery=True)

    assert seen[0][2]["wait_for_delivery"] is True


@CHANNELS
def test_a_numpy_bool_for_wait_for_delivery_is_sent_as_the_bool_it_holds(channel):
    # A flag read from pandas is a NumPy bool, which JSON cannot carry as it is.
    seen = []

    send(channel, recording(seen, QUEUED), "Stop-loss hit", wait_for_delivery=numpy.bool_(True))

    assert [body["wait_for_delivery"] for _, _, body in seen] == [True]


@CHANNELS
def test_a_message_is_sent_as_written_and_the_server_does_any_escaping(channel):
    # The server shows mentions as text and escapes '&', '<' and '>' for Slack; the SDK
    # sending them escaped would show the escapes in the channel.
    written = "**NIFTY** & BANKNIFTY <@here> @everyone"
    seen = []

    send(channel, recording(seen, QUEUED), written)

    assert seen[0][2]["message"] == written


@CHANNELS
def test_the_queued_answer_comes_back_as_it_is(channel):
    assert send(channel, answering(200, QUEUED), "hello") == QUEUED


@CHANNELS
def test_the_delivered_answer_comes_back_as_it_is(channel):
    delivered = {"status": "success", "message": f"Message sent to {channel.label}."}

    answer = send(channel, answering(200, delivered), "hello", wait_for_delivery=True)

    assert answer == delivered


def refusals(channel: Channel) -> list[tuple[int, str]]:
    """Every refusal the channel's API page lists: the status and the server's message."""
    label = channel.label
    return [
        (400, "'message' is required."),
        (400, f"'message' must be at most {channel.max_characters} characters."),
        (400, "'wait_for_delivery' must be true or false."),
        (400, "Send the request as a JSON object."),
        (401, "Invalid or missing API key"),
        (409, f"{label} alerts are turned off. Turn them on from the Alerts page in OpenAlgo."),
        (409, f"No {label} webhook is saved. Add one on the {label} page in OpenAlgo."),
        (
            502,
            f"{label} no longer accepts this webhook. It may have been deleted, or its "
            f"channel archived. Create a new webhook and save it on the {label} page.",
        ),
        (502, f"OpenAlgo could not reach {label}. Check this server's internet connection."),
        (503, f"Too many {label} messages are waiting to be sent. Try again shortly."),
    ]


@pytest.mark.parametrize(
    ("channel", "status", "message"),
    [(channel, *refusal) for channel in (DISCORD, SLACK) for refusal in refusals(channel)],
    ids=lambda value: value.method if isinstance(value, Channel) else str(value)[:24],
)
def test_a_refusal_comes_back_with_the_servers_message_and_its_status_code(
    channel, status, message
):
    refused = {"status": "error", "message": message}

    answer = send(channel, answering(status, refused), "hello")

    assert answer == {"status": "error", "message": message, "code": status}


@CHANNELS
def test_the_rate_limit_refusal_comes_back_with_its_retry_time(channel):
    # The server's answer to a caller over 30 requests a minute (app.py, the 429 handler).
    over_limit = {
        "status": "error",
        "message": "Rate limit exceeded. Please slow down your requests.",
        "retry_after": 60,
    }

    answer = send(channel, answering(429, over_limit), "hello")

    assert answer == {
        "status": "error",
        "message": "Rate limit exceeded. Please slow down your requests.",
        "retry_after": 60,
        "code": 429,
    }


@CHANNELS
@pytest.mark.parametrize(
    ("message", "wait_for_delivery", "server_says"),
    [
        ("", False, "'message' is required."),
        ("   ", False, "'message' is required."),
        ("x", "yes", "'wait_for_delivery' must be true or false."),
    ],
    ids=["empty", "blank", "wait-not-a-bool"],
)
def test_input_the_server_refuses_is_sent_as_given_and_its_refusal_comes_back(
    channel, message, wait_for_delivery, server_says
):
    # The SDK leaves these checks to the server, as telegram() and whatsapp() do.
    refused = {"status": "error", "message": server_says}
    seen = []

    answer = send(
        channel,
        recording(seen, refused, status=400),
        message,
        wait_for_delivery=wait_for_delivery,
    )

    assert seen[0][2] == {
        "apikey": API_KEY,
        "message": message,
        "wait_for_delivery": wait_for_delivery,
    }
    assert answer == {"status": "error", "message": server_says, "code": 400}


@CHANNELS
def test_a_message_over_the_limit_is_sent_whole_and_the_servers_refusal_comes_back(channel):
    too_long = "x" * (channel.max_characters + 1)
    server_says = f"'message' must be at most {channel.max_characters} characters."
    refused = {"status": "error", "message": server_says}
    seen = []

    answer = send(channel, recording(seen, refused, status=400), too_long)

    assert len(seen[0][2]["message"]) == channel.max_characters + 1
    assert answer == {"status": "error", "message": server_says, "code": 400}


@CHANNELS
@pytest.mark.parametrize(
    ("failure", "error_type"),
    [
        (httpx.ReadTimeout("no answer"), "timeout_error"),
        (httpx.ConnectError("refused"), "connection_error"),
    ],
    ids=["timed-out", "unreachable"],
)
def test_a_server_that_does_not_answer_gives_the_error_telegram_gives(channel, failure, error_type):
    with client_answering(failing(failure)) as client:
        answer = getattr(client, channel.method)("hello")
        telegram_answer = client.telegram(username="trader", message="hello")

    assert answer == telegram_answer
    assert answer["error_type"] == error_type


@CHANNELS
def test_wait_for_delivery_is_keyword_only(channel):
    with pytest.raises(TypeError):
        send(channel, answering(200, QUEUED), "hello", True)
