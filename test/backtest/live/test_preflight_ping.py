"""The SDK's `ping` (openalgo/account.py), which the preflight reads the broker from: the
request it sends and the answers it returns. The SDK's HTTP client is replaced by one
whose transport answers in-process, so nothing reaches a server.

The server's answers are those of restx_api/ping.py and services/ping_service.py.
"""

from __future__ import annotations

import json

import httpx

from openalgo import api

API_KEY = "test-key-not-a-real-one"


def client_answering(handler) -> api:
    """An SDK client whose HTTP calls go to `handler`, a function of the request."""
    client = api(api_key=API_KEY, host="http://openalgo.invalid")
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def test_ping_posts_the_api_key_to_the_ping_endpoint():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"status": "success", "data": {}})

    with client_answering(handler) as client:
        client.ping()

    assert seen == [("POST", "/api/v1/ping", {"apikey": API_KEY})]


def test_ping_returns_the_servers_answer_naming_the_broker():
    pong = {"status": "success", "data": {"message": "pong", "broker": "zerodha"}}

    with client_answering(lambda request: httpx.Response(200, json=pong)) as client:
        answer = client.ping()

    assert answer == pong


def test_a_refused_ping_comes_back_in_the_sdks_http_error():
    refused = {"status": "error", "message": "Invalid openalgo apikey"}

    with client_answering(lambda request: httpx.Response(403, json=refused)) as client:
        answer = client.ping()

    assert answer["status"] == "error"
    assert answer["code"] == 403
    assert answer["error_type"] == "http_error"
    assert answer["message"].startswith("HTTP 403: ")
    assert json.loads(answer["message"].removeprefix("HTTP 403: ")) == refused


def test_ping_sends_the_extra_arguments_it_is_given_and_drops_those_that_are_none():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"status": "success", "data": {}})

    with client_answering(handler) as client:
        client.ping(probe="yes", unused=None)

    assert seen == [{"apikey": API_KEY, "probe": "yes"}]
