"""An SDK client whose REST calls are answered in-process, for the REST layer tests.

The client's pooled HTTP client is replaced by one whose transport is a function of the
request, so nothing reaches a server. `recording` keeps what each request carried;
`answering` gives one fixed answer; `failing` raises the transport error a dead or slow
server would.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx

from openalgo import api

API_KEY = "test-key-not-a-real-one"
HOST = "http://openalgo.invalid"

Handler = Callable[[httpx.Request], httpx.Response]


def client_answering(handler: Handler) -> api:
    """An `openalgo.api` client whose HTTP calls go to `handler`, a function of the request."""
    client = api(api_key=API_KEY, host=HOST)
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def answering(status: int, body: object) -> Handler:
    """A handler that answers every request with `status` and the JSON `body`."""
    return lambda request: httpx.Response(status, json=body)


def recording(seen: list[tuple[str, str, object]], body: object, status: int = 200) -> Handler:
    """A handler that appends (method, URL, JSON body) of each request to `seen` and
    answers `status` with the JSON `body`."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, str(request.url), json.loads(request.content)))
        return httpx.Response(status, json=body)

    return handler


def failing(error: httpx.HTTPError) -> Handler:
    """A handler that raises `error` for every request, as the transport does when the
    server does not answer in time or cannot be reached."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    return handler
