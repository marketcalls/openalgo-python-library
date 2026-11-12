"""What a data callback receives and what a getter returns are copies: a
callback or a caller that edits them, at any depth, cannot change what the feed
has stored.

Fake socket only: see feed_fakes.py. Nothing here reaches the network.
"""

import copy

import pytest
from feed_fakes import KINDS, message_for, subscribe_kind
from feed_fixtures import *  # noqa: F403

# The getter that returns what an update of each kind stored.
GETTERS = {"ltp": "get_ltp", "quote": "get_quotes", "depth": "get_depth", "order": "get_orders"}


def overwrite_every_value(value):
    """Replace every value inside a dict or list, at every depth, with None."""
    items = value.items() if isinstance(value, dict) else enumerate(value)
    for key, item in list(items):
        if isinstance(item, (dict, list)):
            overwrite_every_value(item)
        else:
            value[key] = None


@pytest.mark.parametrize("kind", KINDS)
def test_callback_that_edits_its_update_leaves_the_stored_snapshot_as_received(
    server, client, kind
):
    getter = getattr(client, GETTERS[kind])
    seen_by_the_callback = []

    def edits_after_looking(update):
        seen_by_the_callback.append(copy.deepcopy(getter()))
        overwrite_every_value(update)

    assert client.connect() is True
    assert subscribe_kind(client, kind, edits_after_looking) is True
    server.last.deliver(message_for(kind, 7))
    assert getter() == seen_by_the_callback[0]


@pytest.mark.parametrize("kind", KINDS)
def test_caller_that_edits_what_a_getter_returned_leaves_the_stored_snapshot_as_received(
    server, client, kind
):
    getter = getattr(client, GETTERS[kind])
    assert client.connect() is True
    server.last.deliver(message_for(kind, 7))
    returned = getter()
    as_received = copy.deepcopy(returned)
    overwrite_every_value(returned)
    assert getter() == as_received
