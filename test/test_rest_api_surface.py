"""The `openalgo.api` client carries the Charges, Discord and Slack calls, builds its
shared HTTP state once, and importing `openalgo` and building a client still needs no
server.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import openalgo
from openalgo import BaseAPI, api

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("method", ["charges", "discord", "slack"])
def test_the_client_has_the_method(method):
    assert callable(getattr(api, method, None))


@pytest.mark.parametrize(("mixin", "module"), [("DiscordAPI", "discord"), ("SlackAPI", "slack")])
def test_the_alert_mixin_is_exported_and_composed_into_the_client(mixin, module):
    exported = getattr(openalgo, mixin)

    assert exported.__module__ == f"openalgo.{module}"
    assert issubclass(api, exported)


def test_building_the_client_runs_the_shared_setup_once(monkeypatch):
    calls = []
    setup = BaseAPI.__init__

    def counted(self, *args, **kwargs):
        calls.append(args)
        setup(self, *args, **kwargs)

    monkeypatch.setattr(BaseAPI, "__init__", counted)

    with api(api_key="test-key-not-a-real-one", host="http://openalgo.invalid"):
        pass

    assert len(calls) == 1


# Runs in a fresh interpreter: every way to open a connection records the attempt and is
# refused, then the package is imported and a client is built with its three new calls.
NO_SERVER = """
import json
import socket
import sys

sys.path.insert(0, {root!r})
attempts = []


def refuse(name):
    def refused(*args, **kwargs):
        attempts.append(name)
        raise OSError("network access refused in this test: " + name)

    return refused


socket.socket.connect = refuse("socket.connect")
socket.socket.connect_ex = refuse("socket.connect_ex")
socket.create_connection = refuse("socket.create_connection")
socket.getaddrinfo = refuse("socket.getaddrinfo")

import openalgo

client = openalgo.api(api_key="test-key-not-a-real-one")
methods = [callable(getattr(client, name)) for name in ("charges", "discord", "slack")]
client.close()
print(json.dumps({{"attempts": attempts, "methods": methods}}))
"""


def test_importing_openalgo_and_building_a_client_contacts_no_server():
    finished = subprocess.run(
        [sys.executable, "-c", NO_SERVER.format(root=str(ROOT))],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=300,
    )

    assert finished.returncode == 0, finished.stderr
    report = json.loads(finished.stdout.strip().splitlines()[-1])
    assert report == {"attempts": [], "methods": [True, True, True]}
