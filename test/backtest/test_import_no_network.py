"""Importing `openalgo` and `openalgo.backtest` opens no socket and makes no network call
(01-requirements 6.4: a network call on import is one of the pitfalls the engine must
not repeat).

Each check runs a fresh interpreter. Before anything else is imported, the child replaces
socket creation, connection and name lookup with stand-ins that record the attempt and
refuse it, then imports the package, touches every lazily loaded name of
`openalgo.backtest` (and, in the second check, loads the built-in calendar, reference
tables and cost schedules), and prints the attempts it recorded as JSON on its last
line. An attempt is recorded even when the code that made it swallows the refusal.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Runs first in the child: every way to reach the network goes through one of these.
BLOCK_NETWORK = """
import json
import socket
import sys

sys.path.insert(0, {root!r})
attempts = []


def refuse(name):
    def blocked(*args, **kwargs):
        attempts.append(name)
        raise OSError(f"network access refused in this test: {{name}}")

    return blocked


class RefusedSocket(socket.socket):
    def __init__(self, *args, **kwargs):
        attempts.append("socket.socket")
        raise OSError("network access refused in this test: socket.socket")


socket.socket = RefusedSocket
for name in ("create_connection", "getaddrinfo", "gethostbyname", "gethostbyname_ex",
             "gethostbyaddr", "socketpair", "fromfd"):
    setattr(socket, name, refuse("socket." + name))
"""

IMPORT_EVERYTHING = """
import openalgo
import openalgo.backtest as bt

for name in bt.__all__:
    getattr(bt, name)
"""

LOAD_BUILT_IN_DATA = """
bt.calendar.default_calendar().sessions("NSE", "2026-01-01", "2026-12-31")
bt.reference.default_tables().coverage()
bt.costs.trade_cost("BUY", 1000.0, 10, "2026-11-04", segment="NSE_EQ_DELIVERY")
"""

REPORT = """
print(json.dumps(attempts))
"""


def network_attempts(body: str) -> list[str]:
    """Run `body` in a fresh interpreter with the network blocked; the attempts made."""
    code = BLOCK_NETWORK.format(root=str(ROOT)) + body + REPORT
    finished = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=300
    )
    assert finished.returncode == 0, finished.stderr
    return json.loads(finished.stdout.strip().splitlines()[-1])


def test_the_blocker_records_a_connection_attempt():
    # The check itself works: a plain connection attempt is caught and recorded.
    body = "try:\n    socket.create_connection(('127.0.0.1', 5000))\nexcept OSError:\n    pass\n"
    assert network_attempts(body) == ["socket.create_connection"]


def test_importing_openalgo_and_every_backtest_name_makes_no_network_call():
    assert network_attempts(IMPORT_EVERYTHING) == []


def test_loading_the_built_in_calendar_tables_and_costs_makes_no_network_call():
    assert network_attempts(IMPORT_EVERYTHING + LOAD_BUILT_IN_DATA) == []
