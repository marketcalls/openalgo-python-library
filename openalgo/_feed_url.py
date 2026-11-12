"""Default WebSocket URL of the feed (private to openalgo.feed)."""

from typing import Any
from urllib.parse import urlsplit

# The port of the WebSocket proxy of a self-hosted OpenAlgo.
DEFAULT_WS_PORT = 8765
# Where the OpenAlgo installers serve the feed on a domain, behind TLS.
TLS_PATH = "/ws"


def derive_ws_url(host: str, ws_port: Any = None) -> tuple[str, str]:
    """
    Return (ws_host, ws_url) for a REST host. ws_port is None when the caller
    did not choose a port.

    - http://<host>, or a host without a scheme (local or LAN, self-hosted):
      ws://<host>:<ws_port>, on port 8765 unless one was chosen.
    - https://<host> (a server hosted on a domain, behind TLS): wss://<host>/ws.
    - https://<host> with a port chosen: ws://<host>:<ws_port>, the URL such a
      deployment has always had.

    Only the hostname of host is used: its port and its path belong to the
    REST API.
    """
    port = DEFAULT_WS_PORT if ws_port is None else ws_port
    try:
        # urlsplit finds a network location only behind "//".
        parts = urlsplit(host if "://" in host else f"//{host}")
        hostname = parts.hostname or ""
    except ValueError:
        # Not a URL (for example an unbalanced bracket). The feed never
        # refused such text, so keep it and let connect() fail on it.
        return host, f"ws://{host}:{port}"

    # urlsplit lower-cases the hostname. Cut the same text out of the network
    # location, to keep the capitalisation the caller wrote.
    start = parts.netloc.lower().rfind(hostname)
    ws_host = parts.netloc[start : start + len(hostname)]
    if ":" in ws_host:
        # An IPv6 literal needs its brackets in a URL.
        ws_host = f"[{ws_host}]"
    if parts.scheme == "https" and ws_port is None:
        return ws_host, f"wss://{ws_host}{TLS_PATH}"
    return ws_host, f"ws://{ws_host}:{port}"
