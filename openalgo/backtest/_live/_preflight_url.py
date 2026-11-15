"""The OpenAlgo server a client talks to, read from the URL of its API (`BaseAPI.base_url`,
"<host>/api/v1/"): the server's root, whether it runs on this computer, and the key that
names it for the lock of one session per server (`_preflight_lock`).
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from types import MappingProxyType

import httpx

from ..errors import ConfigError

#: The port of each scheme a server URL may have when it names none.
_DEFAULT_PORTS = MappingProxyType({"http": 80, "https": 443})
#: The host every loopback host is named by in a server's key: localhost, 127.0.0.1 and
#: ::1 reach the same server on this computer.
_LOOPBACK_HOST = "localhost"


@dataclass(frozen=True)
class ServerAddress:
    """Where an OpenAlgo server listens: `scheme` ("http" or "https"), `host` (lower
    case, an IPv6 address without brackets) and `port`."""

    scheme: str
    host: str
    port: int

    @classmethod
    def of(cls, url: str) -> ServerAddress:
        """The address in a server URL (the client's `base_url`, or its host).

        Raises ConfigError unless the URL is http or https and names a host.
        """
        try:
            parsed = httpx.URL(url)
        except (httpx.InvalidURL, TypeError) as error:
            raise ConfigError(
                f"the OpenAlgo server's URL {url!r} cannot be read: {error}"
            ) from error
        scheme = parsed.scheme.lower()
        if scheme not in _DEFAULT_PORTS or not parsed.host:
            raise ConfigError(
                f"the OpenAlgo server's URL must start with http:// or https:// and name a "
                f"host, not {url!r}"
            )
        return cls(scheme, parsed.host.lower(), parsed.port or _DEFAULT_PORTS[scheme])

    @property
    def on_this_computer(self) -> bool:
        """True for a loopback host: localhost, 127.0.0.0/8 or ::1."""
        if self.host == _LOOPBACK_HOST:
            return True
        try:
            return ipaddress.ip_address(self.host).is_loopback
        except ValueError:
            return False

    @property
    def root(self) -> str:
        """The server's root URL, "<scheme>://<host>[:<port>]/", with no path, query or
        credentials."""
        host = f"[{self.host}]" if ":" in self.host else self.host
        port = "" if self.port == _DEFAULT_PORTS[self.scheme] else f":{self.port}"
        return f"{self.scheme}://{host}{port}/"

    @property
    def key(self) -> str:
        """The name of the server: "<scheme>://<host>:<port>", every loopback host named
        localhost, so one server reached by two names has one key."""
        host = _LOOPBACK_HOST if self.on_this_computer else self.host
        return f"{self.scheme}://{host}:{self.port}"


__all__ = ["ServerAddress"]
