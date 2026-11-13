# -*- coding: utf-8 -*-
"""
OpenAlgo Python Library
"""

from .base import BaseAPI
from .orders import OrderAPI
from .gtt import GTTAPI
from .data import DataAPI
from .account import AccountAPI
from .strategy import Strategy
from .strategy_api import StrategyAPI
from .feed import FeedAPI
from .options import OptionsAPI
from .telegram import TelegramAPI
from .whatsapp import WhatsAppAPI
from .utilities import UtilitiesAPI
from .indicators import ta

# ------------------------------------------------------------------
# Indicator math runs in the Rust core (openalgo._oaindicators); numba/llvmlite
# are no longer dependencies. The shim below keeps legacy decorator imports working.
# ------------------------------------------------------------------
from .numba_shim import jit as _jit_shim, prange as _prange, HAS_NUMBA  # noqa: E402

# Back-compat aliases for any user code that referenced these.
nbjit = _jit_shim
prange = _prange

class api(OrderAPI, GTTAPI, DataAPI, AccountAPI, FeedAPI, OptionsAPI,
          StrategyAPI, TelegramAPI, WhatsAppAPI, UtilitiesAPI):
    """
    OpenAlgo API client class
    """
    def __init__(self, api_key, host="http://127.0.0.1:5000", version="v1", timeout=120.0,
                 ws_port=None, ws_url=None, verbose=False, auto_reconnect=True):
        """
        Initialize the OpenAlgo API client.

        Args:
            api_key (str): User's API key.
            host (str): Base URL for the API endpoints. Defaults to localhost.
            version (str): API version. Defaults to "v1".
            timeout (float): Request timeout in seconds. Defaults to 120.0 seconds.
            ws_port (int, optional): WebSocket server port. Defaults to 8765, which is
                also what self.ws_port reads when it is not passed.
            ws_url (str, optional): Custom WebSocket URL. Overrides host and ws_port.
                Without it the WebSocket URL is derived from host:
                - http://<host>, or a host without a scheme (local or LAN):
                  ws://<host>:<ws_port>
                - https://<host> (a server hosted on a domain): wss://<host>/ws,
                  where the OpenAlgo installers serve the feed
                - https://<host> with ws_port passed: ws://<host>:<ws_port>
                The port and the path of host belong to the REST API and are not used.
            verbose (int): Logging verbosity level. Defaults to False.
                - 0 or False: Silent mode (errors only)
                - 1 or True: Basic info (connection, auth, subscription status)
                - 2: Full debug (all market data updates)
            auto_reconnect (bool): If True (default), the SDK transparently
                reconnects, re-authenticates, and replays all active
                subscriptions after a drop with exponential backoff. Set to
                False to keep the previous manual-reconnect behaviour.
        """
        # Initialize BaseAPI for REST functionality
        BaseAPI.__init__(self, api_key, host, version, timeout)

        # Initialize FeedAPI WebSocket attributes
        # (with the helper FeedAPI.__init__ also runs, so the two cannot drift apart)
        self._init_feed(host, ws_port, ws_url, verbose, auto_reconnect)

__version__ = "2.1.1"

# Export main components for easy access
__all__ = ['api', 'Strategy', 'ta', 'nbjit', 'prange', 'HAS_NUMBA']
