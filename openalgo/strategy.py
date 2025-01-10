"""
OpenAlgo Strategy Module for TradingView Integration
"""

from typing import Optional

class Strategy:
    def __init__(self, host_url: str, webhook_id: str):
        """
        Initialize strategy with host URL and webhook ID
        
        Args:
            host_url (str): OpenAlgo server URL (e.g., "http://127.0.0.1:5000")
            webhook_id (str): Strategy's webhook ID from OpenAlgo
        """
        self._host_url = host_url.rstrip('/')
        self._webhook_id = webhook_id
        self._webhook_url = None

    @property
    def webhook_url(self) -> str:
        """
        Cached property for webhook URL to avoid reconstructing it every time
        """
        if self._webhook_url is None:
            self._webhook_url = f"{self._host_url}/strategy/webhook/{self._webhook_id}"
        return self._webhook_url

    def strategyorder(self, symbol: str, action: str, position_size: Optional[int] = None) -> dict:
        """
        Format the strategy order message for TradingView webhook.
        The strategy mode (LONG_ONLY, SHORT_ONLY, BOTH) is configured in OpenAlgo.
        
        Args:
            symbol (str): Trading symbol (e.g., "RELIANCE", "NIFTY")
            action (str): Order action ("BUY" or "SELL")
            position_size (Optional[int]): Position size, required for BOTH mode
            
        Returns:
            dict: Message format for TradingView webhook
        """
        message = {
            "symbol": symbol,
            "action": action.upper()
        }
        
        if position_size is not None:
            message["position_size"] = str(position_size)
            
        return message
