# -*- coding: utf-8 -*-
"""
layr0_imc REST API Documentation - Telegram Notification Methods
    https://docs.layr0.org
"""

from .base import BaseAPI

class TelegramAPI(BaseAPI):
    """
    Telegram notification API methods for layr0_imc.
    Inherits from the BaseAPI class.
    """

    def telegram(self, *, username, message, priority=5, **kwargs):
        """
        Send Custom Alert Messages to Telegram Users.

        Prerequisites:
        1. Telegram Bot Must Be Running
           - Start the bot from layr0_imc Telegram settings
           - Bot must be active to send alerts
        2. User Must Be Linked
           - User must have linked their account using /link command in Telegram
           - Username must match your layr0_imc login username (NOT Telegram username)
        3. Valid API Key
           - API key must be active and valid

        Parameters:
        - username (str): layr0_imc login username (the username used to login to layr0_imc app,
                         NOT Telegram username). Required.
        - message (str): Alert message to send. Required.
                        Supports emojis, markdown, and line breaks (use \\n).
                        Maximum length: 4096 characters (Telegram limit).
        - priority (int, optional): Message priority (1-10, higher = more urgent). Defaults to 5.
                                   Priority Levels:
                                   - 1-3: Low Priority (General updates, market news)
                                   - 4-6: Normal Priority (Trade signals, daily summaries)
                                   - 7-8: High Priority (Price alerts, position updates)
                                   - 9-10: Urgent (Stop loss hits, risk alerts)

        Returns:
        dict: JSON response containing:
            - status: success/error
            - message: Response message

        Message Formatting:
        - Bold: *text* or **text**
        - Italic: _text_ or __text__
        - Code: `text`
        - Pre-formatted: ```text```
        - Links: [text](url)
        - Line breaks: Use \\n in message string
        - Emojis: Standard Unicode emojis supported

        Example:
            # Basic notification
            result = api.telegram(
                username="john_trader",
                message="NIFTY crossed 24000! Consider taking profit on long positions."
            )

            # High priority alert
            result = api.telegram(
                username="john_trader",
                message="🚨 URGENT: Stop loss hit on BANKNIFTY position!",
                priority=10
            )

            # Multi-line trading summary
            result = api.telegram(
                username="john_trader",
                message="📊 Daily Trading Summary\\n"
                        "─────────────────────\\n"
                        "✅ Winning Trades: 8\\n"
                        "❌ Losing Trades: 2\\n"
                        "💰 Net P&L: +₹15,450\\n"
                        "📈 Win Rate: 80%\\n\\n"
                        "🎯 Great day! Keep it up!",
                priority=5
            )

            # Price alert
            result = api.telegram(
                username="trader_123",
                message="🔔 Price Alert: RELIANCE reached target price ₹2,850",
                priority=8
            )

            # Strategy signal
            result = api.telegram(
                username="algo_trader",
                message="📈 BUY Signal: RSI oversold on NIFTY 24000 CE\\n"
                        "Entry: ₹145.50\\n"
                        "Target: ₹165.00\\n"
                        "SL: ₹138.00",
                priority=9
            )

            # Risk management alert
            result = api.telegram(
                username="trader_123",
                message="⚠️ Risk Alert: Daily loss limit reached (-₹25,000)\\n"
                        "No new positions recommended.",
                priority=10
            )

            # Trade execution confirmation
            result = api.telegram(
                username="trader_123",
                message="✅ Order Executed\\n"
                        "Symbol: BANKNIFTY 48000 CE\\n"
                        "Action: BUY\\n"
                        "Qty: 30\\n"
                        "Price: ₹245.75\\n"
                        "Total: ₹7,372.50",
                priority=7
            )

        Common Use Cases:
        1. Price Alert Notifications
        2. Strategy Signal Alerts
        3. Risk Management Alerts
        4. Market Update Notifications
        5. Trade Execution Confirmation
        6. Technical Indicator Alerts
        7. Daily/Weekly Trading Summaries
        8. Position Updates

        Error Messages:
        - "Invalid or missing API key": API key is incorrect or expired
        - "User not found or not linked to Telegram": User hasn't linked account OR using wrong username
          (Use layr0_imc login username, not Telegram username. Link via /link command)
        - "Username and message are required": Missing required fields
        - "Failed to send notification": Bot is not running

        Rate Limiting:
        - Limit: 30 requests per minute per user
        - Response: 429 status code if limit exceeded

        Notes:
        - Username is your layr0_imc login username (case-sensitive), NOT @telegram_handle
        - User must link account via /link command in Telegram bot first
        - Bot must be running in layr0_imc settings to send messages
        - Messages are queued if delivery fails temporarily
        - Keep messages under 4096 characters (Telegram limit)
        """
        payload = {
            "apikey": self.api_key,
            "username": username,
            "message": message,
            "priority": priority
        }
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value

        return self._make_request("telegram/notify", payload)
