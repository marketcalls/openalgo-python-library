# -*- coding: utf-8 -*-
"""
layr0_imc REST API Documentation - Account Methods
    https://docs.layr0.org
"""

from typing import List, Dict, Any, Optional, Union
from .base import BaseAPI

class AccountAPI(BaseAPI):
    """
    Account management API methods for layr0_imc.
    Inherits from the BaseAPI class.
    """

    def funds(self, **kwargs):
        """
        Get funds and margin details of the connected trading account.

        Parameters:
        - **kwargs: Optional additional parameters for future API extensions.

        Returns:
        dict: JSON response containing funds data with format:
            {
                "data": {
                    "availablecash": "amount",
                    "collateral": "amount",
                    "m2mrealized": "amount",
                    "m2munrealized": "amount",
                    "utiliseddebits": "amount"
                },
                "status": "success"
            }
        """
        payload = {
            "apikey": self.api_key
        }
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value
        return self._make_request("funds", payload)

    def orderbook(self, **kwargs):
        """
        Get orderbook details from the broker with basic orderbook statistics.

        Parameters:
        - **kwargs: Optional additional parameters for future API extensions.

        Returns:
        dict: JSON response containing orders data with format:
            {
                "data": {
                    "orders": [
                        {
                            "action": "BUY/SELL",
                            "exchange": "exchange_code",
                            "order_status": "status",
                            "orderid": "id",
                            "price": price_value,
                            "pricetype": "type",
                            "product": "product_type",
                            "quantity": quantity_value,
                            "symbol": "symbol_name",
                            "timestamp": "DD-MMM-YYYY HH:MM:SS",
                            "trigger_price": trigger_price_value
                        },
                        ...
                    ],
                    "statistics": {
                        "total_buy_orders": count,
                        "total_completed_orders": count,
                        "total_open_orders": count,
                        "total_rejected_orders": count,
                        "total_sell_orders": count
                    }
                },
                "status": "success"
            }
        """
        payload = {
            "apikey": self.api_key
        }
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value
        return self._make_request("orderbook", payload)

    def tradebook(self, **kwargs):
        """
        Get tradebook details from the broker.

        Parameters:
        - **kwargs: Optional additional parameters for future API extensions.

        Returns:
        dict: JSON response containing trades data with format:
            {
                "data": [
                    {
                        "action": "BUY/SELL",
                        "average_price": price_value,
                        "exchange": "exchange_code",
                        "orderid": "id",
                        "product": "product_type",
                        "quantity": quantity_value,
                        "symbol": "symbol_name",
                        "timestamp": "DD-MMM-YYYY HH:MM:SS",
                        "trade_value": value
                    },
                    ...
                ],
                "status": "success"
            }
        """
        payload = {
            "apikey": self.api_key
        }
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value
        return self._make_request("tradebook", payload)

    def positionbook(self, **kwargs):
        """
        Get positionbook details from the broker.

        Parameters:
        - **kwargs: Optional additional parameters for future API extensions.

        Returns:
        dict: JSON response containing positions data with format:
            {
                "data": [
                    {
                        "average_price": "price_value",
                        "exchange": "exchange_code",
                        "product": "product_type",
                        "quantity": quantity_value,
                        "symbol": "symbol_name"
                    },
                    ...
                ],
                "status": "success"
            }
        """
        payload = {
            "apikey": self.api_key
        }
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value
        return self._make_request("positionbook", payload)

    def holdings(self, **kwargs):
        """
        Get stock holdings details from the broker.

        Parameters:
        - **kwargs: Optional additional parameters for future API extensions.

        Returns:
        dict: JSON response containing holdings data with format:
            {
                "data": {
                    "holdings": [
                        {
                            "exchange": "exchange_code",
                            "pnl": pnl_value,
                            "pnlpercent": percentage_value,
                            "product": "product_type",
                            "quantity": quantity_value,
                            "symbol": "symbol_name"
                        },
                        ...
                    ],
                    "statistics": {
                        "totalholdingvalue": value,
                        "totalinvvalue": value,
                        "totalpnlpercentage": percentage,
                        "totalprofitandloss": value
                    }
                },
                "status": "success"
            }
        """
        payload = {
            "apikey": self.api_key
        }
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value
        return self._make_request("holdings", payload)

    def analyzerstatus(self, **kwargs):
        """
        Get analyzer status information.

        Parameters:
        - **kwargs: Optional additional parameters for future API extensions.

        Returns:
        dict: JSON response containing analyzer status with format:
            {
                "data": {
                    "analyze_mode": false,
                    "mode": "live",
                    "total_logs": 2
                },
                "status": "success"
            }
        """
        payload = {
            "apikey": self.api_key
        }
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value
        return self._make_request("analyzer", payload)

    def analyzertoggle(self, mode, **kwargs):
        """
        Toggle analyzer mode between analyze and live modes.

        Args:
            mode (bool): True for analyze mode (simulated), False for live mode
            **kwargs: Optional additional parameters for future API extensions.

        Returns:
        dict: JSON response containing analyzer toggle result with format:
            {
                "status": "success",
                "data": {
                    "mode": "live/analyze",
                    "analyze_mode": true/false,
                    "total_logs": 2,
                    "message": "Analyzer mode switched to live"
                }
            }
        """
        payload = {
            "apikey": self.api_key,
            "mode": mode
        }
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value
        return self._make_request("analyzer/toggle", payload)

    def margin(self, *, positions: List[Dict[str, Union[str, int, float]]], **kwargs) -> Dict[str, Any]:
        """
        Calculate margin requirements for a basket of positions.

        This function calculates the total margin required for one or multiple positions,
        taking into account broker-specific margin benefits for hedged positions.

        Parameters:
        - positions (list): List of position dictionaries (max: 50 positions). Required.
            Each position dictionary must contain:
                - symbol (str): Trading symbol (e.g., "SBIN", "NIFTY30DEC2526000CE"). Required.
                - exchange (str): Exchange code (NSE/BSE/NFO/BFO/CDS/MCX). Required.
                - action (str): BUY or SELL. Required.
                - product (str): Product type (CNC/MIS/NRML). Required.
                - pricetype (str): Price type (MARKET/LIMIT/SL/SL-M). Required.
                - quantity (str/int): Quantity to trade. Required.
                - price (str/float, optional): Price for LIMIT orders. Defaults to "0".
                - trigger_price (str/float, optional): Trigger price for SL/SL-M orders. Defaults to "0".

        Returns:
        dict: JSON response containing margin data with format:
            Success:
            {
                "status": "success",
                "data": {
                    "total_margin_required": 328482.00,
                    "span_margin": 258482.00,      # Available for most brokers
                    "exposure_margin": 70000.00    # Available for most brokers
                }
            }

            Error:
            {
                "status": "error",
                "message": "Error description"
            }

        Examples:
        >>> # Single position margin
        >>> client.margin(positions=[{
        ...     "symbol": "SBIN",
        ...     "exchange": "NSE",
        ...     "action": "BUY",
        ...     "product": "MIS",
        ...     "pricetype": "LIMIT",
        ...     "quantity": "10",
        ...     "price": "750.50"
        ... }])

        >>> # Multiple positions (basket margin)
        >>> client.margin(positions=[
        ...     {
        ...         "symbol": "NIFTY30DEC2526000CE",
        ...         "exchange": "NFO",
        ...         "action": "SELL",
        ...         "product": "NRML",
        ...         "pricetype": "LIMIT",
        ...         "quantity": "75",
        ...         "price": "150.75"
        ...     },
        ...     {
        ...         "symbol": "NIFTY30DEC2526000PE",
        ...         "exchange": "NFO",
        ...         "action": "SELL",
        ...         "product": "NRML",
        ...         "pricetype": "LIMIT",
        ...         "quantity": "75",
        ...         "price": "125.50"
        ...     }
        ... ])

        Notes:
        - Maximum 50 positions allowed per request
        - Basket margin calculation may provide margin benefit for hedged positions
        - Broker-specific behavior varies:
            * Angel One: Supports batch margin for up to 50 positions
            * Zerodha: Uses basket API for multiple positions
            * Dhan/Firstock/Kotak/Paytm: Single position only, aggregated for multiple
            * Groww: Basket margin only for FNO segment
            * 5paisa: Returns account-level margin
        - For MARKET orders, price can be "0"
        - For LIMIT orders, price is required
        - For SL/SL-M orders, trigger_price is required
        """
        # Validate positions parameter
        if not isinstance(positions, list):
            return {
                'status': 'error',
                'message': 'Positions must be an array',
                'error_type': 'validation_error'
            }

        if len(positions) == 0:
            return {
                'status': 'error',
                'message': 'Positions array cannot be empty',
                'error_type': 'validation_error'
            }

        if len(positions) > 50:
            return {
                'status': 'error',
                'message': 'Maximum 50 positions allowed',
                'error_type': 'validation_error'
            }

        # Process positions to ensure all numeric values are strings
        processed_positions = []
        for position in positions:
            processed_position = {}
            for key, value in position.items():
                # Convert numeric values to strings
                if key in ['quantity', 'price', 'trigger_price'] and value is not None:
                    processed_position[key] = str(value)
                else:
                    processed_position[key] = value

            # Set defaults for optional parameters if not provided
            if 'price' not in processed_position:
                processed_position['price'] = "0"
            if 'trigger_price' not in processed_position:
                processed_position['trigger_price'] = "0"

            processed_positions.append(processed_position)

        payload = {
            "apikey": self.api_key,
            "positions": processed_positions
        }
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value

        return self._make_request("margin", payload)
