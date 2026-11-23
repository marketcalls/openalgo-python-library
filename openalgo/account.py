# -*- coding: utf-8 -*-
"""
OpenAlgo REST API Documentation - Account Methods
    https://docs.openalgo.in
"""

import httpx
from typing import List, Dict, Any, Optional, Union
from ._payload import plain_value
from .base import BaseAPI

class AccountAPI(BaseAPI):
    """
    Account management API methods for OpenAlgo.
    Inherits from the BaseAPI class.
    """

    def _make_request(self, endpoint, payload):
        """Make HTTP request with proper error handling"""
        url = self.base_url + endpoint
        try:
            response = self.client.post(url, json=payload, headers=self.headers, timeout=self.timeout)
            return self._handle_response(response)
        except httpx.TimeoutException:
            return {
                'status': 'error',
                'message': 'Request timed out. The server took too long to respond.',
                'error_type': 'timeout_error'
            }
        except httpx.ConnectError:
            return {
                'status': 'error',
                'message': 'Failed to connect to the server. Please check if the server is running.',
                'error_type': 'connection_error'
            }
        except httpx.HTTPError as e:
            return {
                'status': 'error',
                'message': f'HTTP error occurred: {str(e)}',
                'error_type': 'http_error'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'An unexpected error occurred: {str(e)}',
                'error_type': 'unknown_error'
            }
    
    def _handle_response(self, response):
        """Helper method to handle API responses"""
        try:
            if response.status_code != 200:
                return {
                    'status': 'error',
                    'message': f'HTTP {response.status_code}: {response.text}',
                    'code': response.status_code,
                    'error_type': 'http_error'
                }
            
            data = response.json()
            if data.get('status') == 'error':
                return {
                    'status': 'error',
                    'message': data.get('message', 'Unknown error'),
                    'code': response.status_code,
                    'error_type': 'api_error'
                }
            return data
            
        except ValueError:
            return {
                'status': 'error',
                'message': 'Invalid JSON response from server',
                'raw_response': response.text,
                'error_type': 'json_error'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'error_type': 'unknown_error'
            }

    def ping(self, **kwargs):
        """
        Check that the API key is valid and see which broker OpenAlgo is logged in to.

        Parameters:
        - **kwargs: Optional additional parameters for future API extensions.

        Returns:
        dict: JSON response with format:
            {
                "data": {
                    "broker": "broker_name",
                    "message": "pong"
                },
                "status": "success"
            }
        An unknown API key, or a broker session that was logged out, is answered
        with HTTP 403 ("Invalid openalgo apikey").
        """
        payload = {
            "apikey": self.api_key
        }
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value
        return self._make_request("ping", payload)

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

    def charges(self, *, symbol, exchange, action, product, quantity, price):
        """
        Brokerage and statutory charges for one order, before it is placed.

        OpenAlgo works the figures out from the brokerage plan saved on its
        /sandbox/commissions page and the published statutory rates. No broker
        is called, so the answer is the same in live and analyzer mode. They are
        the server's estimate of what the order costs, separate from the backtest
        engine's own cost model in openalgo.backtest.costs. One call prices one
        order: a round trip is two calls, one per side.

        Parameters:
        - symbol (str): Trading symbol in OpenAlgo format, e.g. "RELIANCE". Required.
        - exchange (str): Exchange code: NSE, BSE, NFO, BFO, CDS, BCD, MCX, NCDEX,
                          NCO or CRYPTO. Required.
        - action (str): BUY or SELL. Required.
        - product (str): MIS, CNC or NRML. On NSE and BSE the product decides the
                         segment: MIS is intraday, CNC is delivery. Required.
        - quantity (int): Order quantity in units, never lots: one NIFTY lot of
                          65 is 65. Required.
        - price (float): Price per unit the order would trade at, above zero. For
                         an option, the premium; for a MARKET order, the last
                         traded price. Required.

        Returns:
        dict: JSON response with format:
            {
                "status": "success",
                "data": {
                    "symbol": "RELIANCE",
                    "exchange": "NSE",
                    "product": "CNC",
                    "action": "BUY",
                    "quantity": 50,
                    "price": 1350.0,
                    "segment": "Equity Delivery",
                    "trade_value": 67500.0,
                    "charges": {
                        "total": 104.13,
                        "brokerage": 20.0,
                        "taxes": {"gst": 3.99, "stt": 68.0, "stamp_duty": 10.0},
                        "other_charges": {
                            "transaction": 2.0,
                            "clearing": 0.0,
                            "ipft": 0.07,
                            "sebi_turnover": 0.07
                        }
                    }
                }
            }

        Data fields:
        - symbol, exchange, product, action, quantity, price: the order as it was priced.
        - segment: the rate-card segment, e.g. "Equity Delivery" or "F&O Options".
        - trade_value: price times quantity, times the contract multiplier where
          the symbol has one: in rupees on the Indian exchanges, in the
          instrument's quote currency on CRYPTO.
        - charges: every line in rupees on the Indian exchanges. On CRYPTO every
          line is zero until a venue fee is entered on /sandbox/commissions, and
          is then that fee, charged as entered.
            - total: everything the order pays, the sum of every line below.
            - brokerage: brokerage under the saved plan for the segment.
            - taxes.gst: GST on brokerage, exchange transaction charges and SEBI fees.
            - taxes.stt: Securities Transaction Tax, or Commodities Transaction Tax
              on a commodity exchange.
            - taxes.stamp_duty: stamp duty, charged on a buy only.
            - other_charges.transaction: exchange transaction charges.
            - other_charges.clearing: clearing charges, always 0.
            - other_charges.ipft: Investor Protection Fund Trust levy, where the
              exchange has one.
            - other_charges.sebi_turnover: SEBI turnover fee.

        STT and stamp duty are rounded to the nearest rupee, as on a contract note
        for that one order (CTT on a commodity exchange is kept to the paisa);
        every other line is kept to the paisa. Not included:
        depository (DP) charges on a delivery sell, and STT on an option that is
        exercised at expiry.

        Errors come back as the server's JSON body with the HTTP status in "code":
        - 400: a field is missing or not an allowed value (the message names the
          field), the symbol is not in the master contract, or the exchange is an
          index exchange.
        - 403: "Invalid openalgo apikey".

        Example:
            response = client.charges(
                symbol="RELIANCE",
                exchange="NSE",
                action="BUY",
                product="CNC",
                quantity=50,
                price=1350
            )
        """
        payload = {
            "apikey": self.api_key,
            "symbol": symbol,
            "exchange": exchange,
            "action": action,
            "product": product,
            "quantity": plain_value(quantity),
            "price": plain_value(price)
        }
        return self._post("charges", payload)
