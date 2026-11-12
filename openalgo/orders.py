# -*- coding: utf-8 -*-
"""
OpenAlgo REST API Documentation - Order Methods
    https://docs.openalgo.in
"""

import httpx
from .base import BaseAPI

class OrderAPI(BaseAPI):
    """
    Order management API methods for OpenAlgo.
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

    def placeorder(self, *, strategy="Python", symbol, action, exchange, price_type="MARKET", product="MIS", quantity=1, **kwargs):
        """
        Place an order with the given parameters. All parameters after 'strategy' must be named explicitly.

        Parameters:
        - strategy (str, optional): The trading strategy name. Defaults to "Python".
        - symbol (str): Trading symbol. Required.
        - action (str): BUY or SELL. Required.
        - exchange (str): Exchange code. Required.
        - price_type (str, optional): Type of price. Defaults to "MARKET".
        - product (str, optional): Product type. Defaults to "MIS".
        - quantity (int/str, optional): Quantity to trade. Defaults to 1.
        - **kwargs: Optional parameters like:
            - price (str): Required for LIMIT orders
            - trigger_price (str): Required for SL and SL-M orders
            - disclosed_quantity (str): Disclosed quantity
            - target (str): Target price
            - stoploss (str): Stoploss price
            - trailing_sl (str): Trailing stoploss points

        Returns:
        dict: JSON response from the API.
        """
        payload = {
            "apikey": self.api_key,
            "strategy": strategy,
            "symbol": symbol,
            "action": action,
            "exchange": exchange,
            "pricetype": price_type,
            "product": product,
            "quantity": str(quantity)
        }
        # Convert numeric values to strings
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = str(value)
        
        return self._make_request("placeorder", payload)
    
    def placesmartorder(self, *, strategy="Python", symbol, action, exchange, price_type="MARKET", product="MIS", quantity=1, position_size, **kwargs):
        """
        Place a smart order that considers the current position size.

        Parameters:
        - strategy (str, optional): The trading strategy name. Defaults to "Python".
        - symbol (str): Trading symbol. Required.
        - action (str): BUY or SELL. Required.
        - exchange (str): Exchange code. Required.
        - price_type (str, optional): Type of price. Defaults to "MARKET".
        - product (str, optional): Product type. Defaults to "MIS".
        - quantity (int/str, optional): Quantity to trade. Defaults to 1.
        - position_size (int/str): Required position size.
        - **kwargs: Optional parameters like:
            - price (str): Required for LIMIT orders
            - trigger_price (str): Required for SL and SL-M orders
            - disclosed_quantity (str): Disclosed quantity
            - target (str): Target price
            - stoploss (str): Stoploss price
            - trailing_sl (str): Trailing stoploss points

        Returns:
        dict: JSON response from the API.
        """
        payload = {
            "apikey": self.api_key,
            "strategy": strategy,
            "symbol": symbol,
            "action": action,
            "exchange": exchange,
            "pricetype": price_type,
            "product": product,
            "quantity": str(quantity),
            "position_size": str(position_size)
        }
        # Convert numeric values to strings
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = str(value)
        
        return self._make_request("placesmartorder", payload)

    def basketorder(self, *, strategy="Python", orders, **kwargs):
        """
        Place multiple orders simultaneously.

        Parameters:
        - strategy (str, optional): The trading strategy name. Defaults to "Python".
        - orders (list): List of order dictionaries. Each order dictionary should contain:
            - symbol (str): Trading symbol. Required.
            - exchange (str): Exchange code. Required.
            - action (str): BUY or SELL. Required.
            - quantity (str/int): Quantity to trade. Required.
            - pricetype (str, optional): Type of price. Defaults to "MARKET".
            - product (str, optional): Product type. Defaults to "MIS".
            Optional parameters:
            - price (str): Required for LIMIT orders
            - trigger_price (str): Required for SL and SL-M orders
            - disclosed_quantity (str): Disclosed quantity
        - **kwargs: Optional additional parameters for future API extensions.

        Returns:
        dict: JSON response containing results for each order with format:
            {
                "results": [
                    {
                        "orderid": "order_id",
                        "status": "success/error",
                        "symbol": "symbol_name"
                    },
                    ...
                ],
                "status": "success/error"
            }
        """
        # Ensure all numeric values are strings
        processed_orders = []
        for order in orders:
            processed_order = {}
            for key, value in order.items():
                processed_order[key] = str(value) if isinstance(value, (int, float)) else value
            processed_orders.append(processed_order)

        payload = {
            "apikey": self.api_key,
            "strategy": strategy,
            "orders": processed_orders
        }
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = str(value)

        return self._make_request("basketorder", payload)

    def splitorder(self, *, strategy="Python", symbol, action, exchange, quantity, splitsize, price_type="MARKET", product="MIS", **kwargs):
        """
        Split a large order into multiple smaller orders.

        Parameters:
        - strategy (str, optional): The trading strategy name. Defaults to "Python".
        - symbol (str): Trading symbol. Required.
        - action (str): BUY or SELL. Required.
        - exchange (str): Exchange code. Required.
        - quantity (int/str): Total quantity to trade. Required.
        - splitsize (int/str): Size of each split order. Required.
        - price_type (str, optional): Type of price. Defaults to "MARKET".
        - product (str, optional): Product type. Defaults to "MIS".
        - **kwargs: Optional parameters like:
            - price (str): Required for LIMIT orders
            - trigger_price (str): Required for SL and SL-M orders
            - disclosed_quantity (str): Disclosed quantity

        Returns:
        dict: JSON response containing results for each split order with format:
            {
                "results": [
                    {
                        "order_num": 1,
                        "orderid": "order_id",
                        "quantity": quantity,
                        "status": "success"
                    },
                    ...
                ],
                "split_size": splitsize,
                "status": "success",
                "total_quantity": total_quantity
            }
        """
        payload = {
            "apikey": self.api_key,
            "strategy": strategy,
            "symbol": symbol,
            "action": action,
            "exchange": exchange,
            "quantity": str(quantity),
            "splitsize": str(splitsize),
            "pricetype": price_type,
            "product": product
        }
        
        # Convert numeric values to strings
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = str(value)
        
        return self._make_request("splitorder", payload)

    def orderstatus(self, *, order_id, strategy="Python", **kwargs):
        """
        Get the current status of an order.

        Parameters:
        - order_id (str): The ID of the order to check. Required.
        - strategy (str, optional): The trading strategy name. Defaults to "Python".
        - **kwargs: Optional additional parameters for future API extensions.

        Returns:
        dict: JSON response containing order details with format:
            {
                "data": {
                    "action": "BUY/SELL",
                    "exchange": "exchange_code",
                    "order_status": "complete/pending/cancelled/etc",
                    "orderid": "order_id",
                    "price": price_value,
                    "pricetype": "MARKET/LIMIT/etc",
                    "product": "product_type",
                    "quantity": quantity_value,
                    "symbol": "symbol_name",
                    "timestamp": "DD-MMM-YYYY HH:MM:SS",
                    "trigger_price": trigger_price_value
                },
                "status": "success"
            }
        """
        payload = {
            "apikey": self.api_key,
            "strategy": strategy,
            "orderid": order_id
        }
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = str(value)
        return self._make_request("orderstatus", payload)

    def openposition(self, *, strategy="Python", symbol, exchange, product, **kwargs):
        """
        Get the current open position for a specific symbol.

        Parameters:
        - strategy (str, optional): The trading strategy name. Defaults to "Python".
        - symbol (str): Trading symbol. Required.
        - exchange (str): Exchange code. Required.
        - product (str): Product type. Required.
        - **kwargs: Optional additional parameters for future API extensions.

        Returns:
        dict: JSON response containing position details with format:
            {
                "quantity": position_quantity,
                "status": "success"
            }
        """
        payload = {
            "apikey": self.api_key,
            "strategy": strategy,
            "symbol": symbol,
            "exchange": exchange,
            "product": product
        }
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = str(value)
        return self._make_request("openposition", payload)
    
    def modifyorder(self, *, order_id, strategy="Python", symbol, action, exchange, price_type="LIMIT", product, quantity, price, disclosed_quantity="0", trigger_price="0", **kwargs):
        """
        Modify an existing order.

        Parameters:
        - order_id (str): The ID of the order to modify. Required.
        - strategy (str, optional): The trading strategy name. Defaults to "Python".
        - symbol (str): Trading symbol. Required.
        - action (str): BUY or SELL. Required.
        - exchange (str): Exchange code. Required.
        - price_type (str, optional): Type of price. Defaults to "LIMIT".
        - product (str): Product type. Required.
        - quantity (int/str): Quantity to trade. Required.
        - price (str): New price for the order. Required.
        - disclosed_quantity (str): Disclosed quantity. Required.
        - trigger_price (str): Trigger price. Required.
        - **kwargs: Optional parameters

        Returns:
        dict: JSON response from the API.
        """
        payload = {
            "apikey": self.api_key,
            "orderid": order_id,
            "strategy": strategy,
            "symbol": symbol,
            "action": action,
            "exchange": exchange,
            "pricetype": price_type,
            "product": product,
            "quantity": str(quantity),
            "price": str(price),
            "disclosed_quantity": str(disclosed_quantity),
            "trigger_price": str(trigger_price)
        }
        # Convert numeric values to strings
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = str(value)
        
        return self._make_request("modifyorder", payload)
    
    def cancelorder(self, *, order_id, strategy="Python", **kwargs):
        """
        Cancel an existing order.

        Parameters:
        - order_id (str): The ID of the order to cancel. Required.
        - strategy (str, optional): The trading strategy name. Defaults to "Python".
        - **kwargs: Optional additional parameters for future API extensions.

        Returns:
        dict: JSON response from the API.
        """
        payload = {
            "apikey": self.api_key,
            "orderid": order_id,
            "strategy": strategy
        }
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = str(value)
        return self._make_request("cancelorder", payload)
    
    def closeposition(self, *, strategy="Python", **kwargs):
        """
        Close ALL open positions in the account (square off everything).

        WARNING: this call is account-wide. It is NOT scoped to a strategy.
        The server squares off every open position of the account that owns the
        API key, across all symbols, exchanges and products, whichever strategy,
        application or manual trade opened it. In analyzer (sandbox) mode it
        closes every open sandbox position of the user instead.

        To exit one position only, send an opposite order for that symbol
        instead of calling this method.

        Parameters:
        - strategy (str, optional): A label only. The server requires the field
          and uses it for logging; it does not filter positions by it.
          Defaults to "Python".
        - **kwargs: Optional additional parameters for future API extensions.
          The endpoint accepts only the API key and the strategy label today,
          so there is no symbol, exchange or product filter to pass here.

        Returns:
        dict: JSON response from the API indicating the result of the close position action.
        """
        payload = {
            "apikey": self.api_key,
            "strategy": strategy
        }
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = str(value)
        return self._make_request("closeposition", payload)
    
    def cancelallorder(self, *, strategy="Python", **kwargs):
        """
        Cancel ALL open orders in the account.

        WARNING: this call is account-wide. It is NOT scoped to a strategy.
        The server cancels every open and trigger-pending order of the account
        that owns the API key, whichever strategy, application or manual action
        placed it. That includes pending stop-loss orders that protect other
        positions. In analyzer (sandbox) mode it cancels every open sandbox
        order of the user instead.

        To cancel one order only, use cancelorder(order_id=...).

        Parameters:
        - strategy (str, optional): A label only. The server requires the field
          and uses it for logging; it does not filter orders by it.
          Defaults to "Python".
        - **kwargs: Optional additional parameters for future API extensions.
          The endpoint accepts only the API key and the strategy label today,
          so there is no symbol, exchange or product filter to pass here.

        Returns:
        dict: JSON response from the API indicating the result of the cancel all orders action.
              On success it carries the lists "canceled_orders" and
              "failed_cancellations".
        """
        payload = {
            "apikey": self.api_key,
            "strategy": strategy
        }
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = str(value)
        return self._make_request("cancelallorder", payload)
