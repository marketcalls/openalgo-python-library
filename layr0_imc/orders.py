# -*- coding: utf-8 -*-
"""
layr0_imc REST API Documentation - Order Methods
    https://docs.layr0.org
"""

from .base import BaseAPI

class OrderAPI(BaseAPI):
    """
    Order management API methods for layr0_imc.
    Inherits from the BaseAPI class.
    """

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

    def positionsopen(self, *, strategy="Python", symbol, exchange, product, **kwargs):
        """
        Read the current strategy exposure for a specific symbol.

        This is a read-only, mode-checked reconciliation endpoint. It requires
        the caller to pass the current mode precondition fields through kwargs:
        expected_mode, expected_balance_type, expected_mode_version, and request_id.

        Parameters:
        - strategy (str, optional): The trading strategy name. Defaults to "Python".
        - symbol (str): Trading symbol. Required.
        - exchange (str): Exchange code. Required.
        - product (str): Product type. Required.
        - **kwargs: Mode precondition fields and optional future API extensions.

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
        return self._make_request("positionsopen", payload)
    
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
        Close all Strategy exposures for a given strategy.

        Parameters:
        - strategy (str, optional): The trading strategy name. Defaults to "Python".
        - **kwargs: Optional additional parameters for future API extensions.

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
        Cancel all orders for a given strategy.

        Parameters:
        - strategy (str, optional): The trading strategy name. Defaults to "Python".
        - **kwargs: Optional additional parameters for future API extensions.

        Returns:
        dict: JSON response from the API indicating the result of the cancel all orders action.
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
