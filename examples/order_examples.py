"""
OpenAlgo Order Management Examples
"""

import os
from openalgo import api
import json

def print_response(title, response):
    """Helper function to print responses in a readable format"""
    print(f"\n{title}:")
    print(json.dumps(response, indent=2))

# Initialize the API client
client = api(
    api_key=os.getenv("OPENALGO_API_KEY", "your_api_key_here"),
    host="http://127.0.0.1:5000"
)

def place_order_example():
    """Example of placing regular orders"""
    try:
        # Market order
        response = client.placeorder(
            symbol="RELIANCE",
            action="BUY",
            exchange="NSE",
            price_type="MARKET",
            product="MIS",
            quantity=1
        )
        print_response("Market Buy Order", response)

        # Limit order
        response = client.placeorder(
            symbol="TATAMOTORS",
            action="SELL",
            exchange="NSE",
            price_type="LIMIT",
            product="MIS",
            quantity=1,
            price="800.00"  # Price as string
        )
        print_response("Limit Sell Order", response)

    except Exception as e:
        print(f"Error in place_order_example: {e}")

def place_smart_order_example():
    """Example of placing smart orders"""
    try:
        # Smart order with position size
        response = client.placesmartorder(
            symbol="TATAMOTORS",
            action="SELL",
            exchange="NSE",
            price_type="MARKET",
            product="MIS",
            quantity=1,
            position_size=5
        )
        print_response("Smart Order", response)

    except Exception as e:
        print(f"Error in place_smart_order_example: {e}")

def modify_order_example():
    """Example of modifying orders"""
    try:
        # Modify a limit order
        response = client.modifyorder(
            order_id="12345678",
            symbol="INFY",
            action="SELL",
            exchange="NSE",
            price_type="LIMIT",
            product="CNC",
            quantity=2,
            price="1500.00",  # Price as string
            disclosed_quantity="0",  # Required parameter
            trigger_price="0"   # Required parameter
        )
        print_response("Modify Order", response)

    except Exception as e:
        print(f"Error in modify_order_example: {e}")

def cancel_order_example():
    """Example of canceling specific orders"""
    try:
        response = client.cancelorder(
            order_id="12345678",
            strategy="Python"
        )
        print_response("Cancel Order", response)

    except Exception as e:
        print(f"Error in cancel_order_example: {e}")

def cancel_all_orders_example():
    """Example of canceling all orders"""
    try:
        response = client.cancelallorder(
            strategy="Python"
        )
        print_response("Cancel All Orders", response)

    except Exception as e:
        print(f"Error in cancel_all_orders_example: {e}")

def close_position_example():
    """Example of closing positions"""
    try:
        response = client.closeposition(
            strategy="Python"
        )
        print_response("Close Position", response)

    except Exception as e:
        print(f"Error in close_position_example: {e}")

def full_order_workflow_example():
    """Example demonstrating a complete order workflow"""
    try:
        print("\nComplete Order Workflow Example")
        print("=" * 50)

        # 1. Place a market order
        print("\nStep 1: Placing market order")
        order_response = client.placeorder(
            symbol="RELIANCE",
            action="BUY",
            exchange="NSE",
            price_type="MARKET",
            product="MIS",
            quantity=1
        )
        print_response("Market Order Response", order_response)

        # 2. Place a smart order
        print("\nStep 2: Placing smart order")
        smart_response = client.placesmartorder(
            symbol="TATAMOTORS",
            action="BUY",
            exchange="NSE",
            price_type="MARKET",
            product="MIS",
            quantity=1,
            position_size=5
        )
        print_response("Smart Order Response", smart_response)

        # 3. Cancel all orders
        print("\nStep 3: Canceling all orders")
        cancel_response = client.cancelallorder(strategy="Python")
        print_response("Cancel All Orders Response", cancel_response)

        # 4. Close all positions
        print("\nStep 4: Closing all positions")
        close_response = client.closeposition(strategy="Python")
        print_response("Close Positions Response", close_response)

    except Exception as e:
        print(f"Error in full_order_workflow_example: {e}")

if __name__ == "__main__":
    print("Running Order Management Examples...")
    print("=" * 50)
    
    print("\nTesting Individual Order Functions:")
    place_order_example()
    place_smart_order_example()
    modify_order_example()
    cancel_order_example()
    cancel_all_orders_example()
    close_position_example()
    
    print("\nTesting Complete Order Workflow:")
    full_order_workflow_example()
