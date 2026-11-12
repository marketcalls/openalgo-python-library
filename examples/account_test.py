# -*- coding: utf-8 -*-
"""
OpenAlgo Account API Test Examples
"""

import os
from openalgo import api

def main():
    # Initialize API with test credentials
    client = api(
        api_key=os.getenv("OPENALGO_API_KEY", "your_api_key_here"),
        host="http://127.0.0.1:5000"
    )

    # Test 1: Get Funds
    print("\nTest 1: Funds")
    funds_result = client.funds()
    print("Funds Result:", funds_result)

    # Test 2: Get Orderbook
    print("\nTest 2: Orderbook")
    orderbook_result = client.orderbook()
    print("Orderbook Result:", orderbook_result)
    if orderbook_result.get('status') == 'success':
        stats = orderbook_result['data']['statistics']
        print("\nOrder Statistics:")
        print(f"Total Buy Orders: {stats['total_buy_orders']}")
        print(f"Total Sell Orders: {stats['total_sell_orders']}")
        print(f"Total Completed Orders: {stats['total_completed_orders']}")
        print(f"Total Open Orders: {stats['total_open_orders']}")
        print(f"Total Rejected Orders: {stats['total_rejected_orders']}")

    # Test 3: Get Tradebook
    print("\nTest 3: Tradebook")
    tradebook_result = client.tradebook()
    print("Tradebook Result:", tradebook_result)
    if tradebook_result.get('status') == 'success':
        trades = tradebook_result['data']
        print(f"\nTotal Trades: {len(trades)}")
        if trades:
            print("Latest Trade:")
            latest = trades[0]
            print(f"Symbol: {latest['symbol']}")
            print(f"Action: {latest['action']}")
            print(f"Quantity: {latest['quantity']}")
            print(f"Average Price: {latest['average_price']}")
            print(f"Trade Value: {latest['trade_value']}")

    # Test 4: Get Positionbook
    print("\nTest 4: Positionbook")
    positionbook_result = client.positionbook()
    print("Positionbook Result:", positionbook_result)
    if positionbook_result.get('status') == 'success':
        positions = positionbook_result['data']
        print(f"\nTotal Positions: {len(positions)}")
        for pos in positions:
            print(f"\nSymbol: {pos['symbol']}")
            print(f"Quantity: {pos['quantity']}")
            print(f"Average Price: {pos['average_price']}")
            print(f"Product: {pos['product']}")

    # Test 5: Get Holdings
    print("\nTest 5: Holdings")
    holdings_result = client.holdings()
    print("Holdings Result:", holdings_result)
    if holdings_result.get('status') == 'success':
        data = holdings_result['data']
        holdings = data['holdings']
        stats = data['statistics']
        print("\nHoldings Statistics:")
        print(f"Total Holdings Value: {stats['totalholdingvalue']}")
        print(f"Total Investment Value: {stats['totalinvvalue']}")
        print(f"Total P&L: {stats['totalprofitandloss']}")
        print(f"Total P&L %: {stats['totalpnlpercentage']}")
        print(f"\nTotal Holdings: {len(holdings)}")
        for holding in holdings:
            print(f"\nSymbol: {holding['symbol']}")
            print(f"Quantity: {holding['quantity']}")
            print(f"P&L: {holding['pnl']}")
            print(f"P&L %: {holding['pnlpercent']}")

if __name__ == "__main__":
    main()
