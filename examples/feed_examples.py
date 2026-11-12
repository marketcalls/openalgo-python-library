"""
OpenAlgo WebSocket Feed Example (LTP)

Verbose levels:
  - verbose=False (default): Silent mode - no SDK output, only your prints
  - verbose=True or 1: Basic info - connection, auth, subscription status
  - verbose=2: Full debug - all market data updates from SDK
"""

import os
from openalgo import api
import time

# Initialize feed client with explicit parameters
# Set verbose=True for basic logging, verbose=2 for full debug
client = api(
    api_key=os.getenv("OPENALGO_API_KEY", "your_api_key_here"),  # Set OPENALGO_API_KEY in your environment
    host="http://127.0.0.1:5000",  # Replace with your API host
    ws_url="ws://127.0.0.1:8765",  # Explicit WebSocket URL (can be different from REST API host)
    verbose=False  # Silent mode (default) - set to True or 2 for debugging
)

# MCX instruments for testing
instruments_list = [
    {"exchange": "MCX", "symbol": "GOLDPETAL30MAY25FUT"},
    {"exchange": "MCX", "symbol": "GOLD05JUN25FUT"}
]

def on_data_received(data):
    print("LTP Update:")
    print(data)

# Connect and subscribe
client.connect()
client.subscribe_ltp(instruments_list, on_data_received=on_data_received)

# Poll LTP data a few times
for i in range(100):
    print(f"\nPoll {i+1}:")
    print(client.get_ltp())
    time.sleep(0.5)

# Cleanup
client.unsubscribe_ltp(instruments_list)
client.disconnect()
