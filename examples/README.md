# OpenAlgo Examples

This directory contains example scripts demonstrating the usage of OpenAlgo's Python API.

## Setup

Before running the examples, you need to install the openalgo package in development mode. From the root directory of the project:

```bash
# Activate your virtual environment if you haven't already
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows

# Install the package in development mode
pip install -e .
```

## Running Examples

After installation, you can run any of the example scripts:

```bash
# Run order management examples
python examples/order_examples.py

# Run market data examples
python examples/data_examples.py
```

## Available Examples

1. `order_examples.py` - Demonstrates order management functionality:
   - Place regular orders
   - Place smart orders
   - Modify orders
   - Cancel orders
   - Close positions
   - Cancel all orders

2. `data_examples.py` - Demonstrates market data functionality:
   - Get real-time quotes
   - Get market depth
   - Get historical data
   - Get supported intervals
   - Combined market data example

3. WebSocket Feed Examples:
   - `feed_examples.py` - Demonstrates WebSocket LTP feed:
     - Connect to WebSocket server
     - Subscribe to LTP updates
     - Real-time callbacks for LTP data
     - Polling for latest LTP data
     - Unsubscribe and disconnect

   - `quote_example.py` - Demonstrates WebSocket quote feed:
     - Connect to WebSocket server
     - Subscribe to quote updates with OHLC data
     - Real-time callbacks for quote data
     - Polling for latest quote data
     - Unsubscribe and disconnect

   - `depth_example.py` - Demonstrates WebSocket market depth feed:
     - Connect to WebSocket server
     - Subscribe to market depth updates
     - Real-time callbacks for order book data
     - Polling for latest market depth data
     - Unsubscribe and disconnect

## API Key

No API key is stored in this directory. The order, data, account and WebSocket feed
examples (and `test.ipynb`) read the key from the `OPENALGO_API_KEY` environment variable
and fall back to a placeholder:
```python
api_key = os.getenv("OPENALGO_API_KEY", "your_api_key_here")
host = "http://127.0.0.1:5000"
```

Set the variable to your own key before running them:

```bash
export OPENALGO_API_KEY="your_api_key_here"   # On Unix/macOS
# or
$env:OPENALGO_API_KEY = "your_api_key_here"   # On Windows PowerShell
```

The other examples carry the `your_api_key_here` placeholder in the source: replace it
locally, and do not commit a real key.

Make sure your OpenAlgo server is running at the specified host before running the examples.
