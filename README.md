# OpenAlgo - Python API Client for Automated Trading

## Installation

```bash
pip install openalgo
```

## Complete Code Examples

### Order Management Example
```python
from openalgo import api

# Initialize the API client
client = api(
    api_key="your_api_key",
    host="http://127.0.0.1:5000"
)

# Place a market order
market_order = client.placeorder(
    symbol="RELIANCE",  # Symbol without -EQ suffix
    action="BUY",
    exchange="NSE",
    price_type="MARKET",
    product="MIS",
    quantity="1"  # Numeric values as strings
)
print("Market Order:", market_order)

# Place a limit order
limit_order = client.placeorder(
    symbol="TATAMOTORS",
    action="SELL",
    exchange="NSE",
    price_type="LIMIT",
    product="MIS",
    quantity="1",
    price="800.00"  # Price as string
)
print("Limit Order:", limit_order)

# Place a smart order
smart_order = client.placesmartorder(
    symbol="TATAMOTORS",
    action="SELL",
    exchange="NSE",
    price_type="MARKET",
    product="MIS",
    quantity="1",
    position_size="5"
)
print("Smart Order:", smart_order)

# Modify an order
modify_order = client.modifyorder(
    order_id="12345678",
    symbol="INFY",
    action="SELL",
    exchange="NSE",
    price_type="LIMIT",
    product="CNC",
    quantity="2",
    price="1500.00",
    disclosed_quantity="0",  # Required parameter
    trigger_price="0"       # Required parameter
)
print("Modify Order:", modify_order)

# Cancel an order
cancel_order = client.cancelorder(
    order_id="12345678",
    strategy="Python"
)
print("Cancel Order:", cancel_order)

# Cancel all orders
cancel_all = client.cancelallorder(
    strategy="Python"
)
print("Cancel All Orders:", cancel_all)

# Close all positions
close_positions = client.closeposition(
    strategy="Python"
)
print("Close Positions:", close_positions)
```

### Market Data Example
```python
from openalgo import api
import pandas as pd

# Initialize the API client
client = api(
    api_key="your_api_key",
    host="http://127.0.0.1:5000"
)

def print_dataframe(title, df):
    print(f"\n{title}:")
    if isinstance(df, pd.DataFrame):
        print("\nFirst few rows:")
        print(df.head().to_string())
        print("\nDataFrame Info:")
        print(df.info())
    else:
        print(df)

# Get real-time quotes
quotes = client.quotes(
    symbol="RELIANCE",
    exchange="NSE"
)
print("\nReal-time Quotes:", quotes)

# Get market depth
depth = client.depth(
    symbol="SBIN",
    exchange="NSE"
)
print("\nMarket Depth:", depth)

# Get supported intervals
intervals = client.interval()
print("\nSupported Intervals:", intervals)

# Get 1-minute data
minute_data = client.history(
    symbol="SBIN",
    exchange="NSE",
    interval="1m",
    start_date="2024-12-01",
    end_date="2024-12-31"
)
print_dataframe("1-Minute Data", minute_data)

# Get 5-minute data
five_min_data = client.history(
    symbol="SBIN",
    exchange="NSE",
    interval="5m",
    start_date="2024-12-01",
    end_date="2024-12-31"
)
print_dataframe("5-Minute Data", five_min_data)

# Get hourly data
hourly_data = client.history(
    symbol="SBIN",
    exchange="NSE",
    interval="1h",
    start_date="2024-12-01",
    end_date="2024-12-31"
)
print_dataframe("Hourly Data", hourly_data)

# Get daily data
daily_data = client.history(
    symbol="SBIN",
    exchange="NSE",
    interval="D",
    start_date="2024-12-01",
    end_date="2024-12-31"
)
print_dataframe("Daily Data", daily_data)

# Example of data analysis
if isinstance(daily_data, pd.DataFrame):
    # Calculate daily returns
    daily_data['returns'] = daily_data['close'].pct_change()
    
    # Calculate simple moving averages
    daily_data['SMA_5'] = daily_data['close'].rolling(window=5).mean()
    daily_data['SMA_20'] = daily_data['close'].rolling(window=20).mean()
    
    print("\nDaily Data with Indicators:")
    print(daily_data.tail().to_string())
    
    print("\nSummary Statistics:")
    print(daily_data.describe().to_string())
```

### Example Output

```python
# Intraday data (1-minute):
#                                     close    high     low    open  volume
# timestamp                                                        
# 2024-12-02 09:15:00+05:30  836.40  841.10  836.00  838.95  121671
# 2024-12-02 09:16:00+05:30  835.75  836.60  835.10  836.15   40517

# Daily data:
#                      close    high    low    open    volume
# timestamp                                          
# 2024-12-02  836.40  842.00  832.7  838.95   6651119
# 2024-12-03  853.95  856.60  836.9  838.00  12186182
```

## Important Notes

1. Order Parameters:
   - All numeric values (quantity, price, etc.) must be passed as strings
   - Symbols should be used without -EQ suffix
   - Modify orders require disclosed_quantity and trigger_price parameters

2. Historical Data:
   - Intraday data (seconds, minutes, hours) includes IST timezone (+05:30)
   - Daily data uses clean date format without timezone
   - OHLCV columns with proper data types
   - Timestamps are sorted chronologically

For more detailed usage and additional methods, refer to the [OpenAlgo REST API Documentation](https://docs.openalgo.in/api-documentation/v1)
