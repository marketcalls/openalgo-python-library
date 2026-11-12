"""
Configuration for playground testing
"""

import os

# OpenAlgo API Configuration
# Update these values with your actual settings

API_KEY = os.getenv('OPENALGO_API_KEY', 'your_api_key_here')
HOST = 'http://127.0.0.1:5001'

# Default test parameters
DEFAULT_SYMBOL = "RELIANCE"
DEFAULT_EXCHANGE = "NSE"
DEFAULT_INTERVAL = "D"
DEFAULT_START_DATE = "2024-01-01"
DEFAULT_END_DATE = "2025-01-14"
DEFAULT_ADX_PERIOD = 14

# Test symbols for comparison
TEST_SYMBOLS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]