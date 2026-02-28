# -*- coding: utf-8 -*-
"""
OpenAlgo REST API Documentation - Data Methods
    https://docs.openalgo.in
"""

import httpx
import pandas as pd
from datetime import datetime
import time
from .base import BaseAPI

class DataAPI(BaseAPI):
    """
    Data API methods for OpenAlgo.
    Inherits from the BaseAPI class.
    """

    def _make_request(self, endpoint, payload):
        """Make HTTP request with proper error handling"""
        url = self.base_url + endpoint
        try:
            response = httpx.post(url, json=payload, headers=self.headers, timeout=self.timeout)
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

    def quotes(self, *, symbol, exchange, **kwargs):
        """
        Get real-time quotes for a symbol.

        Parameters:
        - symbol (str): Trading symbol. Required.
        - exchange (str): Exchange code. Required.
        - **kwargs: Optional additional parameters for future API extensions.

        Returns:
        dict: JSON response containing quote data including bid, ask, ltp, volume etc.
        """
        payload = {
            "apikey": self.api_key,
            "symbol": symbol,
            "exchange": exchange
        }
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value
        return self._make_request("quotes", payload)

    def multiquotes(self, *, symbols, **kwargs):
        """
        Get real-time quotes for multiple symbols in a single request.

        Parameters:
        - symbols (list): List of symbol-exchange pairs. Required.
            Each item should be a dict with 'symbol' and 'exchange' keys.
            Example: [{"symbol": "RELIANCE", "exchange": "NSE"}, {"symbol": "TCS", "exchange": "NSE"}]
        - **kwargs: Optional additional parameters for future API extensions.

        Returns:
        dict: JSON response containing quote data for all requested symbols.
            On success, returns a dict with 'status': 'success' and quote data for each symbol.
            On error, returns a dict with 'status': 'error' and an error message.

        Examples:
            # Get quotes for multiple NSE stocks
            result = api.multiquotes(symbols=[
                {"symbol": "RELIANCE", "exchange": "NSE"},
                {"symbol": "TCS", "exchange": "NSE"},
                {"symbol": "INFY", "exchange": "NSE"}
            ])

            # Get quotes across different exchanges
            result = api.multiquotes(symbols=[
                {"symbol": "RELIANCE", "exchange": "NSE"},
                {"symbol": "NIFTY", "exchange": "NSE_INDEX"},
                {"symbol": "GOLD", "exchange": "MCX"}
            ])

            if result['status'] == 'success':
                for quote in result.get('data', []):
                    print(f"{quote['symbol']}: LTP = {quote.get('ltp')}")
        """
        payload = {
            "apikey": self.api_key,
            "symbols": symbols
        }
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value
        return self._make_request("multiquotes", payload)

    def depth(self, *, symbol, exchange, **kwargs):
        """
        Get market depth (order book) for a symbol.

        Parameters:
        - symbol (str): Trading symbol. Required.
        - exchange (str): Exchange code. Required.
        - **kwargs: Optional additional parameters for future API extensions.

        Returns:
        dict: JSON response containing market depth data including top 5 bids/asks.
        """
        payload = {
            "apikey": self.api_key,
            "symbol": symbol,
            "exchange": exchange
        }
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value
        return self._make_request("depth", payload)

    def symbol(self, *, symbol, exchange, **kwargs):
        """
        Get symbol details from the API.

        Parameters:
        - symbol (str): Trading symbol. Required.
        - exchange (str): Exchange code. Required.
        - **kwargs: Optional additional parameters for future API extensions.

        Returns:
        dict: JSON response containing symbol details like token, lot size, tick size, etc.
        """
        payload = {
            "apikey": self.api_key,
            "symbol": symbol,
            "exchange": exchange
        }
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value
        return self._make_request("symbol", payload)
        
    def search(self, *, query, exchange=None, **kwargs):
        """
        Search for symbols across exchanges.

        Parameters:
        - query (str): Search query for symbol. Required.
        - exchange (str): Exchange filter. Optional.
            Supported exchanges: NSE, NFO, BSE, BFO, MCX, CDS, BCD, NCDEX, NSE_INDEX, BSE_INDEX, MCX_INDEX
        - **kwargs: Optional additional parameters for future API extensions.

        Returns:
        dict: JSON response containing matching symbols with details like:
            - symbol: Trading symbol
            - name: Company/instrument name
            - exchange: Exchange code
            - token: Unique instrument token
            - instrumenttype: Type of instrument
            - lotsize: Lot size
            - strike: Strike price (for options)
            - expiry: Expiry date (for derivatives)
        """
        payload = {
            "apikey": self.api_key,
            "query": query
        }
        if exchange:
            payload["exchange"] = exchange
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value

        return self._make_request("search", payload)
        
    def history(self, *, symbol, exchange, interval, start_date, end_date, source="api", **kwargs):
        """
        Get historical data for a symbol in pandas DataFrame format.

        Parameters:
        - symbol (str): Trading symbol. Required.
        - exchange (str): Exchange code. Required.
        - interval (str): Time interval for the data. Required.
                       Use interval() method to get supported intervals.
        - start_date (str): Start date in format 'YYYY-MM-DD'. Required.
        - end_date (str): End date in format 'YYYY-MM-DD'. Required.
        - source (str): Data source. 'api' fetches from broker API (default),
                       'db' fetches from OpenAlgo DuckDB/Historify database.
                       The db source stores 1m and D intervals physically and
                       computes all other intervals on-the-fly via SQL aggregation.
                       Custom intraday intervals (2m, 3m, 4m, 6m, 7m, 2h, 3h, 4h, etc.)
                       and daily-based intervals (W, M, Q, Y with multiples like 2W, 3M)
                       are only available with source='db'.
        - **kwargs: Optional additional parameters for future API extensions.

        Returns:
        pandas.DataFrame or dict: DataFrame with historical data if successful,
                                error dict if failed. DataFrame has timestamp as index.
                                For intraday data (non-daily timeframes), timestamps
                                are converted to IST. Daily data is already in IST.
        """
        payload = {
            "apikey": self.api_key,
            "symbol": symbol,
            "exchange": exchange,
            "interval": interval,
            "start_date": start_date,
            "end_date": end_date,
            "source": source
        }
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value

        result = self._make_request("history", payload)
        
        if result.get('status') == 'success' and 'data' in result:
            try:
                df = pd.DataFrame(result['data'])
                if df.empty:
                    return {
                        'status': 'error',
                        'message': 'No data available for the specified period',
                        'error_type': 'no_data'
                    }
                
                # Convert timestamp to datetime
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                
                # Convert to IST for intraday timeframes
                if interval not in ['D', 'W', 'M']:  # Not daily/weekly/monthly
                    df["timestamp"] = df["timestamp"].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
                
                # Set timestamp as index
                df.set_index('timestamp', inplace=True)
                
                # Sort index and remove duplicates
                df = df.sort_index()
                df = df[~df.index.duplicated(keep='first')]
                
                return df
            except Exception as e:
                return {
                    'status': 'error',
                    'message': f'Failed to process historical data: {str(e)}',
                    'error_type': 'processing_error',
                    'raw_data': result['data']
                }
        return result

    def intervals(self, **kwargs):
        """
        Get supported time intervals for historical data from the API.

        Parameters:
        - **kwargs: Optional additional parameters for future API extensions.

        Returns:
        dict: JSON response containing supported intervals categorized by type
              (seconds, minutes, hours, days, weeks, months)
        """
        payload = {
            "apikey": self.api_key
        }
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value
        return self._make_request("intervals", payload)
        
    def interval(self):
        """
        Legacy method. Use intervals() instead.
        Get supported time intervals for historical data.

        Returns:
        dict: JSON response containing supported intervals
        """
        return self.intervals()

    def expiry(self, *, symbol, exchange, instrumenttype, **kwargs):
        """
        Get expiry dates for a symbol.

        Parameters:
        - symbol (str): Trading symbol. Required.
        - exchange (str): Exchange code. Required.
        - instrumenttype (str): Instrument type (futures/options). Required.
        - **kwargs: Optional additional parameters for future API extensions.

        Returns:
        dict: JSON response containing expiry dates for the symbol
        """
        payload = {
            "apikey": self.api_key,
            "symbol": symbol,
            "exchange": exchange,
            "instrumenttype": instrumenttype
        }
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value
        return self._make_request("expiry", payload)

    def instruments(self, *, exchange=None):
        """
        Download all trading symbols and instruments with optional exchange filtering.

        This function retrieves instrument data from the OpenAlgo platform and returns
        it as a pandas DataFrame. When no exchange is specified, downloads ALL exchanges.

        Parameters:
        - exchange (str, optional): Exchange to filter instruments. If not specified, downloads ALL exchanges.
            Supported exchanges: NSE, BSE, NFO, BFO, BCD, CDS, MCX, NSE_INDEX, BSE_INDEX
            Default: None (downloads all exchanges)

        Returns:
        pandas.DataFrame or dict:
            - Success: DataFrame containing instrument data
            - Error: dict with error details

        DataFrame Columns:
        - symbol: OpenAlgo standard symbol
        - brsymbol: Broker-specific symbol
        - name: Instrument name
        - exchange: Exchange code
        - token: Instrument identifier
        - expiry: Expiry date (F&O only)
        - strike: Strike price (options only)
        - lotsize: Lot size
        - instrumenttype: Instrument type (EQ, FUT, CE, PE, etc.)
        - tick_size: Minimum price movement

        Examples:
            # Download ALL instruments (all exchanges)
            all_df = api.instruments()
            print(f"Total instruments across all exchanges: {len(all_df)}")

            # Download NSE equities only
            nse_df = api.instruments(exchange="NSE")
            print(f"Total NSE instruments: {len(nse_df)}")

            # Download NFO derivatives
            nfo_df = api.instruments(exchange="NFO")

            # Filter and analyze instruments
            df = api.instruments(exchange="NSE")
            if isinstance(df, pd.DataFrame):
                # Filter by instrument type
                equities = df[df['instrumenttype'] == 'EQ']

                # Search for specific symbol
                reliance = df[df['symbol'] == 'RELIANCE']

                # Export to CSV
                df.to_csv('nse_instruments.csv', index=False)

                # Get unique symbols
                symbols = df['symbol'].unique()

            # Download all and filter by exchange in pandas
            all_instruments = api.instruments()
            nse_only = all_instruments[all_instruments['exchange'] == 'NSE']
            nfo_only = all_instruments[all_instruments['exchange'] == 'NFO']

        Notes:
        - Without exchange parameter: Downloads ALL exchanges (NSE, BSE, NFO, BFO, BCD, CDS, MCX, NSE_INDEX, BSE_INDEX)
        - With exchange parameter: Downloads only specified exchange
        - Rate limit: 50 requests/second
        - Data updates when master contracts are downloaded
        - Returns pandas DataFrame for easy filtering, searching, and analysis
        """
        # If no exchange specified, fetch all exchanges and combine
        if exchange is None:
            all_exchanges = ['NSE', 'BSE', 'NFO', 'BFO', 'MCX', 'CDS', 'BCD', 'NSE_INDEX', 'BSE_INDEX']
            all_dfs = []

            for exch in all_exchanges:
                params = {
                    "apikey": self.api_key,
                    "exchange": exch
                }

                url = f"{self.base_url}instruments"
                try:
                    response = httpx.get(url, params=params, timeout=self.timeout)
                    result = self._handle_response(response)

                    if result.get('status') == 'success' and 'data' in result:
                        try:
                            df = pd.DataFrame(result['data'])
                            if not df.empty:
                                all_dfs.append(df)
                        except Exception:
                            pass  # Skip exchanges that fail
                except Exception:
                    pass  # Skip exchanges that fail

            # Combine all dataframes
            if all_dfs:
                combined_df = pd.concat(all_dfs, ignore_index=True)
                return combined_df
            else:
                return {
                    'status': 'error',
                    'message': 'Failed to fetch instruments from any exchange',
                    'error_type': 'no_data'
                }

        # Fetch single exchange
        params = {
            "apikey": self.api_key,
            "exchange": exchange
        }

        # Make GET request
        url = f"{self.base_url}instruments"
        try:
            # Don't include Content-Type header for GET requests
            response = httpx.get(url, params=params, timeout=self.timeout)

            # Handle JSON response
            result = self._handle_response(response)

            # Convert to DataFrame if successful
            if result.get('status') == 'success' and 'data' in result:
                try:
                    df = pd.DataFrame(result['data'])
                    if df.empty:
                        return {
                            'status': 'error',
                            'message': 'No instruments available for the specified exchange',
                            'error_type': 'no_data'
                        }
                    return df
                except Exception as e:
                    return {
                        'status': 'error',
                        'message': f'Failed to process instruments data: {str(e)}',
                        'error_type': 'processing_error',
                        'raw_data': result['data']
                    }
            return result

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

    def syntheticfuture(self, *, underlying, exchange, expiry_date, **kwargs):
        """
        Calculate synthetic futures price using ATM Call and Put options.

        A synthetic future replicates the payoff of a futures contract by combining
        options at the same strike price. This function automatically finds the ATM
        strike and calculates the synthetic future price.

        Formula: Synthetic Future Price = Strike Price + Call Premium - Put Premium

        Parameters:
        - underlying (str): Underlying symbol (e.g., NIFTY, BANKNIFTY, RELIANCE). Required.
        - exchange (str): Exchange code. Required.
            Supported exchanges: NSE_INDEX, BSE_INDEX, NSE, BSE
            - NSE_INDEX: For NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY
            - BSE_INDEX: For SENSEX, BANKEX, SENSEX50
            - NSE: For equity stocks (RELIANCE, TCS, INFY, etc.)
            - BSE: For equity stocks
        - expiry_date (str): Expiry date in DDMMMYY format (e.g., 28NOV25). Required.

        Returns:
        dict: JSON response containing:
            - status: "success" or "error"
            - underlying: Underlying symbol from request
            - underlying_ltp: Current Last Traded Price of underlying
            - expiry: Expiry date in DDMMMYY format
            - atm_strike: ATM strike price used for calculation
            - synthetic_future_price: Calculated synthetic futures price

        Examples:
            # NIFTY Index
            result = api.syntheticfuture(
                underlying="NIFTY",
                exchange="NSE_INDEX",
                expiry_date="28NOV25"
            )
            if result['status'] == 'success':
                print(f"NIFTY Synthetic Future: {result['synthetic_future_price']}")
                print(f"ATM Strike: {result['atm_strike']}")
                print(f"Underlying LTP: {result['underlying_ltp']}")

            # BANKNIFTY Index
            result = api.syntheticfuture(
                underlying="BANKNIFTY",
                exchange="NSE_INDEX",
                expiry_date="27NOV25"
            )

            # RELIANCE Equity
            result = api.syntheticfuture(
                underlying="RELIANCE",
                exchange="NSE",
                expiry_date="26DEC25"
            )

            # Compare with actual futures
            synthetic = api.syntheticfuture(
                underlying="NIFTY",
                exchange="NSE_INDEX",
                expiry_date="28NOV25"
            )
            if synthetic['status'] == 'success':
                spot = synthetic['underlying_ltp']
                future_synthetic = synthetic['synthetic_future_price']
                basis = future_synthetic - spot
                print(f"Spot: {spot}, Synthetic Future: {future_synthetic}, Basis: {basis}")

        Notes:
        - This API only calculates and returns data; it does not place any trades
        - Uses current market prices for real-time calculations
        - Automatically determines ATM strike from available strikes in database
        - Index exchanges (NSE_INDEX, BSE_INDEX) are automatically mapped to options exchanges (NFO, BFO)
        - Useful for arbitrage opportunities between synthetic and actual futures
        """
        payload = {
            "apikey": self.api_key,
            "underlying": underlying,
            "exchange": exchange,
            "expiry_date": expiry_date
        }
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value
        return self._make_request("syntheticfuture", payload)
