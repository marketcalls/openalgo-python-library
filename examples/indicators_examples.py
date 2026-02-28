"""
OpenAlgo Technical Indicators Examples

This example demonstrates how to use the OpenAlgo technical indicators library
with both real market data and sample data.
"""

import numpy as np
import pandas as pd
from openalgo import api, ta

def basic_indicators_example():
    """Basic usage of technical indicators with sample data"""
    print("Basic Technical Indicators Example")
    print("=" * 50)
    
    # Create sample OHLCV data
    np.random.seed(42)
    n = 100
    
    # Generate realistic price data as numpy arrays
    close_np = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high_np = close_np + np.random.rand(n) * 2
    low_np = close_np - np.random.rand(n) * 2
    volume_np = np.random.randint(1000, 10000, n)
    
    # Create pandas Series with datetime index for comparison
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    close_pd = pd.Series(close_np, index=dates)
    high_pd = pd.Series(high_np, index=dates)
    low_pd = pd.Series(low_np, index=dates)
    volume_pd = pd.Series(volume_np, index=dates)
    
    print(f"Sample data length: {len(close_np)} periods")
    print(f"Price range: {low_np.min():.2f} - {high_np.max():.2f}")
    
    # Demonstrate format preservation
    print("\n--- FORMAT PRESERVATION DEMO ---")
    
    # NumPy input → NumPy output
    sma_np = ta.sma(close_np, 20)
    print(f"NumPy input → Output type: {type(sma_np)}")
    
    # Pandas input → Pandas output (with preserved index)
    sma_pd = ta.sma(close_pd, 20)
    print(f"Pandas input → Output type: {type(sma_pd)}")
    print(f"Index preserved: {sma_pd.index[:3]}")
    
    # Trend Indicators
    print("\n--- TREND INDICATORS ---")
    
    sma_20 = ta.sma(close_np, 20)
    ema_20 = ta.ema(close_np, 20)
    wma_20 = ta.wma(close_np, 20)
    
    print(f"SMA(20) latest value: {sma_20[-1]:.2f}")
    print(f"EMA(20) latest value: {ema_20[-1]:.2f}")
    print(f"WMA(20) latest value: {wma_20[-1]:.2f}")
    
    # Supertrend with format preservation
    supertrend_np, direction_np = ta.supertrend(high_np, low_np, close_np, 10, 3)
    supertrend_pd, direction_pd = ta.supertrend(high_pd, low_pd, close_pd, 10, 3)
    
    trend_signal = "BULLISH" if direction_np[-1] == -1 else "BEARISH"
    print(f"Supertrend (NumPy): {supertrend_np[-1]:.2f} ({trend_signal})")
    print(f"Supertrend (Pandas): {supertrend_pd.iloc[-1]:.2f} (index: {supertrend_pd.index[-1].date()})")
    
    # Momentum Indicators
    print("\n--- MOMENTUM INDICATORS ---")
    
    rsi = ta.rsi(close_np, 14)
    macd_line, signal_line, histogram = ta.macd(close_np, 12, 26, 9)
    k_percent, d_percent = ta.stochastic(high_np, low_np, close_np, k_period=14, smooth_k=3, d_period=3)
    
    print(f"RSI(14): {rsi[-1]:.2f}")
    print(f"MACD: {macd_line[-1]:.4f}, Signal: {signal_line[-1]:.4f}")
    print(f"Stochastic: %K={k_percent[-1]:.2f}, %D={d_percent[-1]:.2f}")
    
    # Volatility Indicators
    print("\n--- VOLATILITY INDICATORS ---")
    
    atr = ta.atr(high_np, low_np, close_np, 14)
    upper, middle, lower = ta.bbands(close_np, 20, 2)
    
    print(f"ATR(14): {atr[-1]:.2f}")
    print(f"Bollinger Bands - Upper: {upper[-1]:.2f}, Middle: {middle[-1]:.2f}, Lower: {lower[-1]:.2f}")
    
    # Volume Indicators
    print("\n--- VOLUME INDICATORS ---")
    
    obv = ta.obv(close_np, volume_np)
    vwap = ta.vwap(high_np, low_np, close_np, volume_np)
    mfi = ta.mfi(high_np, low_np, close_np, volume_np, 14)
    
    print(f"OBV: {obv[-1]:.0f}")
    print(f"VWAP: {vwap[-1]:.2f}")
    print(f"MFI(14): {mfi[-1]:.2f}")
    
    # Pandas Series Demo for DataFrame Integration
    print("\n--- PANDAS INTEGRATION DEMO ---")
    
    # Create a DataFrame and add indicators (perfect for time-series analysis)
    df = pd.DataFrame({
        'close': close_pd,
        'high': high_pd,
        'low': low_pd,
        'volume': volume_pd
    })
    
    # Add indicators as new columns - format is automatically preserved!
    df['sma_20'] = ta.sma(df['close'], 20)
    df['rsi_14'] = ta.rsi(df['close'], 14)
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], 14)
    
    print("DataFrame with indicators (last 5 rows):")
    print(df[['close', 'sma_20', 'rsi_14', 'atr_14']].tail())
    print(f"\nAll indicators maintain DatetimeIndex: {type(df.index)}")


def strategy_example():
    """Example of using indicators for a simple trading strategy"""
    print("\n\nTrading Strategy Example")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(123)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.3)
    high = close + np.random.rand(n) * 1.5
    low = close - np.random.rand(n) * 1.5
    volume = np.random.randint(5000, 15000, n)
    
    # Calculate indicators
    ema_20 = ta.ema(close, 20)
    ema_50 = ta.ema(close, 50)
    rsi = ta.rsi(close, 14)
    upper, middle, lower = ta.bbands(close, 20, 2)
    
    # Generate signals
    bullish_ema = ta.crossover(ema_20, ema_50)
    bearish_ema = ta.crossunder(ema_20, ema_50)
    
    # Strategy: Buy when EMA20 crosses above EMA50 and RSI < 70
    # Sell when EMA20 crosses below EMA50 or RSI > 80
    buy_signals = bullish_ema & (rsi < 70)
    sell_signals = bearish_ema | (rsi > 80)
    
    # Count signals
    total_buy = np.sum(buy_signals)
    total_sell = np.sum(sell_signals)
    
    print(f"Strategy results over {n} periods:")
    print(f"Buy signals: {total_buy}")
    print(f"Sell signals: {total_sell}")
    
    # Show last few signals
    print("\nLast 10 periods signal analysis:")
    for i in range(-10, 0):
        period = n + i
        signal = ""
        if buy_signals[period]:
            signal = "BUY"
        elif sell_signals[period]:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        print(f"Period {period}: Close={close[period]:.2f}, "
              f"EMA20={ema_20[period]:.2f}, RSI={rsi[period]:.2f}, "
              f"Signal={signal}")


def real_data_example():
    """Example using real market data from OpenAlgo API"""
    print("\n\nReal Market Data Example")
    print("=" * 50)
    
    try:
        # Initialize API client (you need to replace with your credentials)
        client = api(
            api_key="your_api_key_here",
            host="http://127.0.0.1:5000"
        )
        
        print("Note: To run this example with real data, update the API credentials")
        print("and uncomment the following code:")
        
        # Uncomment this section when you have valid API credentials
        """
        # Get historical data
        df = client.history(
            symbol="RELIANCE",
            exchange="NSE", 
            interval="5m",
            start_date="2024-01-01",
            end_date="2024-01-31"
        )
        
        if isinstance(df, pd.DataFrame) and not df.empty:
            print(f"Downloaded {len(df)} records for RELIANCE")
            
            # Calculate indicators
            df['sma_20'] = ta.sma(df['close'], 20)
            df['ema_20'] = ta.ema(df['close'], 20)
            df['rsi'] = ta.rsi(df['close'], 14)
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], 14)
            
            # Bollinger Bands
            upper, middle, lower = ta.bbands(df['close'], 20, 2)
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            
            # Supertrend
            supertrend, direction = ta.supertrend(df['high'], df['low'], df['close'])
            df['supertrend'] = supertrend
            df['trend'] = direction
            
            # Display results
            print("Latest indicator values:")
            print(f"Close: {df['close'].iloc[-1]:.2f}")
            print(f"SMA(20): {df['sma_20'].iloc[-1]:.2f}")
            print(f"EMA(20): {df['ema_20'].iloc[-1]:.2f}")
            print(f"RSI(14): {df['rsi'].iloc[-1]:.2f}")
            print(f"ATR(14): {df['atr'].iloc[-1]:.2f}")
            print(f"Supertrend: {df['supertrend'].iloc[-1]:.2f}")
            
            # Save to CSV
            df.to_csv('reliance_with_indicators.csv')
            print("Data with indicators saved to 'reliance_with_indicators.csv'")
        else:
            print("Failed to fetch data or empty dataset received")
        """
        
    except Exception as e:
        print(f"Error in real data example: {e}")
        print("Make sure you have valid API credentials and OpenAlgo server is running")


def performance_test():
    """Test the performance of indicators with large datasets"""
    print("\n\nPerformance Test")
    print("=" * 50)
    
    import time
    
    # Test with different data sizes
    sizes = [1000, 10000, 100000]
    
    for size in sizes:
        print(f"\nTesting with {size:,} data points:")
        
        # Generate test data
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(size) * 0.1)
        high = close + np.random.rand(size) * 0.5
        low = close - np.random.rand(size) * 0.5
        volume = np.random.randint(1000, 10000, size)
        
        # Test SMA
        start_time = time.time()
        sma_result = ta.sma(close, 20)
        sma_time = time.time() - start_time
        print(f"SMA(20): {sma_time*1000:.2f}ms")
        
        # Test RSI
        start_time = time.time()
        rsi_result = ta.rsi(close, 14)
        rsi_time = time.time() - start_time
        print(f"RSI(14): {rsi_time*1000:.2f}ms")
        
        # Test Bollinger Bands
        start_time = time.time()
        bb_result = ta.bbands(close, 20, 2)
        bb_time = time.time() - start_time
        print(f"Bollinger Bands: {bb_time*1000:.2f}ms")
        
        # Test Supertrend
        start_time = time.time()
        st_result = ta.supertrend(high, low, close, 10, 3)
        st_time = time.time() - start_time
        print(f"Supertrend: {st_time*1000:.2f}ms")


def crossover_signals_example():
    """Example of using crossover utility functions"""
    print("\n\nCrossover Signals Example")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(456)
    n = 50
    close = 100 + np.cumsum(np.random.randn(n) * 0.2)
    
    # Calculate moving averages
    ema_fast = ta.ema(close, 10)
    ema_slow = ta.ema(close, 20)
    
    # Find crossover points
    bullish_cross = ta.crossover(ema_fast, ema_slow)
    bearish_cross = ta.crossunder(ema_fast, ema_slow)
    
    print("EMA Crossover Analysis:")
    print(f"Total bullish crossovers: {np.sum(bullish_cross)}")
    print(f"Total bearish crossovers: {np.sum(bearish_cross)}")
    
    print("\nCrossover events:")
    for i in range(len(close)):
        if bullish_cross[i]:
            print(f"Period {i}: BULLISH crossover at price {close[i]:.2f}")
        elif bearish_cross[i]:
            print(f"Period {i}: BEARISH crossover at price {close[i]:.2f}")


if __name__ == "__main__":
    print("OpenAlgo Technical Indicators Examples")
    print("=====================================\n")
    
    # Run examples
    basic_indicators_example()
    strategy_example()
    crossover_signals_example()
    performance_test()
    real_data_example()
    
    print("\n\nAll examples completed!")
    print("Check the code to see how to use each indicator.")
    print("For more information, visit: https://docs.openalgo.in")