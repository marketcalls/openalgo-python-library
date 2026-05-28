# -*- coding: utf-8 -*-
"""
Fetch benchmark / parity OHLCV data for the Rust-indicator migration.

Source priority (first that returns data wins, per symbol+interval):
  1. OpenAlgo DuckDB / Historify  (source="db")   -- preferred once populated
  2. OpenAlgo broker history       (source="api")  -- Dhan etc. when subscribed
  3. yfinance                                       -- fallback while brokers are down

Output: benchmark/data/<SYMBOL>_<INTERVAL>.csv with columns
        timestamp, open, high, low, close, volume   (timestamp as index)

Both Daily ("D") and 1-minute ("1m") series are fetched for RELIANCE and SBIN.
yfinance only serves ~7 days of 1m history, which is plenty for parity testing.
"""
import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")  # legacy numba stack is broken on numpy>=2

from pathlib import Path
import pandas as pd

API_KEY = "c9aeca4e3255182e59d2c58994c6d55fc04aecce09dac6c49b2c1ba39d480254"
HOST = "http://127.0.0.1:5000"
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["RELIANCE", "SBIN"]
INTERVALS = ["D", "1m"]
YF_SUFFIX = ".NS"  # NSE on Yahoo Finance
COLS = ["open", "high", "low", "close", "volume"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df = df[[c for c in COLS if c in df.columns]].copy()
    df.index.name = "timestamp"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df.dropna()


def from_openalgo(symbol, interval, source):
    from openalgo import api
    client = api(api_key=API_KEY, host=HOST)
    df = client.history(symbol=symbol, exchange="NSE", interval=interval,
                        start_date="2015-01-01", end_date="2026-05-29", source=source)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return _normalize(df), f"openalgo:{source}"
    return None, None


def from_yfinance(symbol, interval):
    import yfinance as yf
    ticker = symbol + YF_SUFFIX
    if interval == "D":
        raw = yf.download(ticker, period="max", interval="1d",
                          auto_adjust=False, progress=False)
    else:  # 1m -> Yahoo serves last ~7 days
        raw = yf.download(ticker, period="7d", interval="1m",
                          auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        return None, None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return _normalize(raw), "yfinance"


def fetch(symbol, interval):
    for loader, args in [
        (from_openalgo, (symbol, interval, "db")),
        (from_openalgo, (symbol, interval, "api")),
        (from_yfinance, (symbol, interval)),
    ]:
        try:
            df, src = loader(*args)
        except Exception as exc:  # noqa: BLE001 - any source can fail; just try next
            df, src = None, None
            print(f"    {loader.__name__}{args}: {type(exc).__name__}: {exc}")
        if df is not None and len(df):
            return df, src
    return None, None


def main():
    print(f"Writing to {DATA_DIR}")
    summary = []
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            print(f"[{symbol} {interval}] fetching...")
            df, src = fetch(symbol, interval)
            if df is None:
                print(f"  -> NO DATA from any source")
                summary.append((symbol, interval, "FAILED", 0, "", ""))
                continue
            out = DATA_DIR / f"{symbol}_{interval}.csv"
            df.to_csv(out)
            lo, hi = str(df.index.min()), str(df.index.max())
            print(f"  -> {len(df)} rows via {src}  [{lo} .. {hi}]  {out.name}")
            summary.append((symbol, interval, src, len(df), lo, hi))

    print("\nSUMMARY")
    print(f"{'symbol':10} {'iv':4} {'source':14} {'rows':>7}  range")
    for sym, iv, src, rows, lo, hi in summary:
        print(f"{sym:10} {iv:4} {src:14} {rows:>7}  {lo} .. {hi}")


if __name__ == "__main__":
    main()
