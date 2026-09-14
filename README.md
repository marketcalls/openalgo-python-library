# OpenAlgo Python Library

A Python library for algorithmic trading using OpenAlgo's REST APIs and WebSocket feeds, with **100+ high-performance technical indicators powered by a Rust core**.

- Python Library Docs: https://docs.openalgo.in/trading-platform/python
- Technical Indicators (100+): https://docs.openalgo.in/trading-platform/python/indicators
- WebSocket & Verbose Control: https://docs.openalgo.in/trading-platform/python/websockets-verbose-control
- API Reference: https://docs.openalgo.in/api-documentation/v1
- General Documentation: https://docs.openalgo.in
- Source: https://github.com/marketcalls/openalgo-python-library

## What's New in 2.0.5

**Bug-fix release: NaN handling in the indicator kernels.** Chaining one indicator
into another was silently producing nothing, because every indicator emits warm-up
NaNs and the moving-average kernels let a single NaN poison their running
accumulator for the rest of the series ([#2029]).

```python
df["RSI_wma"] = ta.rsi(ta.wma(df["close"], 55), 14)
df["RSI_avg"] = ta.wma(df["RSI_wma"], 9)      # was: NaN on every row
ta.crossover(df["RSI_wma"], df["RSI_avg"]).sum()   # was: 0 across 1600 bars
```

Three of these failed silently with plausible numbers rather than NaN:

- **`stdev` returned `0.0`**, not NaN, for the rest of the series after a NaN
  (`f64::max` yields the *other* operand against NaN), collapsing Bollinger width.
- **`rsi` returned a flat `100`** over an upstream indicator's warm-up: a NaN delta
  is neither a gain nor a loss, so `avg_loss` seeded to 0 and took the
  "no losses" branch.
- **`median` panicked** through the PyO3 boundary on any window containing a NaN.

`ta.crossover` / `ta.crossunder` were never at fault and are unchanged.

**The NaN contract is now stated and enforced on both backends.** Rolling-window
kernels are *window-local*: a NaN blanks only the windows containing it and the
series recovers once it slides out, the same rule as pandas `.rolling(period)`.
Recursive kernels (the EMA family) skip leading NaNs and seed at the first finite
value. The Rust core and the pure-NumPy fallback had drifted apart here, so the
same script gave different answers from a wheel and from a source checkout; they
are now checked against each other in CI.

`ta.vi` and `ta.ulcerindex` returned all-NaN on clean OHLCV and now produce values.
For NaN-free input every other indicator is unchanged, bit for bit.

[#2029]: https://github.com/marketcalls/openalgo/issues/2029

## What's New in 2.0.4

- **GTT (Good Till Triggered) orders**: `placegttorder`, `modifygttorder`,
  `cancelgttorder` and `gttorderbook`, covering both SINGLE and OCO triggers. An
  impossible trigger spec is refused locally before anything is sent.
- **Strategy module API**: nine methods on the API-key surface -
  `strategylist`, `strategystatus`, `strategystart`, `strategystop`,
  `strategycloseall`, `strategycloseleg`, `strategyruns`, `strategyorders` and
  `strategyevents`.
- **`Strategy` webhook client revamped** for the new protocol: `start(mode)` and
  `stop()` for batch strategies, `long_entry` / `long_exit` / `short_entry` /
  `short_exit` for signal strategies. `mode` on start has no default, the token
  is never rendered in a repr, and every documented rejection is returned with
  its `result` label instead of raised.
- **Breaking**: `Strategy.strategyorder()` was removed. The webhook it posted to
  no longer exists. See [Strategy Module](#strategy-module) for the replacements.

## What's New in 2.0.0

Version 2.0.0 replaces the old Numba/JIT indicator engine with a **Rust core** (via PyO3):

- **No optional extra, no Numba.** Indicators are compiled into the wheel. The legacy
  `pip install openalgo[indicators]` extra and the `numba`/`llvmlite` dependencies are
  removed; `pip install openalgo` is all you need.
- **Python 3.12, 3.13 and 3.14** are all supported (abi3 wheels). Numba previously
  blocked newer Python/NumPy versions.
- **New TA-Lib-compatible indicators:** `mom`, `rocp`, `rocr`, `rocr100`, `apo`,
  `midpoint`, `midprice`, `avgprice`, `medprice`, `typprice`, `wclprice`, `plus_dm`,
  `minus_dm`, `dx`, `adxr`, `stochf`, `linregangle`, `linregintercept`.
- **Performance:** every indicator is O(n). Benchmarked head-to-head with TA-Lib on
  924k bars, the regression/statistics family (`linreg`, `tsf`, `stddev`, `cci`,
  `macd`, ...) runs faster than TA-Lib; the rest are on par. See the
  [performance comparison](https://github.com/marketcalls/openalgo-python-library/blob/master/benchmark/TALIB_PERF_COMPARE.md)
  and [TA-Lib compatibility notes](https://github.com/marketcalls/openalgo-python-library/blob/master/docs/TALIB_COMPATIBILITY.md).
- **Backward compatible:** the `from openalgo import ta` API is unchanged - existing
  code keeps working without modification.

## Installation

To install the OpenAlgo Python library, use pip:

```bash
pip install openalgo
```

The 100+ technical indicators are built in (powered by a Rust core); no extra install
step or optional dependency is required.

## Get the OpenAlgo apikey

Make sure that your OpenAlgo Application is running. Login to OpenAlgo Application with valid credentials and get the OpenAlgo apikey.

For detailed function parameters refer to the [API Documentation](https://docs.openalgo.in/api-documentation/v1).

## Getting Started with OpenAlgo

First, import the `api` class from the OpenAlgo library and initialize it with your API key:

```python
from openalgo import api

# Replace 'your_api_key_here' with your actual API key
# Specify the host URL with your hosted domain or ngrok domain.
# If running locally in windows then use the default host value.
client = api(api_key='your_api_key_here', host='http://127.0.0.1:5000')
```

## Check OpenAlgo Version

```python
import openalgo
openalgo.__version__
```

## Technical Indicators (100+)

OpenAlgo ships **100+ technical indicators** powered by a Rust core (via PyO3) —
including trend, momentum, volatility, volume, oscillators, statistics, and hybrid
indicators. They are built in; no optional dependency or extra install step:

```bash
pip install openalgo
```

Quick example:

```python
import numpy as np
from openalgo import ta

close = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], dtype=float)
high  = close + 0.5
low   = close - 0.5

# Trend
sma   = ta.sma(close, period=5)
ema   = ta.ema(close, period=5)
supertrend, direction = ta.supertrend(high, low, close, period=7, multiplier=3.0)

# Momentum
rsi   = ta.rsi(close, period=14)
macd_line, signal_line, hist = ta.macd(close, fast=12, slow=26, signal=9)

# Volatility
atr   = ta.atr(high, low, close, period=14)
upper, middle, lower = ta.bbands(close, period=20, std=2.0)
```

Many indicators are value-compatible with TA-Lib; where OpenAlgo intentionally follows
TradingView/Pine conventions instead (EMA/ATR/ADX seeding, etc.), the differences are
documented in the
[TA-Lib compatibility notes](https://github.com/marketcalls/openalgo-python-library/blob/master/docs/TALIB_COMPATIBILITY.md).

Full indicator catalog and parameter reference: https://docs.openalgo.in/trading-platform/python/indicators

## WebSocket Verbose Control

The streaming feed supports verbosity levels (`0` silent, `1` connection/auth/subscription info, `2` full debug with every tick):

```python
client = api(
    api_key="your_api_key",
    host="http://127.0.0.1:5000",
    ws_url="ws://127.0.0.1:8765",
    verbose=1,                 # 0 / 1 / True / 2
)
```

Details: https://docs.openalgo.in/trading-platform/python/websockets-verbose-control

## Examples

Please refer to the documentation on [order constants](https://docs.openalgo.in/api-documentation/v1/order-constants), and consult the API reference for details on optional parameters.

### PlaceOrder example

To place a new market order:

```python
response = client.placeorder(
    strategy="Python",
    symbol="NHPC",
    action="BUY",
    exchange="NSE",
    price_type="MARKET",
    product="MIS",
    quantity=1
)
print(response)
```

Place Market Order Response:

```json
{"orderid": "250408000989443", "status": "success"}
```

To place a new limit order:

```python
response = client.placeorder(
    strategy="Python",
    symbol="YESBANK",
    action="BUY",
    exchange="NSE",
    price_type="LIMIT",
    product="MIS",
    quantity="1",
    price="16",
    trigger_price="0",
    disclosed_quantity="0",
)
print(response)
```

Place Limit Order Response:

```json
{"orderid": "250408001003813", "status": "success"}
```

### PlaceSmartOrder Example

To place a smart order considering the current position size:

```python
response = client.placesmartorder(
    strategy="Python",
    symbol="TATAMOTORS",
    action="SELL",
    exchange="NSE",
    price_type="MARKET",
    product="MIS",
    quantity=1,
    position_size=5
)
print(response)
```

Place Smart Market Order Response:

```json
{"orderid": "250408000997543", "status": "success"}
```

### OptionsOrder Example

To place an ATM options order:

```python
response = client.optionsorder(
    strategy="python",
    underlying="NIFTY",
    exchange="NSE_INDEX",
    expiry_date="28OCT25",
    offset="ATM",
    option_type="CE",
    action="BUY",
    quantity=75,
    pricetype="MARKET",
    product="NRML",
    splitsize=0
)
print(response)
```

Place Options Order Response:

```json
{
  "exchange": "NFO",
  "offset": "ATM",
  "option_type": "CE",
  "orderid": "25102800000006",
  "status": "success",
  "symbol": "NIFTY28OCT2525950CE",
  "underlying": "NIFTY28OCT25FUT",
  "underlying_ltp": 25966.05
}
```

To place an ITM options order:

```python
response = client.optionsorder(
    strategy="python",
    underlying="NIFTY",
    exchange="NSE_INDEX",
    expiry_date="28OCT25",
    offset="ITM4",
    option_type="PE",
    action="BUY",
    quantity=75,
    pricetype="MARKET",
    product="NRML",
    splitsize=0
)
print(response)
```

Place Options Order Response:

```json
{
  "exchange": "NFO",
  "offset": "ITM4",
  "option_type": "PE",
  "orderid": "25102800000007",
  "status": "success",
  "symbol": "NIFTY28OCT2526150PE",
  "underlying": "NIFTY28OCT25FUT",
  "underlying_ltp": 25966.05
}
```

To place an OTM options order:

```python
response = client.optionsorder(
    strategy="python",
    underlying="NIFTY",
    exchange="NSE_INDEX",
    expiry_date="28OCT25",
    offset="OTM5",
    option_type="CE",
    action="BUY",
    quantity=75,
    pricetype="MARKET",
    product="NRML",
    splitsize=0
)
print(response)
```

Place Options Order Response:

```json
{
  "exchange": "NFO",
  "mode": "analyze",
  "offset": "OTM5",
  "option_type": "CE",
  "orderid": "25102800000008",
  "status": "success",
  "symbol": "NIFTY28OCT2526200CE",
  "underlying": "NIFTY28OCT25FUT",
  "underlying_ltp": 25966.05
}
```

### OptionsMultiOrder Example

To place an Iron Condor (same expiry):

```python
response = client.optionsmultiorder(
    strategy="Iron Condor Test",
    underlying="NIFTY",
    exchange="NSE_INDEX",
    expiry_date="25NOV25",
    legs=[
        {"offset": "OTM6", "option_type": "CE", "action": "BUY", "quantity": 75},
        {"offset": "OTM6", "option_type": "PE", "action": "BUY", "quantity": 75},
        {"offset": "OTM4", "option_type": "CE", "action": "SELL", "quantity": 75},
        {"offset": "OTM4", "option_type": "PE", "action": "SELL", "quantity": 75}
    ]
)
print(response)
```

Place OptionsMultiOrder Response:

```json
{
  "status": "success",
  "underlying": "NIFTY",
  "underlying_ltp": 26050.45,
  "results": [
    {
      "action": "BUY",
      "leg": 1,
      "mode": "analyze",
      "offset": "OTM6",
      "option_type": "CE",
      "orderid": "25111996859688",
      "status": "success",
      "symbol": "NIFTY25NOV2526350CE"
    },
    {
      "action": "BUY",
      "leg": 2,
      "mode": "analyze",
      "offset": "OTM6",
      "option_type": "PE",
      "orderid": "25111996042210",
      "status": "success",
      "symbol": "NIFTY25NOV2525750PE"
    },
    {
      "action": "SELL",
      "leg": 3,
      "mode": "analyze",
      "offset": "OTM4",
      "option_type": "CE",
      "orderid": "25111922189638",
      "status": "success",
      "symbol": "NIFTY25NOV2526250CE"
    },
    {
      "action": "SELL",
      "leg": 4,
      "mode": "analyze",
      "offset": "OTM4",
      "option_type": "PE",
      "orderid": "25111919252668",
      "status": "success",
      "symbol": "NIFTY25NOV2525850PE"
    }
  ]
}
```

To place a Diagonal Spread (different expiry):

```python
response = client.optionsmultiorder(
    strategy="Diagonal Spread Test",
    underlying="NIFTY",
    exchange="NSE_INDEX",
    legs=[
        {"offset": "ITM2", "option_type": "CE", "action": "BUY", "quantity": 75, "expiry_date": "30DEC25"},
        {"offset": "OTM2", "option_type": "CE", "action": "SELL", "quantity": 75, "expiry_date": "25NOV25"}
    ]
)
print(response)
```

Place OptionsMultiOrder Response:

```json
{
  "results": [
    {
      "action": "BUY",
      "leg": 1,
      "mode": "analyze",
      "offset": "ITM2",
      "option_type": "CE",
      "orderid": "25111933337854",
      "status": "success",
      "symbol": "NIFTY30DEC2525950CE"
    },
    {
      "action": "SELL",
      "leg": 2,
      "mode": "analyze",
      "offset": "OTM2",
      "option_type": "CE",
      "orderid": "25111957475473",
      "status": "success",
      "symbol": "NIFTY25NOV2526150CE"
    }
  ],
  "status": "success",
  "underlying": "NIFTY",
  "underlying_ltp": 26052.65
}
```

### BasketOrder example

To place a new basket order:

```python
basket_orders = [
    {
        "symbol": "BHEL",
        "exchange": "NSE",
        "action": "BUY",
        "quantity": 1,
        "pricetype": "MARKET",
        "product": "MIS"
    },
    {
        "symbol": "ZOMATO",
        "exchange": "NSE",
        "action": "SELL",
        "quantity": 1,
        "pricetype": "MARKET",
        "product": "MIS"
    }
]
response = client.basketorder(orders=basket_orders)
print(response)
```

Basket Order Response:

```json
{
  "status": "success",
  "results": [
    {"symbol": "BHEL", "status": "success", "orderid": "250408000999544"},
    {"symbol": "ZOMATO", "status": "success", "orderid": "250408000997545"}
  ]
}
```

### SplitOrder example

To place a new split order:

```python
response = client.splitorder(
    symbol="YESBANK",
    exchange="NSE",
    action="SELL",
    quantity=105,
    splitsize=20,
    price_type="MARKET",
    product="MIS"
)
print(response)
```

SplitOrder Response:

```json
{
  "status": "success",
  "split_size": 20,
  "total_quantity": 105,
  "results": [
    {"order_num": 1, "orderid": "250408001021467", "quantity": 20, "status": "success"},
    {"order_num": 2, "orderid": "250408001021459", "quantity": 20, "status": "success"},
    {"order_num": 3, "orderid": "250408001021466", "quantity": 20, "status": "success"},
    {"order_num": 4, "orderid": "250408001021470", "quantity": 20, "status": "success"},
    {"order_num": 5, "orderid": "250408001021471", "quantity": 20, "status": "success"},
    {"order_num": 6, "orderid": "250408001021472", "quantity": 5, "status": "success"}
  ]
}
```

### ModifyOrder Example

To modify an existing order:

```python
response = client.modifyorder(
    order_id="250408001002736",
    strategy="Python",
    symbol="YESBANK",
    action="BUY",
    exchange="NSE",
    price_type="LIMIT",
    product="CNC",
    quantity=1,
    price=16.5
)
print(response)
```

Modify Order Response:

```json
{"orderid": "250408001002736", "status": "success"}
```

### CancelOrder Example

To cancel an existing order:

```python
response = client.cancelorder(
    order_id="250408001002736",
    strategy="Python"
)
print(response)
```

Cancelorder Response:

```json
{"orderid": "250408001002736", "status": "success"}
```

### CancelAllOrder Example

To cancel all open orders and trigger pending orders:

```python
response = client.cancelallorder(strategy="Python")
print(response)
```

Cancelallorder Response:

```json
{
  "status": "success",
  "message": "Canceled 5 orders. Failed to cancel 0 orders.",
  "canceled_orders": [
    "250408001042620",
    "250408001042667",
    "250408001042642",
    "250408001043015",
    "250408001043386"
  ],
  "failed_cancellations": []
}
```

### ClosePosition Example

To close all open positions across various exchanges:

```python
response = client.closeposition(strategy="Python")
print(response)
```

ClosePosition Response:

```json
{"message": "All Open Positions Squared Off", "status": "success"}
```

### OrderStatus Example

To get the current order status:

```python
response = client.orderstatus(
    order_id="250828000185002",
    strategy="Test Strategy"
)
print(response)
```

Orderstatus Response:

```json
{
  "data": {
    "action": "BUY",
    "average_price": 18.95,
    "exchange": "NSE",
    "order_status": "complete",
    "orderid": "250828000185002",
    "price": 0,
    "pricetype": "MARKET",
    "product": "MIS",
    "quantity": "1",
    "symbol": "YESBANK",
    "timestamp": "28-Aug-2025 09:59:10",
    "trigger_price": 0
  },
  "status": "success"
}
```

### OpenPosition Example

To get the current open position:

```python
response = client.openposition(
    strategy="Test Strategy",
    symbol="YESBANK",
    exchange="NSE",
    product="MIS"
)
print(response)
```

OpenPosition Response:

```json
{"quantity": "-10", "status": "success"}
```

### PlaceGTTOrder Example

A GTT (Good Till Triggered) order is a price trigger that sits with the broker until
LTP crosses your level, then places the underlying order automatically.

There are two shapes, and picking the wrong one is the usual mistake:

| Type | Use when | Triggers | Orders fired |
|------|----------|----------|--------------|
| `SINGLE` | One entry or exit at a level | 1 | 1 |
| `OCO` | You hold a position and want both a stoploss and a target, whichever hits first | 2 | 1 of 2, the other is auto-cancelled |

For a **SINGLE**, exactly one of `triggerprice_sl` / `triggerprice_tg` carries your
level and the other stays `0`. Pick by where the trigger sits relative to LTP:
`triggerprice_sl` for a level **below** LTP (sell stop-loss, buy the dip),
`triggerprice_tg` for one **above** (breakout buy, sell at target). A SINGLE has no
stoploss leg, so the suffix is only a directional hint.

For an **OCO**, the suffix is a real role and all four fields are required:
`triggerprice_sl` with its `stoploss` limit, and `triggerprice_tg` with its `target`
limit, where `triggerprice_sl < triggerprice_tg`.

GTT accepts `CNC` and `NRML` only. `MIS` is refused: a GTT can sit for days and MIS is
squared off the same session.

```python
# SINGLE - "Buy IDEA if it dips to 9.55, with a LIMIT order at 9.50"
# LTP is above 9.55, so the trigger sits below it -> triggerprice_sl
response = client.placegttorder(
    strategy="My GTT Strategy",
    symbol="IDEA",
    action="BUY",
    exchange="NSE",
    product="CNC",
    quantity=1,
    price_type="LIMIT",
    price=9.50,
    triggerprice_sl=9.55
)
print(response)
```

```python
# SINGLE - "Buy RELIANCE at MARKET if it breaks above 1450"
# LTP is below 1450, so the trigger sits above it -> triggerprice_tg
response = client.placegttorder(
    strategy="My GTT Strategy",
    symbol="RELIANCE",
    action="BUY",
    exchange="NSE",
    product="CNC",
    quantity=1,
    price_type="MARKET",
    price=0,
    triggerprice_tg=1450
)
```

```python
# OCO - "I am short 5 INFY. Stop me out at 1480, take profit at 1620"
# price is 0: OCO prices each leg separately through stoploss and target
response = client.placegttorder(
    strategy="Bracket OCO",
    trigger_type="OCO",
    symbol="INFY",
    action="SELL",
    exchange="NSE",
    product="CNC",
    quantity=5,
    price_type="LIMIT",
    price=0,
    triggerprice_sl=1480,
    stoploss=1478,
    triggerprice_tg=1620,
    target=1622
)
```

PlaceGTTOrder Response:

```json
{"status": "success", "trigger_id": "23132604291205"}
```

Save the `trigger_id`: modify and cancel both need it.

### ModifyGTTOrder Example

Modify is a **full replacement**, not a patch. Every field on the trigger is replaced
by what the call sends, so pass everything you want to keep rather than only the
values that changed.

```python
response = client.modifygttorder(
    trigger_id="23132604291205",
    strategy="My GTT Strategy",
    symbol="IDEA",
    action="BUY",
    exchange="NSE",
    product="CNC",
    quantity=1,
    price_type="LIMIT",
    price=9.60,           # was 9.50
    triggerprice_sl=9.65  # was 9.55
)
print(response)
```

ModifyGTTOrder Response:

```json
{"status": "success", "trigger_id": "23132604291205"}
```

Trigger prices, limit prices, quantity and pricetype are modifiable. `trigger_type`,
`symbol`, `exchange` and `action` are not - cancel and re-place instead. Only active
GTTs can be modified; triggered, cancelled and expired ones are immutable.

### CancelGTTOrder Example

```python
response = client.cancelgttorder(
    trigger_id="23132604291205",
    strategy="My GTT Strategy"
)
print(response)
```

CancelGTTOrder Response:

```json
{"status": "success", "trigger_id": "23132604291205"}
```

Cancelling an OCO removes both legs atomically; there is no per-leg cancel.

### GTTOrderBook Example

By default this lists **active** triggers only, the ones that can still fire. Pass
`status="all"` to include the history as well (triggered, cancelled, expired,
rejected), ordered active first; in analyzer mode a fired leg also carries the
`triggered_order_id` of the sandbox order it placed.

```python
# Active triggers only (default)
response = client.gttorderbook()
print(response)

# Active triggers first, then the triggered / cancelled / expired history
response = client.gttorderbook(status="all")
for gtt in response["data"]:
    print(gtt["trigger_id"], gtt["status"], gtt["symbol"], gtt["trigger_prices"])
```

GTTOrderBook Response:

```json
{
  "status": "success",
  "data": [
    {
      "trigger_id": "23132604291205",
      "trigger_type": "single",
      "status": "active",
      "symbol": "IDEA",
      "exchange": "NSE",
      "trigger_prices": [9.55],
      "last_price": 9.50,
      "legs": [
        {
          "action": "BUY",
          "quantity": 1,
          "price": 9.50,
          "pricetype": "LIMIT",
          "product": "CNC"
        }
      ],
      "created_at": "2026-04-29 12:18:42",
      "updated_at": "",
      "expires_at": ""
    }
  ]
}
```

`trigger_prices` is sorted ascending: a SINGLE has one element and one leg, an OCO has
two of each with the stoploss first.

The SDK refuses an impossible trigger spec before anything leaves the machine, and
returns the refusal in the same shape as an API error:

```python
# SINGLE with no trigger price at all
client.placegttorder(symbol="IDEA", action="BUY", exchange="NSE",
                     product="CNC", quantity=1, price=9.50)
# {'status': 'error',
#  'message': 'SINGLE GTT requires a positive triggerprice_sl or triggerprice_tg.',
#  'error_type': 'validation_error'}
```

### Quotes Example

```python
response = client.quotes(symbol="RELIANCE", exchange="NSE")
print(response)
```

Quotes Response:

```json
{
  "status": "success",
  "data": {
    "open": 1172.0,
    "high": 1196.6,
    "low": 1163.3,
    "ltp": 1187.75,
    "ask": 1188.0,
    "bid": 1187.85,
    "prev_close": 1165.7,
    "volume": 14414545
  }
}
```

### MultiQuotes Example

```python
response = client.multiquotes(symbols=[
    {"symbol": "RELIANCE", "exchange": "NSE"},
    {"symbol": "TCS", "exchange": "NSE"},
    {"symbol": "INFY", "exchange": "NSE"}
])
print(response)
```

MultiQuotes Response:

```json
{
  "status": "success",
  "results": [
    {
      "symbol": "RELIANCE",
      "exchange": "NSE",
      "data": {
        "open": 1542.3, "high": 1571.6, "low": 1540.5, "ltp": 1569.9,
        "prev_close": 1539.7, "ask": 1569.9, "bid": 0, "oi": 0, "volume": 14054299
      }
    },
    {
      "symbol": "TCS",
      "exchange": "NSE",
      "data": {
        "open": 3118.8, "high": 3178, "low": 3117, "ltp": 3162.9,
        "prev_close": 3119.2, "ask": 0, "bid": 3162.9, "oi": 0, "volume": 2508527
      }
    },
    {
      "symbol": "INFY",
      "exchange": "NSE",
      "data": {
        "open": 1532.1, "high": 1560.3, "low": 1532.1, "ltp": 1557.9,
        "prev_close": 1530.6, "ask": 0, "bid": 1557.9, "oi": 0, "volume": 7575038
      }
    }
  ]
}
```

### Depth Example

```python
response = client.depth(symbol="SBIN", exchange="NSE")
print(response)
```

Depth Response:

```json
{
  "status": "success",
  "data": {
    "open": 760.0,
    "high": 774.0,
    "low": 758.15,
    "ltp": 769.6,
    "ltq": 205,
    "prev_close": 746.9,
    "volume": 9362799,
    "oi": 161265750,
    "totalbuyqty": 591351,
    "totalsellqty": 835701,
    "asks": [
      {"price": 769.6,  "quantity": 767},
      {"price": 769.65, "quantity": 115},
      {"price": 769.7,  "quantity": 162},
      {"price": 769.75, "quantity": 1121},
      {"price": 769.8,  "quantity": 430}
    ],
    "bids": [
      {"price": 769.4,  "quantity": 886},
      {"price": 769.35, "quantity": 212},
      {"price": 769.3,  "quantity": 351},
      {"price": 769.25, "quantity": 343},
      {"price": 769.2,  "quantity": 399}
    ]
  }
}
```

### History Example

Download data directly from broker API:

```python
response = client.history(
    symbol="SBIN",
    exchange="NSE",
    interval="5m",
    start_date="2025-04-01",
    end_date="2025-04-08",
    source="api"
)
print(response)
```

Download data from Historify DuckDB (stored data):

```python
response = client.history(
    symbol="SBIN",
    exchange="NSE",
    interval="5m",
    start_date="2025-04-01",
    end_date="2025-04-08",
    source="db"
)
print(response)
```

History Response:

```text
                            close    high     low    open  volume
timestamp
2025-04-01 09:15:00+05:30  772.50  774.00  763.20  766.50  318625
2025-04-01 09:20:00+05:30  773.20  774.95  772.10  772.45  197189
2025-04-01 09:25:00+05:30  775.15  775.60  772.60  773.20  227544
2025-04-01 09:30:00+05:30  777.35  777.50  774.85  775.15  134596
2025-04-01 09:35:00+05:30  778.00  778.00  776.25  777.50  145385
...                           ...     ...     ...     ...     ...
2025-04-08 14:00:00+05:30  768.25  770.70  767.85  768.50  142478
2025-04-08 14:05:00+05:30  769.10  769.80  766.60  768.15  128283
2025-04-08 14:10:00+05:30  769.05  769.85  768.40  769.10  119084
2025-04-08 14:15:00+05:30  770.05  770.50  769.05  769.05  158299
2025-04-08 14:20:00+05:30  769.95  770.50  769.40  770.05  125485

[437 rows x 5 columns]
```

### Intervals Example

```python
response = client.intervals()
print(response)
```

Intervals Response:

```json
{
  "status": "success",
  "data": {
    "months": [],
    "weeks": [],
    "days": ["D"],
    "hours": ["1h"],
    "minutes": ["10m", "15m", "1m", "30m", "3m", "5m"],
    "seconds": []
  }
}
```

### OptionChain Example

Note: To fetch the entire option chain for an expiry, omit the `strike_count` parameter.

```python
chain = client.optionchain(
    underlying="NIFTY",
    exchange="NSE_INDEX",
    expiry_date="30DEC25",
    strike_count=10
)
```

OptionChain Response:

```json
{
  "status": "success",
  "underlying": "NIFTY",
  "underlying_ltp": 26215.55,
  "expiry_date": "30DEC25",
  "atm_strike": 26200.0,
  "chain": [
    {
      "strike": 26100.0,
      "ce": {
        "symbol": "NIFTY30DEC2526100CE", "label": "ITM2",
        "ltp": 490, "bid": 490, "ask": 491,
        "open": 540, "high": 571, "low": 444.75,
        "prev_close": 496.8, "volume": 1195800, "oi": 0,
        "lotsize": 75, "tick_size": 0.05
      },
      "pe": {
        "symbol": "NIFTY30DEC2526100PE", "label": "OTM2",
        "ltp": 193, "bid": 191.2, "ask": 193,
        "open": 204.1, "high": 229.95, "low": 175.6,
        "prev_close": 215.95, "volume": 1832700, "oi": 0,
        "lotsize": 75, "tick_size": 0.05
      }
    },
    {
      "strike": 26200.0,
      "ce": {
        "symbol": "NIFTY30DEC2526200CE", "label": "ATM",
        "ltp": 427, "bid": 425.05, "ask": 427,
        "open": 449.95, "high": 503.5, "low": 384,
        "prev_close": 433.2, "volume": 2994000, "oi": 0,
        "lotsize": 75, "tick_size": 0.05
      },
      "pe": {
        "symbol": "NIFTY30DEC2526200PE", "label": "ATM",
        "ltp": 227.4, "bid": 227.35, "ask": 228.5,
        "open": 251.9, "high": 269.15, "low": 205.95,
        "prev_close": 251.9, "volume": 3745350, "oi": 0,
        "lotsize": 75, "tick_size": 0.05
      }
    }
  ]
}
```

### Symbol Example

```python
response = client.symbol(
    symbol="NIFTY30DEC25FUT",
    exchange="NFO"
)
print(response)
```

Symbol Response:

```json
{
  "data": {
    "brexchange": "NSE_FO",
    "brsymbol": "NIFTY FUT 30 DEC 25",
    "exchange": "NFO",
    "expiry": "30-DEC-25",
    "freeze_qty": 1800,
    "id": 57900,
    "instrumenttype": "FUT",
    "lotsize": 75,
    "name": "NIFTY",
    "strike": 0,
    "symbol": "NIFTY30DEC25FUT",
    "tick_size": 10,
    "token": "NSE_FO|49543"
  },
  "status": "success"
}
```

### Search Example

```python
response = client.search(query="NIFTY 26000 DEC CE", exchange="NFO")
print(response)
```

Search Response:

```json
{
  "data": [
    {
      "brexchange": "NSE_FO",
      "brsymbol": "NIFTY 26000 CE 30 DEC 25",
      "exchange": "NFO",
      "expiry": "30-DEC-25",
      "freeze_qty": 1800,
      "instrumenttype": "CE",
      "lotsize": 75,
      "name": "NIFTY",
      "strike": 26000,
      "symbol": "NIFTY30DEC2526000CE",
      "tick_size": 5,
      "token": "NSE_FO|71399"
    }
  ],
  "message": "Found 7 matching symbols",
  "status": "success"
}
```

### OptionSymbol Example

ATM Option:

```python
response = client.optionsymbol(
    underlying="NIFTY",
    exchange="NSE_INDEX",
    expiry_date="30DEC25",
    offset="ATM",
    option_type="CE"
)
print(response)
```

OptionSymbol Response:

```json
{
  "status": "success",
  "symbol": "NIFTY30DEC2525950CE",
  "exchange": "NFO",
  "lotsize": 75,
  "tick_size": 5,
  "freeze_qty": 1800,
  "underlying_ltp": 25966.4
}
```

ITM Option:

```python
response = client.optionsymbol(
    underlying="NIFTY",
    exchange="NSE_INDEX",
    expiry_date="30DEC25",
    offset="ITM3",
    option_type="PE"
)
print(response)
```

OptionSymbol Response:

```json
{
  "status": "success",
  "symbol": "NIFTY30DEC2526100PE",
  "exchange": "NFO",
  "lotsize": 75,
  "tick_size": 5,
  "freeze_qty": 1800,
  "underlying_ltp": 25966.4
}
```

OTM Option:

```python
response = client.optionsymbol(
    underlying="NIFTY",
    exchange="NSE_INDEX",
    expiry_date="30DEC25",
    offset="OTM4",
    option_type="CE"
)
print(response)
```

OptionSymbol Response:

```json
{
  "status": "success",
  "symbol": "NIFTY30DEC2526150CE",
  "exchange": "NFO",
  "lotsize": 75,
  "tick_size": 5,
  "freeze_qty": 1800,
  "underlying_ltp": 25966.4
}
```

### SyntheticFuture Example

```python
response = client.syntheticfuture(
    underlying="NIFTY",
    exchange="NSE_INDEX",
    expiry_date="25NOV25"
)
print(response)
```

SyntheticFuture Response:

```json
{
  "atm_strike": 25900.0,
  "expiry": "25NOV25",
  "status": "success",
  "synthetic_future_price": 25980.05,
  "underlying": "NIFTY",
  "underlying_ltp": 25910.05
}
```

### OptionGreeks Example

```python
response = client.optiongreeks(
    symbol="NIFTY25NOV2526000CE",
    exchange="NFO",
    interest_rate=0.00,
    underlying_symbol="NIFTY",
    underlying_exchange="NSE_INDEX"
)
print(response)
```

OptionGreeks Response:

```json
{
  "days_to_expiry": 28.5071,
  "exchange": "NFO",
  "expiry_date": "25-Nov-2025",
  "greeks": {
    "delta": 0.4967,
    "gamma": 0.000352,
    "rho": 9.733994,
    "theta": -7.919,
    "vega": 28.9489
  },
  "implied_volatility": 15.6,
  "interest_rate": 0.0,
  "option_price": 435,
  "option_type": "CE",
  "spot_price": 25966.05,
  "status": "success",
  "strike": 26000.0,
  "symbol": "NIFTY25NOV2526000CE",
  "underlying": "NIFTY"
}
```

### Expiry Example

```python
response = client.expiry(
    symbol="NIFTY",
    exchange="NFO",
    instrumenttype="options"
)
print(response)
```

Expiry Response:

```json
{
  "data": [
    "10-JUL-25", "17-JUL-25", "24-JUL-25", "31-JUL-25",
    "07-AUG-25", "28-AUG-25", "25-SEP-25", "24-DEC-25",
    "26-MAR-26", "25-JUN-26", "31-DEC-26", "24-JUN-27",
    "30-DEC-27", "29-JUN-28", "28-DEC-28", "28-JUN-29",
    "27-DEC-29", "25-JUN-30"
  ],
  "message": "Found 18 expiry dates for NIFTY options in NFO",
  "status": "success"
}
```

### Instruments Example

```python
response = client.instruments(exchange="NSE")
print(response.tail())
```

Instruments Response:

```text
     brexchange           brsymbol exchange expiry instrumenttype  lotsize  \
3041        NSE      NSE:NEOGEN-EQ      NSE   None             EQ        1
3042        NSE     NSE:ALANKIT-EQ      NSE   None             EQ        1
3043        NSE  NSE:EVERESTIND-EQ      NSE   None             EQ        1
3044        NSE   NSE:VIKASLIFE-EQ      NSE   None             EQ        1
3045        NSE    NSE:ONEPOINT-EQ      NSE   None             EQ        1

                          name  strike      symbol  tick_size           token
3041  NEOGEN CHEMICALS LIMITED    -1.0      NEOGEN       0.10  10100000009917
3042           ALANKIT LIMITED    -1.0     ALANKIT       0.01  10100000009921
3043    EVEREST INDUSTRIES LTD    -1.0  EVERESTIND       0.05   1010000000993
3044    VIKAS LIFECARE LIMITED    -1.0   VIKASLIFE       0.01  10100000009931
3045     ONE POINT ONE SOL LTD    -1.0    ONEPOINT       0.01  10100000009939
```

### Telegram Alert Example

```python
response = client.telegram(
    username="<openalgo_loginid>",
    message="NIFTY crossed 26000!"
)
print(response)
```

Telegram Alert Response:

```json
{
  "message": "Notification sent successfully",
  "status": "success"
}
```

### Funds Example

```python
response = client.funds()
print(response)
```

Funds Response:

```json
{
  "status": "success",
  "data": {
    "availablecash": "320.66",
    "collateral": "0.00",
    "m2mrealized": "3.27",
    "m2munrealized": "-7.88",
    "utiliseddebits": "679.34"
  }
}
```

### Margin Example

```python
response = client.margin(positions=[
    {
        "symbol": "NIFTY25NOV2525000CE",
        "exchange": "NFO",
        "action": "BUY",
        "product": "NRML",
        "pricetype": "MARKET",
        "quantity": "75"
    },
    {
        "symbol": "NIFTY25NOV2525500CE",
        "exchange": "NFO",
        "action": "SELL",
        "product": "NRML",
        "pricetype": "MARKET",
        "quantity": "75"
    }
])
```

Margin Response:

```json
{
  "status": "success",
  "data": {
    "total_margin_required": 91555.7625,
    "span_margin": 0.0,
    "exposure_margin": 91555.7625
  }
}
```

### OrderBook Example

```python
response = client.orderbook()
print(response)
```

OrderBook Response:

```json
{
  "status": "success",
  "data": {
    "orders": [
      {
        "action": "BUY",
        "symbol": "RELIANCE",
        "exchange": "NSE",
        "orderid": "250408000989443",
        "product": "MIS",
        "quantity": "1",
        "price": 1186.0,
        "pricetype": "MARKET",
        "order_status": "complete",
        "trigger_price": 0.0,
        "timestamp": "08-Apr-2025 13:58:03"
      },
      {
        "action": "BUY",
        "symbol": "YESBANK",
        "exchange": "NSE",
        "orderid": "250408001002736",
        "product": "MIS",
        "quantity": "1",
        "price": 16.5,
        "pricetype": "LIMIT",
        "order_status": "cancelled",
        "trigger_price": 0.0,
        "timestamp": "08-Apr-2025 14:13:45"
      }
    ],
    "statistics": {
      "total_buy_orders": 2.0,
      "total_sell_orders": 0.0,
      "total_completed_orders": 1.0,
      "total_open_orders": 0.0,
      "total_rejected_orders": 0.0
    }
  }
}
```

### TradeBook Example

```python
response = client.tradebook()
print(response)
```

TradeBook Response:

```json
{
  "status": "success",
  "data": [
    {
      "action": "BUY",
      "symbol": "RELIANCE",
      "exchange": "NSE",
      "orderid": "250408000989443",
      "product": "MIS",
      "quantity": 0.0,
      "average_price": 1180.1,
      "timestamp": "13:58:03",
      "trade_value": 1180.1
    },
    {
      "action": "SELL",
      "symbol": "NHPC",
      "exchange": "NSE",
      "orderid": "250408001086129",
      "product": "MIS",
      "quantity": 0.0,
      "average_price": 83.74,
      "timestamp": "14:28:49",
      "trade_value": 83.74
    }
  ]
}
```

### PositionBook Example

```python
response = client.positionbook()
print(response)
```

PositionBook Response:

```json
{
  "status": "success",
  "data": [
    {
      "symbol": "NHPC",
      "exchange": "NSE",
      "product": "MIS",
      "quantity": "-1",
      "average_price": "83.74",
      "ltp": "83.72",
      "pnl": "0.02"
    },
    {
      "symbol": "RELIANCE",
      "exchange": "NSE",
      "product": "MIS",
      "quantity": "0",
      "average_price": "0.0",
      "ltp": "1189.9",
      "pnl": "5.90"
    },
    {
      "symbol": "YESBANK",
      "exchange": "NSE",
      "product": "MIS",
      "quantity": "-104",
      "average_price": "17.2",
      "ltp": "17.31",
      "pnl": "-10.44"
    }
  ]
}
```

### Holdings Example

```python
response = client.holdings()
print(response)
```

Holdings Response:

```json
{
  "status": "success",
  "data": {
    "holdings": [
      {"symbol": "RELIANCE",  "exchange": "NSE", "product": "CNC", "quantity": 1, "pnl": -149.0, "pnlpercent": -11.10},
      {"symbol": "TATASTEEL", "exchange": "NSE", "product": "CNC", "quantity": 1, "pnl": -15.0,  "pnlpercent": -10.41},
      {"symbol": "CANBK",     "exchange": "NSE", "product": "CNC", "quantity": 5, "pnl": -69.0,  "pnlpercent": -13.43}
    ],
    "statistics": {
      "totalholdingvalue": 1768.0,
      "totalinvvalue": 2001.0,
      "totalprofitandloss": -233.15,
      "totalpnlpercentage": -11.65
    }
  }
}
```

### Holidays Example

```python
response = client.holidays(year=2026)
print(response)
```

Holidays Response:

```json
{
  "data": [
    {
      "closed_exchanges": ["NSE", "BSE", "NFO", "BFO", "CDS", "BCD", "MCX"],
      "date": "2026-01-26",
      "description": "Republic Day",
      "holiday_type": "TRADING_HOLIDAY",
      "open_exchanges": []
    },
    {
      "closed_exchanges": [],
      "date": "2026-02-19",
      "description": "Chhatrapati Shivaji Maharaj Jayanti",
      "holiday_type": "SETTLEMENT_HOLIDAY",
      "open_exchanges": []
    },
    {
      "closed_exchanges": ["NSE", "BSE", "NFO", "BFO", "CDS", "BCD"],
      "date": "2026-03-10",
      "description": "Holi",
      "holiday_type": "TRADING_HOLIDAY",
      "open_exchanges": [
        {"end_time": 1741677900000, "exchange": "MCX", "start_time": 1741624200000}
      ]
    }
  ]
}
```

### Timings Example

```python
response = client.timings(date="2025-12-19")
print(response)
```

Timings Response:

```json
{
  "data": [
    {"end_time": 1766138400000, "exchange": "NSE", "start_time": 1766115900000},
    {"end_time": 1766138400000, "exchange": "BSE", "start_time": 1766115900000},
    {"end_time": 1766138400000, "exchange": "NFO", "start_time": 1766115900000},
    {"end_time": 1766138400000, "exchange": "BFO", "start_time": 1766115900000},
    {"end_time": 1766168700000, "exchange": "MCX", "start_time": 1766115000000},
    {"end_time": 1766143800000, "exchange": "BCD", "start_time": 1766115000000},
    {"end_time": 1766143800000, "exchange": "CDS", "start_time": 1766115000000}
  ],
  "status": "success"
}
```

### Analyzer Status Example

```python
response = client.analyzerstatus()
print(response)
```

Analyzer Status Response:

```json
{
  "data": {"analyze_mode": true, "mode": "analyze", "total_logs": 2},
  "status": "success"
}
```

### Analyzer Toggle Example

```python
# Switch to analyze mode (simulated responses)
response = client.analyzertoggle(mode=True)
print(response)
```

Analyzer Toggle Response:

```json
{
  "data": {
    "analyze_mode": true,
    "message": "Analyzer mode switched to analyze",
    "mode": "analyze",
    "total_logs": 2
  },
  "status": "success"
}
```

## Strategy Module

OpenAlgo's `/strategy` module runs multi-leg options strategies with end-to-end risk
management, plus a signal-driven mode for TradingView alerts. Two surfaces reach it,
and they take different credentials:

| Surface | Credential | Use for |
|---------|-----------|---------|
| `api(api_key=...)` | Your OpenAlgo API key | Lifecycle and reads: list, status, start, stop, close_all, close_leg, runs, orders, events |
| `Strategy(...)` | The strategy's `oaws_` webhook token | The public webhook at `/strategy/webhook/<token>`, which is what TradingView posts to |

Building a strategy stays in the browser wizard at `/strategy`. The API-key surface is
lifecycle plus reads only: nothing on it can create a strategy, edit its
configuration, enable live trading, rotate a webhook token, or delete anything.

Two strategy kinds, and each refuses the other's vocabulary:

- **batch** - a multi-leg spread entered and exited as a unit. `start` / `stop`.
- **signal** - one alert moves one leg. `long_entry` / `long_exit` / `short_entry` /
  `short_exit`. There is no start and no mode: the first signal after the platform
  session boundary opens the run.

Four rules worth knowing before you call anything:

1. **`mode` on start is required and is never defaulted**, in the SDK or on the
   server. It is a keyword argument with no default, so omitting it is a `TypeError`
   rather than a live order.
2. **Live is opt-in per strategy.** A strategy is created sandbox-only, and
   `mode="live"` is refused with a 409 until the operator enables live trading on the
   strategy page.
3. **An accepted stop is not proof of flatness.** Read `stop_pending` and the per-leg
   outcomes; never infer flatness from the HTTP status.
4. **A strategy that is not yours answers 404**, identical to one that does not exist,
   so the id space cannot be probed.

### StrategyList Example

```python
response = client.strategylist()
print(response)

# Optional filters. An out-of-vocabulary status is a 400, not an empty list.
client.strategylist(status="running")
client.strategylist(q="NIFTY")
```

StrategyList Response:

```json
{
  "status": "success",
  "data": [
    {
      "id": 7,
      "name": "NIFTY Short Straddle",
      "strategy_kind": "batch",
      "direction": "both",
      "underlying": "NIFTY",
      "underlying_exchange": "NSE_INDEX",
      "strategy_type": "intraday",
      "entry_time": "09:20",
      "exit_time": "15:10",
      "product": "NRML",
      "pricetype": "MARKET",
      "overall_sl_mtm": -5000.0,
      "overall_target_mtm": 8000.0,
      "live_enabled": false,
      "status": "running",
      "current_run_id": 42,
      "last_finalized_run": {"id": 41, "pnl_realized": 1250.0, "stopped_at": "2026-08-29T09:40:11.482913+00:00"}
    }
  ]
}
```

The list form omits `legs`; call `strategystatus` for one strategy's legs. For a
stopped strategy, `last_finalized_run.pnl_realized` is the durable final P&L.

### StrategyStatus Example

```python
response = client.strategystatus(strategy_id=7)
print(response)
```

StrategyStatus Response:

```json
{
  "status": "success",
  "data": {
    "id": 7,
    "name": "NIFTY Short Straddle",
    "status": "running",
    "current_run_id": 42,
    "legs": [
      {"id": 1, "segment": "options", "position": "S", "lots": 1, "option_type": "CE",
       "strike_mode": "atm", "atm_offset": "ATM", "expiry": "weekly",
       "sl_pts": 30, "target_pts": 60, "trail": {"x": 10, "y": 5}}
    ]
  },
  "run": {
    "id": 42,
    "mode": "sandbox",
    "broker": "sandbox",
    "started_at": "2026-08-30T03:50:11.402118+00:00",
    "stopped_at": null,
    "stop_reason": null,
    "stop_requested_at": null,
    "stop_requested_reason": null,
    "pnl_realized": 0.0,
    "pnl_peak": 0.0,
    "pnl_trough": 0.0,
    "trigger_source": "manual",
    "resolved_expiries": {"1": "04-SEP-26", "2": "04-SEP-26"}
  }
}
```

`run` is `null` whenever the strategy has no current run, which is the normal state of
a stopped strategy. Prefer it over the strategy's own `status` when you need to know
whether anything is actually open. A populated `stop_requested_reason` means a stop is
durable but not yet confirmed flat: the run is still current and still managed.

### StrategyStart Example

Starts a **batch** strategy: every leg's entry order is placed.

```python
response = client.strategystart(strategy_id=7, mode="sandbox")
print(response)

# Partial success is a 200. Check each leg rather than assuming they all
# reached the market.
for leg in response.get("legs", []):
    if not leg["ok"]:
        print(f"leg {leg['leg_id']} rejected: {leg['error']}")
```

StrategyStart Response:

```json
{
  "status": "success",
  "run_id": 42,
  "mode": "sandbox",
  "legs": [
    {"leg_id": 1, "ok": true, "acknowledged": true,
     "symbol": "NIFTY04SEP2624500CE", "broker_order_id": "26083004118201", "error": null},
    {"leg_id": 2, "ok": true, "acknowledged": true,
     "symbol": "NIFTY04SEP2624500PE", "broker_order_id": "26083004118244", "error": null}
  ]
}
```

`ok: true` with `acknowledged: false` is a real broker order whose id could not be
written back, not a rejection - it reconciles itself. A second start against a running
strategy answers 409, so two triggers firing at once cannot both place a full set of
entries.

### StrategyStop Example

Exits every owned position at market.

```python
response = client.strategystop(strategy_id=7)
print(response)
```

StrategyStop Response:

```json
{
  "status": "success",
  "run_id": 42,
  "stop_pending": true,
  "exits": [
    {"leg_id": 1, "ok": true, "position_ref": "969bc536b1c14d15992f730c2c136d7a",
     "exit_owner": "live", "error": null}
  ]
}
```

`stop_pending: true` means the request is durable and its exits were accepted, but the
run stays open, subscribed and managed until fills prove every position is flat. A 409
can also carry `stop_pending: true` when an unfilled entry or a refused exit still
needs management - retry the stop in that case.

### StrategyCloseAll Example

Same stop mechanics as `strategystop`, different audit intent: a `close_all_manual`
event is written first, which proves an operator asked for a flatten.

```python
response = client.strategycloseall(strategy_id=7)
print(response)
```

### StrategyCloseLeg Example

Exits one leg at market; the run continues with the rest. `leg_id` is the id the
wizard assigned within the strategy, the same value that appears in `legs[].id` on
`strategystatus`. It is not an order id.

```python
response = client.strategycloseleg(strategy_id=7, leg_id=2)
print(response)
```

StrategyCloseLeg Response:

```json
{
  "status": "success",
  "run_id": 42,
  "leg_id": 2,
  "run_stopped": false,
  "exits": [
    {"leg_id": 2, "ok": true, "position_ref": "80bb5fc9333f4922a582229f06a0fe45",
     "exit_owner": "live", "error": null}
  ]
}
```

`run_stopped` reports only what this call could prove. A live broker normally
acknowledges before its fill, so even the last accepted exit returns `false` and the
fill finalises the run later. A `leg_id` that names no open leg is a 409, not a 404.

### StrategyRuns Example

Every activation of a strategy, newest first.

```python
response = client.strategyruns(strategy_id=7, limit=10)
print(response)
```

StrategyRuns Response:

```json
{
  "status": "success",
  "data": [
    {
      "id": 42,
      "strategy_id": 7,
      "mode": "sandbox",
      "broker": "sandbox",
      "started_at": "2026-08-30T03:50:11.402118+00:00",
      "stopped_at": "2026-08-30T09:40:02.771905+00:00",
      "stop_reason": "eod",
      "pnl_realized": 3140.5,
      "pnl_peak": 4880.0,
      "pnl_trough": -1220.25,
      "trigger_source": "manual",
      "resolved_expiries": {"1": "04-SEP-26", "2": "04-SEP-26"}
    }
  ]
}
```

`limit` is 1 to 500 and is bounded rather than clamped: a value outside the range is a
400, so you learn it was refused. An overall threshold triggers an exit, it does not
promise the result - market exits fill at the available bid/ask, so `pnl_realized` can
differ from the threshold that caused the stop.

### StrategyOrders Example

Every order the engine placed, oldest first, so an entry always precedes its exit.

```python
response = client.strategyorders(strategy_id=7)

# Narrow a long history to one run. A run belonging to another strategy matches
# nothing rather than leaking its orders.
response = client.strategyorders(strategy_id=7, run_id=42)
print(response)
```

StrategyOrders Response:

```json
{
  "status": "success",
  "data": [
    {
      "id": 318,
      "run_id": 42,
      "leg_id": 1,
      "kind": "entry",
      "position_ref": "969bc536b1c14d15992f730c2c136d7a",
      "broker_order_id": "26083004118201",
      "symbol": "NIFTY04SEP2624500CE",
      "exchange": "NFO",
      "action": "SELL",
      "qty": 75,
      "product": "NRML",
      "pricetype": "MARKET",
      "price": 0.0,
      "status": "complete",
      "placed_at": "2026-08-30T03:50:11.610224+00:00",
      "filled_at": "2026-08-30T03:50:12.004881+00:00",
      "avg_fill_price": 142.35,
      "filled_qty": 75,
      "reject_reason": null
    }
  ]
}
```

A row is written **before** the broker answers, so an order can appear with
`status: "pending"` and a null `broker_order_id`. That is deliberate: an order that
reached the broker but was never recorded would be invisible to crash recovery.

### StrategyEvents Example

The risk-event audit trail, newest first. The trail is append-only.

```python
response = client.strategyevents(strategy_id=7, limit=100)

# Filters. An out-of-vocabulary kind or severity is a 400, not an empty list.
client.strategyevents(strategy_id=7, run_id=42)
client.strategyevents(strategy_id=7, severity="critical")
client.strategyevents(strategy_id=7, kind="run_stop_failed")
```

StrategyEvents Response:

```json
{
  "status": "success",
  "data": [
    {
      "id": 2041,
      "run_id": 42,
      "strategy_id": 7,
      "ts": "2026-08-30T06:21:40.104112+00:00",
      "kind": "leg_sl_hit",
      "severity": "warn",
      "leg_id": 1,
      "message": "stop loss hit: last price 172.8 is at or above the stop 172.35 on a short position",
      "payload": null
    }
  ]
}
```

Events an operator should not ignore:

| Kind | Severity | Meaning |
|------|----------|---------|
| `run_stop_requested` | info | The stop is durable and new signal entries are gated. Not proof the broker is flat |
| `run_stop_failed` | critical | The broker refused a stop's exits and the run is **still holding** those positions |
| `order_ack_unrecorded` | critical | The broker accepted an order but its acknowledgement could not be written; it reconciles itself |
| `leg_expiry_fallback` | warn | The chain did not list the expiry rank the leg asked for, so a nearer one was used |
| `flip_outgoing_exit_rejected` | critical | The outgoing side of a signal flip is still held |

### Strategy Webhook Example

The public webhook is what TradingView and other alert senders post to. It is not
under `/api/v1` and takes no API key: the `oaws_` token in the URL is the whole
credential. It is shown exactly once, in the browser, when the strategy is created or
its token is rotated - no endpoint returns it. Treat it as a password.

```python
from openalgo import Strategy

strategy = Strategy(
    host_url="http://127.0.0.1:5000",
    webhook_token="oaws_your_webhook_token_here"
)

# Batch strategy: mode is required on start and never defaulted
print(strategy.start("sandbox"))
print(strategy.stop())
```

Webhook Start Response:

```json
{
  "status": "success",
  "result": "ok",
  "message": "Strategy start accepted",
  "strategy_id": 7,
  "run_id": 42
}
```

```python
# Signal strategy: one alert moves one leg. Name the leg by id, or by symbol
# and exchange. leg_id wins when both are given.
strategy.long_entry(leg_id=1)
strategy.long_exit(leg_id=1)
strategy.short_entry(symbol="RELIANCE", exchange="NSE")
strategy.short_exit(symbol="RELIANCE", exchange="NSE")
```

Every documented outcome is **returned, not raised**, because the `result` label is
the contract:

```python
response = strategy.start("sandbox")
result = response.get("result")

if result == "ok":
    print(f"accepted, run {response['run_id']}")
elif result == "rejected_dedupe":
    print("duplicate delivery within 60s, already handled")   # HTTP 200
elif result == "rejected_cooling_off":
    print("stopped within the last 30s, try again shortly")   # HTTP 409
elif result == "rejected_live_disabled":
    print("enable live trading on the strategy page first")   # HTTP 403
```

A signal that does nothing is a **success with a note**, not a failure:
`Signal accepted (already_long)`. The notes are `already_long`, `already_short`,
`no_matching_position`, `outside_entry_window` and `outside_trading_window`. Reporting
a no-op as a failure invites a retry, and a retry on an order path is how one alert
becomes two positions. Being *refused* is different: a signal blocked by the
strategy's direction, or naming a leg that does not exist, answers
`rejected_invalid_action` with the engine's own message.

### LTP Data (Streaming WebSocket)

```python
from openalgo import api
import time

# Initialize OpenAlgo client
client = api(
    api_key="your_api_key",                  # Replace with your actual OpenAlgo API key
    host="http://127.0.0.1:5000",            # REST API host
    ws_url="ws://127.0.0.1:8765"             # WebSocket host
)

# Define instruments to subscribe for LTP
instruments = [
    {"exchange": "NSE", "symbol": "RELIANCE"},
    {"exchange": "NSE", "symbol": "INFY"}
]

# Callback function for LTP updates
def on_ltp(data):
    print("LTP Update Received:")
    print(data)

# Connect and subscribe
client.connect()
client.subscribe_ltp(instruments, on_data_received=on_ltp)

# Run for a few seconds to receive data
try:
    time.sleep(10)
finally:
    client.unsubscribe_ltp(instruments)
    client.disconnect()
```

### Quotes (Streaming WebSocket)

```python
from openalgo import api
import time

client = api(
    api_key="your_api_key",
    host="http://127.0.0.1:5000",
    ws_url="ws://127.0.0.1:8765"
)

instruments = [
    {"exchange": "NSE", "symbol": "RELIANCE"},
    {"exchange": "NSE", "symbol": "INFY"}
]

def on_quote(data):
    print("Quote Update Received:")
    print(data)

client.connect()
client.subscribe_quote(instruments, on_data_received=on_quote)

try:
    time.sleep(10)
finally:
    client.unsubscribe_quote(instruments)
    client.disconnect()
```

### Depth (Streaming WebSocket)

```python
from openalgo import api
import time

client = api(
    api_key="your_api_key",
    host="http://127.0.0.1:5000",
    ws_url="ws://127.0.0.1:8765"
)

instruments = [
    {"exchange": "NSE", "symbol": "RELIANCE"},
    {"exchange": "NSE", "symbol": "INFY"}
]

def on_depth(data):
    print("Market Depth Update Received:")
    print(data)

client.connect()
client.subscribe_depth(instruments, on_data_received=on_depth)

try:
    time.sleep(10)
finally:
    client.unsubscribe_depth(instruments)
    client.disconnect()
```

Every tick carries the fields this SDK documents for its mode, plus any other
field the server sent for that tick under the broker's own key. On an F&O depth
feed that commonly means OI, volume and OHLC alongside the order book:

```python
def on_depth(data):
    tick = data["data"]
    print(tick["ltp"], tick["depth"])
    print(tick.get("oi"), tick.get("volume"))  # present when the broker sends them
```

Read the extras with `.get()`: which ones arrive depends on the broker and the
instrument, so none is guaranteed. The polled snapshots (`get_ltp`, `get_quotes`,
`get_depth`) return the fixed shapes documented on them and are unchanged.

## More Examples

The `examples/` directory in the source repository contains runnable scripts:

- `account_test.py` — account-related functions
- `margin_example.py` — margin calculation for single and multiple positions
- `order_test.py` — order management
- `data_examples.py` — market data
- `feed_examples.py` — WebSocket LTP feeds
- `quote_example.py` — WebSocket quote feeds
- `depth_example.py` — WebSocket market depth feeds
- `options_examples.py` — Options API (Greeks, symbol resolution, orders)
- `telegram_examples.py` — Telegram notification API

## License

MIT — see the [LICENSE](LICENSE) file for details.
