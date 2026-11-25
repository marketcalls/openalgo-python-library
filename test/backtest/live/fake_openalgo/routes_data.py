"""The data and notification routes: `symbol`, `intervals`, `quotes`, `history`,
`market/holidays` and `telegram/notify`, each a function of the venue and the
request's JSON that returns (body, HTTP status), run under the venue's lock.

Quotes and history come from the broker in either mode, so an expired login fails them
(services/quotes_service.py and history_service.py answer the broker's exception with
HTTP 500, in the reference broker's words: broker/zerodha/api/data.py); an instrument
not in the master contract is refused with HTTP 400 before the broker is asked. `symbol` and
`intervals` read OpenAlgo's own tables. A schema refusal of these routes carries the
field errors as an object (restx_api/symbol.py, quotes.py, history.py). History is
modelled for the broker's 1-minute candles (`source` "api"), and its daily candles made
from them: each IST date's first open, highest high, lowest low, last close and summed
volume, stamped at that date's IST midnight, as the reference broker dates its daily rows.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from openalgo.backtest._layout import NANOS_PER_SECOND

from .answers import EXPIRED_TOKEN, UNKNOWN_SYMBOL, Unmodelled
from .holidays import holidays_answer
from .log import TELEGRAM
from .rows import symbol_row
from .schema import VALID_EXCHANGES
from .venue import TelegramMessage, Venue

Answer = tuple[dict[str, Any], int]

#: utils/constants.py, SUPPORTED_INTERVALS: what `history` accepts.
SUPPORTED_INTERVALS = (
    "1s", "5s", "10s", "15s", "30s", "45s", "1m", "2m", "3m", "5m", "10m", "15m", "20m",
    "30m", "1h", "2h", "3h", "4h", "D", "W", "M", "Q", "Y",
)  # fmt: skip
#: The reference broker's intervals as `intervals` lists them (broker/zerodha/api/
#: data.py, timeframe_map, limited to the supported ones by services/intervals_service.py).
BROKER_INTERVALS = {
    "seconds": [],
    "minutes": ["1m", "3m", "5m", "10m", "15m", "30m"],
    "hours": ["1h"],
    "days": ["D"],
    "weeks": [],
    "months": [],
}
#: The intervals the fake builds candles for: 1-minute, and daily made from them.
CANDLE_INTERVAL = "1m"
DAILY_INTERVAL = "D"
_SECONDS_PER_DAY = 24 * 3600
_IST_OFFSET_S = 19_800
_MISSING = ["Missing data for required field."]
_EXCHANGES_WORDS = [f"Must be one of: {', '.join(VALID_EXCHANGES)}."]
# The broker's exceptions, wrapped by its data handler (broker/zerodha/api/data.py).
_QUOTES_FAILED = "Error fetching quotes: "
_HISTORY_FAILED = "Error fetching historical data: "
_BROKER_TOKEN = f"API Error: {EXPIRED_TOKEN}"
# The priorities the notify route takes, and the one it uses for any other value
# (restx_api/telegram_bot.py, SendNotification).
_TELEGRAM_PRIORITIES = range(1, 11)
_TELEGRAM_DEFAULT_PRIORITY = 5


def symbol(venue: Venue, payload: dict[str, Any]) -> Answer:
    """`symbol` (services/symbol_service.py)."""
    refused = _instrument_fields(payload)
    if refused:
        return refused
    keys = list(venue.markets)
    key = (payload["symbol"], payload["exchange"])
    if key not in venue.markets:
        words = f"Symbol {key[0]} not found in exchange {key[1]}"
        return {"status": "error", "message": words}, 404
    row = symbol_row(venue.markets[key].instrument, keys.index(key) + 1)
    return {"data": row, "status": "success"}, 200


def intervals(venue: Venue, payload: dict[str, Any]) -> Answer:
    """`intervals` (services/intervals_service.py)."""
    del venue, payload  # the broker's fixed list
    return {
        "status": "success",
        "data": {name: list(v) for name, v in BROKER_INTERVALS.items()},
    }, 200


def quotes(venue: Venue, payload: dict[str, Any]) -> Answer:
    """`quotes`: the broker's quote of an instrument (services/quotes_service.py)."""
    refused = _instrument_fields(payload) or _unknown_symbol(venue, payload)
    if refused:
        return refused
    if venue.session_expired:
        return {"status": "error", "message": _QUOTES_FAILED + _BROKER_TOKEN}, 500
    market = venue.markets[(payload["symbol"], payload["exchange"])]
    data = {
        "ask": market.best_ask,
        "bid": market.best_bid,
        "high": market.day_high,
        "low": market.day_low,
        "ltp": market.last_price or 0.0,
        "open": market.day_open,
        "prev_close": market.prev_close,
        "volume": int(market.day_volume),
        "oi": 0,
    }
    return {"status": "success", "data": data}, 200


def history(venue: Venue, payload: dict[str, Any]) -> Answer:
    """`history`: the broker's 1-minute candles of the IST dates asked for, both
    included, the forming candle with them; each row stamped at its start in epoch
    seconds (broker/zerodha/api/data.py, get_history)."""
    errors = _history_errors(payload)
    if errors:
        return {"status": "error", "message": errors}, 400
    interval = payload["interval"]
    if payload.get("source", "api") != "api" or interval not in (CANDLE_INTERVAL, DAILY_INTERVAL):
        raise Unmodelled(
            f"history is modelled for intervals {CANDLE_INTERVAL!r} and {DAILY_INTERVAL!r} "
            f"from source 'api', not {interval!r} from {payload.get('source')!r}"
        )
    refused = _unknown_symbol(venue, payload)
    if refused:
        return refused
    if venue.session_expired:
        return {"status": "error", "message": _HISTORY_FAILED + _BROKER_TOKEN}, 500
    market = venue.markets[(payload["symbol"], payload["exchange"])]
    rows = [
        {
            "timestamp": candle.start_ns // NANOS_PER_SECOND,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": int(candle.volume),
            "oi": 0,
        }
        for candle in market.candles(payload["start_date"], payload["end_date"])
    ]
    if interval == DAILY_INTERVAL:
        rows = _daily(rows)
    return {"status": "success", "data": rows}, 200


def _daily(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The daily candles of 1-minute `rows` (module docstring), oldest first."""
    days: dict[int, dict[str, Any]] = {}
    for row in rows:
        midnight = (row["timestamp"] + _IST_OFFSET_S) // _SECONDS_PER_DAY * _SECONDS_PER_DAY
        stamp = midnight - _IST_OFFSET_S
        day = days.get(stamp)
        if day is None:
            days[stamp] = {**row, "timestamp": stamp}
            continue
        day["high"] = max(day["high"], row["high"])
        day["low"] = min(day["low"], row["low"])
        day["close"] = row["close"]
        day["volume"] += row["volume"]
    return [days[stamp] for stamp in sorted(days)]


def holidays(venue: Venue, payload: dict[str, Any]) -> Answer:
    """`market/holidays` (services/market_calendar_service.py): the rows of the year
    asked for, from OpenAlgo's own table; the server serves 2020 to 2050."""
    year = payload.get("year")
    if not isinstance(year, int) or not 2020 <= year <= 2050:
        return {"status": "error", "message": "Year must be between 2020 and 2050"}, 400
    return holidays_answer(year, venue.holidays), 200


def telegram_notify(venue: Venue, payload: dict[str, Any]) -> Answer:
    """`telegram/notify` (restx_api/telegram_bot.py, SendNotification): queued for
    delivery, or sent when `wait_for_delivery` is true."""
    if not venue.telegram_bot_active:
        words = "Telegram bot is stopped. Start the bot to send notifications."
        return {"status": "error", "message": words}, 409
    username, message = payload.get("username"), payload.get("message")
    if not username or not message:
        return {"status": "error", "message": "Username and message are required"}, 400
    if username not in venue.telegram_linked:
        return {"status": "error", "message": "User not found or not linked to Telegram"}, 404
    try:
        priority = int(payload.get("priority", _TELEGRAM_DEFAULT_PRIORITY))
    except (TypeError, ValueError):
        priority = _TELEGRAM_DEFAULT_PRIORITY
    if priority not in _TELEGRAM_PRIORITIES:
        priority = _TELEGRAM_DEFAULT_PRIORITY
    now = venue.clock.now()
    venue.telegram_sent.append(TelegramMessage(username, message, priority, now))
    venue.log.record(now, TELEGRAM, username, message=message, priority=priority)
    if payload.get("wait_for_delivery"):
        return {"status": "success", "message": "Notification sent successfully"}, 200
    return {"status": "success", "message": "Notification queued for delivery"}, 200


def _unknown_symbol(venue: Venue, payload: dict[str, Any]) -> Answer | None:
    """The 400 of quotes and history for an instrument not in the master contract,
    before the broker is asked (services/quotes_service.py:118-120,
    services/history_service.py:132-134)."""
    symbol, exchange = payload["symbol"], payload["exchange"]
    if venue.market(symbol, exchange) is not None:
        return None
    words = UNKNOWN_SYMBOL.format(symbol=symbol, exchange=exchange)
    return {"status": "error", "message": words}, 400


def _instrument_fields(payload: dict[str, Any]) -> Answer | None:
    errors: dict[str, list[str]] = {}
    for name in ("symbol", "exchange"):
        if not isinstance(payload.get(name), str):
            errors[name] = _MISSING if name not in payload else ["Not a valid string."]
    if "exchange" not in errors and payload["exchange"] not in VALID_EXCHANGES:
        errors["exchange"] = _EXCHANGES_WORDS
    return ({"status": "error", "message": errors}, 400) if errors else None


def _history_errors(payload: dict[str, Any]) -> dict[str, list[str]]:
    refused = _instrument_fields(payload)
    errors: dict[str, list[str]] = dict(refused[0]["message"]) if refused else {}
    interval = payload.get("interval")
    if interval is None:
        errors["interval"] = _MISSING
    elif interval not in SUPPORTED_INTERVALS:
        errors["interval"] = [f"Must be one of: {', '.join(SUPPORTED_INTERVALS)}."]
    for name in ("start_date", "end_date"):
        if name not in payload:
            errors[name] = _MISSING
            continue
        try:
            dt.date.fromisoformat(str(payload[name]))
        except ValueError:
            errors[name] = ["Not a valid date."]
    return errors


__all__ = [
    "BROKER_INTERVALS",
    "CANDLE_INTERVAL",
    "SUPPORTED_INTERVALS",
    "history",
    "intervals",
    "quotes",
    "symbol",
    "telegram_notify",
]
