"""Values and helpers shared by the live journal's tests (`test_journal*.py`)."""

from datetime import date, datetime, timedelta, timezone

from openalgo.backtest._live._types import Mode, OrderIntent, OrderStatus, OrderUpdate, Purpose
from openalgo.backtest._live.journal import Journal

ACCOUNT = "trader1"
BROKER = "zerodha"
DAY = date(2026, 11, 16)
NEXT_DAY = date(2026, 11, 17)
LEG = ("ema", "SBIN", "NSE")
IST = timezone(timedelta(hours=5, minutes=30))


def ist_ns(day: date, hour: int, minute: int = 0) -> int:
    """UTC nanoseconds of an IST wall-clock time."""
    moment = datetime(day.year, day.month, day.day, hour, minute, tzinfo=IST)
    return int(moment.timestamp()) * 1_000_000_000


DAY_NS = ist_ns(DAY, 10)


class ManualClock:
    """UTC nanoseconds that move only when a test sets them."""

    def __init__(self, ns: int = DAY_NS) -> None:
        self.ns = ns

    def __call__(self) -> int:
        return self.ns


def journal_path(tmp_path):
    return tmp_path / "live" / ACCOUNT / "journal.sqlite"


def open_journal(path, *, broker=BROKER, mode=Mode.SANDBOX, account=ACCOUNT, now=None):
    return Journal.open(path, account=account, broker=broker, mode=mode, now=now or ManualClock())


def started_journal(path, *, day=DAY, mode=Mode.SANDBOX, now=None):
    """A journal with the session of `day` started on it."""
    journal = open_journal(path, mode=mode, now=now)
    journal.start_session(day, mode, BROKER)
    return journal


def an_intent(intent_id="i1", **changes):
    values = {
        "intent_id": intent_id,
        "strategy": "ema",
        "symbol": "SBIN",
        "exchange": "NSE",
        "action": "BUY",
        "quantity": 10,
        "price_type": "MARKET",
        "product": "MIS",
        "purpose": Purpose.ENTRY,
        "created_ns": DAY_NS,
    }
    return OrderIntent(**values | changes)


def an_update(order_id="o1", status=OrderStatus.OPEN, **changes):
    values = {
        "order_id": order_id,
        "symbol": "SBIN",
        "exchange": "NSE",
        "action": "BUY",
        "quantity": 10,
        "filled_quantity": 0,
        "pending_quantity": 10,
        "average_price": 0.0,
        "status": status,
        "rejection_reason": "",
        "mode": Mode.SANDBOX,
        "ts_recv_ns": DAY_NS + 1_000,
        "source": "stream",
    }
    return OrderUpdate(**values | changes)


def a_bound_order(journal, intent_id, order_id, **changes):
    journal.record_intent(an_intent(intent_id, **changes))
    journal.bind(intent_id, order_id)


def a_stop_intent(order_id, leg=LEG, **changes):
    """The intent of the stop order `order_id` of `leg` (strategy, symbol, exchange)."""
    strategy, symbol, exchange = leg
    values = {"strategy": strategy, "symbol": symbol, "exchange": exchange, "action": "SELL"}
    stop = {"price_type": "SL-M", "purpose": Purpose.STOP}
    return an_intent(f"intent-of-{order_id}", **values | stop | changes)


def a_stop_order(journal, order_id, leg=LEG, **changes):
    """A bound order sent as the stop of `leg` (strategy, symbol, exchange)."""
    intent = a_stop_intent(order_id, leg, **changes)
    journal.record_intent(intent)
    journal.bind(intent.intent_id, order_id)
