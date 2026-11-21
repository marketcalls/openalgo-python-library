"""The combined tables of a portfolio run: every strategy's table in one, the account's rows,
the orders the account refused and the net position of each instrument over time.

Conventions are `Result`'s (IST times beside the kernel's UTC nanosecond `*_ts` columns,
rupees, units) with a `strategy` column first. Kernel ids (order ids, trade ids, event
sequence numbers) restart in every strategy's engine, so a combined row is identified by
(`strategy`, the id), never by the id alone. Rows are in time order, rows of one instant
in call order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._result_tables import ist_times
from ._trade_list import STRATEGY
from .result import Result

#: How each combined table is put in time order: the columns of its sort key, the kernel's
#: UTC nanoseconds and the step of the slice's sequence where a table has them. Rows of
#: one key keep their call order (a stable sort of the strategies' tables in call order).
TIME_ORDER: Mapping[str, tuple[str, ...]] = {
    "fills": ("slice_ts", "step"),
    "orders": ("time",),
    "trades": ("exit_ts",),
    "events": ("slice_ts", "step"),
    "costs": ("time",),
    "contract_notes": ("date",),
    "final_positions": (),
}
#: The columns of the account's rows (`Fund.finish`), rupees, as `PortfolioResult.account`
#: has them.
ACCOUNT_COLUMNS = ("equity_rupees", "free_cash_rupees", "margin_rupees", "daily_loss_rupees")
#: The columns of `refused`, in order.
REFUSED_COLUMNS = (
    STRATEGY,
    "order_id",
    "symbol",
    "side",
    "time",
    "slice",
    "slice_ts",
    "step",
    "reason",
)


def stacked(frames: Sequence[pd.DataFrame], names: Sequence[str]) -> pd.DataFrame:
    """The strategies' `frames` one after another (call order), with a `strategy` column
    first naming the strategy of each row; the columns of the first frame when every
    frame is empty."""
    filled = [(frame, name) for frame, name in zip(frames, names, strict=True) if len(frame)]
    if not filled:
        joined = frames[0].iloc[0:0].copy()
        joined.insert(0, STRATEGY, pd.Series(dtype=object))
        return joined.reset_index(drop=True)
    joined = pd.concat([frame for frame, _ in filled], ignore_index=True)
    labels = np.repeat([name for _, name in filled], [len(frame) for frame, _ in filled])
    joined.insert(0, STRATEGY, labels)
    return joined


def combined(results: Mapping[str, Result], table: str) -> pd.DataFrame:
    """Every strategy's `table` (a `Result` table named in `TIME_ORDER`) in one, with a
    `strategy` column, in time order (rows of one instant in call order)."""
    frame = stacked([getattr(result, table) for result in results.values()], list(results))
    order = list(TIME_ORDER[table])
    if order and len(frame):
        frame = frame.sort_values(order, kind="stable", ignore_index=True)
    return frame


def combined_trade_list(results: Mapping[str, Result]) -> pd.DataFrame:
    """Every strategy's `trade_list()` in one, indexed by (strategy, trade), in the order
    the trades closed (trades closing at one instant in call order)."""
    lists = [result.trade_list() for result in results.values()]
    frame = stacked([each.reset_index() for each in lists], list(results))
    if len(frame):
        frame = frame.sort_values("exit_time", kind="stable", ignore_index=True)
    return frame.set_index([STRATEGY, "trade"])


def account_frame(records: Mapping[str, Any]) -> pd.DataFrame:
    """The account's row of every round of the fund, indexed by the IST close of the
    round (`time`): its equity, free cash, margin and daily loss measure, rupees."""
    rows = records["account"]
    index = pd.DatetimeIndex(ist_times(rows["ts"]), name="time")
    return pd.DataFrame({name: rows[name] for name in ACCOUNT_COLUMNS}, index=index)


def refused_frame(records: Mapping[str, Any], results: Mapping[str, Result]) -> pd.DataFrame:
    """The orders the account refused or ended (`REFUSED_COLUMNS`): the strategy, its
    kernel order id, the order's symbol and side, when (the IST open of the slice of the
    refusal, its slice and step) and why ("account_margin", "account_loss_gate",
    "account_profit_gate", "account_trade_limit", or the lock "account_loss" or
    "account_profit" for a working order an account breach cancelled), in time order (one
    instant's in call order, then by order id)."""
    names = list(results)
    raw = pd.DataFrame(records["refused"])
    if not len(raw):
        return pd.DataFrame({name: pd.Series(dtype=object) for name in REFUSED_COLUMNS})
    raw = raw.sort_values(["slice_ts", "step", STRATEGY, "order_id"], kind="stable")
    orders = {name: result.orders.set_index("order_id") for name, result in results.items()}
    strategies = [names[index] for index in raw[STRATEGY]]
    ids = raw["order_id"]
    found = [orders[name].loc[order] for name, order in zip(strategies, ids, strict=True)]
    frame = pd.DataFrame(
        {
            STRATEGY: strategies,
            "order_id": raw["order_id"].to_numpy(),
            "symbol": [order["symbol"] for order in found],
            "side": [order["side"] for order in found],
            "time": ist_times(raw["slice_ts"]),
            "slice": raw["slice"].to_numpy(),
            "slice_ts": raw["slice_ts"].to_numpy(),
            "step": raw["step"].to_numpy(),
            "reason": raw["reason"].to_numpy(),
        }
    )
    return frame.reset_index(drop=True)


@dataclass(frozen=True)
class OppositeSides:
    """The first instant two strategies held one instrument and product on opposite
    sides: the instrument (`symbol`, `exchange`, `product`), the strategies long and
    short then (call order) and the IST open of that slice (`time`)."""

    symbol: str
    exchange: str
    product: str
    long: tuple[str, ...]
    short: tuple[str, ...]
    time: pd.Timestamp


def _position_changes(results: Mapping[str, Result]) -> pd.DataFrame:
    """Every strategy's position history (`Result.positions`) as changes of units: the
    slice open (`slice_ts`), the strategy's call position (`order`), its name, the
    instrument (symbol, exchange, product) and its signed units from then on."""
    frames = []
    for order, (name, result) in enumerate(results.items()):
        levels = result.positions
        exchanges = [entry["exchange"] for entry in result.manifest["instruments"]]
        frames.append(
            pd.DataFrame(
                {
                    "slice_ts": levels["slice_ts"].to_numpy(dtype=np.int64),
                    "order": order,
                    STRATEGY: name,
                    "symbol": levels["symbol"].to_numpy(),
                    "exchange": [exchanges[index] for index in levels["instrument"]],
                    "product": levels["product"].to_numpy(),
                    "units": levels["quantity_units"].to_numpy(dtype=float),
                }
            )
        )
    changes = pd.concat(frames, ignore_index=True)
    return changes.sort_values(["slice_ts", "order"], kind="stable", ignore_index=True)


def _units_column(name: str) -> str:
    return f"{name}_units"


def net_positions(results: Mapping[str, Result]) -> tuple[pd.DataFrame, list[OppositeSides]]:
    """(the net position of each instrument and product over time, the instruments two
    strategies held on opposite sides, each at the first instant they did).

    The table has a row for each slice in which the units a strategy held of an
    instrument and product changed: `time` (the IST open of the slice) and `slice_ts`,
    `symbol`, `exchange` and `product`, `net_units` (the strategies' signed units added
    up: what a broker holds, netted), `holders` (the strategies holding units then, in
    call order, joined by ", ") and a `<strategy>_units` column per strategy, from then
    on, in the order of the instants (one instant's instruments in the order they first
    changed)."""
    names = list(results)
    changes = _position_changes(results)
    held: dict[tuple[str, str, str], dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    sides: dict[tuple[str, str, str], OppositeSides] = {}
    for slice_ts, moment in changes.groupby("slice_ts", sort=True):
        touched: list[tuple[str, str, str]] = []
        for change in moment.itertuples(index=False):
            key = (change.symbol, change.exchange, change.product)
            units = held.setdefault(key, dict.fromkeys(names, 0.0))
            if units[change.strategy] != change.units and key not in touched:
                touched.append(key)
            units[change.strategy] = change.units
        for key in touched:
            units = held[key]
            rows.append(_net_row(key, int(slice_ts), units))
            long = tuple(name for name in names if units[name] > 0)
            short = tuple(name for name in names if units[name] < 0)
            if long and short and key not in sides:
                sides[key] = OppositeSides(*key, long, short, ist_times([slice_ts])[0])
    columns = ["slice_ts", "symbol", "exchange", "product", "net_units", "holders"]
    frame = pd.DataFrame(rows, columns=[*columns, *map(_units_column, names)])
    frame.insert(0, "time", ist_times(frame["slice_ts"].astype("int64")))
    return frame, list(sides.values())


def _net_row(
    key: tuple[str, str, str], slice_ts: int, units: Mapping[str, float]
) -> dict[str, Any]:
    """One row of `net_positions` for the instrument `key` at `slice_ts`."""
    symbol, exchange, product = key
    return {
        "slice_ts": slice_ts,
        "symbol": symbol,
        "exchange": exchange,
        "product": product,
        "net_units": sum(units.values()),
        "holders": ", ".join(name for name, value in units.items() if value != 0.0),
        **{_units_column(name): value for name, value in units.items()},
    }


__all__ = [
    "ACCOUNT_COLUMNS",
    "REFUSED_COLUMNS",
    "STRATEGY",
    "TIME_ORDER",
    "OppositeSides",
    "account_frame",
    "combined",
    "combined_trade_list",
    "net_positions",
    "refused_frame",
    "stacked",
]
