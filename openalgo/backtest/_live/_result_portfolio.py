"""`SessionResult.portfolio`: the strategies of a live session as one account, a
`PortfolioResult` (18-multi-strategy 1.3 and 4.11; unit M5).

- **The strategies** are the session's own `Result`s (`SessionResult.results`, the same
  objects), so the portfolio covers this start only, as they do: after a restart it begins
  at the restart, and the day's figures across restarts stay `SessionResult.summary`'s.
- **The account** is the strategies together on the sum of their capitals, each
  strategy's figures measured on its own capital, as fixed allocations are in a backtest.
  At each instant a strategy's slice closed, the account's equity is every strategy's
  equity at its last slice close (its capital before its first), so it equals
  `metrics.combine_equity` of theirs with their capitals; its margin is the margin their
  positions block as their engines count it, its free cash its equity less that margin (a
  live session trades MIS: no delivery is held and no sale's proceeds are waited for), and
  its daily loss measure their measures added up, each counted from this start: realized
  P&L plus the open P&L at the marks, less the charges on the net `risk_basis`. Each sum is
  rounded to the paisa. After a restart the account's daily limits stay held to the day's
  measure (the restored kernels keep the day's figures), which `SessionResult.summary`
  gives; this account's measure is this start's.
- **The broker's account** holds every strategy: its margin check is the account's, so no
  order is refused for the account's margin here. The account's refusals are those the
  strategies' kernels make for its daily limits, read from their event streams as a
  backtest's fund reads them (`refused_records`): an entry at the account's trade cap,
  `account_trade_limit`, is in `refused` and counted in `contribution`. An entry a
  strategy's kill switch locked out after the account's daily loss limit or profit target
  fired it stays in the strategy's orders as `live_lock`, the session's entry lock, whose
  rejection does not carry the lock's reason (`SessionResult.locks` has it), and the
  working entries and adds the kill switch cancelled are in the strategy's orders as
  cancelled. The Assumptions block says so, with the funds OpenAlgo reported at the start,
  in place of a backtest's pool, and, in a session with an instrument several strategies
  share, what the session does there in place of a backtest's netting rules. When the
  account refused nothing, the combined page's card of the account's refusals says so in
  the words of the limits the session set (`nothing_refused`; a backtest's words when it
  set none), in the card only.

Amounts are rupees; the account's rows carry UTC nanoseconds, shown as IST times.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import reduce
from typing import Any

import numpy as np
import pandas as pd

from .._portfolio_figures import REFUSALS
from .._portfolio_notes import rupees_text
from .._portfolio_tables import ACCOUNT_COLUMNS, STRATEGY
from .._risk_settings import NET_BASIS
from .._strategy_args import PAISE_PER_RUPEE, SHARED_POOL, Account
from ..portfolio_result import PortfolioResult
from ..report._dashboard.portfolio_tables import REFUSED_NOTE
from ..result import Result
from ._groups import SHARED_RULE
from ._plan_args import LiveArguments

#: The account's line of a live session (module docstring): the sum of the capitals, the
#: funds OpenAlgo reported at the start, the broker's margin check, the account's
#: refusals and the start its daily loss measure counts from.
ACCOUNT_NOTE = (
    "the broker's account, which every strategy of the session shares: {capitals} here is "
    "the sum of the strategies' capitals, each strategy's figures measured on its own and "
    "the account's theirs added up; OpenAlgo reported {funds} of funds at the start (the "
    "sandbox's own in SANDBOX mode); the broker's margin check is the account's, so no "
    "order is refused here for the account's margin, and its margin and free cash are the "
    "engines' model; the entries the account's trade cap refused are listed in refused, "
    "while the entries a strategy's kill switch locked out once the account's daily loss "
    "limit or profit target fired it stay in its orders with the code live_lock (the "
    "lock's reason is in SessionResult.locks), and the working entries and adds the kill "
    "switch cancelled stay in its orders as cancelled; the daily loss measure counts this "
    "start only: after a restart the account's daily limits stay held to the day's "
    "measure, whose figures are SessionResult.summary's"
)
#: Where the entries the account's daily loss limit or profit target locked out are, and
#: the kill switch's cancels, as the empty card of the account's refusals says it
#: (`nothing_refused`), the limits the session set named first.
LOCKED_OUT_NOTE = (
    "the entries the account's {limits} locked out are in the strategies' orders with the "
    "code live_lock, and the working entries and adds its kill switch cancelled are in "
    "their orders as cancelled"
)
#: The stamp of a refusal the fund records beside the strategy and the reason: the order
#: in the strategy's own engine and its rejection's slice, slice open (UTC nanoseconds) and
#: step (`_portfolio_tables.refused_frame` reads them).
REFUSAL_STAMP = ("order_id", "slice", "slice_ts", "step")
#: The Assumptions line of a session with an instrument several strategies share, in
#: place of a backtest's netting rules (18-multi-strategy 4.13).
SHARED_NOTE = (
    f"the strategies trading one instrument {SHARED_RULE}; each strategy's charges are "
    "modelled on its own fills, where the broker charges one contract note for the account"
)


def _strategy_rows(result: Result) -> pd.DataFrame:
    """A strategy's figures at its slice closes, by IST time: its equity, free cash,
    margin and daily loss measure counted from this start (module docstring)."""
    rows = result.equity
    measure = rows["realized_rupees"] + rows["unrealized_rupees"]
    if result.manifest["arguments"]["risk"]["risk_basis"] == NET_BASIS:
        measure = measure - rows["costs_rupees"]
    figures = {
        "equity_rupees": rows["equity_rupees"],
        "free_cash_rupees": rows["equity_rupees"] - rows["margin_rupees"],
        "margin_rupees": rows["margin_rupees"],
        "daily_loss_rupees": measure,
    }
    return pd.DataFrame(figures, index=rows.index, columns=list(ACCOUNT_COLUMNS))


def _as_of(frame: pd.DataFrame, times: pd.DatetimeIndex, opening: list[float]) -> np.ndarray:
    """`frame`'s rows at `times` (both in time order): at each time its last row at or
    before it, the `opening` row before its first."""
    values = np.vstack([opening, frame.to_numpy(dtype=float)])
    return values[frame.index.searchsorted(times, side="right")]


def account_rows(results: Mapping[str, Result]) -> dict[str, np.ndarray]:
    """The account's rows of `results` (strategy name to its `Result`, call order), as a
    portfolio run's fund records them: `ts` (UTC nanoseconds of each instant a
    strategy's slice closed, in order) and each of `ACCOUNT_COLUMNS` in rupees, every
    strategy's figure at its last slice close added up in call order (module
    docstring)."""
    frames = [(_strategy_rows(result), result) for result in results.values()]
    times = reduce(pd.Index.union, [frame.index for frame, _ in frames]).unique()
    total = np.zeros((len(times), len(ACCOUNT_COLUMNS)))
    for frame, result in frames:
        capital = float(result.manifest["arguments"]["capital"])
        # Before its first slice close a strategy holds its capital, all of it free.
        opening = {"equity_rupees": capital, "free_cash_rupees": capital}
        total += _as_of(frame, times, [opening.get(name, 0.0) for name in ACCOUNT_COLUMNS])
    paise = np.round(total * PAISE_PER_RUPEE) / PAISE_PER_RUPEE
    stamps = pd.DatetimeIndex(times).as_unit("ns").asi8
    return {"ts": stamps, **{name: paise[:, i] for i, name in enumerate(ACCOUNT_COLUMNS)}}


def refused_records(results: Mapping[str, Result]) -> dict[str, list[Any]]:
    """The orders the account refused among `results` (strategy name to its `Result`, call
    order), as a portfolio run's fund records them: every rejection in a strategy's event
    stream with a code of the account's (`_portfolio_figures.REFUSALS`; in a live session
    `account_trade_limit`, the only one its kernels make), with the strategy's place in
    call order (`strategy`), its `REFUSAL_STAMP` and the code as its `reason`."""
    rows: dict[str, list[Any]] = {STRATEGY: [], **{name: [] for name in REFUSAL_STAMP}}
    rows["reason"] = []
    for place, result in enumerate(results.values()):
        events = result.events
        refused = events[events["reject_code"].isin(REFUSALS)]
        rows[STRATEGY] += [place] * len(refused)
        for name in REFUSAL_STAMP:
            rows[name] += refused[name].astype("int64").tolist()
        rows["reason"] += refused["reject_code"].tolist()
    return rows


def session_account(results: Mapping[str, Result], arguments: LiveArguments) -> Account:
    """The account of a session of `results` with the checked `bt.live` arguments: one
    account every strategy shares, its funds the sum of the capitals, and the account's
    daily limits the call gave."""
    return Account(
        pool=SHARED_POOL,
        account_capital=None,
        account_max_daily_loss=arguments.account_max_daily_loss,
        capitals=tuple(float(r.manifest["arguments"]["capital"]) for r in results.values()),
        account_max_trades_per_day=arguments.account_max_trades_per_day,
        account_daily_profit_target=arguments.account_daily_profit_target,
    )


def nothing_refused(arguments: LiveArguments) -> str | None:
    """The words of the empty card of the account's refusals of a session with the checked
    `bt.live` `arguments` (module docstring): the account's trade cap, when set, refused no
    order, and the entries its daily loss limit or profit target, those set, locked out are
    in the strategies' orders; None when the session set no account limit (the card says
    a backtest's words then)."""
    capped = arguments.account_max_trades_per_day is not None
    limits = [
        name
        for name, given in (
            ("daily loss limit", arguments.account_max_daily_loss),
            ("profit target", arguments.account_daily_profit_target),
        )
        if given is not None
    ]
    if not capped and not limits:
        return None
    words = "the account's trade cap refused no order" if capped else "the account refused no order"
    if limits:
        words += "; " + LOCKED_OUT_NOTE.format(limits=" or ".join(limits))
    return words


def session_portfolio(
    results: Mapping[str, Result],
    arguments: LiveArguments,
    funds_at_start: float,
    *,
    shared: bool,
) -> PortfolioResult:
    """The portfolio of a live session (module docstring): its strategies' `results`
    (name to `Result`, call order), the checked `bt.live` `arguments`, the funds OpenAlgo
    reported at the start in rupees (`funds_at_start`), and whether an instrument was
    shared by several strategies (`shared`). Raises what the strategies' tables raise
    when they cannot be read."""
    account = session_account(results, arguments)
    funds = sum(account.capitals)
    records: dict[str, Any] = {
        "account_capital": funds,
        "account": account_rows(results),
        "refused": refused_records(results),
    }
    words = ACCOUNT_NOTE.format(capitals=rupees_text(funds), funds=rupees_text(funds_at_start))
    notes = {"account": words}
    nothing = nothing_refused(arguments)
    if nothing is not None and not records["refused"][STRATEGY]:
        notes[REFUSED_NOTE] = nothing
    if shared:
        notes["shared_instrument_rules"] = SHARED_NOTE
    return PortfolioResult(account, results, records, notes=notes)


__all__ = [
    "ACCOUNT_NOTE",
    "LOCKED_OUT_NOTE",
    "REFUSAL_STAMP",
    "SHARED_NOTE",
    "account_rows",
    "nothing_refused",
    "refused_records",
    "session_account",
    "session_portfolio",
]
