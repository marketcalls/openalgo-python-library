"""`SessionResult`: what `bt.live` returns when a live session ends.

A frozen dataclass of plain values and pandas DataFrames. Every time is a tz-aware IST
datetime, kept beside the UTC nanoseconds it came from in the `*_ns` columns; prices and
money are rupees, quantities units (never lots). Every table has its columns whether or
not it has rows.

- `mode`: "sandbox" or "live", as OpenAlgo's analyzer switch was when the session started
  (the session adopts it; no argument chooses it). `broker`, `account` and `day` (the IST
  trading day). `started` and `ended`: when this start began and ended, IST.
- `stopped_by`: "session_end" (flat past the square-off), "ctrl_c", "mode_breach" (the
  analyzer switch changed during the session) or "error".
- `restarts`: how many times the session of this day and mode was started before this
  start.
- `strategies`: the strategy names, in call order. `results`: each strategy's
  `bt.Result`, built from its engine by the backtest's own result builder, covering this
  start only: a restarted engine's run begins at the restart, so its equity, metrics and
  gross P&L leave out the realized P&L and charges of the day's earlier starts, which
  its account carried (a position carried open across the restart keeps its whole P&L,
  as its trade does). Its Assumptions block says OpenAlgo priced the fills (no fill
  model, no slippage).
- `summary`: the day's figures across every start of the day: `gross_pnl`, `charges`
  (modelled: the broker's trade book carries none), `net_pnl`, `slippage` (None: not
  measured yet), `closed_trades`, `rejected_orders`, `unprotected_seconds`,
  `reconcile_mismatches`, `undelivered_alerts` and `flat`, and the legs the close names
  (`differing`, `unconfirmed`) and the strategies of earlier starts (`earlier`), in
  words; `summary_text()` gives them as the DAILY_SUMMARY alert sent them.
- `orders`: every order the session placed through OpenAlgo, with its intent ID, broker
  order ID, purpose, how far it got at the broker and its final status. `fills`: every
  fill the engine took, with where it was heard (`source`) and whether its price was
  released unconfirmed (`estimated`) or may lie on another tick (`derived_from_average`).
- `drift`: each broker candle against the bar built from the feed's ticks. `unprotected`:
  each window a position had no confirmed stop at the broker, and why. `locks`: each
  entry lock, when it was set and lifted. `reconciliations`: each comparison with the
  broker's books that differed, and what was done. `skipped_decisions`: the strategy bars
  whose `on_bar` was skipped for missing data.
- `open_positions`: the positions still open at the end, each with its stop at the
  broker, whether that stop was confirmed working, and a note saying what to do. A leg
  the close names (`unconfirmed_legs`: OpenAlgo's books and the engine did not agree on
  it, or no read confirmed it) is there only at a figure the session could carry to the
  close (none for a difference a late read saw): a warning names each, and `flat` is
  False.
- `mode_breach`: the mode guard's text when the analyzer switch changed, else None.
  `warnings`: what the reader must know, the mode first. `paths`: the journal, the day's
  log and the event stream.
- `flat`: True only when the session ended with no position open, a strategy of an
  earlier start included, and no leg the close names.

`RecordSummary`, what `bt.record` returns, is re-exported here for type hints. It loads
on first use, so importing this module loads none of the live layer.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from ._live._alerts_summary import SessionSummary
    from ._live._recorder_summary import RecordSummary
    from .result import Result


@dataclass(frozen=True, eq=False)
class SessionResult:
    """The result of one live session (see the module docstring for every field). Two
    results are equal only when they are the same object: their tables are DataFrames."""

    mode: str
    broker: str
    account: str
    day: dt.date
    started: pd.Timestamp
    ended: pd.Timestamp
    stopped_by: str
    restarts: int
    strategies: tuple[str, ...]
    results: Mapping[str, Result]
    summary: SessionSummary
    orders: pd.DataFrame
    fills: pd.DataFrame
    drift: pd.DataFrame
    unprotected: pd.DataFrame
    locks: pd.DataFrame
    reconciliations: pd.DataFrame
    skipped_decisions: pd.DataFrame
    open_positions: pd.DataFrame
    mode_breach: str | None
    warnings: tuple[str, ...]
    paths: Mapping[str, str]

    @property
    def flat(self) -> bool:
        """True only when the session ended with no position open and the close named no
        leg it could not confirm (`unconfirmed_legs`)."""
        return bool(self.summary.flat) and not self.unconfirmed_legs

    @property
    def unconfirmed_legs(self) -> tuple[str, ...]:
        """The legs the close named, in the summary's words: those whose position
        OpenAlgo's books and the engine did not agree on at the close, and those no read
        confirmed (module docstring, `open_positions`)."""
        return (*self.summary.differing, *self.summary.unconfirmed)

    def summary_text(self) -> str:
        """The day's figures as plain text, as the DAILY_SUMMARY alert sent them: a title
        line with the day, the mode and the broker, then one line per figure."""
        return import_module("._live._alerts_summary", __package__).summary_text(self.summary)

    def __repr__(self) -> str:
        figures = self.summary
        return (
            f"SessionResult({self.mode.upper()} {self.day.isoformat()}, broker "
            f"{self.broker}, stopped by {self.stopped_by}, {figures.closed_trades} closed "
            f"trades, net P&L Rs {figures.net_pnl:,.2f}, "
            f"{len(self.open_positions)} positions open{self._legs_named()})"
        )

    def _legs_named(self) -> str:
        count = len(self.unconfirmed_legs)
        if not count:
            return ""
        return f", {count} {'leg' if count == 1 else 'legs'} not confirmed at the close"


# Names re-exported from the live layer, loaded on first use.
_LIVE_EXPORTS = {"RecordSummary": "._live._recorder_summary"}


def __getattr__(name: str) -> Any:
    if name in _LIVE_EXPORTS:
        value = getattr(import_module(_LIVE_EXPORTS[name], __package__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["RecordSummary", "SessionResult"]
