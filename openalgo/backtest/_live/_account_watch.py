"""The account's daily rules in a live session (16-session-design 6.3; 18-multi-strategy
2.5 and 2.5a rules 6 to 8), for every strategy of the session together:

- **The daily loss limit and the daily profit target** (`AccountRules`, run by the risk
  watch, `watch`): in each loop step, after its ticks and bars, the account's measure
  (`account_measure`: the strategies' `daily_loss_measure()` of the session day added up,
  the kernel's measure on each strategy's risk basis, on tick marks) at or past minus
  `account_max_daily_loss` fires the kill switch (`_kill_switch`) of every strategy with
  the reason "account_loss"; at or past `account_daily_profit_target`, with
  "account_profit". Each fires once a session, with an ERROR alert; the kill switch locks
  every strategy's entries for the day (the lock is the reason) and exits what it holds,
  and its exits record the exit reasons `account_loss_limit` and `account_profit_target`.
  The rule (`reached`) is the backtest fund's (`rust/oa_bt/src/fund/rules.rs`), compared
  in whole paise, and one table of cases pins both
  (`test/backtest/data/account_rule_cases.json`, design rule 4).
- **The account's trade cap** (`AccountTrades`, run by the loop before each strategy's
  decisions): the strategy's kernel is told `account_max_trades_per_day` and the entries
  the session's other strategies have counted today (`Engine.set_account_trades`), so an
  entry at the cap is rejected `account_trade_limit` at submission; nothing is exited.

A restart keeps both: each strategy's count of the day comes back with its saved kernel
state (`Engine.restore_live_state`), the entries of a strategy an earlier start of the day
ran and this one does not still count toward the cap (`DayFigures.earlier_entries`), and
the locks "account_loss" and "account_profit" are set again from the saved state
(`_session_saved.DAY_LOCKS`); a measure still past a rule at the restart fires the kill
switch again, which exits what the earlier start left open. The measure is the running
strategies' only, as the account's daily loss limit always took it: a strategy that is no
longer run has no marks. Amounts are rupees. Used by the loop only.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from typing import Any

from .._engine import call
from ..report._format import indian_digits
from ._kill_switch import ACCOUNT_LOSS, ACCOUNT_PROFIT
from ._session_trader import Trader
from ._types import LOGGER_NAME, AlertKind

logger = logging.getLogger(LOGGER_NAME)

#: Paise in one rupee: the rules compare whole paise, as the kernel's do.
PAISE_PER_RUPEE = 100


def _paise(rupees: float) -> int:
    """`rupees` rounded to the nearest paisa, halves away from zero (as the kernel rounds)."""
    return int(math.copysign(math.floor(abs(rupees) * PAISE_PER_RUPEE + 0.5), rupees))


def account_measure(day: Any, engines: Iterable[tuple[float, Any]]) -> float:
    """The account's measure at session day `day` (module docstring): the sum, in call
    order, of each strategy's (measure, the session day of its figures), a measure
    counting only on `day` (18-multi-strategy 2.5, H3), rounded to the paisa. A live
    session's strategies all trade its one day."""
    counted = sum(measure for measure, of in engines if of == day)
    return _paise(counted) / PAISE_PER_RUPEE


def reached(
    measure_rupees: float, limit_rupees: float | None, target_rupees: float | None
) -> str | None:
    """The account's rule the measure reaches: "account_loss" at or past minus the loss
    limit, "account_profit" at or past the profit target, None otherwise or with neither
    set. A measure cannot reach both."""
    measure = _paise(measure_rupees)
    if limit_rupees is not None and measure <= -_paise(limit_rupees):
        return ACCOUNT_LOSS
    if target_rupees is not None and measure >= _paise(target_rupees):
        return ACCOUNT_PROFIT
    return None


#: Each rule's words in its alert: (what it is, the setting it is).
_WORDS = {
    ACCOUNT_LOSS: ("daily loss", "limit"),
    ACCOUNT_PROFIT: ("daily profit", "target"),
}


class AccountRules:
    """The account's daily loss limit and daily profit target of `session`, a
    `_session_loop.Session` (module docstring), firing `kill` (its `KillSwitch`)."""

    def __init__(self, session: Any, kill: Any) -> None:
        self._session = session
        self._kill = kill
        self._fired: set[str] = set()

    def duties(self) -> None:
        """Fire the kill switch of every strategy once a rule is reached (module
        docstring); each rule once."""
        session = self._session
        arguments = session.parts.spec.arguments
        limit, target = arguments.account_max_daily_loss, arguments.account_daily_profit_target
        if limit is None and target is None:
            return
        measures = [float(call(t.engine.daily_loss_measure)) for t in session.traders]
        measure = account_measure(None, ((each, None) for each in measures))
        rule = reached(measure, limit, target)
        if rule is None or rule in self._fired:
            return
        self._fired.add(rule)
        amount = limit if rule == ACCOUNT_LOSS else target
        session.event(rule, measure=measure, limit=amount)
        self._alert(rule, measure, amount)
        for trader in session.traders:
            self._kill.fire(trader, rule)

    def _alert(self, rule: str, measure: float, amount: float) -> None:
        what, setting = _WORDS[rule]
        # After a mode change nothing is sent: the kill switch only alerts (`_kill_switch`).
        exits = (
            "nothing is exited, as OpenAlgo's mode changed: close the positions by hand in OpenAlgo"
            if self._session.locked
            else "the kill switch exits every strategy's position"
        )
        figure = -measure if rule == ACCOUNT_LOSS else measure
        self._session.alert(
            AlertKind.ERROR,
            f"the account's {what} Rs {indian_digits(figure)} is at or past its {setting} Rs "
            f"{indian_digits(amount)}: every strategy's entries are locked for the day and "
            f"{exits}",
        )


class AccountTrades:
    """The account's trade cap of `session`, a `_session_loop.Session` (module
    docstring)."""

    def __init__(self, session: Any) -> None:
        self._session = session

    def before_decisions(self, trader: Trader) -> None:
        """Tell `trader`'s kernel the cap and the other strategies' entries of today,
        those of the strategies only an earlier start of the day ran among them; nothing
        without a cap or for a finished kernel session."""
        session = self._session
        limit = session.parts.spec.arguments.account_max_trades_per_day
        if limit is None or trader.finished:
            return
        others = session.figures.earlier_entries + sum(
            int(call(other.engine.trades_today)) for other in session.traders if other is not trader
        )
        call(trader.engine.set_account_trades, limit, others)


__all__ = ["AccountRules", "AccountTrades", "account_measure", "reached"]
