"""The arguments of `bt.live`, checked before anything is called (16-session-design 1.2
and 3.1 step 1; decisions 3 and 9 of its section 10.0; unit 7). Part of `plan`.

- One strategy class or a list of them (decision 3). Every per-strategy setting a session
  takes is one value for every strategy or a dict by strategy class name (18-multi-
  strategy 1.3, decision M5): `capital`, `max_daily_loss`, the risk settings
  (`_strategy_args.RISK_SETTINGS`: `after_risk_exit`, `max_positions`, `drawdown_halt`,
  `cooldown`, `entry_from`, `entry_until`, `square_off`) and the daily limits; with
  several strategies `params` is a dict of each one's parameter values by class name (one
  strategy takes its own values, as in `bt.backtest`). The rules of a dict by name are
  `_strategy_args`', the ones `bt.backtest` follows for a list of strategies; a class
  attribute of the same name still wins, as in a backtest.
- No argument chooses or confirms the mode: sandbox or live is the analyzer switch inside
  OpenAlgo, which the session adopts (D21). An argument that looks like a mode choice is
  refused, saying so.
- A `bt.backtest` argument that has no meaning live is refused, naming why (`NOT_LIVE`),
  and so is any other argument `bt.live` does not take. `bars="websocket"`, the one bar
  source of this release, is accepted; `bars="candles"` is not built yet.
- The account the journal, logs and locks are kept under defaults to the broker name
  `ping` reports (decision 9; `LiveArguments.account_name`).
- The settings a live session shares with `bt.backtest` take `bt.backtest`'s own
  defaults, read from its signature (`BACKTEST_DEFAULT`), so the two cannot drift
  apart. `capital` and `max_daily_loss` are mandatory in the call, for each strategy; a
  class attribute `max_daily_loss` must agree with the call's (`_plan_strategies`).
- The daily limits of 18-multi-strategy 2.5a (`DAILY_LIMITS`: `max_trades_per_day`,
  `daily_profit_target`, `max_order_value`) take one value for every strategy or a dict
  naming each by class name, as in `bt.backtest` given a list; the account's
  (`account_max_trades_per_day`, `account_daily_profit_target`) one value for the session.
  `max_orders_per_second` is the session's order rate (rule 12, `_order_throttle`).

Every other value is checked where `bt.backtest` checks it, when the plan resolves each
strategy's settings (`_plan_strategies`), so a live session and a backtest refuse the
same values with the same words. Amounts are rupees.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .._messages import not_supported
from .._products import MIS
from .._risk_settings import rupees_amount
from .._strategy_args import (
    MANDATORY,
    account_trade_cap,
    by_name,
    by_strategy,
    class_names,
    params_by_name,
    repeated,
    strategy_classes,
)
from ..errors import ConfigError
from ..strategy import Strategy
from ._order_rate import DEFAULT_ORDERS_PER_SECOND
from ._preflight_plan import positive_amount
from .data_worker import HISTORY_SOURCE
from .series import DEFAULT_BASE

#: What `on_stop` takes: flatten every leg before stopping, or keep the legs open with
#: their broker-held stops (16-session-design 3.3, decision 1 of its section 10.0), or
#: None, the default: flatten the MIS legs and keep the CNC and NRML ones with their GTTs
#: (20-positional-live 10.2, decision 11.8; an MIS-only session flattens every leg).
ON_STOP = (None, "flatten", "keep")


def kept_on_stop(on_stop: str | None, product: str) -> bool:
    """Whether a Ctrl+C keeps open a position of `product` ("MIS", "CNC", "NRML") with its
    broker stop under `on_stop` (`ON_STOP`), rather than exiting it."""
    if on_stop is None:
        return product != MIS
    return on_stop == "keep"


#: The bar source of this release: bars built from the feed's ticks (08-live-trading
#: section 5); bars from broker candles come later.
BARS_FROM_TICKS = "websocket"
BARS_FROM_CANDLES = "candles"
#: The call a dict by strategy name is checked against, as its messages name it.
SESSION = "this session"
#: Names a caller might use to choose or confirm the mode in code, which `bt.live` never
#: takes: the analyzer switch inside OpenAlgo is the only control (D21).
MODE_ARGUMENTS = frozenset(
    {"mode", "live", "sandbox", "paper", "analyzer", "analyze", "confirm", "dry_run"}
)
#: The `bt.backtest` arguments with no meaning in a live session, and why (16-session-
#: design 1.2): the broker prices fills, computes margin and gives the contract facts,
#: and the session trades today, on broker data.
NOT_LIVE = MappingProxyType(
    {
        "data": "the session trades on the broker's data",
        "start": "the session trades today",
        "end": "the session trades today",
        "now": "the session runs on the clock",
        "source": "the session reads its history from the broker",
        "tz": "the session reads no data of its own",
        "corporate_actions": "the session reads the broker's history as the broker gives it",
        "base": "a live session's bars are built on 1-minute slices",
        "fill": "the broker fills the orders",
        "delay_entries": "a live session sends each order when it is decided",
        "delay_exits": "a live session sends each order when it is decided",
        "slippage": "the broker prices the fills",
        "margin": "the broker computes the margin",
        "quantity_mode": "a live session trades whole shares and lots",
        "instrument": "OpenAlgo gives the lot size, tick size and freeze quantity",
        "product": "set product on each strategy class (MIS by default, CNC or NRML to hold "
        "positions overnight)",
        "roll": "a live session trades one contract named by its symbol",
        "benchmark": "a live session has no benchmark",
        "calendar": "a live session follows the package calendar",
        "pool": "the broker's account is the one account every strategy of the session shares",
        "account_capital": "the account's funds are the broker's",
    }
)


class _BacktestDefault:
    """The default of a setting a live session shares with `bt.backtest`: the one
    `bt.backtest`'s signature gives it, read when the settings are resolved."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<bt.backtest's default>"


#: Stands in the signature of `live_arguments` for `bt.backtest`'s default of a shared
#: setting (module docstring).
BACKTEST_DEFAULT: Any = _BacktestDefault()


def _backtest_defaults() -> dict[str, Any]:
    """`bt.backtest`'s keyword arguments with their defaults: one source of the defaults
    a live session shares with a backtest. `bt.backtest` is imported here rather than
    with the module, so the run module can import the live session without a cycle."""
    from ..run import backtest

    return {
        name: parameter.default
        for name, parameter in inspect.signature(backtest).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


@dataclass(frozen=True)
class LiveArguments:
    """The checked arguments of one `bt.live` call (module docstring).

    `strategies` are the strategy classes in call order and `names` their class names.
    `capital`, `max_daily_loss` and `params` are each strategy's values as given, by
    name (`max_daily_loss` None where only the class may set it). `settings` are the
    `bt.backtest` settings of the session that the call gave (`costs`, `risk_basis`,
    `round_levels`, `strict_config`: one value for every strategy) and `risk` each
    strategy's risk settings the call gave (`_strategy_args.RISK_SETTINGS`), by name; a
    strategy's class attributes of the same names override them as in a backtest, and
    the settings not
    given take `bt.backtest`'s defaults (`backtest_arguments`). The rest are the session's
    own: `account_max_daily_loss` in rupees, `alerts_to`, `account`, `live_dir` and
    `on_stop`; `daily_limits` each strategy's daily limits by name (`DAILY_LIMITS`, None
    where not set), the account's `account_max_trades_per_day` and
    `account_daily_profit_target` (rupees), and `max_orders_per_second`, the order rate.
    """

    strategies: tuple[type[Strategy], ...]
    names: tuple[str, ...]
    capital: Mapping[str, Any]
    max_daily_loss: Mapping[str, Any]
    params: Mapping[str, Mapping[str, Any] | None]
    settings: Mapping[str, Any]
    risk: Mapping[str, Mapping[str, Any]]
    account_max_daily_loss: float | None
    alerts_to: str | None
    account: str | None
    live_dir: str | os.PathLike[str] | None
    on_stop: str | None
    daily_limits: Mapping[str, Mapping[str, Any]]
    account_max_trades_per_day: int | None
    account_daily_profit_target: float | None
    max_orders_per_second: int

    def backtest_arguments(self, name: str, day: Any, present_ns: int) -> dict[str, Any]:
        """The arguments of the backtest of strategy `name` on the IST date `day` with
        the present `present_ns` (UTC nanoseconds), as `bt.backtest` would take them:
        its defaults, the session's own settings, and the live terms (start and end
        today, broker history, a 1-minute base). Raises KeyError for an unknown name."""
        return {
            **_backtest_defaults(),
            **self.settings,
            **self.risk[name],
            "capital": self.capital[name],
            "max_daily_loss": self.max_daily_loss[name],
            "params": self.params[name],
            **self.daily_limits[name],
            "start": day,
            "end": day,
            "now": present_ns,
            "source": HISTORY_SOURCE,
            "base": DEFAULT_BASE,
        }

    def account_name(self, broker: str) -> str:
        """The account the journal, logs and locks are kept under: `account`, else the
        broker name `ping` reported (`broker`)."""
        return self.account if self.account is not None else broker


def _refuse_not_live(not_live: Mapping[str, Any]) -> None:
    """Raise for every argument `bt.live` does not take (module docstring): ConfigError,
    or NotImplementedError for `bars="candles"`."""
    taken = sorted(set(inspect.signature(live_arguments).parameters) - {"strategy", "not_live"})
    for name, value in not_live.items():
        if name in MODE_ARGUMENTS:
            raise ConfigError(
                f"bt.live takes no {name}=: sandbox or live is chosen inside OpenAlgo with "
                "its analyzer switch, and the session adopts the mode it finds at the start; "
                f"remove {name}="
            )
        if name in NOT_LIVE:
            raise ConfigError(f"bt.live takes no {name}=: {NOT_LIVE[name]}; remove {name}=")
        if name != "bars":
            raise ConfigError(
                f"bt.live has no argument {name}=; it takes {', '.join(taken)}, and bars"
            )
        if value == BARS_FROM_CANDLES:
            raise not_supported(
                "bars='candles' (bars from broker candles)",
                f"use bars='{BARS_FROM_TICKS}', the bars built from the feed's ticks",
            )
        if value != BARS_FROM_TICKS:
            raise ConfigError(f"bars takes '{BARS_FROM_TICKS}', not {value!r}")


def _names(strategies: Sequence[type[Strategy]]) -> tuple[str, ...]:
    names = class_names(strategies)
    twice = repeated(names)
    if twice:
        raise not_supported(
            f"two strategies with the same name ({', '.join(twice)}) in one session",
            "give each strategy class its own name: OpenAlgo shows it on every order",
        )
    return names


def _session_options(
    account_max_daily_loss: Any, alerts_to: Any, account: Any, live_dir: Any, on_stop: Any
) -> None:
    if account_max_daily_loss is not None and positive_amount(account_max_daily_loss) is None:
        raise ConfigError(
            "account_max_daily_loss takes a positive number of rupees for all strategies "
            f"together, or None, not {account_max_daily_loss!r}"
        )
    for name, value in (("alerts_to", alerts_to), ("account", account)):
        if value is not None and not (isinstance(value, str) and value.strip()):
            raise ConfigError(f"{name} takes a name (a non-blank string) or None, not {value!r}")
    if live_dir is not None and not isinstance(live_dir, str | os.PathLike):
        raise ConfigError(f"live_dir takes a folder path or None, not {live_dir!r}")
    if on_stop not in ON_STOP:
        raise ConfigError(
            "on_stop takes None (the default: exit the MIS positions and keep the CNC and "
            "NRML ones with their GTT stops), 'flatten' (exit every position, then stop) or "
            f"'keep' (leave the positions with their broker stops), not {on_stop!r}"
        )


def _orders_per_second(value: Any) -> int:
    """`max_orders_per_second` checked: a whole number of order calls a second, at least
    1. Raises ConfigError otherwise."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(
            "max_orders_per_second takes a whole number of order calls a second, at least 1 "
            f"({DEFAULT_ORDERS_PER_SECOND} by default: OpenAlgo's 10 less 2 for your own "
            f"orders), not {value!r}"
        )
    return value


def _given(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The settings the call gave: those not left at `bt.backtest`'s default."""
    return {name: value for name, value in settings.items() if value is not BACKTEST_DEFAULT}


def live_arguments(
    strategy: type[Strategy] | Sequence[type[Strategy]],
    *,
    capital: float | Mapping[str, float] | None = None,
    max_daily_loss: Any = None,
    account_max_daily_loss: float | None = None,
    alerts_to: str | None = None,
    account: str | None = None,
    params: Mapping[str, Any] | None = None,
    costs: Any = BACKTEST_DEFAULT,
    risk_basis: str = BACKTEST_DEFAULT,
    after_risk_exit: str | Mapping[str, str] = BACKTEST_DEFAULT,
    entry_from: str | Mapping[str, str | None] | None = BACKTEST_DEFAULT,
    entry_until: str | Mapping[str, str | None] | None = BACKTEST_DEFAULT,
    square_off: str | Mapping[str, str | None] | None = BACKTEST_DEFAULT,
    max_positions: int | Mapping[str, int | None] | None = BACKTEST_DEFAULT,
    drawdown_halt: Any = BACKTEST_DEFAULT,
    cooldown: int | Mapping[str, int | None] | None = BACKTEST_DEFAULT,
    round_levels: bool = BACKTEST_DEFAULT,
    strict_config: bool = BACKTEST_DEFAULT,
    live_dir: str | os.PathLike[str] | None = None,
    on_stop: str | None = None,
    max_trades_per_day: Any = None,
    daily_profit_target: Any = None,
    max_order_value: Any = None,
    account_max_trades_per_day: int | None = None,
    account_daily_profit_target: Any = None,
    max_orders_per_second: int = DEFAULT_ORDERS_PER_SECOND,
    **not_live: Any,
) -> LiveArguments:
    """Check the arguments of a `bt.live` call (16-session-design 1.2; module docstring)
    and return them by strategy.

    Raises ConfigError for an argument `bt.live` does not take (a mode choice, a
    `bt.backtest` argument with no meaning live, an unknown name), for a strategy that is
    not a `bt.Strategy` subclass, for no `capital`, for a dict of values keyed by a name
    the session does not run or, for `capital`, the risk settings and the daily limits,
    missing one, and for an unusable `account_max_daily_loss`,
    `account_max_trades_per_day`, `account_daily_profit_target`, `max_orders_per_second`,
    `alerts_to`, `account`, `live_dir` or `on_stop`; NotImplementedError for
    `bars="candles"` and for two strategies with the same name."""
    _refuse_not_live(not_live)
    strategies = strategy_classes(strategy, "bt.live")
    names = _names(strategies)
    if capital is None:
        raise ConfigError(
            "bt.live needs capital=: the rupees allocated to each strategy, one number for "
            "every strategy or a dict by strategy class name (a risk limit, mandatory live)"
        )
    _session_options(account_max_daily_loss, alerts_to, account, live_dir, on_stop)
    session = {
        "costs": costs,
        "risk_basis": risk_basis,
        "round_levels": round_levels,
        "strict_config": strict_config,
    }
    risk = {
        "after_risk_exit": after_risk_exit,
        "entry_from": entry_from,
        "entry_until": entry_until,
        "square_off": square_off,
        "max_positions": max_positions,
        "drawdown_halt": drawdown_halt,
        "cooldown": cooldown,
    }
    return LiveArguments(
        strategies=strategies,
        names=names,
        capital=by_name(capital, names, "capital", owner=SESSION, required=MANDATORY),
        max_daily_loss=by_name(max_daily_loss, names, "max_daily_loss", owner=SESSION),
        params=params_by_name(params, names, owner=SESSION),
        settings=_given(session),
        risk=by_strategy(_given(risk), names, owner=SESSION),
        account_max_daily_loss=positive_amount(account_max_daily_loss),
        alerts_to=alerts_to,
        account=account,
        live_dir=live_dir,
        on_stop=on_stop,
        daily_limits=by_strategy(
            {
                "max_trades_per_day": max_trades_per_day,
                "daily_profit_target": daily_profit_target,
                "max_order_value": max_order_value,
            },
            names,
            owner=SESSION,
        ),
        account_max_trades_per_day=account_trade_cap(account_max_trades_per_day),
        account_daily_profit_target=rupees_amount(
            account_daily_profit_target, "account_daily_profit_target"
        ),
        max_orders_per_second=_orders_per_second(max_orders_per_second),
    )


__all__ = [
    "BACKTEST_DEFAULT",
    "BARS_FROM_CANDLES",
    "BARS_FROM_TICKS",
    "MODE_ARGUMENTS",
    "NOT_LIVE",
    "ON_STOP",
    "kept_on_stop",
    "LiveArguments",
    "live_arguments",
]
