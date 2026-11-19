"""The arguments of `bt.live`, checked before anything is called (16-session-design 1.2
and 3.1 step 1; decisions 3 and 9 of its section 10.0; unit 7). Part of `plan`.

- One strategy class or a list of them (decision 3): each trades its own symbol in the
  one session. `capital` and `max_daily_loss` are one value for every strategy or a dict
  by strategy class name; with several strategies `params` is a dict of each one's
  parameter values by class name (one strategy takes its own values, as in
  `bt.backtest`).
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
from ..errors import ConfigError
from ..strategy import Strategy
from ._preflight_plan import positive_amount
from .data_worker import HISTORY_SOURCE
from .series import DEFAULT_BASE

#: What `on_stop` takes: flatten every leg before stopping, or keep the legs open with
#: their broker-held stops (16-session-design 3.3, decision 1 of its section 10.0).
ON_STOP = ("flatten", "keep")
#: The bar source of this release: bars built from the feed's ticks (08-live-trading
#: section 5); bars from broker candles come later.
BARS_FROM_TICKS = "websocket"
BARS_FROM_CANDLES = "candles"
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
        "product": "a live session trades MIS (set product on the class, or leave it unset)",
        "roll": "a live session trades one contract named by its symbol",
        "benchmark": "a live session has no benchmark",
        "calendar": "a live session follows the package calendar",
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
    `bt.backtest` settings every strategy shares that the call gave (`costs` and the risk
    settings, which a strategy's class attributes of the same names override as in a
    backtest); the others take `bt.backtest`'s defaults (`backtest_arguments`). The rest are
    the session's own: `account_max_daily_loss` in rupees, `alerts_to`, `account`,
    `live_dir` and `on_stop`.
    """

    strategies: tuple[type[Strategy], ...]
    names: tuple[str, ...]
    capital: Mapping[str, Any]
    max_daily_loss: Mapping[str, Any]
    params: Mapping[str, Mapping[str, Any] | None]
    settings: Mapping[str, Any]
    account_max_daily_loss: float | None
    alerts_to: str | None
    account: str | None
    live_dir: str | os.PathLike[str] | None
    on_stop: str

    def backtest_arguments(self, name: str, day: Any, present_ns: int) -> dict[str, Any]:
        """The arguments of the backtest of strategy `name` on the IST date `day` with
        the present `present_ns` (UTC nanoseconds), as `bt.backtest` would take them:
        its defaults, the session's own settings, and the live terms (start and end
        today, broker history, a 1-minute base). Raises KeyError for an unknown name."""
        return {
            **_backtest_defaults(),
            **self.settings,
            "capital": self.capital[name],
            "max_daily_loss": self.max_daily_loss[name],
            "params": self.params[name],
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


def _strategies(strategy: Any) -> tuple[type[Strategy], ...]:
    found = tuple(strategy) if isinstance(strategy, list | tuple) else (strategy,)
    if not found:
        raise ConfigError("bt.live takes a bt.Strategy subclass, or a list of them")
    for each in found:
        if not (isinstance(each, type) and issubclass(each, Strategy)):
            raise ConfigError(
                f"bt.live takes bt.Strategy subclasses, not {each!r}: pass the class itself, "
                "or a list of classes"
            )
    return found


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
    names = tuple(strategy.__name__ for strategy in strategies)
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise not_supported(
            f"two strategies with the same name ({', '.join(repeated)}) in one session",
            "give each strategy class its own name: OpenAlgo shows it on every order",
        )
    return names


def _by_name(value: Any, names: tuple[str, ...], what: str, *, required: bool) -> dict:
    """`value` for each strategy: a dict keyed by strategy class name, or one value for
    every strategy. Raises ConfigError for a key that names no strategy of the session,
    and, when `required`, for a strategy the dict leaves out."""
    if not isinstance(value, Mapping):
        return dict.fromkeys(names, value)
    unknown = [key for key in value if key not in names]
    if unknown:
        raise ConfigError(
            f"{what} names {unknown}, which this session does not run: key it by strategy "
            f"class name, one of {list(names)}"
        )
    missing = [name for name in names if name not in value]
    if required and missing:
        raise ConfigError(
            f"{what} is mandatory for every strategy, and the dict gives none for {missing}"
        )
    return {name: value.get(name) for name in names}


def _params(params: Any, names: tuple[str, ...]) -> dict[str, Mapping[str, Any] | None]:
    if params is not None and not isinstance(params, Mapping):
        raise ConfigError(f"params takes a dict of parameter values, not {params!r}")
    if len(names) == 1:
        return {names[0]: params}
    return _by_name(params or {}, names, "params", required=False)


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
            f"on_stop takes 'flatten' (exit every position, then stop) or 'keep' (leave "
            f"the positions with their broker stops), not {on_stop!r}"
        )


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
    after_risk_exit: str = BACKTEST_DEFAULT,
    entry_from: str | None = BACKTEST_DEFAULT,
    entry_until: str | None = BACKTEST_DEFAULT,
    square_off: str | None = BACKTEST_DEFAULT,
    max_positions: int | None = BACKTEST_DEFAULT,
    drawdown_halt: Any = BACKTEST_DEFAULT,
    cooldown: int | None = BACKTEST_DEFAULT,
    round_levels: bool = BACKTEST_DEFAULT,
    strict_config: bool = BACKTEST_DEFAULT,
    live_dir: str | os.PathLike[str] | None = None,
    on_stop: str = "flatten",
    **not_live: Any,
) -> LiveArguments:
    """Check the arguments of a `bt.live` call (16-session-design 1.2; module docstring)
    and return them by strategy.

    Raises ConfigError for an argument `bt.live` does not take (a mode choice, a
    `bt.backtest` argument with no meaning live, an unknown name), for a strategy that is
    not a `bt.Strategy` subclass, for no `capital`, for a dict of values keyed by a name
    the session does not run or, for `capital`, missing one, and for an unusable
    `account_max_daily_loss`, `alerts_to`, `account`, `live_dir` or `on_stop`;
    NotImplementedError for `bars="candles"` and for two strategies with the same name."""
    _refuse_not_live(not_live)
    strategies = _strategies(strategy)
    names = _names(strategies)
    if capital is None:
        raise ConfigError(
            "bt.live needs capital=: the rupees allocated to each strategy, one number for "
            "every strategy or a dict by strategy class name (a risk limit, mandatory live)"
        )
    _session_options(account_max_daily_loss, alerts_to, account, live_dir, on_stop)
    shared = {
        "costs": costs,
        "risk_basis": risk_basis,
        "after_risk_exit": after_risk_exit,
        "entry_from": entry_from,
        "entry_until": entry_until,
        "square_off": square_off,
        "max_positions": max_positions,
        "drawdown_halt": drawdown_halt,
        "cooldown": cooldown,
        "round_levels": round_levels,
        "strict_config": strict_config,
    }
    return LiveArguments(
        strategies=strategies,
        names=names,
        capital=_by_name(capital, names, "capital", required=True),
        max_daily_loss=_by_name(max_daily_loss, names, "max_daily_loss", required=False),
        params=_params(params, names),
        settings={name: value for name, value in shared.items() if value is not BACKTEST_DEFAULT},
        account_max_daily_loss=positive_amount(account_max_daily_loss),
        alerts_to=alerts_to,
        account=account,
        live_dir=live_dir,
        on_stop=on_stop,
    )


__all__ = [
    "BACKTEST_DEFAULT",
    "BARS_FROM_CANDLES",
    "BARS_FROM_TICKS",
    "MODE_ARGUMENTS",
    "NOT_LIVE",
    "ON_STOP",
    "LiveArguments",
    "live_arguments",
]
