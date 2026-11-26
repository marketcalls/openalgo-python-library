"""Per-strategy arguments by strategy class name: one set of rules for `bt.backtest` given a
list of strategies and for `bt.live`.

A per-strategy setting (`PER_STRATEGY`) takes one value for every strategy, or a dict keyed
by strategy class name ({"SbinTrend": 200_000, "SbinBreakout": 150_000}); `bt.live` takes
every one of them but `product` and `base`, which a live session refuses. A key that names
no strategy of the call is refused, naming the strategies it may name. A strategy the dict
leaves out is refused where the setting must be given for each one (`capital`, and every
setting but `params` and, in `bt.live`, `max_daily_loss`, which its plan checks for each
strategy); `params` leaves a strategy it does not name on its class defaults. None for a
strategy in a dict is its value: for a setting off when None (`OFF_WHEN_NONE`) it leaves
that strategy without the setting, and the refusal of a dict that leaves a strategy out
says so. A strategy class attribute of the same name still wins over the run argument, as
for one strategy (`_settings`). With one strategy, `params` holds that strategy's own
values, as `bt.backtest` takes them.

The account's settings (`ACCOUNT_SETTINGS`) are one value for all the strategies of a call
together. They describe the account of a list of strategies: a run of one strategy leaves
them at their defaults (`refuse_account_settings`), so its figures are those it has always
had. They include the account's daily limits of 18-multi-strategy 2.5a,
`account_max_trades_per_day` (a whole number of entries of every strategy together in one
session day) and `account_daily_profit_target`. Amounts are rupees.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ._calendar_convert import native_attr
from ._risk_settings import rupees_amount
from .errors import ConfigError
from .report._format import indian_digits
from .strategy import Strategy

#: The daily limits of a strategy (18-multi-strategy 2.5a), per strategy in both verbs.
DAILY_LIMITS = ("max_trades_per_day", "daily_profit_target", "max_order_value")
#: The risk settings of a strategy, per strategy in both verbs (18-multi-strategy 1.1 and
#: 1.3, decision M5).
RISK_SETTINGS = (
    "after_risk_exit",
    "max_positions",
    "drawdown_halt",
    "cooldown",
    "entry_from",
    "entry_until",
    "square_off",
)
#: The settings a call takes per strategy: one value for every strategy, or a dict by
#: strategy class name (18-multi-strategy 1.1).
PER_STRATEGY = (
    "capital",
    "params",
    "product",
    "base",
    "max_daily_loss",
    *RISK_SETTINGS,
    *DAILY_LIMITS,
    "positional_stops",
)
#: The per-strategy settings that are off for a strategy given None (no cap on positions,
#: no drawdown halt, no cooldown, no daily limit). None for the others is a default (the
#: calendar's times, the lock after a risk exit), and `max_daily_loss` is mandatory live.
OFF_WHEN_NONE = ("max_positions", "drawdown_halt", "cooldown", *DAILY_LIMITS)
#: The pool of a portfolio run when none is given: one account the strategies share.
SHARED_POOL = "shared"
#: The account's settings of a list of strategies and their defaults, which a run of one
#: strategy keeps (the defaults of `bt.backtest`'s signature).
ACCOUNT_SETTINGS = MappingProxyType(
    {
        "pool": SHARED_POOL,
        "account_capital": None,
        "account_max_daily_loss": None,
        "account_max_trades_per_day": None,
        "account_daily_profit_target": None,
    }
)
#: Why a missing name is refused, as the message says it: `capital` must be given for
#: each strategy in every call; a backtest's other settings name every strategy or none.
MANDATORY = "is mandatory for every strategy"
EVERY_STRATEGY = (
    "as a dict must name every strategy of the run (give one value for all of them, or "
    "a value for each)"
)
#: Paise per rupee: a separate pool's account equals the sum of the capitals to the paisa,
#: as the native fund compares them.
PAISE_PER_RUPEE = 100


def strategy_classes(strategy: Any, verb: str) -> tuple[type[Strategy], ...]:
    """The strategy classes of a call of `verb` ("bt.live", "bt.backtest"), in call order:
    one class, or each class of a list or tuple. Raises ConfigError for an empty list or a
    value that is not a `bt.Strategy` subclass."""
    found = tuple(strategy) if isinstance(strategy, list | tuple) else (strategy,)
    if not found:
        raise ConfigError(f"{verb} takes a bt.Strategy subclass, or a list of them")
    for each in found:
        if not (isinstance(each, type) and issubclass(each, Strategy)):
            raise ConfigError(
                f"{verb} takes bt.Strategy subclasses, not {each!r}: pass the class itself, "
                "or a list of classes"
            )
    return found


def class_names(strategies: Sequence[type[Strategy]]) -> tuple[str, ...]:
    """The strategies' class names, in call order: the names every dict by name keys."""
    return tuple(strategy.__name__ for strategy in strategies)


def repeated(names: Sequence[str]) -> list[str]:
    """The names that occur more than once, sorted (each verb words its own refusal)."""
    return sorted({name for name in names if names.count(name) > 1})


def by_name(
    value: Any, names: Sequence[str], what: str, *, owner: str, required: str | None = None
) -> dict[str, Any]:
    """`value` for each strategy of `names`: a dict keyed by strategy class name, or one
    value for every strategy (anything but a Mapping). `owner` names the call in messages
    ("this session", "this backtest"); `required`, when given, is why a strategy the dict
    leaves out is refused (`MANDATORY`, `EVERY_STRATEGY`), and without it such a strategy
    takes None. Raises ConfigError for a key that names no strategy of the call, and for a
    strategy left out when `required`; for a setting of `OFF_WHEN_NONE` the refusal adds
    that None leaves a strategy without it."""
    if not isinstance(value, Mapping):
        return dict.fromkeys(names, value)
    unknown = [key for key in value if key not in names]
    if unknown:
        raise ConfigError(
            f"{what} names {unknown}, which {owner} does not run: key it by strategy "
            f"class name, one of {list(names)}"
        )
    missing = [name for name in names if name not in value]
    if required is not None and missing:
        # None is a value a dict may give: say so where it means "without the setting".
        how = f": give None for a strategy that has no {what}" if what in OFF_WHEN_NONE else ""
        raise ConfigError(f"{what} {required}, and the dict gives none for {missing}{how}")
    return {name: value.get(name) for name in names}


def by_strategy(
    given: Mapping[str, Any], names: Sequence[str], *, owner: str
) -> dict[str, dict[str, Any]]:
    """Each strategy's value of every setting in `given` (a setting's name to one value
    for every strategy, or a dict by strategy class name naming each of `names`), by
    strategy name in call order. Raises ConfigError as `by_name` does with
    `EVERY_STRATEGY`."""
    resolved = {
        setting: by_name(value, names, setting, owner=owner, required=EVERY_STRATEGY)
        for setting, value in given.items()
    }
    return {name: {setting: values[name] for setting, values in resolved.items()} for name in names}


def params_by_name(
    params: Any, names: Sequence[str], *, owner: str
) -> dict[str, Mapping[str, Any] | None]:
    """Each strategy's parameter values (None for its class defaults): with one strategy,
    `params` itself; with several, a dict of each one's values by class name, a strategy
    it leaves out keeping its defaults. Raises ConfigError for a `params` that is not a
    dict and, with several strategies, as `by_name` does."""
    if params is not None and not isinstance(params, Mapping):
        raise ConfigError(f"params takes a dict of parameter values, not {params!r}")
    if len(names) == 1:
        return {names[0]: params}
    return by_name(params or {}, names, "params", owner=owner)


def refuse_account_settings(args: Mapping[str, Any]) -> None:
    """Raise ConfigError when the arguments of a run of one strategy give an account
    setting (`ACCOUNT_SETTINGS`) other than its default: they describe the account of a
    list of strategies. An argument left out counts as its default."""
    given = [
        name for name, default in ACCOUNT_SETTINGS.items() if args.get(name, default) != default
    ]
    if given:
        names = ", ".join(f"{name}=" for name in given)
        raise ConfigError(
            f"{names} set the account of a list of strategies, as in bt.backtest([SbinTrend, "
            "SbinBreakout], ...); a single strategy's own daily limits are max_daily_loss=, "
            "max_trades_per_day= and daily_profit_target="
        )


@dataclass(frozen=True)
class Account:
    """The account of a portfolio run, checked: `pool` ("shared" or "separate"),
    `account_capital` in rupees (None for the sum of the strategies' capitals),
    `account_max_daily_loss` in rupees (None for no limit); `capitals` are the strategies'
    capitals in rupees, in call order; `account_max_trades_per_day` the entries of every
    strategy together in one session day and `account_daily_profit_target` in rupees
    (each None for none)."""

    pool: str
    account_capital: float | None
    account_max_daily_loss: float | None
    capitals: tuple[float, ...]
    account_max_trades_per_day: int | None = None
    account_daily_profit_target: float | None = None

    def fund_config(self) -> dict[str, Any]:
        """The config dict the native `Fund` reads."""
        return {
            "pool": self.pool,
            "account_capital": self.account_capital,
            "account_max_daily_loss": self.account_max_daily_loss,
            "account_max_trades_per_day": self.account_max_trades_per_day,
            "account_daily_profit_target": self.account_daily_profit_target,
        }


def account_settings(args: Mapping[str, Any], capitals: Sequence[float]) -> Account:
    """The account of a list of strategies from the run arguments `args` and the
    strategies' `capitals` (positive rupees, call order). Raises ConfigError for a pool
    other than those the engine runs, an account capital, daily loss limit or daily profit
    target that is not a positive amount of rupees (a number or bt.rupees(x)), a trade cap
    that is not a whole number of at least 1, and, with pool="separate", an account
    capital other than the sum of the capitals."""
    pools = tuple(native_attr("POOLS"))
    pool = args.get("pool", SHARED_POOL)
    if pool not in pools:
        raise ConfigError(
            f"pool takes {' or '.join(map(repr, pools))} (one account the strategies share, "
            f"or fixed allocations), not {pool!r}"
        )
    funds = rupees_amount(args.get("account_capital"), "account_capital")
    limit = rupees_amount(args.get("account_max_daily_loss"), "account_max_daily_loss")
    total = sum(capitals)
    summed = funds is None or round(funds * PAISE_PER_RUPEE) == round(total * PAISE_PER_RUPEE)
    if pool == "separate" and not summed:
        raise ConfigError(
            f"pool='separate' gives each strategy its own capital, so the account is their "
            f"sum, Rs {indian_digits(total)}: leave account_capital out, or give "
            f"pool='shared' for an account of Rs {indian_digits(funds)}"
        )
    return Account(
        pool,
        funds,
        limit,
        tuple(float(capital) for capital in capitals),
        account_trade_cap(args.get("account_max_trades_per_day")),
        rupees_amount(args.get("account_daily_profit_target"), "account_daily_profit_target"),
    )


def account_trade_cap(value: Any) -> int | None:
    """`account_max_trades_per_day` checked: None, or a whole number of entries of every
    strategy together in one session day, at least 1. Raises ConfigError otherwise."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(
            "account_max_trades_per_day takes a whole number of entries of every strategy "
            f"together in one session day, at least 1, or None, not {value!r}"
        )
    return value


__all__ = [
    "ACCOUNT_SETTINGS",
    "DAILY_LIMITS",
    "EVERY_STRATEGY",
    "MANDATORY",
    "OFF_WHEN_NONE",
    "PER_STRATEGY",
    "RISK_SETTINGS",
    "SHARED_POOL",
    "Account",
    "account_settings",
    "account_trade_cap",
    "by_name",
    "by_strategy",
    "class_names",
    "params_by_name",
    "refuse_account_settings",
    "repeated",
    "strategy_classes",
]
