"""The driver of a portfolio run: the native fund advances every strategy's engine over one
account, and each strategy decides at its own pauses as it would alone.

    fund = Fund(engines, names, account)        # the engines in call order
    a condition strategy: its conditions to the fund, decided in Rust
    while (step := fund.advance()) is not None:
        index, pause = step                     # strategy `index` stands at `pause`
        that strategy's decision                # as `_driver.drive` makes it
    the account's records = fund.finish()
    a condition strategy: its skipped entries recorded, then on_stop

An event strategy's decision at a pause is `_driver.drive`'s, pause for pause: the bars of
the slice revealed and the events delivered, then at a timer pause the timers fired, at
the "finished" pause `on_stop`, at a bar close `on_bar`; before the strategy's first
decision (`first_decision`, its indicators ready) the pauses only reveal their bars and
deliver their events. A strategy decides on its own engine while the fund stands at its
pause. The engines are finished once the fund has returned None (`_pipeline.conclude`),
never at a strategy's "finished" pause, since the fund lends every engine at each advance.

The loop runs once per pause of every event strategy, so its own cost is part of every
callback, as `_driver.drive`'s is: it decides inline, from the strategy's place in call
order, with no Python call per pause beyond the strategy's own.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._calendar_convert import native_attr
from ._conditions import has_conditions, record_skipped, signals_of
from ._driver import deliver
from ._engine import call
from ._pipeline import Begun
from ._strategy_args import Account
from .errors import BacktestError, ConfigError

#: A first decision before every pause: a strategy that decides from its first pause.
FROM_THE_FIRST_PAUSE = -1


def _start(fund: Any, members: Sequence[Begun]) -> list[int | None]:
    """Hand every condition strategy's conditions to `fund` (decided in Rust) and return,
    per strategy in call order, the instant an event strategy decides from (UTC
    nanoseconds), None for a condition strategy."""
    firsts: list[int | None] = []
    for index, member in enumerate(members):
        ready_at = member.state.first_decision
        if has_conditions(member.instance):
            call(fund.add_signals, index, signals_of(member.instance, member.engine, ready_at))
            firsts.append(None)
        else:
            firsts.append(FROM_THE_FIRST_PAUSE if ready_at is None else ready_at)
    return firsts


def _decide_every_pause(fund: Any, members: Sequence[Begun], firsts: list[int | None]) -> None:
    """Advance `fund` to its end, each event strategy deciding at its pauses (module
    docstring). Raises ConfigError or BacktestError for the fund's errors (as `_engine
    .call` converts them) and the strategies' own errors unchanged."""
    # Per strategy in call order: what a pause of it needs, read once.
    lanes = [
        (
            member.instance,
            member.context,
            member.context.reveal,
            first,
            member.instance.symbol is not None or member.instance.underlying is not None,
        )
        for member, first in zip(members, firsts, strict=True)
    ]
    advance = fund.advance
    while True:
        try:
            step = advance()
        except native_attr("NativeEngineInputError") as exc:
            raise ConfigError(str(exc)) from None
        except native_attr("NativeEngineError") as exc:
            raise BacktestError(str(exc)) from None
        if step is None:
            return
        # Only an event strategy pauses: a condition strategy is decided in Rust.
        index, (kind, decided_at, rows, events, timers) = step
        strategy, ctx, reveal, first, single = lanes[index]
        bars = reveal(decided_at, rows)
        if events:
            deliver(strategy, ctx, events)
        if kind == "finished":
            strategy.on_stop()
        elif decided_at < first:
            continue
        elif kind == "timers":
            ctx.timers.fire(timers)
        elif bars:
            strategy.on_bar(next(iter(bars.values())) if single else bars)


def drive_fund(members: Sequence[Begun], names: Sequence[str], account: Account) -> dict:
    """Run every strategy of a portfolio run to the end of its grid on one account and
    return the account's records (`Fund.finish`: the account's rows, its breaches, the
    orders it refused and the entries the Rust drivers' sizers skipped). `members` are
    the strategies begun in call order (`_pipeline.begin`), `names` their class names.
    The engines are left for `_pipeline.conclude` to finish. Raises ConfigError for an
    account the fund refuses, and the kernel's and the strategies' errors unchanged, an
    engine's error starting with its strategy's name."""
    engines = [member.engine for member in members]
    fund = call(native_attr("Fund"), engines, list(names), account.fund_config())
    firsts = _start(fund, members)
    _decide_every_pause(fund, members, firsts)
    records = call(fund.finish)
    for member, skipped, first in zip(members, records["skipped"], firsts, strict=True):
        if first is None:
            record_skipped(member.instance, skipped)
            member.instance.on_stop()
    return records


__all__ = ["drive_fund"]
