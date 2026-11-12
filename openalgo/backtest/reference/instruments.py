"""Instrument definition used by the backtest, paper and live engines.

An `Instrument` is an immutable record of what the engine needs to size, price and
settle one tradable contract on one date. Three conventions are fixed here:

* Quantity is always in units, never in lots. One lot is `lot_size` units, and a
  derivative order must be a whole multiple of `lot_size`. The unit is the one the
  OpenAlgo API uses: shares for equity, units of the underlying for NSE and BSE
  derivatives (one NIFTY lot is 75 units when the lot size is 75) and the exchange's
  trading unit on MCX (CRUDEOIL trades in lots of 100 barrels, so one lot is
  quantity 100; GOLD trades in lots of 1 kg, so one lot is quantity 1).
* `multiplier` is the rupee value of a one-rupee price move for ONE UNIT of
  quantity, so profit and loss in rupees is `(exit - entry) * quantity * multiplier`.
  It is 1 wherever the price is quoted for the unit that is traded (equity, NSE and
  BSE derivatives, CRUDEOIL, SILVER) and differs from 1 where the quotation unit is
  not the trading unit (GOLD is traded per kg and quoted per 10 g: 100; ZINC is
  traded per tonne and quoted per kg: 1000; GOLDM is traded per gram and quoted per
  10 g: 0.1).
* `contract_multiplier` is `lot_size * multiplier`: the rupee value of a one-rupee
  move for one lot (GOLD 100, CRUDEOIL 100, NATURALGAS 1250).
* `tick_size` is in rupees. It is None when the exchange defines the tick by price
  slab and no reference price was given, or when the instrument's kind is not known;
  `ReferenceTables.round_price(...)` then says what it needs.
* A derivative has an `expiry` (the listed date: the date in its symbol and in the
  exchange's expiry column) and a `last_trading_day` (the day it stops trading, the day
  the kernel settles it). They differ when an exchange moves only the maturity of a
  contract. Both are as they stood on the trade date the instrument was resolved for:
  an exchange can re-date a contract after listing it, and the symbol changes with it.
  `contract_key` stays the same across such a re-dating.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

from ..errors import ConfigError

KINDS = ("equity", "index", "future", "option")
OPTION_TYPES = ("CE", "PE")

# delivery: cash equity settled by delivery of shares
# cash:     settled in cash at the final settlement price (index derivatives)
# physical: settled by delivery of the underlying (stock derivatives, MCX delivery contracts)
# futures:  an option that devolves into the underlying futures contract (MCX options)
# none:     not tradable (an index)
DERIVATIVE_SETTLEMENTS = ("cash", "physical", "futures")
SETTLEMENTS = ("delivery", *DERIVATIVE_SETTLEMENTS, "none")

# An instrument that trades in single units: a share. An index, which does not trade,
# carries the same value.
SINGLE_UNIT_LOT = 1
# The generic cash-equity instrument the engine assumes when it is given price data
# with no instrument definition (PRD 14, section 6: lot 1, tick 0.05). These are a
# stated assumption that a result reports, not reference data for any real symbol.
GENERIC_LOT_SIZE = SINGLE_UNIT_LOT
GENERIC_TICK_SIZE = 0.05
# Multiplier where the price is quoted for the same unit that quantity counts.
UNIT_MULTIPLIER = 1.0
# Decimal places kept in `contract_multiplier`; removes binary noise from products
# such as 100 * 0.1 without touching any real multiplier.
_MULTIPLIER_DECIMALS = 9


def _is_whole_number(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _is_plain_date(value: object) -> bool:
    return isinstance(value, _dt.date) and not isinstance(value, _dt.datetime)


def _check_identity(instrument: Instrument) -> None:
    for name in ("symbol", "exchange"):
        value = getattr(instrument, name)
        if not isinstance(value, str) or not value:
            raise ConfigError(f"Instrument.{name} must be a non-empty string, got {value!r}")
    if instrument.kind not in KINDS:
        raise ConfigError(f"Instrument.kind must be one of {KINDS}, got {instrument.kind!r}")
    if instrument.settlement not in SETTLEMENTS:
        raise ConfigError(
            f"Instrument.settlement must be one of {SETTLEMENTS}, got {instrument.settlement!r}"
        )


def _check_sizes(instrument: Instrument) -> None:
    if not _is_whole_number(instrument.lot_size):
        raise ConfigError(
            "Instrument.lot_size must be a whole number of units, 1 or more, "
            f"got {instrument.lot_size!r}"
        )
    if instrument.freeze_qty is not None and not _is_whole_number(instrument.freeze_qty):
        raise ConfigError(
            "Instrument.freeze_qty must be a whole number of units or None, "
            f"got {instrument.freeze_qty!r}"
        )
    if instrument.tick_size is not None and not (instrument.tick_size > 0):
        raise ConfigError(
            f"Instrument.tick_size must be positive rupees or None, got {instrument.tick_size!r}"
        )
    if not (instrument.multiplier > 0):
        raise ConfigError(f"Instrument.multiplier must be positive, got {instrument.multiplier!r}")


def _iso(value: _dt.date | None) -> str | None:
    return None if value is None else value.isoformat()


def _check_contract_dates(instrument: Instrument) -> None:
    label = f"{instrument.kind} {instrument.symbol!r}"
    for name in ("last_trading_day", "contract_date"):
        value = getattr(instrument, name)
        if value is None:
            continue
        if not _is_plain_date(value):
            raise ConfigError(f"{label}: {name} must be a datetime.date, got {value!r}")
        if instrument.expiry is None:
            raise ConfigError(f"{label} cannot carry a {name} without an expiry")


def _check_contract_terms(instrument: Instrument) -> None:
    """Expiry, strike and option type must fit the kind of instrument."""
    label = f"{instrument.kind} {instrument.symbol!r}"
    if instrument.expiry is not None and not _is_plain_date(instrument.expiry):
        raise ConfigError(f"{label}: expiry must be a datetime.date, got {instrument.expiry!r}")
    if (instrument.expiry is not None) != (instrument.kind in ("future", "option")):
        needs = "needs" if instrument.expiry is None else "cannot carry"
        raise ConfigError(f"{label} {needs} an expiry date")
    if instrument.kind != "option":
        if instrument.strike is not None or instrument.option_type is not None:
            raise ConfigError(f"{label} cannot carry a strike or an option_type")
        return
    if instrument.option_type not in OPTION_TYPES:
        raise ConfigError(
            f"{label}: option_type must be 'CE' or 'PE', got {instrument.option_type!r}"
        )
    if instrument.strike is None or not (instrument.strike > 0):
        raise ConfigError(f"{label}: strike must be a positive number, got {instrument.strike!r}")


@dataclass(frozen=True)
class Instrument:
    """One instrument as it stood on one date.

    Units (see the module docstring): `lot_size` and `freeze_qty` in units of
    quantity, `tick_size` and `strike` in rupees, `multiplier` in rupees per one-rupee
    price move per unit of quantity, `expiry` an IST session date. `tick_size` None
    means the tick depends on a price slab; `freeze_qty` None means no limit is known.

    Raises `ConfigError` when a value is out of range or the expiry, strike and option
    type do not fit `kind`.
    """

    symbol: str
    exchange: str
    kind: str
    lot_size: int = SINGLE_UNIT_LOT
    tick_size: float | None = None
    freeze_qty: int | None = None
    multiplier: float = UNIT_MULTIPLIER
    expiry: _dt.date | None = None
    strike: float | None = None
    option_type: str | None = None
    underlying: str | None = None
    settlement: str = "delivery"
    # The day the contract stops trading, as in force on the trade date; None when not known.
    last_trading_day: _dt.date | None = None
    # The contract's identity date: the expiry its weekday rule gives (see contract_key).
    contract_date: _dt.date | None = None
    # True when the values are the generic cash-equity assumption, not reference data.
    generic: bool = False
    # (field, source, status) for every value taken from a reference row. Not part of
    # equality: two instruments with the same terms are the same instrument.
    provenance: tuple[tuple[str, str, str], ...] = field(default=(), compare=False, repr=False)

    def __post_init__(self) -> None:
        _check_identity(self)
        _check_sizes(self)
        _check_contract_terms(self)
        _check_contract_dates(self)

    @classmethod
    def cash_equity(cls, symbol: str, exchange: str = "NSE") -> Instrument:
        """The generic cash-equity instrument assumed when no definition is supplied.

        Lot size 1, tick size 0.05, no freeze quantity, multiplier 1, delivery
        settlement. `generic` is True so that a result can report the assumption.
        """
        return cls(
            symbol=symbol,
            exchange=exchange,
            kind="equity",
            lot_size=GENERIC_LOT_SIZE,
            tick_size=GENERIC_TICK_SIZE,
            freeze_qty=None,
            multiplier=UNIT_MULTIPLIER,
            settlement="delivery",
            generic=True,
            provenance=(
                ("lot_size", "generic cash-equity default", "assumed"),
                ("tick_size", "generic cash-equity default", "assumed"),
            ),
        )

    @property
    def is_derivative(self) -> bool:
        """True for a future or an option."""
        return self.kind in ("future", "option")

    @property
    def is_tradable(self) -> bool:
        """False for an index, which can be quoted but not traded."""
        return self.kind != "index"

    @property
    def contract_key(self) -> tuple:
        """What identifies the contract across re-datings of its listed expiry.

        Two symbols of one contract (before and after the exchange moved its expiry)
        have equal keys: (exchange, underlying, kind, contract_date, strike, option_type).
        """
        return (
            self.exchange,
            self.underlying or self.symbol,
            self.kind,
            self.contract_date,
            self.strike,
            self.option_type,
        )

    @property
    def contract_multiplier(self) -> float:
        """Rupee value of a one-rupee price move for one lot (`lot_size * multiplier`)."""
        return round(self.lot_size * self.multiplier, _MULTIPLIER_DECIMALS)

    def notional(self, price: float, quantity: float) -> float:
        """Contract value in rupees of `quantity` units at `price`."""
        return price * quantity * self.multiplier

    def pnl(self, entry_price: float, exit_price: float, quantity: float) -> float:
        """Gross profit in rupees of a long position of `quantity` units."""
        return (exit_price - entry_price) * quantity * self.multiplier

    def to_dict(self) -> dict[str, object]:
        """Plain values for a run manifest (the expiry becomes an ISO date string)."""
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "kind": self.kind,
            "lot_size": self.lot_size,
            "tick_size": self.tick_size,
            "freeze_qty": self.freeze_qty,
            "multiplier": self.multiplier,
            "expiry": _iso(self.expiry),
            "strike": self.strike,
            "option_type": self.option_type,
            "underlying": self.underlying,
            "settlement": self.settlement,
            "last_trading_day": _iso(self.last_trading_day),
            "contract_date": _iso(self.contract_date),
            "generic": self.generic,
            "provenance": [list(item) for item in self.provenance],
        }


__all__ = [
    "DERIVATIVE_SETTLEMENTS",
    "GENERIC_LOT_SIZE",
    "GENERIC_TICK_SIZE",
    "KINDS",
    "OPTION_TYPES",
    "SETTLEMENTS",
    "SINGLE_UNIT_LOT",
    "UNIT_MULTIPLIER",
    "Instrument",
]
