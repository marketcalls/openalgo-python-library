"""The market settings of a run: the cost model, the slippage and the per-instrument
facts the kernel's market rules read (`rust/oa_py/src/backtest/engine/market.rs`).

    costs:    "zerodha" (the default) or another preset name, a brokerage preset from
              `bt.costs` (`bt.costs.flat(20)`), a `bt.costs.CostModel` (for calculator
              rounding, user schedules or backfill), or None (no charges; every report
              warns). Statutory charges apply under every preset.
    slippage: see `openalgo.backtest.slippage`.
    margin:   a futures contract's own margin, from `margin=` (`_futures_margin`).

The cost segment of each instrument is checked here for the run's product, and the cost
model is asked for the run's first and last day, so an instrument or a date the model
cannot price stops the run before it starts (CostError) rather than at its first fill.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from . import costs as costs_module
from . import slippage as slippage_module
from ._futures_margin import MarginSetting, own_margin
from .errors import CostError, ReferenceDataError
from .reference import Instrument, default_tables

DERIVATIVE_EXCHANGES = ("NFO", "BFO")
# The cost model's instrument codes (`costs.segment_for`) by Instrument.kind.
SEGMENT_KINDS = {"equity": "EQ", "future": "FUT", "option": "OPT"}


@dataclass(frozen=True)
class MarketSettings:
    """What `market_settings` resolved: the config entries of the native engine, the
    Python cost model (None when cost-free), labels for the Assumptions block and the
    warnings to report."""

    config: dict[str, Any]
    cost_model: costs_module.CostModel | None
    costs_label: str
    slippage_label: str
    warnings: tuple[str, ...]

    @property
    def assumptions(self) -> dict[str, str]:
        """The costs and slippage lines of the Assumptions block."""
        return {"costs": self.costs_label, "slippage": self.slippage_label}

    @property
    def cost_schedules(self) -> list[str]:
        """Ids of the cost model's schedule versions, oldest first (empty when cost-free)."""
        model = self.cost_model
        return [] if model is None else [version["id"] for version in model.schedules()]


def cost_model(value: object) -> costs_module.CostModel | None:
    """The cost model of a `costs=` argument (None for no charges).

    Raises CostError for an unknown preset name or a value of another type.
    """
    if value is None:
        return None
    if isinstance(value, costs_module.CostModel):
        return value
    if isinstance(value, str | costs_module.Brokerage):
        return costs_module.CostModel(brokerage=value)
    raise CostError(
        "costs takes a preset name ('zerodha'), a preset from bt.costs (bt.costs.flat(20)), "
        f"a bt.costs.CostModel or None, got {value!r}"
    )


def _costs_label(model: costs_module.CostModel | None) -> str:
    if model is None:
        return "none"
    return f"{model.brokerage.name} ({model.rounding.replace('_', '-')} rounding)"


def _check_costs(
    model: costs_module.CostModel, instruments: list[Instrument], product: str, days: tuple
) -> None:
    """Check that the model prices every instrument on the run's first and last day.

    Raises CostError naming the instrument; warns (UnverifiedScheduleWarning) when the
    rates of those days are not verified.
    """
    for instrument in instruments:
        if instrument.kind not in SEGMENT_KINDS:
            continue  # an index is never traded, so never charged
        try:
            segment = costs_module.segment_for(
                instrument.exchange,
                product,
                SEGMENT_KINDS[instrument.kind],
                None,
                instrument.underlying,
            )
            for day in dict.fromkeys(days):
                model.trade("BUY", 1.0, 1.0, day, segment)
        except CostError as exc:
            raise CostError(
                f"{instrument.symbol}: the charges cannot be priced ({exc}); pass costs=None to "
                "run without charges"
            ) from None


def _slippage_class(instrument: Instrument, table: Any, day: dt.date) -> tuple[str | None, str]:
    """The default-table class of an instrument and a note (empty when nothing was
    assumed)."""
    if instrument.kind == "equity":
        low = instrument.symbol in table.low_volume
        return ("EQ_LOW_VOLUME" if low else "EQ_LARGE_MID"), ""
    suffix = "FUT" if instrument.kind == "future" else "OPT"
    if instrument.exchange == "MCX":
        return f"MCX_{suffix}", ""
    if instrument.exchange == "CDS":
        return "CDS", ""
    if instrument.exchange not in DERIVATIVE_EXCHANGES or instrument.underlying is None:
        return None, ""
    try:
        kind = default_tables().underlying_type(instrument.underlying, instrument.exchange, day)
    except ReferenceDataError:
        note = (
            f"{instrument.symbol}: {instrument.underlying} is not a listed index underlying; "
            "the stock-derivative slippage row is used"
        )
        return f"STOCK_{suffix}", note
    return (f"INDEX_{suffix}" if kind == "index" else f"STOCK_{suffix}"), ""


def _facts(
    instruments: list[Instrument], slippage: Any, day: dt.date, margin: MarginSetting
) -> tuple[list, list[str]]:
    facts, notes = [], []
    for instrument in instruments:
        own, class_code = None, None
        if isinstance(slippage, slippage_module.SlippageTable):
            class_code, note = _slippage_class(instrument, slippage, day)
            notes += [note] if note else []
            override = slippage.overrides.get(class_code) if class_code else None
            own = None if override is None else override.to_native()
        facts.append(
            {
                "underlying": instrument.underlying,
                "bse_group": None,
                "slippage_class": class_code,
                "slippage": own,
                "futures_margin": own_margin(margin, instrument),
            }
        )
    return facts, notes


def market_settings(
    plan: Any, instruments: list[Instrument], args: dict[str, Any]
) -> MarketSettings:
    """The market settings of a run of `plan` on `instruments` with the run arguments
    `args` (`costs`, `slippage`, and `margin`, the parsed `margin=`, absent for the
    default). Raises CostError or ConfigError for settings that cannot be used."""
    model = cost_model(args["costs"])
    if model is not None:
        _check_costs(model, instruments, plan.product, (plan.start, plan.end))
    slippage = slippage_module.resolve(args["slippage"])
    facts, notes = _facts(instruments, slippage, plan.start, args.get("margin"))
    run_form = (
        ("segment_default", 0.0, 0)
        if isinstance(slippage, slippage_module.SlippageTable)
        else slippage.to_native()
    )
    config = {
        "costs": None if model is None else model._kernel,
        "slippage": run_form,
        "market": facts,
    }
    return MarketSettings(
        config,
        model,
        _costs_label(model),
        slippage_module.label(args["slippage"]),
        tuple(notes),
    )


__all__ = ["MarketSettings", "cost_model", "market_settings"]
