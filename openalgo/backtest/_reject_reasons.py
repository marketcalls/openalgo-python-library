"""What each reject code of the orders table means and what to do about it: the wording
of the run's rejection warnings (`_rejections`).

The codes are the kernel's (`RejectCode::code` in `rust/oa_bt/src/exec/checks.rs`), except
`rate_limit`, which no kernel check makes: a backtest enforces no order rate limit. Each
entry is (what happened, what to do). The product rule covers three refusals that need
different advice, so it has one entry per case, told apart by the order and its
instrument (`_rejections.product_rule_case`).
"""

from __future__ import annotations

from types import MappingProxyType

PRODUCT_RULE = "product_rule"

REASONS = MappingProxyType(
    {
        "quantity_rule": (
            "the quantity was not a whole number of lots (derivatives trade in whole "
            "lots), or not a whole number of shares in venue mode",
            "give derivatives with lots= or a multiple of the lot size, or size with "
            "bt.size, which rounds to the lot",
        ),
        "tick": (
            "a limit or trigger price was not a multiple of the instrument's tick size",
            "round prices to self.instrument().tick_size",
        ),
        "freeze_qty": (
            "the quantity was above the exchange's freeze quantity for one order",
            "keep each order at or below self.instrument().freeze_qty; a bt.size sizer "
            "caps at it on its own",
        ),
        "margin": (
            "the margin the order needed was more than the free cash",
            "raise capital, lower the size, or set size_caps, for example size_caps = "
            "bt.size.caps(leverage=1) for a position of at most 1x the equity",
        ),
        "entry_window": (
            "orders after the MIS entry cut-off (entry_until) are rejected, and so are "
            "entries before entry_from",
            "check self.now or set entry_until",
        ),
        "market_closed": (
            "an intraday order decided at the session's last close could only fill in the "
            "next session",
            "stop placing orders on the session's last bar (check self.now)",
        ),
        "daily_loss_gate": (
            "an entry came after the day's loss reached max_daily_loss, which stops new "
            "entries for the rest of the day",
            "this is the limit working; raise max_daily_loss if it is too tight",
        ),
        "cooldown": (
            "an entry came within cooldown base bars of an exit",
            "lower cooldown",
        ),
        "max_positions": (
            "an entry would have opened more positions than max_positions allows",
            "raise max_positions",
        ),
        "halted": (
            "the trading state was REDUCING or HALTED and the order would have opened or "
            "added to a position",
            "result.risk_events shows what changed the state",
        ),
        "instrument_unknown": (
            "the order named a symbol the run does not trade",
            "add the symbol to the strategy's symbols",
        ),
        "reverse_not_allowed": (
            "a buy or sell was larger than the opposite position, and only reverse=True "
            "lets the excess open the other side",
            "close with self.exit(), pass reverse=True, or flip with self.target(...)",
        ),
        "scale_in_limit": (
            "an add broke a scale-in limit (max_adds, max_position, min_spacing or "
            "max_risk) or averaged down",
            "set allow_averaging_down = True or loosen the scale-in limits",
        ),
    }
)

PRODUCT_RULE_CASES = MappingProxyType(
    {
        "delivery_short": (
            "a delivery (CNC) short is not allowed in Indian cash markets",
            "short a stock intraday with MIS, or trade futures",
        ),
        "gtt_on_mis": (
            "MIS cannot hold a GTT order",
            "place a GTT on CNC or NRML",
        ),
        "index": (
            "an index cannot be traded",
            "trade its futures or options",
        ),
    }
)

__all__ = ["PRODUCT_RULE", "PRODUCT_RULE_CASES", "REASONS"]
