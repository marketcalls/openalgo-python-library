"""What each reject code of the orders table means and what to do about it: the wording
of the run's rejection warnings (`_rejections`).

The codes are the kernel's (`RejectCode::code` in `rust/oa_bt/src/exec/checks.rs`), except
`rate_limit`, which no kernel check makes: a backtest enforces no order rate limit. `venue`
and `exit_cover` are a live session's: the broker's own rejection of an order, and a
reduce-only order that would leave more units resting to close a position than it has
(`rust/oa_bt/src/engine/venue_cover.rs`). So are `live_lock` and `live_in_flight`, the
entry gates a live session adds (`rust/oa_bt/src/engine/venue_gates.rs`), and the freeze
quantity is a cap on the whole position there, not only on each order. Each
entry is (what happened, what to do). The product rule covers three refusals that need
different advice, so it has one entry per case, told apart by the order and its
instrument (`_rejections.product_rule_case`). The entry window is the window working, not
a mistake: its line names the window's times (`_rejections.entry_window_warning`).

`RUN_RULE_CODES` are the refusals of a rule of the run working as it intends: the MIS
entry window (given or the calendar's default) and the limits a strategy or run sets
(max_daily_loss, drawdown_halt, cooldown, max_positions, and the REDUCING or HALTED
state they lead to). The end-of-run notice leaves them out (`_rejection_notice`).
"""

from __future__ import annotations

from types import MappingProxyType

PRODUCT_RULE = "product_rule"
ENTRY_WINDOW = "entry_window"
RUN_RULE_CODES = frozenset({ENTRY_WINDOW, "daily_loss_gate", "cooldown", "max_positions", "halted"})

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
            "the quantity was above the exchange's freeze quantity for one order; in a "
            "live session also an entry or add that would have taken the position above "
            "it, since one stop order and one exit order must cover the whole position",
            "keep each order, and in a live session each position, at or below "
            "self.instrument().freeze_qty; a bt.size sizer caps each order at it on its "
            "own",
        ),
        "margin": (
            "the margin the order needed was more than the free cash",
            "raise capital, lower the size, or set size_caps, for example size_caps = "
            "bt.size.caps(leverage=1) for a position of at most 1x the equity",
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
        "venue": (
            "the broker or the exchange refused the order in a live session, for a "
            "reason of its own",
            "read the broker's reason in the session's journal and the OpenAlgo order book",
        ),
        "exit_cover": (
            "a live session keeps at most the position's units resting to close it, so "
            "that the broker cannot fill them all and reverse the position; this "
            "reduce-only order would have gone beyond them",
            "hold a target with the strategy's own logic, or, once it is built, with "
            "target=, which the engine holds on ticks",
        ),
        "live_lock": (
            "a live session had locked entries (a change of OpenAlgo's mode, a feed with "
            "no ticks, the start or the stop of the session, a failed exit, a "
            "reconciliation mismatch or the account's loss limit), so an entry or add was "
            "not sent; exits still were",
            "the session's log and alerts name the lock; entries resume when it lifts, and "
            "a lock that stands for the day needs a new session",
        ),
        "live_in_flight": (
            "a live session sends no entry or add while another order of the symbol is "
            "working (an entry whose fill is not reported yet, a resting order), so that "
            "a fill reported late cannot double the position",
            "enter or add once the earlier order has filled or been cancelled: check "
            "self.orders and self.position in on_bar",
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

__all__ = ["ENTRY_WINDOW", "PRODUCT_RULE", "PRODUCT_RULE_CASES", "REASONS", "RUN_RULE_CODES"]
