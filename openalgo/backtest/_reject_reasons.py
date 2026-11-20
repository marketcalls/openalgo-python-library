"""What each reject code of the orders table means and what to do about it: the wording
of the run's rejection warnings (`_rejections`).

The codes are the kernel's (`RejectCode::code` in `rust/oa_bt/src/exec/checks.rs`), except
`rate_limit`, which no kernel check makes: a backtest enforces no order rate limit. `venue`
and `exit_cover` are a live session's: the broker's own rejection of an order, and a
reduce-only order that would leave more units resting to close a position than it has
(`rust/oa_bt/src/engine/venue_cover.rs`). So are `live_lock` and `live_in_flight`, the
entry gates a live session adds (`rust/oa_bt/src/engine/venue_gates.rs`), and the freeze
quantity is a cap on the whole position there, not only on each order; and
`group_other_side` and `shared_resting`, its gates on a symbol several of its strategies
trade (`rust/oa_bt/src/engine/venue_group.rs`). `account_margin` and
`account_loss_gate` are a portfolio run's: the shared account's free cash refusing an
opening fill, or a sized order at its submission, that the strategy's own free cash
covers, and the account's daily loss limit
locking entries (`rust/oa_bt/src/engine/account_link.rs`, `account_breach.rs`), and so are
`account_trade_limit` and `account_profit_gate`. `trade_limit`, `daily_profit_gate` and
`order_value_limit` are a strategy's daily limits (`rust/oa_bt/src/engine/daily_caps.rs`,
18-multi-strategy 2.5a). Each entry is (what happened, what to do). The product rule
covers three refusals that need different advice, so it has one entry per case, told
apart by the order and its
instrument (`_rejections.product_rule_case`). The entry window is the window working, not
a mistake: its line names the window's times (`_rejections.entry_window_warning`).

`RUN_RULE_CODES` are the refusals of a rule of the run working as it intends: the MIS
entry window (given or the calendar's default), the limits a strategy or run sets
(max_daily_loss, drawdown_halt, cooldown, max_positions, and the REDUCING or HALTED
state they lead to, the daily limits max_trades_per_day, daily_profit_target and
max_order_value, and the account's daily loss limit, daily profit target and trade cap of
a portfolio run) and the delay a run holds its orders for (delay_entries=,
delay_exits=). The end-of-run notice leaves them out (`_rejection_notice`); each is told
once in `result.warnings` (`_rejections`).
"""

from __future__ import annotations

from types import MappingProxyType

PRODUCT_RULE = "product_rule"
ENTRY_WINDOW = "entry_window"
DELAY_PAST_CLOSE = "delay_past_close"
ACCOUNT_LOSS_GATE = "account_loss_gate"
ACCOUNT_MARGIN = "account_margin"
ACCOUNT_PROFIT_GATE = "account_profit_gate"
ACCOUNT_TRADE_LIMIT = "account_trade_limit"
#: The codes of a strategy's daily limits (18-multi-strategy 2.5a).
DAILY_LIMIT_CODES = ("trade_limit", "daily_profit_gate", "order_value_limit")
RUN_RULE_CODES = frozenset(
    {
        ENTRY_WINDOW,
        "daily_loss_gate",
        "cooldown",
        "max_positions",
        "halted",
        DELAY_PAST_CLOSE,
        ACCOUNT_LOSS_GATE,
        ACCOUNT_PROFIT_GATE,
        ACCOUNT_TRADE_LIMIT,
        *DAILY_LIMIT_CODES,
    }
)

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
        DELAY_PAST_CLOSE: (
            "an order held by delay_entries= or delay_exits= would have waited past the last "
            "close it could fill after: its session's close for an intraday order, the run's "
            "last strategy bar for any order",
            "this is the delay working: an order decided that near the session's or the "
            "run's end cannot fill that late; lower the delay to fill such orders",
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
        "account_margin": (
            "in a portfolio run on a shared account, the margin the order needed was within "
            "the strategy's own free cash but more than the account's free cash: the other "
            "strategies held it",
            "give the account more funds (account_capital=), lower the size, or give each "
            'strategy its own capital with pool="separate"',
        ),
        ACCOUNT_LOSS_GATE: (
            "an entry came after the account's loss for the day reached "
            "account_max_daily_loss, which stops every strategy's new entries for the rest "
            "of the day",
            "this is the account's limit working; raise account_max_daily_loss if it is too tight",
        ),
        "trade_limit": (
            "an entry came once the strategy's entries of the day (those that filled and "
            "those still working) had reached max_trades_per_day; a reversal at the cap was "
            "cut to its close, its opening part refused",
            "this is the cap working: exits still go; raise max_trades_per_day if it is too tight",
        ),
        ACCOUNT_TRADE_LIMIT: (
            "an entry came once the entries of the day of every strategy of the account "
            "together had reached account_max_trades_per_day",
            "this is the account's cap working: exits still go; raise "
            "account_max_trades_per_day if it is too tight",
        ),
        "daily_profit_gate": (
            "an entry or add came after the day's profit reached daily_profit_target, which "
            "exits every position and stops new entries for the rest of the day",
            "this is the target working; raise daily_profit_target to keep trading longer",
        ),
        ACCOUNT_PROFIT_GATE: (
            "an entry or add came after the account's profit for the day reached "
            "account_daily_profit_target, which exits every strategy's positions and stops "
            "their new entries for the rest of the day",
            "this is the account's target working; raise account_daily_profit_target to "
            "keep trading longer",
        ),
        "order_value_limit": (
            "an entry or add was worth more than max_order_value (its units times its "
            "limit, an SL-M's trigger or the last close, times the multiplier); exits are "
            "never refused by it",
            "lower the size, or raise max_order_value",
        ),
        "live_in_flight": (
            "a live session sends no entry or add while another order of the symbol is "
            "working (an entry whose fill is not reported yet, a resting order), so that "
            "a fill reported late cannot double the position; a sale or buy against the "
            "position beyond what the orders already working to close it leave counts as "
            "an entry, since their fills could leave the position flat and it would then "
            "open the other side",
            "enter, add or close once the earlier order has filled or been cancelled: "
            "check self.orders and self.position in on_bar, and close a position with "
            "self.exit(), which cancels the orders on its closing side first",
        ),
        "group_other_side": (
            "another strategy of the live session holds the symbol, or is entering it, on "
            "the other side: a broker nets the two into one position, so a position on the "
            "other side is refused",
            "exit first, or trade the other side in its own session on another account",
        ),
        "shared_resting": (
            "the symbol is traded by several strategies of the live session, which take "
            "only market orders there: a resting LIMIT, SL or SL-M order would keep their "
            "combined position from being checked against the broker's",
            "use market orders on that symbol; targets and the leg's stop (target=, stop=) "
            "are held for you",
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

__all__ = [
    "ACCOUNT_LOSS_GATE",
    "ACCOUNT_MARGIN",
    "ACCOUNT_PROFIT_GATE",
    "ACCOUNT_TRADE_LIMIT",
    "DAILY_LIMIT_CODES",
    "DELAY_PAST_CLOSE",
    "ENTRY_WINDOW",
    "PRODUCT_RULE",
    "PRODUCT_RULE_CASES",
    "REASONS",
    "RUN_RULE_CODES",
]
