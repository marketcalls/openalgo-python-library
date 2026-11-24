"""The layout of the live journal's file (`_journal_file`): its tables, the version of that
layout, and what takes a file of an earlier version to it.

Every table is STRICT, so a value of the wrong type is refused instead of kept. Each has a
`ledger` column, a `Mode` value: the mode of the handle that wrote the row, or for an
update the mode it was handled in. `day` is an IST trading day, YYYY-MM-DD; a broker may
give an order ID again on a later day, so an order ID is unique in a day. Times are UTC
nanoseconds.

- Version 1 (openalgo 2.3.0): `meta`, `intents`, `updates`, `stops`, `risk_states`,
  `user_states`, `sessions`. An intent's `state` is "recorded" until its outcome is known,
  then "bound" to its order, "unbound" (outcome unknown), "queued" (semi-auto put it in
  the Action Center; its `reason` is then the pending order ID there) or "not_placed".
- Version 2 (openalgo 2.4.0, 18-multi-strategy 12.1): the intents' `version_id` (its
  strategy's version, "" in a file version 1 wrote) and `versions`, one strategy's version
  at one start of the day: its id and its run manifest, as JSON.
- Version 3 (openalgo 2.5.0, 20-positional-live 3, 4 and 8; `_journal_gtt`,
  `_journal_carry`): what a position held overnight needs across days. `gtts`, each GTT
  the session placed, written before its place call (its `seq` is that request's number
  in the journal, `request_id` the order worker's), bound to the broker's `trigger_id`,
  with what it is for (`purpose`, a leg's stop or a strategy's entry, a `Purpose` value),
  its terms, its state and its leg's units when a read last showed it resting
  (`baseline_units`, NULL before: what a GTT gone later is judged by); `gtt_states`,
  each state a GTT took and when; `gtt_children`, the child order a fired GTT placed, by
  its broker and kernel order IDs; `day_closes`,
  one record per strategy and trading day of what the next day's session restores;
  `kept_refusals`, a broker's refusal of a stop GTT kept across starts; `decided_bars`,
  the last bar each strategy decided.
"""

from __future__ import annotations

from types import MappingProxyType

#: The layout of the journal file this version reads and writes.
SCHEMA_VERSION = 3

_VERSIONS = (
    """CREATE TABLE versions (
        ledger TEXT NOT NULL, day TEXT NOT NULL, strategy TEXT NOT NULL,
        version_id TEXT NOT NULL, manifest TEXT NOT NULL, started_ns INTEGER NOT NULL) STRICT""",
    "CREATE INDEX versions_by_day ON versions (ledger, day, strategy)",
)
_POSITIONAL = (
    """CREATE TABLE gtts (
        seq INTEGER PRIMARY KEY, ledger TEXT NOT NULL, day TEXT NOT NULL,
        strategy TEXT NOT NULL, gtt_id INTEGER NOT NULL, request_id INTEGER NOT NULL,
        trigger_id TEXT, symbol TEXT NOT NULL, exchange TEXT NOT NULL, product TEXT NOT NULL,
        purpose TEXT NOT NULL, side TEXT NOT NULL, trigger_price REAL NOT NULL,
        limit_price REAL NOT NULL, quantity INTEGER NOT NULL, state TEXT NOT NULL,
        recorded_ns INTEGER NOT NULL, changed_ns INTEGER NOT NULL, baseline_units INTEGER)
        STRICT""",
    "CREATE UNIQUE INDEX gtts_by_trigger ON gtts (ledger, trigger_id)",
    "CREATE INDEX gtts_by_state ON gtts (ledger, state, seq)",
    """CREATE TABLE gtt_states (
        ledger TEXT NOT NULL, gtt_seq INTEGER NOT NULL, state TEXT NOT NULL,
        note TEXT NOT NULL, at_ns INTEGER NOT NULL) STRICT""",
    "CREATE INDEX gtt_states_by_gtt ON gtt_states (ledger, gtt_seq)",
    """CREATE TABLE gtt_children (
        ledger TEXT NOT NULL, day TEXT NOT NULL, gtt_seq INTEGER NOT NULL,
        order_id TEXT NOT NULL, kernel_order INTEGER NOT NULL, bound_ns INTEGER NOT NULL,
        PRIMARY KEY (ledger, gtt_seq)) STRICT""",
    "CREATE UNIQUE INDEX gtt_children_by_order ON gtt_children (ledger, day, order_id)",
    """CREATE TABLE day_closes (
        ledger TEXT NOT NULL, strategy TEXT NOT NULL, day TEXT NOT NULL,
        carried TEXT NOT NULL, user_state TEXT NOT NULL, version_id TEXT NOT NULL,
        saved_ns INTEGER NOT NULL, PRIMARY KEY (ledger, strategy, day)) STRICT""",
    """CREATE TABLE kept_refusals (
        ledger TEXT NOT NULL, strategy TEXT NOT NULL, symbol TEXT NOT NULL,
        exchange TEXT NOT NULL, product TEXT NOT NULL, message TEXT NOT NULL,
        day TEXT NOT NULL, kept_ns INTEGER NOT NULL,
        PRIMARY KEY (ledger, strategy, symbol, exchange, product)) STRICT""",
    """CREATE TABLE decided_bars (
        ledger TEXT NOT NULL, strategy TEXT NOT NULL, bar_ns INTEGER NOT NULL,
        decided_ns INTEGER NOT NULL, PRIMARY KEY (ledger, strategy)) STRICT""",
)
#: Every table and index of a new file, `SCHEMA_VERSION`'s layout.
SCHEMA = (
    "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT",
    """CREATE TABLE intents (
        seq INTEGER PRIMARY KEY, intent_id TEXT NOT NULL UNIQUE, ledger TEXT NOT NULL,
        day TEXT NOT NULL, strategy TEXT NOT NULL, symbol TEXT NOT NULL,
        exchange TEXT NOT NULL, action TEXT NOT NULL, quantity INTEGER NOT NULL,
        price_type TEXT NOT NULL, product TEXT NOT NULL, purpose TEXT NOT NULL,
        created_ns INTEGER NOT NULL, price REAL NOT NULL, trigger_price REAL NOT NULL,
        state TEXT NOT NULL, order_id TEXT, reason TEXT NOT NULL,
        changed_ns INTEGER NOT NULL, version_id TEXT NOT NULL DEFAULT '') STRICT""",
    "CREATE UNIQUE INDEX intents_by_order ON intents (ledger, day, order_id)",
    "CREATE INDEX intents_by_day ON intents (ledger, day, state, seq)",
    """CREATE TABLE updates (
        seq INTEGER PRIMARY KEY, ledger TEXT NOT NULL, day TEXT NOT NULL,
        order_id TEXT NOT NULL, symbol TEXT NOT NULL, exchange TEXT NOT NULL,
        action TEXT NOT NULL, quantity INTEGER NOT NULL, filled_quantity INTEGER NOT NULL,
        pending_quantity INTEGER NOT NULL, average_price REAL NOT NULL,
        status TEXT NOT NULL, terminal INTEGER NOT NULL, rejection_reason TEXT NOT NULL,
        mode TEXT NOT NULL, ts_recv_ns INTEGER NOT NULL, source TEXT NOT NULL) STRICT""",
    "CREATE INDEX updates_by_order ON updates (ledger, day, order_id, terminal, seq)",
    """CREATE TABLE stops (
        ledger TEXT NOT NULL, day TEXT NOT NULL, strategy TEXT NOT NULL,
        symbol TEXT NOT NULL, exchange TEXT NOT NULL, order_id TEXT NOT NULL,
        set_ns INTEGER NOT NULL, PRIMARY KEY (ledger, day, strategy, symbol, exchange)) STRICT""",
    """CREATE TABLE risk_states (
        ledger TEXT NOT NULL, strategy TEXT NOT NULL, day TEXT NOT NULL,
        state TEXT NOT NULL, saved_ns INTEGER NOT NULL,
        PRIMARY KEY (ledger, strategy, day)) STRICT""",
    """CREATE TABLE user_states (
        ledger TEXT NOT NULL, strategy TEXT NOT NULL, state TEXT NOT NULL,
        saved_ns INTEGER NOT NULL, PRIMARY KEY (ledger, strategy)) STRICT""",
    """CREATE TABLE sessions (
        ledger TEXT NOT NULL, day TEXT NOT NULL, broker TEXT NOT NULL,
        first_started_ns INTEGER NOT NULL, last_started_ns INTEGER NOT NULL,
        starts INTEGER NOT NULL, PRIMARY KEY (ledger, day)) STRICT""",
    *_VERSIONS,
    *_POSITIONAL,
)
#: What takes a file of each earlier version one version on, by its version: version 1
#: gains the intents' version ids (its intents keep "") and the versions table; version 2
#: the tables of positions held overnight. A file is taken through each step in turn, its
#: rows untouched.
UPGRADES = MappingProxyType(
    {
        1: ("ALTER TABLE intents ADD COLUMN version_id TEXT NOT NULL DEFAULT ''", *_VERSIONS),
        2: _POSITIONAL,
    }
)


__all__ = ["SCHEMA", "SCHEMA_VERSION", "UPGRADES"]
