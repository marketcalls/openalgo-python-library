"""Worked-trace fixtures for the backtest kernel and the helpers that load and check them.

The fixture files (`trace-NN-*.json`) and the specification review (`spec_review.json`)
live next to these modules. The fixture format is documented in
`test/backtest/test_traces.py`.

Modules:

    vocabulary       field names and value vocabularies of the format (data only)
    timegrid         IST timestamps, session grids, strategy-bar boundaries
    bars             bar rows as exact decimals, aggregation of base bars
    fixture_loader   reads the files and resolves bars derived from another case
    schema           shape of a trace file and of a case's inputs
    schema_expected  shape of a case's expected section
    ledger           positions and gross P&L replayed from the expected fills
    timeline         read access to the level timeline and to fills by slice
    checks_*         consistency rules, one function per rule, grouped by subject:
                     run, bars, fills, levels, ratchets, triggers, strategy, events,
                     risk_events, review, text
    kernel_hook      where the kernel plugs in, and the comparison with a fixture
"""
