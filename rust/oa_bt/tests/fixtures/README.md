# Test fixtures of the kernel crate

## risk_vectors.json

The golden vectors of OpenAlgo's per-position risk core, copied unchanged from the
OpenAlgo server repository:

- Source: `test/risk/vectors.json` of the OpenAlgo server (the vectors of
  `services/risk/position.py`), last changed in commit 55984920 of 2026-08-30.
- Copied on 2026-11-12. SHA-256 of the copy:
  `88b704d5baba3e9a8195ea8a9f8d439e600ee3cdf0fa91fbd3c93fe61ce44bb0`.

Each case is one last price against one position's state. They bind the engine's leg
rules to the live module (05-risk-management 8, NFR-COR-03) and are run by
`src/rms/leg/tests/vectors.rs`, through the tick path (`LegState::on_tick`, the live
module's order) and through the bar path (steps 3 and 4 on a bar whose four prices are
the last price). The test file lists the cases whose result differs by design and why.

Update the copy only from the server's file, and record the new commit, date and hash
here.

## futures_margin_pages.csv

One contract of every futures row of the dated margin table that came from a broker's
margin calculator page: NSE stock futures from https://zerodha.com/margin-calculator/Futures/
and MCX futures from https://zerodha.com/margin-calculator/Commodity/, both "Last updated:
12 Nov 2026", read on 2026-11-12. Each line is the page's lot (quantity in units), price
and NRML margin in rupees; for MCX the multiplier is the one the page's figures imply
(margin = percent x price x lot x multiplier), since the page does not state it.
`tests/futures_margin.rs` checks that every row's percent reproduces the page's rupee
margin within the rounding of a two-decimal percent. Update it together with the table
rows, from the same page read.
