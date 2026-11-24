"""OpenAlgo's products, as its `product` field names them: the one place the backtest
package and its live layer spell them.

- `MIS`: intraday, squared off by the broker the same day.
- `CNC`: cash delivery, held overnight as holdings once the buy settles.
- `NRML`: derivatives held overnight.
"""

from __future__ import annotations

MIS = "MIS"
CNC = "CNC"
NRML = "NRML"
#: Every product, in the order messages name them.
PRODUCTS = (MIS, CNC, NRML)
#: The products whose positions are held overnight (20-positional-live), the products a
#: GTT order takes.
OVERNIGHT_PRODUCTS = (CNC, NRML)

__all__ = ["CNC", "MIS", "NRML", "OVERNIGHT_PRODUCTS", "PRODUCTS"]
