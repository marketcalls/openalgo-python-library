"""Private parts of `openalgo.backtest.costs`. Import from `openalgo.backtest.costs`.

| Module | Responsibility |
|---|---|
| `kernel` | The native cost functions, the tables they publish, error translation |
| `inputs` | Turning dates, sides, segments and numbers into what the kernel takes |
| `presets` | Brokerage presets |
| `lots` | A quantity in lots of a contract as price units, from the reference tables |
| `schedules` | Schedule versions as dicts, to and from the kernel's tuples |
| `results` | `CostBreakdown` and the unverified-schedule warning |
| `model` | `CostModel` |
| `notes` | Contract-note rounding of per-trade accruals |
"""
