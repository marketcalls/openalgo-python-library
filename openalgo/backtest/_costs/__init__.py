"""Private parts of `openalgo.backtest.costs`. Import from `openalgo.backtest.costs`.

| Module | Responsibility |
|---|---|
| `kernel` | The native cost functions, the tables they publish, error translation |
| `inputs` | Turning dates, sides, segments and numbers into what the kernel takes |
| `presets` | Brokerage presets |
| `schedules` | Schedule versions as dicts, to and from the kernel's tuples |
| `results` | `CostBreakdown` and the unverified-schedule warning |
| `model` | `CostModel` |
| `notes` | Contract-note rounding of per-trade accruals |
"""
