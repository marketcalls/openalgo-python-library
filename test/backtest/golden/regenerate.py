"""Write the golden fixtures of `test_golden.py` again, from the runs of `golden_runs`.

The pins are regression pins: they say what the engine produced when they were made,
after that output was checked by hand. Regenerate them only on purpose, after a change
of behaviour that is meant, on the reference platform (Windows x86_64), and review the
diff of every file before committing it:

    python test/backtest/golden/regenerate.py --write

Without `--write` the script refuses to run; no test calls it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # the repository root: this source tree first
sys.path.insert(0, str(HERE.parent))  # test/backtest, for golden_runs

import golden_runs  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Write every pin file; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--write", action="store_true", required=True, help="overwrite the pinned files"
    )
    parser.parse_args(argv)
    if not golden_runs.on_reference_platform():
        print(
            f"refused: pins are made on {golden_runs.REFERENCE_PLATFORM}, this is another platform",
            file=sys.stderr,
        )
        return 1
    for name, run in golden_runs.RUNS.items():
        path = golden_runs.write_pins(name, golden_runs.snapshot(run()))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
