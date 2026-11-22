"""Write the golden tearsheet pages of `test_report_golden.py` again, from the runs of
`tearsheet_runs`: each run's default page in each report mode.

The pages are regression pins: they say what the tearsheet looked like when they were
made, byte for byte, so a change to the default page shows as a diff. Regenerate them
only on purpose, after a change of the default page that is meant, on the reference
platform (Windows x86_64), and review the diff of every file before committing it:

    python test/backtest/golden/tearsheets/regenerate.py --write

Without `--write` the script refuses to run; no test calls it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))  # the repository root: this source tree first
sys.path.insert(0, str(HERE.parents[1]))  # test/backtest, for tearsheet_runs

import golden_runs  # noqa: E402
import tearsheet_runs  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Write every golden page; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--write", action="store_true", required=True, help="overwrite the golden pages"
    )
    parser.parse_args(argv)
    if not golden_runs.on_reference_platform():
        print(
            f"refused: pages are made on {golden_runs.REFERENCE_PLATFORM}, this is another "
            "platform",
            file=sys.stderr,
        )
        return 1
    total = 0
    for name, run in tearsheet_runs.RUNS.items():
        result = run()
        for mode in tearsheet_runs.MODES:
            page = tearsheet_runs.pinned(tearsheet_runs.default_page(result, mode))
            path = tearsheet_runs.write_page(name, mode, page)
            total += path.stat().st_size
            print(f"wrote {path}")
    if total > tearsheet_runs.MAX_TOTAL_BYTES:
        print(f"the pages take {total:,} bytes, over {tearsheet_runs.MAX_TOTAL_BYTES:,}")
        return 1
    print(f"{total:,} bytes in all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
