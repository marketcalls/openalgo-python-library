"""The exports of a `PortfolioResult`: every table by name, as CSV files or as one Excel
workbook (`PortfolioExport`), as `_result_export.ExportMethods` gives a `Result` its own.

The account's tables come first (`PORTFOLIO_TABLES`), then each strategy's own tables
(`EXPORTED_TABLES`) prefixed by its name, or in a workbook by its place in call order
(`STRATEGY_SHEET`), since a sheet name holds at most 31 characters. Times are plain IST
date-times in the files; the `*_ts` columns stay UTC nanoseconds.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from ._result_export import EXPORTED_TABLES, write_csv, write_excel
from .result import Result

#: The account's tables `to_dataframes` returns first, in order: the properties of these
#: names; then every strategy's own tables (`EXPORTED_TABLES`), prefixed by its name.
PORTFOLIO_TABLES = (
    "account",
    "contribution",
    "refused",
    "net_positions",
    "trades",
    "fills",
    "orders",
    "events",
    "costs",
    "contract_notes",
    "final_positions",
)
#: The letter of a strategy's sheet prefix in a workbook (`to_excel`): S1, S2, ...
STRATEGY_SHEET = "S"


class PortfolioExport:
    """`to_dataframes`, `to_csv` and `to_excel` of a `PortfolioResult`, which has a
    property for each name of `PORTFOLIO_TABLES` and its strategies' `Result`s by name in
    call order (`results`)."""

    results: Mapping[str, Result]

    def _account_tables(self) -> dict[str, pd.DataFrame]:
        """The account's tables by name (`PORTFOLIO_TABLES`), `contribution` with its
        strategy names as a column."""
        tables = {name: getattr(self, name) for name in PORTFOLIO_TABLES}
        tables["contribution"] = tables["contribution"].reset_index()
        return tables

    def _strategy_tables(self, prefixes: Sequence[str]) -> dict[str, pd.DataFrame]:
        """Each strategy's own tables (`EXPORTED_TABLES`) named `<prefix>_<table>`, with
        `prefixes` one per strategy in call order."""
        return {
            f"{prefix}_{table}": getattr(result, table)
            for prefix, result in zip(prefixes, self.results.values(), strict=True)
            for table in EXPORTED_TABLES
        }

    def to_dataframes(self) -> dict[str, pd.DataFrame]:
        """Every table by name: the account's (`PORTFOLIO_TABLES`; `contribution` with its
        strategy names as a column), then each strategy's own tables (`Result
        .to_dataframes`) named `<strategy>_<table>`."""
        return self._account_tables() | self._strategy_tables(list(self.results))

    def to_csv(self, folder: str | os.PathLike) -> list[Path]:
        """Write every table of `to_dataframes()` to `folder` (created when missing), one
        CSV file each named after it, and return the paths. Times are plain IST
        date-times; the `*_ts` columns stay UTC nanoseconds."""
        return write_csv(self.to_dataframes(), folder)

    def to_excel(self, path: str | os.PathLike) -> Path:
        """Write every table of `to_dataframes()` to one Excel workbook at `path`, a sheet
        each, and return the path. A sheet name holds at most 31 characters, too few for
        a strategy's name and a table's, so the account's tables keep their names, a
        `strategies` sheet gives each strategy a prefix by its place in call order (S1 for
        the first) and its tables are the sheets `<prefix>_<table>` (`S1_fills`). Needs
        openpyxl, which openalgo does not install: raises MissingDependencyError (an
        ImportError) when it is missing."""
        names = list(self.results)
        prefixes = [f"{STRATEGY_SHEET}{place}" for place in range(1, len(names) + 1)]
        index = pd.DataFrame({"sheet_prefix": prefixes, "strategy": names})
        sheets = self._account_tables() | {"strategies": index}
        return write_excel(sheets | self._strategy_tables(prefixes), path)


__all__ = ["PORTFOLIO_TABLES", "STRATEGY_SHEET", "PortfolioExport"]
