"""Rows a broker's book leaves out of its next reads: a book that lists a new GTT or a new
order a few seconds late, or drops a row from one read (20-positional-live 3.1, 3.2). A
test names the row by its ID before or after it exists (`FakeOpenAlgo.hide_gtt`,
`hide_order`); every answer of that book that succeeds counts as one read, whether the row
existed then or not.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

#: The books a row can be left out of, and the field of their rows that names it.
GTT_BOOK, ORDER_BOOK = "trigger_id", "orderid"


class HiddenRows:
    """The rows each book leaves out (module docstring). Used under the venue's lock."""

    def __init__(self) -> None:
        self._reads: dict[tuple[str, str], int] = {}

    def hide(self, book: str, row_id: str, reads: int) -> None:
        """Leave row `row_id` out of the next `reads` answers of `book` (`GTT_BOOK` or
        `ORDER_BOOK`). Raises ValueError for another book or fewer than 1 read."""
        if book not in (GTT_BOOK, ORDER_BOOK):
            raise ValueError(f"rows are left out of the GTT book or the order book, not {book!r}")
        if isinstance(reads, bool) or not isinstance(reads, int) or reads < 1:
            raise ValueError(f"a row is left out of 1 read or more, not {reads!r}")
        self._reads[(book, row_id)] = reads

    def shown(self, book: str, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """`rows`, one answer of `book` that succeeded, less the rows it leaves out; that
        answer counts as one read of every row hidden from `book`."""
        hidden = {row_id for (named, row_id), left in self._reads.items() if named == book}
        for row_id in hidden:
            key = (book, row_id)
            self._reads[key] -= 1
            if self._reads[key] == 0:
                del self._reads[key]
        return [row for row in rows if str(row.get(book)) not in hidden]


__all__ = ["GTT_BOOK", "ORDER_BOOK", "HiddenRows"]
