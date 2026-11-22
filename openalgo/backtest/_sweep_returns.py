"""The session returns of a `bt.optimize` sweep as one array (sessions, combinations),
which the optimizer's overfitting figures read (`_robust._optimizer_checks`; design 17,
section 5).

The sweep keeps 8 bytes a session and combination, with no session index per
combination, and fills the array as the combinations are measured: the array is made
when the first combination comes back, on its sessions, and each combination's returns
are copied into its column as they come back and then let go. A combination's returns are
therefore held twice only while one is copied: in this process one combination at a
time, and from worker processes the chunks that have come back and wait to be copied.

A combination whose sessions differ from the array's is placed on the sessions of both:
on the array's when it has no session the array lacks (the sessions it has no return for
count 0), else on the array grown to every session either has, in date order. Growing
copies the columns filled so far into a new array, so for that moment they are held
twice: the union of the sessions is not known before the combination that brings a new
one has run. In a sweep with such a combination, a return with no number counts 0 as
well, so each column is the one an outer join of the combinations' returns gives with
its gaps filled with 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ReturnsMatrix:
    """The array of the module docstring, filled column by column in grid order."""

    def __init__(self, combinations: int) -> None:
        """An empty array for `combinations` combinations; `add` fills it."""
        self._combinations = combinations
        self._sessions = pd.Index([])
        self._array = np.empty((0, 0))
        self._filled = 0
        self._aligned = True

    def add(self, returns: pd.Series) -> None:
        """Copy the next combination's session returns (fractions, indexed by IST session
        date, `Performance.session_returns`) into its column."""
        values = returns.to_numpy(dtype=float)
        if self._filled == 0:
            self._sessions = returns.index
            # Fortran order: each combination's returns lie together, so a sum over
            # sessions runs in NumPy's pairwise order (`_robust._overfitting_probability`).
            self._array = np.zeros((len(returns.index), self._combinations), order="F")
        if returns.index.equals(self._sessions):
            self._array[:, self._filled] = values
        else:
            self._aligned = False
            if not returns.index.isin(self._sessions).all():
                self._grow(returns.index)
            self._array[self._sessions.get_indexer(returns.index), self._filled] = values
        self._filled += 1

    def _grow(self, sessions: pd.Index) -> None:
        """Place the array on every session it has or `sessions` has, in date order: the
        columns filled so far move to their sessions' rows, the new rows read 0. The old
        array is let go before the grown one is made, so only the filled columns are held
        twice."""
        union = self._sessions.union(sessions).sort_values()
        filled = self._filled
        kept = self._array[:, :filled].copy()
        self._array = np.empty((0, 0))
        grown = np.zeros((len(union), self._combinations), order="F")
        grown[union.get_indexer(self._sessions), :filled] = kept
        self._sessions, self._array = union, grown

    def array(self) -> np.ndarray:
        """The array, (sessions, combinations) in grid order; (0, 0) when nothing was
        added. In a sweep whose combinations' sessions differ, a return with no number
        reads 0 (module docstring)."""
        if not self._aligned:
            for position in range(self._filled):
                column = self._array[:, position]
                column[np.isnan(column)] = 0.0
        return self._array


__all__ = ["ReturnsMatrix"]
