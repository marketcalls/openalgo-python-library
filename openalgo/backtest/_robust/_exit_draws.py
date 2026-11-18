"""The draws of the randomized-exits check (design 17, section 2.12): each run's factors
for its trades' exit distances, and its values of the exit parameters the user names.

Run r of the stream `seed` reads `width` draws (`_random.block`, draws r x width
onwards): the parameters' first, one each in the order given, then one factor for each
trade the run may open (`factors`, at least the run's strategy bars on every symbol, so
no trade of a rerun reuses another's factor unless it opens several trades at one
decision). A factor is 1 plus `spread` times a draw uniform from -1 to 1 (a fraction f
from `_random.fractions` gives 1 + spread x (2f - 1)), so it lies from 1 - spread to
below 1 + spread; the kernel multiplies each trade's leg-rule distances by its factor
(`engine::exit_scale`). A parameter's value is one of the values it names, each as
likely (`_random.below`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import _random


@dataclass(frozen=True)
class ExitDraws:
    """What each run's exits are drawn from (module docstring): the stream `seed`, the
    `spread` of the factors, the (name, values) of each exit parameter and the number of
    factors a run draws."""

    seed: int
    spread: float
    params: tuple[tuple[str, tuple[Any, ...]], ...]
    factors: int

    @property
    def width(self) -> int:
        """The draws a run reads."""
        return len(self.params) + self.factors

    def values(self, run: int) -> dict[str, Any]:
        """The parameter values of run `run`, by name."""
        draws = _random.outputs(self.seed, run * self.width, len(self.params))
        return {
            name: choices[int(_random.below(draws[k : k + 1], len(choices))[0])]
            for k, (name, choices) in enumerate(self.params)
        }

    def draw(self, run: int) -> tuple[dict[str, Any], tuple[float, ...]]:
        """(the parameter values, the factors) of run `run`."""
        start = run * self.width + len(self.params)
        fractions = _random.fractions(_random.outputs(self.seed, start, self.factors))
        factors = 1.0 + self.spread * (2.0 * fractions - 1.0)
        return self.values(run), tuple(float(factor) for factor in factors)


__all__ = ["ExitDraws"]
