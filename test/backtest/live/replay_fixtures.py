"""The pytest fixture of the replay's recorded sessions (`replay_samples`). A test module
takes it with

    from replay_fixtures import *  # noqa: F403

since pytest finds a fixture by its name in the module that uses it.
"""

from __future__ import annotations

import logging

import pytest
from replay_samples import SCENARIOS, Recorded, recorded

from openalgo.backtest._live._types import LOGGER_NAME


@pytest.fixture(params=list(SCENARIOS), ids=list(SCENARIOS))
def session(request, make_fake_openalgo, monkeypatch, tmp_path, caplog) -> Recorded:
    """Each scenario of `replay_samples`, recorded on the fake (`recorded`)."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return recorded(request.param, make_fake_openalgo, monkeypatch, tmp_path)


__all__ = ["session"]
