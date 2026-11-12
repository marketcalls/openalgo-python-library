"""Interval strings: what `parse_interval` reads and what it refuses with a ConfigError."""

import pytest

from openalgo.backtest._layout import parse_interval
from openalgo.backtest.errors import ConfigError


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        ("1m", ("intraday", 60)),
        ("15m", ("intraday", 900)),
        ("1h", ("intraday", 3600)),
        ("D", ("D", 1)),
        ("W", ("W", 1)),
    ],
)
def test_an_interval_is_read_as_its_kind_and_count(interval, expected):
    assert parse_interval(interval) == expected


# A superscript digit is a digit to str.isdigit but not a number int() can read; it
# must be refused like any other unknown form, never raise a bare ValueError.
@pytest.mark.parametrize("interval", ["0m", "-5m", "²m", "5x", "m", "", 5])
def test_an_unusable_interval_raises_a_config_error(interval):
    with pytest.raises(ConfigError):
        parse_interval(interval)
