"""The provider protocol, the registry and entry-point plugins (FR-DATA-03).

Plugins are simulated by replacing the entry-point lookup; nothing is installed.
"""

import datetime as dt
import subprocess
import sys

import pytest
from data_fixtures.fakes import (
    FakeCalendar,
    at_ist,
    canonical_frame,
    days_between,
    ist_frame,
    truth_rows,
)

from openalgo.backtest.data import (
    DataProvider,
    Files,
    Frame,
    OpenAlgoProvider,
    available_providers,
    get_provider,
    load_bars,
    providers,
    register_provider,
    unregister_provider,
)
from openalgo.backtest.data.providers import (
    ENTRY_POINT_GROUP,
    discover_providers,
    provider_load_errors,
    validate_provider,
)
from openalgo.backtest.errors import ConfigError, DataError, TimestampError

MON = dt.date(2024, 1, 1)
FRI = dt.date(2024, 1, 5)
CAL = FakeCalendar()
LONG_AFTER = at_ist(dt.date(2024, 3, 15), 18, 0)
ROWS = truth_rows(CAL, "NSE", days_between(MON, FRI))


class VendorProvider:
    """A third-party provider: serves canonical bars it built itself."""

    name = "myvendor"
    capabilities = {"bars"}
    timestamp_convention = "utc_open"

    def __init__(self, rows=ROWS, shift_seconds=0):
        self._rows = [dict(row, timestamp=row["timestamp"] + shift_seconds) for row in rows]

    def bars(self, symbol, exchange, interval, start, end):
        return canonical_frame(self._rows)


class _EntryPoint:
    def __init__(self, name, target):
        self.name = name
        self._target = target

    def load(self):
        if isinstance(self._target, Exception):
            raise self._target
        return self._target


@pytest.fixture
def registry(monkeypatch):
    """A fresh registry holding only the built-in providers; restored afterwards."""
    monkeypatch.setattr(providers, "_REGISTRY", providers._Registry())
    return providers._REGISTRY


@pytest.fixture
def plugins(monkeypatch, registry):
    """Install fake entry points: call it with {name: target}."""

    def install(entries):
        def entry_points(*, group):
            assert group == ENTRY_POINT_GROUP
            return [_EntryPoint(name, target) for name, target in entries.items()]

        monkeypatch.setattr(providers._metadata, "entry_points", entry_points)

    return install


# --- the protocol -------------------------------------------------------------------


def test_protocol_has_the_members_of_the_prd():
    assert DataProvider.__annotations__ == {
        "name": str, "capabilities": set[str], "timestamp_convention": str,
    }  # fmt: skip
    assert callable(DataProvider.bars)
    assert ENTRY_POINT_GROUP == "openalgo.backtest.providers"


def test_third_party_object_with_the_four_members_is_a_data_provider():
    assert isinstance(VendorProvider(), DataProvider)
    assert validate_provider(VendorProvider()).name == "myvendor"


@pytest.mark.parametrize(
    "member, value, message",
    [
        ("name", "", "non-empty string attribute 'name'"),
        ("capabilities", ["bars"], "'capabilities' must be a set of strings"),
        ("capabilities", {"ticks"}, "does not declare the 'bars' capability"),
        ("timestamp_convention", None, "must declare 'timestamp_convention'"),
        ("bars", None, "has no bars"),
    ],
)
def test_provider_that_breaks_the_protocol_is_a_config_error(member, value, message):
    provider = VendorProvider()
    setattr(provider, member, value)
    with pytest.raises(ConfigError, match=message):
        validate_provider(provider)


# --- the registry -------------------------------------------------------------------


def test_built_in_providers_are_registered(registry):
    assert available_providers() == ["files", "frame", "openalgo"]
    assert registry.providers == {"openalgo": OpenAlgoProvider, "frame": Frame, "files": Files}


def test_no_yfinance_provider_ships_with_the_engine(registry):
    assert "yfinance" not in available_providers()
    code = "import sys, openalgo.backtest.data; assert 'yfinance' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0


def test_built_in_provider_is_built_with_its_options(registry):
    frame = get_provider("frame", data=ist_frame(ROWS), calendar=CAL, now=LONG_AFTER)
    assert isinstance(frame, Frame)
    assert len(frame.bars("X", "NSE", "1m", MON, FRI)) == len(ROWS)


def test_registered_instance_is_returned_as_it_is(registry):
    vendor = VendorProvider()
    register_provider("myvendor", vendor)
    assert get_provider("myvendor") is vendor


def test_registered_class_is_built_on_request(registry):
    register_provider("myvendor", VendorProvider)
    assert isinstance(get_provider("myvendor", shift_seconds=0), VendorProvider)


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: register_provider("frame", VendorProvider()), "already registered"),
        (lambda: register_provider("", VendorProvider()), "non-empty string"),
        (lambda: register_provider("broken", object()), "non-empty string attribute 'name'"),
        (
            lambda: get_provider("nowhere"),
            "unknown data provider 'nowhere'. Registered: files, frame",
        ),
        (lambda: get_provider("frame", no_such_option=1), "could not be created"),
    ],
    ids=["name taken", "empty name", "not a provider", "unknown name", "bad options"],
)
def test_registry_misuse_is_a_config_error(registry, call, message):
    with pytest.raises(ConfigError, match=message):
        call()


def test_options_for_a_registered_instance_are_a_config_error(registry):
    register_provider("myvendor", VendorProvider())
    with pytest.raises(ConfigError, match="registered as an instance and takes no options"):
        get_provider("myvendor", shift_seconds=60)


def test_replace_overrides_a_registered_provider(registry):
    register_provider("myvendor", VendorProvider())
    replacement = VendorProvider()
    register_provider("myvendor", replacement, replace=True)
    assert get_provider("myvendor") is replacement


def test_unregistered_provider_is_gone(registry):
    register_provider("myvendor", VendorProvider())
    unregister_provider("myvendor")
    assert "myvendor" not in available_providers()


# --- entry-point plugins ------------------------------------------------------------


def test_entry_point_plugin_is_discovered_on_first_use(plugins):
    plugins({"myvendor": VendorProvider})
    assert isinstance(get_provider("myvendor"), VendorProvider)
    assert available_providers() == ["files", "frame", "myvendor", "openalgo"]


def test_entry_points_are_read_once_per_process(plugins):
    plugins({"myvendor": VendorProvider})
    assert discover_providers() == ["myvendor"]
    assert discover_providers() == []
    assert discover_providers(reload=True) == ["myvendor"]


def test_broken_plugin_is_reported_and_does_not_hide_the_others(plugins):
    plugins({"broken": ImportError("No module named 'vendor_sdk'"), "myvendor": VendorProvider})
    with pytest.warns(RuntimeWarning, match="'broken'.*was not registered.*vendor_sdk"):
        assert discover_providers() == ["myvendor"]
    assert provider_load_errors() == {"broken": "ImportError: No module named 'vendor_sdk'"}


def test_unknown_provider_error_says_when_its_plugin_failed_to_load(plugins):
    plugins({"broken": ImportError("No module named 'vendor_sdk'")})
    with (
        pytest.warns(RuntimeWarning),
        pytest.raises(ConfigError, match="failed to load: ImportError"),
    ):
        get_provider("broken")


def test_plugin_cannot_take_the_name_of_a_built_in_provider(plugins, registry):
    plugins({"frame": VendorProvider})
    with pytest.warns(RuntimeWarning, match="reserved for a built-in provider"):
        discover_providers()
    assert registry.providers["frame"] is Frame


def test_plugin_that_is_not_a_provider_is_reported(plugins):
    plugins({"notaprovider": object()})
    with pytest.warns(RuntimeWarning, match="non-empty string attribute 'name'"):
        assert discover_providers() == []


# --- plugins through load_bars ------------------------------------------------------


def test_load_bars_uses_a_registered_provider_by_name(registry):
    register_provider("myvendor", VendorProvider())
    result = load_bars("X", "NSE", "1m", MON, FRI, source="myvendor", calendar=CAL, now=LONG_AFTER)
    assert (result.meta.provider, result.meta.source) == ("myvendor", "myvendor")
    assert result.meta.timestamp_convention == "utc_open"
    assert result.bars["ts_open"].iloc[0] == at_ist(MON, 9, 15)
    assert result.bars["ts_close"].iloc[-1] == at_ist(FRI, 15, 30)


def test_load_bars_accepts_a_provider_object_as_source():
    result = load_bars(
        "X", "NSE", "1m", MON, FRI, source=VendorProvider(), calendar=CAL, now=LONG_AFTER
    )
    assert list(result.meta.partition_hashes()) == ["myvendor/NSE/X/1m/2024-01"]


def test_plugin_bars_are_validated_against_the_session_calendar():
    shifted_vendor = VendorProvider(shift_seconds=19_800)
    with pytest.raises(TimestampError, match="5 hours 30 minutes late"):
        load_bars("X", "NSE", "1m", MON, FRI, source=shifted_vendor, calendar=CAL, now=LONG_AFTER)


def test_plugin_that_returns_another_layout_is_a_data_error():
    class WrongLayout(VendorProvider):
        def bars(self, symbol, exchange, interval, start, end):
            return ist_frame(ROWS)

    with pytest.raises(
        DataError, match="bars from data provider 'myvendor': columns must be exactly"
    ):
        load_bars("X", "NSE", "1m", MON, FRI, source=WrongLayout(), calendar=CAL, now=LONG_AFTER)
