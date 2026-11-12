"""The built-in reference tables against the server's master contract (read-only).

For the near-month futures of three MCX commodities and of NIFTY and BANKNIFTY, the
lot size, tick size and expiry the server lists must equal what
`default_tables().instrument(...)` resolves for today. The snapshot check runs the
tables' own guard for MCX: a symbol-master snapshot whose lot size is in another unit
than the tables' (lots instead of units) makes `instrument` raise.
"""

import pytest
from live_support import live_only, near_month_future, server_expiry, today_ist

from openalgo.backtest import reference as ref

pytestmark = [
    live_only,
    pytest.mark.filterwarnings("ignore::openalgo.backtest.reference.ReferenceDataWarning"),
]

FUTURES = [
    ("CRUDEOIL", "MCX"),
    ("GOLDM", "MCX"),
    ("NATURALGAS", "MCX"),
    ("NIFTY", "NFO"),
    ("BANKNIFTY", "NFO"),
]
MCX_FUTURES = [name for name, exchange in FUTURES if exchange == "MCX"]
IDS = [name for name, _ in FUTURES]


def _reference_price(client, name: str, exchange: str) -> float | None:
    """The price that selects an NFO index future's tick slab: the index's last close.

    The rule names the close of the last trading day of the previous month; the
    previous session's close falls in the same slab unless the index sits on a
    slab boundary. MCX ticks have no slabs.
    """
    if exchange != "NFO":
        return None
    quote = client.quotes(symbol=name, exchange="NSE_INDEX")
    return float(quote["data"]["prev_close"])


@pytest.fixture(scope="module")
def contracts(client):
    """(server row, resolved Instrument) per future, looked up once."""
    tables, today = ref.default_tables(), today_ist()
    resolved = {}
    for name, exchange in FUTURES:
        row = near_month_future(client, name, exchange)
        price = _reference_price(client, name, exchange)
        resolved[name] = (row, tables.instrument(row["symbol"], exchange, today, price))
    return resolved


@pytest.mark.parametrize("name", IDS)
def test_the_lot_size_matches_the_master_contract(contracts, name):
    row, instrument = contracts[name]
    assert int(row["lotsize"]) == instrument.lot_size


@pytest.mark.parametrize("name", IDS)
def test_the_tick_size_matches_the_master_contract(contracts, name):
    row, instrument = contracts[name]
    assert float(row["tick_size"]) == pytest.approx(instrument.tick_size)


@pytest.mark.parametrize("name", IDS)
def test_the_expiry_is_the_last_trading_day(contracts, name):
    row, instrument = contracts[name]
    assert server_expiry(row) == instrument.last_trading_day


def test_mcx_lot_sizes_from_the_server_are_in_the_tables_unit(client, broker, tmp_path):
    today = today_ist()
    ref.write_symbol_master_snapshot(
        client, tmp_path, exchanges=("MCX",), broker=broker, as_of=today
    )
    tables = ref.load_tables(snapshots=tmp_path)
    for name in MCX_FUTURES:
        row = near_month_future(client, name, "MCX")
        instrument = tables.instrument(row["symbol"], "MCX", today, allow_missing=("freeze_qty",))
        assert instrument.lot_size == int(row["lotsize"]), name
