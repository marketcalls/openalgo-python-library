"""
OpenAlgo Strategy Module Examples

Two separate surfaces reach OpenAlgo's /strategy module, and they need
different credentials:

  api(api_key=...)  - the API-key surface under /api/v1/strategy/. Lifecycle
                      plus reads: list, status, start, stop, close_all,
                      close_leg, runs, orders, events. Nothing here can create
                      a strategy, edit it, enable live trading, rotate a token
                      or delete anything; building a strategy stays in the
                      browser wizard at /strategy.

  Strategy(...)     - the public webhook at /strategy/webhook/<token>, which is
                      what charting-platform alerts and other senders post to. It is
                      not under /api/v1 and takes no API key: the token in the
                      URL is the whole credential.

Two strategy kinds, and each refuses the other's vocabulary:

  batch  - a multi-leg spread entered and exited as a unit. start / stop.
  signal - one alert moves one leg. long_entry / long_exit / short_entry /
           short_exit. There is no start and no mode: the first signal after
           the platform session boundary opens the run.
"""

from openalgo import api, Strategy
import json


def print_response(title, response):
    """Helper function to print responses in a readable format"""
    print(f"\n{title}:")
    print(json.dumps(response, indent=2))


client = api(
    api_key="your_api_key_here",
    host="http://127.0.0.1:5000"
)


# ----------------------------------------------------------------------
# API-key surface
# ----------------------------------------------------------------------

def list_strategies_example():
    """List the strategies this API key owns, newest first."""
    response = client.strategylist()
    for s in response.get("data", []):
        print(f"  id={s['id']:<4} kind={s['strategy_kind']:<7} "
              f"status={s['status']:<8} live_enabled={s['live_enabled']} "
              f"{s['name']}")

    # Filters. An out-of-vocabulary status is a 400, not an empty list.
    print_response("Running only", client.strategylist(status="running"))
    print_response("Name search", client.strategylist(q="NIFTY"))


def status_example(strategy_id):
    """One strategy's full configuration including legs, plus its current run."""
    response = client.strategystatus(strategy_id=strategy_id)
    print_response("Strategy status", response)

    # run is null whenever the strategy has no current run, which is the normal
    # state of a stopped strategy. Prefer it over the strategy's own status
    # field when you need to know whether anything is actually open.
    run = response.get("run")
    if run is None:
        print("  nothing is running")
    elif run.get("stop_requested_reason"):
        # A durable stop is pending: the run is still current and still managed
        # until fills prove every position is flat.
        print(f"  stop pending: {run['stop_requested_reason']}")
    else:
        print(f"  run {run['id']} in {run['mode']} mode on {run['broker']}")

    for leg in response.get("data", {}).get("legs", []):
        print(f"  leg {leg['id']}: {leg}")


def start_example(strategy_id):
    """Start a batch strategy. Every leg's entry order is placed.

    mode is required and is never defaulted, here or on the server. The default
    a hurried reader would reach for is the one that places real orders, so
    there is no default at all: omitting it is a TypeError.

    Live is opt-in per strategy. A strategy is created sandbox-only and
    mode="live" is refused with a 409 until the operator enables live trading
    on the strategy page.
    """
    response = client.strategystart(strategy_id=strategy_id, mode="sandbox")
    print_response("Start", response)

    # Partial success is a 200. Check each leg rather than assuming they all
    # reached the market.
    for leg in response.get("legs", []):
        if not leg["ok"]:
            print(f"  leg {leg['leg_id']} REJECTED: {leg['error']}")
        elif leg.get("acknowledged") is False:
            # A real broker order whose id could not be written back. Not a
            # rejection: it reconciles itself.
            print(f"  leg {leg['leg_id']} placed, acknowledgement unrecorded")
        else:
            print(f"  leg {leg['leg_id']} {leg['symbol']} -> {leg['broker_order_id']}")

    return response.get("run_id")


def stop_example(strategy_id):
    """Stop the current run: exit every owned position at market.

    An accepted stop is not necessarily flat. A 200 with stop_pending true
    means the request is durable and its exits were accepted, but the run stays
    open, subscribed and managed until fills prove every position is flat.
    Never infer flatness from the HTTP status.
    """
    response = client.strategystop(strategy_id=strategy_id)
    print_response("Stop", response)

    if response.get("stop_pending"):
        print("  exits are working; the run is still managed")
    for exit_ in response.get("exits", []):
        if not exit_["ok"]:
            print(f"  leg {exit_['leg_id']} exit REFUSED: {exit_['error']} - retry the stop")


def close_all_example(strategy_id):
    """Same stop mechanics as strategystop, different audit intent.

    A close_all_manual event is written before the stop persists, which proves
    an operator asked for a flatten. It is not proof the broker became flat -
    run_stopped is that terminal evidence.
    """
    print_response("Close all", client.strategycloseall(strategy_id=strategy_id))


def close_leg_example(strategy_id, leg_id):
    """Exit one leg at market; the run continues with the rest.

    leg_id is the id the wizard assigned within the strategy, the same value
    that appears in legs[].id on strategystatus. It is not an order id.
    """
    response = client.strategycloseleg(strategy_id=strategy_id, leg_id=leg_id)
    print_response("Close leg", response)

    # run_stopped reports only what this call could prove. A live broker
    # normally acknowledges before its fill, so even the last accepted exit
    # returns false and the fill finalises the run later.
    print("  run stopped by this call:", response.get("run_stopped"))


def history_examples(strategy_id):
    """Runs, orders and events: what happened and why."""
    runs = client.strategyruns(strategy_id=strategy_id, limit=10)
    print("\nRuns (newest first):")
    for run in runs.get("data", []):
        print(f"  run {run['id']:<4} {run['mode']:<8} {run['stop_reason'] or 'open':<16} "
              f"pnl={run['pnl_realized']:<10} peak={run['pnl_peak']} trough={run['pnl_trough']}")

    # Orders are oldest first, so an entry always precedes its exit. This is
    # the opposite ordering to runs and events.
    orders = client.strategyorders(strategy_id=strategy_id)
    print("\nOrders (oldest first):")
    for o in orders.get("data", []):
        print(f"  leg {o['leg_id']} {o['kind']:<18} {o['action']} {o['qty']} "
              f"{o['symbol']:<26} {o['status']:<10} @ {o['avg_fill_price']}")

    # Narrow a long history with run_id. A run belonging to another strategy
    # matches nothing rather than leaking its orders.
    latest = (runs.get("data") or [{}])[0].get("id")
    if latest:
        print_response(f"Orders for run {latest}",
                       client.strategyorders(strategy_id=strategy_id, run_id=latest))

    events = client.strategyevents(strategy_id=strategy_id, limit=20)
    print("\nEvents (newest first):")
    for e in events.get("data", []):
        print(f"  {e['severity']:<8} {e['kind']:<24} {e['message']}")

    # The events an operator must not ignore.
    critical = client.strategyevents(strategy_id=strategy_id, severity="critical")
    for e in critical.get("data", []):
        print(f"  CRITICAL {e['kind']}: {e['message']}")

    # Filter by kind. An out-of-vocabulary kind is a 400, not an empty list.
    print_response("Stop-failure events",
                   client.strategyevents(strategy_id=strategy_id, kind="run_stop_failed"))


# ----------------------------------------------------------------------
# Public webhook
# ----------------------------------------------------------------------

def batch_webhook_example():
    """Drive a batch strategy from an alert.

    The token is shown exactly once, in the browser, when the strategy is
    created or its token is rotated. No endpoint returns it. Treat it as a
    credential: anyone who can post to the URL can start or stop the strategy.
    """
    strategy = Strategy(
        host_url="http://127.0.0.1:5000",
        webhook_token="oaws_your_webhook_token_here"
    )

    with strategy:
        # mode is required on start and never defaulted.
        print_response("Webhook start", strategy.start("sandbox"))

        # A stop is accepted, not necessarily flat. Read stop_pending.
        response = strategy.stop()
        print_response("Webhook stop", response)
        if response.get("stop_pending"):
            print("  exits are working; the run is still managed")


def signal_webhook_example():
    """Drive a signal strategy from an alert: one alert moves one leg.

    A signal that does nothing is a success with a note, not a failure:
    "Signal accepted (already_long)". Reporting a no-op as a failure invites a
    retry, and a retry on an order path is how one alert becomes two positions.
    """
    strategy = Strategy(
        host_url="http://127.0.0.1:5000",
        webhook_token="oaws_your_webhook_token_here"
    )

    with strategy:
        # Name the leg by its id...
        print_response("Long entry by leg id", strategy.long_entry(leg_id=1))

        # ...or by symbol and exchange. leg_id wins when both are given.
        print_response("Short exit by symbol",
                       strategy.short_exit(symbol="RELIANCE", exchange="NSE"))

        # An opposite entry squares the existing side first, then opens.
        print_response("Short entry", strategy.short_entry(leg_id=1))
        print_response("Long exit", strategy.long_exit(leg_id=1))


def webhook_result_handling_example():
    """Every documented outcome is returned, not raised.

    The result label is the contract: a 200 rejected_dedupe and a 409
    rejected_cooling_off both need reading rather than a traceback.
    """
    strategy = Strategy(
        host_url="http://127.0.0.1:5000",
        webhook_token="oaws_your_webhook_token_here"
    )

    with strategy:
        response = strategy.start("sandbox")
        result = response.get("result")

        if result == "ok":
            print(f"accepted, run {response.get('run_id')}")
        elif result == "rejected_dedupe":
            # An identical signal within the last 60 seconds. Reported as a
            # success because the caller's intent was already satisfied.
            print("duplicate delivery, already handled")
        elif result == "rejected_cooling_off":
            # The strategy stopped within the last 30 seconds, so a start is
            # held off: a misconfigured pair of alerts cannot oscillate.
            print("cooling off, try again shortly")
        elif result == "rejected_live_disabled":
            print("enable live trading on the strategy page first")
        elif result == "rejected_token":
            print("unknown or rotated token")
        else:
            print(f"refused: {result} - {response.get('message')}")


def main():
    """Run the strategy examples. Set STRATEGY_ID to one of your own."""
    STRATEGY_ID = 1

    try:
        list_strategies_example()
        status_example(STRATEGY_ID)
        history_examples(STRATEGY_ID)

        # Lifecycle, in sandbox mode. Uncomment to place simulated orders.
        # start_example(STRATEGY_ID)
        # close_leg_example(STRATEGY_ID, leg_id=2)
        # stop_example(STRATEGY_ID)
        # close_all_example(STRATEGY_ID)

        # Webhook, needs a real oaws_ token from the browser.
        # batch_webhook_example()
        # signal_webhook_example()
        # webhook_result_handling_example()
    finally:
        client.close()


if __name__ == "__main__":
    main()
