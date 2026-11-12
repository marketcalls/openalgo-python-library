# -*- coding: utf-8 -*-
"""
OpenAlgo REST API Documentation - Strategy Module Methods
    https://docs.openalgo.in

The API-key surface for OpenAlgo's ``/strategy`` module: multi-leg options
strategies with end-to-end risk management, plus the signal-driven mode that
reacts to individual alerts.

It is lifecycle plus reads only. Building a strategy stays in the browser
wizard at ``/strategy``; nothing here can create a strategy, edit its
configuration, enable live trading, rotate a webhook token or delete anything.

Five rules worth knowing before calling anything in this module:

1. ``mode`` on :meth:`StrategyAPI.strategystart` is required and is never
   defaulted, here or on the server. It is a keyword argument with no default,
   so omitting it is a TypeError rather than a live order.
2. Live is opt-in per strategy. A strategy is created sandbox-only and
   ``mode="live"`` is refused with a 409 until the operator enables live
   trading on the strategy page.
3. A strategy that is not yours answers 404, identical to one that does not
   exist, so the id space cannot be probed.
4. An accepted stop is not proof of flatness. Read ``stop_pending`` and the
   per-leg outcomes; a 200 with ``stop_pending: true`` means the exits were
   accepted and are still working.
5. No endpoint returns a webhook token. Only its SHA-256 digest is stored, and
   the plaintext is shown once in the browser at creation and rotation.
"""

from .base import BaseAPI

#: Run modes. Exact and case-sensitive on the server: "LIVE" is not "live",
#: and a near miss such as "paper" is refused rather than read as sandbox.
RUN_MODES = ("live", "sandbox")

#: Strategy statuses, accepted by strategylist's status filter.
STRATEGY_STATUSES = ("stopped", "running", "paused", "errored")

#: Event severities, accepted by strategyevents.
EVENT_SEVERITIES = ("info", "warn", "critical")


class StrategyAPI(BaseAPI):
    """
    Strategy module API methods for OpenAlgo.
    Inherits from the BaseAPI class.

    Every route is a POST with the identifier in the body. External alert senders
    (charting platforms, spreadsheets, screeners) cannot always choose a method or
    set a header, so there is no GET form and no path parameter.
    """

    def _strategy_request(self, endpoint, payload, **kwargs):
        """POST to a /api/v1/strategy/ route, dropping unset optional filters.

        A ``None`` filter is the same as omitting it, so it is not sent at all
        rather than sent as JSON null.
        """
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value
        return self._post(f"strategy/{endpoint}", payload)

    def strategylist(self, *, status=None, q=None, **kwargs):
        """
        List the strategies this API key owns, newest first.

        Parameters:
        - status (str, optional): Filter by strategy status - stopped, running,
          paused or errored. An out-of-vocabulary value is a 400, not an empty list.
        - q (str, optional): Case-insensitive substring match on the strategy
          name, 100 characters at most.
        - **kwargs: Passed through unchanged for future API extensions.

        Returns:
        dict: {"status": "success", "data": [...]} with one object per strategy
        carrying id, name, strategy_kind, direction, product, the risk settings,
        live_enabled, status, current_run_id and last_finalized_run.

        Notes:
        - The list form omits ``legs``. Call strategystatus for one strategy's legs.
        - For a stopped strategy, ``last_finalized_run.pnl_realized`` is the
          durable final P&L; do not infer it from an earlier checkpoint.
        """
        return self._strategy_request(
            "list", {"apikey": self.api_key}, status=status, q=q, **kwargs
        )

    def strategystatus(self, *, strategy_id, **kwargs):
        """
        Get one strategy's full configuration including legs, and its current run.

        Parameters:
        - strategy_id (int): Strategy id, a positive integer. Required.
        - **kwargs: Passed through unchanged for future API extensions.

        Returns:
        dict: {"status": "success", "data": {...}, "run": {...} or null}. ``run``
        is null whenever the strategy has no current run, which is the normal
        state of a stopped strategy.

        Notes:
        - A populated ``run.stop_requested_reason`` means a stop is durable but
          not yet confirmed flat. The run is still current and still managed.
        - ``run.resolved_expiries`` is the snapshot taken at run start, so a
          positional strategy does not roll to a new contract mid-run.
        - Prefer ``run`` over the strategy's own ``status`` when you need to know
          whether anything is actually open.
        """
        return self._strategy_request(
            "status", {"apikey": self.api_key, "strategy_id": int(strategy_id)}, **kwargs
        )

    def strategystart(self, *, strategy_id, mode, **kwargs):
        """
        Start a run of a batch strategy. Every leg's entry order is placed.

        Parameters:
        - strategy_id (int): Strategy id, a positive integer. Required.
        - mode (str): "live" or "sandbox". Required, exact and case-sensitive.
          There is no default anywhere in the chain - the default a hurried
          reader would reach for is the one that places real orders.
        - **kwargs: Passed through unchanged for future API extensions.

        Returns:
        dict: {"status": "success", "run_id": int, "mode": str, "legs": [...]}
        with one entry per leg carrying leg_id, ok, acknowledged, symbol,
        broker_order_id and error.

        Notes:
        - Partial success is a 200. Check ``legs[].ok`` rather than assuming
          every leg reached the market.
        - ``ok: true`` with ``acknowledged: false`` is a real broker order whose
          id could not be written back, not a rejection. It reconciles itself.
        - A run whose entries were all rejected is closed immediately and the
          call answers 400.
        - A second start against a running strategy answers 409, so two triggers
          firing at once cannot both place a full set of entries.
        - This is for batch strategies. A signal strategy has no start: its first
          inbound signal after the platform session boundary opens the run.
        """
        payload = {
            "apikey": self.api_key,
            "strategy_id": int(strategy_id),
            # Passed through exactly as given. The SDK does not lower-case or
            # otherwise normalise it, so the server's strictness stands: a typo
            # is refused rather than quietly read as sandbox.
            "mode": mode,
        }
        return self._strategy_request("start", payload, **kwargs)

    def strategystop(self, *, strategy_id, **kwargs):
        """
        Stop the strategy's current run: exit every owned position at market.

        Parameters:
        - strategy_id (int): Strategy id, a positive integer. Required. The run
          is resolved from the strategy; a caller never supplies a run id.
        - **kwargs: Passed through unchanged for future API extensions.

        Returns:
        dict: {"status": "success", "run_id": int, "stop_pending": bool,
        "exits": [...]} with one entry per leg carrying leg_id, ok,
        position_ref, exit_owner and error.

        Notes:
        - An accepted stop is not necessarily flat. ``stop_pending: true`` means
          the request is durable and its exits were accepted, but the run stays
          open, subscribed and managed until fills prove every position is flat.
          Never infer flatness from the HTTP status.
        - 409 when nothing is running, and 409 can also carry
          ``stop_pending: true`` when an unfilled entry or a refused exit still
          needs management. Retry the stop in that case.
        - A leg whose entry was accepted but not filled is not exited and the
          stop is reported as refused: there is no confirmed quantity to close.
        """
        return self._strategy_request(
            "stop", {"apikey": self.api_key, "strategy_id": int(strategy_id)}, **kwargs
        )

    def strategycloseall(self, *, strategy_id, **kwargs):
        """
        Close every leg of the current run, recording an operator's intent.

        Same stop mechanics as strategystop, different audit intent: a
        ``close_all_manual`` event is written before the stop persists, which
        proves an operator asked for a flatten. It is not proof that the broker
        became flat - ``run_stopped`` is that terminal evidence.

        Parameters:
        - strategy_id (int): Strategy id, a positive integer. Required.
        - **kwargs: Passed through unchanged for future API extensions.

        Returns:
        dict: {"status": "success", "run_id": int, "stop_pending": bool,
        "exits": [...]}, the same shape as strategystop.
        """
        return self._strategy_request(
            "close_all", {"apikey": self.api_key, "strategy_id": int(strategy_id)}, **kwargs
        )

    def strategycloseleg(self, *, strategy_id, leg_id, **kwargs):
        """
        Exit one leg of the current run at market. The run continues with the rest.

        Parameters:
        - strategy_id (int): Strategy id, a positive integer. Required.
        - leg_id (int): The leg's id within the strategy - the same value that
          appears in ``legs[].id`` on strategystatus. Not an order id and not a
          run id. Required.
        - **kwargs: Passed through unchanged for future API extensions.

        Returns:
        dict: {"status": "success", "run_id": int, "leg_id": int,
        "run_stopped": bool, "exits": [...]}.

        Notes:
        - ``run_stopped`` reports only what this call could prove. A live broker
          normally acknowledges before its fill, so even the last accepted exit
          returns false and the fill finalises the run later.
        - A refused exit is a failure, not a close: read the per-leg ``ok``.
          The leg stays retryable, so its stop, its target and the scheduler's
          square-off can all still reach it.
        - A leg_id that names no open leg is a 409, not a 404. The strategy was
          found; the leg is simply not in a state that can be closed.
        """
        payload = {
            "apikey": self.api_key,
            "strategy_id": int(strategy_id),
            "leg_id": int(leg_id),
        }
        return self._strategy_request("close_leg", payload, **kwargs)

    def strategyruns(self, *, strategy_id, limit=None, **kwargs):
        """
        Get every activation of a strategy, newest first.

        Parameters:
        - strategy_id (int): Strategy id, a positive integer. Required.
        - limit (int, optional): How many runs to return, 1 to 500. Server
          default is 100. A value outside the range is a 400, not a clamp.
        - **kwargs: Passed through unchanged for future API extensions.

        Returns:
        dict: {"status": "success", "data": [...]} with one object per run
        carrying id, mode, broker, started_at, stopped_at, stop_reason,
        stop_requested_at, stop_requested_reason, pnl_realized, pnl_peak,
        pnl_trough, trigger_source, webhook_event_id and resolved_expiries.

        Notes:
        - The row at the top is the current run only while ``stopped_at`` is null.
        - ``pnl_peak`` and ``pnl_trough`` are authoritative only once the run has
          stopped; while it is open the checkpoints are the authority.
        - An overall threshold triggers an exit, it does not promise the result.
          Market exits fill at the available bid/ask, so ``pnl_realized`` can
          differ from the threshold that caused the stop.
        """
        return self._strategy_request(
            "runs", {"apikey": self.api_key, "strategy_id": int(strategy_id)},
            limit=limit, **kwargs
        )

    def strategyorders(self, *, strategy_id, run_id=None, **kwargs):
        """
        Get every order the engine placed across a strategy's runs.

        Parameters:
        - strategy_id (int): Strategy id, a positive integer. Required.
        - run_id (int, optional): Narrow the result to one run. A run belonging
          to another strategy matches nothing rather than leaking its orders.
        - **kwargs: Passed through unchanged for future API extensions.

        Returns:
        dict: {"status": "success", "data": [...]} with one object per order
        carrying id, run_id, leg_id, kind, position_ref, broker_order_id,
        symbol, exchange, action, qty, product, pricetype, price, status,
        placed_at, filled_at, avg_fill_price, filled_qty and reject_reason.

        Notes:
        - Rows are oldest first, so an entry always precedes its exit. This is
          the opposite ordering to strategyruns and strategyevents.
        - A row is written before the broker answers, so an order can appear
          with status "pending" and a null broker_order_id. An order that
          reached the broker but was never recorded would be invisible to
          crash recovery.
        - A positive ``filled_qty`` proves exposure in every status, including
          cancelled and rejected.
        - There is no limit parameter here. Narrow with run_id instead.
        """
        return self._strategy_request(
            "orders", {"apikey": self.api_key, "strategy_id": int(strategy_id)},
            run_id=run_id, **kwargs
        )

    def strategyevents(self, *, strategy_id, run_id=None, kind=None,
                       severity=None, limit=None, **kwargs):
        """
        Get the risk-event audit trail for a strategy, newest first.

        Parameters:
        - strategy_id (int): Strategy id, a positive integer. Required.
        - run_id (int, optional): Narrow the result to one run. Configuration
          events carry run_id null, so this filter excludes them.
        - kind (str, optional): Filter by event kind. Must be a member of the
          event-kind vocabulary; anything else is a 400.
        - severity (str, optional): info, warn or critical.
        - limit (int, optional): How many events to return, 1 to 1000. Server
          default is 500. A value outside the range is a 400, not a clamp.
        - **kwargs: Passed through unchanged for future API extensions.

        Returns:
        dict: {"status": "success", "data": [...]} with one object per event
        carrying id, run_id, strategy_id, ts, kind, severity, leg_id, message
        and payload.

        Event kinds an operator should not ignore:
        - run_stop_requested (info): the stop is durable and new signal entries
          are gated. It is not proof the broker is flat.
        - run_stop_failed (critical): the broker refused a stop's exits and the
          run is still holding those positions.
        - order_ack_unrecorded (critical): the broker accepted an order but its
          acknowledgement could not be written; it reconciles itself.
        - leg_expiry_fallback (warn): the chain did not list the expiry rank the
          leg asked for, so a nearer one was used.
        - flip_outgoing_exit_rejected (critical): the outgoing side of a signal
          flip is still held.

        Notes:
        - The trail is append-only; nothing updates or deletes a row.
        - ``payload`` is free-form JSON and null on most events. Do not assume
          one shape across kinds.
        """
        return self._strategy_request(
            "events", {"apikey": self.api_key, "strategy_id": int(strategy_id)},
            run_id=run_id, kind=kind, severity=severity, limit=limit, **kwargs
        )
