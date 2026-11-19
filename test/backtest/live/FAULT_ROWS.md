# Fault rows of the live session and the tests that cover them

16-session-design 7.2 lists the fault-injection scenarios F01 to F93 of a live session on
the fake OpenAlgo. Each row below names the tests (module and name, under
`test/backtest/live/`) that set up the row's fault calls and assert its Expected and
Assertion columns. A parametrized test is named with its parameter id in brackets.

Every one of these tests also runs the session invariants of 7.1 when it ends
(`session_invariants.py` with `session_invariants_changes.py`, `session_invariants_end.py`
and `session_invariants_stream.py`, over the order changes `session_trace.py` records as
the fake makes them, a sandbox modify included): `session_samples.rig` installs the
recording for every session test (`session_hooks.py`), and the `make_fake_openalgo`
fixture runs the checks. The checks "at any instant" run at every change of an order at
the fake, another client's included. A test whose row 7.1 lists (F35, F36, F60, the F77
variant, F80) leaves its leg without a confirmed stop with `trace.leave_unprotected`; the
leg must then be listed in `open_positions` with no confirmed stop and named by an ERROR
alert, and only the protection check is lifted for it. A test whose other client sells by
hand names what the decided end it plays leaves at the fake (`trace.kept_stop_by_decision`,
`trace.oversold_by_decision`; the section "Sales by hand" below). A test whose session
cannot keep one check waives that check alone, with its reason (`trace.waive`); every
other check still runs. `test_session_invariants.py` and `test_session_invariants_end.py`
check the checks on hand-written changes and runs, and the trace's recording on the fake.

| Row | Tests |
|---|---|
| F01 | `test_stops_unit8_rows.py::test_an_entry_bound_from_the_book_gets_one_stop_for_its_fill[F01]`, `test_session_orders.py::test_an_entry_whose_answer_is_lost_after_it_reached_the_broker_is_bound_once[F01]` |
| F02 | `test_session_orders.py::test_F02_an_entry_that_never_reached_the_broker_is_rejected_after_the_window` |
| F03 | `test_session_orders.py::test_F03_nothing_is_sent_for_any_instrument_between_a_timeout_and_an_answered_read`, `test_stops_unit8_rows.py::test_an_entry_bound_from_the_book_gets_one_stop_for_its_fill[F03]` |
| F04 | `test_session_orders.py::test_F04_a_gateway_error_is_not_taken_as_not_placed_within_the_brokers_window`, `test_stops_unit8_rows.py::test_an_entry_bound_from_the_book_gets_one_stop_for_its_fill[F04]` |
| F05 | `test_stops_unit8_rows.py::test_an_entry_bound_from_the_book_gets_one_stop_for_its_fill[F05]`, `test_session_orders.py::test_an_entry_whose_answer_is_lost_after_it_reached_the_broker_is_bound_once[F05]` |
| F06 | `test_session_orders.py::test_F06_an_entry_refused_a_connection_is_rejected_within_seconds` |
| F07 | `test_stops_unit8_rows.py::test_an_entry_bound_from_the_book_gets_one_stop_for_its_fill[F07]`, `test_session_orders.py::test_an_entry_whose_answer_is_lost_after_it_reached_the_broker_is_bound_once[F07]` |
| F08 | `test_session_orders.py::test_F08_a_rate_limited_entry_is_rejected_once_and_not_sent_again` |
| F09 | `test_stops_unit8_rows.py::test_F09_a_rate_limited_exit_after_the_stops_cancel_is_asked_for_again_and_fills`, `test_session_orders.py::test_F09_a_rate_limited_exit_is_asked_for_again_a_second_later_and_fills` |
| F10 | `test_session_faults_orders.py::test_F10_a_slow_entry_answer_leaves_no_false_unprotected_exit_and_the_stop_confirmed`, `test_session_orders.py::test_F10_a_slow_answer_moves_the_clock_and_the_session_catches_up` |
| F11 | `test_stops_placing.py::test_F11_a_stop_rejected_after_acceptance_exits_at_market_at_once` (unprotected under 2 s from the rejection to flat, after the stop's usual wait for its first word) |
| F12 | `test_stops_placing.py::test_F12_an_option_stop_refused_at_placement_exits_at_market_at_once` |
| F13 | `test_stops_placing.py::test_F13_an_options_stop_is_a_stop_limit_5_percent_beyond_its_trigger_on_the_tick` |
| F14 | `test_stops_placing.py::test_F14_a_stop_accepted_and_never_confirmed_exits_the_leg_after_10_seconds` |
| F15 | `test_stops_placing.py::test_F15_a_stop_placement_unanswered_holds_the_leg_alerts_at_once_and_binds_later` |
| F16 | `test_stops_placing.py::test_F16_a_stop_placement_that_never_reached_the_broker_is_placed_again` |
| F17 | `test_stops_placing.py::test_F17_a_part_fill_gets_a_stop_for_its_units_then_a_modify_to_the_whole` |
| F18 | `test_stops_placing.py::test_F18_an_add_modifies_the_stop_up_to_the_position` |
| F19 | `test_stops_exits.py::test_F19_a_partial_exit_cuts_the_stop_down_before_it_goes`, `test_stops_exits.py::test_F19_two_targets_cut_the_stop_for_the_first_then_cancel_it_for_the_second` |
| F20 | `test_stops_exits.py::test_F20_a_target_reached_on_a_tick_cancels_the_stop_then_exits` |
| F21 | `test_stops_exits.py::test_F21_a_stop_that_fills_while_its_cancel_is_sent_closes_the_leg_with_no_exit` (one orderstatus read of the stop after the cancel) |
| F22 | `test_stops_exits.py::test_F22_a_cancel_that_times_out_is_settled_by_a_read_before_the_exit_goes` |
| F23 | `test_stops_price.py::test_F23_a_stop_limit_run_through_is_repriced_to_a_marketable_limit_after_2_seconds` |
| F24 | `test_stops_placing.py::test_F24_a_stops_status_flips_change_nothing` |
| F25 | `test_stops_exits.py::test_F25_trailing_modifies_are_a_quarter_percent_and_5_seconds_apart` |
| F26 | `test_stops_exits.py::test_F26_a_trailing_modify_unanswered_is_settled_by_a_read_of_its_terms[timeout]`, `[bad_gateway]` (the variant) |
| F27 | `test_stops_exits.py::test_F27_a_trailing_modify_refused_keeps_the_old_trigger_until_the_next_is_accepted` |
| F28 | `test_stops_exits.py::test_F28_an_exit_rejected_five_times_halts_entries_and_places_the_stop_again` |
| F29 | `test_session_stream.py::test_F29_with_the_order_stream_silent_all_day_polls_give_the_same_trades`, `test_stops_unit8_rows.py::test_F29_with_the_order_stream_silent_the_stop_is_confirmed_by_its_polls` |
| F30 | `test_session_stream.py::test_F30_an_entry_fill_whose_update_is_lost_is_found_by_polling`, `test_stops_unit8_rows.py::test_F30_an_entry_fill_whose_words_are_lost_still_places_the_stop` |
| F31 | `test_session_stream.py::test_F31_an_update_that_arrives_twice_is_one_fill`, `test_stops_unit8_rows.py::test_F31_an_entry_word_that_arrives_twice_places_one_stop` |
| F32 | `test_session_stream.py::test_F32_updates_released_newest_first_fill_the_whole_order_at_its_average`, `test_stops_unit8_rows.py::test_F32_part_fills_released_newest_first_end_with_one_stop_of_the_whole` |
| F33 | `test_session_stream.py::test_F33_a_fill_reported_at_price_0_is_priced_from_the_tradebook`, `test_stops_unit8_rows.py::test_F33_a_fill_reported_at_price_0_and_priced_by_the_tradebook_places_the_stop` |
| F34 | `test_stops_held.py::test_F34_a_fill_the_tradebook_never_prices_is_released_at_the_feeds_price` |
| F35 | `test_session_guard.py::test_F35_after_the_switch_flips_to_sandbox_nothing_is_sent_or_read`, `test_session_guard.py::test_F35_a_session_locked_by_the_guard_ends_at_the_close_plus_its_grace`, `test_stops_unit8_rows.py::test_F35_after_a_mode_breach_the_live_stop_still_works_and_the_alert_names_it` |
| F36 | `test_session_faults_orders.py::test_F36_after_a_sandbox_session_sees_live_mode_nothing_reaches_the_live_book`, `test_session_guard.py::test_F36_after_the_switch_flips_to_live_nothing_reaches_the_live_book` |
| F37 | `test_session_guard.py::test_F37_semi_auto_mode_queues_the_entry_and_the_session_sends_nothing_more` |
| F38 | `test_stops_exits.py::test_F67_a_stop_that_fills_while_the_login_is_expired_is_found_after_the_restore` (a fill while expired: its word lost, found after the restore), `test_watch_feed.py::test_F38_an_expired_login_kills_the_feed_and_locks_entries`, `test_F38_refused_quotes_are_alerted_as_a_login_that_may_have_expired`, `test_F38_the_refused_quotes_alert_is_repeated_at_most_once_a_minute`, `test_F38_after_the_login_is_restored_the_lock_lifts_a_whole_slice_later`, `test_F38_the_session_ends_with_the_engine_and_the_broker_flat`, `test_F38_a_square_off_refused_while_the_login_is_expired_is_retried_until_it_fills`, `test_F38_a_square_off_due_while_the_books_cannot_be_read_waits_for_them` |
| F39 | `test_watch_feed.py::test_F39_no_entry_is_sent_while_the_login_is_revoked`, `test_F39_the_guards_failed_reads_are_alerted_after_three`, `test_F39_entries_resume_a_whole_slice_after_the_login_is_restored` |
| F40 | `test_watch_feed.py::test_F40_a_dead_feed_is_alerted_once`, `test_F40_a_dead_feed_locks_entries_from_10_s_after_its_last_tick`, `test_F40_the_lock_lifts_at_the_close_of_the_first_whole_slice_after_the_feed_returns`, `test_F40_quotes_are_read_every_2_s_while_the_feed_is_dead`, `test_F40_a_target_reached_on_a_quote_exits_while_the_feed_is_dead`, `test_F40_the_slice_of_the_dead_spell_makes_no_bar_and_is_marked_missing`, `test_F40_the_first_bar_after_the_dead_spell_holds_none_of_its_volume`, `test_F40_no_candle_is_read_while_the_feed_is_dead`, `test_F40_the_decision_on_a_bar_holding_a_dead_minute_is_skipped` |
| F41 | `test_watch_feed_drop.py::test_F41_a_fill_while_the_connection_was_down_is_found_by_the_stream_gaps_poll`, `test_F41_the_bar_builders_resume_over_the_outages_span`, `test_F41_the_sdk_replays_the_subscriptions_on_the_new_connection`, `test_F41_a_reconnect_held_past_the_dead_time_is_a_dead_feed_until_ticks_return` |
| F42 | `test_session_candles.py::test_F42_a_tick_gap_is_drift_and_the_series_continues_from_the_candle` |
| F43 | `test_stops_exits.py::test_F43_the_square_off_comes_by_the_clock_with_no_tick_and_cancels_the_stop_first` |
| F44 | `test_stops_exits.py::test_F44_a_refused_square_off_is_asked_for_again_with_alerts_each_minute_until_flat` (the "still not flat" ERROR at the square-off plus 30 s and again a minute later, then flat) |
| F45 | `test_stops_exits.py::test_F45_the_daily_loss_on_ticks_exits_mid_bar_and_locks_entries` (the later entries rejected `daily_loss_gate`) |
| F46 | `test_watch_kill_switch.py::test_F46_a_combined_loss_at_the_account_limit_fires_the_kill_switch_of_every_strategy`, `test_F46_the_kill_switch_exits_both_positions_as_risk_exits`, `test_F46_entries_stay_locked_for_the_day_after_the_kill_switch`, `test_F46_the_account_limit_is_alerted_once`, `test_F46_the_kill_switch_reports_each_strategy_flat_at_the_broker`, `test_F46_a_flat_report_waits_for_a_read_begun_after_the_strategy_went_flat` |
| F47 | `test_session_faults_start.py::test_F47_a_live_session_with_the_telegram_bot_stopped_fails_its_preflight`, `test_session_faults_start.py::test_F47_a_sandbox_session_with_the_telegram_bot_stopped_runs_and_counts_its_alerts` |
| F48 | `test_reconcile.py::test_F48_a_buy_by_hand_locks_the_strategys_entries_after_two_reads`, `test_reconcile.py::test_F48_nothing_is_sent_for_the_excess_bought_by_hand` |
| F49 | `test_reconcile.py::test_F49_the_stops_trade_is_attributed_to_the_stop_and_nothing_is_adopted` (orderstatus recovers after the attribution, and its COMPLETE changes nothing), `test_reconcile.py::test_F49_a_stop_filled_in_two_parts_is_attributed_part_by_part` (the variant) |
| F50 | `test_restart.py::test_F50_a_restart_resumes_the_position_with_its_stop_and_sends_nothing_new`, `test_F50_the_banner_says_it_is_a_restart_and_names_the_resumed_leg`, `test_F50_the_restarted_engine_trades_on_from_where_it_was`, `test_F50_trailing_resumes_after_the_restart` |
| F51 | `test_restart.py::test_F51_a_stop_that_filled_while_down_is_attributed_to_the_stop` |
| F52 | `test_restart.py::test_F52_a_missing_stop_is_alerted_at_the_start_and_placed_again`, `test_restart.py::test_F52_the_stop_is_working_before_the_banner` (variants: F75, F76) |
| F53 | `test_restart.py::test_F53_a_broker_position_on_a_traded_symbol_refuses_a_fresh_start` |
| F54 | `test_restart.py::test_F54_an_unbound_entry_is_settled_before_anything_is_sent` |
| F55 | `test_restart.py::test_F55_a_restart_past_the_square_off_flattens_in_the_start_phase` |
| F56 | `test_restart.py::test_F56_a_journal_bound_to_another_broker_refuses_the_start` |
| F57 | `test_stops_protect.py::test_F57_ctrl_c_mid_position_cancels_the_stop_then_exits_flat`, `test_session_stop.py::test_F57_ctrl_c_mid_position_exits_and_stops_flat`, `test_stops_protect.py::test_F57_a_second_ctrl_c_right_after_the_stops_cancel_protects_instead_of_exiting` (the variant) |
| F58 | `test_session_stop.py::test_F58_ctrl_c_in_the_countdown_sends_nothing_and_stops_within_a_second` |
| F59 | `test_session_stream.py::test_F59_another_clients_orders_on_the_stream_change_nothing` |
| F60 | `test_session_faults_orders.py::test_F60_an_entry_the_book_cannot_tell_apart_gets_no_stop_and_holds_the_instrument`, `test_F60_the_ambiguous_entrys_units_are_an_open_position_with_no_stop`, `test_F60_the_ambiguous_entrys_units_count_unprotected_until_the_end`, `test_session_orders.py::test_F60_an_entry_the_book_cannot_tell_apart_locks_the_instrument` |
| F61 | `test_watch_clock.py::test_F61a_the_banner_warns_that_zerodhas_feed_cannot_check_the_clock`, `test_F61a_the_results_warnings_repeat_it`, `test_F61a_no_estimate_is_made_and_nothing_is_alerted`, `test_F61b_the_estimate_is_about_3_s`, `test_F61b_a_clock_3_s_ahead_is_alerted_once`, `test_F61b_the_results_warnings_keep_the_skew`, `test_F61b_a_broker_with_exchange_times_gets_no_unmeasured_warning` |
| F62 | `test_stops_unit8_rows.py::test_F62_a_strategys_exception_cancels_the_stop_exits_then_is_raised`, `test_session_stop.py::test_F62_a_strategys_exception_stops_safely_then_is_raised` |
| F63 | `test_session_threads.py::test_F63_while_an_order_call_hangs_slices_close_and_decisions_are_made_on_time` |
| F64 | `test_session_threads.py::test_F64_an_entry_queued_behind_a_blocked_worker_expires_and_the_exit_goes` |
| F65 | `test_restart.py::test_F65_a_queued_entry_is_rejected_at_the_restart_and_its_approval_is_a_mismatch` |
| F66 | `test_stops_exits.py::test_F66_a_rate_limited_cancel_is_sent_again_a_second_after_its_read` |
| F67 | `test_stops_exits.py::test_F67_a_cancel_refused_while_the_login_is_expired_leaves_the_stop_working`, `test_F67_a_stop_that_fills_while_the_login_is_expired_is_found_after_the_restore` (the stop fills while expired). The row's "refused at 1, 2, 5 s, then every 5 s" is the retry of a refusal (429, F66); with the login expired OpenAlgo answers the cancel with the broker's error (HTTP 500), so the session reads the order instead. The test asserts that path: one cancel answered 500, at least three failed reads, the ERROR after the third, a second cancel after the restore answered 200, then the exit |
| F68 | `test_stops_held.py::test_F68_a_held_cancel_the_position_book_shows_unfilled_is_released_and_exits`, `test_stops_held.py::test_F68_a_held_cancel_of_a_stop_that_filled_part_releases_those_units_estimated` (the variant) |
| F69 | `test_stops_exits.py::test_F69_a_second_trade_of_the_day_records_its_own_stop`, `test_restart.py::test_F69_a_restart_during_the_second_trade_keeps_its_stop_as_the_legs` (the variant) |
| F70 | `test_session_guard.py::test_F70_a_sandbox_session_then_a_live_one_on_one_journal_keep_their_own_intents` |
| F71 | `test_stops_threads.py::test_F71_a_quantity_change_behind_a_queued_trailing_modify_goes_after_it_in_order` |
| F72 | `test_stops_price.py::test_F72_a_partly_filled_stop_is_repriced_for_its_total_units`, `test_stops_price.py::test_F72_a_reprice_refused_exits_the_rest_at_a_marketable_limit` (the variant) |
| F73 | `test_restart.py::test_F73_a_stop_saved_and_never_sent_is_placed_again_once` |
| F74 | `test_stops_protect.py::test_F74_a_safe_stop_whose_exit_is_unanswered_places_no_stop_and_reminds_each_minute` |
| F75 | `test_restart.py::test_F75_a_stop_asked_for_in_the_countdown_flattens_after_the_stop_is_placed` |
| F76 | `test_restart.py::test_F76_a_feed_that_refuses_to_connect_raises_with_the_stop_placed_again` |
| F77 | `test_stops_protect.py::test_F77_an_error_mid_cancel_then_exit_drops_the_exit_and_places_the_stop_again`, `test_stops_protect.py::test_F77_when_the_kernels_protect_fails_too_the_alert_lists_the_leg_without_a_stop` (the variant) |
| F78 | `test_reconcile.py::test_F78_a_lagging_position_book_is_alerted_as_a_lag_and_nothing_adopted`, `test_F78_the_stop_is_not_cancelled_while_the_book_lags`, `test_F78_once_the_book_catches_up_the_held_exits_go`, `test_F78_a_lag_that_lasts_past_the_in_flight_limit_is_alerted_to_check_the_account`, `test_F78_the_square_off_sends_no_exit_beyond_what_the_book_shows`, `test_F78_a_stop_placed_again_is_held_while_the_book_shows_fewer_units`; `test_reconcile_guard.py::test_F78_the_stop_stays_at_the_broker_while_the_flatten_waits_for_the_books`, `test_F78_the_stop_stays_at_the_broker_while_the_square_off_waits_for_the_books` |
| F79 | `test_stops_placing.py::test_F79_a_stop_cancelled_by_another_client_is_placed_again_once` |
| F80 | `test_stops_protect.py::test_F80_an_ambiguous_stop_placement_leaves_the_leg_marked_and_sends_nothing_more` (run past the 15:00 square-off, which is not sent) |
| F81 | `test_stops_placing.py::test_F81_an_add_whose_answer_is_lost_counts_its_units_unprotected_until_bound` |
| F82 | `test_stops_exits.py::test_F26_a_trailing_modify_unanswered_is_settled_by_a_read_of_its_terms[F82_lost_answer]` |
| F83 | `test_stops_exits.py::test_F83_a_cancel_whose_answer_is_lost_is_settled_by_a_read_and_exits` |
| F84 | `test_session_guard.py::test_F84_a_hung_guard_read_does_not_hold_back_an_exit_that_is_due` |
| F85 | `test_watch_feed_drop.py::test_F85_an_immediate_reconnect_resumes_the_bar_builders_over_its_span`, `test_F85_an_immediate_reconnect_polls_the_order_streams_gap`, `test_F85_the_sdk_replays_the_subscriptions_and_ticks_flow_again`, `test_F85_the_slice_of_the_drop_makes_no_bar_and_the_next_bar_holds_its_own_volume` |
| F86 | `test_session_stop.py::test_F86_a_feed_that_never_connects_stops_the_start_with_nothing_sent` |
| F87 | `test_watch_day.py::test_F87_before_the_open_the_preflight_runs_again_each_minute_until_the_login`, `test_F87_the_alert_test_is_sent_once`, `test_F87_a_broker_still_not_logged_in_at_09_00_is_alerted_then`, `test_F87_once_logged_in_the_session_trades_from_the_open`, `test_F87_the_quiet_market_before_the_open_is_no_dead_feed`, each with its `[expired]` login and its zero-funds variant `[zero_funds]` |
| F88 | `test_watch_day.py::test_F88_sbin_is_squared_off_at_15_00`, `test_F88_the_option_is_squared_off_at_15_10`, `test_F88_the_session_ends_once_both_are_flat`, `test_F88_sbin_falling_silent_after_its_square_off_is_no_dead_feed` |
| F89 | `test_stops_unit8_rows.py::test_F89_an_entry_whose_fill_is_late_blocks_a_second_and_gets_one_stop`, `test_session_stream.py::test_F89_an_entry_whose_fill_is_late_blocks_a_second_entry_live_in_flight` |
| F90 | `test_session_stream.py::test_F90_when_the_stream_resumes_the_trades_are_the_baseline_ones` |
| F91 | `test_stops_exits.py::test_F91_an_add_that_would_pass_the_freeze_quantity_is_rejected` |
| F92 | `test_stops_threads.py::test_F92_no_deadline_exit_while_the_worker_is_in_a_call_with_the_confirmation_waiting` |
| F93 | `test_restart.py::test_F93_a_stop_fill_journaled_after_the_last_save_is_in_the_journal_position` (restarted after the entry cut-off and run past the 15:00 square-off, which sends nothing) |

All 93 rows are covered.

## The alerts of 08-live-trading section 8

Each alert of "Operations and safety" is sent in a fault test through the session's alert
path to the fake's Telegram endpoint and asserted by its words as Telegram took them
(`test_session_faults_alerts.py`; an urgent alert's post carries its kind's priority). The
words are counted across the posts, since routine alerts that wait together leave in one
post: "once" means once in all of them.

| Alert | Test |
|---|---|
| Entry | `test_session_faults_alerts.py::test_F31_the_entry_alert_names_the_fill_once_though_its_word_came_twice` |
| Exit | `test_session_faults_alerts.py::test_F09_the_exit_alert_names_the_exit_that_filled_after_a_rate_limited_try` |
| Stop moves | `test_session_faults_alerts.py::test_F25_the_stop_moved_alert_names_the_brokers_new_trigger` |
| Rejects | `test_session_faults_alerts.py::test_F11_the_reject_alert_names_the_stop_and_the_brokers_reason` |
| Mode breach | `test_session_faults_alerts.py::test_F35_the_mode_breach_alert_says_the_switch_changed_and_nothing_more_is_sent` |
| Reconciliation mismatch | `test_session_faults_alerts.py::test_F48_the_reconciliation_mismatch_alert_names_both_figures` |
| Feed dead | `test_session_faults_alerts.py::test_F40_the_feed_dead_alert_says_entries_are_locked_and_quotes_watch_the_position` |
| Daily summary | `test_session_faults_alerts.py::test_F29_the_daily_summary_comes_last_with_the_days_figures` |

## The fake's extensions of 7.3

| Extension | Test of the fake |
|---|---|
| The order stream stops while the login is expired or revoked | `test_fake_openalgo_extensions.py::test_a_fill_while_the_brokers_login_is_gone_sends_no_order_update[expired]`, `[revoked]`, `test_an_update_lost_while_the_login_was_gone_stays_lost_once_it_is_back`, `test_a_sandbox_order_update_still_flows_while_the_login_is_expired`; a row test depends on it: `test_stops_exits.py::test_F67_a_stop_that_fills_while_the_login_is_expired_is_found_after_the_restore` |
| `direct_feed()` | `test_fake_openalgo_direct_feed.py` |
| `refuse_feed()` | `test_fake_openalgo_direct_feed.py::test_a_refused_feed_does_not_connect_and_fires_nothing_until_accepted` and the three after it |
| `lag_positionbook()`, `catch_up_positionbook()` | `test_fake_openalgo_extensions.py::test_a_lagging_position_book_answers_the_positions_from_before_the_latest_fills`, `test_the_order_and_trade_books_show_the_fills_the_lagging_position_book_does_not`, `test_a_position_book_caught_up_shows_every_fill_again` |
| The flavour whose quote-mode last trade time is the trade's exchange time in milliseconds | `test_fake_openalgo_feed.py::test_another_brokers_tick_carries_the_trades_time_in_milliseconds` |
| The broker's own MIS square-off (optional) | Not built: no row's test needs it (F44's variant and 5.3 row 1 are covered by the session's own square-off and the reconciler's adoption tests) |
| The client each call and order came from (for the invariants) | `test_fake_openalgo_extensions.py::test_each_client_of_the_fake_has_its_own_name`, `test_an_order_keeps_the_name_of_the_client_that_placed_it`, `test_the_log_names_the_client_of_every_call` |

## Sales by hand: decided ends beyond 7.1 (for the lead)

A sale by hand is the one change a session learns of only by its comparisons, and 5.3
fits the leg's stop to it on the second comparison that shows it. So at the instant of
the sale, and until that fit, the stop that was working stands beyond the position. The
invariants allow that, at every change, for exactly the exits that were working when the
other client's fill came, by no more than the sale left them beyond the position; an exit
placed after the sale, or the units a stop is raised by, get no such room
(`session_invariants_changes.SaleRoom`; self-tests in `test_session_invariants.py`).

Beyond that the invariants of 7.1 hold with no exemption unless the test names the decided
end it plays. These merged tests do, for the lead to add to 7.1 (or to send back as code
defects):

1. **The session's stop fills against what the sale left, through zero**
   (`trace.oversold_by_decision`; invariant 3). Allowed only for a stop that was working
   when the sale came, through zero by no more than the sale left it beyond the position,
   and only when the run lists the leg in `open_positions` at the fake's units with no
   confirmed stop and an ERROR alert names the strategy.

| Test | Decided end |
|---|---|
| `test_session_close_differing.py`: every test of `CASES` (first sighting, inline and threads, keep and flatten; after the line, inline and threads) and of `STALE` (dropped, inline and threads; slow, threads) | 5.3 (the stop is fitted on the second comparison that shows the sale); 5.4 "As built", rounds 7 to 9 (the close names the short it is the first to see). In the first-sighting cases the market moves inside the very comparison that first sees the sale |
| `test_session_close_fill_in_read.py`: every case `sold 4` (inline and threads, stream live and held, and the `UNJUDGED` ones) | 5.3; 5.4 "As built", round 9 (the stop fills inside the read of the second comparison) |

2. **The stop stays larger than what the sale left, or on the flat position it left**
   (`trace.kept_stop_by_decision`; invariants 2 and 4 at the end). Allowed only for a stop
   that was working when the sale came, beyond the position by no more than the sale left
   it, and only when an ERROR alert of the run names that stop's order ID.

| Test | Decided end |
|---|---|
| `test_session_close_differing.py`: the `all_sold` tests (inline and threads) | 5.4 "As built", round 8: the close names the stop working on a flat position |
| `test_session_close_hand_sale.py`: every test (the final read with no books; the cut held, dropped or asked for as the final read ends) | 3.3 step 5 and 5.4 "As built", rounds 6 to 8: the close does not fit the stop to a sale it has not settled, and names it |
| `test_reconcile_hand_sales.py::test_a_cut_the_broker_refuses_leaves_the_stop_for_10_and_nothing_claims_protection`, `test_the_error_alert_names_a_cut_the_broker_refused` | 3.3 step 5 "As built", round 7: a refused cut is named |
| `test_reconcile_hand_sales.py::test_the_error_alert_names_a_cut_with_no_answer_by_the_close_and_claims_no_protection` (inline only: on the threads the slow cut reaches the broker, so no stop stays beyond the position; unit 16 found the declaration unneeded there) | 3.3 step 5 "As built", round 7: a cut with no answer by the close |
| `test_reconcile_hand_sales.py::test_an_adoption_the_kernel_refuses_leaves_the_stop_for_10_and_sends_nothing`, `test_the_close_names_the_held_cancel_an_adoption_the_kernel_refused_made_way_for`, `test_the_mismatch_alert_of_an_adoption_the_kernel_refused_claims_no_adoption` | 3.3 step 5 "As built", round 8: an adoption the kernel refuses leaves the stop as it was |

What would remove the first list instead: fitting the stop at the first comparison that
shows an explained sale by hand. The kernel fits a stop only with an adoption
(`apply_external_fill`), and `Engine.modify` needs a decision point and keeps a stop's
role, so that means adopting an explained sale on its first sighting (a change of 5.3's
two comparisons, and of every unit 12 test built on them) or a kernel call that cuts a
stop alone (a Rust build). It would remove the "after the line" and fill-in-read cases;
the first-sighting cases need the exemption whatever is built.

As built (unit 16, the lead's decision in 7.1 "As built (unit 14"): another client's fill
on a traded symbol, reported by the order stream, prompts a comparison
`RECONCILE_AFTER_FILL_S` later, so the two comparisons come within seconds of the sale
instead of up to 30 s after it (`test_reconcile_prompted.py`). Each declaration above must
still be needed: the end checks find a decided end that no check took
(`session_invariants_end._declarations_needed`).

## Exemptions beyond 7.1's list (for the lead)

7.1 lists F35, F36, F60, the F77 variant and F80 as the tests that end with a leg without a
confirmed stop. These merged tests end so by a decided end of 16-session-design, named
with `trace.left_unprotected_by_decision`; each leg is listed in `open_positions` with no
confirmed stop and named by an ERROR alert, and only the protection check is lifted. They
are for the lead to add to 7.1's list:

| Test | Decided end |
|---|---|
| `test_reconcile.py::test_F78_a_stop_placed_again_is_held_while_the_book_shows_fewer_units` | 5.4 "As built" (unit 12): the exit guard holds a re-placed stop beyond the lagging book's units |
| `test_reconcile_hand_sales.py::test_the_error_alert_names_a_cut_with_no_answer_by_the_close_and_claims_no_protection` | 3.3 step 5 "As built": a cut with no answer by the close |
| `test_restart_stops.py::test_a_missing_stop_of_a_leg_the_books_show_fewer_units_of_is_not_promised_again` | 5.5 and 5.4 "As built": a lagging book and a stop cancelled while down |
| `test_session_close_lines.py` (a stop cancelled after the last lines; a stop held back at the last lines) | 3.3 step 5 "As built" |
| `test_session_threads.py` (an entry filled after the last lines; a stop cancelled in the final read) | 3.3 step 5 "As built" |
| `test_stops_exits.py` (a stop placed again with its answer lost at the safe stop) | 3.3 step 4 and F74 |
| `test_stops_protect.py::test_a_stop_whose_cancel_has_no_word_at_the_end_is_not_reported_confirmed` | 3.3 step 4: a second Ctrl+C before the cancel's word |
| `test_stops_report.py` (a second Ctrl+C with the exit unanswered; a stop repriced to a resting LIMIT) | 3.3 step 4; 4.6 |

Checks waived by single tests, each with its reason in the test:

| Test | Check waived | Why |
|---|---|---|
| `test_restart_stops.py::test_a_restored_stop_with_no_record_is_placed_again_and_no_exit_is_sent` | one stop per leg, exits within the position, no orphan | the test drops the venue record of a stop the broker holds, a state the save before every send never leaves |
| `test_session_threads.py::test_an_entry_unanswered_at_the_close_is_one_error_alert` | no orphan | the entry's answer comes after the session closed (3.3 step 5 "As built") |
| `test_session_close_hand_sale.py` (the cut held, dropped, or handed as the final read ends) and `test_reconcile_hand_sales.py` (the cut with no answer) | every modify answered | 3.3 step 5 "As built": the stop's cut is not sent or not answered by the close |
| `test_stops_protect.py::test_the_emergency_alert_names_an_exit_at_work_and_the_order_book_comes_first` | books agree | 3.5: the exit fills unheard and no books are read after an error; the alert names the exit in flight |
| `test_stops_report.py::test_an_exit_accepted_with_no_final_word_is_in_flight_and_never_beside_close_it_by_hand` | books agree | 3.3 step 4: the exit fills unheard and the books are unreadable; the leg's line names the exit in flight |
