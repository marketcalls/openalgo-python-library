"""Resolving a run's settings is split by concern, under the package's size guide of about
400 lines a module: the plan the kernel receives (`_run_plan`), the run's dates
(`_run_dates`), the risk settings (`_risk_settings`), and `_settings` resolving them
into one plan with its Assumptions block. The behaviour is the audit's
(`test_settings_audit`) and the runs', unchanged."""

from pathlib import Path

from openalgo.backtest import _risk_settings, _run_dates, _run_plan, _settings

#: The package's guide for the length of a module (CLAUDE.md, code standards).
SIZE_GUIDE_LINES = 400


def test_the_plan_the_dates_and_the_risk_settings_each_have_a_module():
    assert _settings.RunPlan is _run_plan.RunPlan
    assert _run_dates.run_dates.__module__ == "openalgo.backtest._run_dates"
    assert _risk_settings.risk_settings.__module__ == "openalgo.backtest._risk_settings"


def test_the_resolver_stays_under_the_size_guide():
    lines = Path(_settings.__file__).read_text(encoding="utf-8").count("\n")
    assert lines < SIZE_GUIDE_LINES
