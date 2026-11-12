"""Shared setup for the backtest test suite.

These tests never contact an OpenAlgo server or a broker. Anything that needs a
client uses an in-process fake.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
