"""Shared setup for the pytest suites under test/.

Makes the repository root importable so the suites run against the source tree. These
suites never contact an OpenAlgo server or a broker. The older scripts in this folder
that do need a live server are run by hand, not collected by CI.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
