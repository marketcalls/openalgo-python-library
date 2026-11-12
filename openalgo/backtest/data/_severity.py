"""The severities of a data quality finding (PRD 06 section 5), shared by every check.

- error: the data cannot be trusted; `load_bars` stops unless told otherwise.
- warning: the data is usable, and what is wrong or incomplete is counted.
- info: a conversion or a reading the user should know about.
"""

ERROR = "error"
WARNING = "warning"
INFO = "info"
SEVERITIES = (ERROR, WARNING, INFO)

__all__ = ["ERROR", "INFO", "SEVERITIES", "WARNING"]
