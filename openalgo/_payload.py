"""Values a REST payload carries, made ready for JSON."""

import numpy


def plain_value(value):
    """Return `value` in a form JSON can carry.

    A NumPy scalar becomes the Python value it holds: numpy.int64(65) becomes 65 and
    numpy.bool_(True) becomes True. A figure read from pandas (.iloc, .at, iterrows, a
    lot-size column) is a NumPy scalar, and JSON cannot carry a NumPy integer or bool as
    it is. Any other value is returned as it is, so the server still checks it and
    refuses what it does not accept.
    """
    return value.item() if isinstance(value, numpy.generic) else value
