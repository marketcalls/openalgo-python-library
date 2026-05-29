# -*- coding: utf-8 -*-
"""
TA-Lib-compatible indicators added to OpenAlgo (Rust/numpy backend via _backend).

These fill gaps where TA-Lib had a function OpenAlgo lacked, and they match TA-Lib's
values exactly (by definition). Price transforms, MIDPOINT/MIDPRICE, MOM, the ROC
variants (ROCP/ROCR/ROCR100), and APO.
"""
import numpy as np
import pandas as pd
from typing import Union
from .base import BaseIndicator
from . import _backend


class _Single(BaseIndicator):
    """Helper for single-series-in / single-series-out indicators."""
    _fn = None
    _args = ()

    def calculate(self, data, *params):
        validated, input_type, index = self.validate_input(data)
        result = type(self)._fn(validated, *params)
        return self.format_output(result, input_type, index)


class MOM(BaseIndicator):
    def __init__(self):
        super().__init__("Momentum")

    def calculate(self, data, period: int = 10):
        validated, input_type, index = self.validate_input(data)
        return self.format_output(_backend.mom(validated, period), input_type, index)


class ROCP(BaseIndicator):
    def __init__(self):
        super().__init__("ROCP")

    def calculate(self, data, period: int = 10):
        validated, input_type, index = self.validate_input(data)
        return self.format_output(_backend.rocp(validated, period), input_type, index)


class ROCR(BaseIndicator):
    def __init__(self):
        super().__init__("ROCR")

    def calculate(self, data, period: int = 10):
        validated, input_type, index = self.validate_input(data)
        return self.format_output(_backend.rocr(validated, period), input_type, index)


class ROCR100(BaseIndicator):
    def __init__(self):
        super().__init__("ROCR100")

    def calculate(self, data, period: int = 10):
        validated, input_type, index = self.validate_input(data)
        return self.format_output(_backend.rocr100(validated, period), input_type, index)


class MIDPOINT(BaseIndicator):
    def __init__(self):
        super().__init__("MidPoint")

    def calculate(self, data, period: int = 14):
        validated, input_type, index = self.validate_input(data)
        return self.format_output(_backend.midpoint(validated, period), input_type, index)


class APO(BaseIndicator):
    def __init__(self):
        super().__init__("APO")

    def calculate(self, data, fast_period: int = 12, slow_period: int = 26, ma_type: str = "SMA"):
        validated, input_type, index = self.validate_input(data)
        return self.format_output(_backend.apo(validated, fast_period, slow_period, ma_type),
                                  input_type, index)


class _HLC(BaseIndicator):
    """Helper for high/low(/close)-input price transforms (single output)."""
    def _hlc(self, high, low, close=None):
        high_data, input_type, index = self.validate_input(high)
        low_data, _, _ = self.validate_input(low)
        if close is None:
            high_data, low_data = self.align_arrays(high_data, low_data)
            return high_data, low_data, None, input_type, index
        close_data, _, _ = self.validate_input(close)
        high_data, low_data, close_data = self.align_arrays(high_data, low_data, close_data)
        return high_data, low_data, close_data, input_type, index


class MEDPRICE(_HLC):
    def __init__(self):
        super().__init__("Median Price")

    def calculate(self, high, low):
        h, l, _, it, idx = self._hlc(high, low)
        return self.format_output(_backend.medprice(h, l), it, idx)


class TYPPRICE(_HLC):
    def __init__(self):
        super().__init__("Typical Price")

    def calculate(self, high, low, close):
        h, l, c, it, idx = self._hlc(high, low, close)
        return self.format_output(_backend.typprice(h, l, c), it, idx)


class WCLPRICE(_HLC):
    def __init__(self):
        super().__init__("Weighted Close Price")

    def calculate(self, high, low, close):
        h, l, c, it, idx = self._hlc(high, low, close)
        return self.format_output(_backend.wclprice(h, l, c), it, idx)


class MIDPRICE(_HLC):
    def __init__(self):
        super().__init__("Midpoint Price")

    def calculate(self, high, low, period: int = 14):
        h, l, _, it, idx = self._hlc(high, low)
        return self.format_output(_backend.midprice(h, l, period), it, idx)


class AVGPRICE(BaseIndicator):
    def __init__(self):
        super().__init__("Average Price")

    def calculate(self, open_prices, high, low, close):
        o, input_type, index = self.validate_input(open_prices)
        h, _, _ = self.validate_input(high)
        l, _, _ = self.validate_input(low)
        c, _, _ = self.validate_input(close)
        o, h, l, c = self.align_arrays(o, h, l, c)
        return self.format_output(_backend.avgprice(o, h, l, c), input_type, index)


# --- TA-Lib-faithful directional-movement family (matches TA-Lib, not TradingView) ---

class PLUS_DM(_HLC):
    def __init__(self):
        super().__init__("Plus Directional Movement")

    def calculate(self, high, low, period: int = 14):
        h, l, _, it, idx = self._hlc(high, low)
        return self.format_output(_backend.plus_dm(h, l, period), it, idx)


class MINUS_DM(_HLC):
    def __init__(self):
        super().__init__("Minus Directional Movement")

    def calculate(self, high, low, period: int = 14):
        h, l, _, it, idx = self._hlc(high, low)
        return self.format_output(_backend.minus_dm(h, l, period), it, idx)


class DX(_HLC):
    def __init__(self):
        super().__init__("Directional Movement Index")

    def calculate(self, high, low, close, period: int = 14):
        h, l, c, it, idx = self._hlc(high, low, close)
        return self.format_output(_backend.dx(h, l, c, period), it, idx)


class ADXR(_HLC):
    def __init__(self):
        super().__init__("Average Directional Movement Rating")

    def calculate(self, high, low, close, period: int = 14):
        h, l, c, it, idx = self._hlc(high, low, close)
        return self.format_output(_backend.adxr(h, l, c, period), it, idx)


class STOCHF(_HLC):
    def __init__(self):
        super().__init__("Stochastic Fast")

    def calculate(self, high, low, close, fastk_period: int = 5, fastd_period: int = 3):
        h, l, c, it, idx = self._hlc(high, low, close)
        k, d = _backend.stochf(h, l, c, fastk_period, fastd_period)
        return self.format_multiple_outputs((k, d), it, idx)


class LINEARREG_ANGLE(BaseIndicator):
    def __init__(self):
        super().__init__("Linear Regression Angle")

    def calculate(self, data, period: int = 14):
        validated, input_type, index = self.validate_input(data)
        return self.format_output(_backend.linreg_angle(validated, period), input_type, index)


class LINEARREG_INTERCEPT(BaseIndicator):
    def __init__(self):
        super().__init__("Linear Regression Intercept")

    def calculate(self, data, period: int = 14):
        validated, input_type, index = self.validate_input(data)
        return self.format_output(_backend.linreg_intercept(validated, period), input_type, index)
