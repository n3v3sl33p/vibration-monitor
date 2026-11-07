"""Stateful block-compatible exponential moving average."""
from numbers import Integral, Real
import numpy as np


def ema(values, period=3, last=None):
    """Return (filtered, last); first sample initializes an empty EMA state."""
    if isinstance(period, bool) or not isinstance(period, Integral) or period <= 0:
        raise ValueError("период EMA должен быть положительным целым числом")
    if last is not None and (isinstance(last, bool) or not isinstance(last, Real) or not np.isfinite(last)):
        raise ValueError("состояние EMA должно быть конечным числом")
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("вход EMA должен быть конечным одномерным массивом")
    result = np.empty_like(values)
    alpha = 2 / (period + 1)
    for i, value in enumerate(values):
        last = value if last is None else alpha * value + (1 - alpha) * last
        result[i] = last
    return result, last
