"""Finite, index-clocked signal acquisition; no wall-clock timing or threads."""
from dataclasses import dataclass, field
from numbers import Integral, Real
import numpy as np
from generator import SignalConfig, generate
from processing import ema


@dataclass
class SessionBuffer:
    """An append-only finite-session buffer. Rejected blocks leave it unchanged."""
    capacity: int
    t: np.ndarray = field(init=False)
    raw: np.ndarray = field(init=False)
    quantized: np.ndarray = field(init=False)
    filtered: np.ndarray = field(init=False)
    mask: np.ndarray = field(init=False)
    saturation_count: int = field(default=0, init=False)

    def __post_init__(self):
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int) or self.capacity <= 0:
            raise ValueError("вместимость буфера должна быть положительным целым")
        self.t = np.empty(0)
        self.raw = np.empty(0)
        self.quantized = np.empty(0)
        self.filtered = np.empty(0)
        self.mask = np.empty(0, dtype=bool)

    def append(self, t, raw, quantized, filtered, mask, saturation_count=0):
        arrays = [np.asarray(v, dtype=float) for v in (t, raw, quantized, filtered)]
        mask = np.asarray(mask, dtype=bool)
        if any(v.ndim != 1 for v in [*arrays, mask]) or len({len(v) for v in [*arrays, mask]}) != 1:
            raise ValueError("длины одномерных массивов блока должны совпадать")
        if len(self.t) + len(arrays[0]) > self.capacity:
            raise ValueError("превышена вместимость буфера сеанса")
        if not all(np.isfinite(v).all() for v in arrays):
            raise ValueError("значения блока должны быть конечными")
        times = arrays[0]
        if np.any(np.diff(times) <= 0) or (len(self.t) and len(times) and times[0] <= self.t[-1]):
            raise ValueError("время должно строго возрастать между блоками")
        if (isinstance(saturation_count, bool) or not isinstance(saturation_count, int)
                or not 0 <= saturation_count <= len(times)):
            raise ValueError("некорректное число насыщений блока")
        self.t, self.raw, self.quantized, self.filtered = [
            np.concatenate((old, new)) for old, new in zip(
                (self.t, self.raw, self.quantized, self.filtered), arrays)]
        self.mask = np.concatenate((self.mask, mask))
        self.saturation_count += saturation_count


def quantize(values, bits=12, xmin=-4.0, xmax=4.0):
    """Uniform endpoint-inclusive ADC; ties are rounded to the even code."""
    if isinstance(bits, bool) or not isinstance(bits, Integral) or not 1 <= bits <= 32:
        raise ValueError("разрядность АЦП должна быть целым числом от 1 до 32")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) for v in (xmin, xmax)):
        raise ValueError("границы диапазона АЦП должны быть конечными числами")
    if xmin >= xmax or not np.isfinite(xmax - xmin):
        raise ValueError("диапазон АЦП должен иметь конечную положительную ширину")
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("вход АЦП должен быть конечным одномерным массивом")
    step = (xmax - xmin) / (2**bits - 1)
    result = xmin + np.rint((np.clip(values, xmin, xmax) - xmin) / step) * step
    return result, int(np.count_nonzero((values < xmin) | (values > xmax)))


def run_session(config: SignalConfig, bits=12, xmin=-4.0, xmax=4.0,
                ema_period=3, chunk_size=128) -> SessionBuffer:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size <= 0:
        raise ValueError("размер блока должен быть положительным целым числом")
    count = round(config.fs * config.duration)
    buffer = SessionBuffer(capacity=count)
    rng = np.random.default_rng(config.seed)
    last = None
    for start in range(0, count, chunk_size):
        t = np.arange(start, min(start + chunk_size, count)) / config.fs
        raw, mask = generate(t, config, rng)
        quantized, saturated = quantize(raw, bits, xmin, xmax)
        filtered, last = ema(quantized, ema_period, last)
        buffer.append(t, raw, quantized, filtered, mask, saturated)
    return buffer
