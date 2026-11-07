"""Discrete mathematical model of a virtual vibration sensor (arbitrary units)."""
from dataclasses import dataclass
from numbers import Real, Integral
import numpy as np


@dataclass(frozen=True)
class SignalConfig:
    amplitudes: tuple[float, ...] = (1.0, 0.5, 0.25)
    frequencies: tuple[float, ...] = (0.05, 0.1, 0.4)
    fs: float = 20.0
    duration: float = 60.0
    noise_std: float = 0.1
    seed: int = 42
    scenario: str = "normal"
    event_start: float = 20.0
    event_end: float = 30.0
    amplitude_factor: float = 2.0
    frequency_factor: float = 2.0
    impulse_width: float = 0.1
    impulse_height: float = 2.0

    def __post_init__(self):
        a = np.asarray(self.amplitudes, dtype=float)
        f = np.asarray(self.frequencies, dtype=float)
        if a.ndim != 1 or f.ndim != 1 or not len(a) or len(a) != len(f):
            raise ValueError("A и F должны быть непустыми одномерными векторами одинаковой длины")
        if not np.isfinite(a).all() or not np.isfinite(f).all() or np.any(a < 0) or np.any(f <= 0):
            raise ValueError("A должны быть конечными ≥ 0, F — конечными > 0")
        scalars = (self.fs, self.duration, self.noise_std, self.event_start, self.event_end,
                   self.amplitude_factor, self.frequency_factor, self.impulse_width, self.impulse_height)
        if any(isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) for v in scalars):
            raise ValueError("числовые параметры должны быть конечными числами")
        if self.fs <= 0 or self.duration <= 0 or self.noise_std < 0:
            raise ValueError("fs и длительность должны быть > 0, шум ≥ 0")
        samples = self.fs * self.duration
        if (not np.isfinite(samples) or not 1 <= samples <= 1_000_000
                or not np.isclose(samples, round(samples), rtol=0, atol=1e-9)):
            raise ValueError("fs × длительность должно быть целым числом от 1 до 1 000 000")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or self.seed < 0:
            raise ValueError("seed должен быть неотрицательным целым числом")
        if self.scenario not in ("normal", "impulse", "amplitude", "frequency"):
            raise ValueError("неизвестный сценарий")
        if min(self.amplitude_factor, self.frequency_factor, self.impulse_width, self.impulse_height) <= 0:
            raise ValueError("сила и длительность отклонения должны быть > 0")
        if self.scenario != "normal":
            if not 0 <= self.event_start < self.event_end <= self.duration:
                raise ValueError("интервал события должен находиться внутри сеанса: 0 ≤ начало < конец ≤ длительность")
            if self.scenario == "impulse" and self.impulse_width > self.event_end - self.event_start:
                raise ValueError("импульс должен помещаться в интервал события")
        max_frequency = float(np.max(f))
        if self.scenario == "frequency":
            max_frequency = max(max_frequency, f[0] * self.frequency_factor)
        if self.fs <= 2 * max_frequency:
            raise ValueError("условие Найквиста: fs должна быть больше удвоенной максимальной частоты гармоник")
        object.__setattr__(self, "amplitudes", tuple(a))
        object.__setattr__(self, "frequencies", tuple(f))


def generate(t, config: SignalConfig, rng: np.random.Generator):
    """Sample harmonics and discrete Gaussian noise at the supplied times."""
    t = np.asarray(t, dtype=float)
    raw = np.zeros(len(t))
    for k, (a, f) in enumerate(zip(config.amplitudes, config.frequencies)):
        phase_time = t
        if config.scenario == "frequency" and k == 0:
            # Integrate the piecewise frequency; retain the phase offset after the event.
            phase_time = t + (config.frequency_factor - 1) * np.clip(
                t - config.event_start, 0, config.event_end - config.event_start)
        raw += a * np.sin(2 * np.pi * f * phase_time)
    mask = np.zeros(len(t), dtype=bool)
    if config.scenario == "frequency":
        mask = (t >= config.event_start) & (t < config.event_end)
    elif config.scenario == "amplitude":
        mask = (t >= config.event_start) & (t < config.event_end)
        raw *= np.where(mask, config.amplitude_factor, 1)
    elif config.scenario == "impulse":
        mask = (t >= config.event_start) & (t < config.event_start + config.impulse_width)
        raw += config.impulse_height * mask
    raw += rng.normal(0, config.noise_std, len(t))
    return raw, mask
