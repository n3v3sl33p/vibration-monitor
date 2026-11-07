"""Vertical RED→GREEN regression tests for the finite acquisition pipeline."""
import numpy as np
import pytest


def test_normal_session_tracer_bullet():
    from generator import SignalConfig
    from acquisition import run_session

    cfg = SignalConfig(amplitudes=(1.0,), frequencies=(1.0,), fs=8, duration=1,
                       noise_std=0)
    result = run_session(cfg)
    np.testing.assert_array_equal(result.t, np.arange(8) / 8)
    np.testing.assert_allclose(result.raw, np.sin(2 * np.pi * result.t))
    assert len(result.quantized) == len(result.filtered) == len(result.mask) == 8
    assert not result.mask.any()
    assert np.isfinite(result.filtered).all()
    assert result.saturation_count == 0


def test_generate_uses_given_clock_and_reproducible_rng():
    from generator import SignalConfig, generate

    cfg = SignalConfig()
    t = np.arange(1200) / cfg.fs
    raw, mask = generate(t, cfg, np.random.default_rng(cfg.seed))
    expected = sum(a * np.sin(2 * np.pi * f * t)
                   for a, f in zip(cfg.amplitudes, cfg.frequencies))
    expected += np.random.default_rng(42).normal(0, 0.1, len(t))
    np.testing.assert_array_equal(raw, expected)
    assert not mask.any()
    np.testing.assert_array_equal(raw, generate(t, cfg, np.random.default_rng(42))[0])
    assert not np.array_equal(raw, generate(t, cfg, np.random.default_rng(43))[0])


def test_quantize_clips_to_exact_levels_and_counts_saturation():
    from acquisition import quantize

    values = np.array([-5, -4, -2.8, 0, 2.8, 4, 5.0])
    result, count = quantize(values, bits=3, xmin=-4, xmax=4)
    step = 8 / 7
    expected = -4 + np.rint((np.clip(values, -4, 4) + 4) / step) * step
    np.testing.assert_array_equal(result, expected)
    assert count == 2
    assert result.min() == -4 and result.max() == 4
    np.testing.assert_allclose((result + 4) / step,
                               np.rint((result + 4) / step), atol=1e-14)


def test_ema_known_recurrence_and_continuation():
    from processing import ema

    values = np.array([1.0, 2.0, 4.0, -1.0])
    result, last = ema(values, period=9)
    np.testing.assert_allclose(result, [1, 1.2, 1.76, 1.208])
    assert last == result[-1]
    first, state = ema(values[:2], period=9)
    second, state = ema(values[2:], period=9, last=state)
    np.testing.assert_array_equal(np.r_[first, second], result)
    empty, same_state = ema([], period=9, last=state)
    assert empty.size == 0 and same_state == state
    np.testing.assert_array_equal(ema(values, period=1)[0], values)


def test_buffer_append_is_bounded_aligned_and_atomic():
    from acquisition import SessionBuffer

    buffer = SessionBuffer(capacity=3)
    assert buffer.t.size == 0
    buffer.append([0, 0.1], [1, 2], [1, 2], [1, 1.2], [False, True], 1)
    buffer.append([0.2], [3], [3], [1.56], [False], 0)
    np.testing.assert_array_equal(buffer.t, [0, 0.1, 0.2])
    assert buffer.saturation_count == 1
    before = buffer.raw.copy()
    with pytest.raises(ValueError, match="вместимость"):
        buffer.append([0.3], [4], [4], [2], [False])
    np.testing.assert_array_equal(buffer.raw, before)
    partial = SessionBuffer(capacity=5)
    with pytest.raises(ValueError, match="длины"):
        partial.append([0, 0.1], [1], [1, 2], [1, 2], [False, False])
    assert partial.t.size == 0
    with pytest.raises(ValueError, match="врем"):
        partial.append([0.1, 0], [1, 2], [1, 2], [1, 2], [False, False])
    with pytest.raises(ValueError, match="конеч"):
        partial.append([0], [float('nan')], [1], [1], [False])
    with pytest.raises(ValueError, match="вместимость"):
        SessionBuffer(capacity=0)


def test_chunked_acquisition_keeps_index_clock_noise_and_ema_state(monkeypatch):
    from acquisition import SessionBuffer, run_session
    from generator import SignalConfig

    lengths = []
    real_append = SessionBuffer.append

    def record_real_append(self, t, *args, **kwargs):
        lengths.append(len(t))
        return real_append(self, t, *args, **kwargs)

    monkeypatch.setattr(SessionBuffer, "append", record_real_append)
    cfg = SignalConfig()
    whole = run_session(cfg, chunk_size=1200)
    lengths.clear()
    blocks = run_session(cfg, chunk_size=127)
    assert lengths == [min(127, 1200 - start) for start in range(0, 1200, 127)]
    np.testing.assert_array_equal(blocks.t, np.arange(1200) / 20)
    assert blocks.capacity == len(blocks.t) == 1200
    assert np.all(np.diff(blocks.t) > 0)
    for field in ('t', 'raw', 'quantized', 'filtered', 'mask'):
        np.testing.assert_array_equal(getattr(blocks, field), getattr(whole, field))
    singles = run_session(cfg, chunk_size=1)
    np.testing.assert_array_equal(singles.filtered, whole.filtered)


def test_amplitude_event_is_half_open_and_doubles_only_harmonics():
    from generator import SignalConfig, generate

    t = np.array([19.95, 20.0, 20.05, 29.95, 30.0, 30.05])
    cfg = SignalConfig(scenario='amplitude')
    raw, mask = generate(t, cfg, np.random.default_rng(42))
    expected_mask = (t >= 20) & (t < 30)
    base = sum(a * np.sin(2 * np.pi * f * t)
               for a, f in zip(cfg.amplitudes, cfg.frequencies))
    expected = base * np.where(expected_mask, 2, 1)
    expected += np.random.default_rng(42).normal(0, 0.1, len(t))
    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_equal(raw, expected)


def test_impulse_is_one_rectangular_pulse_and_mask_marks_only_its_support():
    from generator import SignalConfig, generate

    cfg = SignalConfig(scenario='impulse', noise_std=0)
    t = np.arange(1200) / cfg.fs
    raw, mask = generate(t, cfg, np.random.default_rng(42))
    base = sum(a * np.sin(2 * np.pi * f * t)
               for a, f in zip(cfg.amplitudes, cfg.frequencies))
    expected_mask = (t >= 20) & (t < 20.1)
    assert expected_mask.sum() == 2
    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_equal(raw, base + 2 * expected_mask)


def test_frequency_event_integrates_first_harmonic_phase_across_both_edges():
    from generator import SignalConfig, generate
    from acquisition import run_session

    cfg = SignalConfig(scenario='frequency', noise_std=0,
                       frequencies=(0.073, 0.1, 0.4))
    t = np.array([19.999999, 20, 20.000001, 29.999999, 30, 30.000001, 40])
    raw, mask = generate(t, cfg, np.random.default_rng(42))
    phase_time = t + np.clip(t - 20, 0, 10)
    expected = np.sin(2 * np.pi * 0.073 * phase_time)
    expected += 0.5 * np.sin(2 * np.pi * 0.1 * t)
    expected += 0.25 * np.sin(2 * np.pi * 0.4 * t)
    np.testing.assert_allclose(raw, expected, atol=1e-14)
    np.testing.assert_array_equal(mask, (t >= 20) & (t < 30))
    assert abs(raw[0] - raw[2]) < 1e-5
    assert abs(raw[3] - raw[5]) < 1e-5
    whole = run_session(cfg, chunk_size=1200)
    blocks = run_session(cfg, chunk_size=137)
    np.testing.assert_array_equal(blocks.raw, whole.raw)
    np.testing.assert_array_equal(blocks.mask, whole.mask)


@pytest.mark.parametrize('updates', [
    {'amplitudes': ()}, {'frequencies': ()}, {'amplitudes': (1,)},
    {'amplitudes': (1, float('nan'), 1)}, {'amplitudes': (1, -1, 1)},
    {'frequencies': (0.05, float('inf'), 0.4)}, {'frequencies': (0, 0.1, 0.4)},
    {'fs': 0}, {'fs': -1}, {'fs': float('inf')}, {'fs': True},
    {'duration': 0}, {'duration': -1}, {'duration': float('nan')},
    {'duration': 1.001}, {'duration': 1e7}, {'noise_std': -0.1},
    {'noise_std': float('nan')}, {'seed': -1}, {'seed': 1.5}, {'seed': True},
    {'scenario': 'unknown'}, {'fs': 0.8},
    {'scenario': 'frequency', 'frequencies': (5, 0.1, 0.4)},
    {'scenario': 'amplitude', 'event_start': -1},
    {'scenario': 'amplitude', 'event_end': 61},
    {'scenario': 'amplitude', 'event_start': 30},
    {'event_start': float('nan')}, {'amplitude_factor': 0},
    {'frequency_factor': float('inf')}, {'impulse_width': 0},
    {'scenario': 'impulse', 'impulse_width': 11}, {'impulse_height': 0},
])
def test_config_rejects_invalid_parameters(updates):
    from generator import SignalConfig

    with pytest.raises(ValueError):
        SignalConfig(**updates)


@pytest.mark.parametrize('values, updates', [
    ([0], {'bits': 0}), ([0], {'bits': 1.5}), ([0], {'bits': True}),
    ([0], {'bits': 33}), ([0], {'xmin': 4}), ([0], {'xmax': -4}),
    ([0], {'xmin': float('nan')}), ([0], {'xmax': float('inf')}),
    ([float('inf')], {}), ([float('nan')], {}), ([[1, 2]], {}),
])
def test_quantize_rejects_invalid_adc_input(values, updates):
    from acquisition import quantize

    with pytest.raises(ValueError):
        quantize(values, **updates)


@pytest.mark.parametrize('values, updates', [
    ([1], {'period': 0}), ([1], {'period': -1}), ([1], {'period': 1.5}),
    ([1], {'period': True}), ([float('nan')], {}), ([[1]], {}),
    ([1], {'last': float('inf')}),
])
def test_ema_rejects_invalid_input(values, updates):
    from processing import ema

    with pytest.raises(ValueError):
        ema(values, **updates)


@pytest.mark.parametrize('chunk_size', [-1, True, 1.5])
def test_run_session_rejects_invalid_chunk_size(chunk_size):
    from acquisition import run_session
    from generator import SignalConfig

    with pytest.raises(ValueError, match='блок'):
        run_session(SignalConfig(), chunk_size=chunk_size)


def test_default_ema_period_is_experimentally_selected_three():
    from acquisition import run_session
    from generator import SignalConfig
    from processing import ema

    np.testing.assert_array_equal(ema([1, 2, 4])[0], [1, 1.5, 2.75])
    cfg = SignalConfig(noise_std=0)
    result = run_session(cfg)
    np.testing.assert_array_equal(result.filtered, ema(result.quantized, period=3)[0])
