"""Signals & Systems tests: filter responses, stability, zero-phase, sampling, convolution, Parseval."""

import numpy as np
import pytest
from scipy import signal as sg

from src.data_loader import generate_synthetic_ecg
from src.sns_lab import (
    FS, alias_frequency, bandpass_sos, conv_manual, highpass_sos, notch_sos,
)


def _gain_db(sos, f):
    _, h = sg.sosfreqz(sos, worN=[f], fs=FS)
    return 20 * np.log10(abs(h[0]) + 1e-12)


@pytest.mark.parametrize("f0", [50.0, 60.0])
def test_notch_attenuates_mains_and_passes_ecg_band(f0):
    sos = notch_sos(f0, 30.0)
    assert _gain_db(sos, f0) < -40.0
    assert _gain_db(sos, 10.0) > -0.5


def test_bandpass_edges_are_minus_3db():
    sos = bandpass_sos(0.5, 40.0, 4)
    assert _gain_db(sos, 0.5) == pytest.approx(-3.0, abs=0.5)
    assert _gain_db(sos, 40.0) == pytest.approx(-3.0, abs=0.5)
    assert _gain_db(sos, 10.0) > -0.2


def test_highpass_blocks_dc_and_passes_qrs_band():
    sos = highpass_sos(0.5, 1)
    assert _gain_db(sos, 0.01) < -25.0
    assert _gain_db(sos, 20.0) > -0.1


@pytest.mark.parametrize("sos", [
    highpass_sos(0.5, 1), highpass_sos(0.5, 4), notch_sos(50.0, 30.0), bandpass_sos(0.5, 40.0, 4),
])
def test_filters_are_stable_all_poles_inside_unit_circle(sos):
    _, p, _ = sg.sos2zpk(sos)
    assert np.all(np.abs(p) < 1.0)


def test_notch_zeros_lie_on_unit_circle():
    z, _, _ = sg.sos2zpk(notch_sos(50.0, 30.0))
    assert np.allclose(np.abs(z), 1.0, atol=1e-9)


def test_filtfilt_is_zero_phase_while_causal_filter_delays_qrs():
    sig, _, ann, _ = generate_synthetic_ecg(
        duration=20.0, fs=FS, rhythm_type="normal", noise_level=0.01,
        power_noise_level=0.0, baseline_noise_level=0.0, seed=5)
    sos = bandpass_sos(0.5, 40.0, 4)
    yc, yz = sg.sosfilt(sos, sig), sg.sosfiltfilt(sos, sig)
    p = int(ann[ann > 6 * FS][0])
    lo, hi = p - int(0.1 * FS), p + int(0.4 * FS)
    shift_z = abs(lo + int(np.argmax(yz[lo:hi])) - p)
    shift_c = lo + int(np.argmax(yc[lo:hi])) - p
    assert shift_z <= 1
    assert shift_c > shift_z


def test_filtfilt_squares_the_magnitude_response():
    sos = bandpass_sos(0.5, 40.0, 4)
    n = 4096
    t = np.arange(n) / FS
    x = np.sin(2 * np.pi * 40.0 * t)
    y = sg.sosfiltfilt(sos, x)
    measured = np.max(np.abs(y[n // 4: 3 * n // 4]))
    expected = 10 ** (2 * _gain_db(sos, 40.0) / 20)
    assert measured == pytest.approx(expected, rel=0.05)


def test_alias_frequency_formula():
    assert alias_frequency(50, 90) == pytest.approx(40.0)
    assert alias_frequency(50, 360) == pytest.approx(50.0)
    assert alias_frequency(50, 72) == pytest.approx(22.0)


def test_sampled_tone_is_indistinguishable_from_its_alias():
    fs, f = 90.0, 50.0
    n = np.arange(200)
    assert np.allclose(np.cos(2 * np.pi * f * n / fs),
                       np.cos(2 * np.pi * alias_frequency(f, fs) * n / fs), atol=1e-9)


def test_manual_convolution_matches_numpy():
    rng = np.random.RandomState(0)
    x, h = rng.randn(500), np.array([2, 1, 0, -1, -2]) / 8.0
    assert np.allclose(conv_manual(x, h), np.convolve(x, h), atol=1e-12)


def test_parseval_theorem():
    x = np.random.RandomState(1).randn(1000)
    e_t = np.sum(x ** 2)
    e_f = np.sum(np.abs(np.fft.fft(x)) ** 2) / len(x)
    assert e_t == pytest.approx(e_f, rel=1e-9)


def test_fourier_series_converges_with_more_harmonics():
    sig, _, ann, _ = generate_synthetic_ecg(
        duration=10.0, fs=FS, rhythm_type="normal", noise_level=0.0,
        power_noise_level=0.0, baseline_noise_level=0.0, seed=13)
    seg = sig[int(ann[4]) - int(0.25 * FS): int(ann[5]) - int(0.25 * FS)]
    X = np.fft.rfft(seg)

    def err(N):
        Xk = np.zeros_like(X)
        Xk[:N + 1] = X[:N + 1]
        return np.mean((np.fft.irfft(Xk, len(seg)) - seg) ** 2)

    assert err(5) > err(20) > err(60)
