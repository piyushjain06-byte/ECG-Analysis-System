"""
Signals & Systems Lab - interactive demonstrations of the S&S theory that the
rest of the project uses silently.

Place this file at  src/sns_lab.py  and call render_sns_lab_page() from app.py.

Tabs
  1  Filters as LTI systems (|H|, phase, group delay, pole-zero, h[n], step, difference equation)
  2  filtfilt (zero-phase) vs causal filtering
  3  Sampling, Nyquist and aliasing
  4  Convolution, LTI chain (Pan-Tompkins) and matched filter
  5  Windowing and spectral leakage
  6  STFT / spectrogram (time-frequency tradeoff)
  7  Fourier series and Parseval's theorem
  8  Quantitative filter evaluation (SNR, 50 Hz power, drift)
  9  FIR vs IIR
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import signal as sg

from src.data_loader import generate_synthetic_ecg
from src.preprocessing import full_dsp_pipeline
from src.signal_analysis import compute_psd_welch
from src.visualization import (
    AMBER, AXIS_LINE, CLEAN_TEAL, ECG_RED, FONT_COLOR, GRID, PAPER, PLOT_BG, PURPLE, TITLE_COLOR,
)

FS = 360.0
NYQ = FS / 2.0
BLUE = "#38BDF8"


# --------------------------------------------------------------------------- #
# Pure helpers (no Streamlit) - these are also unit-tested in tests/test_dsp.py
# --------------------------------------------------------------------------- #
def alias_frequency(f: float, fs: float) -> float:
    """Apparent frequency (Hz) of a tone f sampled at fs."""
    return float(abs(f - fs * np.round(f / fs)))


def conv_manual(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolution as a sum of shifted, scaled copies of h: y[n] = sum_k h[k] x[n-k]."""
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    y = np.zeros(len(x) + len(h) - 1)
    for k, hk in enumerate(h):
        y[k:k + len(x)] += hk * x
    return y


def group_delay_ms(f_hz: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Numerical group delay  tau = -d(phase)/d(omega)  in milliseconds."""
    phase = np.unwrap(np.angle(h))
    return -np.gradient(phase, 2 * np.pi * f_hz) * 1000.0


def highpass_sos(fc, order=1, fs=FS):
    return sg.butter(order, fc / (fs / 2), "highpass", output="sos")


def notch_sos(f0, q=30.0, fs=FS):
    b, a = sg.iirnotch(f0 / (fs / 2), q)
    return sg.tf2sos(b, a)


def bandpass_sos(lo=0.5, hi=40.0, order=4, fs=FS):
    return sg.butter(order, [lo / (fs / 2), hi / (fs / 2)], "band", output="sos")


def difference_equation(sos):
    """Difference equation text for low-order systems (<= 2nd order), else None."""
    b, a = sg.sos2tf(sos)
    if len(a) > 3:
        return None

    def idx(k):
        return "n" if k == 0 else f"n-{k}"

    xs = " ".join(f"{b[k] / a[0]:+.5f}*x[{idx(k)}]" for k in range(len(b)))
    ys = " ".join(f"{-a[k] / a[0]:+.5f}*y[{idx(k)}]" for k in range(1, len(a)))
    return f"y[n] = {xs} {ys}"


def spectrum_db(x, fs, window="boxcar"):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    w = sg.get_window(window, len(x))
    X = np.fft.rfft(x * w) / np.sum(w) * 2.0
    return np.fft.rfftfreq(len(x), 1.0 / fs), 20 * np.log10(np.abs(X) + 1e-9)


# --------------------------------------------------------------------------- #
# Plot + data helpers
# --------------------------------------------------------------------------- #
def _style(fig, title="", height=320, xt=None, yt=None):
    fig.update_layout(
        title=dict(text=title, font=dict(color=TITLE_COLOR, size=14)),
        height=height, paper_bgcolor=PAPER, plot_bgcolor=PLOT_BG, font=dict(color=FONT_COLOR),
        margin=dict(l=50, r=18, t=46, b=42), legend=dict(orientation="h", y=1.14),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS_LINE, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS_LINE, zeroline=False)
    if xt:
        fig.update_xaxes(title_text=xt)
    if yt:
        fig.update_yaxes(title_text=yt)
    return fig


def _show(fig):
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data(show_spinner=False)
def _pair(rhythm="normal", seed=11, hf=0.03, power=0.08, base=0.15, duration=20.0):
    """(noisy, clean, annotated R-peak indices). The clean signal uses the same
    seed with all noise sources set to zero, so the beats are identical and the
    clean version is a true ground truth."""
    kw = dict(duration=duration, fs=FS, heart_rate=72.0, rhythm_type=rhythm, seed=seed)
    noisy, _, ann, _ = generate_synthetic_ecg(
        noise_level=hf, power_noise_level=power, baseline_noise_level=base, **kw)
    clean, _, _, _ = generate_synthetic_ecg(
        noise_level=0.0, power_noise_level=0.0, baseline_noise_level=0.0, **kw)
    return noisy, clean, np.asarray(ann)


# --------------------------------------------------------------------------- #
# Tab 1 - filters as LTI systems
# --------------------------------------------------------------------------- #
def _tab_filters():
    st.markdown(
        "Every filter in the DSP pipeline is an **LTI system**, fully described by its "
        "transfer function $H(z)$ - its poles and zeros. Pick one and inspect it."
    )
    kind = st.radio(
        "System", ["High-pass (baseline removal)", "Notch (mains)", "Band-pass (QRS band)"],
        horizontal=True, key="sns_kind")
    c = st.columns(3)
    if kind.startswith("High"):
        fc = c[0].slider("Cutoff (Hz)", 0.05, 5.0, 0.5, 0.05, key="sns_hp_fc")
        order = c[1].slider("Butterworth order", 1, 6, 1, key="sns_hp_o")
        sos = highpass_sos(fc, order)
    elif kind.startswith("Notch"):
        f0 = c[0].selectbox("Mains frequency f0 (Hz)", [50.0, 60.0], key="sns_nf")
        q = c[1].slider("Quality factor Q", 2, 60, 30, key="sns_nq")
        sos = notch_sos(f0, q)
        c[2].caption(f"-3 dB bandwidth = f0/Q = {f0 / q:.2f} Hz. Higher Q = narrower notch, longer ringing.")
    else:
        lo = c[0].slider("Low cutoff (Hz)", 0.05, 5.0, 0.5, 0.05, key="sns_bp_lo")
        hi = c[1].slider("High cutoff (Hz)", 10.0, 100.0, 40.0, 1.0, key="sns_bp_hi")
        order = c[2].slider("Butterworth order (per edge)", 1, 6, 4, key="sns_bp_o")
        sos = bandpass_sos(lo, hi, order)

    w, h = sg.sosfreqz(sos, worN=4096, fs=FS)
    mag = 20 * np.log10(np.maximum(np.abs(h), 1e-6))
    phase = np.degrees(np.unwrap(np.angle(h)))
    gd = group_delay_ms(w, h)

    logx = st.checkbox("Log frequency axis (shows the 0.5 Hz region clearly)", value=False, key="sns_logx")
    m = w > 0 if logx else np.ones_like(w, dtype=bool)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("Magnitude |H(e^jw)| (dB)", "Phase (deg)", "Group delay (ms)"))
    fig.add_trace(go.Scatter(x=w[m], y=mag[m], line=dict(color=CLEAN_TEAL), name="|H| dB"), 1, 1)
    fig.add_trace(go.Scatter(x=w[m], y=phase[m], line=dict(color=PURPLE), name="phase"), 2, 1)
    fig.add_trace(go.Scatter(x=w[m], y=np.clip(gd[m], -50, 500), line=dict(color=AMBER), name="group delay"), 3, 1)
    fig.update_yaxes(range=[-80, 5], row=1, col=1)
    _style(fig, "Frequency-domain description", height=640, xt=None)
    fig.update_xaxes(title_text="Frequency (Hz)", row=3, col=1)
    if logx:
        fig.update_xaxes(type="log")
    fig.update_layout(showlegend=False)
    _show(fig)

    z, p, _ = sg.sos2zpk(sos)
    max_pole = float(np.max(np.abs(p))) if len(p) else 0.0
    left, right = st.columns(2)
    with left:
        th = np.linspace(0, 2 * np.pi, 400)
        pz = go.Figure()
        pz.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), mode="lines",
                                line=dict(color="#94A3B8", dash="dot"), name="unit circle"))
        pz.add_trace(go.Scatter(x=z.real, y=z.imag, mode="markers", name="zeros",
                                marker=dict(symbol="circle-open", size=12, color=CLEAN_TEAL, line=dict(width=2))))
        pz.add_trace(go.Scatter(x=p.real, y=p.imag, mode="markers", name="poles",
                                marker=dict(symbol="x", size=11, color=ECG_RED)))
        _style(pz, "Pole-zero plot (z-plane)", height=380, xt="Real", yt="Imag")
        pz.update_yaxes(scaleanchor="x", scaleratio=1)
        _show(pz)
        if max_pole < 1.0:
            st.success(f"STABLE (causal): all poles inside the unit circle. max |p| = {max_pole:.6f}")
        else:
            st.error(f"UNSTABLE: pole outside/on the unit circle. max |p| = {max_pole:.6f}")
    with right:
        n = int(FS * 2)
        delta = np.zeros(n)
        delta[0] = 1.0
        hn = sg.sosfilt(sos, delta)
        sn = sg.sosfilt(sos, np.ones(n))
        t = np.arange(n) / FS
        ts = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                           subplot_titles=("Impulse response h[n]", "Step response s[n]"))
        ts.add_trace(go.Scatter(x=t, y=hn, line=dict(color=BLUE)), 1, 1)
        ts.add_trace(go.Scatter(x=t, y=sn, line=dict(color=ECG_RED)), 2, 1)
        _style(ts, "Time-domain description", height=380)
        ts.update_xaxes(title_text="Time (s)", row=2, col=1)
        ts.update_layout(showlegend=False)
        _show(ts)

    de = difference_equation(sos)
    if de:
        st.markdown("**Difference equation** (from H(z)):")
        st.code(de)
    else:
        st.caption(f"Order {2 * len(sos)} system implemented as {len(sos)} cascaded second-order sections (SOS) "
                   "for numerical stability.")
    st.info("Zeros on the unit circle kill that frequency (notch zeros at +/-w0). Poles near the unit circle "
            "create sharp resonances; the closer to |z|=1, the longer the impulse response rings.")


# --------------------------------------------------------------------------- #
# Tab 2 - filtfilt vs causal
# --------------------------------------------------------------------------- #
def _tab_filtfilt():
    st.markdown(
        "The project uses `filtfilt` everywhere. This shows **why**: a causal filter delays and distorts the QRS; "
        "forward-backward filtering gives **zero phase** but squares the magnitude response."
    )
    noisy, _, ann = _pair("normal", 5, 0.01, 0.0, 0.0)
    sos = bandpass_sos(0.5, 40.0, 4)
    yc = sg.sosfilt(sos, noisy)
    yz = sg.sosfiltfilt(sos, noisy)
    p = int(ann[ann > 6 * FS][0])
    lo, hi = p - int(0.1 * FS), p + int(0.4 * FS)
    t = (np.arange(lo, hi) - p) / FS * 1000.0

    def rshift(y):
        return (lo + int(np.argmax(y[lo:hi])) - p) / FS * 1000.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=noisy[lo:hi], name="Input", line=dict(color="#94A3B8", dash="dot")))
    fig.add_trace(go.Scatter(x=t, y=yc[lo:hi], name="Causal (sosfilt)", line=dict(color=ECG_RED)))
    fig.add_trace(go.Scatter(x=t, y=yz[lo:hi], name="Zero-phase (sosfiltfilt)", line=dict(color=CLEAN_TEAL)))
    fig.add_vline(x=0, line_dash="dot", line_color="#94A3B8", annotation_text="true R-peak")
    _style(fig, "One heartbeat: causal vs zero-phase", height=360, xt="Time relative to R-peak (ms)", yt="mV")
    _show(fig)

    w, h = sg.sosfreqz(sos, worN=4096, fs=FS)
    i40 = int(np.argmin(np.abs(w - 40.0)))
    single = 20 * np.log10(abs(h[i40]))
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("R-peak shift, causal", f"{rshift(yc):+.1f} ms")
    k2.metric("R-peak shift, filtfilt", f"{rshift(yz):+.1f} ms")
    k3.metric("Gain at 40 Hz, single pass", f"{single:.2f} dB")
    k4.metric("Gain at 40 Hz, filtfilt", f"{2 * single:.2f} dB")

    gdc = group_delay_ms(w, h)
    mag1 = 20 * np.log10(np.maximum(np.abs(h), 1e-6))
    fm = go.Figure()
    fm.add_trace(go.Scatter(x=w, y=mag1, name="|H| (causal)", line=dict(color=ECG_RED)))
    fm.add_trace(go.Scatter(x=w, y=2 * mag1, name="|H|^2 (filtfilt)", line=dict(color=CLEAN_TEAL)))
    fm.update_xaxes(range=[0, 80])
    fm.update_yaxes(range=[-60, 3])
    _style(fm, "Effective magnitude response", height=300, xt="Frequency (Hz)", yt="dB")
    _show(fm)
    st.latex(r"\text{causal: } H(e^{j\omega})\qquad \text{filtfilt: } |H(e^{j\omega})|^2,\ \ \angle = 0")
    st.caption(f"The causal filter's group delay at 10 Hz is about {gdc[int(np.argmin(np.abs(w - 10)))]:.1f} ms and it "
               "varies with frequency (non-linear phase), which changes the QRS shape. Zero phase needs the whole "
               "signal in advance, so filtfilt is offline only; a real-time monitor must use a causal filter.")


# --------------------------------------------------------------------------- #
# Tab 3 - sampling and aliasing
# --------------------------------------------------------------------------- #
def _tab_sampling():
    st.markdown("**Nyquist:** a tone at $f$ sampled at $f_s$ appears at "
                r"$f_a = |f - f_s\,\mathrm{round}(f/f_s)|$. Above $f_s/2$ it folds back to a false frequency.")
    st.markdown("##### A. Single tone")
    c1, c2 = st.columns(2)
    f = c1.slider("Tone frequency f (Hz)", 1, 200, 50, key="sns_tone_f")
    fs_s = c2.slider("Sampling rate fs (Hz)", 20, 400, 90, 5, key="sns_tone_fs")
    fa = alias_frequency(f, fs_s)
    T = 0.2
    td = np.linspace(0, T, 4000)
    ts = np.arange(0, int(T * fs_s) + 1) / fs_s
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=td, y=np.cos(2 * np.pi * f * td), name=f"True {f} Hz", line=dict(color=BLUE)))
    fig.add_trace(go.Scatter(x=ts, y=np.cos(2 * np.pi * f * ts), mode="markers", name="Samples",
                             marker=dict(color=AMBER, size=9)))
    fig.add_trace(go.Scatter(x=td, y=np.cos(2 * np.pi * fa * td), name=f"What the samples look like: {fa:.1f} Hz",
                             line=dict(color=ECG_RED, dash="dash")))
    _style(fig, "Sampling a tone", height=320, xt="Time (s)", yt="Amplitude")
    _show(fig)
    if f > fs_s / 2:
        st.error(f"f = {f} Hz > fs/2 = {fs_s / 2:.0f} Hz -> ALIASING. The samples are indistinguishable from {fa:.1f} Hz.")
    else:
        st.success(f"f = {f} Hz <= fs/2 = {fs_s / 2:.0f} Hz: Nyquist satisfied, no aliasing.")

    st.markdown("##### B. ECG with 50 Hz hum, downsampled from 360 Hz")
    noisy, _, _ = _pair("normal", 3, 0.02, 0.15, 0.0)
    x = sg.sosfiltfilt(highpass_sos(0.5, 1), noisy)
    M = st.select_slider("Decimation factor M", options=[2, 3, 4, 5, 6, 8], value=4, key="sns_M")
    fs_new = FS / M
    naive = x[::M]
    proper = sg.decimate(x, M, ftype="iir", zero_phase=True)
    f0, s0 = spectrum_db(x, FS)
    f1, s1 = spectrum_db(naive, fs_new)
    f2, s2 = spectrum_db(proper, fs_new)
    a1, a2 = st.columns(2)
    with a1:
        fo = go.Figure(go.Scatter(x=f0, y=s0, line=dict(color=BLUE)))
        fo.add_vline(x=50, line_dash="dot", line_color="#94A3B8", annotation_text="50 Hz hum")
        _style(fo, f"Original, fs = {FS:.0f} Hz", height=320, xt="Frequency (Hz)", yt="dB")
        _show(fo)
    with a2:
        fd = go.Figure()
        fd.add_trace(go.Scatter(x=f1, y=s1, name="No anti-alias filter (x[::M])", line=dict(color=ECG_RED)))
        fd.add_trace(go.Scatter(x=f2, y=s2, name="With anti-alias filter (decimate)", line=dict(color=CLEAN_TEAL)))
        fd.add_vline(x=alias_frequency(50, fs_new), line_dash="dot", line_color=AMBER, annotation_text="alias of 50 Hz")
        _style(fd, f"Downsampled, fs = {fs_new:.0f} Hz (Nyquist {fs_new / 2:.0f} Hz)", height=320,
               xt="Frequency (Hz)", yt="dB")
        _show(fd)
    if 50 > fs_new / 2:
        st.warning(f"50 Hz is above the new Nyquist ({fs_new / 2:.0f} Hz). Without filtering it folds to "
                   f"{alias_frequency(50, fs_new):.1f} Hz, where it is indistinguishable from real signal. "
                   "The anti-alias filter removes it before sampling.")
    else:
        st.info("50 Hz is still below the new Nyquist, so it is just preserved (no aliasing). Try M = 4, 5, 6 or 8.")


# --------------------------------------------------------------------------- #
# Tab 4 - convolution, LTI chain, matched filter
# --------------------------------------------------------------------------- #
def _tab_convolution():
    st.markdown("The Pan-Tompkins detector is a **cascade of LTI systems** (derivative -> square -> moving-window "
                "integrator). The squarer is the one non-linear block.")
    hf = st.slider("EMG noise sigma (mV)", 0.0, 0.30, 0.05, 0.01, key="sns_conv_hf")
    noisy, clean, ann = _pair("normal", 21, hf, 0.0, 0.1)
    x = full_dsp_pipeline(noisy, FS)
    N = int(0.150 * FS)
    N += (N % 2 == 0)
    h_d = np.array([2, 1, 0, -1, -2]) / 8.0
    h_m = np.ones(N) / N

    xs = x[int(2 * FS):int(4 * FS)]
    yd = conv_manual(xs, h_d)[:len(xs)]
    sq = yd ** 2
    mw = conv_manual(sq, h_m)[:len(xs)]
    err = float(np.max(np.abs(conv_manual(xs, h_d) - np.convolve(xs, h_d))))
    t = np.arange(len(xs)) / FS

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("x[n]: band-passed ECG", "x * h_derivative", "squared (non-linear)",
                                        "squared * h_moving-average"))
    for r, (y, col) in enumerate([(xs, CLEAN_TEAL), (yd, PURPLE), (sq, ECG_RED), (mw, AMBER)], start=1):
        fig.add_trace(go.Scatter(x=t, y=y, line=dict(color=col)), r, 1)
    _style(fig, "Pan-Tompkins as a system chain", height=640)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_text="Time (s)", row=4, col=1)
    _show(fig)
    st.metric("max |manual convolution - np.convolve|", f"{err:.2e}")
    st.caption("Manual convolution = sum of shifted, scaled copies of h[n]. It matches `np.convolve` to machine precision.")

    l, r = st.columns(2)
    with l:
        fb = make_subplots(rows=2, cols=1, vertical_spacing=0.18,
                           subplot_titles=("h_derivative[n]", f"h_moving-average[n] (N={N})"))
        fb.add_trace(go.Bar(x=np.arange(len(h_d)), y=h_d, marker_color=PURPLE), 1, 1)
        fb.add_trace(go.Bar(x=np.arange(N), y=h_m, marker_color=AMBER), 2, 1)
        _style(fb, "Impulse responses (FIR)", height=360)
        fb.update_layout(showlegend=False)
        _show(fb)
    with r:
        wd, Hd = sg.freqz(h_d, 1, worN=2048, fs=FS)
        wm, Hm = sg.freqz(h_m, 1, worN=2048, fs=FS)
        fr = go.Figure()
        fr.add_trace(go.Scatter(x=wd, y=20 * np.log10(np.maximum(abs(Hd), 1e-6)), name="derivative (high-pass-ish)",
                                line=dict(color=PURPLE)))
        fr.add_trace(go.Scatter(x=wm, y=20 * np.log10(np.maximum(abs(Hm), 1e-6)), name="moving average (sinc-like low-pass)",
                                line=dict(color=AMBER)))
        fr.update_yaxes(range=[-70, 10])
        _style(fr, "Frequency responses", height=360, xt="Frequency (Hz)", yt="dB")
        _show(fr)

    st.markdown("##### Matched filter = cross-correlation with a QRS template")
    cf = full_dsp_pipeline(clean, FS)
    p0 = int(ann[5])
    pre, post = int(0.06 * FS), int(0.08 * FS)
    tpl = cf[p0 - pre:p0 + post].copy()
    tpl -= tpl.mean()
    tpl /= np.linalg.norm(tpl)
    c = sg.correlate(x, tpl, mode="same")
    peaks, _ = sg.find_peaks(c, height=0.4 * np.max(c), distance=int(0.25 * FS))
    peaks = peaks - (len(tpl) // 2 - pre)
    tol = int(0.05 * FS)
    ann_in = ann[(ann > FS) & (ann < len(x) - FS)]
    det = peaks[(peaks > FS) & (peaks < len(x) - FS)]
    tp = sum(np.min(np.abs(det - a)) <= tol for a in ann_in) if len(det) else 0
    fp = sum(np.min(np.abs(ann_in - d)) > tol for d in det) if len(ann_in) else 0
    m1, m2, m3 = st.columns(3)
    m1.metric("True beats", len(ann_in))
    m2.metric("Sensitivity", f"{tp / max(len(ann_in), 1):.1%}")
    m3.metric("False detections", int(fp))
    tt = np.arange(len(x)) / FS
    s = int(8 * FS)
    fm = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                       subplot_titles=("Processed ECG + matched-filter detections", "Correlation output"))
    fm.add_trace(go.Scatter(x=tt[:s], y=x[:s], line=dict(color=CLEAN_TEAL)), 1, 1)
    d8 = det[det < s]
    fm.add_trace(go.Scatter(x=d8 / FS, y=x[d8], mode="markers",
                            marker=dict(symbol="triangle-up", color=AMBER, size=10)), 1, 1)
    fm.add_trace(go.Scatter(x=tt[:s], y=c[:s], line=dict(color=PURPLE)), 2, 1)
    _style(fm, "Matched filter detector", height=440)
    fm.update_layout(showlegend=False)
    fm.update_xaxes(title_text="Time (s)", row=2, col=1)
    _show(fm)
    st.caption("The correlation output peaks where the signal best resembles the template; it is the optimal linear "
               "detector for a known shape in white noise. Raise the noise slider to see it degrade.")


# --------------------------------------------------------------------------- #
# Tab 5 - windowing and leakage
# --------------------------------------------------------------------------- #
def _tab_windowing():
    st.markdown("A finite record is the true signal multiplied by a **window**, so its spectrum is the true spectrum "
                "**convolved** with the window's spectrum. A rectangular window leaks energy into every bin and can "
                "bury weak components.")
    c = st.columns(4)
    n = c[0].selectbox("Record length N", [256, 512, 1024, 2048], index=2, key="sns_wn")
    f1 = c[1].slider("Strong tone (Hz)", 10.0, 80.0, 40.3, 0.1, key="sns_wf1")
    f2 = c[2].slider("Weak tone (Hz)", 10.0, 120.0, 52.0, 0.5, key="sns_wf2")
    a_db = c[3].slider("Weak tone level (dB)", -90, -20, -60, key="sns_wdb")
    wins = st.multiselect("Windows", ["boxcar", "hann", "hamming", "blackman", "flattop"],
                          default=["boxcar", "hann", "blackman"], key="sns_wins")
    k = np.arange(n)
    x = np.sin(2 * np.pi * f1 * k / FS) + 10 ** (a_db / 20) * np.sin(2 * np.pi * f2 * k / FS)
    cols = {"boxcar": ECG_RED, "hann": CLEAN_TEAL, "hamming": PURPLE, "blackman": AMBER, "flattop": BLUE}
    fig = go.Figure()
    for w in wins:
        f, s = spectrum_db(x, FS, w)
        fig.add_trace(go.Scatter(x=f, y=s, name="rectangular" if w == "boxcar" else w, line=dict(color=cols[w])))
    fig.update_yaxes(range=[-140, 5])
    fig.add_vline(x=f2, line_dash="dot", line_color="#94A3B8", annotation_text="weak tone")
    _style(fig, "Spectral leakage", height=380, xt="Frequency (Hz)", yt="dB")
    _show(fig)
    st.caption("Rectangular: narrowest main lobe but high side lobes (weak tone hidden). Hann/Blackman: wider main "
               "lobe, far lower side lobes. This is the resolution-vs-leakage tradeoff.")

    st.markdown("##### On the ECG: periodogram (rectangular) vs Welch (averaged Hann)")
    noisy, _, _ = _pair("arrhythmia", 7, 0.04, 0.08, 0.15)
    xp = full_dsp_pipeline(noisy, FS)
    fp_, pp = sg.periodogram(xp, FS, window="boxcar")
    fw, pw = compute_psd_welch(xp, FS)
    fg = go.Figure()
    fg.add_trace(go.Scatter(x=fp_, y=10 * np.log10(pp + 1e-12), name="Periodogram (rect, high variance)",
                            line=dict(color=ECG_RED, width=1)))
    fg.add_trace(go.Scatter(x=fw, y=10 * np.log10(pw + 1e-12), name="Welch (Hann, averaged)",
                            line=dict(color=CLEAN_TEAL, width=2)))
    fg.update_xaxes(range=[0, 60])
    _style(fg, "Power spectral density of the processed ECG", height=340, xt="Frequency (Hz)", yt="dB/Hz")
    _show(fg)


# --------------------------------------------------------------------------- #
# Tab 6 - STFT
# --------------------------------------------------------------------------- #
def _tab_stft():
    st.markdown("The STFT slides a window along the signal and takes an FFT of each segment. Window length sets the "
                "**time vs frequency resolution** tradeoff (a Heisenberg-style limit: "
                r"$\Delta t\,\Delta f \gtrsim$ const).")
    noisy, _, _ = _pair("arrhythmia", 7, 0.04, 0.08, 0.15, 12.0)
    x = full_dsp_pipeline(noisy, FS)
    c1, c2 = st.columns(2)
    w_short = c1.select_slider("Short window (s)", options=[0.03, 0.05, 0.08, 0.12], value=0.05, key="sns_ws")
    w_long = c2.select_slider("Long window (s)", options=[0.25, 0.5, 1.0, 2.0], value=0.5, key="sns_wl")
    for label, win in (("Short", w_short), ("Long", w_long)):
        nper = max(8, int(win * FS))
        f, tt, S = sg.spectrogram(x, FS, window="hann", nperseg=nper, noverlap=int(nper * 0.9))
        Sdb = 10 * np.log10(S + 1e-12)
        m = f <= 60
        hm = go.Figure(go.Heatmap(z=Sdb[m], x=tt, y=f[m], colorscale="Turbo",
                                  zmin=float(Sdb.max()) - 60, zmax=float(Sdb.max()), colorbar=dict(title="dB")))
        _style(hm, f"{label} window = {win * 1000:.0f} ms (frequency bin = {FS / nper:.2f} Hz)", height=320,
               xt="Time (s)", yt="Frequency (Hz)")
        _show(hm)
    st.caption("Short window: each QRS appears as a sharp vertical bar (good timing, smeared frequency). Long window: "
               "harmonics resolved, but beats blur together. The CWT on the DSP page avoids the fixed tradeoff by "
               "using scale-dependent windows.")


# --------------------------------------------------------------------------- #
# Tab 7 - Fourier series & Parseval
# --------------------------------------------------------------------------- #
def _tab_fourier():
    st.markdown("One heartbeat period treated as one period of a periodic signal is a **Fourier series**: "
                r"$x[n]=\sum_k X_k e^{j2\pi kn/L}$. Add harmonics and watch it converge.")
    _, clean, ann = _pair("normal", 13, 0.0, 0.0, 0.0)
    a = int(ann[4]) - int(0.25 * FS)
    b = int(ann[5]) - int(0.25 * FS)
    seg = clean[a:b]
    L = len(seg)
    f_fund = FS / L
    nmax = L // 2
    N = st.slider("Number of harmonics N", 1, nmax, 5, key="sns_harm")
    X = np.fft.rfft(seg)
    Xk = np.zeros_like(X)
    Xk[:N + 1] = X[:N + 1]
    y = np.fft.irfft(Xk, L)
    t = np.arange(L) / FS
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=seg, name="One beat period", line=dict(color="#94A3B8")))
    fig.add_trace(go.Scatter(x=t, y=y, name=f"Reconstruction, {N} harmonics", line=dict(color=CLEAN_TEAL, width=2)))
    _style(fig, f"Fundamental = {f_fund:.2f} Hz; highest harmonic used = {N * f_fund:.1f} Hz", height=340,
           xt="Time (s)", yt="mV")
    _show(fig)
    e_t = float(np.sum(seg ** 2))
    e_f = float(np.sum(np.abs(np.fft.fft(seg)) ** 2) / L)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Time-domain energy", f"{e_t:.4f}")
    c2.metric("Freq-domain energy", f"{e_f:.4f}")
    c3.metric("Parseval error", f"{abs(e_t - e_f) / e_t:.1e}")
    c4.metric("Energy captured by N", f"{np.sum(y ** 2) / e_t:.1%}")
    st.latex(r"\sum_{n=0}^{L-1}|x[n]|^2=\frac{1}{L}\sum_{k=0}^{L-1}|X[k]|^2 \quad\text{(Parseval)}")
    kk = min(40, len(X))
    bar = go.Figure(go.Bar(x=np.arange(kk) * f_fund, y=2 * np.abs(X[:kk]) / L, marker_color=PURPLE))
    _style(bar, "Harmonic amplitudes", height=280, xt="Frequency (Hz)", yt="Amplitude")
    _show(bar)
    st.caption("Sharp features like the QRS need many harmonics; a smooth T-wave needs few. This is why the ECG "
               "band-pass goes up to 40 Hz.")


# --------------------------------------------------------------------------- #
# Tab 8 - quantitative evaluation
# --------------------------------------------------------------------------- #
def _band_power(f, p, lo, hi):
    m = (f >= lo) & (f <= hi)
    return float(np.sum(p[m]) * (f[1] - f[0]))


def _tab_evaluation():
    st.markdown("Quantify what the filters actually did, using the synthetic generator's **clean ground truth**.")
    c = st.columns(4)
    hf = c[0].slider("EMG noise (mV)", 0.0, 0.3, 0.06, 0.01, key="sns_e_hf")
    pw = c[1].slider("50 Hz hum amplitude (mV)", 0.0, 0.3, 0.10, 0.01, key="sns_e_pw")
    bs = c[2].slider("Baseline wander (mV)", 0.0, 0.5, 0.2, 0.01, key="sns_e_bs")
    nf = c[3].selectbox("Notch frequency (Hz)", [50.0, 60.0], key="sns_e_nf")
    noisy, clean, _ = _pair("normal", 7, hf, pw, bs)
    proc = full_dsp_pipeline(noisy, FS, notch_freq=nf)
    ref = full_dsp_pipeline(clean, FS)
    e = int(FS)
    sl = slice(e, len(noisy) - e)

    def snr(sig, err):
        return 10 * np.log10(np.sum(sig[sl] ** 2) / (np.sum(err[sl] ** 2) + 1e-18))

    snr_in = snr(clean, noisy - clean)
    snr_out = snr(ref, proc - ref)
    fr, pr = sg.welch(noisy[sl], FS, nperseg=int(FS * 8))
    fq, pq = sg.welch(proc[sl], FS, nperseg=int(FS * 8))
    rows = [
        ("SNR (dB)", snr_in, snr_out, snr_out - snr_in),
        ("Power at 49-51 Hz (dB)", 10 * np.log10(_band_power(fr, pr, 49, 51) + 1e-18),
         10 * np.log10(_band_power(fq, pq, 49, 51) + 1e-18), None),
        ("Baseline drift power < 0.5 Hz (dB)", 10 * np.log10(_band_power(fr, pr, 0, 0.5) + 1e-18),
         10 * np.log10(_band_power(fq, pq, 0, 0.5) + 1e-18), None),
    ]
    table = []
    for name, before, after, delta in rows:
        table.append({"Metric": name, "Before": round(before, 2), "After": round(after, 2),
                      "Change (dB)": round(after - before, 2) if delta is None else round(delta, 2)})
    st.dataframe(table, use_container_width=True, hide_index=True)
    if nf != 50.0 and pw > 0:
        st.warning("The synthetic hum is at 50 Hz but the notch is at 60 Hz, so hum is NOT removed. "
                   "Choosing the wrong mains frequency is a real-world failure mode (MIT-BIH was recorded in the US: 60 Hz).")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fr, y=10 * np.log10(pr + 1e-12), name="Before", line=dict(color=ECG_RED)))
    fig.add_trace(go.Scatter(x=fq, y=10 * np.log10(pq + 1e-12), name="After", line=dict(color=CLEAN_TEAL)))
    fig.update_xaxes(range=[0, 80])
    fig.add_vline(x=50, line_dash="dot", line_color="#94A3B8")
    _style(fig, "Power spectral density before vs after", height=340, xt="Frequency (Hz)", yt="dB/Hz")
    _show(fig)
    t = np.arange(len(noisy)) / FS
    s = int(6 * FS)
    ft = go.Figure()
    ft.add_trace(go.Scatter(x=t[:s], y=(proc - ref)[:s], name="Residual noise after filtering", line=dict(color=AMBER)))
    ft.add_trace(go.Scatter(x=t[:s], y=(noisy - clean)[:s], name="Injected noise", line=dict(color="#94A3B8", width=1)))
    _style(ft, "Noise before vs after", height=300, xt="Time (s)", yt="mV")
    _show(ft)
    st.caption("SNR out uses the same filter applied to the clean signal as the reference, so it measures noise "
               "suppression and not the (intentional) band-limiting of the ECG itself.")


# --------------------------------------------------------------------------- #
# Tab 9 - FIR vs IIR
# --------------------------------------------------------------------------- #
def _tab_fir_iir():
    st.markdown("Same 0.5-40 Hz band-pass, two implementations.")
    taps = st.slider("FIR taps", 101, 3001, 1001, 100, key="sns_taps")
    sos = bandpass_sos(0.5, 40.0, 4)
    bfir = sg.firwin(taps, [0.5, 40.0], pass_zero=False, fs=FS, window="hamming")
    wi, hi_ = sg.sosfreqz(sos, worN=4096, fs=FS)
    wf, hf_ = sg.freqz(bfir, 1, worN=4096, fs=FS)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("Magnitude (dB)", "Phase (deg)", "Group delay (ms)"))
    fig.add_trace(go.Scatter(x=wi, y=20 * np.log10(np.maximum(abs(hi_), 1e-6)), name="IIR Butterworth",
                             line=dict(color=ECG_RED)), 1, 1)
    fig.add_trace(go.Scatter(x=wf, y=20 * np.log10(np.maximum(abs(hf_), 1e-6)), name="FIR (firwin)",
                             line=dict(color=CLEAN_TEAL)), 1, 1)
    fig.add_trace(go.Scatter(x=wi, y=np.degrees(np.unwrap(np.angle(hi_))), showlegend=False,
                             line=dict(color=ECG_RED)), 2, 1)
    fig.add_trace(go.Scatter(x=wf, y=np.degrees(np.unwrap(np.angle(hf_))), showlegend=False,
                             line=dict(color=CLEAN_TEAL)), 2, 1)
    fig.add_trace(go.Scatter(x=wi, y=np.clip(group_delay_ms(wi, hi_), -50, 500), showlegend=False,
                             line=dict(color=ECG_RED)), 3, 1)
    fig.add_trace(go.Scatter(x=wf, y=group_delay_ms(wf, hf_), showlegend=False,
                             line=dict(color=CLEAN_TEAL)), 3, 1)
    fig.update_yaxes(range=[-90, 5], row=1, col=1)
    fig.update_xaxes(range=[0, 100])
    _style(fig, "FIR vs IIR", height=680)
    fig.update_xaxes(title_text="Frequency (Hz)", row=3, col=1)
    _show(fig)
    delay = (taps - 1) / 2 / FS * 1000
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("IIR order", 2 * len(sos))
    k2.metric("IIR mults / sample", 5 * len(sos))
    k3.metric("FIR mults / sample", taps)
    k4.metric("FIR constant delay", f"{delay:.0f} ms")
    st.info("FIR: exactly linear phase (constant group delay = (taps-1)/2 samples), always stable, but needs many "
            "taps for a sharp 0.5 Hz edge. IIR: very cheap and sharp, but non-linear phase (frequency-dependent "
            "delay) - which is why the project pairs it with zero-phase filtfilt.")


# --------------------------------------------------------------------------- #
# Page entry point
# --------------------------------------------------------------------------- #
def render_sns_lab_page():
    st.markdown("### Signals & Systems Lab")
    st.caption("Theory behind the pipeline: LTI systems, frequency response, poles/zeros, sampling, convolution, "
               "windowing, time-frequency analysis and Fourier series. All demos use the synthetic ECG generator "
               "(with clean ground truth) so results are reproducible offline.")
    tabs = st.tabs([
        "1 Filters as LTI systems", "2 filtfilt vs causal", "3 Sampling & aliasing",
        "4 Convolution & matched filter", "5 Windowing & leakage", "6 STFT",
        "7 Fourier series & Parseval", "8 Filter evaluation", "9 FIR vs IIR",
    ])
    for tab, fn in zip(tabs, [
        _tab_filters, _tab_filtfilt, _tab_sampling, _tab_convolution, _tab_windowing,
        _tab_stft, _tab_fourier, _tab_evaluation, _tab_fir_iir,
    ]):
        with tab:
            fn()
