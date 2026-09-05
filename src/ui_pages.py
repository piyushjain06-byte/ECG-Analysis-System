import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import generate_synthetic_ecg, load_csv_record
from src.explainability import generate_clinical_explanation, get_model_feature_importance
from src.history_db import clear_analysis_history, get_analysis_history, save_analysis_history
from src.pipeline import CLASS_NAMES, run_full_analysis
from src.report_generator import generate_pdf_report
from src.signal_analysis import compute_wavelet_scalogram
from src.visualization import (
    AXIS_LINE,
    CLEAN_TEAL,
    ECG_RED,
    FONT_COLOR,
    GRID,
    PAPER,
    PLOT_BG,
    PURPLE,
    TITLE_COLOR,
    make_ecg_figure,
    overlay_ecg,
    window_signal,
)


DISCLAIMER = (
    "This project is an academic/research prototype for education and experiment. "
    "It is not a certified medical device and must not be used for diagnosis, treatment, or emergency decisions."
)

LOW_CONFIDENCE_THRESHOLD = 0.55


def _quality_html(status):
    cls = "q-good" if status == "GOOD" else ("q-ok" if status == "ACCEPTABLE" else "q-poor")
    return f'<span class="{cls}">{status}</span>'


@st.cache_data(show_spinner=False)
def _cached_dsp_run(sig_raw, fs, use_baseline, baseline_method, baseline_cutoff,
                     use_notch, notch_freq, use_bandpass, bp_low, bp_high, bp_order):
    """DSP-only pass (no ML inference) so tweaking a filter slider on the DSP
    page doesn't also re-run the classifier every rerun, and repeated reruns
    with identical parameters don't recompute at all (plan §52/§54)."""
    return run_full_analysis(
        sig_raw, fs,
        use_baseline=use_baseline, baseline_method=baseline_method, baseline_cutoff=baseline_cutoff,
        use_notch=use_notch, notch_freq=notch_freq,
        use_bandpass=use_bandpass, bp_low=bp_low, bp_high=bp_high, bp_order=bp_order,
        run_prediction=False,
    )


def _apply_result(result, update_prediction=True):
    st.session_state["quality_metrics"] = result["quality"]
    st.session_state["quality_status"] = result["quality"]["status"]
    st.session_state["quality_details"] = result["quality"]["details"]
    st.session_state["sig_clean"] = result["sig_clean"]
    st.session_state["dsp_stages"] = result["stages"]
    st.session_state["dsp_applied"] = True
    st.session_state["peaks"] = result["peaks"]
    st.session_state["pt_stages"] = result["pt_stages"]
    st.session_state["rr_metrics"] = result["rr"]
    st.session_state["fft_freqs"] = result["fft_freqs"]
    st.session_state["fft_mags"] = result["fft_mags"]
    st.session_state["fft_mags_raw"] = result["fft_mags_raw"]
    st.session_state["dominant_freq"] = result["dominant_freq"]
    st.session_state["spectral_bands"] = result["bands_clean"]
    st.session_state["feat_df"] = result["feat_df"]
    # Prediction fields are only overwritten when this result actually ran
    # prediction - otherwise a DSP-only recompute would silently wipe out
    # whatever prediction was already on screen.
    if update_prediction and result.get("prediction_ran", True):
        st.session_state["pred_df"] = result["pred_df"]
        st.session_state["dominant_class"] = result["dominant_class"]
        st.session_state["pred_confidence"] = result["confidence"]
        st.session_state["model_name"] = result["model_name"]


def _load_and_analyze(sig, fs, record_name, simulated, model_name="Random Forest"):
    """Runs the full pipeline; only commits to session state on success so a
    bad/short/corrupt recording never leaves the app in a broken half-loaded state."""
    with st.spinner("Running DSP → R-peaks → features → classification..."):
        try:
            result = run_full_analysis(sig, fs, model_name=model_name)
        except ValueError as e:
            st.error(f"Could not analyze this recording: {e}")
            return False
        except Exception as e:
            st.error(f"Unexpected error while analyzing the recording: {e}")
            return False

    st.session_state["sig_raw"] = result["sig_raw"]
    st.session_state["fs"] = float(fs)
    st.session_state["record_name"] = record_name
    st.session_state["data_loaded"] = True
    st.session_state["is_simulated"] = simulated
    _apply_result(result)
    return True


def render_home_page():
    st.markdown(
        """
        <div class="hero-card">
            <h1>Intelligent ECG Signal Analysis</h1>
            <p>Academic platform for digital signal processing, feature extraction, and rhythm classification.
            DSP stages stay visible — this is not a black-box “ECG → AI” demo.</p>
            <div class="pipe">
                <span>RAW ECG</span><span>QUALITY</span><span>DSP</span><span>FFT</span>
                <span>R-PEAKS</span><span>FEATURES</span><span>ML</span><span>REPORT</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(DISCLAIMER)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Demo recording (synthetic — offline / demo use only)")
        st.caption(
            "Synthetic ECGs are for demo, UI testing, and fully-offline operation. "
            "They are never mixed with MIT-BIH data in the official reported model metrics."
        )
        demo_type = st.selectbox(
            "Rhythm scenario",
            [
                "Arrhythmia (N + PVC/APC/fusion)",
                "Normal sinus rhythm",
                "Bradycardia",
                "Tachycardia",
            ],
        )
        mapping = {
            "Arrhythmia (N + PVC/APC/fusion)": "arrhythmia",
            "Normal sinus rhythm": "normal",
            "Bradycardia": "bradycardia",
            "Tachycardia": "tachycardia",
        }
        if st.button("Analyze demo ECG", type="primary", use_container_width=True):
            rhythm = mapping[demo_type]
            hr = 70.0
            if rhythm == "bradycardia":
                hr = 45.0
            elif rhythm == "tachycardia":
                hr = 120.0
            sig, fs, ann_s, ann_y = generate_synthetic_ecg(
                duration=40.0,
                fs=360.0,
                heart_rate=hr,
                rhythm_type=rhythm,
                noise_level=0.06,
                power_noise_level=0.03,
                baseline_noise_level=0.16,
                seed=101,
            )
            st.session_state["ann_sample"] = ann_s
            st.session_state["ann_symbol"] = ann_y
            name = f"DEMO_{rhythm.upper()}"
            _load_and_analyze(sig, fs, name, True)

    with c2:
        st.markdown("#### Upload recording")
        uploaded = st.file_uploader("CSV (`time,ecg` or a single ECG column) or WFDB `.dat`", type=["csv", "dat"])
        fs_input = st.number_input("Sampling frequency if missing (Hz)", 1.0, 2000.0, 360.0, 1.0)
        if st.button("Analyze uploaded file", use_container_width=True):
            if uploaded is None:
                st.error("Choose a file first.")
            elif uploaded.size == 0:
                st.error("The uploaded file is empty (0 bytes).")
            else:
                os.makedirs("data/raw", exist_ok=True)
                temp_path = os.path.join("data/raw", uploaded.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                try:
                    if uploaded.name.lower().endswith(".csv"):
                        sig, fs = load_csv_record(temp_path, default_fs=fs_input)
                        st.session_state["ann_sample"] = None
                        st.session_state["ann_symbol"] = None
                        _load_and_analyze(sig, fs, uploaded.name.rsplit(".", 1)[0], False)
                    else:
                        hea = temp_path.replace(".dat", ".hea")
                        if not os.path.exists(hea):
                            st.error(
                                "WFDB `.dat` needs a matching `.hea` file (and ideally `.atr`) "
                                "uploaded into `data/raw/` alongside it - a `.dat` file alone "
                                "has no header describing the signal."
                            )
                        else:
                            import wfdb

                            rec = wfdb.rdrecord(temp_path.replace(".dat", ""))
                            sig = rec.p_signal[:, 0] if rec.p_signal.ndim > 1 else rec.p_signal
                            st.session_state["ann_sample"] = None
                            st.session_state["ann_symbol"] = None
                            try:
                                ann = wfdb.rdann(temp_path.replace(".dat", ""), "atr")
                                st.session_state["ann_sample"] = ann.sample
                                st.session_state["ann_symbol"] = ann.symbol
                            except Exception:
                                pass
                            _load_and_analyze(sig, rec.fs, uploaded.name.rsplit(".", 1)[0], False)
                except ValueError as e:
                    st.error(f"Could not read this file: {e}")
                except Exception as e:
                    st.error(f"Unsupported or unreadable ECG file. {e}")

    if not st.session_state.get("data_loaded"):
        st.info("Load a demo or upload a CSV to generate graphs and classification.")
        return

    sig = st.session_state["sig_raw"]
    fs = st.session_state["fs"]
    q = st.session_state["quality_metrics"] or {}
    rr = st.session_state.get("rr_metrics") or {}

    st.markdown("### Recording overview")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Record", st.session_state["record_name"])
    m2.metric("Sampling rate", f"{fs:.0f} Hz")
    m3.metric("Duration", f"{len(sig) / fs:.1f} s")
    m4.metric("Samples", f"{len(sig):,}")
    m5.metric("Channels", "1")

    q_status = st.session_state["quality_status"]
    sqi = q.get("sqi_score")
    quality_line = f"**Signal quality:** {_quality_html(q_status)}"
    if sqi is not None:
        quality_line += f"&nbsp;&nbsp;·&nbsp;&nbsp;**SQI:** {sqi}/100"
    st.markdown(quality_line, unsafe_allow_html=True)
    st.caption(st.session_state.get("quality_details", ""))
    if q_status == "POOR":
        st.warning("The ECG signal quality is poor. Results may be unreliable.")

    if rr:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Heart rate", f"{rr.get('mean_hr', 0):.1f} BPM")
        k2.metric("Mean RR", f"{rr.get('mean_rr', 0):.0f} ms")
        k3.metric("Detected beats", f"{rr.get('num_beats', 0)}")
        pred = st.session_state.get("dominant_class")
        k4.metric("ML class", CLASS_NAMES.get(pred, pred or "—"))

    t, y, _, _ = window_signal(sig, fs, 0, min(8.0, len(sig) / fs))
    st.plotly_chart(
        make_ecg_figure(t, y, "Raw ECG — first window", name="Raw", color=ECG_RED, height=260),
        use_container_width=True,
    )
    st.caption("Open **DSP Analysis** for baseline / notch / band-pass stages, FFT, and optional wavelets.")


def render_dsp_page():
    st.markdown("### DSP preprocessing")
    st.caption(
        "Baseline wander is mostly <0.5 Hz (respiration). Notch removes 50/60 Hz mains. "
        "A 0.5–40 Hz Butterworth band-pass keeps P/QRS/T content and reduces EMG. "
        "Zero-phase `filtfilt` is used so offline analysis is not delayed."
    )
    if not st.session_state.get("data_loaded"):
        st.warning("Load an ECG on Home first.")
        return

    sig_raw = st.session_state["sig_raw"]
    fs = st.session_state["fs"]
    duration = len(sig_raw) / fs

    c1, c2, c3 = st.columns(3)
    with c1:
        use_baseline = st.checkbox("Baseline removal", True)
        baseline_method = st.selectbox("Method", ["butterworth", "median", "detrend"])
        baseline_cutoff = st.slider("High-pass cutoff (Hz)", 0.05, 2.0, 0.5, 0.05)
    with c2:
        use_notch = st.checkbox("Notch filter", True)
        notch_freq = st.selectbox("Mains frequency", [50.0, 60.0], format_func=lambda x: f"{int(x)} Hz")
    with c3:
        use_bandpass = st.checkbox("Band-pass", True)
        bp_low = st.number_input("Low cutoff (Hz)", 0.05, 10.0, 0.5, 0.1)
        bp_high = st.number_input("High cutoff (Hz)", 10.0, 100.0, 40.0, 1.0)
        bp_order = st.slider("Butterworth order", 1, 8, 4)

    # NOTE: run_prediction is intentionally False here - the classifier is
    # not re-run just because a DSP slider moved. Cached on the exact filter
    # parameters, so identical reruns skip recomputation entirely.
    result = _cached_dsp_run(
        sig_raw, fs, use_baseline, baseline_method, baseline_cutoff,
        use_notch, float(notch_freq), use_bandpass, bp_low, bp_high, bp_order,
    )
    _apply_result(result, update_prediction=False)
    sig_clean = result["sig_clean"]
    stages = result["stages"]

    t0, t1 = st.slider("Time window (s)", 0.0, float(duration), (0.0, min(8.0, duration)), 0.2)
    t, raw_w, i0, i1 = window_signal(sig_raw, fs, t0, t1)

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(make_ecg_figure(t, raw_w, "1 · Raw ECG", color=ECG_RED), use_container_width=True)
        st.plotly_chart(
            make_ecg_figure(t, stages["notch"][i0:i1], "3 · After notch (mains)", color="#A78BFA"),
            use_container_width=True,
        )
    with g2:
        st.plotly_chart(
            make_ecg_figure(t, stages["baseline"][i0:i1], "2 · Baseline-corrected", color="#38BDF8"),
            use_container_width=True,
        )
        st.plotly_chart(
            make_ecg_figure(t, stages["final"][i0:i1], "4 · Final processed ECG", color=CLEAN_TEAL),
            use_container_width=True,
        )

    st.plotly_chart(
        overlay_ecg(
            t,
            [
                (raw_w, "Raw", ECG_RED, "dot"),
                (sig_clean[i0:i1], "Processed", CLEAN_TEAL, "solid"),
            ],
            "Overlay: raw vs processed",
            height=300,
        ),
        use_container_width=True,
    )

    st.markdown("### Frequency domain (FFT)")
    st.info(
        "FFT maps the ECG from time to frequency so you can see mains hum (~50 Hz), "
        "residual baseline, and the QRS band."
    )
    freqs = result["fft_freqs"]
    mag_raw = result["fft_mags_raw"]
    mag_c = result["fft_mags"]
    mask = freqs <= 80
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs[mask], y=mag_raw[mask], name="Raw spectrum", line=dict(color=ECG_RED, dash="dash", width=1.2)))
    fig.add_trace(go.Scatter(x=freqs[mask], y=mag_c[mask], name="Processed spectrum", line=dict(color=PURPLE, width=1.6)))
    fig.add_vline(x=50, line_dash="dot", line_color="#94A3B8", annotation_text="50 Hz")
    fig.update_layout(
        title=dict(text="Magnitude spectrum", font=dict(color=TITLE_COLOR)),
        xaxis_title="Frequency (Hz)", yaxis_title="Magnitude", height=320,
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER, font=dict(color=FONT_COLOR),
        xaxis=dict(gridcolor=GRID, linecolor=AXIS_LINE), yaxis=dict(gridcolor=GRID, linecolor=AXIS_LINE),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, use_container_width=True)
    f1, f2, f3 = st.columns(3)
    f1.metric("Dominant freq (processed)", f"{result['dominant_freq']:.2f} Hz")
    f2.metric("QRS-band energy 0.5–40 Hz", f"{result['bands_clean']['ecg_band_power']:.1%}")
    f3.metric("LF 0.04–0.15 Hz (drift)", f"{result['bands_clean']['lf_power']:.1%}")

    with st.expander("Optional wavelet scalogram (CWT, Morlet)"):
        wav = compute_wavelet_scalogram(sig_clean, fs=fs, max_seconds=6.0)
        if wav is None:
            st.caption("Install PyWavelets to enable this view.")
        else:
            tw, fq, mag = wav
            heat = go.Figure(data=go.Heatmap(z=mag, x=tw, y=fq, colorscale="Turbo", colorbar=dict(title="|CWT|")))
            heat.update_layout(
                title=dict(text="Wavelet energy vs time (non-stationary QRS bursts)", font=dict(color=TITLE_COLOR)),
                xaxis_title="Time (s)", yaxis_title="Frequency (Hz)", height=340,
                paper_bgcolor=PAPER, plot_bgcolor=PLOT_BG, font=dict(color=FONT_COLOR),
            )
            st.plotly_chart(heat, use_container_width=True)

    st.caption(
        "ML classification is not recomputed on this page - open **ML Prediction** to run/refresh it "
        "with the currently-selected DSP settings."
    )


def render_features_page():
    st.markdown("### Time-domain analysis & features")
    if not st.session_state.get("data_loaded"):
        st.warning("Load an ECG on Home first.")
        return
    if not st.session_state.get("dsp_applied"):
        st.warning("Open DSP Analysis once so filters and peaks are computed.")
        return

    sig = st.session_state["sig_clean"]
    fs = st.session_state["fs"]
    peaks = st.session_state["peaks"]
    rr = st.session_state["rr_metrics"]
    pt = st.session_state.get("pt_stages") or {}

    if peaks is None or len(peaks) == 0:
        st.error(
            "Unable to detect reliable R-peaks in this recording. This usually means the signal is "
            "too noisy, too short, or the amplitude scale is unusual. Try adjusting the DSP filters "
            "or use a cleaner recording."
        )
        return

    st.caption(
        "Pan–Tompkins: 5–15 Hz band-pass → derivative → square → 150 ms integration → adaptive peak search. "
        "P/Q/T waves are not claimed as fully detected; QRS duration here is an estimate around each R peak."
    )
    duration = len(sig) / fs
    t0, t1 = st.slider("Display window (s)", 0.0, float(duration), (0.0, min(8.0, duration)), 0.2, key="feat_win")
    t, yw, i0, i1 = window_signal(sig, fs, t0, t1)
    vis_peaks = peaks[(peaks >= i0) & (peaks < i1)]
    st.plotly_chart(
        make_ecg_figure(
            t, yw, "Processed ECG with detected R-peaks", name="Processed", color=CLEAN_TEAL,
            peaks=vis_peaks / fs, peak_y=sig[vis_peaks], height=300,
        ),
        use_container_width=True,
    )

    a, b, c, d = st.columns(4)
    a.metric("Beats", f"{rr['num_beats']}")
    b.metric("Heart rate", f"{rr['mean_hr']:.1f} BPM")
    c.metric("Mean RR", f"{rr['mean_rr']:.0f} ms")
    d.metric("SDRR", f"{rr['std_rr_sdrr']:.1f} ms")
    e, fcol, g = st.columns(3)
    e.metric("Median RR", f"{np.median(rr['rr_intervals']):.0f} ms" if len(rr["rr_intervals"]) else "—")
    fcol.metric("Min / max RR", f"{rr['rr_intervals'].min():.0f} / {rr['rr_intervals'].max():.0f} ms" if len(rr["rr_intervals"]) else "—")
    g.metric("RMSSD", f"{rr['rmssd']:.1f} ms")
    st.caption("Heart rate outside 60–100 BPM is not labeled as disease; it is only a measured rate.")

    if len(rr["rr_intervals"]):
        fig_rr = go.Figure()
        fig_rr.add_trace(
            go.Scatter(
                x=np.arange(len(rr["rr_intervals"])), y=rr["rr_intervals"],
                mode="lines+markers", line=dict(color="#FB923C", width=1.6), marker=dict(size=5),
            )
        )
        fig_rr.update_layout(
            title=dict(text="RR interval vs beat number", font=dict(color=TITLE_COLOR)),
            xaxis_title="Beat", yaxis_title="RR (ms)", height=260,
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER, font=dict(color=FONT_COLOR),
            xaxis=dict(gridcolor=GRID, linecolor=AXIS_LINE), yaxis=dict(gridcolor=GRID, linecolor=AXIS_LINE),
        )
        st.plotly_chart(fig_rr, use_container_width=True)
        with st.expander("RR list (ms)"):
            st.dataframe(pd.DataFrame({"RR_ms": np.round(rr["rr_intervals"], 1)}), use_container_width=True, height=220)

    if pt:
        with st.expander("Pan–Tompkins intermediate signals"):
            t_pt, _, i0p, i1p = window_signal(pt["filtered"], fs, t0, t1)
            fig_pt = go.Figure()
            for key, name, col in [
                ("filtered", "5–15 Hz", "#38BDF8"), ("derivative", "Derivative", "#A78BFA"),
                ("squared", "Squared", "#F472B6"), ("integrated", "Integrated", "#2DD4BF"),
            ]:
                fig_pt.add_trace(go.Scatter(x=t_pt, y=pt[key][i0p:i1p], name=name, line=dict(width=1.2, color=col)))
            fig_pt.update_layout(
                height=340, hovermode="x unified", legend=dict(orientation="h"),
                paper_bgcolor=PAPER, plot_bgcolor=PLOT_BG, font=dict(color=FONT_COLOR),
                xaxis=dict(gridcolor=GRID, linecolor=AXIS_LINE), yaxis=dict(gridcolor=GRID, linecolor=AXIS_LINE),
            )
            st.plotly_chart(fig_pt, use_container_width=True)

    feat_df = st.session_state.get("feat_df")
    st.markdown("#### Beat feature table")
    if feat_df is None or feat_df.empty:
        st.error("No features extracted — the segment may be too short.")
        return
    summary = {
        "Parameter": [
            "Heart rate", "Mean RR", "RR std (SDRR)", "Min RR", "Max RR", "Detected beats",
            "Mean QRS amplitude (est.)", "Mean QRS duration (est.)", "Mean signal energy (beat)",
        ],
        "Value": [
            f"{rr['mean_hr']:.1f} BPM", f"{rr['mean_rr']:.1f} ms", f"{rr['std_rr_sdrr']:.1f} ms",
            f"{rr['rr_intervals'].min():.1f} ms" if len(rr["rr_intervals"]) else "—",
            f"{rr['rr_intervals'].max():.1f} ms" if len(rr["rr_intervals"]) else "—",
            f"{rr['num_beats']}", f"{feat_df['qrs_amplitude'].mean():.3f}",
            f"{feat_df['qrs_duration_est'].mean()*1000:.0f} ms", f"{feat_df['energy'].mean():.3f}",
        ],
    }
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
    st.dataframe(feat_df.head(20), use_container_width=True)
    st.caption(f"{feat_df.shape[0]} beats × {feat_df.shape[1]} columns. QRS duration is a window-based estimate, not a caliper measurement.")


def render_prediction_page():
    st.markdown("### Rhythm classification")
    if not st.session_state.get("data_loaded"):
        st.warning("Load an ECG on Home first.")
        return
    if st.session_state.get("peaks") is None or len(st.session_state.get("peaks", [])) == 0:
        st.warning("Open DSP Analysis or Home (demo) so peaks exist.")
        return

    model_name = st.selectbox(
        "Classifier",
        ["Random Forest", "XGBoost", "Support Vector Machine", "Logistic Regression"],
        index=0,
    )
    if st.button("Re-run prediction", type="primary") or st.session_state.get("pred_df") is None:
        result = run_full_analysis(
            st.session_state["sig_raw"], st.session_state["fs"], model_name=model_name,
        )
        _apply_result(result)

    pred_df = st.session_state.get("pred_df")
    if pred_df is None or (hasattr(pred_df, "empty") and pred_df.empty):
        st.error("Trained models are missing. Run `python -m src.ml_training` from the project folder.")
        return

    dominant = st.session_state.get("dominant_class")
    conf = st.session_state.get("pred_confidence")
    left, right = st.columns([1.4, 1])
    with left:
        st.markdown(f"#### Predicted majority class: **{CLASS_NAMES.get(dominant, dominant)}**")
        st.caption(
            "Majority vote over beat-level predictions. Confidence is the mean of the model's "
            "`predict_proba` for that class — not a calibrated clinical probability."
        )
    with right:
        if conf is not None:
            st.metric("Mean class probability", f"{conf:.1%}")
        st.metric("Model", model_name)

    if conf is not None and conf < LOW_CONFIDENCE_THRESHOLD:
        st.warning(
            f"⚠ Low-confidence prediction ({conf:.1%}). The model is not strongly certain about this "
            f"majority class - treat this result with extra caution and do not rely on it alone."
        )

    counts = pred_df["predicted_label"].value_counts()
    count_df = pd.DataFrame(
        {
            "Code": counts.index,
            "Class": [CLASS_NAMES.get(x, x) for x in counts.index],
            "Beats": counts.values,
            "Share": [f"{x / len(pred_df):.1%}" for x in counts.values],
        }
    )
    st.dataframe(count_df, use_container_width=True, hide_index=True)

    prob_cols = [c for c in pred_df.columns if c.startswith("prob_")]
    if prob_cols:
        avg_probs = pred_df[prob_cols].mean().rename(lambda c: c.replace("prob_", ""))
        st.caption(
            "Average per-class model probability across all detected beats: "
            + ", ".join(f"{CLASS_NAMES.get(k, k).split(' (')[0]}: {v:.1%}" for k, v in avg_probs.items())
        )

    pie = px.pie(count_df, names="Class", values="Beats", color_discrete_sequence=px.colors.qualitative.Set2)
    pie.update_layout(height=280, margin=dict(t=20, b=20))
    st.plotly_chart(pie, use_container_width=True)

    rep = pred_df[pred_df["predicted_label"] == dominant].iloc[0]
    explanation = generate_clinical_explanation(rep)
    st.markdown("#### Why the model leaned this way")
    st.markdown(explanation)
    st.caption("These features contributed to the model's classification. They do not prove a medical diagnosis.")

    feat_imp = get_model_feature_importance(model_name)
    if feat_imp is not None:
        fig_imp = px.bar(
            feat_imp.head(10).iloc[::-1], x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale="Teal",
        )
        fig_imp.update_layout(
            title=dict(text=f"Feature contribution ({model_name})", font=dict(color=TITLE_COLOR)),
            height=340, coloraxis_showscale=False, yaxis_title="", xaxis_title="Importance",
            paper_bgcolor=PAPER, plot_bgcolor=PLOT_BG, font=dict(color=FONT_COLOR),
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("This model does not expose a simple importance vector. Tree models and logistic regression do.")

    st.markdown("#### PDF report")
    patient_id = st.text_input("Patient / subject ID (optional, stored with this analysis)", value="")
    if st.button("Generate PDF report", type="primary", use_container_width=True):
        try:
            pdf_path = generate_pdf_report(
                record_name=st.session_state["record_name"], fs=st.session_state["fs"],
                sig_raw=st.session_state["sig_raw"], sig_processed=st.session_state["sig_clean"],
                peaks=st.session_state["peaks"], rr_metrics=st.session_state["rr_metrics"],
                fft_freqs=st.session_state["fft_freqs"], fft_mags=st.session_state["fft_mags"],
                predicted_label=dominant, confidence=conf or 0.0, model_name=model_name,
                explanation_text=explanation, quality_status=st.session_state["quality_status"],
                quality_details=st.session_state["quality_details"],
            )
            model_version = None
            meta_path = "models/metadata/ml_performance.json"
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as mf:
                    model_version = json.load(mf).get("model_version")
            feature_summary = {}
            fdf = st.session_state.get("feat_df")
            if fdf is not None and not fdf.empty:
                numeric_cols = fdf.select_dtypes(include=[np.number]).columns
                feature_summary = fdf[numeric_cols].mean().to_dict()
            save_analysis_history(
                record_name=st.session_state["record_name"], fs=st.session_state["fs"],
                duration=len(st.session_state["sig_raw"]) / st.session_state["fs"],
                signal_quality=st.session_state["quality_status"],
                heart_rate=st.session_state["rr_metrics"]["mean_hr"],
                prediction=CLASS_NAMES.get(dominant, str(dominant)),
                model_name=model_name, report_path=pdf_path,
                patient_id=patient_id or None, predicted_class=dominant,
                confidence=conf, model_version=model_version, features=feature_summary,
            )
            st.success(f"Saved `{pdf_path}`")
            with open(pdf_path, "rb") as fh:
                st.download_button("Download PDF", fh, file_name=os.path.basename(pdf_path), mime="application/pdf")
        except Exception as e:
            st.error(f"Report failed: {e}")


def render_performance_page():
    st.markdown("### Model comparison (held-out records)")
    meta_path = "models/metadata/ml_performance.json"
    if not os.path.exists(meta_path):
        st.warning("No metrics file. Train with `python -m src.ml_training`.")
        return
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    perf = metadata["performance"]
    st.caption(metadata.get("split_note", "Metrics are computed by the training script — not typed by hand."))

    trained_on = metadata.get("trained_on")
    is_official = metadata.get("is_official_result", trained_on == "mitbih")
    if is_official:
        st.success("**Official result** — trained and evaluated on MIT-BIH only (patient-level 70/15/15 split).")
    else:
        st.warning(
            "**Experimental result** — trained on the synthetic generator only. This is NOT the "
            "official reported performance; it exists for demo/offline testing purposes."
        )

    version_cols = st.columns(4)
    version_cols[0].metric("Model version", metadata.get("model_version", "—"))
    version_cols[1].metric("Dataset version", metadata.get("dataset_version", "—"))
    version_cols[2].metric("Feature version", metadata.get("feature_version", "—"))
    version_cols[3].metric("Classification scheme", metadata.get("classification_version", "—"))

    if "n_train" in metadata:
        st.caption(
            f"Train beats: {metadata['n_train']} · val beats: {metadata.get('n_val', '—')} · "
            f"test beats: {metadata['n_test']} · test records: {', '.join(metadata.get('test_records', []))}"
        )

    sources = metadata.get("data_sources")
    if sources:
        parts = []
        if "mitbih_beats" in sources:
            parts.append(f"{sources['mitbih_beats']} beats from {sources['mitbih_records']} real MIT-BIH record(s)")
        if "synthetic_beats" in sources:
            parts.append(f"{sources['synthetic_beats']} beats from {sources['synthetic_records']} synthetic record(s)")
        st.info(f"**Trained on:** {' + '.join(parts)}.")

    rows = []
    for name, m in perf.items():
        rows.append({
            "Model": name,
            "Accuracy": m.get("accuracy", 0.0),
            "Macro F1": m.get("macro_f1", 0.0),
            "Weighted F1": m.get("f1_weighted", m.get("f1_score", 0.0)),
            "ROC-AUC (macro OvR)": m.get("roc_auc_macro_ovr"),
            "PR-AUC (macro)": m.get("pr_auc_macro"),
            "Sensitivity": m.get("sensitivity_macro"),
            "Specificity": m.get("specificity_macro"),
            "Train time (s)": m.get("training_time_seconds"),
        })
    cdf = pd.DataFrame(rows)
    show = cdf.copy()
    for col in ["Accuracy", "Macro F1", "Weighted F1", "ROC-AUC (macro OvR)", "PR-AUC (macro)", "Sensitivity", "Specificity"]:
        show[col] = show[col].map(lambda v: f"{v:.1%}" if isinstance(v, (int, float)) else "—")
    st.dataframe(show, use_container_width=True, hide_index=True)

    fig = go.Figure()
    for col, color in zip(["Accuracy", "Macro F1", "Weighted F1"], ["#38BDF8", "#2DD4BF", "#A78BFA"]):
        fig.add_trace(go.Bar(x=cdf["Model"], y=cdf[col], name=col, marker_color=color))
    fig.update_layout(
        barmode="group", yaxis_title="Score", height=340,
        yaxis=dict(range=[0, 1], gridcolor=GRID, linecolor=AXIS_LINE),
        xaxis=dict(gridcolor=GRID, linecolor=AXIS_LINE), legend=dict(orientation="h"),
        paper_bgcolor=PAPER, plot_bgcolor=PLOT_BG, font=dict(color=FONT_COLOR),
    )
    st.plotly_chart(fig, use_container_width=True)

    dist = metadata.get("class_distribution", {})
    if dist:
        st.markdown("#### Training-set class counts")
        st.bar_chart(pd.Series(dist, name="beats"))

    models = list(perf.keys())
    sel = st.selectbox("Confusion matrix", models, index=min(2, len(models) - 1))
    labels = [metadata["class_mapping"][k] for k in sorted(metadata["class_mapping"], key=lambda x: int(x))]
    matrix = np.array(perf[sel]["confusion_matrix"]) if perf[sel]["confusion_matrix"] else np.zeros((len(labels), len(labels)))
    fig_cm = px.imshow(
        matrix, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="True label", color="Count"), title=f"Confusion matrix — {sel}",
    )
    fig_cm.update_layout(height=420, title=dict(font=dict(color=TITLE_COLOR)),
                          paper_bgcolor=PAPER, plot_bgcolor=PLOT_BG, font=dict(color=FONT_COLOR))
    st.plotly_chart(fig_cm, use_container_width=True)

    report = perf[sel].get("report", {})
    lines = []
    for key in labels:
        if key in report and isinstance(report[key], dict):
            val = report[key]
            lines.append({
                "Class": key, "Precision": f"{val['precision']:.1%}", "Recall": f"{val['recall']:.1%}",
                "F1": f"{val['f1-score']:.1%}", "Support": int(val["support"]),
            })
    if lines:
        st.dataframe(pd.DataFrame(lines), use_container_width=True, hide_index=True)

    if perf[sel].get("best_params"):
        st.caption(f"Tuned hyperparameters (RandomizedSearchCV, patient-aware CV): {perf[sel]['best_params']}")


def render_history_page():
    st.markdown("### Analysis history")
    st.caption("SQLite log of PDF runs. No patient identifiers are stored beyond what you optionally typed in.")
    df = get_analysis_history()
    if df.empty:
        st.info("No rows yet. Generate a PDF on the ML Prediction page.")
        return

    search = st.text_input("Search by record name, patient ID, or predicted class", value="")
    filtered = df
    if search:
        s = search.lower()
        mask = df.apply(lambda row: s in str(row.get("record_name", "")).lower()
                         or s in str(row.get("patient_id", "")).lower()
                         or s in str(row.get("predicted_class", "")).lower(), axis=1)
        filtered = df[mask]

    st.dataframe(filtered, use_container_width=True)

    with st.expander("Download a past report"):
        for _, row in filtered.head(20).iterrows():
            path = row.get("report_path")
            if path and os.path.exists(path):
                cols = st.columns([3, 1])
                cols[0].write(f"**{row['record_name']}** — {row.get('timestamp', '')}")
                with open(path, "rb") as fh:
                    cols[1].download_button("Download", fh, file_name=os.path.basename(path), key=f"dl_{row['id']}")

    if st.button("Clear history"):
        clear_analysis_history()
        st.rerun()


def render_about_page():
    st.markdown("### About this prototype")
    st.markdown(
        """
**Intelligent ECG Signal Analysis and Arrhythmia Detection Using Digital Signal Processing and Machine Learning**
is a software platform for digital ECG recordings. It applies baseline correction, filtering, frequency analysis,
and R-peak detection, then extracts temporal, morphological, statistical, and spectral features for classical ML
(N / S / V / F AAMI groups). It is an **academic research prototype**, not a replacement for hospital ECG systems.

**Algorithms:** Pan–Tompkins QRS detection (1985); Butterworth / notch / median DSP; FFT + Welch PSD; Random Forest, SVM, logistic regression, XGBoost with bounded RandomizedSearchCV tuning.

**Official dataset policy:** the MIT-BIH Arrhythmia Database (PhysioNet/WFDB) is the only dataset used for the
reported model performance shown on the Model Performance page (`trained_on: "mitbih"`). Run `download_mitbih.py`
then `python -m src.ml_training` (default source is `mitbih`) to reproduce it. The synthetic ECG generator is used
only for the demo button, offline testing, and an explicitly separate `synthetic_experiment` training run - it is
never pooled with MIT-BIH beats in one split.

**Gap we address:** many demos hide filtering. This app shows raw → cleaned → FFT → peaks → features → class.

**Medical disclaimer:** """
        + DISCLAIMER
    )
    st.markdown(
        """
**How to present it**

1. Start from a raw ECG.
2. Show DSP: baseline, notch, band-pass.
3. Time and frequency analysis.
4. R-peaks, RR, heart rate.
5. Features.
6. ML on labeled beats (MIT-BIH official).
7. Supported rhythm categories only.
8. Report + this disclaimer.
"""
    )
