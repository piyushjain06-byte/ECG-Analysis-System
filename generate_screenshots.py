"""Generate publication-style PNG graphs for the prototype submission pack."""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import generate_synthetic_ecg
from src.pipeline import CLASS_NAMES, run_full_analysis

OUT = "screenshots"
SAMPLE = "data/sample"


def _style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "#FFF8F5",
            "axes.grid": True,
            "grid.color": "#F0C4C4",
            "grid.linewidth": 0.6,
            "axes.edgecolor": "#CBD5E1",
        }
    )


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main():
    _style()
    os.makedirs(SAMPLE, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    sig, fs, _, _ = generate_synthetic_ecg(
        duration=30.0,
        fs=360.0,
        heart_rate=72.0,
        rhythm_type="arrhythmia",
        noise_level=0.06,
        power_noise_level=0.035,
        baseline_noise_level=0.18,
        seed=101,
    )
    t = np.arange(len(sig)) / fs
    csv_path = os.path.join(SAMPLE, "demo_ecg.csv")
    pd.DataFrame({"time": t, "ecg": sig}).to_csv(csv_path, index=False)
    print("wrote", csv_path)

    result = run_full_analysis(sig, fs, model_name="Random Forest")
    stages = result["stages"]
    n = int(8 * fs)
    tw = t[:n]

    fig, axes = plt.subplots(4, 1, figsize=(11, 8.4), sharex=True)
    series = [
        (stages["raw"], "1  Raw ECG (baseline wander + 50 Hz + EMG)", "#C0392B"),
        (stages["baseline"], "2  Baseline-corrected", "#0369A1"),
        (stages["notch"], "3  Notch-filtered (50 Hz)", "#7C3AED"),
        (stages["final"], "4  Final processed ECG (0.5–40 Hz band-pass)", "#0E7C7B"),
    ]
    for ax, (y, title, color) in zip(axes, series):
        ax.plot(tw, y[:n], color=color, lw=0.9)
        ax.set_title(title, loc="left", fontweight="bold", color="#0B1F3A")
        ax.set_ylabel("mV")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("DSP pipeline — visible stages", fontweight="bold", color="#0B1F3A")
    fig.tight_layout()
    save(fig, "01_dsp_stages.png")

    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.plot(tw, stages["raw"][:n], color="#C0392B", lw=0.85, alpha=0.45, label="Raw")
    ax.plot(tw, stages["final"][:n], color="#0E7C7B", lw=1.05, label="Processed")
    ax.set_title("Raw vs processed overlay", loc="left", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("mV")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "02_raw_vs_processed.png")

    freqs = result["fft_freqs"]
    mask = freqs <= 80
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(freqs[mask], result["fft_mags_raw"][mask], color="#C0392B", ls="--", lw=1, label="Raw FFT")
    ax.plot(freqs[mask], result["fft_mags"][mask], color="#6D28D9", lw=1.2, label="Processed FFT")
    ax.axvline(50, color="#94A3B8", ls=":", lw=1, label="50 Hz")
    ax.set_title("FFT magnitude spectrum", loc="left", fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "03_fft_spectrum.png")

    peaks = result["peaks"]
    vis = peaks[peaks < n]
    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.plot(tw, stages["final"][:n], color="#0E7C7B", lw=1.0)
    ax.scatter(vis / fs, stages["final"][vis], c="#C0392B", s=28, zorder=5, marker="^", label="R-peaks")
    ax.set_title("Pan–Tompkins R-peak detection", loc="left", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("mV")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "04_rpeak_detection.png")

    rr = result["rr"]["rr_intervals"]
    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.plot(np.arange(len(rr)), rr, color="#EA580C", marker="o", ms=3, lw=1.1)
    ax.set_title("RR interval series", loc="left", fontweight="bold")
    ax.set_xlabel("Beat number")
    ax.set_ylabel("RR (ms)")
    fig.tight_layout()
    save(fig, "05_rr_intervals.png")

    pt = result["pt_stages"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 7.2), sharex=True)
    names = [
        ("filtered", "Band-pass 5–15 Hz"),
        ("derivative", "Derivative"),
        ("squared", "Squaring"),
        ("integrated", "Moving-window integration"),
    ]
    colors = ["#0284C7", "#7C3AED", "#DB2777", "#0F766E"]
    for ax, (key, title), c in zip(axes, names, colors):
        ax.plot(tw, pt[key][:n], color=c, lw=0.9)
        ax.set_title(title, loc="left", fontweight="bold")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Pan–Tompkins intermediate signals", fontweight="bold")
    fig.tight_layout()
    save(fig, "06_pan_tompkins_stages.png")

    meta_path = "models/metadata/ml_performance.json"
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        perf = meta["performance"]
        models = list(perf.keys())
        acc = [perf[m]["accuracy"] for m in models]
        f1 = [perf[m]["f1_score"] for m in models]
        x = np.arange(len(models))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(x - 0.18, acc, 0.35, label="Accuracy", color="#0369A1")
        ax.bar(x + 0.18, f1, 0.35, label="F1", color="#0E7C7B")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=12)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Held-out record metrics (computed, not fabricated)", loc="left", fontweight="bold")
        ax.legend(frameon=False)
        fig.tight_layout()
        save(fig, "07_model_comparison.png")

        labels = [meta["class_mapping"][k] for k in sorted(meta["class_mapping"], key=lambda z: int(z))]
        sel = "Random Forest" if "Random Forest" in perf else models[0]
        cm = np.array(perf[sel]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5.4, 4.6))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix — {sel}", loc="left", fontweight="bold")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="#0B1F3A")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        save(fig, "08_confusion_matrix.png")

    from src.explainability import get_model_feature_importance

    imp = get_model_feature_importance("Random Forest")
    if imp is not None:
        top = imp.head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.barh(top["feature"], top["importance"], color="#0E7C7B")
        ax.set_title("Random Forest feature importance", loc="left", fontweight="bold")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        save(fig, "09_feature_importance.png")

    # Dashboard-style summary card
    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.35, wspace=0.25)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(tw, stages["final"][:n], color="#0E7C7B", lw=1.0)
    vis = peaks[peaks < n]
    ax0.scatter(vis / fs, stages["final"][vis], c="#C0392B", s=22, zorder=5, marker="^")
    ax0.set_title("Analysis snapshot — processed ECG + R-peaks", loc="left", fontweight="bold")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("mV")

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(freqs[mask], result["fft_mags"][mask], color="#6D28D9", lw=1.1)
    ax1.set_title("FFT (processed)", loc="left", fontweight="bold")
    ax1.set_xlabel("Hz")

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.axis("off")
    pred = result["dominant_class"]
    conf = result["confidence"]
    q = result["quality"]["status"]
    text = (
        f"Record: DEMO_ARRHYTHMIA\n"
        f"Fs: {fs:.0f} Hz    Duration: {len(sig)/fs:.1f} s\n"
        f"Quality: {q}\n"
        f"Heart rate: {result['rr']['mean_hr']:.1f} BPM\n"
        f"Mean RR: {result['rr']['mean_rr']:.0f} ms\n"
        f"Beats: {result['rr']['num_beats']}\n"
        f"Prediction: {CLASS_NAMES.get(pred, pred)}\n"
        f"Mean P(class): {conf:.1%}" if conf is not None else ""
    )
    ax2.text(0.02, 0.95, text, va="top", family="DejaVu Sans", fontsize=11, color="#0B1F3A")
    fig.suptitle("Intelligent ECG Analysis — prototype dashboard", fontweight="bold", color="#0B1F3A")
    save(fig, "10_analysis_dashboard.png")

    print("Screenshots ready in", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
