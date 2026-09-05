"""Consistent scientific ECG plots for Streamlit (Plotly) and screenshots. — Dark theme."""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_dark"

ECG_RED = "#FF6B6B"
CLEAN_TEAL = "#2DD4BF"
NAVY = "#E7ECF5"
AMBER = "#FBBF24"
PURPLE = "#A78BFA"

PAPER = "rgba(0,0,0,0)"
PLOT_BG = "rgba(255,255,255,0.03)"
GRID = "rgba(148,163,184,0.16)"
GRID_MAJOR = "rgba(148,163,184,0.30)"
AXIS_LINE = "rgba(148,163,184,0.35)"
FONT_COLOR = "#C7D0E0"
TITLE_COLOR = "#F1F5F9"


def make_ecg_figure(
    t, y, title, name="ECG", color=ECG_RED, peaks=None, peak_y=None, height=280,
    y_title="Amplitude (mV)", x_title="Time (s)",
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t, y=y, mode="lines", name=name,
            line=dict(color=color, width=1.6),
            hovertemplate="%{x:.3f} s<br>%{y:.3f} mV<extra></extra>",
        )
    )
    if peaks is not None and len(peaks) > 0:
        py = peak_y if peak_y is not None else y[peaks] if hasattr(y, "__getitem__") else None
        fig.add_trace(
            go.Scatter(
                x=peaks, y=py, mode="markers", name="R-peaks",
                marker=dict(color=AMBER, size=9, symbol="triangle-up", line=dict(width=0)),
            )
        )
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=TITLE_COLOR, family="Outfit, Segoe UI, sans-serif")),
        height=height, margin=dict(l=50, r=18, t=48, b=42),
        paper_bgcolor=PAPER, plot_bgcolor=PLOT_BG, font=dict(color=FONT_COLOR, size=12),
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title=x_title, showgrid=True, gridcolor=GRID, zeroline=False, showline=True, linecolor=AXIS_LINE),
        yaxis=dict(title=y_title, showgrid=True, gridcolor=GRID, zeroline=False, showline=True, linecolor=AXIS_LINE),
    )
    return fig


def overlay_ecg(t, series, title, height=340):
    fig = go.Figure()
    for y, name, color, dash in series:
        fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=name, line=dict(color=color, width=1.5, dash=dash or "solid")))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=TITLE_COLOR)),
        height=height, margin=dict(l=50, r=18, t=48, b=42),
        paper_bgcolor=PAPER, plot_bgcolor=PLOT_BG, font=dict(color=FONT_COLOR, size=12),
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Time (s)", showgrid=True, gridcolor=GRID, zeroline=False, linecolor=AXIS_LINE),
        yaxis=dict(title="Amplitude (mV)", showgrid=True, gridcolor=GRID, zeroline=False, linecolor=AXIS_LINE),
    )
    return fig


def window_signal(sig, fs, t0, t1):
    n = len(sig)
    i0 = max(0, int(t0 * fs))
    i1 = min(n, int(t1 * fs))
    if i1 <= i0:
        i1 = min(n, i0 + int(fs))
    t = np.arange(i0, i1) / fs
    return t, sig[i0:i1], i0, i1
