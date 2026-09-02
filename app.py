import os
import streamlit as st

st.set_page_config(
    page_title="Intelligent ECG Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
    --bg-0: #0A0E17;
    --bg-1: #0D1220;
    --surface: #141B2D;
    --surface-2: #171F33;
    --border: #253049;
    --text: #E7ECF5;
    --text-muted: #93A0BC;
    --accent-teal: #2DD4BF;
    --accent-red: #FF6B6B;
    --accent-amber: #FBBF24;
    --accent-purple: #A78BFA;
}

html, body, [class*="css"], .stMarkdown, .stText {
    font-family: 'Outfit', 'IBM Plex Sans', sans-serif;
    color: var(--text);
}
.stApp { background: linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 100%); }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] { background: transparent; }

/* Headings / body text */
h1, h2, h3, h4, h5, h6 { color: var(--text) !important; }
p, span, label, .stMarkdown, .stCaption, div[data-testid="stMarkdownContainer"] { color: var(--text); }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--text-muted) !important; }

.hero-card {
    background: linear-gradient(135deg, #0B1F3A 0%, #0E3A52 58%, #0E7C7B 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 28px 32px;
    color: white;
    margin-bottom: 18px;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45);
}
.hero-card h1 { font-size: 1.85rem; font-weight: 800; margin: 0 0 6px 0; letter-spacing: -0.03em; color: white !important; }
.hero-card p { margin: 0; opacity: 0.92; font-size: 0.98rem; font-weight: 400; color: #DCE6EE; }

.pipe {
    display: flex; flex-wrap: wrap; gap: 6px; margin-top: 16px;
}
.pipe span {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.22);
    padding: 4px 10px; border-radius: 999px; font-size: 11px; letter-spacing: 0.04em;
    color: #EAF2F6;
}

.metric-box {
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
    padding: 14px 16px; box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
}
.q-good { color: #34D399; font-weight: 700; }
.q-ok { color: #FBBF24; font-weight: 700; }
.q-poor { color: #F87171; font-weight: 700; }

.disclaimer-banner {
    font-size: 11px; color: #FCA5A5; background: #2A1216;
    border-left: 3px solid #F87171; padding: 10px 12px; border-radius: 8px; margin-top: 12px; line-height: 1.45;
}
.sidebar-title { font-size: 20px; font-weight: 800; color: var(--text); margin-bottom: 2px; }
.sidebar-subtitle { font-size: 11px; color: var(--text-muted); letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 12px; }

div[data-testid="stMetric"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: 8px 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.25);
}
div[data-testid="stMetricLabel"] { color: var(--text-muted) !important; }
div[data-testid="stMetricValue"] { color: var(--text) !important; }

section[data-testid="stSidebar"] { background: var(--bg-1); border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] * { color: var(--text); }

/* Inputs, selects, uploaders, expanders, dataframes: sit on the same dark surface */
div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input,
div[data-testid="stFileUploaderDropzone"], .streamlit-expanderHeader {
    background-color: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
div[data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; }
div[data-testid="stDataFrame"] { background: var(--surface); border-radius: 10px; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
    border-radius: 10px; border: 1px solid var(--border);
    background: var(--surface-2); color: var(--text);
}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
    background: linear-gradient(135deg, #E63946 0%, #FF6B6B 100%);
    border: none; color: white; font-weight: 600;
}

/* Alerts (info/warning/error/success) */
div[data-testid="stAlert"] { border-radius: 10px; }
</style>
""",
    unsafe_allow_html=True,
)

from src.ui_pages import (
    render_home_page,
    render_dsp_page,
    render_features_page,
    render_prediction_page,
    render_performance_page,
    render_history_page,
    render_about_page,
)

defaults = {
    "record_name": "",
    "data_loaded": False,
    "fs": 360.0,
    "sig_raw": None,
    "sig_clean": None,
    "dsp_stages": None,
    "ann_sample": None,
    "ann_symbol": None,
    "dsp_applied": False,
    "peaks": None,
    "pt_stages": None,
    "rr_metrics": None,
    "feat_df": None,
    "pred_df": None,
    "quality_metrics": None,
    "quality_status": "UNKNOWN",
    "quality_details": "",
    "fft_freqs": None,
    "fft_mags": None,
    "fft_mags_raw": None,
    "dominant_freq": 0.0,
    "spectral_bands": None,
    "is_simulated": True,
    "model_name": "Random Forest",
    "dominant_class": None,
    "pred_confidence": None,
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

with st.sidebar:
    st.markdown('<div class="sidebar-title">🩺 CardioDSP Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">ECG · DSP · ML prototype</div>', unsafe_allow_html=True)

    page = st.radio(
        "Workspace",
        [
            "Home",
            "DSP Analysis",
            "ECG Features",
            "ML Prediction",
            "Model Performance",
            "History",
            "About",
        ],
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Active record")
    if st.session_state["data_loaded"]:
        st.success(
            f"**{st.session_state['record_name']}**\n\n"
            f"{st.session_state['fs']:.0f} Hz · quality **{st.session_state['quality_status']}**"
        )
    else:
        st.info("Load a demo or CSV on Home.")

    st.markdown(
        """
    <div class="disclaimer-banner">
        <b>Academic prototype.</b> Not a certified medical device. Do not use for diagnosis or emergency decisions.
    </div>
    """,
        unsafe_allow_html=True,
    )

if page == "Home":
    render_home_page()
elif page == "DSP Analysis":
    render_dsp_page()
elif page == "ECG Features":
    render_features_page()
elif page == "ML Prediction":
    render_prediction_page()
elif page == "Model Performance":
    render_performance_page()
elif page == "History":
    render_history_page()
else:
    render_about_page()