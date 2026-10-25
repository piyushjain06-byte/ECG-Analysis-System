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
:root {
    --bg: #F4F7FB;
    --surface: #FFFFFF;
    --surface-soft: #F8FAFC;
    --border: #D8E1EC;
    --text: #172B4D;
    --text-muted: #52657D;
    --accent-teal: #087E75;
    --accent-red: #C62828;
    --accent-amber: #8A5A00;
    --accent-purple: #6842A6;
}

html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea {
    font-family: Inter, "Segoe UI", Arial, sans-serif;
    color: var(--text);
}
.stApp { background: var(--bg); }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] { background: rgba(244, 247, 251, 0.92); }

h1, h2, h3, h4, h5, h6 { color: var(--text) !important; letter-spacing: -0.02em; }
p, label, .stMarkdown, div[data-testid="stMarkdownContainer"] { color: var(--text); }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--text-muted) !important; }

.hero-card {
    background: linear-gradient(120deg, #FFFFFF 0%, #EAF4FA 58%, #E5F5F1 100%);
    border: 1px solid #C7D9E8;
    border-left: 5px solid var(--accent-teal);
    border-radius: 16px;
    padding: 26px 30px;
    color: var(--text);
    margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(23, 43, 77, 0.07);
}
.hero-card h1 { font-size: 1.85rem; font-weight: 750; margin: 0 0 6px 0; color: var(--text) !important; }
.hero-card p { margin: 0; font-size: 1rem; line-height: 1.55; color: #344B63; }

.pipe {
    display: flex; flex-wrap: wrap; gap: 6px; margin-top: 16px;
}
.pipe span {
    background: #FFFFFF;
    border: 1px solid #C7D9E8;
    padding: 5px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 650;
    color: #17445A;
}

.metric-box {
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
    padding: 14px 16px; box-shadow: 0 4px 14px rgba(23, 43, 77, 0.06);
}
.q-good { color: #146C43; font-weight: 700; }
.q-ok { color: #805400; font-weight: 700; }
.q-poor { color: #A61B1B; font-weight: 700; }

.disclaimer-banner {
    font-size: 0.78rem; color: #7A2020; background: #FFF3F1;
    border: 1px solid #F3C5BF; border-left: 4px solid #B42318;
    padding: 10px 12px; border-radius: 8px; margin-top: 12px; line-height: 1.5;
}
.sidebar-title { font-size: 20px; font-weight: 750; color: var(--text); margin-bottom: 2px; }
.sidebar-subtitle { font-size: 0.72rem; color: var(--text-muted); letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 12px; }

div[data-testid="stMetric"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: 10px 13px; box-shadow: 0 2px 8px rgba(23, 43, 77, 0.05);
}
div[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important; font-size: 0.8rem; line-height: 1.3;
    white-space: normal;
}
div[data-testid="stMetricValue"] {
    color: var(--text) !important; font-size: clamp(1rem, 1.5vw, 1.35rem) !important;
    line-height: 1.3;
}
div[data-testid="stMetricValue"] > div { white-space: normal; overflow-wrap: anywhere; }

section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p { color: var(--text); }

/* Keep controls visibly distinct and keyboard-focusable. */
div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input,
div[data-testid="stFileUploaderDropzone"], .streamlit-expanderHeader,
div[data-baseweb="textarea"] textarea {
    background-color: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
div[data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; }
div[data-testid="stDataFrame"] { background: var(--surface); border-radius: 10px; }
button:focus-visible, input:focus-visible, textarea:focus-visible,
[role="radiogroup"] button:focus-visible {
    outline: 3px solid #176B87 !important;
    outline-offset: 2px !important;
}

.stButton > button, .stDownloadButton > button {
    min-height: 42px;
    border-radius: 9px; border: 1px solid #B8C8D9;
    background: #FFFFFF; color: var(--text);
    font-weight: 600;
}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
    background: #135D72;
    border: 1px solid #135D72; color: #FFFFFF; font-weight: 650;
}
.stButton > button[kind="primary"] p, .stDownloadButton > button[kind="primary"] p {
    color: #FFFFFF !important;
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color: #135D72; color: #104B5B; }
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {
    background: #104B5B; color: #FFFFFF;
}

div[data-testid="stAlert"] { border-radius: 10px; }
button[data-baseweb="tab"] { color: #42566F; }
button[data-baseweb="tab"][aria-selected="true"] { color: #135D72; }
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
from src.sns_lab import render_sns_lab_page

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
            "Signals & Systems Lab",
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
elif page == "Signals & Systems Lab":
    render_sns_lab_page()
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