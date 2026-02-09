# ============================================================================
# CDN
# ============================================================================

FONT_AWESOME_CDN = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
"""

# ============================================================================
# Custom CSS
# ============================================================================

CUSTOM_CSS = """
<style>

/* === LAYOUT & BACKGROUND === */

.stApp {
    background:
        radial-gradient(1200px 600px at 20% 10%, rgba(99,110,250,0.16), transparent 60%),
        radial-gradient(900px 500px at 80% 20%, rgba(0,204,150,0.12), transparent 65%),
        radial-gradient(900px 500px at 60% 80%, rgba(239,85,59,0.10), transparent 65%),
        #0e1117;
}

.main {
    padding: 0rem 1rem;
}

div[data-testid="stAppViewContainer"] .main .block-container {
    padding-top: 1rem !important;
}

/* === TYPOGRAPHY === */

h1 {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    font-size: 3rem;
    font-weight: 700;
    margin-top: 0.2rem !important;
    margin-bottom: 1.2rem !important;
}

.section-title,
.sidebar-center-title {
    text-align: center;
    font-weight: 800;
    background: linear-gradient(90deg, #5c6ac4 0%, #43389b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.section-title {
    font-size: 1.5rem;
    margin-top: 0;
    margin-bottom: 0.5rem;
}

.sidebar-center-title {
    font-size: 1.25rem;
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
    opacity: 0.95;
}

/* === FORM ELEMENTS === */

div[data-testid="stTextArea"] textarea {
    white-space: pre !important;
    overflow-wrap: normal !important;
    word-break: normal !important;
    overflow-x: auto !important;
}

/* === BUTTONS === */

.stButton > button {
    background: linear-gradient(90deg, #5c6ac4 0%, #43389b 100%);
    color: white;
    border: none;
    padding: 0.75rem 2rem;
    font-size: 1.1rem;
    font-weight: 600;
    border-radius: 50px;
    transition: all 0.3s;
    width: 100%;
}

.stButton > button:hover {
    transform: scale(1.03);
    box-shadow: 0 5px 15px rgba(102,126,234,0.4);
}

div[data-testid="stButton"] button[kind="secondary"],
div[data-testid="stButton"] button[data-testid="baseButton-secondary"] {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.16) !important;
    color: rgba(255,255,255,0.92) !important;
    padding: 0.55rem 1.1rem !important;
    border-radius: 14px !important;
    white-space: nowrap !important;
}

div[data-testid="stButton"] button[kind="secondary"]:hover,
div[data-testid="stButton"] button[data-testid="baseButton-secondary"]:hover {
    border-color: rgba(255,255,255,0.26) !important;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25) !important;
    transform: translateY(-1px) !important;
}

/* === ALERTS === */

.stAlert,
.element-container:has(.stWarning) div[role="alert"],
.element-container:has(.stInfo) div[role="alert"],
.element-container:has(.stSuccess) div[role="alert"],
.element-container:has(.stError) div[role="alert"] {
    background-color: rgb(28,31,48) !important;
    border: 1px solid rgb(49,51,63) !important;
    border-radius: 0.5rem !important;
    color: rgba(255,255,255,0.88) !important;
}

.stAlert svg { display: none !important; }
.stAlert p, .stAlert div { color: rgba(255,255,255,0.88) !important; }

/* === METRICS === */

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 10px 26px rgba(0,0,0,0.35);
    transition: transform 120ms ease, border-color 120ms ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: rgba(255,255,255,0.22);
}

div[data-testid="stMetricLabel"] {
    font-size: 0.85rem !important;
    opacity: 0.85;
    letter-spacing: 0.02em;
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p {
    white-space: normal !important;
    overflow-wrap: anywhere !important;
    text-overflow: unset !important;
}

div[data-testid="stMetricValue"] {
    font-size: 2.1rem !important;
    font-weight: 800 !important;
    line-height: 1.15;
}

/* === TOKEN NOTE === */

.token-line-note {
    text-align: center;
    opacity: 0.9;
    font-size: 0.90rem;
    margin-top: -10px;
    white-space: nowrap;
}

/* === TABS === */

.stTabs [data-baseweb="tab-list"] { justify-content: center; }
.stTabs [data-baseweb="tab-list"] button { font-size: 1.1rem; padding: 0.6rem 1.2rem; }
.stTabs [data-baseweb="tab-highlight"] { transition: left 0.2s ease, width 0.2s ease; }

/* === PLOTLY === */

div[data-testid="stPlotlyChart"] .js-plotly-plot .plotly .draglayer,
div[data-testid="stPlotlyChart"] .js-plotly-plot .plotly .draglayer *,
div[data-testid="stPlotlyChart"] .js-plotly-plot .plotly .nsewdrag {
    cursor: crosshair !important;
}

</style>
"""
