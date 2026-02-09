# ============================================================================
# Model Configuration
# ============================================================================

MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_ORG = "sentence-transformers"

# ============================================================================
# External URLs
# ============================================================================

HF_BASE_URL = "https://huggingface.co"
SBERT_URL = "https://www.sbert.net/"
LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0"

# ============================================================================
# Input Limits
# ============================================================================

MAX_INPUT_TEXTS = 15
MAX_SHOWN_LINES = 6

# ============================================================================
# Clustering
# ============================================================================

DEFAULT_MAX_CLUSTERS = 5
MIN_CLUSTERS = 2
MAX_CLUSTERS_SLIDER = 10
DEFAULT_CLUSTERS_SLIDER = 3
KMEANS_N_INIT = 10
RANDOM_SEED = 42

# ============================================================================
# UI Controls
# ============================================================================

THRESHOLD_MIN = 0.0
THRESHOLD_MAX = 1.0
THRESHOLD_DEFAULT = 0.0
THRESHOLD_STEP = 0.05
TEXTAREA_HEIGHT = 350
TEXTAREA_MAX_CHARS = 20000

# ============================================================================
# Chart Dimensions
# ============================================================================

CHART_HEIGHT = 600
CHART_HEIGHT_SMALL = 450
CHART_FONT_SIZE = 16

# ============================================================================
# Hover Formatting
# ============================================================================

HOVER_WIDTH_STANDARD = 70
HOVER_WIDTH_EXTENDED = 80
HOVER_LINES_STANDARD = 12
HOVER_LINES_EXTENDED = 24
HOVER_CHARS_STANDARD = 1400
HOVER_CHARS_EXTENDED = 4000

# ============================================================================
# Node Visualization
# ============================================================================

NODE_SIZE_BASE = 20.0
NODE_SIZE_SCALE = 40.0
CLUSTER_MARKER_SIZE = 22.0


# ============================================================================
# Plotly Configuration
# ============================================================================

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
    "scrollZoom": True,
    "doubleClick": "reset",
}

MODEBAR_STYLE = {
    "orientation": "h",
    "bgcolor": "rgba(0,0,0,0)",
    "color": "rgba(255,255,255,0.85)",
    "activecolor": "rgba(255,255,255,1)",
}

BASE_LAYOUT = {
    "template": "plotly_dark",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "white"},
}

GRID_AXIS = {
    "showgrid": True,
    "zeroline": True,
    "showticklabels": False,
    "gridcolor": "rgba(255,255,255,0.1)",
}

HIDDEN_AXIS = {
    "showgrid": False,
    "zeroline": False,
    "showticklabels": False,
}

LEGEND_CONFIG = {"title": "Clusters", "orientation": "v"}

# ============================================================================
# Color Palette
# ============================================================================

VIRIDIS_COLORS = [
    "#440154",
    "#31688E",
    "#35B779",
    "#FDE725",
    "#3E4989",
    "#1F9E89",
    "#B5DE2B",
    "#482878",
    "#26828E",
    "#6ECE58",
]

# ============================================================================
# Styling
# ============================================================================

FONT_AWESOME_CDN = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
"""

CUSTOM_CSS = """
<style>
/* === TYPOGRAPHY === */
h1{
background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
text-align:center;
font-size:3rem;
font-weight:700;
margin-top:0.2rem!important;
margin-bottom:1.2rem!important;}
.section-title,.sidebar-center-title{
text-align:center;
font-weight:800;
background:linear-gradient(90deg,#5c6ac4 0%,#43389b 100%);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;}
.section-title{
font-size:1.5rem;
margin-top:0;
margin-bottom:0.5rem;}
.sidebar-center-title{
font-size:1.25rem;
margin-top:0.5rem;
margin-bottom:0.5rem;
opacity:0.95;}

/* === LAYOUT & BACKGROUND === */
.main{padding:0rem 1rem;}
div[data-testid="stAppViewContainer"] .main .block-container{padding-top:1rem!important;}
.stApp{
background:radial-gradient(1200px 600px at 20% 10%,rgba(99,110,250,0.16),transparent 60%),
radial-gradient(900px 500px at 80% 20%,rgba(0,204,150,0.12),transparent 65%),
radial-gradient(900px 500px at 60% 80%,rgba(239,85,59,0.10),transparent 65%),
#0e1117;}

/* === ALERTS === */
.stAlert,
.element-container:has(.stWarning) div[role="alert"],
.element-container:has(.stInfo) div[role="alert"],
.element-container:has(.stSuccess) div[role="alert"],
.element-container:has(.stError) div[role="alert"]{
background-color:rgb(28,31,48)!important;
border:1px solid rgb(49,51,63)!important;
border-radius:0.5rem!important;
color:rgba(255,255,255,0.88)!important;}
.stAlert svg{display:none!important;}
.stAlert p,.stAlert div{color:rgba(255,255,255,0.88)!important;}

/* === BUTTONS === */
.stButton > button{
background:linear-gradient(90deg,#5c6ac4 0%,#43389b 100%);
color:white;
border:none;
padding:0.75rem 2rem;
font-size:1.1rem;
font-weight:600;
border-radius:50px;
transition:all 0.3s;
width:100%;}
.stButton > button:hover{
transform:scale(1.03);
box-shadow:0 5px 15px rgba(102,126,234,0.4);}
div[data-testid="stButton"] button[kind="secondary"],
div[data-testid="stButton"] button[data-testid="baseButton-secondary"]{
background:rgba(255,255,255,0.07)!important;
border:1px solid rgba(255,255,255,0.16)!important;
color:rgba(255,255,255,0.92)!important;
padding:0.55rem 1.1rem!important;
border-radius:14px!important;
white-space:nowrap!important;}
div[data-testid="stButton"] button[kind="secondary"]:hover,
div[data-testid="stButton"] button[data-testid="baseButton-secondary"]:hover{
border-color:rgba(255,255,255,0.26)!important;
box-shadow:0 6px 18px rgba(0,0,0,0.25)!important;
transform:translateY(-1px)!important;}

/* === METRICS === */
div[data-testid="stMetric"]{
background:rgba(255,255,255,0.06);
border:1px solid rgba(255,255,255,0.12);
border-radius:14px;
padding:14px 16px;
box-shadow:0 10px 26px rgba(0,0,0,0.35);
transition:transform 120ms ease,border-color 120ms ease;}
div[data-testid="stMetric"]:hover{
transform:translateY(-2px);
border-color:rgba(255,255,255,0.22);}
div[data-testid="stMetricLabel"]{
font-size:0.85rem!important;
opacity:0.85;
letter-spacing:0.02em;}
div[data-testid="stMetricLabel"],div[data-testid="stMetricLabel"] p{
white-space:normal!important;
overflow-wrap:anywhere!important;
text-overflow:unset!important;}
div[data-testid="stMetricValue"]{
font-size:2.1rem!important;
font-weight:800!important;
line-height:1.15;}

/* === FORM ELEMENTS === */
div[data-testid="stTextArea"] textarea{
white-space:pre!important;
overflow-wrap:normal!important;
word-break:normal!important;
overflow-x:auto!important;}

/* === TABS === */
.stTabs [data-baseweb="tab-list"]{justify-content:center;}
.stTabs [data-baseweb="tab-list"] button{font-size:1.1rem;padding:0.6rem 1.2rem;}
.stTabs [data-baseweb="tab-highlight"]{transition:left 0.2s ease,width 0.2s ease;}

/* === CUSTOM COMPONENTS === */
.token-line-note{
text-align:center;
opacity:0.9;
font-size:0.90rem;
margin-top:-10px;
white-space:nowrap;}

/* === PLOTLY OVERRIDES === */
div[data-testid="stPlotlyChart"] .js-plotly-plot .plotly .draglayer,
div[data-testid="stPlotlyChart"] .js-plotly-plot .plotly .draglayer *,
div[data-testid="stPlotlyChart"] .js-plotly-plot .plotly .nsewdrag{cursor:crosshair!important;}
</style>
"""
