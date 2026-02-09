# ============================================================================
# Model Configuration
# ============================================================================

MODEL_NAME = "all-MiniLM-L6-v2"

# ============================================================================
# Input Limits
# ============================================================================

MAX_INPUT_TEXTS = 15
MAX_SHOWN_LINES = 6

# ============================================================================
# Clustering
# ============================================================================

DEFAULT_MAX_CLUSTERS = 5
KMEANS_N_INIT = 10
RANDOM_SEED = 42


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
