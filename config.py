from collections import namedtuple

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

AUTO_CLUSTER_MAX_K = 5
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

# ============================================================================
# Sample Data
# ============================================================================

SAMPLE_TEXTS = """Barcelona is a popular travel destination with its stunning architecture and dynamic atmosphere.
Tokyo is a fascinating travel destination where tradition and innovation meet.
The United States of America offers a wide variety of travel experiences.
Antalya is one of the most popular tourist destinations in Turkey.
The lights and sounds at the festival created an unforgettable experience for everyone.
The movie got more interesting as the story went on.
The use of mirrors in horror films often suggests a hidden danger.
The atmosphere in the stadium during a big match can be electric.
Experimenting with different charts makes it easier to reveal hidden insights.
Data visualization is an effective technique to understand complex information."""

TAB_LABELS = ["🌐 Network", "⚛️ Clusters", "🏆 Top Pairs"]

# ============================================================================
# Metric Descriptions
# ============================================================================

MetricDescription = namedtuple("MetricDescription", ["label", "fmt", "help"])

METRIC_DESCRIPTIONS = {
    "network": [
        MetricDescription(
            "Average pair cosine similarity", ".3f",
            "Average cosine similarity across all text pairs. "
            "Values range from -1 (opposite) to 1 (very similar).",
        ),
        MetricDescription(
            "Maximum pair cosine similarity", ".3f",
            "Highest cosine similarity among all text pairs. "
            "Values near 1 indicate nearly identical embeddings.",
        ),
        MetricDescription(
            "Minimum pair cosine similarity", ".3f",
            "Lowest cosine similarity among all text pairs. "
            "Values near -1 indicate highly dissimilar embeddings.",
        ),
        MetricDescription(
            "Average degree", ".2f",
            "Average number of connections per node "
            "in the similarity network (edges above the threshold).",
        ),
        MetricDescription(
            "Network density", ".3f",
            "Ratio of actual connections to all possible "
            "connections in the similarity network.",
        ),
        MetricDescription(
            "Top degree nodes", "",
            "Nodes (text indices) with the highest number of connections; "
            "displayed as Text ID (degree).",
        ),
    ],
    "cluster_overview": [
        MetricDescription("Number of clusters:", "", None),
        MetricDescription("Smallest cluster size:", "", None),
        MetricDescription("Largest cluster size:", "", None),
    ],
    "cluster_detail": [
        MetricDescription(
            "Silhouette score (cosine distance):", ".3f",
            "Measures how well each text fits within its cluster "
            "compared to other clusters; scores range from -1 to 1.",
        ),
        MetricDescription(
            "Average within-cluster cosine similarity:", ".3f",
            "Average cosine similarity of text pairs within the same cluster.",
        ),
        MetricDescription(
            "Calinski-Harabasz Index:", ".1f",
            "Calinski-Harabasz Index (Variance Ratio Criterion). "
            "Measures cluster separation quality; higher values indicate "
            "better-defined, well-separated clusters.",
        ),
        MetricDescription(
            "Average between-cluster cosine similarity:", ".3f",
            "Average cosine similarity of text pairs belonging to different clusters.",
        ),
    ],
}