from collections import namedtuple

import streamlit as st

import config
import styles
from state import (
    STATE_ANALYSIS_HASH,
    STATE_ANALYZER,
    STATE_CLUSTER_COUNT,
    build_token_limit_error,
    get_analyzer,
    init_state,
    update_token_stats,
)
from ui_components import (
    render_input_actions,
    render_sidebar_controls,
    render_stats_panel,
    render_text_area,
)
from utils import (
    cluster_partitions,
    compute_content_hash,
    parse_texts,
    upper_triangle,
)
from visualizer import PlotlyVisualizer

st.set_page_config(
    page_title="Similyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Constants
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

MetricDescription = namedtuple("MetricDescription", ["label", "format", "help"])

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


# ============================================================================
# UI Helpers
# ============================================================================


def show_chart(figure):
    """Display a Plotly figure with consistent container and config settings."""
    st.plotly_chart(figure, use_container_width=True, config=config.PLOTLY_CONFIG)


def fa_heading(icon, text, level=3):
    """Render a Font Awesome icon heading, centered."""
    st.markdown(
        f"<h{level} style='text-align:center;'>"
        f"<i class='fa-solid fa-{icon}'></i> {text}"
        f"</h{level}>",
        unsafe_allow_html=True,
    )


def format_metric(value, format=".3f"):
    """Format a metric value for display, returning 'N/A' if None."""
    if value is None:
        return "N/A"
    if isinstance(value, str) or not format:
        return str(value)
    return f"{value:{format}}"


def render_metrics_grid(descriptions, values, columns=2):
    """Render a grid of st.metric widgets from descriptions and values."""
    for row_start in range(0, len(descriptions), columns):
        row = zip(descriptions[row_start : row_start + columns],
                  values[row_start : row_start + columns])
        cols = st.columns(columns)
        for col, (desc, value) in zip(cols, row):
            with col:
                st.metric(desc.label, format_metric(value, desc.format), help=desc.help)


# ============================================================================
# Validation
# ============================================================================


def _validate_input(texts, current_hash):
    """Return an error message string if input is invalid, otherwise None."""
    if not texts:
        return "Please enter at least 1 text."

    if len(texts) > config.MAX_INPUT_TEXTS:
        return (
            f"Please use {config.MAX_INPUT_TEXTS} texts or fewer. "
            f"You entered {len(texts)} texts."
        )

    analyzer = get_analyzer(config.MODEL_NAME)
    token_stats = update_token_stats(analyzer, texts, current_hash)
    token_error = build_token_limit_error(token_stats)
    if token_error:
        return token_error

    if st.session_state.get(STATE_ANALYSIS_HASH) == current_hash:
        return "Analysis is already up to date."

    return None


# ============================================================================
# Analysis Pipeline
# ============================================================================


def _run_analysis(analyzer, texts, num_clusters):
    """Execute the full embedding → similarity → clustering pipeline."""
    analyzer.add_sentences(texts)
    analyzer.get_embeddings()
    analyzer.calculate_similarity()
    analyzer.reduce_dimensions()
    analyzer.perform_clustering(num_clusters)


def handle_analyze_click(texts, num_clusters, current_hash):
    """Validate input and trigger analysis when Analyze button is clicked."""
    error = _validate_input(texts, current_hash)
    if error:
        (st.info if "up to date" in error else st.error)(error)
        return

    with st.spinner("Analyzing..."):
        try:
            analyzer = get_analyzer(config.MODEL_NAME)
            _run_analysis(analyzer, texts, num_clusters)
            st.session_state[STATE_ANALYSIS_HASH] = current_hash
            st.session_state[STATE_CLUSTER_COUNT] = num_clusters
        except Exception as error:
            st.error(f"Analysis failed: {error}")


# ============================================================================
# Tab Renderers
# ============================================================================


def render_network_tab(analyzer, visualizer, threshold):
    """Render the similarity network graph and associated metrics."""
    network_figure, network_stats = visualizer.create_similarity_network(threshold=threshold)
    show_chart(network_figure)

    if analyzer.similarity_matrix is None or len(analyzer.sentences) == 0:
        return

    pairwise = upper_triangle(analyzer.similarity_matrix)
    if pairwise.size == 0:
        avg_sim, min_sim, max_sim = 0.0, 0.0, 0.0
    else:
        avg_sim = float(pairwise.mean())
        min_sim = float(pairwise.min())
        max_sim = float(pairwise.max())

    top_nodes = network_stats["top_nodes"]
    top_nodes_display = ", ".join(f"{n} ({d})" for n, d in top_nodes) if top_nodes else "None"

    values = [avg_sim, max_sim, min_sim, network_stats["avg_degree"],
              network_stats["density"], top_nodes_display]
    render_metrics_grid(METRIC_DESCRIPTIONS["network"], values)


def render_clusters_tab(analyzer, visualizer):
    """Render the cluster visualization, metrics, and per-cluster text lists."""
    show_chart(visualizer.create_cluster_visualization())

    if analyzer.cluster_labels is None:
        return

    cluster_indices = cluster_partitions(analyzer.cluster_labels)
    cluster_sizes = [len(members) for members in cluster_indices]

    # Overview
    fa_heading("eye", "Overview")
    overview_values = [len(cluster_sizes), min(cluster_sizes, default=0),
                       max(cluster_sizes, default=0)]
    render_metrics_grid(METRIC_DESCRIPTIONS["cluster_overview"], overview_values, columns=3)

    # Metrics
    fa_heading("chart-line", "Metrics")
    detail_values = [analyzer.silhouette, analyzer.avg_within_cluster,
                     analyzer.calinski_harabasz, analyzer.avg_between_clusters]
    render_metrics_grid(METRIC_DESCRIPTIONS["cluster_detail"], detail_values)

    # Texts by cluster
    fa_heading("layer-group", "Texts by Cluster")
    for cluster_id, text_indices in enumerate(cluster_indices):
        with st.expander(f"Cluster {cluster_id + 1} ({len(text_indices)} texts)"):
            for i in text_indices:
                st.write(f"**Text {i + 1}:** {analyzer.sentences[i]}")


def render_pairs_tab(visualizer):
    """Render the top similar pairs bar chart."""
    show_chart(visualizer.create_top_pairs_chart(num_pairs=10))


# ============================================================================
# Main
# ============================================================================


def _render_results(texts, num_clusters, current_hash, threshold):
    """Check analysis state and render result tabs if ready."""
    analyzer = st.session_state.get(STATE_ANALYZER)
    analysis_hash = st.session_state.get(STATE_ANALYSIS_HASH)

    if analyzer is None or analysis_hash is None:
        return

    # Check if results are stale
    if current_hash != analysis_hash:
        msg = ("Input is empty. Enter texts and click Analyze to see results."
               if not texts else
               "Input or model changed. Click Analyze to refresh results.")
        (st.info if not texts else st.warning)(msg)
        return

    # Re-cluster if cluster count changed (without full re-analysis)
    if num_clusters != st.session_state.get(STATE_CLUSTER_COUNT):
        try:
            analyzer.perform_clustering(num_clusters)
            st.session_state[STATE_CLUSTER_COUNT] = num_clusters
        except Exception as error:
            st.error(f"Re-clustering failed: {error}. Click Analyze to recompute.")
            return

    # Render analysis results
    visualizer = PlotlyVisualizer(analyzer)

    st.markdown("---")
    fa_heading("chart-pie", "Analysis Results", level=2)

    network_tab, clusters_tab, pairs_tab = st.tabs(TAB_LABELS)

    with network_tab:
        render_network_tab(analyzer, visualizer, threshold)
    with clusters_tab:
        render_clusters_tab(analyzer, visualizer)
    with pairs_tab:
        render_pairs_tab(visualizer)


def main():
    st.markdown(styles.FONT_AWESOME_CDN, unsafe_allow_html=True)
    st.markdown(styles.CUSTOM_CSS, unsafe_allow_html=True)
    fa_heading("magnifying-glass-chart", "Similyzer", level=1)
    init_state()

    # Sidebar controls
    num_clusters, threshold = render_sidebar_controls()

    # Header row: input actions and stats title
    header_left, header_right = st.columns([3, 1])
    with header_left:
        render_input_actions(SAMPLE_TEXTS)
    with header_right:
        st.markdown(
            "<div class='section-title'>"
            "<i class='fa-solid fa-chart-simple'></i> Input Statistics"
            "</div>",
            unsafe_allow_html=True,
        )

    # Body row: text area and stats panel
    body_left, body_right = st.columns([3, 1])
    with body_left:
        raw_input = render_text_area()

    texts = parse_texts(raw_input)
    current_hash = compute_content_hash(config.MODEL_NAME, texts)

    with body_right:
        stats_placeholder = st.empty()

    # Analyze button
    analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

    if analyze_clicked:
        handle_analyze_click(texts, num_clusters, current_hash)

    with stats_placeholder.container():
        render_stats_panel(texts, current_hash)

    _render_results(texts, num_clusters, current_hash, threshold)


if __name__ == "__main__":
    main()