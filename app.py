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
# Analysis Pipeline
# ============================================================================


def run_full_analysis(model_name, texts, num_clusters):
    analyzer = get_analyzer(model_name)
    analyzer.add_sentences(texts)
    analyzer.get_embeddings()
    analyzer.calculate_similarity()
    analyzer.reduce_dimensions()
    analyzer.perform_clustering(num_clusters)


def handle_analyze_click(
    analyze_clicked, texts, model_name, num_clusters, current_hash
):
    """Validate input and trigger analysis when Analyze button is clicked."""
    if not analyze_clicked:
        return

    # Validate input count
    if not texts:
        st.error("Please enter at least 1 text.")
        return

    if len(texts) > config.MAX_INPUT_TEXTS:
        st.error(
            f"Please use {config.MAX_INPUT_TEXTS} texts or fewer. You entered {len(texts)} texts."
        )
        return

    # Validate token limits
    analyzer = get_analyzer(model_name)
    token_stats = update_token_stats(analyzer, texts, current_hash)
    token_error = build_token_limit_error(token_stats)
    if token_error:
        st.error(token_error)
        return

    # Skip if analysis is already up to date
    existing_hash = st.session_state.get(STATE_ANALYSIS_HASH)
    if existing_hash == current_hash:
        st.info("Analysis is already up to date.")
        return

    # Run analysis
    with st.spinner("Analyzing..."):
        try:
            run_full_analysis(model_name, texts, num_clusters)
            st.session_state[STATE_ANALYSIS_HASH] = current_hash
            st.session_state[STATE_CLUSTER_COUNT] = num_clusters
        except Exception as error:
            st.error(f"Analysis failed: {error}")


# ============================================================================
# Tab Renderers
# ============================================================================


def format_metric(value, fmt=".3f"):
    """Format a metric value for display, returning 'N/A' if None."""
    return f"{value:{fmt}}" if value is not None else "N/A"


def render_network_tab(analyzer, visualizer, threshold):
    network_figure, network_stats = visualizer.create_similarity_network(threshold=threshold)
    st.plotly_chart(
        network_figure, use_container_width=True, config=config.PLOTLY_CONFIG
    )

    if analyzer.similarity_matrix is None or len(analyzer.sentences) == 0:
        return

    pairwise = upper_triangle(analyzer.similarity_matrix)
    if pairwise.size == 0:
        avg_sim, min_sim, max_sim = 0.0, 0.0, 0.0
    else:
        avg_sim = float(pairwise.mean())
        min_sim = float(pairwise.min())
        max_sim = float(pairwise.max())

    # Row 1: Average and Maximum similarity
    column_average, column_maximum = st.columns(2)
    with column_average:
        average_help = (
            "Average cosine similarity across all text pairs. "
            "Values range from -1 (opposite) to 1 (very similar)."
        )
        st.metric(
            "Average pair cosine similarity",
            format_metric(avg_sim),
            help=average_help,
        )
    with column_maximum:
        maximum_help = (
            "Highest cosine similarity among all text pairs. "
            "Values near 1 indicate nearly identical embeddings."
        )
        st.metric(
            "Maximum pair cosine similarity",
            format_metric(max_sim),
            help=maximum_help,
        )

    # Row 2: Minimum similarity and Average degree
    column_minimum, column_degree = st.columns(2)
    with column_minimum:
        minimum_help = (
            "Lowest cosine similarity among all text pairs. "
            "Values near -1 indicate highly dissimilar embeddings."
        )
        st.metric(
            "Minimum pair cosine similarity",
            format_metric(min_sim),
            help=minimum_help,
        )
    with column_degree:
        degree_help = (
            "Average number of connections per node "
            "in the similarity network (edges above the threshold)."
        )
        st.metric(
            "Average degree",
            format_metric(network_stats["avg_degree"], ".2f"),
            help=degree_help,
        )

    # Row 3: Network density and Top degree nodes
    column_density, column_top_nodes = st.columns(2)
    with column_density:
        density_help = (
            "Ratio of actual connections to all possible "
            "connections in the similarity network."
        )
        st.metric(
            "Network density",
            format_metric(network_stats["density"]),
            help=density_help,
        )
    with column_top_nodes:
        top_nodes = network_stats["top_nodes"]
        if top_nodes:
            top_nodes_display = ", ".join(
                [f"{node} ({degree})" for node, degree in top_nodes]
            )
        else:
            top_nodes_display = "None"
        top_help = (
            "Nodes (text indices) with the highest number of connections; "
            "displayed as Text ID (degree)."
        )
        st.metric(
            "Top degree nodes",
            top_nodes_display,
            help=top_help,
        )


def render_clusters_tab(analyzer, visualizer):
    cluster_figure = visualizer.create_cluster_visualization()
    st.plotly_chart(
        cluster_figure, use_container_width=True, config=config.PLOTLY_CONFIG
    )

    if analyzer.cluster_labels is None:
        return

    cluster_indices = cluster_partitions(analyzer.cluster_labels)
    cluster_sizes = [len(members) for members in cluster_indices]

    # Overview section
    st.markdown(
        "<h3 style='text-align: center;'><i class='fa-solid fa-eye'></i> Overview</h3>",
        unsafe_allow_html=True,
    )
    column_count, column_min_size, column_max_size = st.columns(3)

    with column_count:
        st.metric("Number of clusters:", len(cluster_sizes))
    with column_min_size:
        minimum_size = min(cluster_sizes) if cluster_sizes else 0
        st.metric("Smallest cluster size:", minimum_size)
    with column_max_size:
        maximum_size = max(cluster_sizes) if cluster_sizes else 0
        st.metric("Largest cluster size:", maximum_size)

    # Metrics section
    st.markdown(
        "<h3 style='text-align: center;'><i class='fa-solid fa-chart-line'></i> Metrics</h3>",
        unsafe_allow_html=True,
    )
    column_left_metrics, column_right_metrics = st.columns(2)

    with column_left_metrics:
        silhouette_help = (
            "Measures how well each text fits within its cluster "
            "compared to other clusters; scores range from -1 to 1."
        )
        st.metric(
            "Silhouette score (cosine distance):",
            format_metric(analyzer.silhouette),
            help=silhouette_help,
        )

        calinski_help = (
            "Calinski-Harabasz Index (Variance Ratio Criterion). "
            "Measures cluster separation quality; higher values indicate "
            "better-defined, well-separated clusters."
        )
        st.metric(
            "Calinski-Harabasz Index:",
            format_metric(analyzer.calinski_harabasz, ".1f"),
            help=calinski_help,
        )

    with column_right_metrics:
        within_help = "Average cosine similarity of text pairs within the same cluster."
        st.metric(
            "Average within-cluster cosine similarity:",
            format_metric(analyzer.avg_within_cluster),
            help=within_help,
        )

        between_help = (
            "Average cosine similarity of text pairs belonging to different clusters."
        )
        st.metric(
            "Average between-cluster cosine similarity:",
            format_metric(analyzer.avg_between_clusters),
            help=between_help,
        )

    # Texts by cluster
    st.markdown(
        "<h3 style='text-align: center;'><i class='fa-solid fa-layer-group'></i> Texts by Cluster</h3>",
        unsafe_allow_html=True,
    )
    for cluster_id, text_indices in enumerate(cluster_indices):
        with st.expander(f"Cluster {cluster_id + 1} ({len(text_indices)} texts)"):
            for text_index in text_indices:
                st.write(f"**Text {text_index + 1}:** {analyzer.sentences[text_index]}")


def render_pairs_tab(visualizer):
    pairs_figure = visualizer.create_top_pairs_chart(num_pairs=10)
    st.plotly_chart(pairs_figure, use_container_width=True, config=config.PLOTLY_CONFIG)


# ============================================================================
# Main
# ============================================================================


def main():
    st.markdown(styles.FONT_AWESOME_CDN, unsafe_allow_html=True)
    st.markdown(styles.CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        "<h1><i class='fa-solid fa-magnifying-glass-chart'></i> Similyzer</h1>",
        unsafe_allow_html=True,
    )
    init_state()

    # Sidebar controls
    num_clusters, threshold = render_sidebar_controls()
    model_name = config.MODEL_NAME

    # Header row: input actions and stats title
    header_left, header_right = st.columns([3, 1])
    with header_left:
        render_input_actions(SAMPLE_TEXTS)
    with header_right:
        st.markdown(
            "<div class='section-title'><i class='fa-solid fa-chart-simple'></i> Input Statistics</div>",
            unsafe_allow_html=True,
        )

    # Body row: text area and stats panel
    body_left, body_right = st.columns([3, 1])
    with body_left:
        raw_input = render_text_area()

    texts = parse_texts(raw_input)
    current_hash = compute_content_hash(model_name, texts)

    with body_right:
        stats_placeholder = st.empty()

    # Analyze button
    analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

    handle_analyze_click(
        analyze_clicked=analyze_clicked,
        texts=texts,
        model_name=model_name,
        num_clusters=num_clusters,
        current_hash=current_hash,
    )

    with stats_placeholder.container():
        render_stats_panel(texts, current_hash)

    # Check if analysis results are available
    analyzer = st.session_state.get(STATE_ANALYZER)
    analysis_hash = st.session_state.get(STATE_ANALYSIS_HASH)

    if analyzer is None or analysis_hash is None:
        return

    # Check if results are stale
    if current_hash != analysis_hash:
        if not texts:
            st.info("Input is empty. Enter texts and click Analyze to see results.")
        else:
            st.warning("Input or model changed. Click Analyze to refresh results.")
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
    st.markdown(
        "<h2 style='text-align:center;'><i class='fa-solid fa-chart-pie'></i> Analysis Results</h2>",
        unsafe_allow_html=True,
    )

    network_tab, clusters_tab, pairs_tab = st.tabs(TAB_LABELS)

    with network_tab:
        render_network_tab(analyzer, visualizer, threshold)

    with clusters_tab:
        render_clusters_tab(analyzer, visualizer)

    with pairs_tab:
        render_pairs_tab(visualizer)


if __name__ == "__main__":
    main()
