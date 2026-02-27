import logging
from dataclasses import dataclass
from enum import Enum, auto

import streamlit as st

import config
import state
import styles
import ui_components as ui
import utils
from analyzer import SentenceAnalyzer
from visualizer import PlotlyVisualizer

st.set_page_config(
    page_title="Similyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger(__name__)


# ============================================================================
# Validation
# ============================================================================


class ValidationStatus(Enum):
    OK = auto()
    UP_TO_DATE = auto()
    ERROR = auto()


@dataclass
class ValidationResult:
    status: ValidationStatus
    message: str = ""


def _validate_input(texts: list[str], input_hash: str | None) -> ValidationResult:
    """Pure validation — no side effects."""
    if not texts:
        return ValidationResult(ValidationStatus.ERROR, "Please enter at least 1 text.")

    if len(texts) > config.MAX_INPUT_TEXTS:
        return ValidationResult(
            ValidationStatus.ERROR,
            f"Please use {config.MAX_INPUT_TEXTS} texts or fewer. "
            f"You entered {len(texts)} texts.",
        )

    if st.session_state.get(state.STATE_ANALYSIS_HASH) == input_hash:
        return ValidationResult(ValidationStatus.UP_TO_DATE, "Analysis is already up to date.")

    return ValidationResult(ValidationStatus.OK)


# ============================================================================
# Analysis Pipeline
# ============================================================================


def run_analysis(texts: list[str], n_clusters: int | None, input_hash: str | None) -> None:
    result = _validate_input(texts, input_hash)

    if result.status == ValidationStatus.UP_TO_DATE:
        st.info(result.message)
        return
    if result.status == ValidationStatus.ERROR:
        st.error(result.message)
        return

    analyzer = state.get_analyzer(config.MODEL_NAME)
    token_stats = state.update_token_stats(analyzer, texts, input_hash)
    token_error = state.build_token_limit_error(token_stats)
    if token_error:
        st.error(token_error)
        return

    with st.spinner("Analyzing..."):
        try:
            analyzer.set_sentences(texts)
            analyzer.get_pca_coordinates()
            analyzer.get_cluster_labels(n_clusters)
            st.session_state[state.STATE_ANALYSIS_HASH] = input_hash
            st.session_state[state.STATE_CLUSTER_COUNT] = n_clusters
        except Exception as e:
            logger.exception("Analysis failed")
            st.error(f"Analysis failed: {e}")


# ============================================================================
# Tab Renderers
# ============================================================================


def render_network_tab(
    analyzer: SentenceAnalyzer,
    visualizer: PlotlyVisualizer,
    threshold: float,
) -> None:
    """Render the similarity network graph and associated metrics."""
    network_figure, network_stats = visualizer.create_similarity_network(threshold=threshold)
    ui.show_chart(network_figure)

    if len(analyzer.sentences) == 0:
        return

    avg_sim, min_sim, max_sim = analyzer.get_pairwise_stats()

    top_nodes = network_stats["top_nodes"]
    top_nodes_display = ", ".join(f"{n} ({d})" for n, d in top_nodes) if top_nodes else "None"

    values = [avg_sim, max_sim, min_sim, network_stats["avg_degree"],
              network_stats["density"], top_nodes_display]
    ui.render_metrics_grid(config.METRIC_DESCRIPTIONS["network"], values)


def render_clusters_tab(analyzer: SentenceAnalyzer, visualizer: PlotlyVisualizer) -> None:
    """Render the cluster visualization, metrics, and per-cluster text lists."""
    ui.show_chart(visualizer.create_cluster_visualization())

    if analyzer.cluster_labels is None:
        return

    cluster_indices = utils.cluster_partitions(analyzer.cluster_labels)
    cluster_sizes = [len(members) for members in cluster_indices]

    ui.fa_heading("eye", "Overview")
    overview_values = [len(cluster_sizes), min(cluster_sizes, default=0),
                       max(cluster_sizes, default=0)]
    ui.render_metrics_grid(config.METRIC_DESCRIPTIONS["cluster_overview"], overview_values, columns=3)

    ui.fa_heading("chart-line", "Metrics")
    detail_values = [analyzer.silhouette, analyzer.avg_within_cluster,
                     analyzer.calinski_harabasz, analyzer.avg_between_clusters]
    ui.render_metrics_grid(config.METRIC_DESCRIPTIONS["cluster_detail"], detail_values)

    ui.fa_heading("layer-group", "Texts by Cluster")
    for cluster_id, text_indices in enumerate(cluster_indices):
        with st.expander(f"Cluster {cluster_id + 1} ({len(text_indices)} texts)"):
            for i in text_indices:
                st.write(f"**Text {i + 1}:** {analyzer.sentences[i]}")


# ============================================================================
# Main
# ============================================================================


def _render_results(
    texts: list[str],
    n_clusters: int | None,
    input_hash: str | None,
    threshold: float,
) -> None:
    """Check analysis state and render result tabs if ready."""
    analyzer = st.session_state.get(state.STATE_ANALYZER)
    analysis_hash = st.session_state.get(state.STATE_ANALYSIS_HASH)

    if analyzer is None or analysis_hash is None:
        return

    if input_hash != analysis_hash:
        msg = ("Input is empty. Enter texts and click Analyze to see results."
               if not texts else
               "Input or model changed. Click Analyze to refresh results.")
        (st.info if not texts else st.warning)(msg)
        return

    if n_clusters != st.session_state.get(state.STATE_CLUSTER_COUNT):
        try:
            analyzer.get_cluster_labels(n_clusters)
            st.session_state[state.STATE_CLUSTER_COUNT] = n_clusters
        except Exception as e:
            logger.exception("Re-clustering failed")
            st.error(f"Re-clustering failed: {e}. Click Analyze to recompute.")
            return

    visualizer = PlotlyVisualizer(analyzer)

    st.markdown("---")
    ui.fa_heading("chart-pie", "Analysis Results", level=2)

    network_tab, clusters_tab, pairs_tab = st.tabs(config.TAB_LABELS)

    with network_tab:
        render_network_tab(analyzer, visualizer, threshold)
    with clusters_tab:
        render_clusters_tab(analyzer, visualizer)
    with pairs_tab:
        ui.show_chart(visualizer.create_top_pairs_chart(n_pairs=10))


def main() -> None:
    st.markdown(styles.FONT_AWESOME_CDN, unsafe_allow_html=True)
    st.markdown(styles.CUSTOM_CSS, unsafe_allow_html=True)
    ui.fa_heading("magnifying-glass-chart", "Similyzer", level=1)
    state.init_state()

    n_clusters, threshold = ui.render_sidebar_controls()

    header_left, header_right = st.columns([3, 1])
    with header_left:
        ui.render_input_actions(config.SAMPLE_TEXTS)
    with header_right:
        st.markdown(
            "<div class='section-title'>"
            "<i class='fa-solid fa-chart-simple'></i> Input Statistics"
            "</div>",
            unsafe_allow_html=True,
        )

    body_left, body_right = st.columns([3, 1])
    with body_left:
        raw_text = ui.render_text_area()

    texts = utils.parse_texts(raw_text)
    input_hash = utils.compute_content_hash(config.MODEL_NAME, texts)

    with body_right:
        stats_placeholder = st.empty()

    analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

    if analyze_clicked:
        run_analysis(texts, n_clusters, input_hash)

    with stats_placeholder.container():
        ui.render_stats_panel(texts, input_hash)

    _render_results(texts, n_clusters, input_hash, threshold)


if __name__ == "__main__":
    main()