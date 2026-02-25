import streamlit as st
from itertools import batched

import config
import state


# ============================================================================
# Display Helpers
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


def format_metric(value, fmt=".3f"):
    """Format a metric value for display, returning 'N/A' if None."""
    if value is None:
        return "N/A"
    if isinstance(value, str) or not fmt:
        return str(value)
    return f"{value:{fmt}}"


def render_metrics_grid(descriptions, values, columns=2):
    for chunk in batched(zip(descriptions, values), columns):
        cols = st.columns(columns)
        for col, (desc, value) in zip(cols, chunk):
            with col:
                st.metric(desc.label, format_metric(value, desc.fmt), help=desc.help)


# ============================================================================
# Sidebar
# ============================================================================


def render_sidebar_controls():
    """Render sidebar controls and return user settings."""
    with st.sidebar:
        st.markdown(
            "<div class='sidebar-center-title'>"
            "<i class='fa-solid fa-circle-nodes'></i> Clustering</div>",
            unsafe_allow_html=True,
        )

        auto_cluster = st.checkbox("Auto-detect clusters", value=True)
        n_clusters = None
        if not auto_cluster:
            n_clusters = st.slider(
                "Number of clusters",
                min_value=2,
                max_value=10,
                value=3,
                step=1,
            )

        threshold = st.slider(
            "Network threshold (cosine similarity)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key="threshold",
        )

    return (n_clusters, threshold)


# ============================================================================
# Input Actions
# ============================================================================


def render_input_actions(sample_text):
    _, load_col, clear_col = st.columns([8, 3, 3])

    with load_col:
        load_clicked = st.button(
            "Load sample",
            type="secondary",
            use_container_width=True,
            key="button_load_sample",
            help="Inserting the sample input",
        )
    with clear_col:
        clear_clicked = st.button(
            "Clear",
            type="secondary",
            use_container_width=True,
            key="button_clear",
            help="Clearing the input",
        )

    if load_clicked:
        st.session_state[state.STATE_INPUT_TEXT] = sample_text
        state.invalidate_analysis_state()

    if clear_clicked:
        st.session_state[state.STATE_INPUT_TEXT] = ""
        state.invalidate_analysis_state()


# ============================================================================
# Text Area
# ============================================================================


def render_text_area():
    placeholder = (
        "Type your texts here. Each non-empty line is treated as one text input."
    )
    return st.text_area(
        label="Input texts",
        key=state.STATE_INPUT_TEXT,
        label_visibility="collapsed",
        height=350,
        placeholder=placeholder,
        max_chars=20000,
    )


# ============================================================================
# Statistics Panel
# ============================================================================


def _build_token_note(token_stats, model_max):
    parts = []

    if model_max > 0:
        parts.append(f"Limit: {model_max}")

    indices = token_stats["max_line_indices"]
    if indices:
        shown = [str(i + 1) for i in indices[:config.MAX_SHOWN_LINES]]
        extra = f" (+{len(indices) - config.MAX_SHOWN_LINES} more)" if len(indices) > config.MAX_SHOWN_LINES else ""
        label = "Maximum token line" if len(indices) == 1 else "Maximum token lines"
        parts.append(f"{label}: {', '.join(shown)}{extra}")

    return " • ".join(parts) or "&nbsp;"


def render_stats_panel(texts, current_hash):
    st.metric("Number of texts:", len(texts))
    word_count = sum(len(t.split()) for t in texts)
    st.metric("Total words:", word_count)

    token_stats = st.session_state.get(state.STATE_TOKEN_STATS)
    token_hash = st.session_state.get(state.STATE_TOKEN_STATS_HASH)

    if not (token_stats and token_hash == current_hash):
        st.metric("Maximum tokens per line:", "—")
        st.markdown("<div class='token-line-note'>&nbsp;</div>", unsafe_allow_html=True)
        return

    max_tokens = token_stats["max_tokens"]
    model_max = token_stats["model_max"]

    display_value = f"{model_max}+" if 0 < model_max < max_tokens else max_tokens
    st.metric("Maximum tokens per line:", display_value)

    note = _build_token_note(token_stats, model_max)
    st.markdown(f"<div class='token-line-note'>{note}</div>", unsafe_allow_html=True)