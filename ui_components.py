import streamlit as st

import config
from state import (
    STATE_INPUT_TEXT,
    STATE_TOKEN_STATS,
    STATE_TOKEN_STATS_HASH,
    reset_analysis,
)

def _render_sidebar_links():
    model_url = f"{config.HF_BASE_URL}/{config.MODEL_ORG}/{config.MODEL_NAME}"
    link_style = "color:rgba(255,255,255,0.95); text-decoration:underline;"
    sidebar_html = f"""
    <div style="text-align:center; font-size:0.95rem; margin-top:0.5rem;">
        <div style="margin-bottom:0.25rem;">
            <span style="font-weight:800;">Model card:</span>
            <a href="{model_url}" target="_blank" style="{link_style}">{config.MODEL_NAME}</a>
        </div>
        <div style="margin-bottom:0.25rem;">
            <span style="font-weight:800;">Framework:</span>
            <a href="{config.SBERT_URL}" target="_blank" style="{link_style}">SentenceTransformers</a>
        </div>
        <div>
            <span style="font-weight:800;">License:</span>
            <a href="{config.LICENSE_URL}" target="_blank" style="{link_style}">Apache‑2.0</a>
        </div>
    </div>
    """
    st.markdown(sidebar_html, unsafe_allow_html=True)


def render_sidebar_controls():
    """Render sidebar controls and return user settings."""
    with st.sidebar:
        st.markdown(
            "<div class='sidebar-center-title'><i class='fa-solid fa-cube'></i> Model</div>",
            unsafe_allow_html=True,
        )
        _render_sidebar_links()

        st.markdown(
            "<div class='sidebar-center-title' style='margin-top:2rem;margin-bottom:1rem;'>"
            "<i class='fa-solid fa-circle-nodes'></i> Clustering</div>",
            unsafe_allow_html=True,
        )

        auto_cluster = st.checkbox("Auto-detect clusters", value=True)
        num_clusters = None
        if not auto_cluster:
            num_clusters = st.slider(
                "Number of clusters",
                min_value=config.MIN_CLUSTERS,
                max_value=config.MAX_CLUSTERS_SLIDER,
                value=config.DEFAULT_CLUSTERS_SLIDER,
                step=1,
            )

        threshold = st.slider(
            "Network threshold (cosine similarity)",
            min_value=config.THRESHOLD_MIN,
            max_value=config.THRESHOLD_MAX,
            value=config.THRESHOLD_DEFAULT,
            step=config.THRESHOLD_STEP,
            key="threshold",
        )

    return (config.MODEL_NAME, num_clusters, threshold)


def render_input_actions(sample_text):
    _, load_col, clear_col = st.columns([8, 3, 3])

    with load_col:
        load_clicked = st.button(
            "Load sample",
            type="secondary",
            use_container_width=True,
            help="Inserting the sample input",
            key="button_load_sample",
        )
    with clear_col:
        clear_clicked = st.button(
            "Clear",
            type="secondary",
            use_container_width=True,
            help="Clearing the input",
            key="button_clear",
        )

    if load_clicked:
        st.session_state[STATE_INPUT_TEXT] = sample_text
        reset_analysis()

    if clear_clicked:
        st.session_state[STATE_INPUT_TEXT] = ""
        reset_analysis()


def render_text_area():
    placeholder = (
        "Type your texts here. Each non-empty line is treated as one text input."
    )
    return st.text_area(
        label="Input texts",
        key=STATE_INPUT_TEXT,
        label_visibility="collapsed",
        height=config.TEXTAREA_HEIGHT,
        placeholder=placeholder,
        max_chars=config.TEXTAREA_MAX_CHARS,
    )


def _build_token_note(token_stats, model_max):
    note_parts = []

    if model_max > 0:
        note_parts.append(f"Limit: {model_max}")

    max_line_indices = token_stats.get("max_line_indices") or []
    if max_line_indices:
        line_numbers = [str(i + 1) for i in max_line_indices]
        shown = line_numbers[:5]

        if len(line_numbers) > 5:
            suffix = f" (+{len(line_numbers) - 5} more)"
        else:
            suffix = ""

        if len(line_numbers) == 1:
            label = "Maximum token line"
        else:
            label = "Maximum token lines"

        note_parts.append(f"{label}: {', '.join(shown)}{suffix}")

    if note_parts:
        return " • ".join(note_parts)
    return "&nbsp;"


def render_stats_panel(texts, current_hash):
    # Basic stats
    st.metric("Number of texts:", len(texts))
    word_count = sum(len(t.split()) for t in texts) if texts else 0
    st.metric("Total words:", word_count)

    # Token stats
    token_stats = st.session_state.get(STATE_TOKEN_STATS)
    token_hash = st.session_state.get(STATE_TOKEN_STATS_HASH)

    if not (token_stats and token_hash == current_hash):
        st.metric("Maximum tokens per line:", "—")
        st.markdown("<div class='token-line-note'>&nbsp;</div>", unsafe_allow_html=True)
        return

    max_tokens = int(token_stats.get("max_tokens", 0) or 0)
    model_max = int(token_stats.get("model_max", 0) or 0)

    if 0 < model_max < max_tokens:
        display_value = f"{model_max}+"
    else:
        display_value = max_tokens
    st.metric("Maximum tokens per line:", display_value)

    # Token note
    note = _build_token_note(token_stats, model_max)
    st.markdown(f"<div class='token-line-note'>{note}</div>", unsafe_allow_html=True)
