import streamlit as st

import config
from state import (
    STATE_INPUT_TEXT,
    STATE_TOKEN_STATS,
    STATE_TOKEN_STATS_HASH,
    reset_analysis,
)

def render_sidebar_controls():
    """Render sidebar controls and return user settings."""
    with st.sidebar:
        st.markdown(
            "<div class='sidebar-center-title'>"
            "<i class='fa-solid fa-circle-nodes'></i> Clustering</div>",
            unsafe_allow_html=True,
        )

        auto_cluster = st.checkbox("Auto-detect clusters", value=True)
        num_clusters = None
        if not auto_cluster:
            num_clusters = st.slider(
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
        height=350,
        placeholder=placeholder,
        max_chars=20000,
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
