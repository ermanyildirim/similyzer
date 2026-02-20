import streamlit as st

import config
import state


# ============================================================================
# Constants
# ============================================================================

_BUTTON_ACTIONS = {
    "button_load_sample": {
        "label": "Load sample",
        "help": "Inserting the sample input",
    },
    "button_clear": {
        "label": "Clear",
        "help": "Clearing the input",
    },
}


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

    return (num_clusters, threshold)


# ============================================================================
# Input Actions
# ============================================================================


def render_input_actions(sample_text):
    _, load_col, clear_col = st.columns([8, 3, 3])

    with load_col:
        load_clicked = st.button(
            type="secondary",
            use_container_width=True,
            key="button_load_sample",
            **_BUTTON_ACTIONS["button_load_sample"],
        )
    with clear_col:
        clear_clicked = st.button(
            type="secondary",
            use_container_width=True,
            key="button_clear",
            **_BUTTON_ACTIONS["button_clear"],
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
    note_parts = []

    if model_max > 0:
        note_parts.append(f"Limit: {model_max}")

    max_line_indices = token_stats["max_line_indices"]
    if max_line_indices:
        line_numbers = [str(i + 1) for i in max_line_indices]
        shown = line_numbers[:config.MAX_SHOWN_LINES]

        if len(line_numbers) > config.MAX_SHOWN_LINES:
            suffix = f" (+{len(line_numbers) - config.MAX_SHOWN_LINES} more)"
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
    st.metric("Number of texts:", len(texts))
    word_count = sum(len(t.split()) for t in texts) if texts else 0
    st.metric("Total words:", word_count)

    token_stats = st.session_state.get(state.STATE_TOKEN_STATS)
    token_hash = st.session_state.get(state.STATE_TOKEN_STATS_HASH)

    if not (token_stats and token_hash == current_hash):
        st.metric("Maximum tokens per line:", "—")
        st.markdown("<div class='token-line-note'>&nbsp;</div>", unsafe_allow_html=True)
        return

    max_tokens = token_stats["max_tokens"]
    model_max = token_stats["model_max"]

    if 0 < model_max < max_tokens:
        display_value = f"{model_max}+"
    else:
        display_value = max_tokens
    st.metric("Maximum tokens per line:", display_value)

    note = _build_token_note(token_stats, model_max)
    st.markdown(f"<div class='token-line-note'>{note}</div>", unsafe_allow_html=True)