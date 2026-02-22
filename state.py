import streamlit as st
import numpy as np

import config
from analyzer import SentenceAnalyzer

# ============================================================================
# Session State Keys & Defaults
# ============================================================================

STATE_ANALYZER = "analyzer"
STATE_ANALYSIS_HASH = "analysis_hash"
STATE_CLUSTER_COUNT = "cluster_count"
STATE_INPUT_TEXT = "input_text"
STATE_TOKEN_STATS = "token_stats"
STATE_TOKEN_STATS_HASH = "token_stats_hash"

_STATE_DEFAULTS = {
    STATE_ANALYZER: None,
    STATE_ANALYSIS_HASH: None,
    STATE_CLUSTER_COUNT: None,
    STATE_INPUT_TEXT: "",
    STATE_TOKEN_STATS: None,
    STATE_TOKEN_STATS_HASH: None,
}

_INVALIDATION_KEYS = [
    STATE_ANALYSIS_HASH,
    STATE_CLUSTER_COUNT,
    STATE_TOKEN_STATS,
    STATE_TOKEN_STATS_HASH,
]


# ============================================================================
# Session State Management
# ============================================================================


def init_state():
    for key, default in _STATE_DEFAULTS.items():
        st.session_state.setdefault(key, default)


def invalidate_analysis_state():
    for key in _INVALIDATION_KEYS:
        st.session_state[key] = None


def get_analyzer(model_name):
    """Get or create a cached analyzer instance for the model."""
    analyzer = st.session_state.get(STATE_ANALYZER)
    current_model = getattr(analyzer, "model_name", None)

    if analyzer is None or current_model != model_name:
        analyzer = SentenceAnalyzer(model_name)
        st.session_state[STATE_ANALYZER] = analyzer
        invalidate_analysis_state()

    return analyzer


# ============================================================================
# Token Statistics
# ============================================================================


def _compute_token_stats(model, texts):
    model_max = model.max_seq_length or 0
    token_lengths = model.tokenize(texts)["attention_mask"].sum(axis=1).numpy()

    if not token_lengths.size:
        return {"max_tokens": 0, "model_max": model_max, "too_long_lines": [],
                "too_long": 0, "max_line_indices": []}

    max_tokens = int(token_lengths.max())
    too_long = np.flatnonzero((model_max > 0) & (token_lengths > model_max))
    return {
        "max_tokens": max_tokens,
        "model_max": model_max,
        "too_long_lines": too_long.tolist(),
        "too_long": len(too_long),
        "max_line_indices": np.flatnonzero(token_lengths == max_tokens).tolist(),
    }


def update_token_stats(analyzer, texts, current_hash):
    """Compute and cache token statistics."""
    cached_hash = st.session_state.get(STATE_TOKEN_STATS_HASH)
    cached_stats = st.session_state.get(STATE_TOKEN_STATS)

    if cached_hash == current_hash and cached_stats is not None:
        return cached_stats

    stats = _compute_token_stats(analyzer.model, texts)

    st.session_state[STATE_TOKEN_STATS] = stats
    st.session_state[STATE_TOKEN_STATS_HASH] = current_hash
    return stats


def build_token_limit_error(token_stats):
    if not token_stats or token_stats["model_max"] <= 0 or not token_stats["too_long_lines"]:
        return None

    model_max = token_stats["model_max"]
    indices = token_stats["too_long_lines"]
    shown = [str(i + 1) for i in indices[:config.MAX_SHOWN_LINES]]
    suffix = ", ..." if len(indices) > config.MAX_SHOWN_LINES else ""

    if len(indices) == 1:
        return (f"Input is too long: Line {shown[0]} exceeds the model limit "
                f"({model_max} tokens). Please shorten that line and try again.")

    return (f"Input is too long: {len(indices)} lines exceed the model limit "
            f"({model_max} tokens). Please shorten those lines and try again. "
            f"Over-limit lines: {', '.join(shown)}{suffix}")