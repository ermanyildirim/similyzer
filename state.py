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
    """Compute raw token statistics."""
    model_max = model.max_seq_length or 0
    token_lengths = model.tokenize(texts)["attention_mask"].sum(axis=1).numpy()
    max_tokens = int(token_lengths.max()) if token_lengths.size else 0

    return {
        "max_tokens": max_tokens,
        "model_max": model_max,
        "too_long_lines": np.flatnonzero((model_max > 0) & (token_lengths > model_max)).tolist(),
        "max_line_indices": np.flatnonzero(token_lengths == max_tokens).tolist() if token_lengths.size else [],
    }


def update_token_stats(analyzer, texts, current_hash):
    """Compute and cache token statistics."""
    cached_hash = st.session_state.get(STATE_TOKEN_STATS_HASH)
    cached_stats = st.session_state.get(STATE_TOKEN_STATS)

    if cached_hash == current_hash and cached_stats is not None:
        return cached_stats

    stats = _compute_token_stats(analyzer.model, texts)
    stats["too_long"] = len(stats["too_long_lines"])

    st.session_state[STATE_TOKEN_STATS] = stats
    st.session_state[STATE_TOKEN_STATS_HASH] = current_hash
    return stats


def build_token_limit_error(token_stats):
    """Build an error message when inputs exceed the model token limit."""
    if not token_stats:
        return None

    model_max = token_stats["model_max"]
    overlimit_count = token_stats["too_long"]
    overlimit_indices = token_stats["too_long_lines"]

    if model_max <= 0 or overlimit_count <= 0:
        return None

    shown = [str(i + 1) for i in overlimit_indices[:config.MAX_SHOWN_LINES]]
    suffix = ", ..." if len(overlimit_indices) > config.MAX_SHOWN_LINES else ""

    if overlimit_count == 1:
        return (
            f"Input is too long: Line {shown[0]} exceeds the model limit ({model_max} tokens). "
            f"Please shorten that line and try again."
        )

    return (
        f"Input is too long: {overlimit_count} lines exceed the model limit ({model_max} tokens). "
        f"Please shorten those lines and try again. "
        f"Over-limit lines: {', '.join(shown)}{suffix}"
    )