import streamlit as st

import config
from analyzer import SentenceAnalyzer
from utils import normalize_whitespace

# Session state keys
STATE_ANALYZER = "analyzer"
STATE_ANALYSIS_HASH = "analysis_hash"
STATE_CLUSTER_COUNT = "cluster_count"
STATE_INPUT_TEXT = "input_text"
STATE_TOKEN_STATS = "token_stats"
STATE_TOKEN_STATS_HASH = "token_stats_hash"


# ============================================================================
# Session State Management
# ============================================================================


def init_state():
    st.session_state.setdefault(STATE_ANALYZER, None)
    st.session_state.setdefault(STATE_ANALYSIS_HASH, None)
    st.session_state.setdefault(STATE_CLUSTER_COUNT, None)
    st.session_state.setdefault(STATE_INPUT_TEXT, "")
    st.session_state.setdefault(STATE_TOKEN_STATS, None)
    st.session_state.setdefault(STATE_TOKEN_STATS_HASH, None)


def invalidate_analysis_state():
    st.session_state[STATE_ANALYSIS_HASH] = None
    st.session_state[STATE_CLUSTER_COUNT] = None
    st.session_state[STATE_TOKEN_STATS] = None
    st.session_state[STATE_TOKEN_STATS_HASH] = None


def reset_analysis():
    invalidate_analysis_state()


def get_analyzer(model_name):
    """Get or create a cached analyzer instance for the model."""
    analyzer = st.session_state.get(STATE_ANALYZER)
    current_model = getattr(analyzer, "model_name", None)
    is_new_model = analyzer is None or current_model != model_name

    if is_new_model:
        analyzer = SentenceAnalyzer(model_name)
        st.session_state[STATE_ANALYZER] = analyzer
        invalidate_analysis_state()

    return analyzer


# ============================================================================
# Token Statistics
# ============================================================================


def update_token_stats(analyzer, texts, current_hash):
    """Compute and cache token statistics. Returns stats dict or None."""
    existing_hash = st.session_state.get(STATE_TOKEN_STATS_HASH)
    existing_stats = st.session_state.get(STATE_TOKEN_STATS)

    if existing_hash == current_hash and existing_stats is not None:
        return existing_stats

    model = analyzer.model
    tokenizer = model.tokenizer

    model_max = model.max_seq_length or 0

    encode_kwargs = {"add_special_tokens": True}
    if model_max > 0:
        encode_kwargs["truncation"] = True
        encode_kwargs["max_length"] = model_max + 1

    token_lengths = []
    overlimit_indices = []

    for i, raw_text in enumerate(texts):
        processed = normalize_whitespace(raw_text)
        count = len(tokenizer.encode(processed, **encode_kwargs))
        token_lengths.append(count)
        if 0 < model_max < count:
            overlimit_indices.append(i)

    max_tokens = max(token_lengths) if token_lengths else 0
    max_indices = [i for i, c in enumerate(token_lengths) if c == max_tokens] if token_lengths else []

    stats = {
        "max_tokens": max_tokens,
        "model_max": model_max,
        "too_long": len(overlimit_indices),
        "too_long_lines": overlimit_indices,
        "max_line_indices": max_indices,
    }

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
