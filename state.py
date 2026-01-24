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


###### Session State Management ######


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


def clear_analyzer_inputs():
    analyzer = st.session_state.get(STATE_ANALYZER)
    if analyzer is not None:
        analyzer.add_sentences([])


def reset_analysis():
    invalidate_analysis_state()
    clear_analyzer_inputs()


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


###### Token Statistics ######


def count_tokens(tokenizer, text, cap=None):
    """Count tokens for a single text with optional truncation."""
    if tokenizer is None:
        return 0

    try:
        if cap and cap > 0:
            token_ids = tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=cap + 1,
            )
        else:
            token_ids = tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=False,
            )
        return len(token_ids)
    except (TypeError, AttributeError):
        try:
            return len(tokenizer.encode(text))
        except (TypeError, AttributeError):
            return 0


def compute_token_stats(analyzer, texts):
    # Validate model and tokenizer
    model = getattr(analyzer, "model", None)
    if model is None:
        return None

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return None

    # Setup token limits
    model_max_tokens = int(getattr(model, "max_seq_length", 0) or 0)
    token_cap = model_max_tokens

    # Count tokens for each text
    token_lengths = []
    overlimit_indices = []

    for i, raw_text in enumerate(texts):
        processed = normalize_whitespace(raw_text)
        count = count_tokens(tokenizer, processed, cap=token_cap)
        token_lengths.append(count)

        if 0 < model_max_tokens < count:
            overlimit_indices.append(i)

    # Compute max token info
    if not token_lengths:
        max_tokens = 0
        max_indices = []
    else:
        max_tokens = max(token_lengths)
        max_indices = [
            i for i, count in enumerate(token_lengths) if count == max_tokens
        ]

    return {
        "max_tokens": max_tokens,
        "model_max": model_max_tokens,
        "too_long": len(overlimit_indices),
        "too_long_lines": overlimit_indices,
        "max_line_indices": max_indices,
    }


def update_token_stats(analyzer, texts, current_hash):
    """Update token statistics if input hash has changed."""
    existing_hash = st.session_state.get(STATE_TOKEN_STATS_HASH)
    existing_stats = st.session_state.get(STATE_TOKEN_STATS)

    if existing_hash == current_hash and existing_stats is not None:
        return existing_stats

    stats = compute_token_stats(analyzer, texts)
    st.session_state[STATE_TOKEN_STATS] = stats
    st.session_state[STATE_TOKEN_STATS_HASH] = current_hash
    return stats


def build_token_limit_error(token_stats):
    """Build an error message when inputs exceed the model token limit."""
    if not token_stats:
        return None

    model_max = int(token_stats.get("model_max", 0) or 0)
    overlimit_count = int(token_stats.get("too_long", 0) or 0)
    overlimit_indices = token_stats.get("too_long_lines") or []

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
