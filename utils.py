import hashlib
import html
import textwrap
import numpy as np

from sklearn.preprocessing import StandardScaler




# ============================================================================
# Text Processing
# ============================================================================


def normalize_whitespace(text):
    return " ".join(text.split())


def parse_texts(text):
    if not text:
        return []
    return [normalize_whitespace(line) for line in text.splitlines() if line.strip()]


# ============================================================================
# Hashing
# ============================================================================


def compute_content_hash(model_name, sentences):
    """Compute a hash for cache invalidation. Returns None for empty input."""
    if not sentences:
        return None
    content = f"{model_name}|" + "\n".join(sentences)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ============================================================================
# Hover Formatting
# ============================================================================


def format_sentence_for_hover(sentence, max_width=70, max_lines=12, max_chars=1400):
    """Format user text for Plotly hover labels safely."""
    if not sentence:
        return ""

    text = " ".join(sentence.split())

    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."

    lines = textwrap.wrap(
        text,
        width=max_width,
        break_long_words=False,
        break_on_hyphens=False,
    )

    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines] + ["..."]

    safe_lines = [html.escape(line, quote=True) for line in lines]
    return "<br>".join(safe_lines)


# ============================================================================
# Matrix Utilities
# ============================================================================


def upper_triangle(matrix):
    """Return upper-triangular values excluding the diagonal."""
    return matrix[np.triu_indices_from(matrix, k=1)]


def normalize_coordinates(coordinates):
    return StandardScaler().fit_transform(coordinates).astype(np.float32)


def cluster_partitions(labels):
    if labels is None:
        return []
    return [np.flatnonzero(labels == cluster_id).tolist() for cluster_id in np.unique(labels)]
