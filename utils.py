import hashlib
import html
import textwrap
import numpy as np


# ============================================================================
# Text Processing
# ============================================================================


def parse_texts(raw_text: str) -> list[str]:
    """Split text into non-empty lines with normalized whitespace."""
    if not raw_text:
        return []
    return [" ".join(line.split()) for line in raw_text.splitlines() if line.strip()]


def format_sentence_for_hover(
    sentence: str,
    max_width: int = 70,
    max_lines: int = 12,
    max_chars: int = 1400,
) -> str:
    """Format user text for Plotly hover labels safely."""
    if not sentence:
        return ""

    text = " ".join(sentence.split())

    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."

    lines = textwrap.wrap(
        text,
        width=max_width,
        max_lines=max_lines,
        placeholder=" ...",
        break_long_words=False,
        break_on_hyphens=False,
    )

    return "<br>".join(html.escape(line, quote=True) for line in lines)


# ============================================================================
# Hashing
# ============================================================================


def compute_input_hash(model_name: str, sentences: list[str]) -> str | None:
    """Compute a hash for cache invalidation. Returns None for empty input."""
    if not sentences:
        return None
    content = f"{model_name}|" + "\n".join(sentences)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ============================================================================
# Matrix Utilities
# ============================================================================


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Return upper-triangular values excluding the diagonal."""
    return matrix[np.triu_indices_from(matrix, k=1)]


def cluster_partitions(labels: np.ndarray | None) -> list[list[int]]:
    if labels is None:
        return []
    return [np.flatnonzero(labels == cluster_id).tolist() for cluster_id in np.unique(labels)]