import hashlib
import html
import textwrap

import numpy as np

import config


###### Text Processing ######


def normalize_whitespace(text):
    return " ".join(text.split())


def parse_texts(text):
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


###### Hashing ######


def compute_content_hash(model_name, sentences):
    """Compute a hash for cache invalidation. Returns None for empty input."""
    if not sentences:
        return None
    processed = [normalize_whitespace(sentence) for sentence in sentences]
    content = f"{model_name}|" + "\n".join(processed)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


###### Hover Formatting ######


def format_sentence_for_hover(
    sentence,
    max_width=config.HOVER_WIDTH_STANDARD,
    max_lines=config.HOVER_LINES_STANDARD,
    max_chars=config.HOVER_CHARS_STANDARD,
):
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


###### Matrix Utilities ######


def upper_triangle(matrix):
    """Return upper-triangular values excluding the diagonal."""
    if matrix.shape[0] < 2:
        return np.array([], dtype=matrix.dtype)

    row_index, column_index = np.triu_indices(matrix.shape[0], k=1)
    return matrix[row_index, column_index]


def normalize_coordinates(coordinates, epsilon=1e-10):
    """Normalize coordinates to zero mean and unit variance."""
    coordinates = np.asarray(coordinates, dtype=np.float32)
    mean = coordinates.mean(axis=0)
    standard_deviation = coordinates.std(axis=0)
    standard_deviation = np.where(standard_deviation < epsilon, 1.0, standard_deviation)
    return (coordinates - mean) / standard_deviation



def cluster_partitions(cluster_labels):
    """Group text indices by cluster label."""
    if cluster_labels is None:
        return []

    cluster_map = {}
    for text_index, label in enumerate(cluster_labels):
        cluster_id = int(label)
        cluster_map.setdefault(cluster_id, []).append(text_index)

    return [cluster_map[cid] for cid in sorted(cluster_map)]
