from dataclasses import dataclass

import numpy as np
import networkx as nx
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

import config
from utils import upper_triangle


@st.cache_resource(show_spinner=False)
def load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


# ============================================================================
# Compute Cache
# ============================================================================


@dataclass
class _ComputeCache:
    embeddings: np.ndarray | None = None
    similarity_matrix: np.ndarray | None = None
    pca_coords: np.ndarray | None = None
    network_coords: np.ndarray | None = None
    top_pairs: list | None = None
    cluster_key: int | str | None = None


class SentenceAnalyzer:
    """Compute embeddings, similarity, PCA reduction and clustering for sentences."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = load_model(model_name)
        self.sentences: list[str] = []
        self._reset()

    # ====================================================================
    # Public API
    # ====================================================================

    def set_sentences(self, sentences: list[str]) -> None:
        """Set sentences and reset all cached results."""
        self.sentences = list(sentences)
        self._reset()

    def get_embeddings(self) -> np.ndarray:
        """Compute sentence embeddings (cached)."""
        if self._cache.embeddings is None:
            if not self.sentences:
                raise ValueError("No sentences provided")
            self._cache.embeddings = self.model.encode(
                self.sentences, show_progress_bar=False, convert_to_numpy=True,
            )
        return self._cache.embeddings

    def get_similarity_matrix(self) -> np.ndarray:
        """Compute cosine similarity matrix (cached)."""
        if self._cache.similarity_matrix is None:
            self._cache.similarity_matrix = cosine_similarity(self.get_embeddings())
        return self._cache.similarity_matrix

    def get_pca_coordinates(self) -> np.ndarray:
        """Reduce embeddings to 2D with PCA (cached)."""
        if self._cache.pca_coords is None:
            embeddings = self.get_embeddings()
            n = len(self.sentences)
            if n < 2:
                self._cache.pca_coords = np.zeros((1, 2), dtype=np.float32)
            else:
                components = min(2, embeddings.shape[1], n)
                self._cache.pca_coords = PCA(
                    n_components=components, random_state=config.RANDOM_SEED,
                ).fit_transform(embeddings)
        return self._cache.pca_coords

    def get_network_coordinates(self) -> np.ndarray:
        """Compute force-directed layout from full similarity matrix (cached)."""
        if self._cache.network_coords is None:
            similarity = self.get_similarity_matrix()
            n = similarity.shape[0]
            if n < 2:
                self._cache.network_coords = np.zeros((max(n, 1), 2), dtype=np.float32)
            else:
                weight_matrix = similarity.copy()
                np.fill_diagonal(weight_matrix, 0.0)
                graph = nx.from_numpy_array(weight_matrix)
                pos = nx.spring_layout(graph, weight="weight", seed=config.RANDOM_SEED)
                coords = np.array(list(pos.values()), dtype=np.float32)
                self._cache.network_coords = StandardScaler().fit_transform(coords)
        return self._cache.network_coords

    def get_cluster_labels(self, n_clusters: int | None = None) -> np.ndarray:
        """Cluster sentences. Auto-detect count when n_clusters is None."""
        embeddings = self.get_embeddings()
        n = embeddings.shape[0]

        if n < 2:
            self._cache.cluster_key = None
            self.cluster_labels = np.zeros(n, dtype=np.int32)
        elif n_clusters is None:
            if self._cache.cluster_key == "auto":
                return self.cluster_labels
            self.cluster_labels = self._auto_cluster(embeddings)
            self._cache.cluster_key = "auto"
        else:
            k = int(np.clip(n_clusters, 2, n))
            if self._cache.cluster_key == k:
                return self.cluster_labels
            self.cluster_labels = self._best_labels_for_k(embeddings, k)[0]
            self._cache.cluster_key = k

        self._compute_cluster_metrics()
        return self.cluster_labels

    def get_top_pairs(self) -> list:
        """Return sentence pairs sorted by similarity (cached)."""
        if self._cache.top_pairs is None:
            self._cache.top_pairs = util.paraphrase_mining_embeddings(self.get_embeddings())
        return self._cache.top_pairs

    def get_pairwise_stats(self) -> tuple[float, float, float]:
        """Return (mean, min, max) of upper-triangle similarities."""
        sims = upper_triangle(self.get_similarity_matrix())
        if sims.size == 0:
            return 0.0, 0.0, 0.0
        return float(sims.mean()), float(sims.min()), float(sims.max())

    # ====================================================================
    # Clustering
    # ====================================================================

    def _fit_kmeans(self, embeddings: np.ndarray, k: int) -> np.ndarray:
        return KMeans(
            n_clusters=k,
            random_state=config.RANDOM_SEED,
            n_init=config.KMEANS_N_INIT,
        ).fit_predict(embeddings)

    def _fit_agglomerative(self, embeddings: np.ndarray, k: int) -> np.ndarray:
        return AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average",
        ).fit_predict(embeddings)

    def _best_labels_for_k(
        self, embeddings: np.ndarray, k: int,
    ) -> tuple[np.ndarray, float | None]:
        """Compare KMeans and Agglomerative for a given k, return (labels, score)."""
        candidates = [
            self._fit_kmeans(embeddings, k),
            self._fit_agglomerative(embeddings, k),
        ]
        scored = [
            (labels, self._silhouette(embeddings, labels))
            for labels in candidates
        ]
        valid = [(labels, score) for labels, score in scored if score is not None]

        if not valid:
            return scored[0]
        return max(valid, key=lambda x: x[1])

    def _auto_cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Select best clustering by highest silhouette score across k values."""
        n = embeddings.shape[0]
        max_k = min(config.AUTO_CLUSTER_MAX_K, n - 1)

        if max_k < 2:
            return np.zeros(n, dtype=np.int32)

        best_labels, best_score = None, -np.inf

        for k in range(2, max_k + 1):
            labels, score = self._best_labels_for_k(embeddings, k)
            if score is not None and score > best_score:
                best_labels, best_score = labels, score

        return best_labels if best_labels is not None else np.zeros(n, dtype=np.int32)

    @staticmethod
    def _silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float | None:
        """Return silhouette score, or None on failure."""
        if np.unique(labels).size < 2:
            return None
        try:
            return float(silhouette_score(embeddings, labels, metric="cosine"))
        except ValueError:
            return None

    # ====================================================================
    # Metrics
    # ====================================================================

    def _compute_cluster_metrics(self) -> None:
        """Compute within/between cluster similarities, silhouette and CH index."""
        similarity = self.get_similarity_matrix()
        labels = self.cluster_labels
        n = labels.size
        n_clusters = np.unique(labels).size

        self.avg_within_cluster = None
        self.avg_between_clusters = None
        self.silhouette = None
        self.calinski_harabasz = None

        if n < 2:
            return

        if n_clusters <= 1:
            sims = upper_triangle(similarity)
            if sims.size:
                self.avg_within_cluster = float(sims.mean())
            return

        row_idx, col_idx = np.triu_indices(n, k=1)
        pair_sims = similarity[row_idx, col_idx]
        same_cluster = labels[row_idx] == labels[col_idx]

        within = pair_sims[same_cluster]
        between = pair_sims[~same_cluster]

        if within.size:
            self.avg_within_cluster = float(within.mean())
        if between.size:
            self.avg_between_clusters = float(between.mean())

        if n <= n_clusters:
            return

        embeddings = self.get_embeddings()
        self.silhouette = self._silhouette(embeddings, labels)

        try:
            self.calinski_harabasz = float(calinski_harabasz_score(embeddings, labels))
        except ValueError:
            pass

    # ====================================================================
    # Internals
    # ====================================================================

    def _reset(self) -> None:
        self._cache = _ComputeCache()
        self.cluster_labels: np.ndarray | None = None
        self.silhouette: float | None = None
        self.calinski_harabasz: float | None = None
        self.avg_within_cluster: float | None = None
        self.avg_between_clusters: float | None = None