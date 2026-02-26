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
def load_model(model_name):
    return SentenceTransformer(model_name)


class SentenceAnalyzer:
    """Compute embeddings, similarity, PCA reduction and clustering for sentences."""

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = load_model(model_name)
        self.sentences = []
        self._reset()

    # ====================================================================
    # Public API
    # ====================================================================

    def add_sentences(self, sentences):
        """Set sentences and reset all cached results."""
        self.sentences = list(sentences)
        self._reset()

    def get_embeddings(self):
        """Compute sentence embeddings (cached)."""
        if self._embeddings is None:
            if not self.sentences:
                raise ValueError("No sentences provided")
            self._embeddings = self.model.encode(
                self.sentences, show_progress_bar=False, convert_to_numpy=True,
            )
        return self._embeddings

    def get_similarity_matrix(self):
        """Compute cosine similarity matrix (cached)."""
        if self._similarity_matrix is None:
            self._similarity_matrix = cosine_similarity(self.get_embeddings())
        return self._similarity_matrix

    def get_pca_coordinates(self):
        """Reduce embeddings to 2D with PCA (cached)."""
        if self._pca_coords is None:
            embeddings = self.get_embeddings()
            n = len(self.sentences)
            if n < 2:
                self._pca_coords = np.zeros((1, 2), dtype=np.float32)
            else:
                components = min(2, embeddings.shape[1], n)
                self._pca_coords = PCA(
                    n_components=components, random_state=config.RANDOM_SEED,
                ).fit_transform(embeddings)
        return self._pca_coords

    def get_network_coordinates(self):
        """Compute force-directed layout from full similarity matrix (cached)."""
        if self._network_coords is None:
            similarity = self.get_similarity_matrix()
            n = similarity.shape[0]
            if n < 2:
                self._network_coords = np.zeros((max(n, 1), 2), dtype=np.float32)
            else:
                adj = similarity.copy()
                np.fill_diagonal(adj, 0.0)
                graph = nx.from_numpy_array(adj)
                pos = nx.spring_layout(graph, weight="weight", seed=config.RANDOM_SEED)
                coords = np.array(list(pos.values()), dtype=np.float32)
                self._network_coords = StandardScaler().fit_transform(coords)
        return self._network_coords

    def get_cluster_labels(self, n_clusters=None):
        """Cluster sentences. Auto-detect count when n_clusters is None."""
        embeddings = self.get_embeddings()
        n = embeddings.shape[0]

        if n < 2:
            self._cluster_key = None
            self.cluster_labels = np.zeros(n, dtype=np.int32)
        elif n_clusters is None:
            if self._cluster_key == "auto":
                return self.cluster_labels
            self.cluster_labels = self._auto_cluster(embeddings)
            self._cluster_key = "auto"
        else:
            k = int(np.clip(n_clusters, 2, n))
            if self._cluster_key == k:
                return self.cluster_labels
            self.cluster_labels = self._best_for_k(embeddings, k)[0]
            self._cluster_key = k

        self._compute_cluster_metrics()
        return self.cluster_labels

    def get_top_pairs(self):
        """Return sentence pairs sorted by similarity (cached)."""
        if self._top_pairs is None:
            self._top_pairs = util.paraphrase_mining_embeddings(self.get_embeddings())
        return self._top_pairs

    def get_pairwise_stats(self):
        """Return (mean, min, max) of upper-triangle similarities."""
        sims = upper_triangle(self.get_similarity_matrix())
        if sims.size == 0:
            return 0.0, 0.0, 0.0
        return float(sims.mean()), float(sims.min()), float(sims.max())

    # ====================================================================
    # Clustering
    # ====================================================================

    def _fit_kmeans(self, embeddings, k):
        return KMeans(
            n_clusters=k,
            random_state=config.RANDOM_SEED,
            n_init=config.KMEANS_N_INIT,
        ).fit_predict(embeddings)

    def _fit_agglomerative(self, embeddings, k):
        return AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average",
        ).fit_predict(embeddings)

    def _best_for_k(self, embeddings, k):
        """Compare KMeans and Agglomerative for a given k, return (labels, score)."""
        candidates = [
            (self._fit_kmeans(embeddings, k), "kmeans"),
            (self._fit_agglomerative(embeddings, k), "agglo"),
        ]
        scored = [
            (labels, self._silhouette(embeddings, labels))
            for labels, _ in candidates
        ]
        valid = [(l, s) for l, s in scored if s is not None]

        if not valid:
            return scored[0]
        return max(valid, key=lambda x: x[1])

    def _auto_cluster(self, embeddings):
        """Select best clustering by highest silhouette score across k values."""
        n = embeddings.shape[0]
        max_k = min(config.AUTO_CLUSTER_MAX_K, n - 1)

        if max_k < 2:
            return np.zeros(n, dtype=np.int32)

        best_labels, best_score = None, -np.inf

        for k in range(2, max_k + 1):
            labels, score = self._best_for_k(embeddings, k)
            if score is not None and score > best_score:
                best_labels, best_score = labels, score

        return best_labels if best_labels is not None else np.zeros(n, dtype=np.int32)

    @staticmethod
    def _silhouette(embeddings, labels):
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

    def _compute_cluster_metrics(self):
        """Compute within/between cluster similarities, silhouette and CH index."""
        sim_matrix = self.get_similarity_matrix()
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
            sims = upper_triangle(sim_matrix)
            if sims.size:
                self.avg_within_cluster = float(sims.mean())
            return

        row_idx, col_idx = np.triu_indices(n, k=1)
        pair_sims = sim_matrix[row_idx, col_idx]
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

    def _reset(self):
        self._embeddings = None
        self._similarity_matrix = None
        self._pca_coords = None
        self._network_coords = None
        self._top_pairs = None
        self._cluster_key = None
        self.cluster_labels = None
        self.silhouette = None
        self.calinski_harabasz = None
        self.avg_within_cluster = None
        self.avg_between_clusters = None