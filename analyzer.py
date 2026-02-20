import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score

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
        self._reset_all()

    # ====================================================================
    # Public API
    # ====================================================================

    def add_sentences(self, sentences):
        """Set sentences and reset cached results."""
        self.sentences = list(sentences)
        self._reset_all()

    def get_embeddings(self):
        """Compute sentence embeddings (or return cached)."""
        if self.embeddings is not None:
            return self.embeddings

        if not self.sentences:
            raise ValueError("No sentences provided")

        self.embeddings = np.asarray(
            self.model.encode(self.sentences, show_progress_bar=False),
            dtype=np.float32,
        )
        return self.embeddings

    def get_similarity_matrix(self):
        """Compute cosine similarity matrix (or return cached)."""
        if self.similarity_matrix is not None:
            return self.similarity_matrix

        if self.embeddings is None:
            self.get_embeddings()

        self.similarity_matrix = util.cos_sim(self.embeddings, self.embeddings).numpy()
        return self.similarity_matrix

    def get_pca_coordinates(self):
        """Reduce embeddings to 2D using PCA for visualization (or return cached)."""
        if self.pca_coordinates is not None:
            return self.pca_coordinates

        if self.embeddings is None:
            self.get_embeddings()

        n_sentences = len(self.sentences)

        if n_sentences < 2:
            self.pca_coordinates = np.array([[0.0, 0.0]], dtype=np.float32)
            return self.pca_coordinates

        n_components = min(2, self.embeddings.shape[1], n_sentences)
        pca = PCA(n_components=n_components, random_state=config.RANDOM_SEED)
        self.pca_coordinates = pca.fit_transform(self.embeddings)
        return self.pca_coordinates

    def get_cluster_labels(self, n_clusters):
        """Perform K-Means clustering and compute quality metrics."""
        if self.embeddings is None:
            self.get_embeddings()
        if self.similarity_matrix is None:
            self.get_similarity_matrix()

        n_sentences = self.embeddings.shape[0]

        if n_sentences < 2:
            self.cluster_labels = np.zeros(n_sentences, dtype=np.int32)
            self._compute_cluster_metrics()
            return self.cluster_labels

        if n_clusters is None:
            self._auto_cluster()
        else:
            self._kmeans_cluster(int(n_clusters))

        self._compute_cluster_metrics()
        return self.cluster_labels

    def get_top_pairs(self):
        """Return scored pairs sorted by similarity (or return cached)."""
        if self.top_pairs is not None:
            return self.top_pairs

        if self.embeddings is None:
            self.get_embeddings()

        self.top_pairs = util.paraphrase_mining_embeddings(self.embeddings)
        return self.top_pairs

    def get_pairwise_stats(self):
        """Return (mean, min, max) of upper-triangle similarity values."""
        if self.similarity_matrix is None:
            self.get_similarity_matrix()

        pairwise = upper_triangle(self.similarity_matrix)
        if pairwise.size == 0:
            return 0.0, 0.0, 0.0
        return float(pairwise.mean()), float(pairwise.min()), float(pairwise.max())

    # ====================================================================
    # Private: Clustering
    # ====================================================================

    def _create_kmeans(self, n_clusters):
        """Create a KMeans instance with standard config."""
        return KMeans(
            n_clusters=n_clusters,
            random_state=config.RANDOM_SEED,
            n_init=config.KMEANS_N_INIT,
        )

    def _auto_cluster(self):
        """Select optimal cluster count using silhouette score with early stopping."""
        n_sentences = self.embeddings.shape[0]
        upper_bound = min(config.DEFAULT_MAX_CLUSTERS, n_sentences - 1)

        self.cluster_labels = np.zeros(n_sentences, dtype=np.int32)

        if upper_bound < 2:
            return

        best_score = None
        best_labels = None
        consecutive_drops = 0

        for n_clusters in range(2, upper_bound + 1):
            result = self._try_clustering(n_clusters)
            if result is None:
                continue
            score, labels = result

            if best_score is not None and score < best_score:
                consecutive_drops += 1
                if consecutive_drops >= 2:
                    break
            else:
                consecutive_drops = 0

            if best_score is None or score > best_score:
                best_score = score
                best_labels = labels

        if best_labels is not None:
            self.cluster_labels = best_labels

    def _kmeans_cluster(self, n_clusters):
        n_sentences = self.embeddings.shape[0]
        clamped = max(1, min(n_clusters, n_sentences))
        self.cluster_labels = self._create_kmeans(clamped).fit_predict(self.embeddings)

    def _try_clustering(self, n_clusters):
        """Attempt K-Means and return (score, labels) or None on failure."""
        try:
            labels = self._create_kmeans(n_clusters).fit_predict(self.embeddings)

            if np.unique(labels).size < 2:
                return None

            score = float(silhouette_score(self.embeddings, labels, metric="cosine"))
            return score, labels
        except ValueError:
            return None

    # ====================================================================
    # Private: Metrics
    # ====================================================================

    def _safe_score(self, func, *args, **kwargs):
        """Compute a scoring function, returning None on failure."""
        try:
            return float(func(*args, **kwargs))
        except ValueError:
            return None

    def _compute_cluster_metrics(self):
        """Compute within/between cluster similarities, silhouette and Calinski-Harabasz index."""
        self.avg_within_cluster = None
        self.avg_between_clusters = None
        self.calinski_harabasz = None
        self.silhouette = None

        n_sentences = self.cluster_labels.size
        n_clusters = np.unique(self.cluster_labels).size

        if n_clusters <= 1 or n_sentences < 2:
            upper_sims = upper_triangle(self.similarity_matrix)
            if upper_sims.size:
                self.avg_within_cluster = float(np.mean(upper_sims))
            return

        row_index, column_index = np.triu_indices(n_sentences, k=1)
        pairwise = self.similarity_matrix[row_index, column_index].astype(np.float32)

        same_cluster = self.cluster_labels[row_index] == self.cluster_labels[column_index]
        within = pairwise[same_cluster]
        between = pairwise[~same_cluster]

        if within.size:
            self.avg_within_cluster = float(within.mean())
        if between.size:
            self.avg_between_clusters = float(between.mean())

        if n_sentences > n_clusters:
            self.silhouette = self._safe_score(
                silhouette_score, self.embeddings, self.cluster_labels, metric="cosine"
            )
            self.calinski_harabasz = self._safe_score(
                calinski_harabasz_score, self.embeddings, self.cluster_labels
            )

    # ====================================================================
    # Private: Utilities
    # ====================================================================

    def _reset_all(self):
        self.embeddings = None
        self.similarity_matrix = None
        self.pca_coordinates = None
        self.cluster_labels = None
        self.top_pairs = None
        self.silhouette = None
        self.calinski_harabasz = None
        self.avg_within_cluster = None
        self.avg_between_clusters = None