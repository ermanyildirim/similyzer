import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
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
        if self.embeddings is None:
            if not self.sentences:
                raise ValueError("No sentences provided")
            self.embeddings = self.model.encode(
                self.sentences, show_progress_bar=False, convert_to_numpy=True,
            )
        return self.embeddings

    def get_similarity_matrix(self):
        """Compute cosine similarity matrix (cached)."""
        if self.similarity_matrix is None:
            embeddings = self.get_embeddings()
            self.similarity_matrix = cosine_similarity(embeddings)
        return self.similarity_matrix

    def get_pca_coordinates(self):
        """Reduce embeddings to 2D with PCA (cached)."""
        if self.pca_coordinates is None:
            embeddings = self.get_embeddings()
            n_sentences = len(self.sentences)
            if n_sentences < 2:
                self.pca_coordinates = np.zeros((1, 2), dtype=np.float32)
            else:
                n_components = min(2, embeddings.shape[1], n_sentences)
                self.pca_coordinates = PCA(
                    n_components=n_components, random_state=config.RANDOM_SEED,
                ).fit_transform(embeddings)
        return self.pca_coordinates

    def get_cluster_labels(self, n_clusters=None):
        """Cluster sentences. Auto-detect count when n_clusters is None."""
        embeddings = self.get_embeddings()
        n_sentences = embeddings.shape[0]

        if n_sentences < 2:
            self.cluster_labels = np.zeros(n_sentences, dtype=np.int32)
        elif n_clusters is None:
            self.cluster_labels = self._auto_cluster(embeddings)
        else:
            clamped = int(np.clip(n_clusters, 1, n_sentences))
            self.cluster_labels = self._fit_kmeans(embeddings, clamped)

        self._compute_cluster_metrics()
        return self.cluster_labels

    def get_top_pairs(self):
        """Return sentence pairs sorted by similarity (cached)."""
        if self.top_pairs is None:
            self.top_pairs = util.paraphrase_mining_embeddings(self.get_embeddings())
        return self.top_pairs

    def get_pairwise_stats(self):
        """Return (mean, min, max) of upper-triangle similarities."""
        upper_sims = upper_triangle(self.get_similarity_matrix())
        if upper_sims.size == 0:
            return 0.0, 0.0, 0.0
        return float(upper_sims.mean()), float(upper_sims.min()), float(upper_sims.max())

    # ====================================================================
    # Clustering
    # ====================================================================

    def _auto_cluster(self, embeddings):
        """Select best k by trying all candidates and picking highest silhouette."""
        n_sentences = embeddings.shape[0]
        max_k = min(config.AUTO_CLUSTER_MAX_K, n_sentences - 1)

        if max_k < 2:
            return np.zeros(n_sentences, dtype=np.int32)

        scores = {}
        for k in range(2, max_k + 1):
            labels = self._fit_kmeans(embeddings, k)
            score = self._silhouette(embeddings, labels)
            if score is not None:
                scores[k] = (score, labels)

        if not scores:
            return np.zeros(n_sentences, dtype=np.int32)

        best_k = max(scores, key=lambda k: scores[k][0])
        return scores[best_k][1]

    def _fit_kmeans(self, embeddings, n_clusters):
        """Run KMeans and return cluster labels."""
        return KMeans(
            n_clusters=n_clusters,
            random_state=config.RANDOM_SEED,
            n_init=config.KMEANS_N_INIT,
        ).fit_predict(embeddings)

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
        n_sentences = labels.size
        n_unique_clusters = np.unique(labels).size

        self.avg_within_cluster = None
        self.avg_between_clusters = None
        self.silhouette = None
        self.calinski_harabasz = None

        if n_sentences < 2:
            return

        # Single cluster — overall similarity is the within-cluster similarity
        if n_unique_clusters <= 1:
            upper_sims = upper_triangle(sim_matrix)
            if upper_sims.size:
                self.avg_within_cluster = float(upper_sims.mean())
            return

        # Within vs between cluster similarity
        row_idx, col_idx = np.triu_indices(n_sentences, k=1)
        upper_sims = sim_matrix[row_idx, col_idx]
        same_cluster = labels[row_idx] == labels[col_idx]

        within_sims = upper_sims[same_cluster]
        between_sims = upper_sims[~same_cluster]

        if within_sims.size:
            self.avg_within_cluster = float(within_sims.mean())
        if between_sims.size:
            self.avg_between_clusters = float(between_sims.mean())

        # Sklearn scoring
        if n_sentences <= n_unique_clusters:
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
        self.embeddings = None
        self.similarity_matrix = None
        self.pca_coordinates = None
        self.cluster_labels = None
        self.top_pairs = None
        self.silhouette = None
        self.calinski_harabasz = None
        self.avg_within_cluster = None
        self.avg_between_clusters = None