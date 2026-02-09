import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score

import config
from utils import normalize_whitespace, upper_triangle


@st.cache_resource(show_spinner=False)
def load_model(model_name):
    return SentenceTransformer(model_name)


class SentenceAnalyzer:
    """Compute embeddings, similarity, PCA reduction and clustering for sentences."""

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = load_model(model_name)

        # Input data
        self.sentences = []
        self.processed_sentences = []

        # Embedding results
        self.embeddings = None
        self.similarity_matrix = None

        self._pca_coordinates = None

        # Clustering results
        self.cluster_labels = None
        self.silhouette = None
        self.calinski_harabasz = None
        self.avg_within_cluster = None
        self.avg_between_clusters = None

    # ====================================================================
    # Public API
    # ====================================================================

    def add_sentences(self, sentences):
        """Set sentences, apply whitespace normalization and reset cached results."""
        self.sentences = list(sentences)
        self.processed_sentences = [
            normalize_whitespace(sentence) for sentence in self.sentences
        ]
        self._reset_all()

    def get_embeddings(self):
        """Compute sentence embeddings (or return cached)."""
        if self.embeddings is not None:
            return self.embeddings

        if not self.processed_sentences:
            raise ValueError("No sentences provided")

        embeddings = self.model.encode(
            self.processed_sentences,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        self.embeddings = embeddings
        return embeddings

    def calculate_similarity(self):
        """Compute cosine similarity matrix (or return cached)."""
        if self.similarity_matrix is not None:
            return self.similarity_matrix

        if self.embeddings is None:
            self.get_embeddings()

        similarity_matrix = np.dot(self.embeddings, self.embeddings.T)

        self.similarity_matrix = similarity_matrix
        return similarity_matrix

    def reduce_dimensions(self):
        """Reduce embeddings to 2D using PCA for visualization."""
        if self._pca_coordinates is not None:
            return self._pca_coordinates

        if self.embeddings is None:
            self.get_embeddings()

        num_sentences = len(self.sentences)

        if num_sentences < 2:
            self._pca_coordinates = np.array([[0.0, 0.0]], dtype=np.float32)
            return self._pca_coordinates

        num_components = min(2, self.embeddings.shape[1], num_sentences)

        pca = PCA(n_components=num_components, random_state=config.RANDOM_SEED)
        coordinates = pca.fit_transform(self.embeddings)

        if coordinates.shape[1] == 1:
            zeros_column = np.zeros(num_sentences, dtype=np.float32)
            coordinates = np.column_stack([coordinates[:, 0], zeros_column])

        self._pca_coordinates = coordinates
        return coordinates

    def perform_clustering(self, num_clusters):
        """Perform K-Means clustering and compute quality metrics."""
        if self.embeddings is None:
            self.get_embeddings()
        if self.similarity_matrix is None:
            self.calculate_similarity()

        num_samples = self.embeddings.shape[0]

        if num_samples < 2:
            self.cluster_labels = np.zeros(num_samples, dtype=np.int32)
            self.silhouette = None
            self._compute_cluster_metrics()
            return self.cluster_labels

        if num_clusters is None:
            self._auto_cluster()
        else:
            self._kmeans_cluster(int(num_clusters))

        self._compute_cluster_metrics()
        return self.cluster_labels

    # ====================================================================
    # Private: Clustering
    # ====================================================================

    def _create_kmeans(self, num_clusters):
        return KMeans(
            n_clusters=num_clusters,
            random_state=config.RANDOM_SEED,
            n_init=config.KMEANS_N_INIT,
        )

    def _auto_cluster(self):
        """Select optimal cluster count using silhouette score."""
        num_samples = self.embeddings.shape[0]
        upper_bound = min(config.DEFAULT_MAX_CLUSTERS, num_samples - 1)

        self.cluster_labels = np.zeros(num_samples, dtype=np.int32)
        self.silhouette = None

        if upper_bound < 2:
            return

        best_score = None
        best_labels = None

        for cluster_count in range(2, upper_bound + 1):
            result = self._try_clustering(cluster_count)
            if result is None:
                continue
            score, labels = result
            if best_score is None or score > best_score:
                best_score = score
                best_labels = labels

        if best_labels is not None:
            self.cluster_labels = best_labels
            self.silhouette = float(best_score)

    def _kmeans_cluster(self, num_clusters):
        num_samples = self.embeddings.shape[0]
        cluster_count = max(1, min(num_clusters, num_samples))

        kmeans = self._create_kmeans(cluster_count)
        self.cluster_labels = kmeans.fit_predict(self.embeddings)

        self.silhouette = None
        has_multiple_clusters = np.unique(self.cluster_labels).size >= 2

        if has_multiple_clusters and cluster_count < num_samples:
            try:
                self.silhouette = float(
                    silhouette_score(
                        self.embeddings, self.cluster_labels, metric="cosine"
                    )
                )
            except ValueError:
                pass

    def _try_clustering(self, num_clusters):
        """Attempt K-Means and return (score, labels) or None on failure."""
        try:
            kmeans = self._create_kmeans(num_clusters)
            labels = kmeans.fit_predict(self.embeddings)

            if np.unique(labels).size < 2:
                return None

            score = float(silhouette_score(self.embeddings, labels, metric="cosine"))
            return score, labels
        except ValueError:
            return None

    def _compute_cluster_metrics(self):
        """Compute within/between cluster similarities and Calinski-Harabasz index."""
        self.avg_within_cluster = None
        self.avg_between_clusters = None
        self.calinski_harabasz = None

        if self.cluster_labels is None or self.similarity_matrix is None:
            return

        num_samples = self.cluster_labels.size
        num_clusters = np.unique(self.cluster_labels).size

        if num_clusters <= 1 or num_samples < 2:
            upper_sims = upper_triangle(self.similarity_matrix)
            if upper_sims.size:
                self.avg_within_cluster = float(np.mean(upper_sims))
            return

        row_index, column_index = np.triu_indices(num_samples, k=1)
        pairwise = self.similarity_matrix[row_index, column_index].astype(np.float32)

        same_cluster = self.cluster_labels[row_index] == self.cluster_labels[column_index]
        within = pairwise[same_cluster]
        between = pairwise[~same_cluster]

        if within.size:
            self.avg_within_cluster = float(within.mean())
        if between.size:
            self.avg_between_clusters = float(between.mean())

        # Calinski-Harabasz Index (Variance Ratio Criterion)
        if num_clusters >= 2 and num_samples > num_clusters:
            try:
                self.calinski_harabasz = float(
                    calinski_harabasz_score(self.embeddings, self.cluster_labels)
                )
            except ValueError:
                pass

    # ====================================================================
    # Private: Utilities
    # ====================================================================

    def _reset_all(self):
        self.embeddings = None
        self.similarity_matrix = None
        self._pca_coordinates = None
        self.cluster_labels = None
        self.silhouette = None
        self.calinski_harabasz = None
        self.avg_within_cluster = None
        self.avg_between_clusters = None

