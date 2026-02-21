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
            emb = self.get_embeddings()
            self.similarity_matrix = util.cos_sim(emb, emb).numpy()
        return self.similarity_matrix

    def get_pca_coordinates(self):
        """Reduce embeddings to 2D with PCA (cached)."""
        if self.pca_coordinates is None:
            emb = self.get_embeddings()
            n = len(self.sentences)
            if n < 2:
                self.pca_coordinates = np.zeros((1, 2), dtype=np.float32)
            else:
                n_comp = min(2, emb.shape[1], n)
                self.pca_coordinates = PCA(
                    n_components=n_comp, random_state=config.RANDOM_SEED,
                ).fit_transform(emb)
        return self.pca_coordinates

    def get_cluster_labels(self, n_clusters=None):
        """Cluster sentences. Auto-detect count when n_clusters is None."""
        emb = self.get_embeddings()
        n = emb.shape[0]

        if n < 2:
            self.cluster_labels = np.zeros(n, dtype=np.int32)
        elif n_clusters is None:
            self.cluster_labels = self._auto_cluster(emb)
        else:
            k = int(np.clip(n_clusters, 1, n))
            self.cluster_labels = self._fit_kmeans(emb, k)

        self._compute_cluster_metrics()
        return self.cluster_labels

    def get_top_pairs(self):
        """Return sentence pairs sorted by similarity (cached)."""
        if self.top_pairs is None:
            self.top_pairs = util.paraphrase_mining_embeddings(self.get_embeddings())
        return self.top_pairs

    def get_pairwise_stats(self):
        """Return (mean, min, max) of upper-triangle similarities."""
        tri = upper_triangle(self.get_similarity_matrix())
        if tri.size == 0:
            return 0.0, 0.0, 0.0
        return float(tri.mean()), float(tri.min()), float(tri.max())

    # ====================================================================
    # Clustering
    # ====================================================================

    def _auto_cluster(self, embeddings):
        """Select best k via silhouette score, with early stopping after 2 drops."""
        n = embeddings.shape[0]
        max_k = min(config.DEFAULT_MAX_CLUSTERS, n - 1)

        if max_k < 2:
            return np.zeros(n, dtype=np.int32)

        best_score = None
        best_labels = None
        drops = 0

        for k in range(2, max_k + 1):
            labels = self._fit_kmeans(embeddings, k)

            if np.unique(labels).size < 2:
                continue

            score = self._silhouette(embeddings, labels)
            if score is None:
                continue

            if best_score is None or score > best_score:
                best_score, best_labels, drops = score, labels, 0
            else:
                drops += 1
                if drops >= 2:
                    break

        return best_labels if best_labels is not None else np.zeros(n, dtype=np.int32)

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
        try:
            return float(silhouette_score(embeddings, labels, metric="cosine"))
        except ValueError:
            return None

    # ====================================================================
    # Metrics
    # ====================================================================

    def _compute_cluster_metrics(self):
        """Compute within/between cluster similarities, silhouette and CH index."""
        sim = self.get_similarity_matrix()
        labels = self.cluster_labels
        n = labels.size
        n_unique = np.unique(labels).size

        self.avg_within_cluster = None
        self.avg_between_clusters = None
        self.silhouette = None
        self.calinski_harabasz = None

        if n < 2:
            return

        # Single cluster — overall similarity is the within-cluster similarity
        if n_unique <= 1:
            tri = upper_triangle(sim)
            if tri.size:
                self.avg_within_cluster = float(tri.mean())
            return

        # Within vs between cluster similarity
        row, col = np.triu_indices(n, k=1)
        pairwise = sim[row, col].astype(np.float32)
        same = labels[row] == labels[col]

        if pairwise[same].size:
            self.avg_within_cluster = float(pairwise[same].mean())
        if pairwise[~same].size:
            self.avg_between_clusters = float(pairwise[~same].mean())

        # Sklearn scoring
        if n <= n_unique:
            return

        emb = self.get_embeddings()
        self.silhouette = self._silhouette(emb, labels)

        try:
            self.calinski_harabasz = float(calinski_harabasz_score(emb, labels))
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