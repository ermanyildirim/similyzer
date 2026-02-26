import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import format_sentence_for_hover


class PlotlyVisualizer:
    """Plotly visualizations for similarity analysis."""

    _CLUSTER_COLORS = [
        "#440154", "#31688E", "#35B779", "#FDE725", "#3E4989",
        "#1F9E89", "#B5DE2B", "#482878", "#26828E", "#6ECE58",
    ]

    _EDGE_COLOR = "102,126,234"
    _EDGE_MIN_OPACITY = 0.25
    _EDGE_MAX_OPACITY = 0.70
    _EDGE_MIN_WIDTH = 1.0
    _EDGE_MAX_WIDTH = 5.0

    _CHART_HEIGHT = 600
    _CHART_HEIGHT_SMALL = 450

    _NODE_SIZE_BASE = 20.0
    _NODE_SIZE_SCALE = 40.0
    _NODE_DEFAULT_SIZE = 22.0
    _NODE_SINGLE_SIZE = 40
    _NODE_BORDER = {"color": "white", "width": 2}
    
    _TEXT_FONT_SIZE = 16

    _ORIGIN_MARKER = {
        "size": 12,
        "color": "rgba(255,255,255,0.4)",
        "symbol": "diamond",
        "line": {"color": "white", "width": 1.2},
    }

    _HOVER_LABEL = {
        "bgcolor": "rgba(0,0,0,0.90)",
        "font_family": "Arial",
        "namelength": 0,
        "align": "left",
    }

    _BASE_LAYOUT = {
        "template": "plotly_dark",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "white"},
    }

    _GRID_AXIS = {
        "showgrid": True,
        "zeroline": True,
        "showticklabels": False,
        "gridcolor": "rgba(255,255,255,0.1)",
    }

    _NO_AXIS = {"showgrid": False, "zeroline": False, "showticklabels": False}

    def __init__(self, analyzer):
        self.analyzer = analyzer

    # ====================================================================
    # Public API
    # ====================================================================

    def create_similarity_network(self, threshold):
        """Build an interactive similarity network graph and return (figure, stats)."""
        empty_stats = {"avg_degree": 0.0, "density": 0.0, "top_nodes": []}

        if len(self.analyzer.sentences) == 1:
            coords = self.analyzer.get_pca_coordinates()
            fig = self._scatter_figure(
                coords[:, 0], coords[:, 1],
                np.zeros(1, dtype=np.int32),
                sizes=[self._NODE_SINGLE_SIZE],
            )
            return self._apply_layout(fig, "Similarity Network", axis_style=self._NO_AXIS, hover_font=14), empty_stats

        similarity = self.analyzer.get_similarity_matrix()
        adjacency = self._threshold_adjacency(similarity, threshold)
        coordinates = self.analyzer.get_network_coordinates()
        node_stats = self._compute_node_stats(similarity, adjacency)

        edge_lines, edge_hover = self._build_edge_traces(
            similarity, adjacency,
            coordinates[:, 0], coordinates[:, 1],
            threshold,
        )

        nodes = self._build_cluster_traces(
            coordinates[:, 0], coordinates[:, 1],
            self.analyzer.cluster_labels,
            avg_similarity=node_stats["avg_similarity"],
            max_similarity=node_stats["max_similarity"],
            sizes=node_stats["sizes"],
        )

        fig = go.Figure(data=edge_lines + [edge_hover] + nodes + [self._origin_trace()])
        fig = self._apply_layout(fig, "Similarity Network", show_legend=True, axis_style=self._GRID_AXIS)
        return fig, node_stats

    def create_cluster_visualization(self):
        """Create cluster visualization using PCA coordinates."""
        coords = self.analyzer.get_pca_coordinates()
        labels = np.asarray(self.analyzer.cluster_labels, dtype=np.int32)

        fig = self._scatter_figure(coords[:, 0], coords[:, 1], labels)
        return self._apply_layout(fig, "Clusters (PCA)", show_legend=True, axis_style=self._GRID_AXIS)

    def create_top_pairs_chart(self, n_pairs):
        """Create side-by-side bar charts for most and least similar pairs."""
        if len(self.analyzer.sentences) < 2:
            return self._empty_figure("Need at least 2 texts")

        pairs = self.analyzer.get_top_pairs()
        k = min(int(n_pairs), len(pairs))

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Most Similar", "Least Similar"),
            horizontal_spacing=0.15,
        )

        for data, color, col in [
            (self._pairs_data(pairs[:k]), self._CLUSTER_COLORS[0], 1),
            (self._pairs_data(pairs[-k:][::-1]), self._CLUSTER_COLORS[-1], 2),
        ]:
            fig.add_trace(go.Bar(
                x=data["similarities"], y=data["labels"],
                orientation="h", hoverinfo="text", hovertext=data["hovers"],
                marker={"color": color}, showlegend=False,
            ), row=1, col=col)

        for col in (1, 2):
            fig.update_xaxes(title_text="Cosine Similarity", autorange=True, tickformat=".2f", row=1, col=col)
            fig.update_yaxes(tickfont={"size": 14}, row=1, col=col)

        return self._apply_layout(fig, height=self._CHART_HEIGHT_SMALL, hover_font=14)

    # ====================================================================
    # Layout & Shared Traces
    # ====================================================================

    def _apply_layout(self, fig, title=None, height=None, show_legend=False, axis_style=None, hover_font=16):
        options = {
            **self._BASE_LAYOUT,
            "height": height or self._CHART_HEIGHT,
            "dragmode": "pan",
            "modebar": {"orientation": "h", "bgcolor": "rgba(0,0,0,0)",
                        "color": "rgba(255,255,255,0.85)", "activecolor": "rgba(255,255,255,1)"},
            "hoverlabel": {**self._HOVER_LABEL, "font_size": hover_font},
            "showlegend": show_legend,
        }
        if title:
            options["title"] = {"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 20}}
        if axis_style:
            options["xaxis"] = axis_style
            options["yaxis"] = axis_style
        if show_legend:
            options["legend"] = {"title": "Clusters", "orientation": "v"}
        fig.update_layout(**options)
        return fig

    def _origin_trace(self):
        return go.Scatter(
            x=[0], y=[0], mode="markers",
            marker=self._ORIGIN_MARKER,
            hoverinfo="skip", showlegend=False,
        )

    def _empty_figure(self, message):
        fig = go.Figure()
        fig.add_annotation(
            text=message, xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font={"size": 20, "color": "gray"},
        )
        return self._apply_layout(fig, height=self._CHART_HEIGHT_SMALL)

    def _scatter_figure(self, x, y, labels, **kwargs):
        """Build a figure with cluster traces and origin marker."""
        traces = self._build_cluster_traces(x, y, labels, **kwargs)
        return go.Figure(data=traces + [self._origin_trace()])

    # ====================================================================
    # Network
    # ====================================================================

    @staticmethod
    def _threshold_adjacency(similarity, threshold):
        adjacency = np.where(similarity > threshold, similarity, 0.0)
        np.fill_diagonal(adjacency, 0.0)
        return adjacency

    def _compute_node_stats(self, similarity, adjacency):
        n = similarity.shape[0]

        if n <= 1:
            return {
                "avg_similarity": [0.0],
                "max_similarity": [0.0],
                "sizes": [self._NODE_SIZE_BASE + self._NODE_SIZE_SCALE],
                "avg_degree": 0.0,
                "density": 0.0,
                "top_nodes": [],
            }

        avg_sim = ((similarity.sum(axis=1) - 1.0) / (n - 1)).tolist()

        diag_mask = np.eye(n, dtype=bool)
        max_sim = np.where(diag_mask, -np.inf, similarity).max(axis=1).tolist()

        degrees = (adjacency > 0).sum(axis=1)
        ratio = degrees / (n - 1)
        sizes = (self._NODE_SIZE_BASE + ratio * self._NODE_SIZE_SCALE).tolist()

        total_edges = n * (n - 1) / 2.0
        actual_edges = float(degrees.sum()) / 2.0

        top_indices = np.argsort(degrees)[-3:][::-1]
        top_nodes = [(int(i) + 1, int(degrees[i])) for i in top_indices if degrees[i] > 0]

        return {
            "avg_similarity": avg_sim,
            "max_similarity": max_sim,
            "sizes": sizes,
            "avg_degree": float(degrees.mean()),
            "density": actual_edges / total_edges,
            "top_nodes": top_nodes,
        }

    def _build_edge_traces(self, similarity, adjacency, node_x, node_y, threshold):
        """Build per-edge line traces with individual width and opacity."""
        n = similarity.shape[0]
        sources, targets = np.triu_indices(n, k=1)
        sims = similarity[sources, targets]
        mask = adjacency[sources, targets] > 0

        if not np.any(mask):
            empty = go.Scatter(x=[], y=[], mode="markers", hoverinfo="text", showlegend=False)
            return [empty], empty

        sources, targets, sims = sources[mask], targets[mask], sims[mask]

        span = max(1e-9, 1.0 - threshold)
        strength = np.clip((sims - threshold) / span, 0.0, 1.0)

        edge_traces = []
        for (s, t), w in zip(zip(sources, targets), strength):
            opacity = self._EDGE_MIN_OPACITY + w * (self._EDGE_MAX_OPACITY - self._EDGE_MIN_OPACITY)
            width = self._EDGE_MIN_WIDTH + w * (self._EDGE_MAX_WIDTH - self._EDGE_MIN_WIDTH)
            edge_traces.append(go.Scatter(
                x=[node_x[s], node_x[t], None],
                y=[node_y[s], node_y[t], None],
                mode="lines", hoverinfo="skip", showlegend=False,
                line={"width": width, "color": f"rgba({self._EDGE_COLOR},{opacity:.2f})"},
            ))

        hover_trace = go.Scatter(
            x=(node_x[sources] + node_x[targets]) * 0.5,
            y=(node_y[sources] + node_y[targets]) * 0.5,
            mode="markers", hoverinfo="text",
            hovertext=[
                f"<b>Text {s + 1} - Text {t + 1}</b><br>Cosine similarity: {v:.3f}"
                for s, t, v in zip(sources.tolist(), targets.tolist(), sims.tolist())
            ],
            marker={"size": 16, "color": "rgba(0,0,0,0)"},
            showlegend=False,
        )
        return edge_traces, hover_trace

    # ====================================================================
    # Cluster Traces
    # ====================================================================

    def _build_node_hover(self, index, avg_sim=None, max_sim=None):
        preview = format_sentence_for_hover(self.analyzer.sentences[index])
        parts = [f"<b>Text {index + 1}</b><br><br>{preview}<br><br>"]
        if avg_sim is not None:
            parts.append(f"<b>Average cosine similarity to all texts:</b> {avg_sim:.3f}<br><br>")
        if max_sim is not None:
            parts.append(f"<b>Maximum cosine similarity to any text:</b> {max_sim:.3f}<br>")
        return "".join(parts)

    def _build_cluster_traces(self, node_x, node_y, labels, avg_similarity=None, max_similarity=None, sizes=None):
        node_x = np.asarray(node_x, dtype=np.float32)
        node_y = np.asarray(node_y, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)

        def to_array(arr):
            return np.asarray(arr, dtype=np.float32) if arr is not None else None

        avg_similarity = to_array(avg_similarity)
        max_similarity = to_array(max_similarity)
        sizes = to_array(sizes)

        traces = []
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            indices = np.flatnonzero(mask)

            hovers = [
                self._build_node_hover(
                    i,
                    avg_similarity[i] if avg_similarity is not None else None,
                    max_similarity[i] if max_similarity is not None else None,
                )
                for i in indices
            ]

            color = self._CLUSTER_COLORS[int(cluster_id) % len(self._CLUSTER_COLORS)]
            marker = {
                "color": color,
                "line": self._NODE_BORDER,
                "size": sizes[mask] if sizes is not None else self._NODE_DEFAULT_SIZE,
            }

            traces.append(go.Scatter(
                x=node_x[mask], y=node_y[mask],
                mode="markers+text",
                text=[f"Text {i + 1}" for i in indices],
                textposition="top center",
                textfont={"size": self._TEXT_FONT_SIZE, "color": "white"},
                hoverinfo="text", hovertext=hovers,
                marker=marker,
                name=f"Cluster {int(cluster_id) + 1}",
                showlegend=True,
            ))
        return traces

    # ====================================================================
    # Pairs
    # ====================================================================

    def _pairs_data(self, pairs):
        if not pairs:
            return {"labels": [], "hovers": [], "similarities": []}

        sims, sources, targets = zip(*pairs)
        sentences = self.analyzer.sentences

        labels = [f"Text {s + 1} - Text {t + 1}" for s, t in zip(sources, targets)]
        hovers = [
            f"<b>Text {s + 1} - Text {t + 1}</b><br><br>"
            f"<b>Text {s + 1}:</b><br>"
            f"{format_sentence_for_hover(sentences[s], max_width=80, max_lines=24, max_chars=4000)}<br><br>"
            f"<b>Text {t + 1}:</b><br>"
            f"{format_sentence_for_hover(sentences[t], max_width=80, max_lines=24, max_chars=4000)}<br><br>"
            f"<b>Cosine similarity:</b> {v:.3f}"
            for s, t, v in zip(sources, targets, sims)
        ]

        return {"labels": labels, "hovers": hovers, "similarities": list(sims)}