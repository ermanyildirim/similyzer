import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler

import config
from utils import format_sentence_for_hover


class PlotlyVisualizer:
    """Plotly visualizations for similarity analysis."""

    _ORIGIN_SIZE = 12
    _SINGLE_SIZE = 40
    _NODE_SIZE_BASE = 20.0
    _NODE_SIZE_SCALE = 40.0
    _TEXT_FONT_SIZE = 16
    _CHART_HEIGHT_SMALL = 450

    _VIRIDIS_COLORS = [
        "#440154", "#31688E", "#35B779", "#FDE725", "#3E4989",
        "#1F9E89", "#B5DE2B", "#482878", "#26828E", "#6ECE58",
    ]

    _MODEBAR_STYLE = {
        "orientation": "h",
        "bgcolor": "rgba(0,0,0,0)",
        "color": "rgba(255,255,255,0.85)",
        "activecolor": "rgba(255,255,255,1)",
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

    _ORIGIN_STYLE = {
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

    def __init__(self, analyzer):
        self.analyzer = analyzer

    # ====================================================================
    # Public API
    # ====================================================================

    def create_similarity_network(self, threshold):
        n_sentences = len(self.analyzer.sentences)

        empty_stats = {"avg_degree": 0.0, "density": 0.0, "top_nodes": []}

        if n_sentences == 0:
            return self._empty_figure("No texts"), empty_stats
        if n_sentences == 1:
            fig = self._single_node_network(self.analyzer.get_pca_coordinates())
            return fig, empty_stats

        similarity = self.analyzer.similarity_matrix.astype(np.float32)

        adjacency = np.where(similarity > threshold, similarity, 0.0)
        np.fill_diagonal(adjacency, 0.0)

        coordinates = self._compute_network_layout(adjacency)
        node_stats = self._compute_node_stats(similarity, adjacency)

        edges, edge_hover = self._build_edge_traces(
            similarity, adjacency, coordinates[:, 0], coordinates[:, 1], threshold
        )

        labels = self.analyzer.cluster_labels

        nodes = self._build_cluster_traces(
            coordinates[:, 0],
            coordinates[:, 1],
            labels,
            node_stats["avg_similarity"],
            node_stats["max_similarity"],
            node_stats["sizes"],
        )

        fig = go.Figure(data=edges + [edge_hover] + nodes + [self._origin_marker()])
        fig = self._apply_layout(
            fig, "Similarity Network", show_legend=True, axis_style=self._GRID_AXIS
        )

        network_stats = {
            "avg_degree": node_stats["avg_degree"],
            "density": node_stats["density"],
            "top_nodes": node_stats["top_nodes"],
        }
        return fig, network_stats

    def create_cluster_visualization(self):
        """Create cluster visualization using PCA coordinates."""
        coordinates = self.analyzer.get_pca_coordinates()
        labels = np.asarray(self.analyzer.cluster_labels, dtype=np.int32)

        if len(self.analyzer.sentences) == 0:
            return self._empty_figure("No texts")

        traces = self._build_cluster_traces(
            coordinates[:, 0], coordinates[:, 1], labels
        )
        fig = go.Figure(data=traces + [self._origin_marker()])
        return self._apply_layout(
            fig, "Clusters (PCA)", show_legend=True, axis_style=self._GRID_AXIS
        )

    def create_top_pairs_chart(self, n_pairs):
        n_sentences = len(self.analyzer.sentences)
        if n_sentences < 2:
            return self._empty_figure("Need at least 2 texts")

        pairs = self.analyzer.get_top_pairs()
        k = min(int(n_pairs), len(pairs))

        most_data = self._pairs_data(*self._extract_pairs(pairs[:k]))
        least_data = self._pairs_data(*self._extract_pairs(pairs[-k:][::-1]))

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Most Similar", "Least Similar"),
            horizontal_spacing=0.15,
        )
        for data, color, col in [
            (most_data, self._VIRIDIS_COLORS[0], 1),
            (least_data, self._VIRIDIS_COLORS[-1], 2),
        ]:
            fig.add_trace(
                go.Bar(
                    x=data["similarities"],
                    y=data["labels"],
                    orientation="h",
                    hoverinfo="text",
                    hovertext=data["hovers"],
                    marker={"color": color},
                    showlegend=False,
                ),
                row=1,
                col=col,
            )

        for col in (1, 2):
            fig.update_xaxes(
                title_text="Cosine Similarity",
                autorange=True,
                tickformat=".2f",
                row=1,
                col=col,
            )
            fig.update_yaxes(tickfont={"size": 14}, row=1, col=col)

        return self._apply_layout(fig, height=self._CHART_HEIGHT_SMALL, hover_font=14)

    # ====================================================================
    # Private: Helpers
    # ====================================================================

    def _empty_figure(self, message):
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 20, "color": "gray"},
        )
        return self._apply_layout(fig, height=self._CHART_HEIGHT_SMALL)

    def _origin_marker(self):
        style = {**self._ORIGIN_STYLE, "size": self._ORIGIN_SIZE}
        return go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            marker=style,
            hoverinfo="skip",
            showlegend=False,
        )

    def _apply_layout(
        self,
        fig,
        title=None,
        height=None,
        show_legend=False,
        axis_style=None,
        hover_font=16,
    ):
        layout_options = {
            **self._BASE_LAYOUT,
            "height": height or 600,
            "dragmode": "pan",
            "modebar": self._MODEBAR_STYLE,
            "hoverlabel": {**self._HOVER_LABEL, "font_size": hover_font},
            "showlegend": show_legend,
        }
        if title:
            layout_options["title"] = {
                "text": title,
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 20},
            }
        if axis_style:
            layout_options["xaxis"] = axis_style
            layout_options["yaxis"] = axis_style
        if show_legend:
            layout_options["legend"] = {"title": "Clusters", "orientation": "v"}
        fig.update_layout(**layout_options)
        return fig

    def _build_node_hover(self, index, avg_similarity=None, max_similarity=None):
        text = self.analyzer.sentences[index]
        preview = format_sentence_for_hover(text)
        hover = f"<b>Text {index + 1}</b><br><br>{preview}<br><br>"
        if avg_similarity is not None:
            hover += f"<b>Average cosine similarity to all texts:</b> {avg_similarity:.3f}<br><br>"
        if max_similarity is not None:
            hover += f"<b>Maximum cosine similarity to any text:</b> {max_similarity:.3f}<br>"
        return hover

    # ====================================================================
    # Private: Network
    # ====================================================================

    def _compute_network_layout(self, adjacency):
        graph = nx.from_numpy_array(adjacency)
        pos = nx.spring_layout(graph, weight="weight", seed=config.RANDOM_SEED, dim=2)
        coordinates = np.array(list(pos.values()), dtype=np.float32)
        return StandardScaler().fit_transform(coordinates).astype(np.float32)

    def _compute_node_stats(self, similarity, adjacency):
        n_sentences = similarity.shape[0]

        if n_sentences <= 1:
            default = self._NODE_SIZE_BASE + self._NODE_SIZE_SCALE
            return {
                "avg_similarity": [0.0],
                "max_similarity": [0.0],
                "sizes": [default],
                "avg_degree": 0.0,
                "density": 0.0,
                "top_nodes": [],
            }

        avg_similarity = ((similarity.sum(axis=1) - 1.0) / (n_sentences - 1)).tolist()
        diagonal_mask = np.eye(n_sentences, dtype=bool)
        max_similarity = (
            np.where(diagonal_mask, -np.inf, similarity).max(axis=1).tolist()
        )

        connections = np.maximum((adjacency > 0).sum(axis=1) - 1, 0)
        ratio = connections / (n_sentences - 1)
        sizes = (self._NODE_SIZE_BASE + ratio * self._NODE_SIZE_SCALE).tolist()

        degrees = connections.astype(np.float32)
        total_edges = n_sentences * (n_sentences - 1) / 2.0
        actual_edges = float(degrees.sum()) / 2.0

        top_indices = np.argsort(degrees)[-3:][::-1]
        top_nodes = [
            (int(i) + 1, int(degrees[i])) for i in top_indices if degrees[i] > 0
        ]

        return {
            "avg_similarity": avg_similarity,
            "max_similarity": max_similarity,
            "sizes": sizes,
            "avg_degree": float(degrees.mean()),
            "density": actual_edges / total_edges,
            "top_nodes": top_nodes,
        }

    def _build_edge_traces(self, similarity, adjacency, node_x, node_y, threshold):
        n_sentences = similarity.shape[0]
        sources, targets = np.triu_indices(n_sentences, k=1)
        similarities = similarity[sources, targets]
        mask = adjacency[sources, targets] > 0

        if not np.any(mask):
            return [], go.Scatter(
                x=[], y=[], mode="markers", hoverinfo="text", showlegend=False
            )

        sources = sources[mask]
        targets = targets[mask]
        similarities = similarities[mask]

        span = max(1e-9, 1.0 - threshold)
        strength = np.clip((similarities - threshold) / span, 0.0, 1.0)

        count = len(sources)
        edge_x = np.empty(count * 3, dtype=np.float32)
        edge_y = np.empty(count * 3, dtype=np.float32)
        edge_x[0::3] = node_x[sources]
        edge_x[1::3] = node_x[targets]
        edge_x[2::3] = np.nan
        edge_y[0::3] = node_y[sources]
        edge_y[1::3] = node_y[targets]
        edge_y[2::3] = np.nan

        median_strength = float(np.median(strength))
        width = 1.0 + median_strength * 5.0
        opacity = 0.15 + median_strength * 0.55

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            hoverinfo="skip",
            showlegend=False,
            line={"width": width, "color": f"rgba(102,126,234,{opacity:.2f})"},
        )

        midpoint_x = (node_x[sources] + node_x[targets]) * 0.5
        midpoint_y = (node_y[sources] + node_y[targets]) * 0.5
        hover_texts = [
            f"<b>Text {s + 1} - Text {t + 1}</b><br>Cosine similarity: {v:.3f}"
            for s, t, v in zip(
                sources.tolist(), targets.tolist(), similarities.tolist()
            )
        ]
        hover_trace = go.Scatter(
            x=midpoint_x,
            y=midpoint_y,
            mode="markers",
            hoverinfo="text",
            hovertext=hover_texts,
            marker={"size": 16, "color": "rgba(0,0,0,0)"},
            showlegend=False,
        )
        return [edge_trace], hover_trace

    def _build_cluster_traces(
        self,
        node_x,
        node_y,
        labels,
        avg_similarity=None,
        max_similarity=None,
        sizes=None,
    ):
        node_x = np.asarray(node_x, dtype=np.float32)
        node_y = np.asarray(node_y, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)

        if sizes is not None:
            sizes = np.asarray(sizes, dtype=np.float32)
        if avg_similarity is not None:
            avg_similarity = np.asarray(avg_similarity, dtype=np.float32)
        if max_similarity is not None:
            max_similarity = np.asarray(max_similarity, dtype=np.float32)

        traces = []
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            indices = np.flatnonzero(mask)

            hovers = []
            for i in indices:
                avg = avg_similarity[i] if avg_similarity is not None else None
                max_sim = max_similarity[i] if max_similarity is not None else None
                hovers.append(self._build_node_hover(i, avg, max_sim))

            color = self._VIRIDIS_COLORS[int(cluster_id) % len(self._VIRIDIS_COLORS)]
            marker = {"color": color, "line": {"color": "white", "width": 2}}
            if sizes is not None:
                marker["size"] = sizes[mask]
            else:
                marker["size"] = 22.0
            traces.append(
                go.Scatter(
                    x=node_x[mask],
                    y=node_y[mask],
                    mode="markers+text",
                    text=[f"Text {i + 1}" for i in indices],
                    textposition="top center",
                    textfont={"size": self._TEXT_FONT_SIZE, "color": "white"},
                    hoverinfo="text",
                    hovertext=hovers,
                    marker=marker,
                    name=f"Cluster {int(cluster_id) + 1}",
                    showlegend=True,
                )
            )
        return traces

    def _single_node_network(self, coordinates):
        x = float(coordinates[0, 0]) if coordinates.size else 0.0
        y = float(coordinates[0, 1]) if coordinates.size else 0.0
        hover = self._build_node_hover(0)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                text=["Text 1"],
                textposition="top center",
                textfont={"size": self._TEXT_FONT_SIZE, "color": "white"},
                hoverinfo="text",
                hovertext=[hover],
                marker={
                    "size": self._SINGLE_SIZE,
                    "color": self._VIRIDIS_COLORS[0],
                    "line": {"color": "white", "width": 2},
                },
                showlegend=False,
            )
        )
        return self._apply_layout(
            fig,
            "Similarity Network",
            axis_style={"showgrid": False, "zeroline": False, "showticklabels": False},
            hover_font=14,
        )

    # ====================================================================
    # Private: Pairs
    # ====================================================================

    def _extract_pairs(self, pairs):
        """Unpack (score, source, target) triples into separate arrays."""
        similarities, sources, targets = zip(*pairs) if pairs else ([], [], [])
        return np.array(sources), np.array(targets), np.array(similarities)

    def _pairs_data(self, sources, targets, similarities):
        sentences = self.analyzer.sentences
        sources, targets, similarities = (
            sources.tolist(),
            targets.tolist(),
            similarities.tolist(),
        )
        labels = [f"Text {s + 1} - Text {t + 1}" for s, t in zip(sources, targets)]

        def format_hover(text):
            return format_sentence_for_hover(
                text, max_width=80, max_lines=24, max_chars=4000
            )

        hovers = [
            f"<b>Text {s + 1} - Text {t + 1}</b><br><br>"
            f"<b>Text {s + 1}:</b><br>{format_hover(sentences[s])}<br><br>"
            f"<b>Text {t + 1}:</b><br>{format_hover(sentences[t])}<br><br>"
            f"<b>Cosine similarity:</b> {v:.3f}"
            for s, t, v in zip(sources, targets, similarities)
        ]
        return {
            "labels": labels,
            "hovers": hovers,
            "similarities": similarities,
        }