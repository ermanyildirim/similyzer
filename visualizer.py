import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config
from utils import format_sentence_for_hover, normalize_coordinates


class PlotlyVisualizer:
    """Plotly visualizations for similarity analysis."""

    _ORIGIN_SIZE = 12
    _SINGLE_SIZE = 40

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

    def _ensure_similarity(self):
        if self.analyzer.similarity_matrix is None:
            self.analyzer.calculate_similarity()

    ###### Public API ######

    def create_similarity_network(self, threshold):
        self._ensure_similarity()

        num_nodes = len(self.analyzer.sentences)
        if self.analyzer.cluster_labels is None and num_nodes >= 2:
            try:
                self.analyzer.perform_clustering(None)
            except Exception:
                pass  # Continue without clusters if clustering fails

        if num_nodes == 0:
            return self._empty_figure("No texts")
        if num_nodes == 1:
            return self._single_node_network(self.analyzer.reduce_dimensions())

        similarity = self.analyzer.similarity_matrix.astype(np.float32)
        coordinates = self._compute_network_layout(similarity, threshold)
        node_stats = self._compute_node_stats(similarity, threshold)

        edges, edge_hover = self._build_edge_traces(
            similarity, coordinates[:, 0], coordinates[:, 1], threshold
        )

        labels = self.analyzer.cluster_labels
        if labels is None or len(labels) != num_nodes:
            labels = np.zeros(num_nodes, dtype=np.int32)

        nodes = self._build_cluster_traces(
            coordinates[:, 0],
            coordinates[:, 1],
            labels,
            node_stats["avg_similarity"],
            node_stats["max_similarity"],
            node_stats["sizes"],
        )

        fig = go.Figure(data=edges + [edge_hover] + nodes + [self._origin_marker()])
        return self._apply_layout(
            fig, "Similarity Network", show_legend=True, axis_style=config.GRID_AXIS
        )

    def compute_network_stats(self, threshold):
        self._ensure_similarity()
        node_stats = self._compute_node_stats(
            self.analyzer.similarity_matrix.astype(np.float32), threshold
        )
        return {
            "avg_degree": node_stats["avg_degree"],
            "density": node_stats["density"],
            "top_nodes": node_stats["top_nodes"],
        }

    def create_cluster_visualization(self):
        """Create cluster visualization using PCA coordinates."""
        if self.analyzer.cluster_labels is None:
            self.analyzer.perform_clustering(None)

        coordinates = self.analyzer.reduce_dimensions()
        labels = np.asarray(self.analyzer.cluster_labels, dtype=np.int32)

        if len(self.analyzer.sentences) == 0:
            return self._empty_figure("No texts")

        traces = self._build_cluster_traces(
            coordinates[:, 0], coordinates[:, 1], labels
        )
        fig = go.Figure(data=traces + [self._origin_marker()])
        return self._apply_layout(
            fig, "Clusters (PCA)", show_legend=True, axis_style=config.GRID_AXIS
        )

    def create_top_pairs_chart(self, num_pairs):
        self._ensure_similarity()

        num_texts = len(self.analyzer.sentences)
        if num_texts < 2:
            return self._empty_figure("Need at least 2 texts")

        similarity = self.analyzer.similarity_matrix
        row_index, column_index = np.triu_indices(num_texts, k=1)
        similarities = similarity[row_index, column_index]
        k = min(int(num_pairs), len(similarities))

        # Top k most similar (reversed for chart display)
        most_indices = np.argpartition(-similarities, k - 1)[:k]
        most_indices = most_indices[np.argsort(-similarities[most_indices])][::-1]

        # Top k least similar (reversed for chart display)
        least_indices = np.argpartition(similarities, k - 1)[:k]
        least_indices = least_indices[np.argsort(similarities[least_indices])][::-1]

        most_data = self._pairs_data(
            row_index[most_indices],
            column_index[most_indices],
            similarities[most_indices],
        )
        least_data = self._pairs_data(
            row_index[least_indices],
            column_index[least_indices],
            similarities[least_indices],
        )

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Most Similar", "Least Similar"),
            horizontal_spacing=0.15,
        )
        for data, color, col in [
            (most_data, config.VIRIDIS_COLORS[0], 1),
            (least_data, config.VIRIDIS_COLORS[-1], 2),
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

        return self._apply_layout(fig, height=config.CHART_HEIGHT_SMALL, hover_font=14)

    ###### Private: Helpers ######

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
        return self._apply_layout(fig, height=config.CHART_HEIGHT_SMALL)

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
            **config.BASE_LAYOUT,
            "height": height or config.CHART_HEIGHT,
            "dragmode": "pan",
            "modebar": config.MODEBAR_STYLE,
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
            layout_options["legend"] = config.LEGEND_CONFIG
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

    ###### Private: Network ######

    def _compute_network_layout(self, similarity, threshold):
        """Compute spring layout positions for network nodes."""
        adjacency = np.where(
            similarity <= threshold, 0.0, similarity.astype(np.float32)
        )
        np.fill_diagonal(adjacency, 0.0)
        graph = nx.from_numpy_array(adjacency)
        pos = nx.spring_layout(graph, weight="weight", seed=config.RANDOM_SEED, dim=2)
        coordinates = np.array(list(pos.values()), dtype=np.float32)
        return normalize_coordinates(coordinates)

    def _compute_node_stats(self, similarity, threshold):
        num_nodes = similarity.shape[0]

        if num_nodes <= 1:
            default = config.NODE_SIZE_BASE + config.NODE_SIZE_SCALE
            return {
                "avg_similarity": [0.0],
                "max_similarity": [0.0],
                "sizes": [default],
                "avg_degree": 0.0,
                "density": 0.0,
                "top_nodes": [],
            }

        # Similarity stats per node
        avg_similarity = ((similarity.sum(axis=1) - 1.0) / (num_nodes - 1)).tolist()
        diagonal_mask = np.eye(num_nodes, dtype=bool)
        max_similarity = (
            np.where(diagonal_mask, -np.inf, similarity).max(axis=1).tolist()
        )

        # Node sizes based on connection count
        connections = np.maximum((similarity > threshold).sum(axis=1) - 1, 0)
        ratio = connections / (num_nodes - 1)
        sizes = (config.NODE_SIZE_BASE + ratio * config.NODE_SIZE_SCALE).tolist()

        # Network-level stats
        degrees = connections.astype(np.float32)
        total_edges = num_nodes * (num_nodes - 1) / 2.0
        actual_edges = float(degrees.sum()) / 2.0

        top_indices = np.argsort(degrees)[-config.TOP_NODES_COUNT :][::-1]
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

    def _build_edge_traces(self, similarity, node_x, node_y, threshold):
        num_nodes = similarity.shape[0]
        sources, targets = np.triu_indices(num_nodes, k=1)
        similarities = similarity[sources, targets]
        mask = similarities > threshold

        # Early return if no edges above threshold
        if not np.any(mask):
            return [], go.Scatter(
                x=[], y=[], mode="markers", hoverinfo="text", showlegend=False
            )

        # Filter to edges above threshold
        sources = sources[mask]
        targets = targets[mask]
        similarities = similarities[mask]

        # Normalize edge strengths and bin into 3 groups
        normalized = np.clip(
            (similarities - threshold) / max(1e-9, 1.0 - threshold), 0.0, 1.0
        )
        strength_bins = np.digitize(normalized, config.EDGE_STRENGTH_BINS)

        # Build edge traces for each strength group
        traces = []
        r, g, b = config.EDGE_COLOR_RGB
        for i in (0, 1, 2):
            bin_mask = strength_bins == i
            if not np.any(bin_mask):
                continue
            bin_sources, bin_targets, count = (
                sources[bin_mask],
                targets[bin_mask],
                bin_mask.sum(),
            )
            edge_x = np.empty(count * 3, dtype=np.float32)
            edge_y = np.empty(count * 3, dtype=np.float32)
            edge_x[0::3], edge_x[1::3], edge_x[2::3] = (
                node_x[bin_sources],
                node_x[bin_targets],
                np.nan,
            )
            edge_y[0::3], edge_y[1::3], edge_y[2::3] = (
                node_y[bin_sources],
                node_y[bin_targets],
                np.nan,
            )
            traces.append(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    hoverinfo="skip",
                    showlegend=False,
                    line={
                        "width": config.EDGE_WIDTHS[i],
                        "color": f"rgba({r},{g},{b},{config.EDGE_OPACITIES[i]})",
                    },
                )
            )

        # Build hover trace at edge midpoints
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
            marker={"size": config.EDGE_HOVER_SIZE, "color": "rgba(0,0,0,0)"},
            showlegend=False,
        )
        return traces, hover_trace

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

            color = config.VIRIDIS_COLORS[int(cluster_id) % len(config.VIRIDIS_COLORS)]
            marker = {"color": color, "line": {"color": "white", "width": 2}}
            if sizes is not None:
                marker["size"] = sizes[mask]
            else:
                marker["size"] = config.CLUSTER_MARKER_SIZE
            traces.append(
                go.Scatter(
                    x=node_x[mask],
                    y=node_y[mask],
                    mode="markers+text",
                    text=[f"Text {i + 1}" for i in indices],
                    textposition="top center",
                    textfont={"size": config.CHART_FONT_SIZE, "color": "white"},
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
                textfont={"size": config.CHART_FONT_SIZE, "color": "white"},
                hoverinfo="text",
                hovertext=[hover],
                marker={
                    "size": self._SINGLE_SIZE,
                    "color": config.VIRIDIS_COLORS[0],
                    "line": {"color": "white", "width": 2},
                },
                showlegend=False,
            )
        )
        return self._apply_layout(
            fig, "Similarity Network", axis_style=config.HIDDEN_AXIS, hover_font=14
        )

    ###### Private: Pairs ######

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
                text,
                max_width=config.HOVER_WIDTH_EXTENDED,
                max_lines=config.HOVER_LINES_EXTENDED,
                max_chars=config.HOVER_CHARS_EXTENDED,
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
