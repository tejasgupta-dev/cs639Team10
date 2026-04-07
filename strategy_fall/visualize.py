from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx

from utils import Graph, Node, Edge


NODE_COLOR_MAP: Dict[Optional[str], str] = {
    "question": "#4C78A8",
    "reasoning": "#72B7B2",
    "result": "#F58518",
    "anchor": "#E45756",
    None: "#B0B0B0",
}

EDGE_COLOR_MAP: Dict[Optional[str], str] = {
    "leads to": "#4C78A8",
    "equivalence": "#54A24B",
    "contradiction": "#E45756",
    "contrapositive": "#B279A2",
    None: "#7F7F7F",
}


def to_networkx(graph: Graph) -> nx.DiGraph:
    """
    Convert custom Graph object into a NetworkX directed graph.
    """
    G = nx.DiGraph()

    for node in graph.nodes:
        G.add_node(
            node.node_id,
            label=_node_label(node),
            node_type=getattr(node, "node_type", None),
            context=getattr(node, "context", ""),
        )

    for edge in graph.edges:
        edge_type = getattr(edge, "edge_type", None)
        G.add_edge(
            edge.source.node_id,
            edge.target.node_id,
            weight=getattr(edge, "weight", 0),
            edge_type=edge_type,
            label=edge_type if edge_type is not None else "",
        )

    return G


def _node_label(node: Node, max_len: int = 28) -> str:
    prefix = f"N{node.node_id}"
    text = (node.context or "").strip().replace("\n", " ")
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return f"{prefix}: {text}" if text else prefix


def draw_graph(
    graph: Graph,
    title: str = "Reasoning Graph",
    figsize: Tuple[int, int] = (12, 8),
    with_edge_labels: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    Draw a single reasoning graph.
    """
    G = to_networkx(graph)

    if len(G.nodes) == 0:
        print("Graph is empty.")
        return

    pos = nx.kamada_kawai_layout(G)

    node_colors = [
        NODE_COLOR_MAP.get(G.nodes[n].get("node_type"), "#B0B0B0")
        for n in G.nodes
    ]
    edge_colors = [
        EDGE_COLOR_MAP.get(G.edges[e].get("edge_type"), "#7F7F7F")
        for e in G.edges
    ]

    labels = {n: G.nodes[n]["label"] for n in G.nodes}
    edge_labels = {(u, v): G.edges[u, v].get("label", "") for u, v in G.edges}

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=2200,
        alpha=0.95,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        arrows=True,
        arrowsize=20,
        width=2.0,
        connectionstyle="arc3,rad=0.05",
    )
    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=8,
        font_weight="bold",
    )

    if with_edge_labels and len(G.edges) > 0:
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_size=8,
        )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("sample_reasoning_graph.png", dpi=300, bbox_inches="tight")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def draw_graph_side_by_side(
    graph_a: Graph,
    graph_b: Graph,
    title_a: str = "Graph A",
    title_b: str = "Graph B",
    figsize: Tuple[int, int] = (16, 7),
    save_path: Optional[str] = None,
) -> None:
    """
    Quick comparison view for two graphs.
    """
    G1 = to_networkx(graph_a)
    G2 = to_networkx(graph_b)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, G, title in [(axes[0], G1, title_a), (axes[1], G2, title_b)]:
        if len(G.nodes) == 0:
            ax.set_title(f"{title} (empty)")
            ax.axis("off")
            continue

        pos = nx.spring_layout(G, seed=42)

        node_colors = [
            NODE_COLOR_MAP.get(G.nodes[n].get("node_type"), "#B0B0B0")
            for n in G.nodes
        ]
        edge_colors = [
            EDGE_COLOR_MAP.get(G.edges[e].get("edge_type"), "#7F7F7F")
            for e in G.edges
        ]
        labels = {n: G.nodes[n]["label"] for n in G.nodes}

        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=1800, alpha=0.95, ax=ax
        )
        nx.draw_networkx_edges(
            G,
            pos,
            edge_color=edge_colors,
            arrows=True,
            arrowsize=18,
            width=2.0,
            connectionstyle="arc3,rad=0.05",
            ax=ax,
        )
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    g = Graph()

    n0 = Node("Solve the equation", node_type="question", node_id=0)
    n1 = Node("Define variables", node_type="reasoning", node_id=1)
    n2 = Node("Apply formula", node_type="reasoning", node_id=2)
    n3 = Node("Compute result", node_type="result", node_id=3)

    e01 = Edge(n0, n1, weight=1.0, edge_type="leads to")
    e12 = Edge(n1, n2, weight=1.0, edge_type="leads to")
    e23 = Edge(n2, n3, weight=1.0, edge_type="leads to")

    g.add_edge(e01)
    g.add_edge(e12)
    g.add_edge(e23)

    draw_graph(g, title="Sample Reasoning Graph")