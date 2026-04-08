from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from utils import Graph, Node, Edge


def to_networkx(graph: Graph) -> nx.DiGraph:
    """
    Convert custom Graph object into a NetworkX directed graph.
    """
    G = nx.DiGraph()

    for node in graph.nodes:
        G.add_node(
            node.node_id,
            label=_node_label(node),
            context=getattr(node, "context", ""),
        )

    for edge in graph.edges:
        G.add_edge(
            edge.source.node_id,
            edge.target.node_id,
            weight=getattr(edge, "weight", 0),
        )

    return G


def _node_label(node: Node, max_len: int = 30) -> str:
    prefix = f"N{node.node_id}"
    text = (node.context or "").strip().replace("\n", " ")
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return f"{prefix}\n{text}" if text else prefix


def get_hierarchical_pos(G: nx.DiGraph) -> Dict:
    """
    Constructs a hierarchical (Top-to-Bottom) layout based on topological depth.
    If the graph has cycles (rare in reasoning), falls back to spring_layout.
    """
    try:
        # Assign 'layers' based on topological order
        levels = {}
        for node in nx.topological_sort(G):
            # level is 1 + max level of predecessors
            preds = list(G.predecessors(node))
            levels[node] = 1 + max([levels[p] for p in preds], default=0)
        
        # multipartite_layout uses 'subset' attribute
        for node, val in levels.items():
            G.nodes[node]['subset'] = val
            
        pos = nx.multipartite_layout(G, subset_key='subset', align='horizontal')
        # Flip Y to ensure top-to-bottom
        for node in pos:
            pos[node][1] *= -1
        return pos
    except nx.NetworkXUnfeasible:
        # Fallback for cycles (shouldn't happen in DAGs)
        return nx.spring_layout(G, seed=42)


def draw_fancy_graph(
    graph: Graph,
    title: str = "Reasoning Strategy Map",
    ax=None,
    save_path: Optional[str] = None,
) -> None:
    """
    A lush, colorful version of the reasoning graph.
    """
    G = to_networkx(graph)
    if len(G.nodes) == 0:
        return

    pos = get_hierarchical_pos(G)
    
    node_weights = np.array([G.degree(n, weight='weight') for n in G.nodes])
    if node_weights.max() > 0:
        node_sizes = 500 + 3000 * (node_weights / node_weights.max())
    else:
        node_sizes = [1500] * len(G.nodes)

    # Node color: distance from start (Topological level)
    levels = {n: pos[n][1] for n in G.nodes}
    min_l, max_l = min(levels.values()), max(levels.values())
    if max_l != min_l:
        colors = [(levels[n] - min_l) / (max_l - min_l) for n in G.nodes]
    else:
        colors = [0.5] * len(G.nodes)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
        
    # Draw Edges with varying thickness and transparency
    edge_weights = [G.edges[e]['weight'] for e in G.edges]
    max_ew = max(edge_weights) if edge_weights else 1
    edge_widths = [1 + 5 * (w / max_ew) for w in edge_weights]
    
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        edge_color="#A9A9A9",
        alpha=0.4,
        arrows=True,
        arrowsize=15,
        connectionstyle="arc3,rad=0.1"
    )

    # Draw Nodes with a lush color map (Viridis or Plasma)
    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=colors,
        cmap=plt.cm.Spectral_r,
        alpha=0.9,
        edgecolors="white",
        linewidths=1.5
    )

    # Draw Labels
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels={n: G.nodes[n]['label'] for n in G.nodes},
        font_size=7,
        font_weight="bold"
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.axis("off")

    if save_path and ax is None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=False)


def draw_graph_side_by_side(
    graph_a: Graph,
    graph_b: Graph,
    title_a: str = "Graph A",
    title_b: str = "Graph B",
    figsize: Tuple[int, int] = (20, 10),
    save_path: Optional[str] = None,
) -> None:
    """
    Generates a high-resolution side-by-side comparison of two strategy graphs.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor="#F8F9FA")
    
    draw_fancy_graph(graph_a, title=title_a, ax=axes[0])
    draw_fancy_graph(graph_b, title=title_b, ax=axes[1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved fancy comparison to: {save_path}")