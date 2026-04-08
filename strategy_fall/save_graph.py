import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from graphviz import Digraph
from utils import Node, Edge, Graph

class GraphSerializer:
    @staticmethod
    def node_to_dict(node) -> Dict[str, Any]:
        return {
            "node_id": node.node_id,
            "context": node.context,
            "node_type": node.node_type,
            "degree": node.degree,
        }

    @staticmethod
    def edge_to_dict(edge) -> Dict[str, Any]:
        edge_type = getattr(edge, "edge_type", None)
        if edge_type is None:
            edge_type = getattr(edge, "type", None)

        return {
            "source": edge.source.node_id,
            "target": edge.target.node_id,
            "weight": edge.weight,
            "edge_type": edge_type,
        }

    @staticmethod
    def graph_to_dict(graph, graph_name: Optional[str] = None) -> Dict[str, Any]:
        return {
            "graph_name": graph_name,
            "nodes": [GraphSerializer.node_to_dict(node) for node in graph.nodes],
            "edges": [GraphSerializer.edge_to_dict(edge) for edge in graph.edges],
        }

    @staticmethod
    def save_graph_json(graph, filepath: str, graph_name: Optional[str] = None) -> None:
        data = GraphSerializer.graph_to_dict(graph, graph_name=graph_name)
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def save_graphs_json(
        graphs: List[Graph],
        filepath: str,
        graph_names: Optional[List[str]] = None
    ) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        if graph_names is None:
            graph_names = [f"graph_{i}" for i in range(len(graphs))]

        data = {
            "graphs": [
                GraphSerializer.graph_to_dict(graph, graph_name=name)
                for graph, name in zip(graphs, graph_names)
            ]
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def save_graph_dot(graph, filepath: str, graph_name: str = "ReasoningGraph") -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [f'digraph "{graph_name}" {{']

        for node in graph.nodes:
            label = str(node.context).replace('"', '\\"')
            node_type = node.node_type if node.node_type is not None else ""
            lines.append(
                f'  N{node.node_id} [label="{label}\\n(type={node_type})"];'
            )

        for edge in graph.edges:
            edge_type = getattr(edge, "edge_type", None)
            if edge_type is None:
                edge_type = getattr(edge, "type", "unknown")

            lines.append(
                f'  N{edge.source.node_id} -> N{edge.target.node_id} '
                f'[label="w={edge.weight}, type={edge_type}"];'
            )

        lines.append("}")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    @staticmethod
    def save_graph_image(
        graph,
        filepath: str,
        graph_name: str = "ReasoningGraph",
        image_format: str = "png"
    ) -> None:
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        dot = Digraph(name=graph_name, format=image_format)
        dot.attr(rankdir="TB")

        for node in graph.nodes:
            node_id = f"N{node.node_id}"
            label_context = str(node.context).replace("\n", "\\n")
            node_type = node.node_type if node.node_type is not None else ""
            label = f"{label_context}\\n(type={node_type})"
            dot.node(node_id, label=label)

        for edge in graph.edges:
            edge_type = getattr(edge, "edge_type", None)
            if edge_type is None:
                edge_type = getattr(edge, "type", "unknown")

            dot.edge(
                f"N{edge.source.node_id}",
                f"N{edge.target.node_id}",
                label=f"w={edge.weight}, type={edge_type}"
            )

        dot.render(str(path), cleanup=True)

    @staticmethod
    def save_all_formats(
        graph,
        base_filepath: str,
        graph_name: str = "ReasoningGraph"
    ) -> None:
        GraphSerializer.save_graph_json(
            graph,
            f"{base_filepath}.json",
            graph_name=graph_name
        )
        GraphSerializer.save_graph_dot(
            graph,
            f"{base_filepath}.dot",
            graph_name=graph_name
        )
        try:
            GraphSerializer.save_graph_image(
                graph,
                base_filepath,
                graph_name=graph_name,
                image_format="png"
            )
        except Exception:
            print("Warning: Graphviz image generation failed. Check if 'dot' is installed.")
