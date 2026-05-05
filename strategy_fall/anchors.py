import numpy as np
from numpy._core.shape_base import stack
from utils import Edge, Node, Graph

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)


def extract_final_answer(response):
    if "Final Answer:" in response:
        return response.split("Final Answer:")[-1].strip()
    return response.strip()


def find_anchor_point(question, chain, result, model, embedding_model, similarity_threshold=0.99, method="default"):
    if method == "default":
        return find_anchor_point_default(question, chain, result, model, embedding_model, similarity_threshold)
    if method == "truncated":
        return find_anchor_point_truncated(question, chain, result, model, embedding_model, similarity_threshold)
    raise ValueError(f"Unknown method: {method}")


def find_anchor_point_default(question, chain, result, model, embedding_model, similarity_threshold=0.99):
    anchor_list = []

    original_embedding = embedding_model.embed_query(result)

    for i in range(len(chain)):
        chain_i = chain[:i] + chain[i + 1:]
        steps = "\n".join(
            f"Step {idx + 1}: {step}" for idx, step in enumerate(chain_i)
        )

        prompt = f"""
            Question: {question}

            Here is a reasoning chain with one intermediate step removed:

            {steps}

            One reasoning step is missing from the original chain.
            Please recompute the reasoning based only on the remaining steps.
            Do not assume the missing step.
            Continue the reasoning step by step and then provide:

            Final Answer: <your answer>
        """.strip()

        response_i = model(prompt)
        result_i = extract_final_answer(response_i)
        embedding_i = embedding_model.embed_query(result_i)

        sim = cosine_similarity(embedding_i, original_embedding)

        if sim < similarity_threshold:
            anchor_list.append(i)

    return anchor_list

def find_anchor_point_truncated(question: str, chain: list[str], result: str, model, embedding_model, similarity_threshold=0.99):
    anchor_list = []

    original_embedding = embedding_model.embed_query(result)

    for i in range(len(chain)):
        chain_i = chain[:i]
        steps = "\n".join(
            f"Step {idx + 1}: {step}" for idx, step in enumerate(chain_i)
        )

        prompt = f"""
            Question: {question}

            Here is a truncated reasoning chain:

            {steps}

            Only partial reasoning chain is provided.
            Please recompute the reasoning based only on the remaining steps.
            Do not assume the missing step.
            Continue the reasoning step by step and then provide:

            Final Answer: <your answer>
        """.strip()

        response_i = model(prompt)
        result_i = extract_final_answer(response_i)
        embedding_i = embedding_model.embed_query(result_i)

        sim = cosine_similarity(embedding_i, original_embedding)

        if sim < similarity_threshold:
            anchor_list.append(i)

    return anchor_list

def find_anchor_point_graph(question: str, graph: Graph, result: str, model, method = "graph", embedding_model = None, similarity_threshold=0.99):
    if method == "graph":
        return find_anchor_point_by_graph_default(question, graph, result, model, embedding_model, similarity_threshold)
    if method == "truncated":
        return find_anchor_point_by_graph_truncated(question, graph, result, model, embedding_model, similarity_threshold)
    raise ValueError(f"Unknown method: {method}")

def find_anchor_point_by_graph_default(question: str, graph: Graph, result: str, model, embedding_model, similarity_threshold=0.99):
    anchor_list = []

    original_embedding = embedding_model.embed_query(result)

    for node in graph.nodes:
        new_graph = pruning(graph, node)
        node_list = new_graph.nodes
        edge_list = new_graph.edges

        node_lines = []
        for node_i in node_list:
            node_lines.append(f"N{node_i.node_id}: {node_i.context}")

        edge_lines = []
        for edge in edge_list:
            edge_type = getattr(edge, "type", None)
            if edge_type is None:
                edge_type = getattr(edge, "edge_type", "unknown")
            edge_lines.append(
                f"N{edge.source.node_id} -> N{edge.target.node_id} "
                f"(weight={edge.weight}, type={edge_type})"
            )

        node_prompt = "\n".join(node_lines)
        edge_prompt = "\n".join(edge_lines)

        prompt = f"""
        Question:
              {question}

        Reasoning Graph:

        Nodes:
              {node_prompt}

        Edges:
            {edge_prompt}

        Task:
        One node from the original reasoning graph has been removed.
        Based only on the remaining nodes and edges, continue the reasoning step by step and provide the final answer.

        Output format:
        Reasoning:
        <your reasoning>

       Final Answer:
       <your answer>
       """.strip()

        answer = model(prompt)
        result_i = extract_final_answer(answer)

        if result_i is None or result_i == "":
            anchor_list.append(node.node_id)
            continue

        embedding = embedding_model.embed_query(result_i)
        sim = cosine_similarity(embedding, original_embedding)

        if sim < similarity_threshold:
            anchor_list.append(node.node_id)

    return anchor_list

def find_anchor_point_by_graph_truncated(question: str, graph: Graph, result: str, model, embedding_model, similarity_threshold=0.99):
    anchor_list = []
    for node in graph.nodes:
        if len(node.inward_edge) == 0 and len(node.outward_edge) > 0:
            chain_list = []
            find_chain_list_by_DFS(node, set(), [], chain_list)
            for chain in chain_list:
                list = find_anchor_point_truncated(question, chain, result, model, embedding_model, similarity_threshold)
                for anchor in list:
                    anchor_list.append(anchor)
    return anchor_list

def find_chain_list_by_DFS(node: Node, visited: set, path: list, chain_list: list):
    if node.node_id in visited:
        return
    visited.add(node.node_id)
    path.append(node)
    if node.node_type == "result":
        chain_list.append(path.copy())
    else:
        for edge in node.outward_edge:
            find_chain_list_by_DFS(edge.target, visited, path, chain_list)
    path.pop()
    visited.remove(node.node_id)

def pruning(graph: Graph, node: Node):
    for inward_edge in node.inward_edge:
        for outward_edge in node.outward_edge:
            new_edge = combining_edge(inward_edge, outward_edge)
            if new_edge is not None:
                graph.add_edge(new_edge)
                graph.remove_edge(inward_edge)
                graph.remove_edge(outward_edge)
    return graph

def truncated_pruning(chain: list[str], node: Node):
    return chain[:node.node_id] + chain[node.node_id + 1:]
            

def combining_edge(edge1: Edge, edge2: Edge):
    pair = {edge1.type, edge2.type}
    type = None

    if edge1.type == "contradiction" and edge2.type == "contradiction":
        type = "equivalence"
    elif edge1.type == "equivalence" and edge2.type == "equivalence":
        type = "equivalence"
    elif edge1.type == "leads to" and edge2.type == "leads to":
        type = "leads to"
    elif pair == {"equivalence", "contradiction"}:
        type = "contradiction"
    elif pair == {"contrapositive", "equivalence"}:
        type = "contrapositive"
    elif pair == {"contrapositive", "leads to"}:
        type = "contrapositive"

    if type is None:
        return None

    new_edge = Edge(edge1.source, edge2.target, edge1.weight + edge2.weight, type)
    return new_edge