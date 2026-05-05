import json
import numpy as np
from json_repair import repair_json
from prompt import build_prompt_reconstruct_endpoints

def cosine_similarity(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))

def retrieve_endpoints(graph):
    endpoints = []
    for node in graph["nodes"]:
        if node.get("endpoint", False) is True:
            endpoints.append(node)
    return endpoints

def retrieve_sources(graph):
    sources = []
    for node in graph["nodes"]:
        if node["depends_on"] == []:
            sources.append(node)
    return sources

def dfs(graph, node, visited):
    visited.add(node)
    for edge in graph["edges"]:
        if edge["source"] == node:
            if edge["target"] not in visited:
                dfs(graph, edge["target"], visited)
    return visited

def connectivity_loss(input_graph, anchors, sources, endpoints):
    loss = 0.0
    count = 0
    for anchor in anchors:
        for source in sources:
            count += 1
            if anchor["node_id"] not in dfs(input_graph, source["node_id"], set()):
                loss += 1.0
        for endpoint in endpoints:
            count += 1
            if endpoint["node_id"] not in dfs(input_graph, anchor["node_id"], set()):
                loss += 1.0
    if count == 0:
        return 0.0
    return loss / count

def premise_loss(input_graph, original_graph, anchors):
    loss = 0.0
    count = 0
    input_node_ids = []
    for node in input_graph["nodes"]:
        input_node_ids.append(node["node_id"])
    for anchor in anchors:
        for edge in original_graph["edges"]:
            if edge["target"] == anchor["node_id"]:
                count += 1
                if edge["source"] not in input_node_ids:
                    loss += 1.0
    if count == 0:
        return 0.0
    return loss / count

def endpoint_similarity(response, expected_endpoints, embedding_model):
    if isinstance(response, str):
        response = response.strip()
        if response.startswith("```json"):
            response = response[len("```json"):].strip()
        if response.startswith("```"):
            response = response[len("```"):].strip()
        if response.endswith("```"):
            response = response[:-3].strip()
        start = response.find("{")
        end = response.rfind("}")
        if start == -1 or end == -1 or end <= start:
            response = {"reconstructed_endpoints": []}
        else:
            response = response[start:end + 1]
            try:
                response = json.loads(response)
            except Exception:
                response = json.loads(repair_json(response))
    response_endpoints = response.get("reconstructed_endpoints", [])
    if len(expected_endpoints) == 0:
        return 1.0
    if len(response_endpoints) == 0:
        return 0.0
    similarity_list = []
    for expected_endpoint in expected_endpoints:
        expected_polarity = None
        if expected_endpoint.get("endpoint_status") == "success":
            expected_polarity = "positive"
        if expected_endpoint.get("endpoint_status") == "failed":
            expected_polarity = "negative"
        best_similarity = 0.0
        for response_endpoint in response_endpoints:
            if expected_polarity != response_endpoint.get("endpoint_polarity"):
                similarity = 0.0
            elif expected_endpoint.get("endpoint_type") != response_endpoint.get("endpoint_type"):
                similarity = 0.0
            else:
                response_embedding = embedding_model.encode(response_endpoint.get("endpoint_text", ""))
                expected_embedding = embedding_model.encode(expected_endpoint.get("text", ""))
                similarity = cosine_similarity(response_embedding, expected_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
        similarity_list.append(best_similarity)
    return float(sum(similarity_list) / len(similarity_list))

def anchor_loss(input_graph, original_graph, question, model, embedding_model):
    prompt = build_prompt_reconstruct_endpoints(question, input_graph)
    response = model.generate(prompt)
    expected_response = retrieve_endpoints(original_graph)
    return 1.0 - endpoint_similarity(response, expected_response, embedding_model)

def structural_loss(input_graph, original_graph, anchors):
    sources = retrieve_sources(original_graph)
    endpoints = retrieve_endpoints(original_graph)
    connectivity_loss_value = connectivity_loss(input_graph, anchors, sources, endpoints)
    premise_loss_value = premise_loss(input_graph, original_graph, anchors)
    return (connectivity_loss_value + premise_loss_value) / 2

def deletion_loss(input_graph, original_graph):
    input_node_ids = []
    for node in input_graph["nodes"]:
        input_node_ids.append(node["node_id"])
    kept_node_count = 0
    kept_edge_count = 0
    for node in original_graph["nodes"]:
        if node["node_id"] in input_node_ids:
            kept_node_count += 1
    for edge in original_graph["edges"]:
        if edge["source"] in input_node_ids and edge["target"] in input_node_ids:
            kept_edge_count += 1
    node_loss = 0.0
    edge_loss = 0.0
    if len(original_graph["nodes"]) > 0:
        node_loss = kept_node_count / len(original_graph["nodes"])
    if len(original_graph["edges"]) > 0:
        edge_loss = kept_edge_count / len(original_graph["edges"])
    return (node_loss + edge_loss) / 2

# def get_gated_deletion_weight(preservation_loss, threshold=0.3, sharpness=10.0, max_deletion_weight=1.0):
#     gate = 1.0 / (1.0 + np.exp(sharpness * (preservation_loss - threshold)))
#     return max_deletion_weight * gate

def get_gated_deletion_weight(preservation_loss):
    threshold = 0.75
    sharpness = 8.0
    min_weight = 0.05
    weight = 1.0 / (1.0 + np.exp(sharpness * (preservation_loss - threshold)))
    return min_weight + (1.0 - min_weight) * weight

def training_loss(input_graph, original_graph, question, model, embedding_model, anchors):
    structural_loss_value = structural_loss(input_graph, original_graph, anchors)
    anchor_loss_value = anchor_loss(input_graph, original_graph, question, model, embedding_model)
    deletion_loss_value = deletion_loss(input_graph, original_graph)
    preservation_loss = structural_loss_value + anchor_loss_value
    weight = get_gated_deletion_weight(preservation_loss)
    return anchor_loss_value + structural_loss_value + weight * deletion_loss_value