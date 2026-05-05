from prompt import build_prompt_reasoning_trace, build_prompt_build_graph, build_prompt_counterfactual_resampling
from loss import retrieve_endpoints, endpoint_similarity, training_loss, anchor_loss, structural_loss, deletion_loss, get_gated_deletion_weight
from model import GraphOfThoughtPrunerGraphSAGE
import json
import torch
import torch.nn.functional as F

from json_repair import repair_json

def prune_graph_by_probability(graph, node_keep_probability, threshold=0.5):
    keep_node_ids = []
    for index in range(len(graph["nodes"])):
        if node_keep_probability[index].item() >= threshold:
            keep_node_ids.append(graph["nodes"][index]["node_id"])
    if len(keep_node_ids) == 0 and len(graph["nodes"]) > 0:
        keep_node_ids.append(graph["nodes"][0]["node_id"])
    pruned_graph = {
        "nodes": [],
        "edges": []
    }
    for node in graph["nodes"]:
        if node["node_id"] in keep_node_ids:
            pruned_graph["nodes"].append(node)
    for edge in graph["edges"]:
        if edge["source"] in keep_node_ids and edge["target"] in keep_node_ids:
            pruned_graph["edges"].append(edge)
    return pruned_graph

def prune_graph_by_top_ratio(graph, node_keep_probability, keep_ratio=0.5):
    keep_count = max(1, int(len(graph["nodes"]) * keep_ratio))
    keep_indices = torch.topk(node_keep_probability, keep_count).indices.tolist()
    keep_node_ids = []
    for index in keep_indices:
        keep_node_ids.append(graph["nodes"][index]["node_id"])
    for node in graph["nodes"]:
        if node["node_id"] in keep_node_ids:
            for dependency in node.get("depends_on", []):
                if dependency not in keep_node_ids:
                    keep_node_ids.append(dependency)
    pruned_graph = {
        "nodes": [],
        "edges": []
    }
    for node in graph["nodes"]:
        if node["node_id"] in keep_node_ids:
            pruned_graph["nodes"].append(node)
    for edge in graph["edges"]:
        if edge["source"] in keep_node_ids and edge["target"] in keep_node_ids:
            pruned_graph["edges"].append(edge)
    return pruned_graph

def evaluate_pruning_model(model, generation_model, embedding_model, dataset, build_data_function, device=None, anchor_model=None, anchor_threshold=0.5, keep_ratios=[0.3, 0.5, 0.7], anchor_loss_threshold=0.3):
    model.eval()
    total_score = 0.0
    count = 0
    with torch.no_grad():
        for line in dataset:
            question = line["question"]
            if "graph" in line:
                graph = line["graph"]
            else:
                reasoning_trace = generation_model.generate(build_prompt_reasoning_trace(question))
                graph = parse_response(generation_model.generate(build_prompt_build_graph(question, reasoning_trace)))
            data = build_data_function(graph)
            if device is not None:
                data = data.to(device)
            node_keep_score = model(data.x, data.edge_index)
            node_keep_probability = torch.sigmoid(node_keep_score / 2.0)
            for keep_ratio in keep_ratios:
                pruned_graph = prune_graph_by_top_ratio(graph, node_keep_probability, keep_ratio)
                if "anchor_labels" in line:
                    anchors = []
                    for node in graph["nodes"]:
                        if line["anchor_labels"].get(node["node_id"], 0.0) >= anchor_threshold:
                            anchors.append(node)
                else:
                    anchors = get_anchors_from_anchor_model(anchor_model, graph, data, anchor_threshold)
                if count == 0:
                    print_loss_parts(pruned_graph, graph, question, generation_model, embedding_model, anchors)
                score = training_loss(pruned_graph, graph, question, generation_model, embedding_model, anchors)
                total_score += score
                count += 1
    if count == 0:
        return 0.0
    return total_score / count

def parse_response(response):
    if isinstance(response, dict):
        return response
    if response is None:
        raise ValueError("Empty model response")
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
        print("Invalid model response:")
        print(response)
        raise ValueError("Model response does not contain JSON")
    response = response[start:end + 1]
    try:
        return json.loads(response)
    except Exception:
        fixed_response = repair_json(response)
        return json.loads(fixed_response)

def train_anchor_model(model, generation_model, embedding_model, dataset, optimizer, build_data_function, epochs=30, sample_count=3, device=None, save_path="best_anchor_model.pt"):
    if device is not None:
        model = model.to(device)
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for line in dataset:
            question = line["question"]
            if "graph" in line:
                graph = line["graph"]
            else:
                reasoning_trace = generation_model.generate(build_prompt_reasoning_trace(question))
                graph = parse_response(generation_model.generate(build_prompt_build_graph(question, reasoning_trace)))
            if "anchor_labels" in line:
                anchor_labels = line["anchor_labels"]
            else:
                anchor_labels = find_anchors_monte_carlo(generation_model, graph, question, embedding_model, sample_count)
                line["graph"] = graph
                line["anchor_labels"] = anchor_labels
            data = build_data_function(graph, anchor_labels)
            if device is not None:
                data = data.to(device)
            optimizer.zero_grad()
            anchor_score = model(data.x, data.edge_index)
            anchor_target = data.anchor_label.float()
            loss = F.binary_cross_entropy_with_logits(anchor_score, anchor_target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        average_loss = total_loss / len(dataset)
        print("epoch:", epoch, "loss:", average_loss)
        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(model.state_dict(), save_path)

def get_anchors_from_anchor_model(anchor_model, graph, data, threshold=0.5):
    anchor_model.eval()
    with torch.no_grad():
        anchor_score = anchor_model(data.x, data.edge_index)
        anchor_probability = torch.sigmoid(anchor_score)
    anchors = []
    for index in range(len(graph["nodes"])):
        if anchor_probability[index].item() >= threshold:
            anchors.append(graph["nodes"][index])
    return anchors

def find_anchors_monte_carlo(generation_model, graph, question, embedding_model, sample_count=100):
    endpoints = retrieve_endpoints(graph)
    anchor_labels = {}
    for vertex in graph["nodes"]:
        similarity_sum = 0.0
        for i in range(sample_count):
            prompt = build_prompt_counterfactual_resampling(question, graph, vertex)
            response = generation_model.generate(prompt)
            similarity_sum += endpoint_similarity(response, endpoints, embedding_model)
        average_similarity = similarity_sum / sample_count
        anchor_score = 1.0 - average_similarity
        anchor_labels[vertex["node_id"]] = anchor_score
    return anchor_labels

def prune_graph_by_mask(graph, keep_mask):
    keep_node_ids = []
    for index in range(len(graph["nodes"])):
        if keep_mask[index].item() == 1:
            keep_node_ids.append(graph["nodes"][index]["node_id"])
    if len(keep_node_ids) == 0 and len(graph["nodes"]) > 0:
        keep_node_ids.append(graph["nodes"][0]["node_id"])
    pruned_graph = {
        "nodes": [],
        "edges": []
    }
    for node in graph["nodes"]:
        if node["node_id"] in keep_node_ids:
            pruned_graph["nodes"].append(node)
    for edge in graph["edges"]:
        if edge["source"] in keep_node_ids and edge["target"] in keep_node_ids:
            pruned_graph["edges"].append(edge)
    return pruned_graph

def train(model, generation_model, embedding_model, dataset, optimizer, build_data_function, epochs=100, device=None, anchor_model=None, anchor_threshold=0.5, keep_ratios=[0.3, 0.5, 0.7], save_path="best_pruning_model.pt", anchor_loss_threshold=0.3):
    if device is not None:
        model = model.to(device)
        if anchor_model is not None:
            anchor_model = anchor_model.to(device)
    best_score = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for line in dataset:
            question = line["question"]
            if "graph" in line:
                graph = line["graph"]
            else:
                reasoning_trace = generation_model.generate(build_prompt_reasoning_trace(question))
                graph = parse_response(generation_model.generate(build_prompt_build_graph(question, reasoning_trace)))
                line["graph"] = graph
            data = build_data_function(graph)
            if device is not None:
                data = data.to(device)
            optimizer.zero_grad()
            node_keep_score = model(data.x, data.edge_index)
            node_keep_probability = torch.sigmoid(node_keep_score)
            distribution = torch.distributions.Bernoulli(node_keep_probability)
            keep_mask = distribution.sample()
            log_probability = distribution.log_prob(keep_mask).sum()
            pruned_graph = prune_graph_by_mask(graph, keep_mask)
            current_anchor_loss = anchor_loss(pruned_graph, graph, question, generation_model, embedding_model)
            if current_anchor_loss > anchor_loss_threshold and "anchor_labels" in line:
                anchors = []
                for node in graph["nodes"]:
                    if line["anchor_labels"].get(node["node_id"], 0.0) >= anchor_threshold:
                        anchors.append(node)
            else:
                anchors = get_anchors_from_anchor_model(anchor_model, graph, data, anchor_threshold)
            graph_loss_value = training_loss(pruned_graph, graph, question, generation_model, embedding_model, anchors)
            graph_loss = torch.tensor(graph_loss_value, dtype=torch.float32, device=node_keep_score.device)
            loss = graph_loss * log_probability
            loss.backward()
            optimizer.step()
            total_loss += float(graph_loss.detach())
        average_loss = total_loss / len(dataset)
        evaluation_score = evaluate_pruning_model(model, generation_model, embedding_model, dataset, build_data_function, device, anchor_model, anchor_threshold, keep_ratios, anchor_loss_threshold)
        print("epoch:", epoch, "loss:", average_loss, "evaluation_score:", evaluation_score)
        if evaluation_score < best_score:
            best_score = evaluation_score
            torch.save(model.state_dict(), save_path)

def print_loss_parts(pruned_graph, original_graph, question, generation_model, embedding_model, anchors):
    anchor_loss_value = anchor_loss(pruned_graph, original_graph, question, generation_model, embedding_model)
    structural_loss_value = structural_loss(pruned_graph, original_graph, anchors)
    deletion_loss_value = deletion_loss(pruned_graph, original_graph)
    preservation_loss = anchor_loss_value + structural_loss_value
    deletion_weight = get_gated_deletion_weight(preservation_loss)
    total_loss = anchor_loss_value + structural_loss_value + deletion_weight * deletion_loss_value
    print("anchor_loss:", anchor_loss_value)
    print("structural_loss:", structural_loss_value)
    print("deletion_loss:", deletion_loss_value)
    print("deletion_weight:", deletion_weight)
    print("total_loss:", total_loss)
    print("kept_nodes:", len(pruned_graph["nodes"]), "/", len(original_graph["nodes"]))
    print("kept_edges:", len(pruned_graph["edges"]), "/", len(original_graph["edges"]))