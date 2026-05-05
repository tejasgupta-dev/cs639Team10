import json

def build_prompt_reasoning_trace(question):
    return f"""
You are solving a reasoning problem.

Generate a concise reasoning trace for the problem.

Rules:
- Break the reasoning into atomic sentences.
- Number each sentence as S1, S2, S3, ...
- Each sentence should contain one reasoning action.
- Include setup, planning, fact retrieval, computation, checking, correction, or conclusion when needed.
- If you try a candidate, method, assumption, or case and later reject it, explicitly write the failed trial and why it fails.
- If a trial fails, end that failed branch with a sentence that clearly states it is rejected, contradicted, abandoned, or failed.
- End the successful path with the final answer.
- Do not include filler such as "let's solve this step by step" unless it performs a real planning function.
- Do not skip important dependencies.

Return only the numbered reasoning trace.

Problem:
{question}
""".strip()

def build_prompt_build_graph(question, reasoning_trace):
    return f"""
You are an expert in interpreting how LLMs solve reasoning problems using multi-step reasoning.

Your task is to analyze a reasoning trace and convert it into a Graph of Thought annotation.

The reasoning trace should be broken into atomic sentences. Each sentence becomes one node in the graph.

For each sentence, label:

1. function_tags:
One or more labels that describe what this sentence is doing functionally in the reasoning process.

2. depends_on:
A list of earlier sentence node ids that this sentence directly depends on.
Only include a dependency if the current sentence clearly uses the information, result, or logic from the earlier sentence.
Do not add dependencies merely because one sentence appears before another.

3. endpoint information:
If a sentence ends a successful answer path or a failed attempt path, mark it as an endpoint.
Every successful path and every meaningful failed trial must end with a sentence labeled final_answer_emission.

Function Tags:

1. problem_setup
Parsing or rephrasing the problem.

2. plan_generation
Stating or deciding on a plan of action, often meta-reasoning.

3. fact_retrieval
Recalling facts, formulas, rules, or problem details without immediate computation.

4. active_computation
Performing algebra, calculations, symbolic manipulation, case testing, or other manipulations toward an answer.

5. result_consolidation
Aggregating intermediate results, summarizing, combining cases, or preparing a conclusion.

6. uncertainty_management
Expressing confusion, re-evaluating, proposing alternative plans, noticing a discrepancy, or backtracking.

7. self_checking
Verifying previous steps, checking calculations, plugging results back into constraints, or reconfirming.

8. final_answer_emission
Explicitly stating the final answer of a path.
This can be either the successful final answer to the problem or the endpoint of a failed trial, rejected candidate, contradiction, or abandoned method.

Dependency Rules:

For each sentence, include a list of earlier node ids that the reasoning in this sentence directly uses.

Examples:
- If a sentence performs a computation based on a plan and a recalled formula, include both dependencies.
- If a sentence verifies a candidate answer, include the candidate answer and the relevant constraint.
- If a sentence rejects a failed trial, include the failed trial and the sentence explaining why it fails.
- If a sentence is an endpoint for a failed branch, it must depend on the sentences that justify why the branch failed.
- If there is no clear dependency, use an empty list.

Important Notes:

- Be precise and conservative.
- Include both short-range and long-range dependencies.
- Do not forget long-range dependencies.
- Make sure there is a dependency path from the problem/setup sentences to the successful final_answer_emission.
- If there is a failed trial, rejected candidate, contradiction, or abandoned method, do not leave it as a dangling branch.
- A failed trial must end with a sentence tagged final_answer_emission and endpoint_status = "failed".
- The successful final answer must also be tagged final_answer_emission and endpoint_status = "success".
- A failed final_answer_emission is not the model's returned final answer. It is the explicit endpoint of a rejected reasoning path.

Endpoint Rules:

Each endpoint node must include:
- endpoint: true
- endpoint_status: "success" or "failed"
- endpoint_type:
  - "final_correct" for the successful final answer
  - "rejected_candidate" for a candidate answer that is rejected
  - "contradiction" for a branch that ends in contradiction
  - "abandoned_method" for a method that is abandoned
  - "failed_trial" for a trial that fails but does not fit the above categories

Non-endpoint nodes must include:
- endpoint: false
- endpoint_status: null
- endpoint_type: null

Output Format:

Return ONLY valid JSON.
Do not include markdown.
Do not include explanations outside the JSON.

Use this exact schema:

{{
  "question": "...",
  "nodes": [
    {{
      "node_id": "N1",
      "step_index": 1,
      "text": "...",
      "function_tags": ["problem_setup"],
      "depends_on": [],
      "endpoint": false,
      "endpoint_status": null,
      "endpoint_type": null
    }}
  ],
  "edges": [
    {{
      "source": "N1",
      "target": "N2",
      "relation": "dependency | support | derivation | computation | aggregation | branching | verification | correction | contradiction"
    }}
  ],
  "final_answer": "..."
}}

Additional constraints:

- Each node_id must be unique.
- Node ids must be N1, N2, N3, ...
- step_index must follow the order of the reasoning trace.
- function_tags must only use the eight allowed tags.
- depends_on must only contain earlier node ids.
- Each depends_on relation should also appear as an edge with relation "dependency" unless a more specific edge relation is clearly appropriate.
- If endpoint is true, function_tags must include "final_answer_emission".
- If endpoint_status is "success", endpoint_type must be "final_correct".
- If endpoint_status is "failed", endpoint_type must be one of:
  - "rejected_candidate"
  - "contradiction"
  - "abandoned_method"
  - "failed_trial"
- There must be exactly one successful endpoint unless the problem genuinely has multiple correct final answers.
- There may be zero or more failed endpoints.
- Failed endpoints should only be included for meaningful failed trials, not for useless filler thoughts.

Problem:
{question}

Reasoning trace:
{reasoning_trace}
""".strip()

def build_prompt_reconstruct_endpoints(question, input_graph):
    graph_without_endpoints = {
        "nodes": [],
        "edges": []
    }
    endpoint_node_ids = []
    for node in input_graph["nodes"]:
        if node.get("endpoint", False) is True:
            endpoint_node_ids.append(node["node_id"])
        else:
            graph_without_endpoints["nodes"].append(node)
    for edge in input_graph["edges"]:
        if edge["source"] not in endpoint_node_ids and edge["target"] not in endpoint_node_ids:
            graph_without_endpoints["edges"].append(edge)
    return f"""
You are reconstructing endpoint conclusions from a Graph of Thought.

You are given:
1. The original question.
2. A Graph of Thought with all endpoint nodes removed.

Your task is to infer which endpoint conclusions are logically supported by the remaining graph.

Endpoint conclusions can be:

1. final_correct
The successful final answer to the original question.

2. rejected_candidate
A candidate answer or case that the graph shows should be rejected.

3. contradiction
A branch that the graph shows is logically inconsistent.

4. abandoned_method
A method or strategy that the graph shows should be abandoned.

5. failed_trial
A trial that the graph shows has failed.

Rules:
- Use only the nodes and edges in the graph without endpoints.
- Do not solve the original question from scratch.
- Do not reconstruct missing reasoning steps from outside knowledge.
- Do not assume removed endpoint nodes.
- If the remaining graph supports a final answer, output it as a final_correct endpoint.
- If the remaining graph supports a rejected candidate, contradiction, abandoned method, or failed trial, output it as a negative endpoint.
- If the remaining graph does not logically support any endpoint, output an empty reconstructed_endpoints list.
- Each reconstructed endpoint must be supported by the remaining graph.
- Do not output explanations outside the JSON.

Question:
{question}

Graph of Thought without endpoints:
{json.dumps(graph_without_endpoints, ensure_ascii=False, indent=2)}

Return ONLY valid JSON using this schema:

{{
  "reconstructed_endpoints": [
    {{
      "endpoint_text": "...",
      "endpoint_type": "final_correct | rejected_candidate | contradiction | abandoned_method | failed_trial",
      "endpoint_polarity": "positive | negative",
      "supporting_node_ids": ["N1", "N2"],
      "confidence": 0.0
    }}
  ]
}}
""".strip()

def build_prompt(question, reasoning_trace=None, input_graph=None, prompt_type="reasoning_trace"):
    if prompt_type == "reasoning_trace":
        return build_prompt_reasoning_trace(question)
    elif prompt_type == "graph_of_thought":
        return build_prompt_build_graph(question, reasoning_trace)
    elif prompt_type == "reconstruct_endpoints":
        return build_prompt_reconstruct_endpoints(question, input_graph)
    elif prompt_type == "counterfactual_resampling":
        return build_prompt_counterfactual_resampling(question, graph, target_node)
    raise ValueError(f"Invalid prompt_type: {prompt_type}")

def build_prompt_counterfactual_resampling(question, graph, target_node):
    prefix_nodes = []
    for node in graph["nodes"]:
        if node["step_index"] < target_node["step_index"]:
            prefix_nodes.append(node)
    return f"""
You are performing counterfactual resampling for a Graph of Thought.

You are given:
1. The original question.
2. The reasoning prefix before a target node.
3. The original target node.

Your task is to generate a counterfactual replacement for the target node and then continue the reasoning from that replacement until endpoint conclusions are reached.

This follows the Thought Anchors idea:
- Keep the reasoning prefix fixed.
- Replace the target reasoning step with a semantically different step.
- Continue reasoning from the replacement.
- Output the endpoint conclusions produced by this counterfactual continuation.

Rules:
- Use only the question and the fixed reasoning prefix.
- Do not copy the original target node.
- The replacement node must be semantically different from the original target node.
- The replacement node should still be a plausible next reasoning step after the prefix.
- Keep the replacement node atomic.
- Continue the reasoning after the replacement node.
- The continuation should be concise.
- If the counterfactual path reaches a correct final answer, output a final_correct endpoint.
- If the counterfactual path reaches a rejected candidate, contradiction, abandoned method, or failed trial, output a negative endpoint.
- Do not output explanations outside the JSON.

Question:
{question}

Fixed reasoning prefix:
{json.dumps(prefix_nodes, ensure_ascii=False, indent=2)}

Original target node:
{json.dumps(target_node, ensure_ascii=False, indent=2)}

Return ONLY valid JSON using this schema:

{{
  "replacement_node": {{
    "node_id": "{target_node["node_id"]}_replacement",
    "step_index": {target_node["step_index"]},
    "text": "...",
    "function_tags": ["problem_setup | plan_generation | fact_retrieval | active_computation | result_consolidation | uncertainty_management | self_checking | final_answer_emission"],
    "semantic_difference_reason": "..."
  }},
  "counterfactual_reasoning_trace": [
    {{
      "step_index": 1,
      "text": "...",
      "function_tags": ["..."]
    }}
  ],
  "reconstructed_endpoints": [
    {{
      "endpoint_text": "...",
      "endpoint_type": "final_correct | rejected_candidate | contradiction | abandoned_method | failed_trial",
      "endpoint_polarity": "positive | negative",
      "supporting_steps": [1, 2],
      "confidence": 0.0
    }}
  ]
}}
""".strip()