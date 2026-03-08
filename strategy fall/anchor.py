import numpy as np

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
    if method == "continue":
        return find_anchor_point_continue(question, chain, result, model, embedding_model, similarity_threshold)
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

def find_anchor_point_truncated(question,chain,result, model, embedding_model, similarity_threshold=0.99):
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

def find_anchor_point_continue(question,chain,result, model, embedding_model, similarity_threshold=0.99):
    return None
