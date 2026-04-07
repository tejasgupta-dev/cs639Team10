import argparse
import json
import os
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Generate reasoning traces using local GPU.")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model path (e.g. Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--num_questions", type=int, default=50, help="Number of GSM8K questions to process")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of trajectories per question")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    return parser.parse_args()

def get_base_prompt(question):
    """Prompt format for Base models (non-instruct). Uses few-shot CoT."""
    return f"""Question: Janet has 3 rabbits. She gets 2 more. How many total?
Answer: Janet started with 3 rabbits. 
She got 2 more. 
3 + 2 = 5.
Final Answer: 5

Question: {question}
Answer:"""

def get_instruct_prompt(question, model_id):
    """Prompt format for Instruct models using chat templates."""
    messages = [
        {"role": "user", "content": f"Solve the following math problem step-by-step. End each step with two newlines (\\n\\n). Provide the final answer as 'Final Answer: <result>'.\n\nQuestion: {question}"}
    ]
    return messages

def main():
    args = parse_args()
    
    print(f"Loading dataset...")
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    questions = gsm8k["question"][:args.num_questions]
    ground_truths = gsm8k["answer"][:args.num_questions]

    print(f"Initializing vLLM for {args.model}...")
    llm = LLM(model=args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    is_instruct = "instruct" in args.model.lower() or "distill" in args.model.lower()
    
    prompts = []
    for q in questions:
        if is_instruct:
            prompt = tokenizer.apply_chat_template(get_instruct_prompt(q, args.model), tokenize=False, add_generation_prompt=True)
        else:
            prompt = get_base_prompt(q)
        prompts.append(prompt)

    sampling_params = SamplingParams(
        n=args.n_samples,
        temperature=args.temp,
        max_tokens=1024,
        stop=["Question:", "Answer:"] if not is_instruct else None
    )

    print(f"Generating {len(questions)} questions with {args.n_samples} samples each...")
    outputs = llm.generate(prompts, sampling_params)

    data = []
    for i, output in enumerate(outputs):
        trajectories = []
        for res in output.outputs:
            text = res.text.strip()
            # Basic block splitting by double newline
            steps = [s.strip() for s in text.split("\n\n") if s.strip()]
            trajectories.append(steps)
            
        data.append({
            "question": questions[i],
            "ground_truth": ground_truths[i],
            "model": args.model,
            "trajectories": trajectories
        })

    model_basename = args.model.split("/")[-1]
    output_path = f"strategy_fall/data/{model_basename}_traces.json"
    os.makedirs("strategy_fall/data", exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Done! Saved {len(data)} results to {output_path}")

if __name__ == "__main__":
    main()


# Base Model: python strategy_fall/data/generate_traces.py --model Qwen/Qwen2.5-7B --num_questions 50
# SFT Model: python strategy_fall/data/generate_traces.py --model Qwen/Qwen2.5-7B-Instruct --num_questions 50
# RL Model: python strategy_fall/data/generate_traces.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --num_questions 50