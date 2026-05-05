import argparse
import json
import os
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# MATH-500 subjects for targeted filtering
MATH_SUBJECTS = [
    "Algebra", "Counting & Probability", "Geometry",
    "Intermediate Algebra", "Number Theory", "Prealgebra", "Precalculus",
]

def parse_args():
    parser = argparse.ArgumentParser(description="Generate reasoning traces using local GPU.")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model path (e.g. Qwen/Qwen2.5-7B-Instruct-AWQ)")
    parser.add_argument("--num_questions", type=int, default=50, help="Number of questions to process")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math"], help="Dataset to use")
    parser.add_argument("--math_level", type=int, default=5, help="Difficulty level for MATH dataset (1-5)")
    parser.add_argument("--math_subject", type=str, default=None, choices=MATH_SUBJECTS,
                        help="(Optional) Filter MATH-500 by subject area for targeted analysis")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of trajectories per question")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    # Memory management — defaults raised for harder (longer) problems
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="Fraction of GPU memory to reserve")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="Max context length; Level 5 MATH solutions need ~3-4k tokens")
    parser.add_argument("--quantization", type=str, default=None,
                        choices=["awq", "gptq", "squeezellm", "compressed-tensors", None],
                        help="Quantization method")
    return parser.parse_args()

def get_base_prompt(question):
    """Prompt format for Base models (non-instruct). Uses few-shot CoT."""
    return f"""Question: Janet has 3 rabbits. She gets 2 more. How many total?
Answer: Janet started with 3 rabbits. \nShe got 2 more. \n3 + 2 = 5.\nFinal Answer: 5\n\nQuestion: {question}\nAnswer:"""

def get_instruct_prompt(question, model_id):
    """Prompt format for Instruct models using chat templates."""
    messages = [
        {"role": "user", "content": f"Solve the following math problem step-by-step. End each step with two newlines (\\n\\n). Provide the final answer as 'Final Answer: #### <result>' or '\\boxed{{<result>}}'.\n\nQuestion: {question}"}
    ]
    return messages

def main():
    args = parse_args()
    
    print(f"Loading dataset {args.dataset}...")
    if args.dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
        questions = ds["question"][:args.num_questions]
        ground_truths = ds["answer"][:args.num_questions]
        subjects = [None] * len(questions)
    else:
        # Load MATH-500 dataset (reliable subset used for R1 evaluation)
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        # Filter by level (required) and optionally by subject
        filtered = [item for item in ds if str(item['level']) == f"Level {args.math_level}"]
        if not filtered:
            # Fallback: some splits store level as plain integer string
            filtered = [item for item in ds if str(item.get('level', '')) == str(args.math_level)]
        if args.math_subject:
            filtered = [item for item in filtered if item.get('type', '') == args.math_subject]
            print(f"Subject filter '{args.math_subject}' applied — {len(filtered)} problems remain.")
        if not filtered:
            raise ValueError(
                f"No MATH-500 problems found for level={args.math_level}"
                + (f", subject={args.math_subject}" if args.math_subject else "")
                + ". Check dataset structure with --math_level and --math_subject."
            )
        filtered = filtered[:args.num_questions]
        questions    = [item['problem']  for item in filtered]
        ground_truths = [item['solution'] for item in filtered]
        subjects     = [item.get('type', 'Unknown') for item in filtered]

    print(f"Loaded {len(questions)} questions.")

    print(f"Initializing vLLM for {args.model}...")
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        quantization=args.quantization,
        dtype="float16",   # RTX 2080 Ti does not support bfloat16
        enforce_eager=True,
    )
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
        max_tokens=args.max_model_len,
        stop=["Question:", "Answer:"] if not is_instruct else None
    )

    print(f"Generating {len(questions)} questions with {args.n_samples} samples each...")
    outputs = llm.generate(prompts, sampling_params)

    data = []
    for i, output in enumerate(outputs):
        trajectories = []
        for res in output.outputs:
            text = res.text.strip()
            # Split on double-newline blocks; for long RL traces also split on <think>/<\think>
            steps = [s.strip() for s in text.split("\n\n") if s.strip()]
            trajectories.append(steps)

        entry = {
            "question": questions[i],
            "ground_truth": ground_truths[i],
            "model": args.model,
            "trajectories": trajectories,
            # Metadata for downstream analysis
            "dataset": args.dataset,
        }
        if args.dataset == "math":
            entry["math_level"] = args.math_level
            entry["math_subject"] = subjects[i]
        data.append(entry)

    model_basename = args.model.split("/")[-1]
    output_dir = "strategy_fall/data"
    os.makedirs(output_dir, exist_ok=True)

    # Build a version tag that embeds the level for MATH so the pipeline
    # version string (e.g. "math_l5") matches the filename suffix.
    if args.dataset == "math":
        version_tag = f"math_l{args.math_level}"
    else:
        version_tag = args.dataset   # "gsm8k"

    output_path = os.path.join(output_dir, f"{model_basename}_traces-{version_tag}.json")

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Done! Saved {len(data)} results to {output_path}")
    print(f"  → version tag for pipeline: '{version_tag}'")
    print(f"  → run analysis with:  bash strategy_fall/run_analysis.sh {version_tag}")

if __name__ == "__main__":
    main()