# 📽️ Midterm Presentation: Quantifying Strategy Collapse (Full Edition)

## Slide 1: Project Title & Recap
**Goal**: Investigating the "Strategy Collapse" phenomenon where RL-trained models sacrifice logical breadth for a singular, reinforced reasoning path.

*   **Key Concept**: If a model collapses, it becomes fragile—unable to pivot if its first line of reasoning fails.
*   **Our Solution**: A custom semantic analyzer that converts unstructured reasoning chains into Quantitative Directed Acyclic Graphs (DAGs).
*   **Baselines**: Qwen2.5-Base (No tuning) and Qwen2.5-Instruct (SFT).

> [!NOTE]
> **Speaker Notes**: "We are looking at whether Reinforcement Learning makes models too narrow. To measure this, we don't just look at accuracy; we look at the 'mental map' of the model. We built a tool that takes thousands of reasoning steps and turns them into a mathematical graph so we can measure how much 'branching' is actually happening."

---

## Slide 2: Experiment Setup (The "How")
*   **The Dataset**: 1,000 Questions from GSM8K/MATH (q1000) with 10 paths each (10,000 total trajectories).
*   **Clustering Engine**: Semantic mapping of ~200,000 reasoning steps using `Sentence-Transformers` + `HDBSCAN`.
*   **Why HDBSCAN?**: It identifies logical "states" naturally without being told how many to find, and it filter out hallucinations as "Noise."

> [!NOTE]
> **Speaker Notes**: "We didn't just test 10 or 20 questions. We analyzed 10,000 reasoning paths. We used a density-based clustering algorithm called HDBSCAN. The beauty of this is that it doesn't force the model's errors into a category; it accurately labels one-off hallucinations as 'noise,' giving us a very clean look at the core logical strategies."

---

## Slide 3: Metrics & Findings
### Discovery: RL Increases Strategic Breadth
*   **Strategy Entropy (Unpredictability)**: 
    *   **RL (3.313)** vs **SFT (3.219)**.
*   **Branching Factor (Logical Flexibility)**: 
    *   **RL (3.483)** vs **SFT (3.232)**.

> [!NOTE]
> **Speaker Notes**: "Our most important finding is counter-intuitive. Everyone expected RL to collapse, but we found that RL actually **increases** diversity. The Entropy metric measures how 'unpredictable' the model is. The RL model is more diverse than the SFT model. This means RL training, when done right, encourages the model to think outside the box rather than just repeating a memorized path."

---

## Slide 4: Case Study 1 - Simple Arithmetic (Q0)
![Question 0 Comparison: SFT vs RL](file:///Users/arushitaneja/Documents/cs639-assignments/cs639Team10/strategy_fall/results/q1000/comparison_q0.png)

**The "Permutation" Pattern:**
- **SFT Behavior:** Linear, single-path logic. `16 - 3 -> 13`, `13 - 4 -> 9`.
- **RL Behavior (14% more branching):** RL models possess a "redundancy engine" that verifies arithmetic by approaching the problem from multiple grouping angles (e.g., `16 - (3+4)` vs `16-3-4`).
- **Finding:** RL doesn't just "reason"—it "explores" permutations of logic even for trivials.

> [!NOTE]
> **Speaker Notes**: "Here is Question 0. On the left (SFT), you see a very thin, linear graph. On the right (RL), notice how the model explores different ways to group the subtraction. This 'exploratory' behavior is a direct result of RL rewarding the correct answer, which forces the model to double-check its work through different logical paths."

---

## Slide 5: Case Study 2 - Complex Logic (Q500)
![Question 500 Comparison: SFT vs RL](file:///Users/arushitaneja/Documents/cs639-assignments/cs639Team10/strategy_fall/results/q1000/comparison_q500.png)

**The "Self-Correction Hub":**
- **Dynamic Branching:** Q500 triggered high-density **Self-Correction Hubs** in the RL model.
- **The "Oops" Loop:** RL paths often started with an incorrect assumption (e.g. naive subtraction), but then **converged** back to a single 'Corrective Node' once a contradiction was hit.
- **SFT Behavior:** SFT models lack these retry loops; an initial error usually cascades into a linear failure.

> [!NOTE]
> **Speaker Notes**: "Question 500 shows the real power of RL. Notice the large circles in the RL graph—we call these 'Communication Hubs.' Many different ways of thinking all converge back onto the correct equation. This shows the model's ability to self-correct mid-thought. If it hits a dead end, it has the logical 'map' to find its way back to a hub and try another path."

---

## Slide 6: Future Plan & Contribution
*   **Difficulty-Based Analysis**: Investigating if collapse happens only on trivial tasks.
*   **Math Specialist RL**: Side-by-side analysis with Qwen-Math-RL.
*   **Contribution**: Arushi (Pipeline/Versioning/Deps), Tejas (Metrics/Branching logic/EDA).

> [!NOTE]
> **Speaker Notes**: "Our next step is to see if this trend holds when the problems get even harder. We want to see if there's a 'Difficulty Threshold' where the model finally breaks. Arushi handled the heavy-lifting of the 10,000-trajectory pipeline, and Tejas developed the branching metrics we're using to quantify this."
