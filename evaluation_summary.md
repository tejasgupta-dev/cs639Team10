# Evaluation Summary

_Generated: 2026-05-05 22:31_

## Structural metrics — GSM8K (q1000)

| model | mean_strategy_entropy | mean_branching_factor | planning_intensity | uncertainty_intensity | avg_unique_paths |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B-Instruct-AWQ_traces-q1000 | 3.213 | 3.239 | 0.0564 | 0.137 | 9.521 |
| DeepSeek-R1-Distill-Qwen-7B-Floppanacci-AWQ_traces-q1000 | 3.313 | 3.541 | 0.0731 | 0.1663 | 9.956 |
| qwen2.5-7b-awq-4bit_traces-q1000 | 1.749 | 1.625 | 0.051 | 0.1592 | 4.795 |

**Deltas:**

- RL - SFT:  mean_strategy_entropy +0.1000  mean_branching_factor +0.3020  planning_intensity +0.0167  uncertainty_intensity +0.0293  avg_unique_paths +0.4350
- RL - base:  mean_strategy_entropy +1.5640  mean_branching_factor +1.9160  planning_intensity +0.0221  uncertainty_intensity +0.0071  avg_unique_paths +5.1610
- SFT - base:  mean_strategy_entropy +1.4640  mean_branching_factor +1.6140  planning_intensity +0.0054  uncertainty_intensity -0.0222  avg_unique_paths +4.7260

## Structural metrics — MATH Level 5

| model | mean_strategy_entropy | mean_branching_factor | planning_intensity | uncertainty_intensity | avg_unique_paths |
| --- | --- | --- | --- | --- | --- |
| DeepSeek-R1-Distill-Qwen-7B-Floppanacci-AWQ_traces-math_l5 | 3.322 | 4.463 | 0.0268 | 0.0593 | 10.0 |
| Qwen2.5-7B-Instruct-AWQ_traces-math_l5 | 3.31 | 3.151 | 0.0463 | 0.116 | 9.94 |

**Deltas:**

- RL - SFT:  mean_strategy_entropy +0.0120  mean_branching_factor +1.3120  planning_intensity -0.0195  uncertainty_intensity -0.0567  avg_unique_paths +0.0600

## Structural metrics — Depth experiment

| model | mean_strategy_entropy | mean_branching_factor | avg_unique_paths |
| --- | --- | --- | --- |
| DeepSeek-depth | 5.626 | 5.467 | 49.64 |
| Qwen-Instruct-depth | 5.476 | 4.472 | 46.82 |

**Deltas:**

- RL - SFT:  mean_strategy_entropy +0.1500  mean_branching_factor +0.9950  avg_unique_paths +2.8200

_(different cluster config — values not comparable to q1000/math_l5)_

## Causal — GSM8K, RL

| tag | control | intervention | drop |
| --- | --- | --- | --- |
| Conclusion | 0.7375 | 0.7562 | -0.0187 |
| Other | 0.5714 | 0.5089 | 0.0625 |
| Planning | 0.6562 | 0.6979 | -0.0417 |
| Uncertainty Management | 0.537 | 0.5278 | 0.0093 |

_n_questions = 50_

## Causal — GSM8K, SFT

| tag | control | intervention | drop |
| --- | --- | --- | --- |
| Conclusion | 0.0 | 0.125 | -0.125 |
| Other | 0.0331 | 0.0294 | 0.0037 |
| Uncertainty Management | 0.0175 | 0.055 | -0.0375 |

_n_questions = 50_

**Note:** SFT-GSM8K outputs use 'Final Answer:' without '####' / '\boxed{}'; absolute accuracies reflect parser limitation, drops still informative.

## Causal — MATH L5, RL

| tag | control | intervention | drop |
| --- | --- | --- | --- |
| Conclusion | 0.625 | 0.5 | 0.125 |
| Uncertainty Management | 0.2734 | 0.2969 | -0.0234 |

_n_questions = 50_

## Causal — MATH L5, SFT

| tag | control | intervention | drop |
| --- | --- | --- | --- |
| Active Computation | 0.0 | 0.0 | 0.0 |
| Conclusion | 0.2812 | 0.25 | 0.0312 |
| Uncertainty Management | 0.3819 | 0.3389 | 0.0431 |

_n_questions = 50_

## Files skipped (legacy / superseded)

- `strategy_fall/results/strategy_collapse_report.csv` — older 4-col schema; superseded by q1000/strategy_collapse_report_q1000.csv
- `strategy_fall/results/Qwen2.5-7B-Instruct-AWQ_details.csv` — 50-row precursor; superseded by q1000/Qwen2.5-7B-Instruct-AWQ_traces-q1000_details.csv
- `strategy_fall/results/DeepSeek-R1-Distill-Qwen-7B-Floppanacci-AWQ_details.csv` — 50-row precursor; superseded by q1000/DeepSeek-R1-Distill-Qwen-7B-Floppanacci-AWQ_traces-q1000_details.csv
- `strategy_fall/results/qwen2.5-7b-awq-4bit_details.csv` — 50-row precursor with empty migration cols; superseded by q1000/qwen2.5-7b-awq-4bit_traces-q1000_details.csv
