# Evaluation Summary

_Generated: 2026-05-01 15:17_

## Structural metrics — GSM8K (q1000)

| model | mean_strategy_entropy | mean_branching_factor | planning_intensity | uncertainty_intensity | avg_unique_paths |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B-Instruct-AWQ_traces-q1000 | 3.213 | 3.239 | 0.0564 | 0.137 | 9.521 |
| DeepSeek-R1-Distill-Qwen-7B-Floppanacci-AWQ_traces-q1000 | 3.313 | 3.541 | 0.0731 | 0.1663 | 9.956 |
| qwen2.5-7b-awq-4bit_traces-q1000 | 1.749 | 1.625 | 0.051 | 0.1592 | 4.795 |

## Structural metrics — MATH Level 5

| model | mean_strategy_entropy | mean_branching_factor | planning_intensity | uncertainty_intensity | avg_unique_paths |
| --- | --- | --- | --- | --- | --- |
| DeepSeek-R1-Distill-Qwen-7B-Floppanacci-AWQ_traces-math_l5 | 3.322 | 4.463 | 0.0268 | 0.0593 | 10.0 |
| Qwen2.5-7B-Instruct-AWQ_traces-math_l5 | 3.31 | 3.151 | 0.0463 | 0.116 | 9.94 |

## Structural metrics — Depth experiment

| model | mean_strategy_entropy | mean_branching_factor | avg_unique_paths |
| --- | --- | --- | --- |
| DeepSeek-depth | 5.626 | 5.467 | 49.64 |
| Qwen-Instruct-depth | 5.476 | 4.472 | 46.82 |

_(different cluster config — values not comparable to q1000/math_l5)_
