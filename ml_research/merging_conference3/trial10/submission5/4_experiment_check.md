# Experimental Evaluation and Scientific Hygiene Check

The experimental evaluation and scientific hygiene in this paper are outstanding and set a very high bar for empirical research. However, there are significant limitations in terms of scale and practical significance that require a rigorous critique.

## 1. Critique of Experimental Scale and Practical Significance
- **Small-Scale Real-World Setup:** The "real-world" multi-task text classification benchmark is conducted using the classical `20newsgroups` dataset, which is a shallow document classification task. 
- **Architectural Simplicity:** The document representation is built on a simple TF-IDF vectorizer (max features $D=1024$), and the "expert adapters" are represented by simple, shallow 2-layer Multi-Layer Perceptrons (MLPs) with 128 hidden units. 
- **Distance from LLM Serving Vision:** While the authors position this work as a solution for dynamic test-time ensembling of lightweight parameter-efficient fine-tuning (PEFT) expert adapters (e.g., LoRA) in Large Language Models (LLMs), the actual empirical evaluation does not use any deep pre-trained models or standard LLM architectures (such as LLaMA, Mistral, or BERT). A shallow MLP on top of static TF-IDF features is a poor proxy for token-by-token autoregressive decoding sequences. The lack of evaluations on standard PEFT benchmarks (like GLUE subcategories, GSM8K, or MATH) under continuous streaming severely limits the practical significance of the empirical results.

## 2. Marginal Gains over Stateless Baselines in the Sandbox
- **Small Accuracy Margin:** In the synthetic Analytical Coordinate Sandbox (ICS), standard SABLE (Stateless, $\tau=0.005$) achieves an accuracy of **74.74%**, which is virtually identical to UGR's **75.08%** (a marginal absolute improvement of only **+0.34%**).
- **Complexity-Performance Trade-off:** SABLE is completely stateless, requiring zero history tracking or spatial-temporal boundary coupling. In contrast, UGR's spatial-temporal coupling introduces a significant boundary shock at layer 4 (with a massive Jitter $L \ge 4$ of **1068.25** compared to SABLE's **889.52**). Under highly dynamic, fast-switching streams (where task transitions occur frequently), the marginal +0.34% accuracy gain of UGR may not justify the added complexity of state tracking and the accompanying boundary shock.

## 3. Scientific Hygiene and Baseline Auditing
Despite the scale limitations, the paper displays an exceptional level of scientific honesty and hygiene:
- **Auditing the Momentum-Merge Baseline:** The authors conducted a code-level audit of Momentum-Merge and discovered a biased initialization discrepancy in its uncoupled configuration. They corrected this discrepancy and re-evaluated it fairly, raising the bar for empirical integrity.
- **Auditing ChemMerge (ODE Numerical Limits):** They analyze ChemMerge's integration step sizes, showing why higher-order solvers (Heun or RK4) multiply latency, highlighting UGR's closed-form advantage.
- **Decomposed Jitter Analysis:** They decompose jitter into Intra-Task Jitter (stability) and Inter-Task Jitter (agility) over a block-structured stream, proving that UGR has a massive 1.8$\times$ separation between stability and plasticity, resolving any apparent jitter gaps.

## 4. Hardware Benchmarking
The authors include a detailed wall-clock timing benchmark on an Intel Xeon Platinum CPU under synchronized sequential execution. This provides highly practical throughput (QPS) and latency measurements, confirming that the Softmax-Free Target variant is faster than ChemMerge while delivering massive accuracy boosts.

## Experimental Conclusion
The empirical validation is conducted with outstanding scientific hygiene and rigorous baseline auditing. However, the practical significance is heavily bottlenecked by the small scale of the "real-world" benchmark (shallow MLPs on TF-IDF features), and the performance gains over simple stateless methods (SABLE) in the synthetic sandbox are remarkably marginal, suggesting that the benefits of stateful coupling are highly workload-dependent.
