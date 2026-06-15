# 4. Experimental Setup and Result Validation

This section critically evaluates the experimental design and conducts a rigorous examination of the quantitative results presented in the paper and the accompanying JSON metrics.

## Baselines and Experimental Design
The authors compare SVS against a reasonable set of baselines: Zero-Shot Base CLIP, Individual Experts, Task Arithmetic (TA), TIES-Merging, and DARE. 
However, the experimental design is highly limited:
- **Single Architecture:** Only one small vision-language model, CLIP-ViT-B/32 (86M parameters), is evaluated. Given that model merging is primarily utilized and researched in the context of multi-billion parameter autoregressive Large Language Models (LLMs), restricting the evaluation to a small vision model limits the generalizability of the findings.
- **Subsampled Test Sets:** The authors evaluate on a subset of only 1,000 samples per dataset. This subsampling is mathematically unnecessary and increases statistical variance, making small accuracy deltas suspect.

## Do the Results Support the Claims?

A rigorous examination of the quantitative results reveals that many of the paper's core claims are unsupported, over-stated, or undermined by the authors' own data.

### 1. "SVS Matches or Exceeds Task Arithmetic" (Overstated and Non-Monotonic Scaling)
The authors emphasize that SVS with rank $k=128$ ($74.83\%$) "strictly outperforms" Task Arithmetic ($74.78\%$). 
- This "outperformance" is a marginal delta of **0.05%**, which is statistically insignificant and well within the noise margin of a 1,000-sample evaluation.
- More importantly, looking at the performance across ranks at the optimal scaling $\lambda=0.5$ (from `results/metrics_summary.json`):
  - Rank $k=16$: **73.23%**
  - Rank $k=32$: **74.50%**
  - Rank $k=64$: **74.58%**
  - Rank $k=128$: **74.83%**
  - Rank $k=256$: **74.68%**
- **Non-Monotonicity:** SVS performance is non-monotonic with respect to rank. If SVS behaves as a noise filter, increasing the rank from $k=128$ to $k=256$ (retaining more of the principal signal) should logically preserve more of the task vector's capabilities and perform closer to (or better than) $k=128$. Instead, rank $k=256$ drops to **74.68%**, which is worse than $k=128$ and even worse than Task Arithmetic ($74.78\%$). This non-monotonic behavior indicates that SVS's performance is highly hyperparameter-sensitive and lacks robust scaling behavior.

### 2. "SVS is Highly Competitive with State-of-the-Art Baselines" (Unsupported)
- SVS ($74.83\%$) is significantly outperformed by **TIES-Merging ($77.98\%$)**, representing a massive **3.15% absolute accuracy gap** under identical training-free, data-free offline constraints.
- SVS is also outperformed by **DARE ($75.18\%$)**.
- On **CIFAR-10**, SVS-128 ($79.60\%$) actually **degrades** the zero-shot performance of the base model ($80.20\%$), whereas TIES-Merging significantly improves it to **85.00%**.
- SVS's inability to match simple coordinate-basis pruning methods (TIES, DARE) proves that spectral-domain low-pass filtering is fundamentally inferior to coordinate sparsification for resolving multi-layer parameter interference. The claim of SVS being a highly competitive merging framework is thus empirically unsupported.

### 3. "SVS Acts as a Robust Regularizer" (Unsupported)
- SVS-16 is consistently worse than Task Arithmetic across the entire lambda sweep (e.g., $73.23\%$ vs $74.78\%$ at $\lambda=0.5$; $57.78\%$ vs $59.25\%$ at $\lambda=0.1$).
- SVS-128 tracks the Task Arithmetic curve almost identically, with minor, fluctuating deltas (e.g., $59.15\%$ vs $59.25\%$ at $\lambda=0.1$; $72.33\%$ vs $72.35\%$ at $\lambda=0.6$).
- The SVS curves do not demonstrate a robust regularizing effect; they simply represent a noisy, low-rank approximation of Task Arithmetic.

### 4. BWN Validation in MLPs (Inconsistent and Contradictory Results)
In Section 4.5, the authors validate BWN on a 3-layer MLP. A close look at `results/mlp_metrics_summary.json` reveals serious anomalies:
- At $\lambda=0.1$, BWN improves accuracy from $29.50\%$ to $30.25\%$ ($+0.75\%$).
- At $\lambda=0.3, 0.5, 1.0$, BWN provides **zero accuracy benefit** ($35.25\% \rightarrow 35.25\%$, $38.25\% \rightarrow 38.25\%$, $32.00\% \rightarrow 32.00\%$).
- At $\lambda=0.7$, BWN **decreases accuracy** from $36.00\%$ to **35.75%**.
- At $\lambda=0.9$, BWN **decreases accuracy** from $35.50\%$ to **35.25%**.
- Far from being a robust scale-preservation operator that "enhances multi-task merging accuracy," BWN actually degrades accuracy in the majority of scaling regimes. This critical negative result is omitted from the main text of the paper.

### 5. "Entropy-SVS Dynamic Rank Allocation Benefits" (Theoretical Bloat)
The authors present Entropy-SVS as a highly sophisticated, information-theoretic rank allocation scheme. However, comparing Entropy-SVS to uniform SVS reveals that this complexity yields **zero practical benefit**:
- At $m_{\text{entropy}}=0.4$, Entropy-SVS allocates an average rank of **43.90** and achieves **74.55%** accuracy.
- Looking at uniform SVS: uniform SVS at rank $k=32$ gets **74.50%** accuracy, and at rank $k=64$ gets **74.58%** accuracy.
- SVS with an average rank of 43.90 performing at **74.55%** is exactly what one would expect from a simple linear interpolation between uniform rank 32 and rank 64.
- In other words, **dynamically allocating ranks via Shannon spectral entropy performs exactly the same as simply applying a uniform rank of the same average size.**
- The mathematical overhead of calculating SVD, extracting singular values, and computing Shannon entropy across every single layer of a deep network is entirely wasted, as it provides no performance benefit over a basic, uniform low-rank projection.
