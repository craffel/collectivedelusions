# Experimental Setup and Results Evaluation

## Strengths of the Experimental Evaluation
1. **Diverse Multi-Task Vision Benchmark:** Evaluating across four visually distinct datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) represents a diverse set of domains (handwritten digits, fashion items, natural objects, and house numbers) with varying difficulty.
2. **Online Streaming Simulation:** Sweep-evaluating over batch sizes ($B \in \{1, 4, 16, 64, 256\}$) under a randomized, shuffled mixed-task stream is an excellent and realistic simulation of online streaming deployment. This setup effectively exposes the vulnerability of batch-averaging routers.
3. **Thorough Ablation Studies:** The paper includes a comprehensive suite of ablation studies, covering:
   - Sensitivity to the low-rank SVD rank ($r$).
   - Differentiable routing via Straight-Through Estimators (STE) vs. zero-shot Activation-Space Mean Initialization.
   - Isolating SVD truncation error via a Full-Rank baseline.
   - Oracle vs. autonomous classification head selection.
   - Statistical robustness sweeps over sequence seeds and random data splits.
   - Quantitative assessment of "routing jitter" across layers.

## Critical Scholarly Critique of the Claims and Results

### 1. Deconstructing "Heterogeneity Collapse" vs. "Soft-Routing Interference"
The authors claim that:
> "Because batch-dependency is applied, the dynamic coefficients smooth out towards a flat uniform weight distribution... as $B$ grows. When this occurs, the model's specialized dynamic benefits degrade back toward the Uniform Merging floor."

However, looking closely at **Table 1** (Joint Test Accuracy), the empirical drop in accuracy for the dynamic baselines as $B$ increases from 1 to 256 is almost negligible:
- **Linear Router:** drops from **59.28%** ($B=1$) to **59.18%** ($B=256$) — a microscopic decrease of **0.10%**.
- **QWS-Merge:** actually *increases* from **56.93%** ($B=1$) to **57.03%** ($B=256$) — a slight increase of **0.10%**.

**Critical Observation:** The "catastrophic performance degradation" or "collapse" *as batch size scales* is not actually visible in the empirical data. Instead, the baselines are already performing near their flat "collapsed" state at $B=1$. 
- **The Reason:** Soft routing methods (using Softmax over $K$ experts) suffer from **soft-routing task interference** even at $B=1$ because they apply a linear superposition of all experts, leading to parameter conflicts and degraded performance. 
- **Clarification Needed:** The paper's primary empirical advantage is that **Top-1 hard gating** completely eliminates soft task interference (yielding **63.87%** joint accuracy at $B=1$), while its **sample-wise activation routing** preserves this advantage batch-independently up to $B=256$. The narrative should be adjusted to clarify that soft-routing interference (at $B=1$) is a much larger bottleneck than batch-averaging degradation (which only accounts for a $\le 0.1\%$ drop in the baselines).

### 2. Extreme Low-Data Subsampling and Under-Trained Experts
The paper uses only **256 training samples** per dataset to fine-tune the experts, which is extremely small. This leads to very low standalone expert ceilings (such as **29.30%** for SVHN).
- **The Justification:** The authors defend this as a deliberate, pragmatic stress-test representing "extreme low-resource transfer learning."
- **The Scholarly Caveat:** While this is a creative defense, it introduces a major confounding factor. The finding that *SVD low-rank truncation acts as an implicit regularizer that outperforms the full-rank baseline by +1.38%* is likely an artifact of these under-trained, overfitted low-shot experts. 
- If the experts were trained on full datasets to convergence (e.g., SVHN at 96% and CIFAR-10 at 90%), they would possess highly robust, saturated feature representations. In that regime, SVD truncation would likely introduce *reconstruction loss* rather than acting as a regularizer, and full-rank weights would likely outperform low-rank ones. 
- **Recommendation:** The authors should explicitly discuss this limitation, acknowledging that the regularization benefits of SVD truncation are specific to low-resource or under-converged regimes, and that high-resource regimes may experience a standard reconstruction trade-off.

### 3. Missing Static and Dynamic Baselines
While the baseline selection is solid, a few key competitors are missing:
- **TIES-Merging / DARE:** These are mentioned in the Related Work as state-of-the-art static merging techniques that specifically resolve parameter conflicts. Since static Task Arithmetic (with standard scaling) is included, evaluating TIES-Merging or DARE would provide a much stronger static baseline and show whether SLD-Merge still outperforms advanced conflict-resolution methods.
- **Oracle Dynamic Merging:** It would be highly informative to include a baseline representing the "Oracle Dynamic Router" (i.e., a router that always selects the correct expert with 100% accuracy) using full-rank weights, to serve as an absolute ceiling for dynamic routing.
