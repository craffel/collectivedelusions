# Paper Summary: Robust Linear Routing (RLR)

## Main Topic and Motivation
The paper addresses the challenge of **dynamic model merging** (or dynamic parameter fusion), which combines multiple task-specific expert neural networks into a single multitask model by predicting sample-specific blending coefficients on-the-fly at runtime. 

The primary motivation of the work is to challenge and deconstruct the escalating complexity of recent dynamic model merging methods—specifically, **Quantum Wavefunction Superposition Merging (QWS-Merge)** (Vance, 2025). QWS-Merge asserted that classical linear routing suffers from structural limitations that lead to a catastrophic representation collapse on high-variance, out-of-distribution datasets (such as SVHN, where it reportedly achieved only 15.30% accuracy). To resolve this, QWS-Merge introduced a convoluted framework of task weight wavefunctions, phase projectors, and wave interference. 

Through the lens of Occam's razor, the authors hypothesize that the reported SVHN collapse of classical linear routing is not a fundamental structural limitation, but rather a standard, preventable overfitting and high-variance logit phenomenon under out-of-distribution shifts.

## Proposed Approach
The authors propose **Robust Linear Routing (RLR)**, a minimalist, mathematically transparent dynamic model merging framework. RLR retains a simple 768-parameter classical gating layer:
1. **Linear Projection:** Raw routing logits $z \in \mathbb{R}^N$ are computed from input representation $x \in \mathbb{R}^d$ via a single linear layer $z = Wx + b$.
2. **Softmax Temperature Scaling:** To convert logits to blending coefficients $a_k$, a constant temperature scaling parameter $T \ge 1$ is introduced:
   $$a_k = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$$
   This softens the routing outputs and guarantees a stable mixture of experts.
3. **$L_2$ Weight Decay:** During offline calibration, the router's weights $W$ are regularized via a Frobenius norm penalty in the cross-entropy objective. This limits the logit magnitudes and prevents extreme hard-switching decisions.

The entire optimization is highly efficient: it trains only 772 parameters (W and b) on a tiny, offline calibration set of 16 samples per task (64 total samples) for 100 steps of Adam, taking less than a second on a single GPU.

## Key Findings and Claims
1. **Deconstruction of SVHN Collapse:** Both unregularized classical linear routing and RLR achieve outstanding classification accuracies on SVHN ($94.87\%$ and $94.36\%$ respectively on Seed 42, and $91.20\% \pm 1.85\%$ across 5 seeds). This completely debunks QWS-Merge's claim of a structural collapse.
2. **Superiority over QWS-Merge:** On a unified ViT-Tiny benchmark, classical linear gating ($95.46\%$ Joint Mean) and RLR ($94.68\%$ Joint Mean) significantly outperform QWS-Merge, whether comparing against reported paper numbers ($59.32\%$) or a local re-implementation ($90.03\%$).
3. **Resilience to Heterogeneous Streams:** In mixed-task test streams where incoming batches contain a mixture of different domains, dynamic merging methods suffer from "heterogeneity collapse" due to batch-wise coefficient averaging. RLR acts as a stabilizer, maintaining a consistent performance buffer over unregularized linear gating as batch size increases (e.g., $75.03\%$ vs. $73.15\%$ at $B=256$).
4. **Diagnostic Insights:** The authors identify that the SVHN collapse in prior work was likely triggered by sub-optimal hyperparameter choices, specifically:
   - Extracting routing representations from deep, task-warped layers rather than early, task-agnostic layers.
   - Employing excessively high learning rates ($>0.1$).
   - Over-optimizing the router for thousands of steps on a tiny dataset.

## Explicitly Claimed Contributions and Evidence
1. **Empirical Deconstruction of QWS-Merge:** Proven via local re-implementation under identical conditions, showing QWS-Merge's performance is inferior to simple classical linear routing (Table 1).
2. **Proposing Robust Linear Routing (RLR):** Confirmed by formulation (Section 3.1) and empirical results (Section 4.2).
3. **Resilience under Mixed-Task Streams:** Proven through mixed-task evaluations with batch sizes $B \in \{1, 16, 256\}$, demonstrating RLR outperforming the classical unregularized baseline (Table 3, Figure 2).
4. **Statistical Rigor:** Supported by multi-seed evaluation across 5 random calibration seeds, showing stable convergence without catastrophic collapse (Section 4.3).
5. **Systematic Ablations and Sensitivity Studies:** Validated by representation layer ablation (Table 4) and a 2D hyperparameter sweep over $\alpha$ and $T$ (Figure 3).
