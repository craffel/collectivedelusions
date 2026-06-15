# Summary Report (`1_summary.md`)

## 1. Main Topic of the Paper
The paper investigates **dynamic layer-wise model merging** in weight space. Specifically, it focuses on the "Layer-Averaging Collapse" (or rank-1 collapse) debate, which claims that layer-wise dynamic routing coefficients inevitably collapse to a single global dimension, rendering layer-wise routers redundant. The paper attempts to deconstruct this theoretical claim from a "critical methodological perspective," arguing that the collapse is an artifact of over-simplified, linear representation-space sandboxes. It proposes a physical dynamic routing framework utilizing a **Bounded Sigmoid (BSigmoid) router** to evaluate routing dimensionality under realistic multi-task settings with varying degrees of semantic conflict.

## 2. Proposed Approach
The authors propose a physical model-merging pipeline with the following key components:
- **Layer-wise Dynamic Model Merging:** Merging $K$ specialized expert models (sharing a common initialization) layer-by-layer using sample-specific dynamic routing coefficients $\lambda_{l, k}(x) \in [0, 1]$ that sum to 1.0 across experts at each layer $l$.
- **Bounded Sigmoid (BSigmoid) Router:** A gating mechanism that projects a high-dimensional input $x \in \mathbb{R}^{784}$ (for Split-MNIST) into a low-dimensional space $d=8$ using a random frozen Gaussian matrix $P_{proj}$. It then computes unconstrained logits, applies independent element-wise sigmoids, and normalizes them post-gating to obtain the final coefficients.
- **SVD Collinearity Audit:** A diagnostic metric that constructs the Batch-Averaged Layer-wise Coefficient Matrix $A \in \mathbb{R}^{L \times K}$ and uses Singular Value Decomposition (SVD) to compute the **Collinearity Ratio** $\rho_{collinear} = \sigma_1 / \sum \sigma_i$. A ratio near 1.0 indicates rank-1 collapse, while a lower ratio indicates multi-dimensional routing.
- **Inter-Layer Cosine Similarity Matrix:** A matrix mapping the directional alignment between pairs of layers to visualize spatial routing specialization.
- **Calibration Pipeline:** Router parameters are optimized on a balanced, few-shot dataset of 128 samples per task over 40 steps using Adam with $L_2$ weight decay.

## 3. Key Findings
- **Absence of Rank-1 Collapse in High Conflict:** The authors report that under their "Cross-Domain" task suite, the SVD Collinearity Ratio drops to $0.4987 \pm 0.08$ on DeepMLP-12 and $0.5673 \pm 0.03$ on TinyCNN-4, indicating multi-dimensional routing.
- **Spatial Specialization:** Inter-layer cosine similarity heatmaps show block-diagonal patterns under Cross-Domain task conflict, which the authors claim aligns with hierarchical feature extraction.
- **BSigmoid Superiority over Softmax:** The authors claim that BSigmoid outperforms Softmax on TinyCNN-4 (e.g., $52.52\%$ vs. $28.33\%$ on Cross-Domain) due to the decoupling of gradient paths during calibration.
- **Calibration Scaling:** The Layer-wise Router's performance improves as the calibration budget scales, eventually crossing over the static OFS-Tune baseline at larger splits (e.g., $B \ge 256$ samples per task).

## 4. Explicitly Claimed Contributions and Accompanying Evidence
The paper explicitly claims four major contributions:
1. **Deconstruction of Rank-1 Collapse:** Claiming that layer-wise routing trajectories occupy a multi-dimensional subspace rather than collapsing.
   - *Evidence:* Table 3 and Figure 1 show SVD Collinearity Ratios dropping to $\approx 0.50$ (MLP) and $\approx 0.57$ (CNN) under Cross-Domain conflict.
2. **Emergence of Depth-Specialized Policies:** Claiming that high-conflict settings force the network to specialize its routing into distinct block-diagonal structures.
   - *Evidence:* Figure 2 shows block-diagonal clusters in inter-layer cosine similarity heatmaps for DeepMLP-12 under Cross-Domain conflict, contrasted with uniform heatmaps under Low-Conflict.
3. **The Capacity-Variance Trade-off:** Claiming that under low-data budgets, static/global routers act as regularizers, but layer-wise routers scale better with larger calibration splits.
   - *Evidence:* Figure 4 plots accuracy vs. calibration budget, showing a crossover around 256 samples where the Layer-wise Router surpasses the static OFS-Tune baseline.
4. **Critical Role of Regularization and Router Design:** Claiming that $L_2$ regularization governs generalization and that the decoupled gradient paths of the BSigmoid router resolve optimization conflict compared to Softmax.
   - *Evidence:* Tables 1 and 2 compare Layer-wise with and without regularization. Section 4.4 discusses BSigmoid vs. Softmax gradients and provides tracked gradient norms.

## 5. Critical Initial Observations on the Claims & Evidence
As a highly skeptical and critical reviewer, several massive red flags stand out immediately regarding these claims and their corresponding evidence:
- **Severe Performance Gap / Functional Failure:** The absolute performance of the merged models is extremely poor. On DeepMLP-12 Cross-Domain, the Layer-wise Router achieves $16.15\%$, which is barely above the $12.5\%$ random guessing threshold for an 8-class task. This is a functional failure—the model is completely unusable. On TinyCNN-4, the "state-of-the-art" dynamic layer-wise router is consistently *outperformed* by the simple offline static baseline **OFS-Tune** across all task-conflict suites in the primary data split (128 samples per task).
- **The "Batch-Averaged Paradox" Undermines Practicality:** The authors dedicate Section 3.5 to describing a fundamental paradox where dynamic model merging is either logically redundant (homogeneous batches) or degrades to a static compromise (mixed batches). If so, the entire premise of dynamic weight-space merging is shown to be practically non-viable in its current form.
- **Collinearity Ratio Interpretation is Overstated:** For $K=2$ tasks, the absolute minimum possible collinearity ratio is $0.5$. The reported ratios of $0.64$ to $0.74$ are actually very high, indicating that the first singular value still heavily dominates. This is far from a complete "deconstruction" of collinearity.
- **Toy Dataset Constraints:** The entire empirical evaluation is conducted on Split-MNIST subsets. The transition to "natural images" is relegated to a tiny, unverified discussion in Section 4.4 and a small ViT simulation in the appendix where only collinearity (and not classification accuracy) is evaluated.
