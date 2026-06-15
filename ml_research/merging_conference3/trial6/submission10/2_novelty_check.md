# 2_novelty_check.md: Novelty and Delta Analysis of the Revised Paper

## 1. Delta from Prior Work
- **Static Model Merging (Task Arithmetic, TIES-Merging, DARE):** These methods interpolate specialized model parameters using fixed global coefficients. The proposed method (like other dynamic merging models) allows input-dependent interpolation.
- **Dynamic Model Merging (AdaMerging, BL-Router):** AdaMerging learns block-wise coefficients but keeps them static at test-time. BL-Router dynamically routes inputs but is susceptible to severe overfitting under low-data calibration (e.g., 16 samples per task). TCPR differentiates itself by introducing a task-relationship prior matrix $S$ as a regularizer to stabilize calibration.
- **Wave-Inspired Merging (QWS-Merge):** QWS-Merge is a state-of-the-art wave-inspired dynamic routing method. The paper positions TCPR as a simpler, more robust alternative that achieves superior performance without complex physical metaphors or non-standard optimization routines.

## 2. Theoretical and Mathematical Critique of the Revised Prior

The revised paper introduces two major modifications to resolve the mathematical and theoretical issues raised in previous reviews:
1. **Off-diagonal Similarity Centering:** Subtracting the off-diagonal mean ($\mu_{\text{off}}$) to create both positive and negative prior similarity elements.
2. **Signature Normalization:** Projecting the routing vectors $\mathbf{w}_i$ and $\mathbf{w}_j$ to unit length before computing their dot product, transforming the penalty term to use bounded cosine similarities.

While these modifications are mathematically elegant on paper, a rigorous analysis of their actual behavior during calibration reveals **significant theoretical and mathematical limitations**:

### A. Vanishing Gradients from Microscopic Priors
The centered parameter-space similarity matrix $S^{\text{centered}}$ consists of off-diagonal elements that are microscopic (near zero). In the experiments:
- The raw parameter similarities are extremely small: $S^{\text{param}}_{i, j} \in [0.014, 0.066]$
- Their off-diagonal mean is $\mu_{\text{off}} \approx 0.0246$
- The centered off-diagonal elements are extremely close to zero: e.g., $S^{\text{centered}}_{\text{mnist}, \text{fashion}} \approx +0.0085$, and $S^{\text{centered}}_{\text{mnist}, \text{cifar}} \approx -0.0100$

Because these centered similarity values are so close to zero, the gradients they pass to the optimizer are microscopic. When scaled by a small regularization coefficient like $\beta = 10^{-4}$, the entire regularization loss is on the order of $10^{-5}$ or $10^{-6}$. Given that the cross-entropy loss is around $2.3$, the regularizer has a mathematically negligible impact on the gradients during the 100 optimization steps. This explains why the optimized alphas and resulting performance of TCPR-Param are **exactly identical** to the unregularized router baseline.

### B. Ineffectiveness and Performance Degradation at Active Scales
To make the regularizer active, one must scale $\beta$ to a much larger value (such as $\beta = 1.0$ or $\beta = 100.0$). However, when $\beta$ is increased to these levels:
- The regularizer does not lead to any performance gains over the unregularized baseline.
- In fact, the performance remains the same or decreases significantly. For instance, the best-performing parameter during the sweep is selected as $\beta = 10^{-4}$ because larger values fail to exceed the unregularized baseline's performance (25.40% mean accuracy).
- For `TCPR-Rep` (Representation-Space), the "best" beta also defaults to $\beta = 10^{-4}$, but the actual performance is **21.70% joint mean**, which is significantly worse than the unregularized sigmoidal router baseline of **25.40%** and even worse than standard isotropic L2 weight decay (**24.00%**).

Thus, there is **no regime** where the proposed task-correlation regularizer actually improves performance over the simple unregularized sigmoidal router baseline. The proposed regularizer is either completely inactive (mathematically dead) or actively detrimental to multi-task performance.
