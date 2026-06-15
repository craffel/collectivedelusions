# 2. Novelty Check

## Key Novel Aspects and the 'Delta' from Prior Work

The paper introduces several distinct novel concepts that differentiate it from previous works in test-time model merging:

1. **PAC-Bayes Control-Space Regularization:** 
   - *Prior Work:* Traditionally, PAC-Bayes theory has been applied to the extremely high-dimensional weight space of deep neural networks (which is computationally intractable and ignores layer structure).
   - *Delta:* This work applies PAC-Bayes generalization bounds directly to the low-dimensional control space of **merging coefficients** $\Lambda \in \mathbb{R}^{L \times K}$ rather than parameter weight space. It leverages Alquier's linear PAC-Bayes bound to justify a linear-in-KL optimization objective.

2. **Continuous GP Spatial Prior & Precision Regularization:**
   - *Prior Work:* Previous methods like RegCalMerge (ESR) utilize disjoint, heuristic penalties to enforce proximity to initialization and adjacent-layer smoothness separately. PolyMerge uses hard constraints by projecting coefficients into low-degree polynomial subspaces, which are overly rigid and cannot capture localized block transitions.
   - *Delta:* GP-BayesMerge places a continuous Gaussian Process prior over normalized network depth coordinates. The resulting Kullback-Leibler complexity penalty naturally simplifies to a single quadratic form governed by the GP precision matrix $\Sigma_{\ell}^{-1}$. This single, positive-definite symmetric operator mathematically unifies both proximity (diagonal) and spatial smoothness (off-diagonal) constraints.

3. **Joint Kronecker Multi-Task GP Prior (MT-GP-BayesMerge):**
   - *Prior Work:* Existing methods assume task coefficients are optimized in complete isolation, ignoring representational conflicts and parameter trade-offs.
   - *Delta:* MT-GP-BayesMerge defines a joint prior over all tasks using a Kronecker product covariance: $\Sigma_{\text{joint}} = B \otimes \Sigma_{\ell}$, where $B$ is a task-similarity matrix. Crucially, the paper proposes estimating $B$ dynamically online and data-free using activation Centered Kernel Alignment (CKA) on incoming target calibration samples.

4. **Linear-Time Analytical OU Precision Matrix:**
   - *Prior Work:* General GP formulations require computing a dense inverse covariance matrix, which scales cubically $O(L^3)$ and can be numerically unstable.
   - *Delta:* The paper derives an exact closed-form analytical inverse for the Ornstein-Uhlenbeck (OU) kernel precision matrix. It shows that $\Sigma_{\text{OU}}^{-1}$ is strictly tridiagonal and can be assembled in $O(L)$ linear time, completely bypassing matrix inversion.

5. **Theoretical Bridge for Unsupervised TTA:**
   - *Prior Work:* Test-time adaptation typically minimizes unsupervised entropy without formal target-risk guarantees.
   - *Delta:* The authors explicitly formalize the **surrogate-to-target risk gap** and prove Theorem 3.4, which bounds true classification risk by the expected prediction entropy under two formal conditions: Margin-Preserving Support and Classifier Calibration.

## Characterization of Novelty

The novelty of this paper is **significant**. Rather than presenting another set of disconnected empirical heuristics, the authors build a cohesive, first-principles mathematical framework. By bridging PAC-Bayes generalization theory, continuous spatial Gaussian Processes, and test-time weight interpolation, they provide deep analytical insights (e.g., how lengthscale $\ell$ smoothly interpolates between independent weight decay and flat spatial averaging). The Kronecker multi-task extension and the analytical OU tridiagonal formulation are exceptionally elegant additions that solve real practical scalability and task-conflict bottlenecks.
