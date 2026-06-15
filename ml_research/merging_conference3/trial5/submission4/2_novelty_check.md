# Originality & Novelty Check: BC-Router

## 1. Originality & Novelty Assessment
**Rating: Good / Excellent**

While the paper does not introduce entirely new mathematical operations (it relies on standard linear projections, Softmax, Sigmoid, and L2 regularization), its **originality is high** due to its rigorous **methodological deconstruction** and application of **Occam's razor** to complex SOTA paradigms.

The paper's key original contributions include:
1.  **Confounder-Driven Framework:** Instead of doing a standard empirical sweep, the authors systematically design three variants (BL-Router, GLS-Router, BSigmoid-Router) to isolate specific structural and optimization confounders:
    *   **The Over-Scaling Confounder:** Tested by capping the coefficients at 0.3.
    *   **The Layer-wise Specialization Confounder:** Tested by introducing layer-wise scaling amplitudes to a global routing head.
    *   **The Zero-Sum Competitive Bottleneck:** Tested by replacing Softmax with independent Sigmoids.
2.  **BSigmoid-Router Gating Genuinely Addressing Calibration Competition:** The mathematical formulation of BSigmoid-Router uses independent, uncoupled Sigmoid functions. This completely sidesteps the Softmax zero-sum competition. This represents an elegant transfer of Mixture-of-Experts (MoE) token-routing insights to parameter-space model merging.
3.  **Nuanced Scientific Reconciliations:** The paper avoids a simple, adversarial "our method is better than SOTA" narrative. Instead, it uncovers a non-trivial scientific insight: QWS-Merge's wave phase-projection equations function as an excellent **structural regularizer** that prevents task-sacrificing behavior and reduces optimization variance under tight calibration budgets, explaining *why* QWS-Merge works so stably compared to unregularized, unconstrained classical baselines. This elevates the work from a simple benchmarking exercise to a mature scientific study.

---

## 2. Positioning Relative to Prior Work
The paper is excellently positioned relative to prior work:
*   **Static Model Merging (Task Arithmetic, TIES-Merging, DARE):** The paper clearly states the trade-offs of static approaches (global compromise representation collapse) and positions dynamic parameter routing as a solution.
*   **Test-Time Adaptation (AdaMerging):** The paper clarifies a major, frequently conflated paradigm distinction. AdaMerging (online TTA) requires active test-time optimization via backward passes, introducing extreme latency and susceptibility to stream noise. Offline-calibrated dynamic routing (like QWS-Merge or BC-Router) optimizes once on a tiny offline calibration set, running as a pure forward pass during inference with zero active optimization.
*   **QWS-Merge:** The paper directly targets QWS-Merge, adopting its calibration protocol but training task experts to true convergence and correcting baseline optimization (L2 regularization, Softmax-free sigmoidal activations) to isolate the true drivers of performance.

The paper is highly novel because it strips away over-engineered, exotic mathematical metaphors to reveal the elegant, simple classical mechanisms that actually drive dynamic model-merging performance.
