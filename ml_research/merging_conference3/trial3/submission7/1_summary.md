# Paper Summary: GranMerge

## 1. Core Question and Motivation
"GranMerge: Deconstructing the Generalization-Granularity Trade-off in Adaptive Model Merging" investigates the structural scale of merging coefficients in test-time adaptive multi-task model merging. While prior work is fragmented (ranging from a single global scale to layer-wise scalars or spline parameterizations), this paper asks a fundamental, previously unexamined question: *At what structural resolution should merging coefficients be defined and optimized, and how does this choice affect multi-task generalization?*

## 2. Proposed Framework: GranMerge
The authors introduce **GranMerge**, a unified framework to systematically evaluate five nested levels of parameter resolution for task vector coefficients:
*   **Level 1: Global Merging (Task Arithmetic):** 1 scalar coefficient $\lambda_k$ per task across the entire model ($K$ parameters total, e.g., 4 parameters for $K=4$).
*   **Level 2: Layer-wise Merging (AdaMerging):** 1 scalar coefficient per layer per task ($L \times K$ parameters total, e.g., 48 parameters for $L=12, K=4$).
*   **Level 3: Block-wise Merging:** 2 coefficients per layer per task (Attention vs. MLP), yielding $2 \times L \times K$ parameters total (e.g., 96 parameters).
*   **Level 4: Component-wise Merging:** 4 coefficients per layer per task ($\{qkv, attn\_out, mlp\_fc1, mlp\_fc2\}$), yielding $4 \times L \times K$ parameters total (e.g., 192 parameters).
*   **Level 5: Tensor-wise Merging:** 6 coefficients per layer per task (corresponding to main projection modules: `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`), yielding $6 \times L \times K$ parameters total (e.g., 288 parameters).

## 3. Adaptation and Optimization Dynamics
At test-time, the coefficients $\Lambda$ are adapted on a small, unlabeled calibration stream ($N=256$) by minimizing the prediction entropy of the model's outputs. The paper compares two fundamentally different optimization paradigms:
1.  **First-order Adam Gradient Descent:** Analytical gradients computed via backpropagation (60 steps, learning rate $\eta=0.02$).
2.  **Zero-order 1+1 Evolution Strategies (ES):** Derivative-free stochastic mutations with dynamic step-size adaptation (100 steps, initial mutation scale $\sigma=0.05$).

## 4. Regularization Techniques
To prevent transductive overfitting at higher granularities, two regularizations are benchmarked at Level 5:
*   **Elastic Spatial Regularization (ESR):** Pulls fine-grained coefficients towards their task-specific layer-wise average.
*   **Total Variation (TV) Smoothness Penalty:** Enforces depth-wise coefficient smoothness across adjacent transformer layers.

## 5. Main Empirical Findings
*   **The Generalization-Granularity Trade-off:** Coarse-grained merging (L1) suffers from underfitting due to low capacity. Moving to intermediate granularities (L2–L4) improves performance. However, high-granularity merging (unregularized L5 Tensor-wise) suffers from severe *transductive overfitting*, collapsing generalization performance on the test set.
*   **Optimizer Dynamics:** Adam gradient descent is highly susceptible to transductive overfitting, whereas zero-order 1+1 ES acts as an implicit regularizer, preserving representation stability.
*   **Dual Interpretation of ES Robustness:** The paper details two competing yet complementary explanations for ES's superior generalization in high dimensions: (1) isotropic search boundaries, and (2) optimization sluggishness (underfitting) due to the curse of dimensionality.
*   **Regularization Effectiveness:** Combining ESR and TV successfully stabilizes Level 5 performance, recovering 0.74% for 1+1 ES and 1.60% for Adam.
*   **The Supremacy of the Uniform Baseline:** Despite adaptation and regularization, *no adaptive configuration outperforms the static, zero-overhead Uniform Task Arithmetic baseline of 30.41%*, highlighting a fundamental misalignment of the unsupervised entropy objective on small calibration streams.
