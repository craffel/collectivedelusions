# 1. Summary of the Paper

## Main Topic and Objective
The paper investigates **multi-task model merging** under test-time adaptive conditions. Specifically, it explores a fundamental and previously unexamined question: *At what level of physical/structural granularity should merging coefficients be defined and optimized, and how does this choice affect multi-task generalization?* 

To address this, the authors introduce **GranMerge**, a unified empirical framework designed to systematically deconstruct the **Generalization-Granularity Trade-off** in adaptive model merging across five nested levels of parameter resolution (from a single global scalar per task up to tensor-wise coefficients) under test-time constraints.

## Proposed Approach (GranMerge)
1. **Task Vector Formulation:** Builds on standard task arithmetic by subtracting pre-trained base weights ($W_{\text{base}}$) from fine-tuned expert weights ($W_k$) to get task vectors ($\theta_k = W_k - W_{\text{base}}$). Merges them dynamically as:
   $$W^{(t)} = W_{\text{base}}^{(t)} + \sum_{k=1}^K \lambda_{k, t} \theta_k^{(t)}$$
   where $\lambda_{k, t}$ is the merging coefficient for task $k$ and parameter tensor $t$.
2. **The Five Nested Levels of Granularity:**
   - **Level 1 (Global / Task Arithmetic):** Exactly 1 scalar per task ($K$ total parameters).
   - **Level 2 (Layer-wise / AdaMerging):** 1 scalar per layer per task ($L \times K$ total parameters).
   - **Level 3 (Block-wise):** 2 scalars per layer (Attention vs. MLP block) per task ($2 \times L \times K$ total parameters).
   - **Level 4 (Component-wise):** 4 scalars per layer ($\{qkv, attn\_out, mlp\_fc1, mlp\_fc2\}$) per task ($4 \times L \times K$ total parameters).
   - **Level 5 (Tensor-wise):** 6 scalars per layer (individual major projection modules: `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`) per task ($6 \times L \times K$ total parameters).
3. **Unsupervised Test-Time Objective:** Minimizes the prediction entropy of the merged model's outputs over a compact, unlabeled calibration stream ($X_{\text{cal}}$) of $N=256$ samples per task:
   $$\mathcal{L}(\Lambda) = \frac{1}{K} \sum_{k=1}^K \mathcal{H}\left(P(Y \mid X_{\text{cal}, k}; W(\Lambda))\right) + \beta \mathcal{R}(\Lambda)$$
4. **Optimizers Analyzed:**
   - **Adam (First-order):** 60 steps with $\eta = 0.02$, unconstrained gradients.
   - **1+1 Evolution Strategies (Zero-order):** 100 mutation steps with step size adaptation, isotropic random search.
5. **Regularizations Proposed:**
   - **Elastic Spatial Regularization (ESR):** Pulls fine-grained coefficients towards their layer average: $\mathcal{R}_{\text{ESR}}(\Lambda) = \sum_{k} \sum_{l} \sum_{t \in T_l} \left( \lambda_{k, t} - \bar{\lambda}_{k, l} \right)^2$.
   - **Depth-wise Total Variation (TV):** Penalizes rapid coefficient changes between adjacent layers: $\mathcal{R}_{\text{TV}}(\Lambda) = \sum_{k} \sum_{l=2}^L \sum_{c} \left( \lambda_{k, l, c} - \lambda_{k, l-1, c} \right)^2$.

## Key Findings and Claims
1. **The Generalization-Granularity Trade-off:** Coarser merging (Level 1 Global) suffers from underfitting due to low capacity. Moving to intermediate granularities (Level 2 to 4) improves generalization. However, high-granularity merging (Level 5 Tensor-wise) suffers from severe **transductive overfitting** on the compact calibration batch ($N=256$), degrading multi-task test-set generalization.
2. **First-Order vs. Zero-Order Dynamics:** First-order Adam gradient descent is highly susceptible to rapid, catastrophic overfitting (dropping performance on Level 5 to 26.91%). Zero-order 1+1 ES is significantly more robust (maintaining 29.43% on unregularized Level 5), which is deconstructed as a combination of:
   - *Isotropic implicit regularization* (self-limiting, non-coordinate-aligned search).
   - *Optimization sluggishness / underfitting* (inherent inefficiency in high dimensions; 1+1 ES fails to optimize away from its near-optimal initialization, effectively preserving baseline performance).
3. **Regularization Effectiveness:** ESR and TV soft regularizers successfully stabilize Level 5 merging, improving ES performance to 30.17% (close to baseline) and Adam to 28.51%. However, they are insufficient to fully stop Adam's chaotic gradient overfitting, indicating that first-order adaptation needs harder structural constraints.
4. **Surrogate Loss Misalignment and Static Baseline Superiority:** No test-time adaptive configuration (even regularized) outperforms the static, zero-overhead **Uniform Task Arithmetic baseline (30.41%)**. This is attributed to a fundamental misalignment where minimizing prediction entropy leads to "confident but incorrect" predictions on the compact calibration set rather than genuine multi-task alignment.

## Claims and Evidence Mapping
- **Claim 1 (Overfitting at Level 5):** Supported by Table 1 results showing unregularized Level 5 Adam (26.91%) and 1+1 ES (29.43%) perform worse than intermediate granularities like Level 2 (29.18% Adam, 29.17% ES).
- **Claim 2 (ES Sluggishness vs. Isotropic search):** Supported by the mathematical logic of high dimensions (288 parameters optimized for 100 steps of 1+1 ES is highly under-optimized), and the empirical trend where ES performance slowly climbs from 24.84% (L1) to 29.98% (L4), reverting back toward the static initialization (30.41%).
- **Claim 3 (Regularization recovery):** Supported by Table 1 where L5 with ESR/TV recovers to 28.51% (Adam) and 30.17% (ES).
- **Claim 4 (Static baseline supremacy):** Supported by the fact that the Uniform Task Arithmetic baseline achieves 30.41%, higher than all adaptive experiments.
