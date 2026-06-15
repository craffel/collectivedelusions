# 1. Summary of the Paper

## Main Topic and Objective
The paper introduces **GranMerge**, a systematic empirical framework designed to investigate the **Generalization-Granularity Trade-off** in adaptive multi-task model merging. Specifically, it examines test-time adaptive merging where merging coefficients are optimized on a small, unlabeled calibration stream at deployment time by minimizing prediction entropy. The central research question is: *At what level of physical granularity should merging coefficients be defined and optimized, and how does this choice affect multi-task generalization?*

## Proposed Approach
To answer this, the authors evaluate merging coefficients across five nested levels of parameter resolution using a 12-layer Vision Transformer (ViT-Tiny):
1. **Level 1: Global Merging (Task Arithmetic):** A single scalar per task across the entire model (4 parameters for 4 tasks).
2. **Level 2: Layer-wise Merging (AdaMerging):** One scalar per layer per task (48 parameters).
3. **Level 3: Block-wise Merging:** Two scalars per layer per task, separating attention vs. MLP blocks (96 parameters).
4. **Level 4: Component-wise Merging:** Four scalars per layer per task, separating sub-components ($qkv$, $attn\_out$, $mlp\_fc1$, $mlp\_fc2$) (192 parameters).
5. **Level 5: Tensor-wise Merging:** Six scalars per layer per task, corresponding to the major projection modules ($q\_proj$, $k\_proj$, $v\_proj$, $out\_proj$, $fc1$, $fc2$) scaling both weights and biases (288 parameters).

The unsupervised surrogate loss minimized at test-time is prediction entropy over a compact local calibration stream ($N=256$). To optimize this, the authors compare two different optimization families:
- **First-order (Adam Gradient Descent):** Updated via analytical gradients for 60 steps.
- **Zero-order (1+1 Evolution Strategies - ES):** Updated via derivative-free isotropic random mutations for 100 steps.

To combat overfitting at higher granularities, the authors introduce and benchmark a joint regularization term $\mathcal{R}(\Lambda)$ scaling two L2 penalties:
- **Elastic Spatial Regularization (ESR):** Pulls fine-grained coefficients towards their layer-wise average.
- **Depth-wise Total Variation (TV) Smoothness:** Penalizes rapid fluctuations in coefficient values between adjacent layers.

---

## Key Findings and Claims

1. **The Degradation of Generalization with Granularity:**
   Increasing structural granularity leads to severe **transductive overfitting** on the local calibration batch ($N=256$). For instance, unregularized Level 5 Tensor-wise merging collapses to **26.91%** for Adam and **29.43%** for 1+1 ES, degrading performance compared to coarser intermediate granularities (e.g., Level 4 ES achieves **29.98%**) or the static baseline.
   
2. **First-order vs. Zero-order Overfitting Dynamics:**
   First-order Adam is highly vulnerable to rapid and severe transductive overfitting because analytical gradients quickly drive coefficients into extreme, unphysical configurations. In contrast, zero-order 1+1 ES maintains higher generalization. The authors provide two explanations:
   - *Isotropic implicit regularization* due to self-bounding random walk trajectories.
   - *Optimization sluggishness (underfitting)* under the curse of dimensionality, where ES fails to converge in 100 steps within a 288-dimensional space, thereby remaining near its robust initialization scale.

3. **Stabilization via Spatial-Depth Regularization:**
   Joint ESR and TV regularization successfully mitigates overfitting. For 1+1 ES, it improves Level 5 performance from 29.43% to **30.17%** (recovering nearly the entire drop). For Adam, it improves Level 5 performance from 26.91% to **28.51%** (a 1.60% recovery), though Adam remains heavily overfitted.

4. **The Supremacy of Static Baselines and Loss Misalignment:**
   A critical and honest finding is that **no test-time adaptive configuration outperforms the static, zero-overhead Uniform Task Arithmetic baseline of 30.41%**. This is driven by a fundamental misalignment where minimizing prediction entropy on a small batch forces the model to make confident but highly incorrect predictions, disrupting underlying representation boundaries.

---

## Explicitly Claimed Contributions
- **GranMerge Framework:** The first systematic deconstruction of model merging resolution from global to tensor-level.
- **Comparative Optimization Analysis:** Directly contrasting first-order (gradient-based) and zero-order (derivative-free) dynamics, showing the implicit regularizing effect of the latter.
- **Regularization Design:** Introducing and validating joint ESR and TV penalties to stabilize fine-grained adaptive weight blending.
- **Critical Diagnostics:** Exposing the fundamental misalignment of prediction entropy as a surrogate test-time loss in low-resource regimes.
