# 2. Novelty Check

## Key Novel Aspects and 'Delta' from Prior Work

The paper introduces several distinct novel elements compared to existing literature on model merging, test-time routing, and streaming system wrappers:

1. **Parameter Space vs. Activation Space Transition:**
   * **Prior Work (e.g., PFSR):** Merges task-specific expert parameters dynamically in parameter space ($W_{\text{merged}} = W_{\text{base}} + \sum \alpha_k V_k$). Under mixed-task streaming batches ($B > 1$), this forces the system to average coefficients over the batch dimension ($\bar{\alpha}_k = \frac{1}{B} \sum \alpha_{k, b}$), causing a catastrophic loss of individual expert specialization (**heterogeneity collapse**).
   * **SABLE Delta:** Shifts the ensembling step entirely from parameter space to activation space ($Y_b = X_b W_{\text{base}} + \sum \alpha_{k, b} (X_b A_k B_k)$). This leverages the distributive property of matrix multiplication to perform ensembling on a per-sample basis during the forward pass, making the model natively immune to batch heterogeneity while maintaining mathematical equivalence to parameter-space ensembling for single-sample batches.

2. **Network-Level Simplicity vs. Systems-Centric Complexity:**
   * **Prior Work (e.g., MBH):** Solves heterogeneity collapse by intercepting the streaming query queue and dynamically scheduling queries into homogeneous micro-batches via temporal buffering and similarity-based sorting. This introduces a stateful serving layer, queuing latencies, and significant memory/compute bottleneck dependencies.
   * **SABLE Delta:** Completely replaces the stateful systems scheduler wrapper with a clean, stateless network-level mathematical formulation. It processes samples immediately upon arrival, eliminating temporal buffers and systems-level queuing latency.

3. **Completely Zero-Data / Zero-Calibration Centroid Refinement:**
   * **Prior Work:** Non-parametric routing often requires small calibration splits or support data to align features, or relies on crude weight-averaging heuristics across class boundaries.
   * **SABLE Delta:** Proposes and mathematically refines a **Completely Zero-Data Centroids** construction method. By applying weight-space L2-normalization to individual class-specific expert head weights before taking their row-wise mean, SABLE prevents vector cancellation (where discriminative class vectors pointing in opposite directions cancel each other out). This yields a pure semantic task prototype directly from pre-trained parameters with absolutely zero calibration data, zero forward-pass samples, and zero support splits.

4. **Layer-Dependent Hybrid-Rank Selection Protocol:**
   * **Prior Work:** PEFT ensembling methods enforce uniform rank constraints across all layers of the model.
   * **SABLE Delta:** Proposes a layer-dependent hybrid-rank protocol. It highlights that final classification heads (low-dimensional projection layers) suffer severely under strict low-rank bottleneck constraints ($r \le 4$), whereas hidden backbone layers (highly dimensional) are highly compressible. By ensembling output heads at full precision and keeping hidden layer adapters at aggressive low rank ($r \le 2$), SABLE bypasses capacity bottlenecks with negligible parameter overhead. This uncovers the **Low-Rank Regularization Paradox**, where an intermediate hidden layer of rank $r=2$ can outperform rank $r=4$ because the tighter bottleneck acts as a powerful low-pass filter to prune cross-task representation noise.

5. **Theoretical Bridge to Production Serving Engines:**
   * SABLE bridges the gap between parameter-space task-vector merging and high-speed multi-tenant PEFT serving engines (such as Punica or S-LoRA). SABLE aligns the mathematical ensembling benefits of task-vector addition with standard, off-the-shelf batched GEMV/GEMM vectorized execution, enabling stateless ensembling that scales gracefully to massive expert pools.

## Characterization of Novelty
The novelty of this work is **significant**. 

While the use of the distributive property of matrix multiplication to shift operations between weight space and activation space is a known mathematical concept, its application as a **systemic solution** to solve "heterogeneity collapse" in test-time model merging is highly creative and elegant. Rather than patching a failure mode of deep learning serving by building increasingly complex, stateful, and latency-heavy systems queues (like MBH), SABLE addresses the root mathematical cause entirely at the network layer. 

Additionally, the paper does not stop at a naive translation to activation space; it introduces rigorous solutions to the practical challenges of this approach:
* **Representational Alignment Paradox:** Solved via Mid-Layer Routing (Late Adaptation), leaving early layers unadapted.
* **Vector Cancellation in Zero-Data Centroids:** Solved via the Refined L2-normalization heuristic.
* **Low-Rank Capacity Bottlenecks:** Solved via the Layer-Dependent Hybrid-Rank protocol.
* **OOD Sensitivity:** Solved via Soft Sigmoid Gating and an Adaptive Dynamic Thresholding algorithm.

This represents a comprehensive, highly mature, and mathematically well-grounded set of innovations that moves test-time dynamic ensembling into clean, stateless execution.
