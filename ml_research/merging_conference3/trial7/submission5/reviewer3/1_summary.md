# Paper Summary: Parameter-Free Activation Blending

## 1. Main Topic and Motivation
The paper addresses the challenge of deploying multiple task-specific Parameter-Efficient Fine-Tuning (PEFT) adapters (like LoRA) on heterogeneous multi-task inference streams without model duplication or parameter bloat. 
While weight-space model merging (e.g., Task Arithmetic, TIES-Merging, DARE) can combine experts statically, it forces a global, uniform compromise that degrades individual expert performance. Dynamic test-time routing methods solve this under homogeneous batches but experience **"heterogeneity collapse"** under mixed-task batches because batch-level average pooling flattens task-routing signals, resulting in poor uniform performance.
To shield model merging from this collapse, the prior state-of-the-art relied on **Micro-Batch Homogenization (MBH)** paired with Parameter-Free Subspace Routing (PFSR). However, MBH requires partitioning heterogeneous streams into homogeneous sub-batches, executing them sequentially, and re-sorting outputs. This shifts the complexity to database-orchestration and systems-serving layers, leading to sequential execution latency that scales linearly with task diversity ($O(G)$ complexity) and heavy systems-level infrastructure bloat.

## 2. Proposed Approach: Parameter-Free Activation Blending (PFAB)
The authors present **Parameter-Free Activation Blending (PFAB)**, a minimalist, training-free framework designed to serve heterogeneous multi-task streams in a single parallelized forward pass of the backbone with constant wall-clock latency ($O(1)$ complexity). 
Instead of merging model weights on-the-fly (which is batch-bound), PFAB blends adapter outputs sample-wise directly in activation/feature space. 

The framework consists of three key components:
1. **Unit-Norm Calibration (UNC):** A training-free normalization technique that projects penultimate hidden representations and classification weights onto the unit hypersphere to neutralize cross-expert representation scale imbalances.
2. **Non-Parametric Task Coordinates (Cosine Similarity Projection):** Computes task-coordinate similarity by projecting the calibrated representation onto normalized classification heads of each expert using maximum cosine similarity. To correct statistical bias from asymmetrical classification cardinalities ($C_k$), a class-size scaling divisor $\sqrt{2\log C'_k / D}$ is applied (constraining $C'_k \ge 2$ to prevent division by zero). Task coordinates are passed through a sharp, temperature-scaled Softmax ($\tau = 0.001$) to resolve task identity with zero trainable parameters and zero calibration data.
3. **Activation-Space Adapter Blending (ASAB):** A highly parallelized feature-modulation layer that scales adapter outputs by sample-specific task coordinates in a single vectorized forward pass.

To resolve the **pipeline causality dilemma** (requiring routing coordinates at intermediate layer 1, which are derived from penultimate features at final layer $L$), the authors propose two architectural pathways:
- **Base-Only Prototyping Pass (PFAB-BOP):** A mathematically precise two-pass strategy. The first pass propagates the batch through the base model (with adapters deactivated) to extract penultimate representations and compute coefficients. The second pass executes with parallel adapters active, scaled by the pre-computed coefficients.
- **Early-Layer Gating with Calibration-Centroids (PFAB-ELC):** A single-pass pathway that extracts early-layer activations (e.g., Layer 0) and projects them onto pre-computed offline task centroids to bypass semantic representation mismatch and execute in a single forward pass ($O(1)$ backbone complexity).

## 3. Key Findings and Quantitative Claims
The paper evaluates the proposed methods on the **Isolating Coordinate Sandbox** (a physical simulation of representation-space coordinate dynamics and scale imbalances) and an organic pilot on **DomainNet** using a pre-trained Vision Transformer (ViT-B/16).
- **Heterogeneous Stream Robustness:** Standard dynamic routing methods (Linear Router, L3-Linear, QWS SOTA) collapse completely to static Uniform Merging performance (28.60% accuracy on Sandbox, 9.35% on DomainNet) when deployed on mixed batches. PFAB-BOP successfully avoids collapse, maintaining **81.50% Joint Mean accuracy** (matching the prior SOTA PFSR+MBH and the theoretical expert ceiling perfectly) on the Sandbox, and **78.80% Joint Mean accuracy** on DomainNet.
- **Latency Speedup:** At 4 active tasks ($G=4$) with batch size $B=64$, MBH's wall-clock latency scales linearly to 14.72 ms. 
  - PFAB-BOP delivers peak accuracy with flat, constant wall-clock latency of **5.84 ms**, representing a **2.52$\times$ latency speedup**.
  - PFAB-ELC resolves requests in constant time at **4.52 ms**, delivering a **3.26$\times$ speedup** (though with a 15.00% absolute accuracy drop to 66.50% Joint Mean accuracy).
- **High-K Scalability:** Using Sample-Wise Sparse Top-$p$ Expert Filtering ($p=2$), PFAB slashes latency under $K=64$ installed experts from 24.84 ms down to **11.22 ms** with negligible accuracy drop (-0.60%).
- **Ablation of UNC:** In the absence of Unit-Norm Calibration, representation scale drift reduces the Joint Mean accuracy to 53.40% on the Sandbox. UNC restores it to 81.50%.
- **SVD Orthogonalization:** Under severe cross-task subspace entanglement ($\epsilon=0.5$), standard activation blending drops to 51.30%. Applying SVD joint parameter-space orthogonalization prior to serving reduces cross-task overlap to zero and restores accuracy to **80.50%** while preserving 99.87% of the experts' original specialized capabilities.

## 4. Explicitly Claimed Contributions (with Evidence)
1. **Pruning Systems Bloat:** By shifting the blending operation to activation-space, the paper removes the need for micro-batch partitioning, on-the-fly model compiling, index tracking, and sequential dispatching. (Evidence: Section 3.4 complexity analysis, Tables 1 and 2).
2. **Unit-Norm Calibration (UNC):** A robust, training-free spatial alignment layer that resolves representation scale drift. (Evidence: Ablation study in Section 4.5).
3. **Two Architectural Pathways (BOP & ELC):** Clean mathematical solutions to the pipeline causality loop. (Evidence: Formulations in Section 3.4, Flowcharts in Appendix G).
4. **Vectorized Parallel Adapter Execution:** Designs a tensor-parallel adapter execution layer using `torch.bmm` and `torch.einsum` to resolve PyTorch kernel launch bottlenecks. (Evidence: Section 4.4, latency sweeps, Appendix D formulation).
5. **Generative LLM Extension Proposals:** Bridges the classification-based gating with generative autoregressive sequence serving via Prompt-Level Semantic Projection (PLSP), Task-Specific Vocabulary-Head Anchoring (TSVHA), and the Dynamic Gate Reset (DGR) safeguard. (Evidence: Section 3.5 theory, Section 4.5 token sequence simulation).
6. **Subspace Entanglement Mitigation:** Proposes joint SVD orthogonalization to insulate expert representations under extreme overlap. (Evidence: Section 4.5, Appendix E.1 pilot).
