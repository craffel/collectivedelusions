# Paper Summary: Parameter-Free Activation Blending (PFAB)

## 1. Main Topic and Problem Formulation
The paper addresses the challenge of serving heterogeneous (mixed-task) inference streams in a multi-task learning environment. In production, deploying multiple task-specific Parameter-Efficient Fine-Tuning (PEFT) adapters (such as LoRA) leads to severe memory and hosting overhead. While weight-space model merging (e.g., Task Arithmetic, TIES-Merging, DARE) solves this parameter overhead, it forces a static, global compromise across tasks, leading to destructive interference and degraded performance on individual tasks. 

Dynamic test-time routing can dynamically merge model weights based on input samples. However, standard dynamic routing methods are batch-bound (producing a single merged model weight state for the entire batch). Under heterogeneous mixed-task streams, batch-average pooling of routing coefficients causes "heterogeneity collapse"—the coefficients average out to a flat, uniform distribution, scrambling intermediate representations and destroying task-specific capabilities.

The prior state-of-the-art resolved this collapse using Micro-Batch Homogenization (MBH) combined with Parameter-Free Subspace Routing (PFSR). However, MBH relies on a heavy serving-layer data-orchestration infrastructure to partition heterogeneous streams into homogeneous sub-batches, execute them sequentially, and re-sort outputs. This results in sequential latency that scales linearly with the number of active tasks ($O(G)$ complexity).

## 2. Proposed Approach: Parameter-Free Activation Blending (PFAB)
The authors propose **Parameter-Free Activation Blending (PFAB)**, a non-parametric framework that shifts the blending of specialized expert adapters from parameter-space to activation-space. Because activations are naturally indexed along the sample dimension (i.e., sample-wise), activation blending allows a heterogeneous batch to be processed concurrently in a single, parallelized forward pass of the base model backbone, retaining constant $O(1)$ systems latency while completely avoiding heterogeneity collapse.

PFAB consists of three main components:
1. **Unit-Norm Calibration (UNC):** A training-free normalization technique that projects penultimate activations and pre-trained classification weights onto the unit hypersphere to resolve cross-expert scale drift and representation scale imbalance.
2. **Non-Parametric Task Coordinates:** Computes sample-wise task gating coefficients by projecting calibrated activations onto the normalized classifier heads using maximum cosine similarity, corrected for class-size/dimension bias via an extreme-value statistical divisor $\sqrt{2\log C'_k / D}$ and passed through a sharp temperature-scaled Softmax.
3. **Activation-Space Adapter Blending (ASAB):** A highly parallelized feature-modulation layer that scales adapter outputs sample-wise by their gating coefficients in a single vectorized forward pass.

To resolve the "pipeline causality dilemma" (intermediate layers need the routing coefficients before they are computed at the penultimate layer), the paper proposes two architectural pathways:
- **PFAB-BOP (Base-Only Prototyping Pass):** A mathematically precise two-pass pathway where the first pass propagates the batch through the base model to compute routing coefficients from penultimate activations, and the second pass executes intermediate activation blending scaled by these coefficients.
- **PFAB-ELC (Calibration-Centroid Early-Layer Gating):** A single-pass pathway that resolves task identity at an early layer (Layer 0 or 1) by projecting early representations onto pre-computed offline task centroids, achieving true $O(1)$ constant-time execution at the expense of requiring a small set of offline calibration samples.

## 3. Explicitly Claimed Contributions
The paper explicitly claims the following contributions:
- **The Activation-Space Blending Paradigm:** Demonstrating that shifting dynamic multi-task PEFT serving from weight-space to activation-space resolves heterogeneity collapse in a single vectorized forward pass.
- **Unit-Norm Calibration & Class-Size Scaling:** Providing training-free, non-parametric calibration techniques to neutralize representation scale imbalances and class-cardinality biases.
- **Layer-Wise Adapter Scaling (LAS):** A training-free scaling mechanism to handle physical feature magnitude differences across independently trained multi-tenant experts.
- **Two Execution Pathways:** Standardizing the two-pass PFAB-BOP for mathematical precision and the single-pass PFAB-ELC for inference efficiency.
- **Memory and Compute Optimizations:** Proposing Sparse Top-$p$ Expert Filtering and Chunked Layer-Wise Execution to bound memory accumulation and computational complexity under massive task counts.
- **Generative LLM Extensions:** Providing formal theoretical designs (PLSP and TSVHA with DGR) to map non-parametric activation blending onto large vocabulary generative models.
- **Empirical Validation:** Showing perfect expert-ceiling matching accuracy (**81.50% Joint Mean**) on the Isolating Coordinate Sandbox with a 2.52$\times$ wall-clock speedup (5.84 ms vs. 14.72 ms) at $G=4$ active tasks, and validating the framework empirically on a pre-trained ViT-B/16 over DomainNet.

## 4. Key Findings
- **Catastrophic Collapse of Weight Routing:** Standard dynamic weight-merging routers collapse completely to the static uniform baseline (28.60% Joint Mean accuracy) under high-entropy mixed-task streams.
- **Expert Ceiling Matching:** PFAB-BOP matches the prior SOTA (PFSR+MBH) and the absolute Expert Ceiling (81.50% Joint Mean accuracy on the Sandbox; 78.80% on DomainNet) under heterogeneous mixed streams.
- **Flat Latency Profiles:** Both PFAB pathways exhibit completely flat, constant wall-clock execution latency curves with respect to the number of active tasks $G$, unlike MBH which scales linearly. At $G=4$, PFAB-BOP and PFAB-ELC achieve 2.52$\times$ and 3.26$\times$ speedups, respectively, over MBH in a standard edge batch size ($B=64$).
- **SVD Orthogonalization Effectiveness:** When cross-task subspaces are heavily entangled ($\epsilon=0.5$), performing joint SVD orthogonalization of adapter weights prior to serving restores activation blending accuracy from 51.30% to 80.50%.
- **Generative LLM Feasibility:** In token-by-token simulations, Periodic Gating with the proposed Dynamic Gate Reset (DGR) safeguard achieves perfect 100.00% gating synchrony while saving 78.00% of GPU vocabulary projection FLOPs.
