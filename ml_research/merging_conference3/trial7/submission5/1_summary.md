# 1. Summary of the Paper

## Title
**Parameter-Free Activation Blending: Applying Occam's Razor to Heterogeneous Multi-Task Model Merging**

## Author
Leo Vance (Stanford University)

## Main Idea and Motivation
The paper addresses the challenge of deploying multi-task expert adapters (e.g., Low-Rank Adaptation/LoRA) on heterogeneous, mixed-task inference streams. Under standard test-time dynamic routing or weight-space model merging, batches containing interleaved samples from different tasks suffer from "heterogeneity collapse." This happens because batch-average pooling of routing coefficients results in a uniform, flat compromise that washes out individual task signals, causing the model to behave as a poor static baseline.

Prior state-of-the-art (SOTA) solutions like Parameter-Free Subspace Routing (PFSR) combined with Micro-Batch Homogenization (MBH) prevent this collapse by dynamically partitioning the heterogeneous batch into homogeneous sub-batches and dispatching them sequentially. However, MBH shifts the complexity to heavy data-orchestration and systems-serving layers (introducing dynamic compilation, index tracking, and sequential execution overhead). This results in a wall-clock latency that scales linearly with task diversity ($O(G)$ sequential passes).

To resolve this complexity dilemma, the author proposes **Parameter-Free Activation Blending (PFAB)**. Guided by a minimalist research philosophy, PFAB performs **sample-wise activation-space blending of expert outputs directly in feature space on-the-fly**, entirely bypassing weight-space model merging. Because activations are naturally indexed by the sample dimension, activation blending allows the model to process heterogeneous requests in a single, parallelized forward pass of the backbone with flat, constant wall-clock latency ($O(1)$) and zero serving-level partitioning.

---

## Technical Methodology
PFAB features three key components:
1. **Unit-Norm Calibration (UNC):** A training-free normalization technique that projects both the final-layer representations ($z_b$) and the pre-trained, frozen classification weights ($W_{k,c}$) onto the unit hypersphere. This neutralizes cross-expert representation scale drift.
2. **Non-Parametric Task Coordinates:** Computes raw similarity gating coordinates via a maximum cosine similarity projection of the calibrated representation onto normalized classification heads, corrected for asymmetric classification cardinality (class size) and passed through a sharp, temperature-scaled Softmax to yield sample-wise routing coefficients ($\alpha_{k, b}$).
3. **Activation-Space Adapter Blending (ASAB):** A vectorized, parallel feature-modulation layer that scales adapter outputs by the sample-specific coefficients in a single, parallel forward pass. To address Python loop/CUDA kernel launch overheads, a parallelized execution layer using batched matrix multiplications (`torch.bmm`) is designed, evaluating all expert adapters concurrently in a single operation per layer.

### Architectural Pathways
To resolve the **pipeline causality dilemma** (the fact that intermediate activation blending requires routing coefficients that are only computed from deep final-layer representations), the paper introduces two pathways:
* **PFAB-BOP (Base-Only Prototyping Pass):** A mathematically exact two-pass strategy. The first pass (prototyping pass) propagates the batch with adapters deactivated to extract base representations and compute routing coefficients. The second pass (execution pass) runs the full backbone with active adapters scaled by the pre-computed coefficients.
* **PFAB-ELC (Exemplar-Locked Centroids):** A single-pass pathway that extracts early-layer activations and projects them onto pre-computed offline task centroids (calibrated using a small set of offline labeled samples). This achieves constant-time single-pass inference.

---

## Summary of Contributions
1. **Minimalist Philosophy:** Proposes a training-free, non-parametric activation blending framework that completely prunes heavy data-orchestration layers from serving infrastructures.
2. **Fine-Grained Sample Routing:** Demonstrates that sample-level feature-space blending avoids the batch-averaging smoothing of weight-space merging, resolving "heterogeneity collapse" with high-fidelity routing.
3. **Extensive System Optimizations:** Develops **Layer-Wise Adapter Scaling (LAS)** to neutralize scale imbalances across disjointly trained experts, **Sparse Top-$p$ Expert Filtering** to bound compute complexity, and **Chunked Layer-Wise Execution** to eliminate GPU memory (VRAM) accumulation during sequence expansions.
4. **LLM Generative Extensions:** Proposes **Prompt-Level Semantic Projection (PLSP)** and **Task-Specific Vocabulary-Head Anchoring (TSVHA)** with a **Dynamic Gate Reset (DGR)** safeguard to handle token-level sequence routing in generative Large Language Models.
5. **Exhaustive Rigorous Evaluation:** Validates the math on a high-fidelity physical *Isolating Coordinate Sandbox* and an organic visual corpus (*DomainNet* using a pre-trained ViT-B/16 backbone), demonstrating SOTA multi-task capabilities, flat constant wall-clock latencies, and strong robustness under severe quantization and subspace entanglement.

---

## Key Empirical Findings
* **Simulated Sandbox (K=4, B=256):** Under mixed-task heterogeneous batching, parametric routers collapse to ~28.60% Joint Mean accuracy. PFAB-BOP maintains the **81.50% Joint Mean accuracy** (matching the Expert Ceiling and PFSR+MBH perfectly) but reduces serving-layer complexity and accelerates execution.
* **Systems-Level Latency (B=64):** At $G=4$ active tasks, PFAB-BOP achieves a **2.52$\times$ latency speedup** (5.84 ms vs. 14.72 ms) over MBH, while the single-pass PFAB-ELC delivers a **3.26$\times$ speedup** (4.52 ms vs. 14.72 ms) with flat, constant-time scaling.
* **Organic DomainNet (K=4, C=20):** PFAB-BOP matches the Expert Ceiling and PFSR+MBH SOTA at **78.80% Joint Mean accuracy**, delivering a **1.31$\times$ speedup** (19.80 ms vs. 25.84 ms) on a pre-trained ViT-B/16 backbone under $G=4$ active tasks.
* **Subspace Entanglement Stress Test:** Under extreme leakage ($\epsilon = 0.5$), standard PFAB-BOP degrades to $51.30\%$. Combining activation blending with a proposed joint SVD parameter orthogonalization perfectly restores performance to **80.50%** with flat $O(1)$ constant systems latency.
* **Generative LLM Sequence Simulation:** TSVHA combined with the DGR safeguard achieves **100.00% Gating Synchrony** (matching continuous gating perfectly) while saving **78.00% of GPU vocabulary projections** and completely eliminating transition latency delays.
