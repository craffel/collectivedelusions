# Evaluation Component 1: Summary of the Paper

## Main Topic and Motivation
The paper presents **Parameter-Free Activation Blending (PFAB)**, a minimalist, training-free, and parameter-free framework for serving heterogeneous, mixed-task inference streams in multi-task expert model deployment. In multi-task serving, fine-tuning lightweight, task-specific adapters (such as LoRA) is a popular paradigm. However, merging these adapters into a single static model compromises task-specific capabilities due to parameter-level interference. Dynamic, test-time routing approaches resolve this in homogeneous batches but suffer from "heterogeneity collapse" in mixed-task batches because batch-level average pooling flattens task signals. The previous state-of-the-art approach, Parameter-Free Subspace Routing (PFSR) combined with Micro-Batch Homogenization (MBH), resolves this collapse by dynamically partitioning heterogeneous streams into homogeneous sub-batches and executing them sequentially. However, MBH introduces heavy serving-infrastructure complexity and a sequential execution latency bottleneck scaling linearly with task diversity ($O(G)$). PFAB aims to eliminate this systems complexity and achieve flat, constant latency ($O(1)$) by shifting the blending operation from parameter-space to activation-space, enabling sample-wise blending directly in a single parallel forward pass.

## Core Approach and Methodology
PFAB operates directly in feature space within each model layer. It introduces:
1. **Unit-Norm Calibration (UNC):** A training-free normalization that projects both penultimate activations and classification weights onto the unit hypersphere to equalize feature scales and prevent scale-imbalance dominance.
2. **Non-Parametric Task Coordinates:** Computes raw similarity via cosine similarity projection onto normalized classification heads, scales them by Class-Size Scaling Calibration (using effective cardinality $C'_k = \max(C_k, 2)$ to avoid division-by-zero) to correct vocabulary cardinality bias, and applies a temperature-scaled Softmax ($\tau = 0.001$) to derive crisp, sample-specific coefficients $\alpha_{k,b}$.
3. **Activation-Space Adapter Blending (ASAB):** Computes shared base model activations and parallel expert adapter deltas, then blends them sample-wise using the gating coefficients in a highly parallelized forward pass via vectorized tensor operations (`torch.bmm`).
4. **Layer-Wise Adapter Scaling (LAS):** Normalizes intermediate expert adapter outputs by their Frobenius norms or running average activation norms to prevent scale imbalances caused by disjoint training configurations.
5. **Architectural Pathways:**
   - **PFAB-BOP (Base-Only Prototyping Pass):** A two-pass execution strategy where a prototyping pass extracts base representations to compute routing coefficients, followed by a second execution pass with active parallel adapters.
   - **PFAB-ELC (Early-Layer Gating with Calibration-Centroids):** A single-pass execution strategy that uses pre-computed offline task centroids from a small set of labeled calibration samples to perform gating at early layers.
6. **Memory/Compute Optimizations:** Includes Sparse Top-$p$ Expert Filtering and Chunked Layer-Wise Execution to bound the serving footprint.
7. **Theoretical LLM Extensions:** Proposes Prompt-Level Semantic Projection (PLSP) and Task-Specific Vocabulary-Head Anchoring (TSVHA) with a Dynamic Gate Reset (DGR) safeguard for autoregressive language models.

## Key Findings and Claimed Contributions
1. **Performance under Homogeneous and Heterogeneous Streams:** 
   - On the Isolating Coordinate Sandbox simulation, PFAB-BOP matches the absolute Expert Ceiling and SOTA PFSR+MBH perfectly at **81.50% Joint Mean accuracy** under both stream configurations.
   - PFAB-ELC achieves **66.50% Joint Mean accuracy** under both streams, sacrificing accuracy for single-pass efficiency.
2. **Systems-Level Latency and Scalability:**
   - On a batch size of $B=64$ with $G=4$ active tasks, PFAB-BOP delivers a **2.52x wall-clock speedup** over MBH (5.84 ms vs. 14.72 ms), while PFAB-ELC delivers a **3.26x speedup** (4.52 ms vs. 14.72 ms), with both pathways maintaining flat, constant latency curves.
3. **Organic Pilot Validation:**
   - On DomainNet with a pre-trained ViT-B/16, PFAB-BOP perfectly matches the Expert Ceiling of **78.80% Joint Mean accuracy** under heterogeneous streams while delivering a **1.31x speedup** over MBH (19.80 ms vs. 25.84 ms). PFAB-ELC drops to **42.50% Joint Mean accuracy** but maintains full single-pass speed.
4. **Unsupervised Streaming and Quantization Robustness:**
   - Demonstrates that Streaming ELC using on-the-fly, unsupervised K-means clustering achieves **58.20% Joint Mean accuracy**.
   - Shows that PFAB-BOP is robust to severe simulated quantization noise, preserving a **45.90% Joint Mean accuracy**.
5. **Generative LLM Sequence Simulation:**
   - Shows that TSVHA with DGR achieves **100% Gating Synchrony** while saving **78% of GPU vocabulary projections** in a token-by-token text generation simulation.
