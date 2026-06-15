# Evaluation Stage 1: Paper Summary

## Main Topic
The paper addresses the challenge of deploying dynamic model ensembling and weight-merging in latent coordinate spaces (e.g., SABLE, ChemMerge, Momentum-Merge) on resource-constrained edge hardware. Specifically, it focuses on the "Quantization Collapse" phenomenon, where standard low-precision quantization (8-bit integer activations [INT8] and 4-bit integer ensembling weights [INT4]) completely degrades the performance of dynamic ensembling, causing it to collapse to simple static uniform merging.

## Proposed Approach
To overcome Quantization Collapse, the authors propose **QA-Merge** (Quantization-Aware Merge), which comprises four lightweight, hardware-compatible mechanisms:
1. **Quantized Centroid Calibration (QCC):** Computes task-specific centroids in the high-precision floating-point space using calibration data first, and then quantizes the final average, maintaining separation in the INT8 space. It also uses scale-invariant cosine similarity instead of negative Euclidean distance to avoid range mismatches.
2. **Straight-Through Estimator (STE) Gating:** Uses the Straight-Through Estimator to allow backpropagation of gradients through the non-differentiable rounding operator during offline calibration of parametric routers.
3. **Error-Feedback Trajectory Stabilization (EF-Smooth):** Tracks blending coefficient rounding errors layer-by-layer and adds them back to the subsequent layer as a high-pass feedback correction. It uses a new sorting-free, permutation-invariant allocation method called **Permutation-Invariant Single-Pass Apportionment (PI-SPA)** to project weights onto the discrete 4-bit simplex.
4. **Activation Error Feedback (AEF):** Residually accumulates sub-grid activation rounding errors layer-by-layer and adds them back to the unquantized representations, overcoming the "Small-Step Quantization Bottleneck" where tiny expert updates are rounded to zero.

## Key Findings & Claims
- **Quantization Collapse:** Naive INT8/INT4 quantization erases the benefits of dynamic ensembling, collapsing performance to uniform merging due to centroid overlap and frozen routing weights.
- **Accuracy Recovery:** QA-Merge successfully recovers $\sim 100\%$ of full-precision ensembling gains on the Coordinate Sandbox (ICS) environment across both small-sample ($N_{\text{cal}}=64$) and large-sample ($N_{\text{cal}}=4000$) regimes.
- **Efficiency and Latency Speedup:** Real physical implementation of the ensembling loop on an STM32H753XI microcontroller (Cortex-M7) yields a **5.2x latency speedup** (0.18 ms vs 0.95 ms) and a **42% power reduction** compared to a Float32 FPU implementation.
- **Generalization:** The authors provide a PyTorch toy implementation demonstrating application to dynamic LoRA-mixtures.

## Claimed Contributions
1. Identification and empirical deconstruction of the *Quantization Collapse* phenomenon.
2. Formulation of the *QA-Merge* framework (QCC, STE, EF-Smooth, and AEF) with theoretical error-bounding proofs (Theorem 3.1).
3. Thorough empirical validation inside the high-fidelity Coordinate Sandbox, along with microarchitectural hardware estimates, physical microcontroller deployment, and a PyTorch LoRA mixture demonstrator.
