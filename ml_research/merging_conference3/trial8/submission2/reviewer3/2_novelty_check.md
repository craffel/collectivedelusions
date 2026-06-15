# Novelty and Originality Assessment

This assessment evaluates the originality, conceptual leaps, and overall novelty of the submission, highlighting the delta between the proposed methods and prior literature.

## Key Delta from Prior Work
The paper positions its proposed framework, **Q-SPS** / **CG-Q-SPS**, against three main lines of prior work:
1. **Static Model Merging (e.g., Task Arithmetic, TIES-Merging):** These methods average weights in parameter space and suffer from "heterogeneity collapse" when evaluated on mixed-task streams.
2. **Dynamic PEFT Serving Systems (e.g., PFSR + MBH):** These systems route inputs to task-specific adapters but require partitioning mixed batches into homogeneous micro-batches. This forces sequential forward passes of the massive base model backbone, leading to latency that scales linearly with the number of active tasks.
3. **Activation-Space Blending (e.g., SABLE, SPS-ZCA):** SPS-ZCA resolves the latency-heterogeneity trade-off by dynamically blending expert activations layer-by-layer in a single parallel forward pass ($O(1)$ backbone latency). However, SPS-ZCA is restricted to unquantized floating-point execution (FP32/FP16).

The explicit technical "delta" introduced in this paper over the foundational **SPS-ZCA** framework consists of:
- **Quantizing LoRA adapters to low-bitwidth integers (INT4/INT8)** and executing the low-rank additions in pure integer precision (Q-SPS).
- **Quantization-Aware Scale Calibration (QASC)**, a post-training scale calibration heuristic that sequentially decouples MSE optimization of weight clipping scales.
- **Conditional Gating (CG-Q-SPS)**, which applies a hard threshold ($\theta = 0.01$) to skip executing expert adapter paths with near-zero routing coefficients.
- **Intra-Task Dispersion Calibration (IDC)**, which divides ZCA routing cosine similarities by their expected in-distribution average over a calibration split.
- **Coordinate GMM Safety Shield**, which fits a low-dimensional diagonal GMM over the routing similarity coordinates to perform out-of-distribution (OOD) query detection.
- **Temporal-Aware Routing Hysteresis**, which applies an Exponentially Weighted Moving Average (EWMA) filter to the routing coordinates to reduce routing flicker under sequential ($B=1$) streaming.

## Characterization of Novelty
While the paper presents a highly polished, mathematically dense, and exhaustive systems-ML narrative, the actual conceptual novelty is **incremental** and composed primarily of standard engineering heuristics rather than paradigm-shifting ideas. 

The individual components represent straightforward adaptations of well-established techniques to the existing SPS-ZCA framework:
1. **Low-Bitwidth Quantization of LoRA (Q-SPS):** Quantizing adapters to low bitwidths is extremely common in modern deep learning (e.g., QLoRA). Applying it to activation-space blending is a natural engineering progression rather than a conceptual leap.
2. **Quantization-Aware Scale Calibration (QASC):** Rather than introducing a new quantization theory, QASC is a post-training calibration heuristic. It sequentially decouples the MSE minimization over weight scales—first for the down-projection, then for the up-projection. This is a practical optimization (reducing search complexity from $O(N^2)$ to $O(N)$) but remains a straightforward greedy/decoupled optimization trick.
3. **Conditional Expert Gating (CG-Q-SPS):** Skipping the execution of paths whose scaling coefficients are tiny ($\alpha_k < 0.01$) is a standard and highly intuitive pruning optimization. Bypassing unexecuted code blocks is standard software engineering and does not represent a new machine learning paradigm.
4. **Intra-Task Dispersion Calibration (IDC):** Normalizing similarities by dividing them by expected in-distribution scales is a standard feature scaling technique.
5. **Coordinate GMM Safety Shield:** Fitting a diagonal GMM over a low-dimensional vector of coordinates (in this case, $K=4$ cosine similarities) for density-based OOD detection is a direct application of classic Gaussian mixture modeling.
6. **Temporal Routing Filter (EWMA):** Applying an EWMA filter to smooth coordinates over sequential steps is a basic first-order low-pass filter, a standard signal processing technique.
7. **Centroid Orthogonalization (GS-CCO & Löwdin SMD):** The authors explore these methods theoretically to handle non-orthogonal task manifolds. However, their empirical analysis concludes that explicit orthogonalization is **mathematically redundant and even detrimental** due to noise spillover. Thus, these theoretical formulations are ultimately discarded in favor of the simpler unorthogonalized ZCA-IDC baseline, meaning they do not contribute to the functional novelty of the final proposed system.

## Overall Synthesis of Novelty
From a conceptual standpoint, the paper does not introduce a fundamentally new serving paradigm. The core paradigm—single-pass activation-space dynamic blending via early-stage centroid-alignment routing—was already fully established by the prior **SPS-ZCA** framework. 

Instead, this paper is a **compilation of pragmatic engineering optimizations and heuristics** designed to compile and compress SPS-ZCA for edge-device constraints. While the collective execution of these optimizations yields impressive simulated speedups (3.97$\times$) and memory savings (87.5%), the conceptual leaps are modest. The work is highly thorough, intellectually honest, and technically detailed, but it operates firmly within an established paradigm, focusing on incremental systems-level tuning and post-hoc heuristics.
