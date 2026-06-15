# Novelty and Delta Analysis

## Conceptual Novelty vs. Engineering Integration
From a rigorous, critical perspective, the conceptual novelty of **SA-QAB** is relatively **incremental**. Rather than introducing a fundamentally new paradigm, the framework represents a practical, hybrid engineering integration of several pre-existing techniques in model merging, post-training quantization, and dynamic routing, which has been repackaged with grandiose nomenclature:

1. **SPS-ZCA / SABLE Adaptation**: The core mechanism of routing input samples dynamically to specific expert adapters based on pre-computed task centroids in activation space is directly inherited from prior works like **SPS-ZCA** and **SABLE** (which the authors explicitly cite in Section 2). The conceptual "delta" here is simply adapting this routing mechanism to execute on integer-only hardware (quantized representations) rather than full-precision (FP16/FP32).
2. **QLoRA-style Heterogeneous Bit-widths (DHQ)**: Decoupling the quantization of the base model (to low-bit, e.g., 4-bit) and the adapters (to 8-bit) is a well-established practice in parameter-efficient fine-tuning (PEFT) and quantization literature. For instance, **QLoRA** quantizes the base model to 4-bit (NF4) while keeping the adapters in FP16 or INT8 during training. Translating this heterogeneous scheme to integer-only inference (INT4 base, INT8 adapters) is a straightforward application of standard TinyML quantization principles rather than a novel conceptual breakthrough.
3. **Quantization Scale Recovery (QSR)**: The technique of matching $L_2$ norm ratios of unquantized and quantized activations over a small calibration dataset is a standard calibration heuristic in PTQ literature (similar to scale matching and activation-aware scaling techniques). Its formulation in Equation 4 is an intuitive, straightforward scaling correction rather than a sophisticated mathematical contribution.
4. **Diagonal GMM OOD Gate**: Fitting a diagonal Gaussian Mixture Model to intermediate activations for Out-of-Distribution (OOD) detection is a highly standard, classical machine learning technique. Utilizing it here as an on-device gate is a standard systems engineering choice rather than a research novelty.

---

## Detailed Delta from Closely Related Prior Works

| Dimension | SABLE / SPS-ZCA | Q-Merge (STE) | PMQ (Post-Merge Quantization) | **SA-QAB (Ours)** |
| :--- | :--- | :--- | :--- | :--- |
| **Merging Domain** | Activation Space (Blending) | Weight Space (Optimized) | Weight Space (Averaged) | **Activation Space (Blending)** |
| **Precision Scheme** | Full-Precision (FP16/FP32) | Uniform Quantized (INT4) | Uniform Quantized (INT4/INT8) | **Decoupled Heterogeneous (INT4 Base / INT8 Adapters)** |
| **Quantization Noise Mitigation** | N/A (Runs in FP) | Straight-Through Estimators (STE) optimization | None (Catastrophic collapse) | **Quantization Scale Recovery (QSR) factors + Q-ZCA noise filtering** |
| **Routing Manifold** | Full-Precision Cosine Sim | Static (No routing) | Static (No routing) | **Quantized Integer-Space Cosine Sim (INT8)** |
| **Task Modularity** | High (Separated adapters) | Low (Requires re-optimization) | Low (Requires re-merging) | **High (Separated adapters on-the-fly)** |
| **Active Expert Compute** | $O(1)$ adapter compute | $O(1)$ (Fully fused weight) | $O(1)$ (Fully fused weight) | **$O(1)$ (Sparse routing via Hard Maxima)** |

---

## Key Assessment of Claims and Delta
- **Is the "Scale Alignment" truly novel?** 
  - Standard PTQ methods (like SmoothQuant) scale activations to match weight scales. SA-QAB's **QSR** scales quantized adapter activations to match their FP16 counterparts. While this specifically addresses low-bit adapter contraction, it is conceptually equivalent to traditional gain-correction or scale-calibration steps standard in DSP and TinyML deployments.
- **Is "Q-ZCA" a significant advance over SABLE's routing?**
  - SABLE computes cosine similarity over FP16 features. Q-ZCA quantizes features and centroids to INT8, performing integer-only dot products. While this requires careful engineering (approximating feature norms with fixed-point arithmetic), it represents an optimization step for low-resource hardware rather than an algorithmic advancement.
- **The "Model Merging" Misnomer**:
  - The paper positions itself in the "model merging" space. However, traditional model merging *fuses* the weights of multiple networks so that only a single model is deployed, resulting in zero additional memory overhead as the number of tasks scales. SA-QAB does not merge weights; it keeps all $K$ task-specific adapters separately in memory and routes to them. Therefore, SA-QAB is technically a **dynamic multi-adapter routing / Mixture-of-Experts (MoE) system**, not a model merging method. Comparing its SRAM/Flash scaling directly against weight-space merging (like PMQ or Q-Merge) is misleading because SA-QAB incurs a linear $O(K)$ flash storage cost, whereas true model merging remains strictly $O(1)$ in storage. This represents a significant deviation from the core goal of model merging.
