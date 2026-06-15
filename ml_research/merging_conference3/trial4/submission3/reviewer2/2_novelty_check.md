# Novelty and Delta Check

This document critically evaluates the novel aspects of the paper, the technical delta from existing work, and whether the proposed methods and concepts represent significant advancements or incremental/rebranded extensions of known techniques.

## 1. Characterization of Conceptual Novelty ("Re-Quantization Silence")
The authors coin the term **"Re-Quantization Silence"** to describe how low-bit quantization of a merged model rounds task-specific adapter updates ($\Delta W$) to zero due to the dominant dynamic range of the base weights ($W_0$).

*   **Critical Evaluation of the Delta:**
    While presented as a "dangerous methodological blindspot" unique to model merging, the phenomenon of high-precision updates being rounded to zero when quantized alongside high-magnitude parameters is a fundamental, well-known limitation of uniform quantization. In post-training quantization, outliers and wide dynamic ranges in weight tensors have long been known to "squash" the quantization bins, leading to severe discretization noise for smaller values. 
    Furthermore, the authors' own control experiment in **Table 6** reveals that under INT4 Symmetric Per-Tensor quantization, even the *unmerged individual experts* (which have zero task interference) suffer a catastrophic **10.90%** performance collapse (dropping from 93.85% to 82.95%). Under the same per-tensor quantization, the merged model (Naive-RQ) drops by **9.90%** (Table 5). This proves that the performance drop is almost entirely driven by the general, severe degradation of the base model representations under per-tensor INT4 quantization, rather than any unique "adapter silencing" specific to the model-merging process. Thus, the conceptual framing of "Re-Quantization Silence" as a novel, merging-specific failure mode is significantly overstated.

## 2. Technical Novelty of Scale-Adaptive Weight Shifting (SAWS)
SAWS is proposed as a "data-free, closed-form scaling mitigation" that projects low-rank updates into a larger dynamic range by scaling them with $\gamma^l$ and applying an output correction factor $c^l \approx 1$ at inference.

*   **Critical Evaluation of the Delta:**
    The core technical mechanism of SAWS is multiplying the adapter updates by a large scaling factor ($\gamma^l \approx 10$ to $100$) relative to the base weights. 
    1. **Equivalence to Standard Scaling:** In LoRA, scaling the adapter update by a scalar (e.g., $\frac{\alpha}{r}$) is a standard practice and a fundamental hyperparameter. Mathematically, scaling the task vectors by $\gamma^l$ during merging is functionally equivalent to scaling the adapter outputs.
    2. **Role of Output Alignment Factor:** The authors derive an "elegant, closed-form alignment factor" $c^l$. However, they admit that because the weight tensors are heavily dominated by the base weights, $c^l \approx 1.0$ (typically $0.99$). Since $c^l \approx 1$, the inference calculation is practically $Y \approx X \tilde{W}_0^T + \gamma^l X \Delta W_{\text{merged}}^T$. This means there is no true mathematical scale preservation happening; instead, the method is simply boosting the adapter weights by $\gamma^l$.
    3. **Novelty vs. Simple Grid Alignment:** The "delta" of SAWS from standard task vector scaling is highly incremental. It computes $\gamma^l$ based on Frobenius norms, but the underlying effect is merely inflating the merging coefficient of the adapters. The paper does not evaluate whether simply searching for a better global scaling factor $\lambda$ in full-precision (FP16) would achieve the exact same performance boost.

## 3. Technical Novelty of Quantization-Aware Adapter Coefficient Search (QA-ACS)
QA-ACS optimizes layer-wise blending coefficients directly through the quantization operator using Straight-Through Estimators (STE) on a tiny calibration set.

*   **Critical Evaluation of the Delta:**
    1. **Incremental over AdaMerging:** AdaMerging (PH-Q) already optimizes layer-wise blending coefficients using prediction entropy minimization. The only technical change in QA-ACS is applying this optimization *through* the quantization operator using STE.
    2. **Worse Performance than Existing Baselines:** The critical issue is that **QA-ACS is systematically outperformed by the existing AdaMerging (PH-Q) baseline** in every single evaluated configuration.
       - Under INT8 Per-Channel: AdaMerging (PH-Q) gets **70.10%** vs. QA-ACS **69.35%** (-0.75%).
       - Under INT4 Symmetric Per-Channel: AdaMerging (PH-Q) gets **68.80%** vs. QA-ACS **68.00%** (-0.80%).
       - Under INT4 Asymmetric Per-Channel: AdaMerging (PH-Q) gets **68.25%** vs. QA-ACS **64.75%** (-3.50%).
       - Under INT4 Symmetric Per-Tensor: AdaMerging (PH-Q) gets **57.25%** vs. QA-ACS **57.00%** (-0.25%).
       Since a method that optimizes weights in full-precision and then quantizes them (AdaMerging PH-Q) strictly outperforms the proposed quantization-aware optimization (QA-ACS), the technical delta and utility of introducing STE-based optimization here are highly questionable. It suggests that QA-ACS is not only incremental but practically counterproductive compared to existing simpler methods.

## Summary of Novelty Assessment
The conceptual novelty of "Re-Quantization Silence" is heavily overstated, as the reported collapse is almost entirely a symptom of general INT4 per-tensor quantization degradation on Vision Transformers, rather than a unique model-merging phenomenon. The proposed methods, SAWS and QA-ACS, represent highly incremental technical deltas: SAWS is functionally a heuristic scale booster for task vectors (equivalent to adjusting standard LoRA alpha), and QA-ACS is a standard test-time optimization that underperforms the existing FP16-optimized baseline (AdaMerging) in all evaluated scenarios.
